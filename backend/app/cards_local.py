"""Condition-card extraction on the local model (no cloud tokens).

The same job the Claude agents did (data/cards/EXTRACTION_SPEC.md), done by the
Mac's own model in small windows. It is slower and less careful, which is safe
here because nothing it writes is trusted: cards_build.verify keeps an item only
when its quote is verbatim in the cited chunk and one of its terms is in the
quote.

    python -m app.cards_local            # every packet without a raw/*.json yet
    python -m app.cards_local --limit 2  # a trial

Resumable: each window's answer is appended to raw/<packet>.windows.jsonl, and a
packet's raw/<packet>.json is written only when all its windows are done.
"""

from __future__ import annotations

import argparse
import json
import re
import time

from . import config
from .cards_build import PACKETS_DIR, RAW_DIR

WINDOW_CHARS = 5000
MAX_TOKENS = 2600

# Emergency-department documents first, so the most useful cards exist soonest.
PRIORITY = (
    "moh-standard-practice", "management-of-dengue", "management-of-asthma", "moh-guideline-management-of-snakebite",
    "moh-training-manual-hypertensive", "moh-antidotes", "early-management-of-head-injury", "moh-pain-management",
    "moh-quick-reference-guide-postpartum", "malaysian-consensus-on-the-management-of-acute", "management-of-non-st",
    "management-of-acute-coronary-syndromes", "management-of-ischaemic-stroke", "management-of-spontaneous",
    "management-of-chronic-obstructive", "prevention-diagnosis-and-management-of-infective", "management-of-non-variceal",
    "management-of-acute-variceal", "management-of-abdominal-trauma", "msic-icu", "paediatric-protocols",
    "national-antimicrobial", "moh-perinatal", "management-of-diabetes-in-pregnancy", "management-of-type-1",
    "management-of-heart-failure", "management-of-gout", "management-of-haemophilia", "management-of-foreign-body",
    "management-of-neonatal-jaundice", "management-of-sore-throat", "prevention-and-treatment-of-venous",
    "management-of-geriatric-hip", "management-of-thyroid", "management-of-bipolar", "management-of-cancer-pain",
)

SYSTEM = """You extract clinical guidance from a Malaysian Ministry of Health guideline into JSON.
Copy every quote EXACTLY, character for character, from ONE chunk of the text given. Never paraphrase.
Only guidance for handling a patient; no study results, no background."""

INSTRUCTIONS = """TEXT (chunks start "## p<page> [r<ref>]"; "§" lines are headings, never quote them):
{text}

For every condition this text gives guidance for, list items. Each item:
- element: definition | red_flag | complication | investigation | treatment | avoid | admission | referral | discharge | monitoring
- label: at most 8 words, as a clinician writes it in a plan
- terms: 1-4 lowercase drug/test/procedure/finding words; at least one MUST appear in the quote
- when: patient states it applies in (e.g. "shock", "hypotension", "pregnant", "fever"), or []
- unless: states it must not be used in, or []
- core: true if every such patient needs it in the first hours
- setting: ed | inpatient | outpatient | any
- ref: the r-number of the chunk the quote is from, e.g. "r123"
- quote: 1-2 sentences copied exactly from that chunk, under 300 characters

Condition fields: name (British spelling), aliases (abbreviations, synonyms), parent (broader condition or ""),
population (adult | paediatric | obstetric | all), acuity (emergency | urgent | routine | chronic).
If the text has no patient guidance, answer {{"conditions": []}}.

Respond ONLY with compact JSON on one line (no indentation): {{"conditions": [{{"name": "", "aliases": [], "parent": "", "population": "all", "acuity": "urgent", "items": [{{"element": "", "label": "", "terms": [], "when": [], "unless": [], "core": false, "setting": "ed", "ref": "", "quote": ""}}]}}]}}"""


def _windows(packet_text: str) -> tuple[str, list[str]]:
    head, _, body = packet_text.partition("\n\n")
    blocks = re.split(r"(?=^## )", body, flags=re.M)
    out, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > WINDOW_CHARS:
            out.append(cur)
            cur = ""
        cur += b
    if cur.strip():
        out.append(cur)
    return head, out


def _json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    start = raw.find("{")
    if start < 0:
        return {"conditions": []}
    depth, end = 0, None
    for i, ch in enumerate(raw[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            end = i + 1
            break
    try:
        return json.loads(raw[start:end] if end else raw[start:])
    except Exception:
        return _salvage(raw[start:])


def _salvage(raw: str) -> dict:
    """A cut-off answer still holds every item that was finished: item objects
    have no nested braces, so each complete one is recovered and filed under
    the condition named before it."""
    conds: list[dict] = []
    marks = [(m.start(), m.group(1)) for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', raw)]
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            item = json.loads(m.group(0))
        except Exception:
            continue
        if "element" not in item or "quote" not in item:
            continue
        name = next((n for pos, n in reversed(marks) if pos < m.start()), "")
        if not name:
            continue
        if not conds or conds[-1]["name"] != name:
            conds.append({"name": name, "aliases": [], "items": []})
        conds[-1]["items"].append(item)
    return {"conditions": conds}


def _order(stems: list[str]) -> list[str]:
    def rank(s: str) -> int:
        return next((i for i, p in enumerate(PRIORITY) if s.startswith(p)), len(PRIORITY))
    return sorted(stems, key=lambda s: (rank(s), s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after this many packets")
    args = ap.parse_args()

    from mlx_lm import generate, load  # noqa: PLC0415
    model, tok = load(config.LOCAL_MLX_MODEL)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in PACKETS_DIR.glob("*.txt") if not (RAW_DIR / f"{p.stem}.json").exists()]
    todo = _order(stems)
    print(f"{len(todo)} packet(s) to extract with {config.LOCAL_MLX_MODEL}", flush=True)
    for n, stem in enumerate(todo, start=1):
        if args.limit and n > args.limit:
            break
        head, windows = _windows((PACKETS_DIR / f"{stem}.txt").read_text(encoding="utf-8"))
        log_path = RAW_DIR / f"{stem}.windows.jsonl"
        done = {}
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                done[row["w"]] = row["conditions"]
        t0 = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            for w, text in enumerate(windows):
                if w in done:
                    continue
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": head + "\n\n" + INSTRUCTIONS.format(text=text)}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
                try:
                    raw = generate(model, tok, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
                    conds = _json(raw).get("conditions") or []
                except Exception as exc:  # one bad window never stops the run
                    print(f"  window {w} failed: {exc}", flush=True)
                    conds = []
                done[w] = conds
                log.write(json.dumps({"w": w, "conditions": conds}, ensure_ascii=False) + "\n")
                log.flush()
                print(f"  [{n}/{len(todo)}] {stem} window {w + 1}/{len(windows)}: "
                      f"{sum(len(c.get('items') or []) for c in conds)} items", flush=True)
        payload = {"packet": stem, "document": head, "extractor": config.LOCAL_MLX_MODEL,
                   "conditions": [c for w in sorted(done) for c in done[w] if isinstance(c, dict)]}
        (RAW_DIR / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"DONE {stem}: {len(payload['conditions'])} condition blocks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
