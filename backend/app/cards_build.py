"""Build the condition cards from the indexed corpus.

Why this exists
---------------
Until 2026-09-30 every per-condition check was written by hand: 44 red-flag
rules, 21 completeness presentations, 16 complication profiles and one
admission rule. Eleven rounds on one rhabdomyolysis case took it from 44 to
96.2; the first new case (ACS with shock) scored ~60. Hand rules do not reach
the conditions nobody has written one for yet, so this builds the same kind of
knowledge for EVERY condition the corpus gives management for.

Pipeline (the extraction step is done by Claude agents, everything around it
is deterministic and re-runnable):

    python -m app.cards_build --packets   # 1. corpus -> data/cards/packets/*.txt
    #   2. agents read each packet and write data/cards/raw/<packet>.json,
    #      following data/cards/EXTRACTION_SPEC.md
    python -m app.cards_build --verify    # 3. verify every quote, merge, write
                                          #    data/condition_cards.json + coverage

Trust boundary: the agents choose WHICH sentence answers WHICH element; they
never supply the words. Step 3 keeps an item only when its quote is found
verbatim (whitespace- and case-blind) in the chunk it cites - or elsewhere in
the same document, in which case the reference is corrected - and when at least
one of its match terms occurs in that quote. Page, title and doc type always
come from the index, never from the agent.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from . import config, distill, population

CARDS_DIR = config.DATA_DIR / "cards"
PACKETS_DIR = CARDS_DIR / "packets"
RAW_DIR = CARDS_DIR / "raw"
REFS = PACKETS_DIR / "refs.json"
OUT = config.DATA_DIR / "condition_cards.json"
COVERAGE = CARDS_DIR / "coverage.json"

ELEMENTS = ("definition", "red_flag", "complication", "investigation", "treatment",
            "avoid", "admission", "referral", "discharge", "monitoring")
SETTINGS = ("ed", "inpatient", "outpatient", "any")
POPULATIONS = ("adult", "paediatric", "obstetric", "all")
ACUITIES = ("emergency", "urgent", "routine", "chronic")

# Documents that are not patient-management guidance, or are read another way:
# FUKKM is a formulary (name lookup), MTS is the triage grid (mts_table.py),
# the HTA report is evidence ABOUT a score, the redirection policy is a flow
# rule (redirection.py).
_SKIP_TYPES = {config.DOC_TYPE_DRUG, config.DOC_TYPE_TRIAGE, config.DOC_TYPE_HTA, config.DOC_TYPE_POLICY}
PACKET_CHARS = 140_000

# A sentence is kept for extraction when it says what to DO, what to look
# for, or when to escalate - or carries a dose. Measured 2026-09-30: 44% of
# the clinical text (5.2 of 11.9M characters) survives; the rest is study
# results, epidemiology and background.
_ACTION = re.compile(
    r"\b(recommend\w*|should|must|consider\w*|indicated|indications?|contraindicat\w*|avoid\w*|do not|don't|"
    r"not be (?:given|used)|cautions?|admi(?:t|ts|tted|ssion)|hospitali[sz]\w*|refer(?:ral|red)?|discharg\w*|"
    r"monitor\w*|repeat\w*|every \d|hourly|check\w*|investigations?|administer\w*|give|given|start|"
    r"initiate\w*|commence\w*|withh[oe]ld\w*|stop\w*|dos(?:e|es|ing|age)|titrat\w*|targets?|criteria|"
    r"diagnosis (?:is|of)|diagnostic|defined|red flags?|warning signs?|danger signs?|complications?|"
    r"life[- ]threatening|emergency|urgent\w*|immediate\w*|first[- ]line|drug of choice|treatment of choice|"
    r"resuscitat\w*|antidote\w*|antivenom|suspect\w*|rule out|excluded?)\b", re.I)
_DOSE = re.compile(r"\d\s*(?:mg|mcg|µg|g\b|ml|units?|iu|mmol|/kg)", re.I)
_EVIDENCE = re.compile(
    r"\b(RCTs?|meta-?analys\w*|systematic review\w*|cohort|trials?|stud(?:y|ies)|odds ratio|relative risk|"
    r"hazard ratio|95% ?CI|CI\s*[:=]?\s*\d|p\s*[<=>]\s*0?\.\d|n\s*=\s*\d|level [I]{1,3}\b|evidence|"
    r"participants|randomi[sz]ed|placebo)\b", re.I)
_STRONG = re.compile(r"\b(should|must|recommended|avoid|do not|contraindicated|is indicated|are indicated)\b", re.I)
# Front matter and apparatus: committee lists, disclaimers, search method,
# evidence grading, contents pages ("82 - 92 ... 93 - 102").
_APPARATUS = re.compile(
    r"committee|external reviewers?|search strategy|literature (?:search|retrieved)|legally binding|"
    r"disclosure|sources? of funding|appraised|grades? of recommendation|levels? of evidence|secretariat|"
    r"acknowledg|\b(?:dr|prof|datuk|dato'?)\b\.?\s+\w+.*\b(?:dr|prof)\b\.?\s+\w+|"
    r"\b\d{1,3}\s*-\s*\d{1,3}\b.*\b\d{1,3}\s*-\s*\d{1,3}\b", re.I)
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,&/()'\-:]{2,80}$")


def keep_sentence(s: str) -> bool:
    if distill.is_citation(s) or _APPARATUS.search(s):
        return False
    if _EVIDENCE.search(s) and not _STRONG.search(s):
        return False
    return bool(_ACTION.search(s) or _DOSE.search(s))


def norm(text: str) -> str:
    """The comparison form of a quote: NFKC, one space, lowercase, typographic
    quotes and dashes unified. Used on BOTH sides, never shown."""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[-]", " ", t)                   # PUA bullets
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[‐-―−]", "-", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:70]


def _collection():
    from .ingest import get_client  # noqa: PLC0415 - no embedding model needed to read
    return get_client().get_collection(config.COLLECTION_NAME)


def load_index() -> dict:
    got = _collection().get(include=["documents", "metadatas"])
    return {"ids": got["ids"], "docs": got["documents"], "metas": [m or {} for m in got["metadatas"]]}


def _covered_by_full(title: str, full_titles: set[str]) -> bool:
    return title in full_titles


# --------------------------------------------------------------- 1. packets
def build_packets(index: dict | None = None) -> list[dict]:
    """One text file per document (large ones split), each chunk under a short
    ref "r<n>" that refs.json maps back to its chunk id."""
    index = index or load_index()
    PACKETS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PACKETS_DIR.glob("*.txt"):
        old.unlink()
    full_titles = {m.get("cpg_title") for m in index["metas"] if m.get("doc_type") == config.DOC_TYPE_CPG_FULL}
    by_doc: dict[tuple, list[int]] = defaultdict(list)
    for i, m in enumerate(index["metas"]):
        dt = m.get("doc_type")
        if dt in _SKIP_TYPES or m.get("status", "active") != "active":
            continue
        # A quick-reference guide is a subset of its full CPG; the full text is
        # what the cards quote. A QR with no full CPG indexed (PPH) is read.
        if dt == config.DOC_TYPE_CPG_QR and _covered_by_full(m.get("cpg_title"), full_titles):
            continue
        by_doc[(dt, m.get("cpg_title", ""))].append(i)

    def order(i: int) -> tuple:
        m = index["metas"][i]
        return (int(m.get("page_number") or 0), str(m.get("section") or ""), int(m.get("chunk_index") or 0))

    refs: dict[str, str] = {}
    packets: list[dict] = []
    n = 0
    for (dt, title), idxs in sorted(by_doc.items(), key=lambda kv: kv[0][1]):
        meta0 = index["metas"][idxs[0]]
        pop = "paediatric" if dt == config.DOC_TYPE_PAEDS else population.title_population(title)
        head = (f"# DOC {title} ({meta0.get('edition_year') or '?'}) | {dt} | population: {pop}"
                + (" | NOT KKM - cited by a KKM document" if dt == config.DOC_TYPE_EXTERNAL else ""))
        parts: list[list[str]] = [[]]
        size = 0
        for i in sorted(idxs, key=order):
            text = index["docs"][i] or ""
            if distill.is_bibliography(text):
                continue
            m = index["metas"][i]
            header, sents = distill.sentences(text)
            lines = []
            for s in sents:
                if keep_sentence(s):
                    lines.append(s)
                elif len(s) <= 90 and _HEADING.match(s):
                    lines.append(f"§ {s}")
            if not any(not ln.startswith("§ ") for ln in lines):
                continue
            n += 1
            ref = f"r{n}"
            refs[ref] = index["ids"][i]
            where = f"p{m.get('page_number')}" if m.get("page_number") else (m.get("section") or "web")
            block = f"## {where} [{ref}]\n" + "\n".join(lines) + "\n"
            if size + len(block) > PACKET_CHARS and parts[-1]:
                parts.append([])
                size = 0
            parts[-1].append(block)
            size += len(block)
        for k, part in enumerate(p for p in parts if p):
            name = slug(title) + (f"-part{k + 1}" if len(parts) > 1 else "")
            path = PACKETS_DIR / f"{name}.txt"
            path.write_text(head + (f" | part {k + 1} of {len(parts)}" if len(parts) > 1 else "")
                            + "\n\n" + "\n".join(part), encoding="utf-8")
            packets.append({"packet": path.stem, "title": title, "doc_type": dt,
                            "chars": sum(len(b) for b in part)})
    REFS.write_text(json.dumps(refs), encoding="utf-8")
    (PACKETS_DIR / "manifest.json").write_text(json.dumps(packets, indent=1), encoding="utf-8")
    return packets


# ---------------------------------------------------------------- 2. verify
_GENERIC_TERMS = frozenset("""patient patients treatment management therapy care assess assessment
consider should the and with for all of dose doses drug drugs medication medications test tests
condition disease acute severe risk sign signs symptom symptoms""".split())


# The local model (cards_local.py) names some elements in its own words; each
# maps to the spec's element meaning the same thing. Anything else - "education",
# "performance measure" - is not patient guidance and stays rejected.
ELEMENT_ALIASES = {
    "management": "treatment", "procedure": "treatment", "surgery": "treatment",
    "prophylaxis": "treatment", "prevention": "treatment", "supplementation": "treatment",
    "pain management": "treatment", "diagnosis": "definition", "diagnostic criteria": "definition",
    "screening": "investigation", "assessment": "investigation", "imaging": "investigation",
    "discharge advice": "discharge", "discharge criteria": "discharge",
}


def _terms(raw) -> list[str]:
    out = []
    for t in raw or []:
        t = norm(str(t)).strip(" .,;:")
        if len(t) >= 2 and t not in _GENERIC_TERMS and t not in out:
            out.append(t)
    return out[:8]


def _term_in(term: str, text_n: str) -> bool:
    # Word-bounded both sides, so "k" never matches "kidney"; a stem ending in
    # a letter may take a suffix ("hyperkalaemi" - agents write the stem).
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}", text_n) is not None


def verify(index: dict | None = None) -> dict:
    index = index or load_index()
    refs = json.loads(REFS.read_text(encoding="utf-8"))
    pos = {cid: i for i, cid in enumerate(index["ids"])}
    normed: dict[int, str] = {}

    def text_n(i: int) -> str:
        if i not in normed:
            normed[i] = norm(index["docs"][i])
        return normed[i]

    doc_rows: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(index["metas"]):
        doc_rows[m.get("cpg_title", "")].append(i)

    stats = Counter()
    rejected: list[dict] = []
    cards: dict[str, dict] = {}
    alias_of: dict[str, str] = {}

    def locate(ref: str, quote_n: str) -> int | None:
        cid = refs.get(ref)
        if cid is None or cid not in pos:
            return None
        i = pos[cid]
        if quote_n in text_n(i):
            return i
        # A chunk boundary or a wrong ref: same document only.
        for j in doc_rows[index["metas"][i].get("cpg_title", "")]:
            if quote_n in text_n(j):
                stats["ref_corrected"] += 1
                return j
        return None

    def card_for(name: str, aliases: list[str]) -> dict:
        key = norm(name)
        target = alias_of.get(key)
        if target is None:
            for a in aliases:
                if norm(a) in cards:          # an alias that IS another card's name
                    target = norm(a)
                    break
        target = target or key
        card = cards.setdefault(target, {"id": slug(target), "name": name, "aliases": [], "parents": [],
                                         "population": set(), "acuity": Counter(), "documents": set(),
                                         "items": []})
        alias_of[key] = target
        return card

    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rejected.append({"file": path.name, "reason": f"invalid JSON: {exc}"})
            continue
        for cond in payload.get("conditions") or []:
            name = str(cond.get("name") or "").strip()
            if not name:
                continue
            aliases = [str(a).strip() for a in cond.get("aliases") or [] if str(a).strip()]
            card = card_for(name, aliases)
            for a in aliases:
                if norm(a) != norm(card["name"]) and norm(a) not in {norm(x) for x in card["aliases"]}:
                    card["aliases"].append(a)
                alias_of.setdefault(norm(a), norm(card["name"]) if norm(card["name"]) in cards else norm(name))
            if cond.get("parent"):
                p = str(cond["parent"]).strip()
                if p and norm(p) != norm(card["name"]) and p not in card["parents"]:
                    card["parents"].append(p)
            if cond.get("population") in POPULATIONS:
                card["population"].add(cond["population"])
            if cond.get("acuity") in ACUITIES:
                card["acuity"][cond["acuity"]] += 1
            for item in cond.get("items") or []:
                stats["items"] += 1
                quote = str(item.get("quote") or "").strip()
                el = str(item.get("element") or "").strip().lower()
                el = ELEMENT_ALIASES.get(el, el)
                terms = _terms(item.get("terms"))
                why = None
                if el not in ELEMENTS:
                    why = f"unknown element {el!r}"
                elif not quote or len(quote) < 12:
                    why = "empty quote"
                elif not terms:
                    why = "no match terms"
                qn = norm(quote)
                idx = None if why else locate(str(item.get("ref") or ""), qn)
                if why is None and idx is None:
                    why = "quote not found verbatim in the cited document"
                if why is None and not any(_term_in(t, qn) for t in terms):
                    why = "no match term occurs in the quote"
                if why:
                    stats["rejected"] += 1
                    rejected.append({"file": path.name, "condition": name, "element": el,
                                     "label": item.get("label"), "reason": why, "quote": quote[:160]})
                    continue
                meta = index["metas"][idx]
                title = meta.get("cpg_title", "")
                # The printed form keeps the source's words: the span of the raw
                # chunk that matches, whitespace collapsed.
                raw = re.sub(r"\s+", " ", re.sub(r"[-]", " ", index["docs"][idx])).strip()
                start = norm(raw).find(qn)
                shown = raw[start:start + len(qn)] if start >= 0 and len(norm(raw)) == len(raw) else quote
                row = {
                    "element": el,
                    "label": str(item.get("label") or terms[0])[:90],
                    "terms": terms,
                    "when": _terms(item.get("when")),
                    "unless": _terms(item.get("unless")),
                    "core": bool(item.get("core")),
                    "setting": item.get("setting") if item.get("setting") in SETTINGS else "any",
                    "population": item.get("population") if item.get("population") in POPULATIONS else "",
                    "quote": shown,
                    "chunk_id": index["ids"][idx],
                    "title": title,
                    "doc_type": meta.get("doc_type"),
                    "page": meta.get("page_number"),
                    "url": meta.get("url", ""),
                    "external": meta.get("doc_type") == config.DOC_TYPE_EXTERNAL,
                }
                dup = next((x for x in card["items"] if x["element"] == el and norm(x["quote"]) == qn), None)
                if dup:
                    stats["duplicate"] += 1
                    continue
                card["items"].append(row)
                card["documents"].add(title)
                stats["kept"] += 1

    out = []
    for card in cards.values():
        if not card["items"]:
            continue
        pops = card["population"] or {"all"}
        card["population"] = next(iter(pops)) if len(pops) == 1 else "all"
        card["acuity"] = card["acuity"].most_common(1)[0][0] if card["acuity"] else "urgent"
        card["documents"] = sorted(card["documents"])
        card["items"].sort(key=lambda x: (ELEMENTS.index(x["element"]), not x["core"], x["external"]))
        out.append(card)
    out.sort(key=lambda c: c["name"].lower())
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (CARDS_DIR / "rejected.json").write_text(json.dumps(rejected, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    cov = coverage(out)
    COVERAGE.write_text(json.dumps(cov, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"cards": len(out), **stats, "coverage": cov["summary"]}


# -------------------------------------------------------------- 3. coverage
# The elements every acute presentation needs an answer for. A chronic
# condition is not expected to state admission criteria.
REQUIRED = {
    "emergency": ("red_flag", "investigation", "treatment", "admission", "monitoring"),
    "urgent": ("red_flag", "investigation", "treatment", "admission"),
    "routine": ("investigation", "treatment", "referral"),
    "chronic": ("treatment", "referral"),
}


def status_of(card: dict, element: str) -> str:
    rows = [x for x in card["items"] if x["element"] == element]
    if not rows:
        return "GAP"
    if all(x["external"] for x in rows):
        return "PARTIAL"            # answered only by a KKM-cited, non-KKM document
    if element == "treatment" and not any(_DOSE.search(x["quote"]) for x in rows):
        return "PARTIAL"            # what to give, but no dose in any KKM sentence
    return "COVERED"


def coverage(cards: list[dict]) -> dict:
    rows, summary = [], Counter()
    for c in cards:
        need = REQUIRED.get(c["acuity"], REQUIRED["urgent"])
        st = {el: status_of(c, el) for el in ELEMENTS}
        gaps = [el for el in need if st[el] == "GAP"]
        rows.append({"id": c["id"], "name": c["name"], "acuity": c["acuity"], "population": c["population"],
                     "documents": c["documents"], "status": st, "gaps": gaps})
        for el in need:
            summary[st[el]] += 1
    return {"summary": dict(summary), "cards": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--packets", action="store_true", help="write extraction packets from the index")
    ap.add_argument("--verify", action="store_true", help="verify agent extracts and write the cards")
    args = ap.parse_args()
    if args.packets:
        packets = build_packets()
        print(f"{len(packets)} packets, {sum(p['chars'] for p in packets):,} characters -> {PACKETS_DIR}")
    elif args.verify:
        print(json.dumps(verify(), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
