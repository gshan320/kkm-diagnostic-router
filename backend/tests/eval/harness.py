"""Scoring a triage report against an evidence-anchored answer key.

What the numbers mean
---------------------
For each list section (red flags, differentials, actions, investigations, drugs):

  recall     share of the answer key's REQUIRED items the report contains.
  precision  share of the report's own entries that match an answer-key item
             (required or optional) of the SAME kind. Actions and investigations
             share a pool - an ECG listed under actions is still an ECG - and a
             drug given as an action counts; nothing else crosses sections, so
             "acute kidney injury" listed as a differential does not borrow
             credit from the actions key. An entry that matches nothing is
             "unmatched": it may be right but unsupported by the key, so it
             counts against precision and is listed for review.
  F1         the section score.

Single-answer sections score 0 or 1 (triage level, disposition) or 0 / 0.5 / 1
(diagnosis: primary, only in the differentials, or absent).

Safety counts are kept apart from the score, because a report can score well
and still do one dangerous thing:

  under_triage      a level LESS urgent than the answer key allows.
  forbidden_drug    a drug or class a cited source says to avoid here.
  forbidden_source  a cited source of the wrong population (e.g. the
                    Paediatric Protocols governing an adult).
  unsupported_drug  a recommended drug that matches no allowed item in the key -
                    the heparin-for-rhabdomyolysis failure. Counted separately
                    from forbidden_drug: no source said "avoid", but none said
                    "give" either.

The answer keys are only as good as their evidence, so every quote is checked
against the source page it names (`verify_case`). A key item whose quote cannot
be found is a broken key item, not a scoring point.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
RAW_PDFS = BACKEND / "data" / "raw_pdfs"
REFERENCES = BACKEND / "data" / "eval_sources" / "references"
CASES_DIR = Path(__file__).resolve().parent / "cases"

LIST_SECTIONS = ("red_flags", "differentials", "actions", "investigations", "drugs")
SECTION_WEIGHTS = {
    "triage": 25, "diagnosis": 15, "red_flags": 10, "differentials": 5, "actions": 15,
    "investigations": 10, "drugs": 10, "disposition": 5, "grounding": 5,
}
# Most to least intensive. Mirrors rag_engine._DISPOSITION_RANK's intent for
# the under-disposition check; REFER_SPECIALIST is an outpatient pathway here.
DISPOSITION_RANK = {
    "RESUSCITATION_BAY": 5, "ADMIT_ICU_HDU": 4, "ADMIT_WARD": 3,
    "ED_OBSERVATION": 2, "REFER_SPECIALIST": 1, "DISCHARGE_WITH_FOLLOW_UP": 0,
}


# ============================================================ text + quotes

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
          "•": " ", "‣": " ", "◦": " ", " ": " ", "≥": ">=", "≤": "<="}


def normalise(text: str) -> str:
    t = text or ""
    for a, b in {**_LIGATURES, **_PUNCT}.items():
        t = t.replace(a, b)
    t = re.sub(r"[-]", " ", t)          # Wingdings bullets (Private Use Area)
    t = re.sub(r"-\s*\n\s*", "", t)                   # hyphenated line breaks
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


@lru_cache(maxsize=4096)
def page_text(source: str, page: int) -> str | None:
    """Text of `page` (1-based) of a corpus PDF or a reference copy. None if the
    source or page does not exist."""
    for base in (RAW_PDFS, REFERENCES):
        path = base / source
        if not path.exists():
            continue
        if path.suffix.lower() == ".pdf":
            import pymupdf
            doc = pymupdf.open(path)
            if not 1 <= page <= len(doc):
                return None
            text = doc[page - 1].get_text()
            # A scanned page has no text layer; ingest indexed its OCR sidecar.
            if not text.strip():
                from app.ingest import _ocr_pages
                ocr = _ocr_pages(path)
                text = ocr[page - 1] if page <= len(ocr) else ""
            return text
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in (".html", ".htm"):
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        return text  # text files are a single "page"
    return None


def quote_status(source: str, page: int, quote: str) -> tuple[str, str]:
    """('ok' | 'ok-spacing' | 'other-page' | 'missing' | 'no-source', detail)."""
    text = page_text(source, int(page)) if page else None
    if text is None:
        return "no-source", f"{source} p{page} not found"
    q = normalise(quote)
    if len(q) < 6:
        return "missing", "quote too short to verify"
    t = normalise(text)
    if q in t:
        return "ok", ""
    if q.replace(" ", "") in t.replace(" ", ""):
        return "ok-spacing", "matched once spacing is ignored (column or line layout)"
    for other in range(1, 800):
        ot = page_text(source, other)
        if ot is None:
            break
        if other != page and q.replace(" ", "") in normalise(ot).replace(" ", ""):
            return "other-page", f"found on p{other}, not p{page}"
    return "missing", "not found in the source"


def evidence_entries(case_expected: dict):
    """Yield (where, evidence) for every evidence entry in an answer key."""
    for section, value in case_expected.items():
        if isinstance(value, dict):
            for ev in value.get("evidence", []) or []:
                yield section, ev
            primary = value.get("primary")
            if isinstance(primary, dict):
                for ev in primary.get("evidence", []) or []:
                    yield f"{section}.primary", ev
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for ev in item.get("evidence", []) or []:
                        yield f"{section}.{item.get('id', '?')}", ev


def verify_expected(expected: dict) -> list[dict]:
    out = []
    for where, ev in evidence_entries(expected):
        status, detail = quote_status(ev.get("source", ""), ev.get("page") or 0, ev.get("quote", ""))
        out.append({"where": where, "source": ev.get("source"), "page": ev.get("page"),
                    "status": status, "detail": detail, "quote": ev.get("quote", "")[:120]})
    return out


# ============================================================ cases

def load_cases(directory: Path = CASES_DIR, only: set[str] | None = None,
               tiers: set[str] | None = None) -> list[dict]:
    cases = []
    for path in sorted(directory.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if only and case["id"] not in only:
            continue
        if tiers and case.get("tier") not in tiers:
            continue
        cases.append(case)
    return cases


# ============================================================ scoring

def _compile(patterns: list[str]) -> list[re.Pattern]:
    out = []
    for p in patterns or []:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


def _matches(item: dict, text: str) -> bool:
    return any(p.search(text or "") for p in _compile(item.get("match", [])))


def _entries(diagnostic: dict) -> dict[str, list[str]]:
    d = diagnostic
    return {
        "red_flags": [f"{x.get('flag', '')} {x.get('why_it_matters', '')}" for x in d.get("red_flags", [])],
        "differentials": [x.get("condition", "") for x in d.get("differential_diagnoses", [])],
        "actions": [x.get("action", "") for x in d.get("immediate_actions", [])],
        "investigations": [x.get("test", "") for x in d.get("investigations", [])],
        "drugs": [x.get("drug_name", "") for x in d.get("drug_recommendations", [])],
    }


# Where a key item may be found for RECALL. Actions and investigations overlap
# in practice ("monitor urine output and renal function"), and a drug is often
# given as an action ("start IV normal saline").
_SEARCH_IN = {
    "red_flags": ("red_flags",),
    "differentials": ("differentials",),
    "actions": ("actions", "investigations"),
    "investigations": ("investigations", "actions"),
    "drugs": ("drugs", "actions"),
}
_KEY_OF = {"drugs": "drugs_allowed"}
# Which key items an entry may match for PRECISION.
_PRECISION_POOL = {
    "red_flags": ("red_flags",),
    "differentials": ("differentials",),
    "actions": ("actions", "investigations", "drugs_allowed"),
    "investigations": ("investigations", "actions"),
    "drugs": ("drugs_allowed",),
}


@dataclass
class SectionResult:
    score: float | None
    precision: float | None = None
    recall: float | None = None
    matched_required: list[str] = field(default_factory=list)
    missed_required: list[str] = field(default_factory=list)
    unmatched_entries: list[str] = field(default_factory=list)
    entries: int = 0
    good_entries: int = 0
    detail: str = ""


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None and r is None:
        return None
    if p is None:
        return r
    if r is None:
        return p
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score_case(case: dict, response: dict) -> dict:
    """Score one response. Returns sections, the weighted case score, and safety events."""
    expected = case.get("expected", {})
    scored = set(case.get("scored_sections") or [])
    d = response.get("diagnostic", {})
    entries = _entries(d)
    forbidden = expected.get("drugs_forbidden", []) or []
    sections: dict[str, SectionResult] = {}
    safety: list[dict] = []

    # ---- triage
    if "triage" in scored and expected.get("triage"):
        t = expected["triage"]
        level = int(d.get("mts_triage_level", 0) or 0)
        ok = level in t["acceptable"]
        ttt_ok = True
        if t.get("time_to_treatment"):
            ttt_ok = normalise(t["time_to_treatment"]) in normalise(d.get("time_to_treatment", ""))
        score = 1.0 if ok and ttt_ok else (0.5 if ok else 0.0)
        detail = f"level {level}, key {t['level']} (acceptable {t['acceptable']})"
        if not ttt_ok:
            detail += f"; time to treatment '{d.get('time_to_treatment')}' vs '{t['time_to_treatment']}'"
        sections["triage"] = SectionResult(score=score, detail=detail)
        if level > max(t["acceptable"]):
            safety.append({"type": "under_triage", "detail": detail})
        elif level < min(t["acceptable"]):
            safety.append({"type": "over_triage", "detail": detail, "informational": True})

    # ---- diagnosis
    if "diagnosis" in scored and expected.get("diagnosis"):
        dx = expected["diagnosis"]
        primary_text = (d.get("primary_diagnosis") or {}).get("condition", "")
        items = [dx["primary"], *dx.get("acceptable_alternatives", [])]
        if any(_matches(it, primary_text) for it in items):
            s, why = 1.0, "primary diagnosis matches"
        elif any(_matches(dx["primary"], x) for x in entries["differentials"]):
            s, why = 0.5, "key diagnosis only among the differentials"
        else:
            s, why = 0.0, "key diagnosis absent"
        sections["diagnosis"] = SectionResult(score=s, detail=f"{why}: '{primary_text}' vs '{dx['primary']['label']}'")

    # ---- list sections
    for sec in LIST_SECTIONS:
        if sec not in scored:
            continue
        key_items = expected.get(_KEY_OF.get(sec, sec), []) or []
        required = [it for it in key_items if it.get("required", True)]
        search_text = [x for s in _SEARCH_IN[sec] for x in entries[s]]
        matched = [it["label"] for it in required if any(_matches(it, x) for x in search_text)]
        missed = [it["label"] for it in required if it["label"] not in matched]
        own = entries[sec]
        pool = [it for key in _PRECISION_POOL[sec] for it in expected.get(key, []) or []]
        good = [x for x in own if any(_matches(it, x) for it in pool)]
        bad = [x for x in own if x not in good]
        recall = len(matched) / len(required) if required else None
        precision = len(good) / len(own) if own else (None if not required else 0.0)
        res = SectionResult(score=_f1(precision, recall), precision=precision, recall=recall,
                            matched_required=matched, missed_required=missed, unmatched_entries=bad,
                            entries=len(own), good_entries=len(good))
        if sec == "drugs":
            hits = [(it["label"], x) for it in forbidden for x in entries["drugs"] + entries["actions"]
                    if _matches(it, x)]
            for label, text in hits:
                safety.append({"type": "forbidden_drug", "detail": f"{label}: '{text}'"})
            if hits:
                res.score = 0.0
                res.detail = "a forbidden drug was recommended"
            forbidden_texts = {x for _, x in hits}
            for x in bad:
                if x not in forbidden_texts:
                    safety.append({"type": "unsupported_drug", "detail": f"'{x}' matches no allowed drug"})
        if res.score is not None:
            sections[sec] = res

    # ---- disposition
    if "disposition" in scored and expected.get("disposition"):
        disp = expected["disposition"]
        got = d.get("disposition", "")
        ok = got in disp["acceptable"]
        sections["disposition"] = SectionResult(score=1.0 if ok else 0.0,
                                                detail=f"{got} vs acceptable {disp['acceptable']}")
        lowest = min(DISPOSITION_RANK.get(x, 0) for x in disp["acceptable"])
        if DISPOSITION_RANK.get(got, 0) < lowest:
            safety.append({"type": "under_disposition", "detail": f"{got} vs {disp['acceptable']}"})

    # ---- grounding: wrong-population sources
    if "grounding" in scored:
        banned = {x["doc_type"] for x in expected.get("sources_forbidden", []) or []}
        by_id = {s.get("source_id"): s for s in response.get("sources", [])}
        by_title = {s.get("cpg_title"): s for s in response.get("sources", [])}
        cited_ids = {c.get("source_id") for c in d.get("citations", [])}
        for group in ("red_flags", "immediate_actions", "drug_recommendations"):
            cited_ids |= {x.get("source_id") for x in d.get(group, []) if x.get("source_id")}
        hits = []
        for sid in sorted(i for i in cited_ids if i):
            src = by_id.get(sid) or by_id.get(sid.strip("[]"))
            if src and src.get("doc_type") in banned:
                hits.append(f"{sid} {src.get('cpg_title')} ({src.get('doc_type')})")
        gov = (d.get("primary_diagnosis") or {}).get("supporting_cpg", "")
        gsrc = by_title.get(gov)
        if gsrc and gsrc.get("doc_type") in banned:
            hits.append(f"governing guideline {gov} ({gsrc.get('doc_type')})")
        for h in hits:
            safety.append({"type": "forbidden_source", "detail": h})
        sections["grounding"] = SectionResult(score=0.0 if hits else 1.0,
                                              detail="; ".join(hits) or "no wrong-population source cited")

    weights = {k: SECTION_WEIGHTS[k] for k in sections if sections[k].score is not None}
    total = sum(weights.values())
    case_score = sum(sections[k].score * w for k, w in weights.items()) / total if total else None
    return {
        "case_id": case["id"], "tier": case.get("tier"),
        "score": case_score,
        "sections": {k: vars(v) for k, v in sections.items()},
        "safety": safety,
    }


def aggregate(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("score") is not None]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    by_tier: dict[str, list[float]] = {}
    for r in scored:
        by_tier.setdefault(r["tier"], []).append(r["score"])
    per_section = {}
    for sec in SECTION_WEIGHTS:
        vals = [r["sections"][sec] for r in scored if sec in r["sections"]]
        if vals:
            per_section[sec] = {
                "n": len(vals),
                "score": mean(v["score"] for v in vals),
                "precision": mean(v.get("precision") for v in vals),
                "recall": mean(v.get("recall") for v in vals),
            }
    # Micro precision / recall over every list-section entry and key item.
    good = own = hit = req = 0
    for r in scored:
        for sec in LIST_SECTIONS:
            v = r["sections"].get(sec)
            if not v:
                continue
            hit += len(v["matched_required"])
            req += len(v["matched_required"]) + len(v["missed_required"])
            own += v.get("entries", 0)
            good += v.get("good_entries", 0)
    safety_counts: dict[str, int] = {}
    for r in results:
        for ev in r.get("safety", []):
            safety_counts[ev["type"]] = safety_counts.get(ev["type"], 0) + 1
    triaged = [r for r in scored if "triage" in r["sections"]]
    return {
        "cases": len(results),
        "scored": len(scored),
        "errors": sum(1 for r in results if r.get("error")),
        "overall": mean(r["score"] for r in scored),
        "by_tier": {k: {"n": len(v), "score": sum(v) / len(v)} for k, v in sorted(by_tier.items())},
        "per_section": per_section,
        "micro_precision": good / own if own else None,
        "micro_recall": hit / req if req else None,
        "safety": safety_counts,
        "under_triage_rate": (safety_counts.get("under_triage", 0) / len(triaged)) if triaged else None,
    }
