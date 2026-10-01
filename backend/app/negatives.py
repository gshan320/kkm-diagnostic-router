"""What the source documents say NOT to give, quoted, and checked against each report.

Why this exists
---------------
The indication gate (indications.py) asks "does anything support this drug?".
This asks the other question: "does any source say to AVOID it here?" - the
MOH heat-illness guideline's "DO NOT administer Paracetamol or Aspirin or other
NSAIDS", the dengue CPGs' warnings against NSAIDs and intramuscular injections.
No clinician is available to sign off a hand-written "do not give" table, so
the table is MINED from the documents themselves, and every warning shown to a
clinician is the source's own sentence with its page.

Build (reads every PDF in raw_pdfs/, a minute or two; the output is
regenerable and git-ignored like the FUKKM JSON):

    python -m app.negatives

Two rules keep it from crying wolf:

  1. A statement applies only when its own sentence - or the text around it
     on the same page - names THIS patient's working diagnosis as a strong
     concept (a phrase, or a long specific word). "Avoid aspirin within 24 h of
     alteplase" sits on a stroke page, and must not fire for "heat stroke".
  2. It WARNS, it never removes. A mined sentence can carry a qualifier the
     pattern does not see ("... unless ..."), so the clinician gets the quote
     and the page, and the decision stays theirs.

Drug classes are matched through a small pharmacology table (NSAIDs ->
ibuprofen, diclofenac, ...). Aspirin is deliberately NOT a member of the NSAID
class here: the ACS CPG warns against NSAIDs in patients for whom aspirin is
the first drug given. A sentence that means aspirin says "aspirin".
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from . import config, indications

OUT = config.RAW_JSON_DIR / "avoid_statements.json"

CLASSES: dict[str, tuple[str, ...]] = {
    "nsaid": ("ibuprofen", "diclofenac", "mefenamic", "naproxen", "indomethacin", "ketorolac",
              "celecoxib", "etoricoxib", "piroxicam", "meloxicam", "parecoxib", "ketoprofen"),
    "corticosteroid": ("prednisolone", "prednisone", "hydrocortisone", "dexamethasone",
                       "methylprednisolone", "betamethasone"),
    "opioid": ("morphine", "fentanyl", "pethidine", "tramadol", "codeine", "oxycodone"),
    "benzodiazepine": ("diazepam", "midazolam", "lorazepam", "clonazepam"),
    "beta blocker": ("propranolol", "metoprolol", "atenolol", "bisoprolol", "carvedilol",
                     "labetalol", "esmolol"),
    "antibiotic": ("amoxicillin", "ampicillin", "amoxicillin clavulanate", "cefuroxime", "ceftriaxone",
                   "cefotaxime", "ceftazidime", "ciprofloxacin", "levofloxacin", "azithromycin",
                   "clarithromycin", "erythromycin", "doxycycline", "metronidazole", "gentamicin",
                   "cloxacillin", "penicillin", "piperacillin", "meropenem", "vancomycin"),
    "anticoagulant": ("heparin", "enoxaparin", "warfarin", "rivaroxaban", "apixaban", "dabigatran",
                      "fondaparinux"),
    "antiplatelet": ("aspirin", "clopidogrel", "ticagrelor", "prasugrel"),
    "diuretic": ("furosemide", "frusemide", "mannitol", "hydrochlorothiazide", "spironolactone",
                 "bumetanide", "indapamide"),
    "antihistamine": ("chlorphenamine", "chlorpheniramine", "promethazine", "diphenhydramine",
                      "cetirizine", "loratadine", "desloratadine", "fexofenadine"),
    "thiazolidinedione": ("pioglitazone", "rosiglitazone"),
    "calcium channel blocker": ("nifedipine", "amlodipine", "diltiazem", "verapamil", "felodipine",
                                "nicardipine"),
}

# How each class is written in guideline prose.
CLASS_PATTERNS: dict[str, str] = {
    "nsaid": r"\bnsaids?\b|non[- ]?steroidal anti[- ]?inflammator|cox[- ]?2 inhibitor",
    "corticosteroid": r"\b(cortico)?steroids?\b",
    "opioid": r"\bopioids?\b|\bopiates?\b",
    "benzodiazepine": r"\bbenzodiazepines?\b",
    "beta blocker": r"\bbeta[- ]?blockers?\b|\bβ[- ]?blockers?\b",
    "antibiotic": r"\bantibiotics?\b|\bantimicrobials?\b",
    "anticoagulant": r"\banticoagula\w*",
    "antiplatelet": r"\bantiplatelets?\b",
    "diuretic": r"\bdiuretics?\b",
    "antihistamine": r"\bantihistamines?\b",
    "thiazolidinedione": r"\bthiazolidinediones?\b|\bglitazones?\b",
    "calcium channel blocker": r"calcium[- ]channel blockers?|\bccbs?\b",
}

_NEGATIVE = re.compile(
    r"\b(avoid(ed|ing)?|do not (give|administer|use|prescribe|start|routinely)|"
    r"should not (be )?(given|used|administered|prescribed|started|routinely)|must not|"
    r"(is|are) contraindicated|contra-?indicated in|not recommended|(is|are) harmful|"
    r"has no role|have no role|no role for|not be given|never (give|be given))\b",
    re.IGNORECASE,
)


def _drug_terms() -> dict[str, re.Pattern]:
    """Single drug names worth looking for: every class member and every class."""
    terms: dict[str, re.Pattern] = {}
    for cls, members in CLASSES.items():
        for m in members:
            terms[m] = re.compile(rf"(?<![a-z]){re.escape(m)}(?![a-z])", re.I)
    for name in ("paracetamol", "acetaminophen", "aspirin", "acetylsalicylic", "salbutamol",
                 "aminophylline", "theophylline", "magnesium", "alteplase", "streptokinase",
                 "tenecteplase", "adrenaline", "atropine", "insulin", "potassium", "sodium bicarbonate",
                 "tranexamic", "vitamin k", "haloperidol", "metoclopramide", "ondansetron",
                 "glyceryl trinitrate", "nitrate", "sildenafil", "digoxin", "amiodarone"):
        terms.setdefault(name, re.compile(rf"(?<![a-z]){re.escape(name)}(?![a-z])", re.I))
    for cls, pattern in CLASS_PATTERNS.items():
        terms[f"class:{cls}"] = re.compile(pattern, re.I)
    return terms


def _sentences(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for part in re.split(r"(?<=[.;])\s+|\n\s*\n|•||●", text):
        clean = re.sub(r"\s+", " ", part).strip()
        if 20 <= len(clean) <= 450:
            out.append((text.find(part, pos), clean))
        pos += len(part)
    return out


def build() -> list[dict]:
    import pymupdf

    terms = _drug_terms()
    rows: list[dict] = []
    for pdf in sorted(Path(config.RAW_PDF_DIR).glob("*.pdf")):
        doc = pymupdf.open(pdf)
        for i, page in enumerate(doc):
            text = re.sub(r"\xad\s*", "", page.get_text())  # soft hyphens split words
            if not _NEGATIVE.search(text):
                continue
            for at, sent in _sentences(text):
                if not _NEGATIVE.search(sent):
                    continue
                named = [t for t, rx in terms.items() if rx.search(sent)]
                if not named:
                    continue
                start = max(0, at - 900)
                rows.append({
                    "source": pdf.name, "page": i + 1, "sentence": sent,
                    "drug_terms": named,
                    # The surrounding text lets a statement's CONDITION be read
                    # from its section, not only from the one sentence.
                    "context": re.sub(r"\s+", " ", text[start:at + len(sent) + 400]),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return rows


@lru_cache(maxsize=1)
def statements() -> list[dict]:
    if not OUT.exists():
        return []
    return json.loads(OUT.read_text(encoding="utf-8"))


def terms_for(drug_name: str) -> set[str]:
    """The table terms that refer to this drug: its own name(s) and its classes."""
    low = (drug_name or "").lower()
    out = set()
    for cls, members in CLASSES.items():
        for m in members:
            if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", low):
                out.add(m)
                out.add(f"class:{cls}")
    for name in ("paracetamol", "acetaminophen", "aspirin", "acetylsalicylic", "salbutamol",
                 "aminophylline", "theophylline", "magnesium", "alteplase", "streptokinase",
                 "tenecteplase", "adrenaline", "atropine", "insulin", "potassium", "sodium bicarbonate",
                 "tranexamic", "vitamin k", "haloperidol", "metoclopramide", "ondansetron",
                 "glyceryl trinitrate", "nitrate", "sildenafil", "digoxin", "amiodarone"):
        if name in low:
            out.add(name)
    if "acetaminophen" in out:
        out.add("paracetamol")
    if "acetylsalicylic" in out:
        out.add("aspirin")
    return out


_AMBIGUOUS = frozenset({
    "stroke", "attack", "arrest", "failure", "infection", "injury", "fever",
    # Adjectives that name a body system, not a condition: they appear on the
    # CKD, AF and ACS pages alike.
    "myocardial", "ischemic", "ischaemic", "coronary", "cerebral", "hypertension",
    "hypertensive", "diabetes", "diabetic", "emergency", "urgency",
})


def strong_mention(concept_set: set[str], text: str, phrases_only: bool = False) -> str | None:
    """Like indications.mentions, but a single word only counts when it is long
    and unambiguous - "stroke" alone would join heat stroke to the stroke CPG.
    `phrases_only` is used for the text AROUND a statement: a page is long
    enough to contain almost any single word, so there only a phrase counts."""
    strong = {c for c in concept_set
              if " " in c or (not phrases_only and len(c) >= 6 and c not in _AMBIGUOUS)}
    return indications.mentions(strong, text)


def _title(source: str) -> str:
    return re.sub(r"\.pdf$", "", source, flags=re.I)


def check(drug_name: str, working: set[str], paediatric: bool = False, limit: int = 2) -> list[dict]:
    """Mined statements that advise against this drug for this working diagnosis,
    strongest first, at most `limit` - two quotes make the point, six build
    alarm fatigue.

    Where the diagnosis is found decides the rank: in the warning sentence
    itself (strongest), in the document's title, or as a phrase in the text
    around it. An adult never gets a statement from the Paediatric Protocols."""
    wanted = terms_for(drug_name)
    if not wanted or not working:
        return []
    ranked = []
    for row in statements():
        if not wanted.intersection(row["drug_terms"]):
            continue
        if not paediatric and "paediatric protocol" in row["source"].lower():
            continue
        for rank, concept in (
            (0, strong_mention(working, row["sentence"])),
            (1, strong_mention(working, _title(row["source"]))),
            (2, strong_mention(working, row["context"], phrases_only=True)),
        ):
            if concept:
                ranked.append((rank, {**row, "concept": concept, "matched_in":
                                      ("sentence", "document title", "surrounding text")[rank]}))
                break
    ranked.sort(key=lambda x: x[0])
    seen, out = set(), []
    for _, row in ranked:
        key = re.sub(r"\W+", " ", row["sentence"].lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    rows = build()
    docs = len({r["source"] for r in rows})
    print(f"{len(rows)} avoid-statements from {docs} documents -> {OUT}")
    sys.exit(0)
