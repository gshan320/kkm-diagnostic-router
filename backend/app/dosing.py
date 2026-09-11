"""Dose parsing and cross-checking against authoritative sources.

The void this closes
--------------------
Prescriber category is now looked up, not reasoned. The DOSE was still the
model's own text: `_check_dose_completeness` caught a *missing* weight-based
dose but never checked that "300mg chewable" matched anything.

The target is not "the model always picks the right dose" - that is a clinical
judgement involving indication, renal function and interactions, and no parser
delivers it. The target is the property that CAN be guaranteed:

    no dose appears on the report without a named source, or an explicit mark
    saying it could not be verified.

Three verdicts, and NOT_COMPARABLE is a real answer
---------------------------------------------------
VERIFIED        the stated dose falls inside a range quoted from FUKKM or the CPG
OUT_OF_RANGE    it matches NEITHER source and no source range contains it
NOT_COMPARABLE  the source says "titrate to effect", or units differ, or nothing
                numeric could be parsed

Silently upgrading NOT_COMPARABLE to VERIFIED would be the whole failure mode
this module exists to prevent.

Why it checks every formulation and the CPG
-------------------------------------------
Measured: "aspirin 300mg" reads OUT OF RANGE against FUKKM's *150mg Dispersible*
entry - but FUKKM also stocks a *300mg Soluble Tablet*, and 300mg is the correct
ACS loading dose in the CPG. A check that consulted one formulation would have
flagged a correct dose, which is the fastest way to teach a clinician to ignore
dose warnings entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit per measurement kind; anything not in the same kind is not
# comparable (mg against ml is a category error, not a discrepancy).
_MASS = {"mcg": 0.001, "microgram": 0.001, "micrograms": 0.001, "ug": 0.001,
         "mg": 1.0, "milligram": 1.0, "milligrams": 1.0,
         "g": 1000.0, "gram": 1000.0, "grams": 1000.0}
_VOLUME = {"ml": 1.0, "millilitre": 1.0, "milliliter": 1.0, "l": 1000.0, "litre": 1000.0}
_UNITS = {"unit": 1.0, "units": 1.0, "iu": 1.0, "u": 1.0}

_ALL_UNITS = "|".join(sorted(set(_MASS) | set(_VOLUME) | set(_UNITS), key=len, reverse=True))
_NUM = r"\d+(?:[.,]\d+)?"
_SEP = r"\s*(?:-|–|—|to|and)\s*"

_PER_KG = re.compile(
    rf"({_NUM})(?:{_SEP}({_NUM}))?\s*({_ALL_UNITS})\s*(?:/|per)\s*kg\b", re.IGNORECASE)
_RANGE = re.compile(rf"({_NUM}){_SEP}({_NUM})\s*({_ALL_UNITS})\b", re.IGNORECASE)
_SINGLE = re.compile(rf"({_NUM})\s*({_ALL_UNITS})\b", re.IGNORECASE)
# A stated ceiling ("maximum of 4 g daily", "not to exceed 15 mg").
_MAX = re.compile(
    rf"(?:max(?:imum)?|not\s+(?:to\s+)?exceed|no\s+dose\s+should\s+exceed|up\s+to)\D{{0,20}}"
    rf"({_NUM})\s*({_ALL_UNITS})\b", re.IGNORECASE)
# Statements that make a numeric comparison meaningless.
_TITRATED = re.compile(
    r"titrat|according to (?:the )?(?:need|response|weight)|individualis|individualiz|"
    r"as (?:required|needed|directed)|adjust(?:ed)? according|to effect|clinical response",
    re.IGNORECASE)


def _kind(unit: str) -> str | None:
    u = unit.lower()
    if u in _MASS:
        return "mass"
    if u in _VOLUME:
        return "volume"
    if u in _UNITS:
        return "units"
    return None


def _canon(value: float, unit: str) -> float:
    u = unit.lower()
    return value * (_MASS.get(u) or _VOLUME.get(u) or _UNITS.get(u) or 1.0)


@dataclass(frozen=True)
class Quantity:
    low: float          # canonical units (mg / ml / units)
    high: float
    kind: str
    per_kg: bool = False
    is_max: bool = False
    text: str = ""

    def absolute(self, weight_kg: float | None) -> "Quantity | None":
        """A per-kg quantity becomes comparable once a weight is known."""
        if not self.per_kg:
            return self
        if not weight_kg:
            return None
        return Quantity(self.low * weight_kg, self.high * weight_kg, self.kind,
                        False, self.is_max, self.text)

    def contains(self, value: float) -> bool:
        # 2% tolerance absorbs rounding ("0.5mg" vs "500mcg"), not clinical drift.
        margin = max(self.high * 0.02, 1e-9)
        return (self.low - margin) <= value <= (self.high + margin)


def _num(raw: str) -> float:
    return float(raw.replace(",", "."))


def parse(text: str) -> list[Quantity]:
    """Every dose quantity in a piece of text, canonicalised."""
    if not text:
        return []
    out: list[Quantity] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(m) -> bool:
        return any(not (m.end() <= s or m.start() >= e) for s, e in consumed)

    maxima = {(m.start(), m.end()) for m in _MAX.finditer(text)}

    for m in _PER_KG.finditer(text):
        k = _kind(m.group(3))
        if not k:
            continue
        lo = _num(m.group(1))
        hi = _num(m.group(2)) if m.group(2) else lo
        out.append(Quantity(_canon(lo, m.group(3)), _canon(hi, m.group(3)), k,
                            per_kg=True, text=m.group(0)))
        consumed.append((m.start(), m.end()))

    for m in _RANGE.finditer(text):
        if overlaps(m):
            continue
        k = _kind(m.group(3))
        if not k:
            continue
        out.append(Quantity(_canon(_num(m.group(1)), m.group(3)),
                            _canon(_num(m.group(2)), m.group(3)), k, text=m.group(0)))
        consumed.append((m.start(), m.end()))

    for m in _SINGLE.finditer(text):
        if overlaps(m):
            continue
        k = _kind(m.group(2))
        if not k:
            continue
        v = _canon(_num(m.group(1)), m.group(2))
        is_max = any(s <= m.start() < e for s, e in maxima)
        out.append(Quantity(v, v, k, is_max=is_max, text=m.group(0)))
        consumed.append((m.start(), m.end()))
    return out


def is_titrated(text: str) -> bool:
    return bool(_TITRATED.search(text or ""))


VERIFIED = "VERIFIED"
EXCEEDS_MAXIMUM = "EXCEEDS_MAXIMUM"
DIFFERS_FROM_SOURCE = "DIFFERS_FROM_SOURCE"
NOT_COMPARABLE = "NOT_COMPARABLE"
OUT_OF_RANGE = DIFFERS_FROM_SOURCE  # retained name for older callers

# Cohort markers. A single FUKKM dosage field routinely carries BOTH an adult
# and a child regimen ("ADULT: 5 to 20 mg ... CHILD: 0.1 - 0.2 mg/kg"). Matching
# a paediatric prescription against the adult sentence VERIFIED morphine 20 mg
# for a 20 kg child - five times the paediatric maximum. Cohort segmentation is
# therefore a safety requirement, not a refinement.
_ADULT_MARK = re.compile(r"\b(adults?)\b\s*:?", re.IGNORECASE)
_CHILD_MARK = re.compile(
    r"\b(child(?:ren)?|paediatric|pediatric|infants?|neonates?|newborns?)\b\s*:?",
    re.IGNORECASE)


def segment_for_cohort(text: str, paediatric: bool) -> tuple[str, bool]:
    """Return the part of a dosage field that applies to this cohort.

    The second value says whether a cohort-specific section was actually found;
    when a paediatric patient has only an adult regimen available, the caller
    must NOT treat a match as verification."""
    if not text:
        return "", False
    marks = sorted(
        [(m.start(), "child") for m in _CHILD_MARK.finditer(text)]
        + [(m.start(), "adult") for m in _ADULT_MARK.finditer(text)]
    )
    # A cohort word only starts a section if that section actually states a
    # dose. "Use in children under 16 years old is not recommended" is a
    # caution, and treating it as a paediatric section left an adult
    # prescription with no applicable dose text at all.
    sections: list[tuple[int, int, str]] = []
    for i, (pos, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if parse(text[pos:end]):
            sections.append((pos, end, kind))
    if not sections:
        return text, False

    # Text before the first real cohort section is general dosing and applies
    # to everyone ("150mg to be taken daily. Use in children ... ").
    prefix = text[: sections[0][0]]
    want = "child" if paediatric else "adult"
    picked = [text[a:b] for a, b, kind in sections if kind == want]
    body = " ".join([prefix] + picked).strip()
    if picked or parse(prefix):
        return body, bool(picked)
    # Only the other cohort is described, and no general dose precedes it.
    return "", True


@dataclass(frozen=True)
class Verdict:
    status: str
    detail: str
    matched_source: str = ""


def check(
    model_dose: str,
    sources: list[tuple[str, str]],
    weight_kg: float | None = None,
    paediatric: bool = False,
) -> Verdict:
    """Compare a stated dose against every authoritative source text.

    `sources` are (label, text) pairs - a FUKKM dosage field, a CPG sentence.
    A dose is VERIFIED if ANY source contains it: FUKKM carries general dosing
    while the CPG carries indication-specific loading doses, and the correct
    answer may appear in only one of them.
    """
    stated = [q for q in parse(model_dose) if not q.is_max]
    if not stated:
        return Verdict(NOT_COMPARABLE, "No numeric dose stated.")
    if not sources:
        return Verdict(NOT_COMPARABLE, "No authoritative dose text available to compare against.")

    exceeded: list[str] = []
    comparable = False
    cohort_blocked: list[str] = []

    for label, raw in sources:
        text, cohort_specific = segment_for_cohort(raw, paediatric)
        if cohort_specific and not text.strip():
            cohort_blocked.append(label)
            continue
        src = parse(text)
        if not src:
            continue
        for want in stated:
            w = want.absolute(weight_kg)
            if w is None:
                continue
            for have in src:
                h = have.absolute(weight_kg)
                if h is None or h.kind != w.kind:
                    continue
                comparable = True
                if h.is_max:
                    if w.low > h.high:
                        exceeded.append(f"{want.text} exceeds the stated maximum {have.text} ({label})")
                    continue
                # A per-kg source range is the authority for a child; an adult
                # absolute range must never validate a paediatric prescription.
                if paediatric and not have.per_kg and any(q.per_kg for q in src):
                    continue
                if h.contains(w.low) and h.contains(w.high):
                    return Verdict(VERIFIED, f"{want.text} matches {have.text}", label)

    if exceeded:
        # The only verdict that should alarm: an explicit ceiling was breached.
        return Verdict(EXCEEDS_MAXIMUM, "; ".join(exceeded))
    if cohort_blocked and not comparable:
        return Verdict(
            NOT_COMPARABLE,
            "The source states a dose for a different age group only "
            f"({', '.join(cohort_blocked)}); it cannot verify this patient's dose.",
        )
    if not comparable:
        return Verdict(NOT_COMPARABLE, "Source dose text is not numerically comparable.")
    if all(is_titrated(t) for _, t in sources):
        return Verdict(NOT_COMPARABLE, "Source specifies an individualised or titrated dose.")
    quoted = "; ".join(f"{lab}: {txt[:110]}" for lab, txt in sources[:2])
    return Verdict(
        DIFFERS_FROM_SOURCE,
        f"Stated {', '.join(q.text for q in stated)} is not within the dose quoted by "
        f"the sources — {quoted}. The formulary states general dosing; an "
        "indication-specific loading dose may legitimately differ. Verify in the CPG.",
    )
