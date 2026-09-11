"""Who may NOT be redirected away from an Emergency and Trauma Department.

Source: JKN Selangor, "Emergency Medicine and Trauma Services: Redirection
Policy", July 2024, section 4.2 - "Non-critical emergency presentations which
have to be seen in ETDs (Exclusion criteria for case redirection)". Sixteen
numbered conditions, reproduced below one rule at a time with their clause
numbers so a reader can check each against the document.

WHY THIS IS IN CODE. The tool's triage-away path had no written destination
until this document entered the corpus on 2026-09-11: nothing said what "refer
to outpatient services" was allowed to contain, and a DISCHARGE_WITH_FOLLOW_UP
on an MTS 4 or 5 patient was unchecked. Section 4.2 is the check. It is short,
categorical and stated as a list of patient types, which is exactly the shape
that belongs in code rather than in a prompt - the same argument as red_flags.py
and contraindications.py.

WHY IT WARNS AND DOES NOT BLOCK. This is a STATE policy. It binds ETDs in
Selangor and nowhere else, and MTS 2022 - which is national - contains no
redirection criteria at all. A hard override would impose a Selangor rule on a
user in Kelantan. So a matching rule attaches a warning naming the clause, and
the disposition is left as the report set it. `facility` is not used to gate the
warning either: the enum records the KIND of facility, not the state.

WHAT IT DOES NOT COVER. Four of the sixteen clauses need facts the intake does
not carry, and they are listed in UNCHECKABLE below rather than silently
dropped. A clinician reading the warning needs to know the list was not
exhausted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# The dispositions that send a patient out of the department. REFER_SPECIALIST
# is included: section 4.1 redirects to "klinik kesihatan OR SPECIALIST
# OUTPATIENT CLINIC", so an outpatient referral is a redirection too.
REDIRECTING_DISPOSITIONS = frozenset(
    {"DISCHARGE_WITH_FOLLOW_UP", "REFER_SPECIALIST"}
)

POLICY = "JKN Selangor ETD Redirection Policy, July 2024"


@dataclass(frozen=True)
class Exclusion:
    """One clause of section 4.2."""

    clause: str  # the numbered clause, so the warning can be checked
    what: str  # the clause text, near enough verbatim to be recognisable
    test: Callable[["_Ctx"], bool]
    why: str = ""


@dataclass(frozen=True)
class _Ctx:
    """Everything a rule may read. Built from the TriageRequest by `check()`."""

    age: float
    # complaint + history + trauma mechanism, joined as written. NOT lowercased:
    # every pattern here is IGNORECASE, and `_NEGATION` needs the original
    # punctuation to find where a negated clause ends.
    text: str
    comorbidities: tuple[str, ...]
    arrival_mode: str
    pregnant: bool
    pregnancy_unestablished: bool
    trauma_mechanism: str


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


# A complaint field records what the patient does NOT have as often as what they
# do: "Heartburn after meals, no chest pain" matched clause 4.2.6 until this
# existed. The window stops at the nearest comma, semicolon or full stop, so
# "no fever, chest pain since morning" still fires on the chest pain.
_NEGATION = re.compile(
    r"\b(?:no|not|without|denies|denied|nil|negative for|free of|absent|"
    r"resolved|never)\b[^.,;:]{0,20}$",
    re.IGNORECASE,
)


# The other direction: a finding the intake explicitly rules out AFTER naming
# it. "pregnancy excluded" is the case that matters, because clause 4.2.16
# matches on `\bpregnan` - the same substring recorded as a trap in the triage
# modifiers work, where it produced false ABSOLUTE teratogen alerts.
#
# The lean here is OPPOSITE to that one and deliberately so. 4.2.16 is "ALL
# pregnancy related cases", so "suspected ectopic pregnancy" and "pregnancy
# status not established" SHOULD flag - a patient who might be pregnant is
# exactly who the clause protects. Only an explicit exclusion clears it.
#
# Deliberately does NOT include "resolved" or "settled": "chest pain resolved"
# means the patient HAD chest pain, which is still clause 4.2.6. "Excluded" and
# "ruled out" mean the finding was never there.
_NEGATION_AFTER = re.compile(
    r"^[^.,;:]{0,14}\b(?:excluded|ruled out|negative|not present)\b",
    re.IGNORECASE,
)


def _hit(pattern: re.Pattern, text: str) -> bool:
    """True when `pattern` matches somewhere it is not being negated."""
    return any(
        not _NEGATION.search(text[:m.start()])
        and not _NEGATION_AFTER.match(text[m.end():])
        for m in pattern.finditer(text)
    )


# Deliberately broad. A false warning costs a clinician one sentence to dismiss;
# a missed one sends a patient the policy names to a clinic that cannot take
# them. Both `\b` anchored, for the reason recorded against the `\bpregnan`
# substring trap in the triage-modifiers work.
_CHEST_PAIN = _rx(r"\bchest pain\b", r"\bchest discomfort\b", r"\bchest tightness\b",
                  r"\bangina\b", r"\bretrosternal\b")
_ABDO_PAIN = _rx(r"\babdominal pain\b", r"\babdo pain\b", r"\bstomach ache\b",
                 r"\bepigastric pain\b", r"\bcolicky pain\b", r"\bloin pain\b")
_ASTHMA = _rx(r"\basthma\b", r"\bwheez", r"\bbronchospasm\b")
_AUR = _rx(r"\bacute urinary retention\b", r"\bunable to (?:pass|void) urine\b",
           r"\burinary retention\b", r"\bAUR\b")
_FEVER = _rx(r"\bfever\b", r"\bfebrile\b", r"\bpyrexia\b", r"\bdemam\b")
_FEVER_WARNING = _rx(r"\bwarning sign", r"\bbleed", r"\bpetechia", r"\brash\b",
                     r"\blethargy\b", r"\bdrowsy\b", r"\bvomit", r"\bdehydrat",
                     r"\bseizure", r"\bneck stiff", r"\bconfus", r"\bdengue\b")
# "All trauma cases" (4.2.15) is the broadest clause in section 4.2, and the
# first draft of this pattern missed "twisted ankle playing football" - a
# mechanism stated without any of the words "trauma", "injury" or "fall". The
# vocabulary below is therefore mechanism-first. `\bburn\b` does not match
# "heartburn" (no word boundary inside it); "slipped" and "tripped" are spelled
# out because bare "slip" and "trip" match ordinary prose.
_TRAUMA = _rx(
    r"\btrauma\b", r"\binjur", r"\bfracture\b", r"\bwound\b", r"\bfell\b",
    r"\bfall(?:en|ing|s)?\b", r"\baccident\b", r"\bassault", r"\bstab",
    r"\bburn\b", r"\bscald", r"\blaceration\b", r"\bRTA\b", r"\bMVA\b",
    r"\bhead injury\b", r"\btwist(?:ed|ing)?\b", r"\bsprain",
    r"\b(?:muscle|back|neck|groin|hamstring|calf)\s+strain\b",
    r"\bdislocat", r"\bcontusion", r"\bbruis", r"\bcrush", r"\bcut\b",
    r"\bbit(?:e|ten)\b", r"\bsting\b", r"\bstung\b", r"\bslipped\b",
    r"\btripped\b", r"\bcollision\b", r"\bmotorcycle\b", r"\bmotorbike\b",
    r"\bstruck\b", r"\bhit by\b", r"\bblunt\b", r"\bpenetrat",
    r"\bforeign body\b", r"\bamputat", r"\bdegloving\b",
)
_PREGNANCY = _rx(r"\bpregnan", r"\bgravida\b", r"\bper vaginal\b", r"\bPV bleed",
                 r"\bliquor\b", r"\blabour\b", r"\bmiscarriage\b", r"\bectopic\b",
                 r"\bpostpartum\b", r"\bantenatal\b")
_WHEELCHAIR = _rx(r"\bwheelchair\b", r"\bbedbound\b", r"\bbed[- ]bound\b",
                  r"\bbedridden\b", r"\bimmobile\b")


# Section 4.1 Step 3(b) names wound CARE as explicitly redirectable - "Routine
# wound dressing", "STO", "Chronic trauma injuries > 1 month" - while 4.2.15
# excludes "All trauma cases". The document resolves its own tension by context:
# 4.2.15 is about an ACUTE trauma presentation, and 4.1 is about the follow-up
# care of one that has already been treated. Without this carve-out the guard
# warned on every dressing change, which is the single most common thing a
# redirection policy exists to redirect.
_SECTION_41_WOUND_CARE = _rx(
    r"\bwound dressing\b", r"\bdressing change\b", r"\bchange of dressing\b",
    r"\bSTO\b", r"\bsuture (?:removal|to open)\b", r"\bremoval of sutures?\b",
    r"\bchronic\b[^.,;]{0,30}\b(?:trauma|injur)",
)


EXCLUSIONS: tuple[Exclusion, ...] = (
    Exclusion(
        "4.2.1", "Wheelchair dependent",
        lambda c: _hit(_WHEELCHAIR, c.text),
        "Read from the complaint text; the intake has no mobility field.",
    ),
    Exclusion(
        "4.2.2", "Children under 5 years old",
        lambda c: c.age < 5,
    ),
    Exclusion(
        "4.2.3", "Children with comorbidities",
        lambda c: c.age < 12 and bool(c.comorbidities),
    ),
    Exclusion(
        "4.2.5", "Geriatric, over 60 years old",
        lambda c: c.age > 60,
    ),
    Exclusion(
        "4.2.6", "Chest pain",
        lambda c: _hit(_CHEST_PAIN, c.text),
    ),
    Exclusion(
        "4.2.7", "Acute abdominal pain",
        lambda c: _hit(_ABDO_PAIN, c.text),
    ),
    Exclusion(
        "4.2.8", "One Stop Crisis Centre (OSCC) case",
        lambda c: c.arrival_mode == "oscc",
    ),
    Exclusion(
        "4.2.9", "Mild acute exacerbation of asthma",
        lambda c: _hit(_ASTHMA, c.text),
    ),
    Exclusion(
        "4.2.10", "Acute urinary retention",
        lambda c: _hit(_AUR, c.text),
    ),
    Exclusion(
        "4.2.11", "Fever with warning signs",
        lambda c: _hit(_FEVER, c.text) and _hit(_FEVER_WARNING, c.text),
        "Fever alone is not an exclusion; fever WITH a warning sign is.",
    ),
    Exclusion(
        "4.2.13", "Referral letter addressed to the ETD",
        lambda c: c.arrival_mode == "referred_from_clinic",
        "The intake records that the patient was referred from a clinic. "
        "Whether the letter names the ETD is not recorded, so this fires on "
        "the referral itself.",
    ),
    Exclusion(
        "4.2.14", "Police or medicolegal case",
        lambda c: c.arrival_mode == "police_okt",
    ),
    Exclusion(
        "4.2.15", "All trauma cases",
        lambda c: (
            (_hit(_TRAUMA, c.text) or bool(c.trauma_mechanism.strip()))
            and not _hit(_SECTION_41_WOUND_CARE, c.text)
        ),
        "Acute trauma. Section 4.1 Step 3(b) sends routine wound dressing, "
        "suture removal and trauma older than a month the other way.",
    ),
    Exclusion(
        "4.2.16", "All pregnancy related cases",
        lambda c: c.pregnant or _hit(_PREGNANCY, c.text),
    ),
)

# Clauses this module cannot evaluate, named in the warning so the list is not
# silently treated as complete. Adding an intake field for any of them is the
# only thing that would move one of these into EXCLUSIONS.
UNCHECKABLE: tuple[tuple[str, str], ...] = (
    ("4.2.4", "children aged 5-12 arriving after klinik kesihatan hours - the "
              "intake records no time of day"),
    ("4.2.12", "a second visit for the same complaint within 72 hours - the "
               "intake records no previous attendance"),
)


def _age(value) -> float:
    """An intake that cannot state an age must not take the report down with it."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _context(req) -> _Ctx:
    """Duck-typed off TriageRequest, the way mts_table is: no schema import."""
    parts = [
        str(getattr(req, "complaint", "") or ""),
        str(getattr(req, "history", "") or ""),
        str(getattr(req, "trauma_mechanism", "") or ""),
    ]
    arrival = getattr(req, "arrival_mode", None)
    return _Ctx(
        age=_age(getattr(req, "age", 0)),
        text=" \n".join(parts),
        comorbidities=tuple(
            str(getattr(c, "value", c)) for c in getattr(req, "comorbidities", ()) or ()
        ),
        arrival_mode=str(getattr(arrival, "value", arrival) or "").lower(),
        pregnant=bool(getattr(req, "pregnant", False)),
        pregnancy_unestablished=bool(getattr(req, "pregnancy_unestablished", False)),
        trauma_mechanism=str(getattr(req, "trauma_mechanism", "") or ""),
    )


def matches(req) -> list[Exclusion]:
    """Every section 4.2 clause this patient meets. Empty is the common case."""
    ctx = _context(req)
    out = []
    for rule in EXCLUSIONS:
        try:
            if rule.test(ctx):
                out.append(rule)
        except Exception:  # a malformed intake must not break the report
            continue
    return out


def warning(hits: list[Exclusion], disposition: str) -> str:
    """The clinician-facing sentence, naming every clause by number."""
    if not hits:
        return ""
    listed = "; ".join(f"{e.clause} {e.what}" for e in hits)
    unchecked = ", ".join(f"{c} ({w})" for c, w in UNCHECKABLE)
    return (
        f"This report proposes {disposition.replace('_', ' ').lower()}, which is a "
        f"redirection out of the ETD. The {POLICY} section 4.2 names this patient "
        f"as one who must be SEEN IN THE ETD and not redirected: {listed}. "
        f"That policy binds ETDs in Selangor; elsewhere treat it as a prompt to "
        f"justify the decision rather than as a rule. Two clauses could not be "
        f"checked from this intake: {unchecked}."
    )
