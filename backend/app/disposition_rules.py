"""F1: a disposition floor that rests on a quoted admission criterion.

Why this exists
---------------
Ten live runs of the exertional rhabdomyolysis case (2026-09-30) ended in "ED
observation". The MTS floor (_enforce_disposition) only knows triage levels; a
condition-specific criterion needs its own rule. These are the hand-checked
rules; every other condition's admission criteria come from its condition card
(cards.admission), through the same raise-only floor.

Each rule names the diagnosis it applies to, the finding in the intake that
meets the criterion, the destination, and the source sentence verbatim (tests
check each one against its page). It can only RAISE the disposition - the
same direction as every other floor in this codebase - and the report states
the rule, the quote and the finding that triggered it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RANK = {"DISCHARGE_WITH_FOLLOW_UP": 0, "ED_OBSERVATION": 1, "REFER_SPECIALIST": 2,
        "ADMIT_WARD": 3, "ADMIT_ICU_HDU": 4, "RESUSCITATION_BAY": 4}


@dataclass(frozen=True)
class AdmissionRule:
    name: str
    diagnosis: str      # regex on the working diagnosis
    criterion: str      # regex on the (negation-stripped) intake + diagnosis
    target: str         # Disposition value
    quote: str          # verbatim from the source
    source: str         # title and page, as shown
    title: str          # indexed cpg_title of the source (for tests / labelling)
    page: int
    external: bool = False


RULES: tuple[AdmissionRule, ...] = (
    AdmissionRule(
        "eclampsia",
        r"(?<!pre-)(?<!pre )(?<!pre)(?<!\w)eclampsi",
        r"eclampsi|\bfit(?:s|ted|ting)?\b|seizure|convuls",
        "ADMIT_ICU_HDU",
        "ICU admission is indicated in the following: Eclampsia; Cerebrovascular accident; "
        "Pulmonary oedema; Aspiration pneumonitis/Acute Respiratory Distress Syndrome (ARDS); "
        "HELLP Syndrome; Renal complications e.g. Acute Kidney Injury (AKI)",
        "MOH Training Manual Hypertensive Disorders in Pregnancy 2018 p62",
        "MOH Training Manual Hypertensive Disorders in Pregnancy", 62,
    ),
    AdmissionRule(
        "severe pre-eclampsia / HELLP",
        r"(?:severe|imminent)\s+pre-?\s?eclampsi|\bHELLP\b",
        r"(?:severe|imminent)\s+pre-?\s?eclampsi\w*|\bHELLP\b",
        "ADMIT_ICU_HDU",
        "The patient should be admitted and managed in a high dependency area.",
        "MOH Training Manual Hypertensive Disorders in Pregnancy 2018 p41",
        "MOH Training Manual Hypertensive Disorders in Pregnancy", 41,
    ),
)


def applicable(diagnosis: str, intake: str) -> tuple[AdmissionRule, str] | None:
    """The first rule for this diagnosis whose criterion the intake meets, with
    the phrase that met it."""
    for rule in RULES:
        if not re.search(rule.diagnosis, diagnosis or "", re.I):
            continue
        m = re.search(rule.criterion, f"{intake} {diagnosis}", re.I)
        if m:
            return rule, m.group(0)
    return None
