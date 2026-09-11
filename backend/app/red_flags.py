"""Deterministic recall floor for can't-miss presentations.

Layers 1-3 of retrieval (medical embeddings, BM25, cross-encoder rerank) make a
miss rare. None of them makes a miss *impossible* - no embedder offers a recall
guarantee. This module is the guarantee.

Each rule is a plain, auditable statement: "if the intake says X, the guideline
for Y is put in front of the model, whatever the retriever scored it." A
practitioner can read these rules, argue with them, and version them. That is
the point - the safety net is not allowed to be a black box.

Rules only ever ADD sources. They never remove or reorder what retrieval found,
so a firing rule cannot mask a correct retrieval.

The same shape as `_enforce_mts_level` in rag_engine: clinical safety is
enforced in code, not requested in a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RedFlag:
    name: str
    # Fires when any `triggers` pattern matches. If `require` is non-empty, one
    # of those must ALSO match - that is how combination findings are expressed
    # ("calf swelling" alone is weak; with pleuritic chest pain it is not).
    triggers: tuple[str, ...]
    titles: tuple[str, ...]
    note: str
    require: tuple[str, ...] = ()
    _compiled: dict = field(default_factory=dict, compare=False, repr=False)


def _rx(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ---------------------------------------------------------------------------
# The rules. Titles must match `cpg_title` in the index exactly; validate_titles()
# is called at engine start-up so a filename rename can never silently break the
# safety net.
# ---------------------------------------------------------------------------
RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        name="haemoptysis_or_chronic_cough",
        triggers=(
            r"\bhaemoptysis\b", r"\bhemoptysis\b", r"cough(?:ing)?\s+(?:up\s+)?blood",
            r"blood[- ]stained sputum", r"night sweats?",
            r"cough(?:ing)?[^.]{0,40}\b(?:[3-9]|[1-9]\d)\s*(?:weeks?|months?)",
            r"chronic cough",
        ),
        titles=("Management of Tuberculosis",),
        note="TB is endemic in Malaysia and drives an isolation decision at triage, "
             "not merely a treatment decision. Missing it exposes staff and other patients.",
    ),
    RedFlag(
        name="venous_thromboembolism",
        triggers=(
            r"\bcalf (?:swelling|pain|tenderness)\b", r"unilateral leg swelling",
            r"pleuritic", r"\bdvt\b", r"deep vein thrombosis", r"pulmonary embolism",
            # NOT r"\bpe\b": "PE" is physical examination as often as pulmonary
            # embolism in clinical notes; the spelled-out term above covers it.
            r"long[- ]haul", r"long flight", r"immobilis",
            r"recent surgery", r"bed[- ]bound",
        ),
        titles=("Prevention and Treatment of Venous Thromboembolism (VTE)",),
        note="PE is the can't-miss cause of chest pain and breathlessness. Symptom "
             "wording sits far from this CPG's prophylaxis/treatment vocabulary.",
    ),
    RedFlag(
        name="acute_stroke_syndrome",
        triggers=(
            r"\bhemiplegi", r"\bhemipares", r"facial droop", r"slurred speech",
            r"\bdysarthri", r"\bdysphasi", r"\baphasi", r"sudden(?:ly)? weak",
            r"one[- ]sided weakness", r"\bfast\s+positive", r"visual field loss",
        ),
        titles=(
            "Management of Ischaemic Stroke",
            "Management of Spontaneous Intracerebral Haemorrhage",
        ),
        note="Ischaemic and haemorrhagic stroke present identically and are "
             "opposite treatments. BOTH guidelines must be in front of the model - "
             "thrombolysis into an ICH is catastrophic.",
    ),
    RedFlag(
        name="thunderclap_or_anticoagulated_headache",
        triggers=(
            r"thunderclap", r"worst headache", r"sudden(?:ly)? severe headache",
            r"\bwarfarin\b", r"\bdoac\b", r"anticoagulat",
        ),
        require=(
            r"\bheadache\b", r"\bcollapse\b", r"reduced consciousness", r"\bgcs\b",
            r"\bvomit", r"neuro", r"weak", r"\bfit\b", r"\bseizure",
        ),
        titles=("Management of Spontaneous Intracerebral Haemorrhage",),
        note="Anticoagulation plus any neurological finding is intracranial "
             "haemorrhage until proven otherwise.",
    ),
    RedFlag(
        name="acute_coronary_syndrome",
        triggers=(
            r"chest pain", r"chest discomfort", r"chest tightness",
            r"crushing", r"radiat\w* to (?:the )?(?:jaw|arm|shoulder|back)",
            r"\bdiaphore", r"cold sweat",
        ),
        titles=(
            "Management of Acute Coronary Syndromes",
            "Management of Non ST Elevation Myocardial Infarction (NSTE ACS)",
        ),
        note="STEMI and NSTE-ACS diverge on timing and risk score, and the "
             "distinction is not visible from the presenting complaint alone.",
    ),
    RedFlag(
        name="dengue_syndrome",
        triggers=(
            r"\bdengue\b", r"retro[- ]orbital", r"\bmyalgi", r"\barthralgi",
            r"\bpetechia", r"platelet", r"\bthrombocytopeni", r"tourniquet test",
            r"warning signs?", r"haematocrit", r"hematocrit",
        ),
        require=(r"\bfever\b", r"febrile", r"\btemperature\b", r"\bdengue\b", r"day \d+ of illness"),
        titles=(
            "Management of Dengue Infection in Adults",
            "Management of Dengue Fever in Children",
        ),
        note="Highest-volume undifferentiated febrile presentation in Malaysia. "
             "Both adult and paediatric CPGs are offered; cohort is decided downstream.",
    ),
    RedFlag(
        name="head_injury",
        triggers=(
            r"head injury", r"head trauma", r"\bgcs\b", r"loss of consciousness",
            r"\blocs?\b", r"unequal pupils", r"\banisocori", r"skull",
            r"\brta\b", r"road traffic", r"motorcycle", r"motorbike", r"fell from",
        ),
        titles=("Early Management of Head Injury in Adults",),
        note="Malaysia's road-traffic burden makes this the highest-frequency "
             "major trauma presentation; GCS drives triage acuity directly.",
    ),
    RedFlag(
        name="abdominal_trauma",
        triggers=(
            r"abdominal trauma", r"blunt abdomin", r"penetrating", r"seat ?belt sign",
            r"stab wound", r"\bevisceration\b", r"abdominal distension",
        ),
        titles=("Management of Abdominal Trauma in Adults",),
        note="Haemodynamic instability with abdominal injury is time-critical.",
    ),
    RedFlag(
        name="hyperglycaemic_emergency",
        triggers=(
            r"\bketo", r"\bdka\b", r"kussmaul", r"polyuria", r"polydipsia",
            r"glucose\s*(?:of\s*)?(?:[2-9]\d|\d{3})", r"\bhhs\b", r"blood sugar high",
        ),
        titles=(
            "Management of Type 2 Diabetes Mellitus",
            "Management of Type 1 Diabetes Mellitus in Children and Adolescents",
        ),
        note="DKA presents identically in undiagnosed T1DM and decompensated T2DM.",
    ),
    RedFlag(
        name="acute_heart_failure",
        triggers=(
            r"orthopn", r"\bpnd\b", r"paroxysmal nocturnal", r"bibasal",
            r"\bcrepitation", r"\bcrackles\b", r"pedal o?edema", r"leg swelling",
            r"raised jvp", r"\bjvp\b", r"frothy sputum", r"pink froth",
        ),
        titles=("Management of Heart Failure",),
        note="Acute decompensated heart failure is a top breathlessness cause and "
             "is treated oppositely to an asthma/COPD exacerbation.",
    ),
    RedFlag(
        name="obstructive_airway_disease",
        triggers=(
            r"\bwheez", r"\basthma\b", r"\bcopd\b", r"inhaler", r"\bnebuli",
            r"silent chest", r"peak flow", r"accessory muscle",
        ),
        titles=(
            "Management of Asthma in Adults",
            "Management of Chronic Obstructive Pulmonary Disease (COPD)",
        ),
        note="Asthma and COPD overlap clinically and diverge on oxygen targets.",
    ),
    RedFlag(
        name="fragility_hip_fracture",
        triggers=(
            r"shortened and externally rotated", r"externally rotated",
            r"\bhip (?:pain|fracture)\b", r"neck of femur", r"\bnof\b",
            r"cannot weight ?bear", r"unable to weight ?bear",
        ),
        require=(r"\bfell\b", r"\bfall\b", r"\bfallen\b", r"\bhip\b", r"\bfemur\b"),
        titles=("Management of Geriatric Hip Fracture",),
        note="Elderly fall with a shortened, externally rotated leg is a hip "
             "fracture until imaging says otherwise; delay worsens mortality.",
    ),
    RedFlag(
        name="neonatal_jaundice",
        triggers=(
            r"\bneonat", r"\bnewborn\b", r"day[- ]?\d+ of life", r"\bjaundice",
            r"\bkernicterus\b", r"\bbilirubin\b",
        ),
        require=(r"\bjaundice", r"\byellow", r"\bbilirubin\b", r"\bneonat", r"\bnewborn\b"),
        titles=("Management of Neonatal Jaundice",),
        note="Untreated neonatal hyperbilirubinaemia causes irreversible kernicterus.",
    ),
    RedFlag(
        name="self_harm_risk",
        triggers=(
            r"suicid", r"self[- ]harm", r"overdose", r"took .{0,20}tablets",
            r"wants? to die", r"kill (?:him|her|them)self", r"hopeless",
            # NOT r"\bod\b": in a Malaysian prescription "OD" is omni die
            # (once daily), so it would fire self-harm on routine dosing text.
        ),
        titles=("Management of Major Depressive Disorder",),
        note="Self-harm risk changes triage acuity regardless of physical findings.",
    ),
    RedFlag(
        name="thyroid_emergency",
        triggers=(
            r"thyrotoxic", r"thyroid storm", r"\bgoitre\b", r"heat intoleran",
            r"myx"r"oedema", r"\btsh\b", r"exophthalmos",
        ),
        titles=("Management of Thyroid Disorders",),
        note="Thyroid storm and myxoedema coma are reversible causes of shock and coma.",
    ),
    RedFlag(
        name="atrial_fibrillation",
        triggers=(
            r"palpitation", r"irregular(?:ly)? irregular", r"\batrial fibrillation\b",
            r"\bafib\b", r"\baf\b(?!\w)", r"fast heart rate",
        ),
        titles=("Management of Atrial Fibrillation",),
        note="AF is both a symptom cause and a stroke risk requiring anticoagulation.",
    ),
    RedFlag(
        name="diabetic_foot",
        triggers=(r"foot ulcer", r"diabetic foot", r"\bgangrene\b", r"toe (?:ulcer|black)"),
        titles=("Management of Diabetic Foot",),
        note="Common Malaysian presentation and a frequently missed sepsis source.",
    ),
    RedFlag(
        name="diabetes_in_pregnancy",
        triggers=(r"\bpregnan", r"\bgestation", r"\bantenatal\b", r"\bweeks pregnant\b"),
        require=(r"glucose", r"\bdiabet", r"\bsugar\b", r"\bogtt\b", r"\bgdm\b"),
        titles=("Management of Diabetes in Pregnancy",),
        note="Pregnancy changes both glycaemic targets and safe drug choices.",
    ),
    RedFlag(
        name="renal_impairment",
        triggers=(
            r"\bcreatinine\b", r"\begfr\b", r"reduced urine", r"\boliguri", r"\banuri",
            r"\bdialysis\b", r"chronic kidney", r"\bckd\b", r"hyperkalaem", r"hyperkalem",
        ),
        titles=("Management of Chronic Kidney Disease in Adults",),
        note="Renal function gates the dose of most of the formulary; "
             "hyperkalaemia is immediately life-threatening.",
    ),
)

_COMPILED = [(f, _rx(f.triggers), _rx(f.require)) for f in RED_FLAGS]


def match(text: str) -> list[tuple[RedFlag, str]]:
    """Return the rules the intake fires, each with the phrase that fired it."""
    if not text:
        return []
    fired: list[tuple[RedFlag, str]] = []
    for flag, triggers, require in _COMPILED:
        hit = next((m.group(0) for p in triggers if (m := p.search(text))), None)
        if not hit:
            continue
        if require and not any(p.search(text) for p in require):
            continue
        fired.append((flag, hit))
    return fired


def titles_for(text: str) -> dict[str, list[str]]:
    """cpg_title -> the reasons it was forced in. Order is rule order, stable."""
    out: dict[str, list[str]] = {}
    for flag, hit in match(text):
        for title in flag.titles:
            out.setdefault(title, []).append(f"{flag.name} ({hit!r})")
    return out


def all_titles() -> set[str]:
    return {t for f in RED_FLAGS for t in f.titles}


def validate_titles(known: set[str]) -> list[str]:
    """Every title a rule points at must exist in the index, or the rule is dead
    weight that looks like a safety net. Called at engine start-up."""
    return sorted(all_titles() - known)
