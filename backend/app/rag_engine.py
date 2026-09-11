"""Dual-mode retrieval + prompt construction over the KKM corpus using Local MLX.

Mode A (/api/v1/triage)          Retrieves MTS 2022, CPG, and FUKKM context, 
                                 forces the local LLM to output strict JSON.
Mode B (/api/v1/compare-inquire) Wide retrieval, generating a structured markdown response.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace
from functools import cached_property

from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
from pydantic import ValidationError

from . import (
    completeness, config, contraindications, dosing, formulary, mts_table,
    progress, red_flags, redirection,
)
from .ingest import get_collection
from .schemas import (
    Citation,
    CompletenessGap,
    Disposition,
    Contraindication,
    DiagnosticSchema,
    InquiryRequest,
    InquiryResponse,
    InquiryScope,
    RetrievedSource,
    TriageRequest,
    TriageResponse,
    TriageLevel,
    TriageLabel,
    modifier_prose,
)

log = logging.getLogger("rag_engine")

TRIAGE_COLOUR = {1: "RED", 2: "RED", 3: "YELLOW", 4: "GREEN", 5: "GREEN"}

SCOPE_DOC_TYPES: dict[InquiryScope, list[str] | None] = {
    InquiryScope.ALL: None,
    InquiryScope.CPG_ONLY: [config.DOC_TYPE_CPG_FULL, config.DOC_TYPE_CPG_QR],
    InquiryScope.QUICK_REFERENCE_ONLY: [config.DOC_TYPE_CPG_QR],
    InquiryScope.PAEDIATRIC_ONLY: [config.DOC_TYPE_PAEDS],
    InquiryScope.TRIAGE_ONLY: [config.DOC_TYPE_TRIAGE],
    InquiryScope.FORMULARY_ONLY: [config.DOC_TYPE_DRUG],
}

class RagError(RuntimeError):
    """Raised for empty corpora or model failures."""

# Words that carry no clinical discrimination. Kept deliberately short - BM25's
# own IDF handles the rest, and over-pruning would drop real terms.
_LEX_STOP = frozenset(
    "a an the and or of to in on for with without is are was were be been being "
    "at by from as this that these those it its if then than so but not no nor "
    "he she they we you i his her their our your patient presents presenting "
    "history complains complaining old year years".split()
)
_LEX_SPLIT = re.compile(r"[^a-z0-9%/.+-]+")


def _lex_tokens(text: str) -> list[str]:
    """Tokens for BM25. Numbers and units survive ('2g', '0.9%', 'mg/kg') because
    a dose is exactly the kind of term this channel exists to match."""
    out = []
    for tok in _LEX_SPLIT.split((text or "").lower()):
        tok = tok.strip(".-+/")
        if len(tok) > 1 and tok not in _LEX_STOP:
            out.append(tok)
    return out


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict
    # None when a chunk arrived via BM25 only and was never scored by the
    # embedder - a real state, not a missing value, so it must not default to 0.
    distance: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    forced: bool = False

    @property
    def similarity(self) -> float | None:
        return None if self.distance is None else round(1.0 - self.distance, 4)

    @property
    def how(self) -> str:
        """Why this chunk is here - shown in source attribution so a reader can
        tell an embedding match from a literal term match from a safety rule."""
        if self.forced:
            return "red-flag rule"
        bits = []
        if self.distance is not None:
            bits.append("semantic")
        if self.lexical_score is not None:
            bits.append("term match")
        return "+".join(bits) or "retrieved"

# ===========================================================================
# Prompts & Few-Shot Examples
# ===========================================================================

# MTS_SCALE and MTS_2022_GUARDRAILS were two hand-typed restatements of the
# grid that `mts_table` already holds cell by cell, and by 2026-09-11 the second
# had drifted from it on the adult pain bands, the HR bands and the Level 5
# time. `mts_table.prompt_rules()` renders both from ADULT_RULES, so the block
# the model reads and the rules `_enforce_mts_level` applies cannot disagree.
# The reasoning behind each number is in mts_table, next to the number.
MTS_ADULT_RULES = mts_table.prompt_rules()

FEW_SHOT_ACS_EXAMPLE = """
EXAMPLE REASONING PROTOCOL (DO NOT OUTPUT THIS TEXT, MIMIC THE LOGIC IN JSON):
Scenario: 55yo Male, HR 110, Pain Score 8/10, Complaint: Chest pain and profuse sweating.
Step 1 - Pain Score Evaluation: 8/10 -> "Severe Pain" = Level 2.
Step 2 - Complaint Evaluation: Chest pain + profuse sweating = Level 2 modifier.
Step 3 - Vitals Evaluation: HR 110 = Level 3.
Step 4 - Final Decision: The worst single criterion is Level 2. Output Level 2.
"""

# The corpus stopped being one kind of document on 2026-09-11. Each retrieved
# chunk already carries its own SOURCE TYPE line and the [S#] header names the
# doc_type; this states what the types mean. `CLINICAL_DOC_TYPES` enforces the
# drug and dose half of this in code - _check_drug_indications and
# _dose_sentence simply skip a chunk that is not a clinical guideline - so this
# rule is about what the model may ASSERT, not what it may be grounded on.
SOURCE_TYPES = """6a. NOT EVERY RETRIEVED SOURCE IS A CLINICAL GUIDELINE. The
   type is given in each [S#] header.
   - CPG_FULL, CPG_QUICK_REFERENCE, PAEDIATRIC_PROTOCOL: clinical guidelines.
     These are what a diagnosis, an investigation or a treatment rests on.
   - TRIAGE_PROTOCOL: the Malaysian Triage Scale. It sets the triage level and
     nothing else - it is not a treatment guideline.
   - DRUG_FORMULARY: FUKKM. Doses and prescriber categories only.
   - HTA_REPORT: an MOH appraisal OF a tool. Its statistics are study findings
     about how the tool performs, never a recommendation for this patient.
     NEVER compute or report a score from it, and never let it set a level.
   - PATIENT_FLOW_POLICY: who may be redirected out of the department. It is
     STATE-level, so name the state if you rely on it, and it governs
     disposition only - never diagnosis, treatment or triage level."""

DRUG_GROUNDING = """7. DRUG GROUNDING - ABSOLUTE. Never state a dose, route, frequency or
   prescriber category that is not written in the retrieved [S#] chunks.
   If a drug is clinically indicated but NO FUKKM chunk for that drug was
   retrieved, you MUST output it like this and nothing else:
       "prescriber_category": "NOT_IN_RETRIEVED_SOURCES"
       "prescriber_category_meaning": "No FUKKM entry for this drug was retrieved; category unverified."
   and name that drug in "prescriber_category_warning" in one sentence.
   NEVER guess a category letter (A, A*, B, C, C+, A/KK) to fill the field.
   A plausible-looking guess is a patient-safety failure; an explicit
   NOT_IN_RETRIEVED_SOURCES is the correct answer.
8. GAP GROUNDING. A vital sign that was not supplied is a GAP, never a normal
   value. Say so in "evidence_gaps" rather than assuming it is normal."""

TRIAGE_SYSTEM = f"""You are a clinical decision-support engine for the Malaysian Ministry of Health (KKM).

{MTS_ADULT_RULES}

{FEW_SHOT_ACS_EXAMPLE}

EVALUATION SEQUENCE & GROUNDING RULES:
1. FIRST, evaluate the Pain Score. 
2. SECOND, evaluate specific modifiers in the Complaints List (e.g., sweating with chest pain).
3. THIRD, evaluate Vitals.
4. You MUST triage based on the WORST SINGLE CRITERION from steps 1-3.
5. Every red flag, immediate action and drug MUST carry the source_id of the
   [S#] chunk it came from, and name the guideline in supporting_cpg.
6. You MUST output strictly in the requested JSON format without markdown wrappers.

{SOURCE_TYPES}
{DRUG_GROUNDING}"""

INQUIRY_SYSTEM = """You are a KKM Clinical Practice Guideline comparative-analysis engine.
Answer the query clearly using only the retrieved sources, heavily citing [S#].
Structure output with these markdown headers: ## Answer, ## Comparison, ## Key differences, ## Caveats and gaps, ## Sources."""

# ===========================================================================
# Deterministic post-validation guardrails
#
# Prompt-level grounding does not hold on a 4-bit 3B model: it was observed
# emitting prescriber_category "B" for "Fluid replacement therapy", which is
# not a FUKKM entry at all. Anything that must be true is therefore enforced
# here, in code, against the chunks that were actually retrieved.
# ===========================================================================

UNVERIFIED = "NOT_IN_RETRIEVED_SOURCES"
UNVERIFIED_MEANING = (
    "This drug is not listed in the FUKKM formulary under any name this system "
    "recognises, so no prescriber category could be verified. Check the formulary "
    "directly before prescribing."
)

# A leaked slot marker from the JSON template, e.g. "<why this destination>".
# Anchored on a leading lowercase letter so clinical text like "HR <60 >120"
# does not trip it.
_SLOT = re.compile(r"<[a-z][^<>]{2,79}>")


def _formulary_index(sources: list[RetrievedSource]) -> list[tuple[str, RetrievedSource]]:
    """Retrieved FUKKM chunks, as (lowercased drug name, source)."""
    return [
        (str(s.drug_name).strip().lower(), s)
        for s in sources
        if s.doc_type == config.DOC_TYPE_DRUG and str(s.drug_name or "").strip()
    ]


def _match_formulary(
    drug_name: str, index: list[tuple[str, RetrievedSource]]
) -> RetrievedSource | None:
    """Match on the base name only. FUKKM names carry strength and form
    ("Abacavir Sulphate 300mg tablet"), so the model's "Abacavir" must match,
    while "Fluid replacement therapy" must not match anything."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", drug_name.lower()) if len(t) >= 4]
    if not tokens:
        return None
    head = tokens[0]
    for name, source in index:
        if head in name:
            return source
    return None


def _ground_prescriber_categories(
    diagnostic: DiagnosticSchema, sources: list[RetrievedSource]
) -> None:
    """Force every prescriber_category to agree with the FUKKM record, by direct
    name lookup over the whole formulary.

    This used to match only against FUKKM chunks that embedding search happened
    to retrieve, and measured badly: for a shocked STEMI the formulary retrieval
    returned a pneumococcal vaccine, secukinumab, rituximab and fluconazole, so
    aspirin and GTN were both reported "CATEGORY UNVERIFIED" while sitting in the
    file under names the model does not use (acetylsalicylic acid, glyceryl
    trinitrate). FUKKM is a database of ~1,700 rows; it is now queried as one.

    The route matters: prescriber category varies by formulation (glyceryl
    trinitrate is C sublingual but "A, A/KK" as an injection), so the model's
    stated route selects which category applies. Where the route is unknown and
    the formulations disagree, BOTH categories are reported rather than one
    picked silently."""
    unverified: list[str] = []
    corrected: list[str] = []
    ambiguous: list[str] = []

    for drug in diagnostic.drug_recommendations:
        claimed = str(drug.prescriber_category or "").strip()
        # The stated dose participates in the lookup so the category and the
        # quoted dose text cite the SAME formulation. Without it the category
        # cited the 150mg tablet while the dose cited the 300mg one.
        stated_dose = str(drug.adult_dose or drug.paediatric_dose or "")
        match = formulary.lookup(str(drug.drug_name or ""), str(drug.route or ""), stated_dose)

        if match is None:
            if claimed and claimed != UNVERIFIED:
                unverified.append(f"{drug.drug_name} (model claimed category {claimed})")
            else:
                unverified.append(str(drug.drug_name))
            drug.prescriber_category = UNVERIFIED
            drug.prescriber_category_meaning = UNVERIFIED_MEANING
            continue

        actual = match.category_display
        if claimed != actual:
            # UNVERIFIED is an internal sentinel, not something to show a reader.
            was = "not stated" if (not claimed or claimed == UNVERIFIED) else claimed
            corrected.append(f"{drug.drug_name}: {was} to {actual}")
        drug.prescriber_category = actual or UNVERIFIED
        drug.prescriber_category_meaning = (
            f"Matched to {match.best.drug_name} [{match.citation}]"
            + (f". Route '{drug.route}' selected this formulation." if match.route_matched else "")
        )
        if match.varies:
            ambiguous.append(
                f"{drug.drug_name}: category differs by formulation "
                f"({', '.join(match.categories)}) and no route was stated"
            )

    notes: list[str] = []
    if unverified:
        notes.append(
            "Not found in the FUKKM formulary: "
            + "; ".join(unverified)
            + ". Confirm against the formulary before prescribing."
        )
    if corrected:
        notes.append(
            "Prescriber categories corrected against the FUKKM record: "
            + "; ".join(corrected)
            + "."
        )
    if ambiguous:
        notes.append(
            "Category depends on the formulation - state the route to resolve it: "
            + "; ".join(ambiguous)
            + "."
        )
    diagnostic.prescriber_category_warning = " ".join(notes)
    for note in notes:
        log.warning("GROUNDING  %s", note)


# ------------------------------------------- drug indication cross-check
# `_ground_prescriber_categories` verifies PROVENANCE, not INDICATION: it
# confirms a category is real, never that the drug treats the disease. Measured
# consequence - the 3B recommended snake antivenom for dengue, and Qwen3-8B
# recommended Artesunate (an antimalarial) for dengue shock. Both carried a
# genuine FUKKM category and both passed the provenance guardrail.

# Dosage forms and generic ion/salt words carry no identifying information -
# "Sodium Chloride Injection 0.9%" is identified by "saline", not by "sodium".
_DRUG_GENERIC = frozenset(
    "solution injection tablet tablets capsule capsules oral infusion syrup "
    "suspension ampoule vial powder sterile water "
    "sodium chloride potassium calcium magnesium acid hydrochloride hydrochlorid "
    "sulphate sulfate phosphate maleate tartrate besylate succinate fumarate "
    "and or of for the with".split()
)

# The formulary and the guidelines name the same drug differently: FUKKM lists
# "Sodium Chloride Injection 0.9%", the dengue CPG says "normal saline". Without
# this table the correct fluid is reported as unsupported - which the unit test
# caught. Curated and auditable on purpose; a fuzzy matcher here would trade a
# false negative for a silent false positive on a drug recommendation.
_DRUG_SYNONYMS: dict[str, tuple[str, ...]] = {
    "sodium chloride": ("normal saline", "0.9% saline", "0.9%saline", "nacl",
                        "isotonic saline", "crystalloid", "saline"),
    "hartmann": ("ringer lactate", "lactated ringer", "compound sodium lactate",
                 "crystalloid"),
    "adrenaline": ("epinephrine",),
    "epinephrine": ("adrenaline",),
    "noradrenaline": ("norepinephrine",),
    "paracetamol": ("acetaminophen",),
    "acetylsalicylic": ("aspirin",),
    "aspirin": ("acetylsalicylic",),
    "salbutamol": ("albuterol",),
    "frusemide": ("furosemide",),
    "furosemide": ("frusemide",),
    "glyceryl trinitrate": ("gtn", "nitroglycerin", "nitrate"),
    "dextrose": ("glucose",),
    "glucose": ("dextrose",),
}


def _alias_pattern(aliases: set[str]) -> re.Pattern:
    """Word-boundary alternation over a drug's aliases.

    Plain substring matching is unsafe here: the alias "haemato" (from "Haemato
    Polyvalent Snake Antivenom") matches "haematocrit" and "haematologist",
    which made the dengue CPG appear to endorse snake antivenom. Boundaries are
    letter-only lookarounds so aliases like "0.9% saline" still match.
    """
    alt = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])", re.IGNORECASE)


def _drug_aliases(name: str) -> set[str]:
    """Every string that would identify this drug in guideline prose."""
    low = (name or "").lower()
    aliases = {
        t for t in re.split(r"[^a-z0-9]+", low)
        if len(t) > 3 and t not in _DRUG_GENERIC
    }
    for key, syns in _DRUG_SYNONYMS.items():
        if key in low:
            aliases.update(syns)
        elif any(sy in low for sy in syns):
            aliases.add(key)
            aliases.update(syns)
    return aliases


def _check_drug_indications(
    diagnostic: DiagnosticSchema,
    chunks: list[Retrieved],
    corpus: dict | None = None,
) -> None:
    """Two tiers, because "not in the retrieved chunks" and "not in the guideline"
    are very different claims.

    Tier 1 - named in a guideline chunk retrieved for this presentation.
    Tier 2 - named anywhere in the FULL text of a guideline that this
             presentation actually retrieved.

    Only tier 1 counts as supported: dabigatran is genuinely in the ACS CPG (for
    AF patients needing a DOAC alongside DAPT) and passed an "anywhere in the
    guideline" check on a shocked STEMI. Prevalence cannot separate the two
    either - paracetamol appears in 1 chunk (0.72%) of the Dengue CPG, the same
    band as dabigatran in ACS (0.74%), and is the correct antipyretic. So both
    are reported unsupported, with different wording."""
    clinical_types = set(config.CLINICAL_DOC_TYPES) | {config.DOC_TYPE_TRIAGE}
    retrieved_text = " \n".join(
        c.text.lower() for c in chunks if c.metadata.get("doc_type") in clinical_types
    )
    guideline_titles = {
        c.metadata.get("cpg_title")
        for c in chunks
        if c.metadata.get("doc_type") in clinical_types
    }

    def _in_full_guideline(pattern: re.Pattern) -> bool:
        if not corpus or not guideline_titles:
            return False
        for text, meta in zip(corpus["docs"], corpus["metas"]):
            if meta.get("cpg_title") not in guideline_titles:
                continue
            if pattern.search(text):
                return True
        return False

    unsupported: list[str] = []
    elsewhere: list[str] = []
    for drug in diagnostic.drug_recommendations:
        aliases = _drug_aliases(str(drug.drug_name or ""))
        if not aliases:
            drug.indication_supported = None  # unverifiable, not unsupported
            continue
        pattern = _alias_pattern(aliases)
        if pattern.search(retrieved_text):
            drug.indication_supported = True
        elif _in_full_guideline(pattern):
            drug.indication_supported = False
            elsewhere.append(str(drug.drug_name))
        else:
            drug.indication_supported = False
            unsupported.append(str(drug.drug_name))

    notes: list[str] = []
    if unsupported:
        notes.append(
            "NOT FOUND IN THE GUIDELINE FOR THIS PRESENTATION: "
            + "; ".join(unsupported)
            + ". These were matched in the FUKKM formulary, which lists every drug "
            "for every condition, but appear nowhere in the clinical guideline "
            "retrieved for this patient. Confirm the indication before prescribing."
        )
    if elsewhere:
        notes.append(
            "NOT VERIFIED FOR THIS PRESENTATION: "
            + "; ".join(elsewhere)
            + " - the drug appears elsewhere in that guideline but not in the excerpt "
            "retrieved for this patient, so it may belong to a different sub-population "
            "(a comorbidity, another phase of care). Read the guideline section itself "
            "before prescribing; the citation shown does not support it."
        )
    if notes:
        note = " ".join(notes)
        diagnostic.drug_indication_warning = note
        for n in notes:
            log.warning("GROUNDING  %s", n)


# ------------------------------------------------- dose completeness check
# "Administer intravenous fluids" is not an instruction for a 20 kg shocked
# child; the retrieved chunk says 5-7 ml/kg. Measured: every model tested,
# including the 14B, gave the direction without the number.

_DOSE_REQUIRED_RE = re.compile(
    r"\b(fluid|fluids|bolus|resuscitat\w*|crystalloid|colloid|saline|hartmann|"
    r"ringer|dextrose|adrenaline|epinephrine|noradrenaline|infusion|rehydrat\w*)\b",
    re.IGNORECASE,
)
_DOSE_PRESENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:-|to|\u2013)?\s*\d*(?:\.\d+)?\s*"
    r"(?:ml|millilitre|milliliter|mg|microgram|mcg|g|gram|unit|units)\s*/\s*kg",
    re.IGNORECASE,
)


def _check_dose_completeness(
    diagnostic: DiagnosticSchema, req: TriageRequest
) -> None:
    """Flag weight-based instructions that carry no weight-based dose. Only runs
    when the weight is actually known - otherwise the missing dose is already
    reported as a missing input, not a model failure."""
    if not req.weight_kg:
        return
    missing: list[str] = []

    for action in diagnostic.immediate_actions:
        text = str(action.action or "")
        if _DOSE_REQUIRED_RE.search(text) and not _DOSE_PRESENT_RE.search(text):
            missing.append(f"action {action.sequence}: {text[:70]}")

    for drug in diagnostic.drug_recommendations:
        dose_field = (
            str(drug.paediatric_dose or "")
            if req.age < config.PAEDIATRIC_AGE_YEARS
            else str(drug.adult_dose or "")
        )
        if not dose_field.strip():
            missing.append(f"{drug.drug_name}: no dose stated")
        elif _DOSE_REQUIRED_RE.search(
            f"{drug.drug_name} {drug.indication}"
        ) and not _DOSE_PRESENT_RE.search(dose_field):
            missing.append(f"{drug.drug_name}: dose {dose_field[:40]!r} is not weight-based")

    if missing:
        note = (
            f"DOSE INCOMPLETE for a {req.weight_kg:g} kg patient - weight-based "
            "instructions given without a per-kg dose: "
            + "; ".join(missing)
            + ". Look up the dose in the cited guideline before acting."
        )
        diagnostic.dose_completeness_warning = note
        log.warning("GROUNDING  %s", note)


# ---------------------------------------------------- the intake as prose
def _intake_text(req: TriageRequest) -> str:
    """Complaint, history AND the structured modifiers, as one string.

    Every text-matching layer in this engine reads this: the MTS qualifier
    patterns, red_flags.py, and _check_urgency_sanity. Before the Primary
    Triage fields existed those layers saw only what someone typed in prose,
    so a capillary refill of 4 seconds ticked in a box was invisible to the
    shock_signs qualifier that exists to find exactly that.

    `for_matching=True` is not a detail. It swaps the two phrasings whose
    display wording contains "pregnan" while describing a patient who is not
    known to be pregnant - see _MATCHING_PROSE in schemas.py.
    """
    return " ".join(
        p for p in (req.complaint, req.history, req.modifiers_text(for_matching=True))
        if p
    )


def _patient_extras(req: TriageRequest) -> str:
    """The optional intake, as lines for the PATIENT block - or "" if none.

    Written as prose, not as a field dump, because the model reads it as
    clinical text. Anything the clinician left blank is simply absent: the
    prompt must never say "Allergies: none" when the question was not asked,
    which is the same reason the vitals block reports a missing value as a gap
    rather than as normal.
    """
    lines: list[str] = []
    if req.onset.strip():
        lines.append(f"Onset / day of illness: {req.onset.strip()}")
    if req.trauma_mechanism.strip():
        lines.append(f"Mechanism of injury: {req.trauma_mechanism.strip()}")
    if req.allergies.strip():
        lines.append(f"Allergies: {req.allergies.strip()}")
    if req.current_medications.strip():
        lines.append(f"Current medications: {req.current_medications.strip()}")

    # The MTS observations, minus the four already given their own line above.
    _ALREADY_SAID = ("symptom onset", "mechanism of injury",
                     "allergies:", "current medications:")
    observed = [m for m in req.modifier_lines()
                if not m.startswith(_ALREADY_SAID)]
    if observed:
        lines.append("Triage observations: " + "; ".join(observed))

    if req.facility is not None:
        lines.append(
            f"Setting: {req.facility.value.replace('_', ' ')} - "
            f"{req.facility_meaning()} Every recommendation must be one that "
            "can actually be started HERE; where definitive care is elsewhere, "
            "say what to do before and during transfer."
        )
    return ("\n" + "\n".join(lines)) if lines else ""


def _retrieval_terms(req: TriageRequest) -> str:
    """The modifiers worth adding to a RETRIEVAL query, and only those.

    Not the whole intake: a query is a similarity probe, and padding it with
    "arrived by ambulance" moves the probe away from the clinical passage. The
    four kept here each name a different SECTION of a guideline -

      onset          the stroke thrombolysis window, the ACS reperfusion
                     clock, the febrile / critical / recovery phase of dengue
      ECG            STEMI and NSTE-ACS are two separate CPGs in this corpus
                     and nothing else in the intake tells them apart
      comorbidity    renal and hepatic dosing, obstructive airway disease
      mechanism      the head-injury and abdominal-trauma pathways
    """
    bits = [req.onset.strip(), req.trauma_mechanism.strip()]
    bits += [modifier_prose(e.value) for e in req.ecg_findings]
    bits += [modifier_prose(c.value) for c in req.comorbidities]
    return " ".join(b for b in bits if b)


# ------------------------------------------- contraindication checking
def _clinical_context(req: TriageRequest, diagnostic: DiagnosticSchema) -> str:
    """The patient picture a contraindication rule is tested against.

    Derived observations are spelled out in words ("hypotension", "bradycardia",
    "paediatric patient") so a rule fires on the OBSERVATION, not on whether the
    model happened to use that word. The GTN failure happened on a patient whose
    hypotension was in the vitals but never named in the model's prose.
    """
    bits = [req.complaint or "", req.history or "",
            req.modifiers_text(for_matching=True),
            getattr(diagnostic.primary_diagnosis, "condition", "") or "",
            getattr(diagnostic.primary_diagnosis, "reasoning", "") or ""]
    bits += [getattr(d, "condition", "") or "" for d in diagnostic.differential_diagnoses]
    bits += [str(getattr(f, "flag", "") or "") for f in diagnostic.red_flags]

    v = req.vitals
    if v.systolic_bp is not None:
        if v.systolic_bp < _hypotension_threshold(req.age):
            bits.append(f"hypotension (systolic {v.systolic_bp:g} mmHg)")
        if v.systolic_bp >= 180:
            bits.append("severe hypertension")
    if v.heart_rate is not None:
        if v.heart_rate < 50:
            bits.append(f"bradycardia (heart rate {v.heart_rate:g})")
        if v.heart_rate > 130:
            bits.append("tachycardia")
    if v.spo2 is not None and v.spo2 < 94:
        bits.append("hypoxia")
    if v.capillary_blood_glucose is not None:
        if v.capillary_blood_glucose < 4:
            bits.append("hypoglycaemia (glucose below 4)")
        if v.capillary_blood_glucose > 11:
            bits.append("hyperglycaemia")
    if v.gcs is not None and v.gcs < 15:
        bits.append(f"reduced consciousness (GCS {v.gcs})")
    if req.age < 16:
        bits.append("paediatric patient (age under 16)")
    if req.age < 8:
        bits.append("age under 8")
    if req.pregnant:
        bits.append("pregnant")
    return " \n".join(b for b in bits if b)


def _check_contraindications(
    diagnostic: DiagnosticSchema, req: TriageRequest
) -> None:
    """Test every recommendation against this patient's own picture.

    Actions, investigations AND drugs are all scanned: the GTN that prompted
    this check arrived as an immediate action, so inspecting drug entries alone
    would have missed it entirely. Findings annotate, never delete - removing a
    recommendation would hide the model failure instead of exposing it."""
    items: list[tuple[str, str]] = []
    for a in diagnostic.immediate_actions:
        items.append((f"action {a.sequence}", str(a.action or "")))
    for x in diagnostic.investigations:
        items.append((f"investigation: {x.test}", f"{x.test} {x.rationale}"))
    for d in diagnostic.drug_recommendations:
        items.append((f"drug: {d.drug_name}",
                      " ".join(str(getattr(d, f, "") or "") for f in
                               ("drug_name", "indication", "adult_dose",
                                "paediatric_dose", "route", "cautions"))))

    found = contraindications.check(_clinical_context(req, diagnostic), items)
    if not found:
        return

    diagnostic.contraindications = [
        Contraindication(rule=f.rule, severity=f.severity, where=f.where,
                         item=f.item, trigger=f.trigger, reason=f.reason)
        for f in found
    ]
    absolute = [f for f in found if f.severity == "ABSOLUTE"]
    parts = []
    if absolute:
        parts.append(
            "ABSOLUTE CONTRAINDICATION - do not act on the following without "
            "re-reading the guideline: "
            + "; ".join(f'{f.item} in {f.where} (triggered by "{f.trigger}": {f.reason})'
                        for f in absolute)
        )
    caution = [f for f in found if f.severity != "ABSOLUTE"]
    if caution:
        parts.append(
            "CAUTION: "
            + "; ".join(f'{f.item} in {f.where} ("{f.trigger}": {f.reason})' for f in caution)
        )
    note = " ".join(parts)
    diagnostic.contraindication_warning = note
    for f in found:
        log.warning("CONTRAINDICATION  [%s] %s in %s <- %r", f.severity, f.item, f.where, f.trigger)


# --------------------------------------------------------- disposition floor
# Measured twice: the model sent an MTS 1 cardiogenic-shock STEMI to "ED
# observation" with no referral. Under-disposition is the dangerous direction,
# so the floor is applied in code like the triage level itself.
#
# Tiers of CARE, not a list of places. ADMIT_ICU_HDU and RESUSCITATION_BAY are
# the same tier: the resuscitation bay is where an MTS 1 patient is stabilised
# and the ICU is where they go next, so an answer that already says ICU must not
# be "corrected" to an ED bay. Ranking them 4 and 5 did exactly that.
_DISPOSITION_RANK = {
    "DISCHARGE_WITH_FOLLOW_UP": 0,
    "ED_OBSERVATION": 1,
    "REFER_SPECIALIST": 2,
    "ADMIT_WARD": 3,
    "ADMIT_ICU_HDU": 4,
    "RESUSCITATION_BAY": 4,
}
# Minimum tier of care for each MTS level; the value names the destination used
# when the floor has to be applied.
_MIN_DISPOSITION = {1: "RESUSCITATION_BAY", 2: "ADMIT_WARD"}

# The ceiling half. _enforce_disposition floored MTS 1 and 2 upward but capped
# nothing, which is how a Level 4 allergic-rhinitis patient was sent to ED
# observation. A ceiling is a SET of permitted destinations, not a point on the
# _DISPOSITION_RANK ladder: that ladder orders intensity of inpatient care, and
# REFER_SPECIALIST outranks ED_OBSERVATION on it. Capping by rank would block
# the outpatient specialist referral that is the correct answer for a routine
# patient - the ORL clinic referral in the rhinosinusitis algorithm, say.
_PERMITTED_DISPOSITION = {
    4: {"DISCHARGE_WITH_FOLLOW_UP", "REFER_SPECIALIST", "ED_OBSERVATION"},
    5: {"DISCHARGE_WITH_FOLLOW_UP", "REFER_SPECIALIST"},
}
# Where an over-shooting answer lands when the ceiling is applied.
_MAX_DISPOSITION = {4: "ED_OBSERVATION", 5: "DISCHARGE_WITH_FOLLOW_UP"}

# MTS 2022 page 6, verbatim: the triage-away pathway. Already indexed, never
# retrieved. It is the only place the document says what sending a patient to
# other outpatient services actually requires.
_TRIAGE_AWAY_NOTE = (
    "[MTS 2022 p6: patients whose presentation is non-urgent and non-emergency, "
    "and can be better addressed in other outpatient services, may be "
    "triaged-away. It is necessary to ensure their vital signs are normal and "
    "provide them with a note recording down their complaints and vital signs "
    "readings.]"
)
# MTS 2022 page 13, verbatim.
_PAEDIATRIC_TRIAGE_AWAY_NOTE = (
    "[MTS 2022 p13: \"Children should not be routinely triaged-away.\" Confirm "
    "a paediatric discharge against the Complaints List (Paediatrics) before "
    "the child leaves.]"
)


def _apply_disposition_ceiling(diagnostic: DiagnosticSchema, req: TriageRequest) -> bool:
    """Cap the destination a low-acuity patient can be sent to, and attach the
    triage-away requirements when one is being sent home. Returns True if the
    level was handled here."""
    level = int(diagnostic.mts_triage_level)
    permitted = _PERMITTED_DISPOSITION.get(level)
    if not permitted:
        return False
    current = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if current not in permitted:
        ceiling = _MAX_DISPOSITION[level]
        diagnostic.disposition = Disposition(ceiling)
        note = (
            f"[System Guardrail: disposition lowered from "
            f"{current.replace('_', ' ').lower()} to "
            f"{ceiling.replace('_', ' ').lower()} - an MTS {level} patient does "
            "not meet the criteria for a higher level of care.]"
        )
        diagnostic.disposition_justification = (
            f"{diagnostic.disposition_justification} {note}"
        ).strip()
        log.warning("GROUNDING  %s", note)
        current = ceiling

    if current == "DISCHARGE_WITH_FOLLOW_UP":
        extra = [_TRIAGE_AWAY_NOTE]
        if req.age < config.PAEDIATRIC_AGE_YEARS:
            extra.append(_PAEDIATRIC_TRIAGE_AWAY_NOTE)
        diagnostic.disposition_justification = " ".join(
            [diagnostic.disposition_justification.strip(), *extra]
        ).strip()
    return True


def _enforce_disposition(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    level = int(diagnostic.mts_triage_level)
    if _apply_disposition_ceiling(diagnostic, req):
        return
    floor = _MIN_DISPOSITION.get(level)
    if not floor:
        return
    current = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if _DISPOSITION_RANK.get(current, 99) >= _DISPOSITION_RANK[floor]:
        if level <= 2 and not diagnostic.referral_required:
            diagnostic.referral_required = True
            diagnostic.disposition_justification = (
                f"{diagnostic.disposition_justification} [System Guardrail: an MTS {level} "
                "patient requires specialist referral.]"
            ).strip()
        return
    diagnostic.disposition = Disposition(floor)
    diagnostic.referral_required = True
    note = (
        f"[System Guardrail: disposition raised from {current.replace('_', ' ').lower()} "
        f"to {floor.replace('_', ' ').lower()} - an MTS {level} patient cannot be "
        "managed at a lower level of care.]"
    )
    diagnostic.disposition_justification = f"{diagnostic.disposition_justification} {note}".strip()
    log.warning("GROUNDING  %s", note)


def _check_redirection(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Flag a triage-away that the Selangor redirection policy forbids.

    Runs AFTER `_enforce_disposition`, so it reads the disposition the patient
    actually leaves with rather than the one the model first proposed - a report
    whose discharge was already raised to admission needs no warning.

    It warns and does not override, for the reason argued in redirection.py: the
    policy is a state one, MTS 2022 has no redirection criteria of its own, and
    a hard block would impose Selangor practice on every other state.
    """
    disposition = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if disposition not in redirection.REDIRECTING_DISPOSITIONS:
        return
    hits = redirection.matches(req)
    if not hits:
        return
    diagnostic.redirection_warning = redirection.warning(hits, disposition)
    log.warning(
        "GROUNDING  [Redirection policy 4.2: %s]",
        ", ".join(e.clause for e in hits),
    )


# --------------------------------------------------------- urgency sanity
# Two ways a low-acuity report inflates its own urgency, both seen on the same
# allergic-rhinitis run:
#
#   * "Mild pain (Pain Score 1/10)" listed as a RED FLAG. That is an MTS triage
#     modifier presented as a danger sign. A red-flag list that fills up with
#     normal findings is how alarm fatigue starts, and it devalues the flags
#     that matter on the reports where they are real.
#   * "Allergy testing - STAT" ordered on a patient who is going home.
#
# The red-flag list is CLEARED rather than annotated, because a flag nothing
# supports should not be on the page at all; what was removed is named, so the
# clinician can see the check acted. STAT orders are only flagged - the report
# may have a reason, and deleting an investigation is not this check's call.
_STAT_RE = re.compile(r"\b(stat|immediate(?:ly)?|emergent|right away)\b", re.I)


def _check_urgency_sanity(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Urgency the report claimed that its own triage level does not support."""
    level = int(diagnostic.mts_triage_level)
    if level < 4:
        return
    notes: list[str] = []

    intake = _intake_text(req)
    if diagnostic.red_flags and not red_flags.match(intake):
        dropped = [str(getattr(f, "flag", "") or "").strip() for f in diagnostic.red_flags]
        dropped = [d for d in dropped if d]
        diagnostic.red_flags = []
        notes.append(
            f"Red flags removed at MTS {level}: no red-flag rule fired on this "
            f"intake, so nothing here is a danger sign - "
            f"{'; '.join(dropped)}." if dropped else
            f"Red flags removed at MTS {level}: no red-flag rule fired on this intake."
        )

    stat = [
        (x.test or "").strip()
        for x in diagnostic.investigations
        if _STAT_RE.search(x.urgency or "") or _STAT_RE.search(x.test or "")
    ]
    if stat:
        notes.append(
            f"Investigation ordered STAT on an MTS {level} patient "
            f"({MTS_TIME[level]}): {'; '.join(stat)}. Confirm the urgency is "
            "intended, or reorder as routine."
        )

    if notes:
        diagnostic.urgency_warning = " ".join(notes)
        log.warning("GROUNDING  urgency sanity: %s", diagnostic.urgency_warning)


# ---------------------------------------------------- completeness of answer
def _check_completeness(
    diagnostic: DiagnosticSchema, req: TriageRequest
) -> None:
    """Name what the answer left out. Never fills the gap - a checklist that
    invented the missing dose would be a second unreliable reasoner."""
    answer = " \n".join(
        [diagnostic.triage_rationale or "",
         getattr(diagnostic.primary_diagnosis, "reasoning", "") or ""]
        + [f"{a.action} {a.timeframe}" for a in diagnostic.immediate_actions]
        + [f"{x.test} {x.rationale}" for x in diagnostic.investigations]
        + [" ".join(str(getattr(d, f, "") or "") for f in
                    ("drug_name", "indication", "adult_dose", "paediatric_dose",
                     "route", "cautions"))
           for d in diagnostic.drug_recommendations]
        + [f"{diagnostic.disposition_justification} {diagnostic.referral_to}"]
    )
    label, gaps = completeness.check(_clinical_context(req, diagnostic), answer)
    if not gaps:
        return
    diagnostic.completeness_gaps = [
        CompletenessGap(element=g.element, why=g.why, guideline=g.guideline) for g in gaps
    ]
    note = (
        f"INCOMPLETE for {label}: the answer does not address "
        + "; ".join(g.element for g in gaps)
        + f". Read {gaps[0].guideline} before acting on this report."
    )
    diagnostic.completeness_warning = note
    log.warning("GROUNDING  %s", note)


# ------------------------------------------------------- citation integrity
_SID = re.compile(r"S\d+")


# Generic vocabulary shared by most MaHTAS titles. Matching on what is left
# is what lets "Rhinosinusitis CPG" resolve to "Management of Rhinosinusitis in
# Adolescents and Adults" without also matching every other guideline whose
# title happens to start "Management of".
_GENERIC_TITLE_WORDS = {
    "management", "adults", "adult", "children", "child", "paediatric",
    "paediatrics", "guideline", "guidelines", "clinical", "practice", "quick",
    "reference", "edition", "malaysia", "malaysian", "ministry", "health",
    "treatment", "prevention", "early", "national", "protocol", "consensus",
    "statement", "with", "from", "their",
}


def _title_words(title: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]+", (title or "").lower())
        if len(w) > 3 and w not in _GENERIC_TITLE_WORDS
    }


def _ground_citations(
    diagnostic: DiagnosticSchema, sources: list[RetrievedSource]
) -> list[str]:
    """Rebuild the citation table in code from what the report actually cites.

    The model was asked to author its own bibliography and, on the 2026-09-09
    rhinitis run, wrote the formulary and nothing else: `citations` listed FUKKM
    alone while `supporting_cpg` correctly named the rhinosinusitis CPG. The
    report therefore quoted a guideline it never cited, and the page numbers and
    years in that table were the model's transcription rather than the index's.
    Same class as every other failure in this file - a property asked for in a
    prompt is not a property that holds - so the table is derived instead, from
    the [S#] ids the report carries plus the guideline the diagnosis names.

    Only CITED sources are listed. The retrieved set runs to fifteen-odd
    passages and reprinting all of it would bury the clinical content; the
    report's provenance line already names the documents without the
    bookkeeping. Returns notes for the citation warning.
    """
    by_id = {s.source_id: s for s in sources}
    order = {s.source_id: i for i, s in enumerate(sources)}
    cited: set[str] = set()
    notes: list[str] = []

    def note(raw: object) -> None:
        cited.update(i for i in _SID.findall(str(raw or "")) if i in by_id)

    # Fields that carry an explicit source_id.
    for f in diagnostic.red_flags:
        note(f.source_id)
    for a in diagnostic.immediate_actions:
        note(a.source_id)
    for d in diagnostic.drug_recommendations:
        note(d.source_id)
    # Prose carries inline [S#] as well, and a citation referenced only in the
    # rationale is still a citation.
    note(diagnostic.triage_rationale)
    note(getattr(diagnostic.primary_diagnosis, "reasoning", ""))
    note(diagnostic.disposition_justification)
    note(diagnostic.evidence_gaps)
    for x in diagnostic.investigations:
        note(x.rationale)

    # The guideline the diagnosis names in prose, resolved to a real retrieved
    # source. This is the half that was missing: a diagnosis can name its CPG
    # and cite nothing for it, and a reader cannot tell which page it came from.
    named = str(getattr(diagnostic.primary_diagnosis, "supporting_cpg", "") or "").strip()
    wanted = _title_words(named)
    if wanted:
        hits = [t for t in sources if _title_words(t.cpg_title) & wanted]
        if hits:
            cited.add(min(hits, key=lambda t: order[t.source_id]).source_id)
        else:
            notes.append(
                f"the diagnosis cites {named!r}, which is not in the retrieved "
                "sources - treat the diagnosis as ungrounded"
            )

    diagnostic.citations = [
        Citation(
            source_id=src.source_id,
            document=src.cpg_title or src.filename or "KKM document",
            page=str(src.page_number) if src.page_number else "N/A",
            edition_year=src.edition_year or "N/A",
        )
        for src in sorted((by_id[i] for i in cited), key=lambda t: order[t.source_id])
    ]
    if not diagnostic.citations and sources:
        notes.append(
            "no recommendation in this report carries a source id, so nothing "
            "here is traceable to a guideline"
        )
    log.info(
        "CITATIONS  %d of %d retrieved sources cited: %s",
        len(diagnostic.citations), len(sources),
        ", ".join(c.source_id for c in diagnostic.citations) or "none",
    )
    return notes


def _check_citations(
    diagnostic: DiagnosticSchema,
    sources: list[RetrievedSource],
    extra_notes: list[str] | None = None,
) -> None:
    """Every clinical claim should carry a [S#] that actually exists.

    Reported as one line, not a new section: the citation table stays compact so
    a reviewer reads the clinical content, not the bookkeeping."""
    valid = {s.source_id for s in sources}
    uncited: list[str] = []
    invalid: list[str] = []

    def audit(where: str, sid: str) -> None:
        ids = _SID.findall(str(sid or ""))
        if not ids:
            uncited.append(where)
            return
        bad = [i for i in ids if i not in valid]
        if bad:
            invalid.append(f"{where} cites {'/'.join(bad)}")

    for a in diagnostic.immediate_actions:
        audit(f"action {a.sequence}", a.source_id)
    for d in diagnostic.drug_recommendations:
        audit(f"drug {d.drug_name}", d.source_id)

    parts = list(extra_notes or [])
    if uncited:
        parts.append(f"{len(uncited)} recommendation(s) carry no source: " + "; ".join(uncited[:6]))
    if invalid:
        parts.append("citation does not exist in the retrieved set: " + "; ".join(invalid[:6]))
    if not parts:
        return
    note = "CITATION GAP - " + ". ".join(parts) + "."
    diagnostic.citation_warning = note
    log.warning("GROUNDING  %s", note)


# ------------------------------------------------------------ dose grounding
_PUA = re.compile(r"[\uE000-\uF8FF\uFFF0-\uFFFF]")


def _clean_quote(text: str) -> str:
    """Tidy a verbatim quote for print without altering its wording.

    PDF producers embed Wingdings bullets in the Private Use Area; they survive
    extraction and print as tofu boxes. Leading list punctuation is dropped so a
    quoted line starts at the first real word."""
    t = _PUA.sub(" ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t.lstrip("-–—•*·:;, ").strip()


_DOSE_IN_TEXT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:-|–|to)?\s*\d*(?:[.,]\d+)?\s*"
    r"(?:mcg|microgram|mg|g|ml|units?|iu)\b(?:\s*/\s*kg)?", re.IGNORECASE)


def _cpg_dose_sentence(drug_name: str, chunks: list[Retrieved]) -> tuple[str, str]:
    """A verbatim sentence from the retrieved guideline that names this drug AND
    a dose. FUKKM carries general dosing; an indication-specific loading dose
    lives only in the CPG, so aspirin 300mg in ACS can be verified from nowhere
    else."""
    aliases = _drug_aliases(drug_name)
    if not aliases:
        return "", ""
    pattern = _alias_pattern(aliases)
    clinical = set(config.CLINICAL_DOC_TYPES) | {config.DOC_TYPE_TRIAGE}
    for chunk in chunks:
        if chunk.metadata.get("doc_type") not in clinical:
            continue
        for sentence in re.split(r"(?<=[.;])\s+|\n", chunk.text or ""):
            text = re.sub(r"\s+", " ", sentence).strip()
            if len(text) < 12 or len(text) > 320:
                continue
            if pattern.search(text) and _DOSE_IN_TEXT.search(text):
                page = chunk.metadata.get("page_number")
                title = str(chunk.metadata.get("cpg_title", "") or "")
                return _clean_quote(text), f"{title} p{page}" if page else title
    return "", ""


def _check_dosing(
    diagnostic: DiagnosticSchema, req: TriageRequest, chunks: list[Retrieved]
) -> None:
    """Attach the AUTHORITATIVE dose text to every drug and cross-check the
    stated dose against it.

    The guarantee this delivers is not "the dose is correct for this patient" -
    that needs indication, renal function and interactions. It is that no dose
    appears without a named source or an explicit mark saying it could not be
    verified."""
    paediatric = req.age < config.PAEDIATRIC_AGE_YEARS
    alarms: list[str] = []
    unverified: list[str] = []

    for drug in diagnostic.drug_recommendations:
        stated = str((drug.paediatric_dose if paediatric else drug.adult_dose) or "").strip()
        if not stated:
            stated = str(drug.adult_dose or drug.paediatric_dose or "").strip()

        match = formulary.lookup(str(drug.drug_name or ""), str(drug.route or ""), stated)
        sources: list[tuple[str, str]] = []
        if match:
            # EVERY formulation, not just the selected one: strengths differ and
            # the right one may not be the one the route picked.
            for e in match.entries:
                if e.dosage.strip():
                    sources.append((f"FUKKM {e.fukkm_no}", e.dosage))
            drug.dose_source_fukkm = f"{_clean_quote(match.best.dosage)}  [{match.best.citation}]"

        cpg_text, cpg_ref = _cpg_dose_sentence(str(drug.drug_name or ""), chunks)
        if cpg_text:
            sources.append((cpg_ref or "CPG", cpg_text))
            drug.dose_source_cpg = f"{cpg_text}  [{cpg_ref}]"

        verdict = dosing.check(stated, sources, req.weight_kg, paediatric)
        drug.dose_verdict = verdict.status
        drug.dose_verdict_detail = verdict.detail
        if verdict.status == dosing.EXCEEDS_MAXIMUM:
            alarms.append(f"{drug.drug_name}: {verdict.detail}")
        elif verdict.status != dosing.VERIFIED:
            unverified.append(f"{drug.drug_name} ({verdict.status.replace('_', ' ').lower()})")

    notes: list[str] = []
    if alarms:
        notes.append("DOSE EXCEEDS A STATED MAXIMUM - " + "; ".join(alarms) + ".")
    if unverified:
        notes.append(
            "Dose not verified against a source: " + "; ".join(unverified)
            + ". The authoritative dose text is quoted under each drug - read it "
            "rather than the figure above."
        )
    if notes:
        note = " ".join(notes)
        diagnostic.dose_completeness_warning = (
            f"{diagnostic.dose_completeness_warning} {note}".strip()
        )
        for n in notes:
            log.warning("GROUNDING  %s", n)


def _flag_leaked_slots(diagnostic: DiagnosticSchema) -> None:
    """The template ships <slot markers>; a copied one means an unfilled field."""
    leaked = sorted(set(_SLOT.findall(diagnostic.model_dump_json())))
    if not leaked:
        return
    note = (
        "Template placeholders were copied instead of filled in "
        f"({', '.join(leaked[:5])}) - treat those fields as unanswered."
    )
    diagnostic.evidence_gaps = f"{diagnostic.evidence_gaps} {note}".strip()
    log.warning("GROUNDING  %s", note)


# --------------------------------------------------------------- MTS floor
# Objective observations that MTS 2022 maps to a level regardless of what the
# model concludes. The model under-triaged both live test cases (paediatric
# dengue shock -> 3, expected 1; shocked ACS -> 3, expected 2), and under-triage
# is the dangerous direction, so the mapping is applied in code.

# Defined in mts_table, which holds MTS 2022 page 3 alongside the grid pages.
MTS_TIME = mts_table.TIME_TO_TREATMENT

# MTS 2022 page 3, verbatim: "The Triage process is meant to be repeated when new
# symptoms develop, symptoms worsen or the patient's condition appears to change.
# It is also recommended that patients are reassessed every hour, if they have
# not been seen by doctors yet."
#
# A triage level is a statement about a moment, and the document says so on the
# same page it sets the time-to-treatment standards. The report printed the
# standard and never the expiry, which reads as though one assessment holds.
# Levels 1 and 2 are excluded: a patient owed treatment now or within ten
# minutes is not a patient anyone is scheduling a re-look for.
MTS_REASSESSMENT = (
    "Re-triage every hour until seen by a doctor, and immediately if new "
    "symptoms develop, symptoms worsen or the condition appears to change "
    "(MTS 2022, p3)."
)


def _hypotension_threshold(age_years: float) -> float:
    """Systolic BP below which a patient of this age is hypotensive (APLS)."""
    if age_years < 1 / 12:
        return 60.0
    if age_years < 1:
        return 70.0
    if age_years <= 10:
        return 70.0 + 2.0 * age_years
    return 90.0


def _mts_floor(req: TriageRequest) -> tuple[int, list[str]]:
    """Most acute level the observations alone justify, with the reasons for it.

    Delegates to `mts_table`, which holds MTS 2022 pages 4, 7, 13 and 14
    transcribed cell by cell. The half-table this used to implement by hand
    never read temperature or respiratory rate, never checked hypertension or
    hyperglycaemia, and had no paediatric bands at all - a patient at 39.5 C,
    RR 28, BP 230/135 returned "floor 5, no reasons".
    """
    return mts_table.floor(
        req.age, req.vitals, _intake_text(req), req.observations()
    )


def _enforce_mts_level(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Escalate to the floor the vitals demand; de-escalate only in the narrow
    case where nothing objective supports an acute level."""
    floor, reasons = _mts_floor(req)
    level = int(diagnostic.mts_triage_level)
    v = req.vitals
    new: int | None = None
    note = ""

    if level > floor:
        new = floor
        note = (
            f"[System Guardrail: escalated MTS {level} -> {floor} on "
            f"{'; '.join(reasons)}]"
        )
    elif (
        floor >= 4
        and level <= 2
        and (v.pain_score or 0) <= 3
        and (v.gcs if v.gcs is not None else 15) >= 14
    ):
        new = 4 if (v.gcs if v.gcs is not None else 15) == 15 else 3
        note = f"[System Guardrail: corrected MTS {level} -> {new}, no acute criterion met]"

    if new is None:
        return
    diagnostic.mts_triage_level = TriageLevel(new)
    diagnostic.mts_triage_label = TriageLabel(TriageLevel(new).name)
    diagnostic.time_to_treatment = MTS_TIME[new]
    diagnostic.triage_rationale = f"{diagnostic.triage_rationale} {note}".strip()
    log.warning("GROUNDING  %s", note)


def _reassess(level: int) -> str:
    """The page 3 re-triage rule. Empty at Levels 1-2, where nobody is waiting."""
    return "" if level <= 2 else MTS_REASSESSMENT


def _ground_triage_timing(diagnostic: DiagnosticSchema) -> None:
    """Derive time_to_treatment from the final level instead of trusting the model.

    It is a lookup against one five-row table on MTS 2022 page 3, which is the
    same reason `mts_triage_label` and the citation table were taken out of the
    prompt: a value the server can derive exactly should never be spent on
    output tokens or exposed to a transcription error. `_enforce_mts_level`
    already set it, but ONLY when it changed the level - a report the guardrail
    agreed with kept whatever time the model typed, including the schema default
    of "under 30 minutes" on an MTS 5 patient.

    Runs after every level-changing check, so it reads the level the patient
    actually leaves with.
    """
    level = int(diagnostic.mts_triage_level)
    diagnostic.time_to_treatment = MTS_TIME[level]
    diagnostic.reassessment = _reassess(level)


# ------------------------------------------------------- formulary querying
# FUKKM must be searched with the TREATMENT named in the clinical guidance, not
# with the patient's symptoms. "fever, bleeding, vomiting" retrieved snake
# antivenom and antihistamines for a dengue child; the dengue algorithm's own
# words ("Commence Normal Saline at 5-7 ml/kg") retrieve the crystalloids.

_TREATMENT_HINT = re.compile(
    r"(\bmg\b|\bml\b|/\s*kg|\biv\b|\bim\b|\boral\b|infusion|bolus|administer"
    r"|commence|\bgive\b|dose|dosage|saline|crystalloid|colloid|fluid|antibiotic"
    r"|analgesi|oxygen|adrenaline|paracetamol|dextrose|transfus|therapy|treatment)",
    re.I,
)


def _formulary_query(clinical: list[Retrieved], complaint: str) -> str:
    """Build the FUKKM query from treatment sentences in the clinical chunks."""
    picked: list[str] = []
    seen: set[str] = set()
    total = 0
    for chunk in clinical:
        for sentence in re.split(r"(?<=[.;])\s+|\n", chunk.text or ""):
            text = re.sub(r"\s+", " ", sentence).strip()
            if not (20 <= len(text) <= 300) or not _TREATMENT_HINT.search(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(text)
            total += len(text)
            if total > 700:  # all-MiniLM truncates at 256 tokens anyway
                break
        if total > 700:
            break
    if not picked:
        return f"drug dose indication for {complaint}"[:700]
    return " ".join(picked)


# ===========================================================================
# Engine
# ===========================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> str:
    r"""Pull the JSON object out of a model response.

    `re.search(r'\{.*\}')` alone spans the FIRST brace to the LAST one, so any
    stray brace in a reasoning trace or a trailing note silently corrupts the
    payload. Reasoning traces and code fences are stripped first, then braces are
    balanced (ignoring braces inside strings) so the object ends where it
    actually ends."""
    text = _THINK_RE.sub("", raw or "").strip()
    if m := _FENCE_RE.search(text):
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unterminated: the model hit the token cap mid-object. Close what is open
    # so the fields that DID arrive still parse. The schema is ordered with the
    # clinically decisive fields first (triage level, diagnosis, immediate
    # actions) and citations last, so a repaired object keeps what matters and
    # loses only the tail. A hard parse failure would discard all of it.
    tail = text[start:]
    if in_str:
        tail += '"'
    # Drop a trailing partial key/value so the repair does not invent content.
    tail = re.sub(r",\s*\"[^\"]*\"\s*:\s*$", "", tail)
    tail = re.sub(r",\s*$", "", tail)
    depth, in_str, esc = 0, False, False
    stack: list[str] = []
    for ch in tail:
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append(ch)
        elif ch in "}]" and stack: stack.pop()
    return tail + "".join("}" if c == "{" else "]" for c in reversed(stack))


# A field the schema cannot accept must never be defaulted silently. The triage
# level is the one field where a default is itself a clinical claim: falling
# back to URGENT would badge a report with an acuity no model ever chose, and
# `_enforce_mts_level` only raises a level to the floor the vitals demand - it
# cannot tell an invented 3 from a considered one.
_UNSALVAGEABLE = frozenset({"mts_triage_level"})


def _salvage(cleaned: str, exc: ValidationError) -> tuple[DiagnosticSchema | None, list[str]]:
    """Drop only the fields that failed validation and keep the rest.

    Same reasoning as the brace repair in `_extract_json`: a Mode A run costs
    minutes of local compute, and one unusable value is a poor reason to
    discard every good one. Every model-authored field has a default, so the
    object still validates without it.

    The dropped names are RETURNED, never swallowed - the caller puts them in
    `parse_warning`. A section that quietly fell back to its default is exactly
    the kind of omission a reader cannot see, which is the failure mode
    `completeness.py` exists to catch.

    Removal is at the top level, so an error deep in a list drops the whole
    list. That is deliberate: the reported name carries the full path, so the
    warning still says precisely what was lost."""
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, []
    if not isinstance(payload, dict):
        return None, []

    dropped: list[str] = []
    for err in exc.errors():
        loc = err.get("loc") or ()
        if not loc or not isinstance(loc[0], str) or loc[0] in _UNSALVAGEABLE:
            return None, []
        if loc[0] in payload:
            payload.pop(loc[0])
            dropped.append(".".join(str(p) for p in loc))
    if not dropped:
        return None, []
    try:
        return DiagnosticSchema.model_validate(payload), dropped
    except ValidationError:
        return None, dropped


def _gen_kwargs(engine) -> dict:
    """Decode-speed options, applied defensively.

    mlx-lm routes generation through different step functions depending on
    whether a draft model is present, and they do not accept the same kwargs.
    Rather than hard-code which combination this version supports, unsupported
    options are dropped on TypeError by _generate below."""
    kw: dict = {}
    if config.KV_BITS:
        kw["kv_bits"] = config.KV_BITS
    draft = engine._draft_model
    if draft is not None:
        kw["draft_model"] = draft
    return kw


def _generate(engine, prompt: str, max_tokens: int, job_id: str = "") -> str:
    """Streamed so progress reflects real work: prefill is exact (mlx-lm reports
    tokens processed against the total) and decode is counted per token."""
    kw = _gen_kwargs(engine)
    if job_id:
        kw["prompt_progress_callback"] = lambda done, total: progress.prefill(job_id, done, total)
    while True:
        try:
            if not job_id:
                return generate(
                    engine._model, engine._tokenizer, prompt=prompt,
                    max_tokens=max_tokens, verbose=False, **kw,
                )
            out, n = [], 0
            for resp in stream_generate(
                engine._model, engine._tokenizer, prompt=prompt,
                max_tokens=max_tokens, **kw,
            ):
                out.append(resp.text)
                n += 1
                if n % 8 == 0:  # ~8 tokens between updates keeps the lock cheap
                    progress.decode(job_id, n)
            progress.decode(job_id, n)
            return "".join(out)
        except TypeError as exc:
            # Drop whichever accelerator this build will not take, and retry.
            dropped = next((k for k in ("draft_model", "kv_bits") if k in kw and k in str(exc)), None)
            if dropped is None:
                dropped = next(iter(kw), None)
            if dropped is None:
                raise
            log.warning("Generation option %r unsupported here (%s); continuing without it.", dropped, exc)
            kw.pop(dropped)


def _build_prompt(tokenizer, messages: list[dict]) -> str:
    """Render the chat template, disabling reasoning-trace mode where the model
    has one.

    Qwen3 defaults `enable_thinking=True`, which emits a long <think> block
    before the answer. For Mode A that is pure cost: the JSON is what is parsed,
    the trace is discarded, and a 4000-token budget can be spent entirely on
    thinking without ever reaching the closing brace. Templates that do not take
    the flag reject the kwarg, so it is passed defensively."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class RagEngine:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._draft_model = None
        self.model_name = config.LOCAL_MLX_MODEL

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model_loaded(self):
        if self._model is None:
            log.info(f"🔄 Loading local MLX model ({self.model_name}) into M4 Unified Memory...")
            self._model, self._tokenizer = load(self.model_name)
            log.info("✅ Local MLX model loaded successfully!")
            if config.DRAFT_MODEL and self._draft_model is None:
                try:
                    log.info("Loading draft model %s for speculative decoding...", config.DRAFT_MODEL)
                    self._draft_model, _ = load(config.DRAFT_MODEL)
                except Exception as exc:
                    log.warning("Draft model unavailable (%s); decoding without it.", exc)
                    self._draft_model = None

    @cached_property
    def collection(self):
        return get_collection()

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    @cached_property
    def corpus_warnings(self) -> list[str]:
        warnings = []
        if not self.count():
            return ["Vector index empty."]
        missing = red_flags.validate_titles(self.indexed_titles)
        if missing:
            # A dead rule looks like a safety net without being one. Surface it.
            warnings.append(
                "Red-flag rules point at guideline titles that are not in the "
                f"index, so those safety nets cannot fire: {', '.join(missing)}. "
                "A file was renamed or removed - fix red_flags.py or re-ingest."
            )
        return warnings

    # ------------------------------------------------- lexical (BM25) channel
    @cached_property
    def _corpus(self) -> dict:
        """Every chunk, pulled once, for the BM25 index. Built from the live
        collection rather than a side file so it can never drift out of sync
        with what the dense retriever is searching."""
        got = self.collection.get(include=["documents", "metadatas"])
        ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
        log.info("Building BM25 index over %d chunks...", len(ids))
        t0 = time.time()
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        tokenised = [_lex_tokens(d) for d in docs]
        bm25 = BM25Okapi(tokenised)
        log.info("BM25 index ready in %.1fs.", time.time() - t0)
        return {
            "ids": ids,
            "docs": docs,
            "metas": [m or {} for m in metas],
            "bm25": bm25,
            "pos": {cid: i for i, cid in enumerate(ids)},
        }

    @cached_property
    def indexed_titles(self) -> set[str]:
        return {m.get("cpg_title", "") for m in self._corpus["metas"]}

    @cached_property
    def _reranker(self):
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        log.info("Loading cross-encoder %s...", config.RERANKER_MODEL)
        return CrossEncoder(config.RERANKER_MODEL)

    def _bm25_search(
        self, query: str, k: int, doc_types: list[str] | None
    ) -> list[Retrieved]:
        c = self._corpus
        scores = c["bm25"].get_scores(_lex_tokens(query))
        allowed = set(doc_types) if doc_types else None
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        out: list[Retrieved] = []
        for i in idx:
            if scores[i] <= 0:
                break
            meta = c["metas"][i]
            if meta.get("status") != "active":
                continue
            if allowed and meta.get("doc_type") not in allowed:
                continue
            out.append(
                Retrieved(
                    chunk_id=c["ids"][i],
                    text=c["docs"][i],
                    metadata=meta,
                    # BM25 is a relevance score, not a distance. Kept separate so
                    # nothing downstream mistakes it for a cosine value.
                    distance=None,
                    lexical_score=float(scores[i]),
                )
            )
            if len(out) >= k:
                break
        return out

    def _dense_search(
        self, query: str, k: int, doc_types: list[str] | None
    ) -> list[Retrieved]:
        conditions = [{"status": "active"}]
        if doc_types:
            conditions.append({"doc_type": {"$in": doc_types}})
        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RagError(f"ChromaDB query failed: {exc}") from exc
        return [
            Retrieved(chunk_id=cid, text=doc, metadata=meta or {}, distance=dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    @staticmethod
    def _rrf(*ranked: list[Retrieved]) -> list[Retrieved]:
        """Reciprocal Rank Fusion. Uses rank, not score, so a BM25 relevance
        number and a cosine distance never have to be made commensurable."""
        scores: dict[str, float] = {}
        item: dict[str, Retrieved] = {}
        for lst in ranked:
            for rank, r in enumerate(lst, start=1):
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
                cur = item.get(r.chunk_id)
                if cur is None:
                    item[r.chunk_id] = r
                    continue
                # A chunk both channels found must carry BOTH scores, or the
                # provenance shown to the reader under-reports how it was found.
                item[r.chunk_id] = replace(
                    cur,
                    distance=cur.distance if cur.distance is not None else r.distance,
                    lexical_score=(
                        cur.lexical_score if cur.lexical_score is not None else r.lexical_score
                    ),
                )
        order = sorted(scores, key=lambda cid: -scores[cid])
        return [replace(item[cid], fusion_score=scores[cid]) for cid in order]

    def _rerank(self, query: str, chunks: list[Retrieved], k: int) -> list[Retrieved]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        scored = [replace(c, rerank_score=float(s)) for c, s in zip(chunks, scores)]
        scored.sort(key=lambda c: -c.rerank_score)
        return scored[:k]

    # ------------------------------------------------------------ retrieval
    def retrieve(
        self,
        query: str,
        k: int,
        doc_types: list[str] | None = None,
        *,
        rerank: bool | None = None,
    ) -> list[Retrieved]:
        """Dense + BM25 -> RRF -> cross-encoder rerank -> top k.

        Each stage catches what the previous one structurally cannot: BM25 finds
        exact clinical terms an embedder may not place ("haemoptysis"), and the
        cross-encoder judges query/chunk *interactions* a bi-encoder cannot."""
        if self.count() == 0:
            raise RagError("Vector index empty.")

        use_rerank = config.RERANK_ENABLED if rerank is None else rerank
        # Retrieve wide, then narrow. Reranking only helps if the right chunk is
        # somewhere in the candidate pool to begin with.
        width = max(k, config.CANDIDATE_K) if (use_rerank or config.HYBRID_ENABLED) else k

        dense = self._dense_search(query, width, doc_types)
        if not config.HYBRID_ENABLED:
            fused = dense
        else:
            try:
                lexical = self._bm25_search(query, width, doc_types)
            except Exception as exc:
                # Lexical is an enhancement; never let it take down retrieval.
                log.warning("BM25 channel failed (%s: %s); dense only.", type(exc).__name__, exc)
                lexical = []
            fused = self._rrf(dense, lexical) if lexical else dense

        if not use_rerank:
            return fused[:k]
        try:
            return self._rerank(query, fused[: config.CANDIDATE_K], k)
        except Exception as exc:
            log.warning("Reranker failed (%s: %s); returning fused order.", type(exc).__name__, exc)
            return fused[:k]

    # --------------------------------------------------- red-flag recall floor
    def apply_red_flags(
        self, intake: str, chunks: list[Retrieved], doc_types: list[str] | None = None
    ) -> tuple[list[Retrieved], list[str]]:
        """Force in the guidelines the intake demands, whatever retrieval scored.
        Only ever ADDS - a firing rule can never mask a correct retrieval."""
        forced = red_flags.titles_for(intake)
        if not forced:
            return chunks, []
        present = {c.metadata.get("cpg_title") for c in chunks}
        have = {c.chunk_id for c in chunks}
        added, notes = [], []
        for title, reasons in forced.items():
            if title in present:
                # The rule DID fire - retrieval had simply already found this
                # guideline. Recording it distinguishes "no red flag in this
                # intake" from "the safety net checked and was not needed",
                # which are very different things to a reviewing clinician.
                notes.append(f"{title} <- {'; '.join(reasons)} [already retrieved]")
                continue
            budget = config.RED_FLAG_MAX_CHUNKS - len(added)
            if budget <= 0:
                break
            extra = [
                c
                for c in self._title_chunks(intake, title, doc_types)
                if c.chunk_id not in have
            ][: min(config.RED_FLAG_CHUNKS_PER_RULE, budget)]
            if not extra:
                continue
            added.extend(extra)
            have.update(c.chunk_id for c in extra)
            notes.append(f"{title} <- {'; '.join(reasons)}")
        if added:
            log.info("Red flags forced %d chunk(s): %s", len(added), " | ".join(notes))
        return chunks + added, notes

    def _title_chunks(
        self, query: str, title: str, doc_types: list[str] | None
    ) -> list[Retrieved]:
        """Best chunks from one named guideline, ranked against the intake."""
        conditions = [{"status": "active"}, {"cpg_title": title}]
        if doc_types:
            conditions.append({"doc_type": {"$in": doc_types}})
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=config.RED_FLAG_CHUNKS_PER_RULE * 3,
                where={"$and": conditions},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.warning("Red-flag fetch for %r failed: %s", title, exc)
            return []
        got = [
            Retrieved(chunk_id=cid, text=doc, metadata=meta or {}, distance=dist, forced=True)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        if got and config.RERANK_ENABLED:
            try:
                return self._rerank(query, got, config.RED_FLAG_CHUNKS_PER_RULE)
            except Exception:
                pass
        return got[: config.RED_FLAG_CHUNKS_PER_RULE]

    def discriminator_cells(
        self, query: str, level: int, cohort: str, k: int
    ) -> list[Retrieved]:
        """The MTS discriminator cells for a triage level, by metadata filter.

        Item 3's whole point: the triage grid is indexed one cell per level, so
        the criteria for the level in question are FETCHED, not hoped for. The
        adjacent levels come too - a triage decision is a discrimination between
        neighbouring columns, and a model shown only the Level 5 column has
        nothing to rule out.

        Only ever ADDS to the retrieved set, like the red-flag floor.
        """
        lo, hi = max(1, level - 1), min(5, level + 1)
        cohorts = (
            ["PAEDIATRIC", "GENERAL"] if cohort == "paediatric" else ["ADULT", "GENERAL"]
        )
        conditions = [
            {"status": "active"},
            # Added 2026-09-11. These four keys used to be the whole filter, and
            # nothing tied them to the Malaysian Triage Scale: ANY indexed
            # document whose tables happened to print level columns would be
            # served here as triage criteria, with `forced=True`, bypassing the
            # ranking entirely. The corpus now holds a state redirection policy
            # and an HTA report, both of which discuss triage levels. The grid
            # is MTS 2022 and nothing else.
            {"doc_type": config.DOC_TYPE_TRIAGE},
            {"is_table_cell": True},
            # Bounded at 1, which is what keeps MTS_LEVEL_CLEARANCE (0) - the
            # SECONDARY TRIAGE column of pages 4-5 - out of every fetch.
            {"mts_level": {"$in": list(range(lo, hi + 1))}},
            {"cohort": {"$in": cohorts}},
        ]
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=max(k * 3, 24),
                where={"$and": conditions},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.warning("Discriminator-cell fetch failed: %s", exc)
            return []
        got = [
            Retrieved(chunk_id=cid, text=doc, metadata=meta or {}, distance=dist, forced=True)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        if not got:
            log.warning(
                "No discriminator cells for levels %d-%d (%s). The triage grid "
                "was indexed as prose only - re-ingest.", lo, hi, cohort,
            )
            return []
        if config.RERANK_ENABLED:
            try:
                return self._rerank(query, got, k)
            except Exception:
                pass
        return got[:k]

    @staticmethod
    def _dedupe(*groups: list[Retrieved]) -> list[Retrieved]:
        seen = set()
        merged = []
        for group in groups:
            for item in group:
                if item.chunk_id not in seen:
                    seen.add(item.chunk_id)
                    merged.append(item)
        return merged

    # Each guideline now ships as a full CPG *and* an 8-page quick reference, so
    # the same recommendation can occupy several retrieval slots twice over. The
    # full CPG carries the graded recommendation and wins; the QR keeps a couple
    # of slots because its severity/staging tables sit whole in one chunk where
    # the full CPG spreads them over pages. Guidelines with no full CPG indexed
    # (and Mode B's QUICK_REFERENCE_ONLY scope) are untouched.
    MAX_QR_CHUNKS_PER_GUIDELINE = 2

    @classmethod
    def _prefer_full_cpg(cls, chunks: list[Retrieved]) -> list[Retrieved]:
        full_titles = {
            c.metadata.get("cpg_title")
            for c in chunks
            if c.metadata.get("doc_type") == config.DOC_TYPE_CPG_FULL
        }
        kept: list[Retrieved] = []
        qr_seen: dict[str, int] = {}
        for c in chunks:
            title = c.metadata.get("cpg_title")
            if (
                c.metadata.get("doc_type") == config.DOC_TYPE_CPG_QR
                and title in full_titles
            ):
                qr_seen[title] = qr_seen.get(title, 0) + 1
                if qr_seen[title] > cls.MAX_QR_CHUNKS_PER_GUIDELINE:
                    continue
            kept.append(c)
        return kept

    # ------------------------------------------------------- prompt context
    def _context(self, chunks: list[Retrieved]) -> tuple[str, list[RetrievedSource]]:
        blocks = []
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            sid = f"S{i}"
            meta = chunk.metadata
            doc_info = f"[{sid}] document='{meta.get('cpg_title', '?')}' type='{meta.get('doc_type', '?')}' version='{meta.get('version_date', '?')}'"
            blocks.append(f"{doc_info}\n{chunk.text}")
            sources.append(
                RetrievedSource(
                    source_id=sid,
                    filename=str(meta.get("filename", "")),
                    cpg_title=str(meta.get("cpg_title", "")),
                    edition_year=str(meta.get("edition_year", "")),
                    doc_type=str(meta.get("doc_type", "")),
                    page_number=meta.get("page_number"),
                    drug_name=meta.get("drug_name"),
                    prescriber_category=meta.get("prescriber_category"),
                    score=chunk.similarity,
                    retrieval=chunk.how,
                    # Scrubbed: 5% of indexed chunks carry Wingdings glyphs
                    # from the source PDFs that print as tofu boxes.
                    excerpt=_clean_quote(chunk.text)[:600],
                )
            )
        return "\n\n---\n\n".join(blocks), sources

    # ============================================================== MODE A
    def triage(self, req: TriageRequest, job_id: str = "") -> TriageResponse:
        progress.stage(job_id, "queued", "Loading models", 0.5)
        self._ensure_model_loaded()
        started = time.perf_counter()
        progress.stage(job_id, "retrieval", "Searching MTS 2022 triage criteria", 0.1)
        
        vitals = req.vitals.as_clinical_text()
        cohort = "paediatric" if req.age < 12 else "adult"

        # 1. Triage criteria. Free-text search over the triage document, plus
        #    the discriminator cells for the level the vitals themselves justify
        #    - fetched by metadata filter, not left to ranking. The floor is
        #    deterministic and costs nothing, so it is computed here rather than
        #    only after generation.
        triage_chunks = self.retrieve(
            f"MTS 2022 triage {req.complaint} {vitals}",
            config.TRIAGE_K,
            [config.DOC_TYPE_TRIAGE],
        )
        floor_level, _ = _mts_floor(req)
        triage_chunks = self._dedupe(
            triage_chunks,
            self.discriminator_cells(
                f"{req.complaint} {vitals}", floor_level, cohort, config.MTS_CELL_K
            ),
        )

        # 2. Clinical guidance. CLINICAL_DOC_TYPES includes the Paediatric
        #    Protocols (1,190 chunks) - previously excluded here, so the richest
        #    paediatric source in the corpus was never retrieved for Mode A.
        progress.stage(job_id, "retrieval", "Searching clinical practice guidelines", 0.4)
        clinical_chunks = self.retrieve(
            f"{req.complaint} {req.history} {_retrieval_terms(req)} "
            "diagnosis and management",
            config.CPG_K,
            config.CLINICAL_DOC_TYPES,
        )
        if cohort == "paediatric":
            clinical_chunks = self._dedupe(
                clinical_chunks,
                self.retrieve(
                    f"child paediatric management of {req.complaint}",
                    config.PAEDS_K,
                    [config.DOC_TYPE_PAEDS],
                ),
            )

        # 3. Formulary, queried with the treatment named in (2), not the symptoms.
        progress.stage(job_id, "retrieval", "Searching the FUKKM formulary", 0.75)
        drug_chunks = self.retrieve(
            _formulary_query(clinical_chunks, req.complaint),
            config.DRUG_K,
            [config.DOC_TYPE_DRUG],
        )

        chunks = self._prefer_full_cpg(
            self._dedupe(triage_chunks, clinical_chunks, drug_chunks)
        )

        # 4. Recall floor. Whatever the retriever scored, a presentation that
        #    names a can't-miss pattern gets that guideline put in front of the
        #    model. Adds only - it cannot displace a correct retrieval.
        red_flag_notes: list[str] = []
        if config.RED_FLAGS_ENABLED:
            chunks, red_flag_notes = self.apply_red_flags(
                _intake_text(req), chunks, config.CLINICAL_DOC_TYPES
            )

        context, sources = self._context(chunks)
        weight = f"{req.weight_kg:g} kg" if req.weight_kg else "NOT SUPPLIED"

        # Every value is a SLOT MARKER, not a suggestion. The previous template
        # carried plausible clinical prose ("Reason based on worst criterion...")
        # and a 3B model copied it verbatim into the patient's rationale. Markers
        # are obviously non-clinical, so a copy is detectable (see _slots_leaked).
        #
        # "citations" is deliberately ABSENT for the same reason the label is.
        # The model wrote FUKKM and nothing else into it while naming the
        # rhinosinusitis CPG in prose, and the page numbers it did write were
        # its own transcription. _ground_citations builds the table from the
        # index instead, so asking for it here only spends output tokens on a
        # value that is thrown away. Rule 5 below still asks for [S#] on each
        # recommendation - those ids are what the table is derived FROM.
        # mts_triage_label is deliberately ABSENT: it is derived from the level in
        # DiagnosticSchema, so asking for it only spends output tokens on a value
        # that is discarded - and it is what the model got wrong, returning the
        # display form "EARLY CARE" for MTS 4 and failing the whole parse.
        # time_to_treatment left for the same reason on 2026-09-11: it is the
        # page 3 lookup for the level, and _ground_triage_timing now derives it
        # from the level the guardrails settle on rather than the one the model
        # first proposed. `reassessment` comes from the same place.
        json_template = """{
  "mts_triage_level": 3,
  "triage_rationale": "<why THIS patient is that level, naming the worst criterion>",
  "vitals_interpretation": [{"parameter": "<vital>", "value": "<as supplied>", "interpretation": "<what it means>", "mts_level_triggered": "<level as text, e.g. \"1\">"}],
  "red_flags": [{"flag": "<red flag>", "why_it_matters": "<why>", "source_id": "S1"}],
  "primary_diagnosis": {"condition": "<diagnosis>", "confidence": "HIGH", "reasoning": "<reasoning from THIS patient's findings>", "supporting_cpg": "<guideline title>"},
  "differential_diagnoses": [{"condition": "<condition>", "discriminating_feature": "<what separates it>"}],
  "immediate_actions": [{"sequence": 1, "action": "<action, with mg/kg or ml/kg where the weight is known>", "timeframe": "<when>", "source_id": "S1"}],
  "investigations": [{"test": "<test>", "rationale": "<why>", "urgency": "STAT"}],
  "drug_recommendations": [{"drug_name": "<drug>", "indication": "<indication>", "adult_dose": "<dose>", "paediatric_dose": "<mg/kg or ml/kg dose>", "route": "<route>", "frequency": "<frequency>", "duration": "<duration>", "prescriber_category": "<FUKKM category from a retrieved chunk, else NOT_IN_RETRIEVED_SOURCES>", "prescriber_category_meaning": "<what that category means>", "cautions": "<cautions>", "source_id": "S1"}],
  "prescriber_category_warning": "<name any drug whose category was NOT in the retrieved chunks, else empty>",
  "disposition": "ED_OBSERVATION",
  "disposition_justification": "<why this destination>",
  "referral_required": false,
  "referral_to": "<team, or empty>",
  "evidence_gaps": "<vitals not supplied, and anything the retrieved chunks did not cover>"
}"""

        user_prompt = f"""PATIENT
Age: {req.age:g} years ({cohort})
Gender: {req.gender.value}
Weight: {weight}
Vitals: {vitals}
Complaint: {req.complaint}
Pain Score: {getattr(req.vitals, 'pain_score', 'Not specified')}
History: {req.history or "None"}{_patient_extras(req)}

SOURCES ({len(chunks)} chunks)
{context}

TASK: Evaluate based on the Evaluation Sequence.
REMINDER 1: every <angle-bracket> below is a SLOT to fill with THIS patient's
data. Never copy a slot marker into your answer. The numbers and enum values
shown are placeholders illustrating the type - they are NOT recommendations.
REMINDER 2: for any drug with no FUKKM chunk above, prescriber_category MUST be
the literal string "NOT_IN_RETRIEVED_SOURCES" - do not guess a category letter.
Respond ONLY with a complete JSON object matching this exact structure:
{json_template}"""

        messages = [{"role": "system", "content": TRIAGE_SYSTEM}, {"role": "user", "content": user_prompt}]
        
        prompt = _build_prompt(self._tokenizer, messages)
        progress.stage(job_id, "prefill", "Reading the retrieved guideline context", 0.0)
        raw_response = _generate(self, prompt, config.MAX_TOKENS_TRIAGE, job_id)
        
        cleaned = _extract_json(raw_response)
        try:
            diagnostic = DiagnosticSchema.model_validate_json(cleaned)

        except ValidationError as exc:
            diagnostic, dropped = _salvage(cleaned, exc)
            if diagnostic is None:
                log.error(f"JSON Parsing failed. Raw output: {raw_response}")
                raise RagError(f"Model generated invalid JSON structure: {exc}")
            diagnostic.parse_warning = (
                "The model returned these fields unusably, so they show the schema "
                f"default and not any clinical judgement: {', '.join(dropped)}. "
                "Treat each as absent rather than as a finding."
            )
            log.warning("GROUNDING  [Salvaged unusable fields: %s]", ", ".join(dropped))

        except Exception as exc:
            log.error(f"JSON Parsing failed. Raw output: {raw_response}")
            raise RagError(f"Model generated invalid JSON structure: {exc}")

        progress.stage(job_id, "checks", "Running triage, grounding and safety checks", 0.3)
        # Enforced in code, not asked for in the prompt.
        _enforce_mts_level(diagnostic, req)
        _ground_prescriber_categories(diagnostic, sources)
        _check_drug_indications(diagnostic, chunks, self._corpus)
        _check_dose_completeness(diagnostic, req)
        _check_dosing(diagnostic, req, chunks)
        _check_contraindications(diagnostic, req)
        _enforce_disposition(diagnostic, req)
        _check_redirection(diagnostic, req)
        _check_urgency_sanity(diagnostic, req)
        # Last of the checks that can move the level, so it reads the final one.
        _ground_triage_timing(diagnostic)
        _check_completeness(diagnostic, req)
        _check_citations(diagnostic, sources, _ground_citations(diagnostic, sources))
        _flag_leaked_slots(diagnostic)

        return TriageResponse(
            diagnostic=diagnostic,
            triage_colour=TRIAGE_COLOUR[int(diagnostic.mts_triage_level)],
            request=req,
            sources=sources,
            model=self.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            corpus_warnings=self.corpus_warnings,
            retrieval_notes=red_flag_notes,
        )

    # ============================================================== MODE B
    def inquire(self, req: InquiryRequest) -> InquiryResponse:
        self._ensure_model_loaded()
        started = time.perf_counter()
        
        doc_types = SCOPE_DOC_TYPES[req.scope]
        chunks = self.retrieve(req.query, req.max_sources, doc_types)

        if req.scope is InquiryScope.ALL:
            formulary = self.retrieve(req.query, max(4, req.max_sources // 4), [config.DOC_TYPE_DRUG])
            chunks = self._dedupe(chunks, formulary)
        chunks = self._prefer_full_cpg(chunks)

        context, sources = self._context(chunks)
        documents = sorted({s.cpg_title for s in sources if s.cpg_title})

        user_prompt = f"""QUESTION: {req.query}\n\nSOURCES:\n{context}\n\nAnswer using the markdown headers requested."""

        messages = [{"role": "system", "content": INQUIRY_SYSTEM}, {"role": "user", "content": user_prompt}]
        prompt = _build_prompt(self._tokenizer, messages)
        
        markdown = _generate(self, prompt, config.MAX_TOKENS_INQUIRY).strip()
        
        if not markdown:
            raise RagError("The local model returned an empty answer.")

        return InquiryResponse(
            query=req.query,
            answer_markdown=markdown,
            sources=sources,
            documents_scanned=documents,
            model=self.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            corpus_warnings=self.corpus_warnings,
        )

_engine: RagEngine | None = None
def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine