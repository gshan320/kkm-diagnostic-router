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
from dataclasses import dataclass, field, replace
from functools import cached_property

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models import cache as cache_mod
from pydantic import ValidationError

from . import (
    attention_audit, cards, complications, disposition_rules,
    completeness, config, contraindications, distill, dosing, focus, formulary, gpu_queue,
    indications, mts_table, negatives, population, progress, red_flags, redirection,
    references, versioning,
)
from .ingest import get_collection
from .schemas import (
    Confidence,
    ExternalReference,
    TriagePreview,
    WithheldDrug,
    Citation,
    CompletenessGap,
    Disposition,
    DrugRecommendation,
    ImmediateAction,
    Investigation,
    Contraindication,
    DiagnosticSchema,
    InquiryRequest,
    InquiryResponse,
    InquiryScope,
    KnowledgeGap,
    RedFlag,
    RetrievedSource,
    SourceCaution,
    VitalInterpretation,
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
     disposition only - never diagnosis, treatment or triage level.
   - EXTERNAL_GUIDELINE: a non-KKM international guideline, indexed only where
     no KKM document covers the condition. Use it for what the KKM sources do
     not say; where both speak, the KKM source wins."""

_DRUG_GROUNDING_FULL = """7. DRUG GROUNDING - ABSOLUTE. Never state a dose, route, frequency or
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

# A4.4: categories are read from the FUKKM record by _ground_prescriber_categories,
# so the model is no longer asked for them - fewer output tokens, and no guessed
# letter to correct.
_DRUG_GROUNDING_DIET = """7. DRUG GROUNDING - ABSOLUTE. Never state a dose, route or frequency that is
   not written in the retrieved [S#] chunks. Do not write prescriber categories:
   the server reads them from the FUKKM record.
8. GAP GROUNDING. A vital sign that was not supplied is a GAP, never a normal
   value. Say so in "evidence_gaps" rather than assuming it is normal."""

DRUG_GROUNDING = _DRUG_GROUNDING_DIET if config.OUTPUT_DIET else _DRUG_GROUNDING_FULL

_REMINDER_CATEGORY = """REMINDER 2: for any drug with no FUKKM chunk above, prescriber_category MUST be
the literal string "NOT_IN_RETRIEVED_SOURCES" - do not guess a category letter.
"""

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

# The second pass (RagEngine._second_stage) asks for ONLY what the first answer
# left out, for a diagnosis that is already fixed. Short on purpose: it runs in
# the same GPU budget as the first pass and must not re-open the triage.
STAGE2_SYSTEM = """You complete a Malaysian emergency-department management plan.
The working diagnosis is FIXED - do not change it, do not re-triage.
Use ONLY the SOURCES given. Every item must cite the [S#] it comes from.
Add only items that address the MISSING list; do not repeat anything in ALREADY IN THE PLAN.
Give a drug dose ONLY if the cited source states it; otherwise leave the dose empty.
Respond ONLY with one JSON object."""

# Attention layer A2: the model names what the patient most likely has BEFORE
# anything is retrieved, so retrieval can look up those conditions instead of
# every document whose wording resembles the symptoms. Deliberately tiny.
HYPOTHESIS_SYSTEM = """You are an emergency physician forming a working differential.
From the KEY FINDINGS only, name the most likely diagnoses and the dangerous
alternatives that must be excluded. Use standard medical terms, no explanation.
Respond ONLY with one JSON object."""

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
        # A4.4: the model is no longer asked for the category - the record is
        # the answer - so only a category it DID state and got wrong is news.
        if claimed and claimed != UNVERIFIED and claimed != actual:
            corrected.append(f"{drug.drug_name}: {claimed} to {actual}")
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
    "and or of for the with "
    # Routes and vehicles, not drugs. Found 2026-09-30: "Intravenous calcium"
    # took the alias "intravenous", so an action "Administer intravenous
    # fluids" was reported as also naming withheld calcium.
    "intravenous intramuscular subcutaneous sublingual nebulised nebulized inhaled "
    "topical rectal fluid fluids".split()
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
        if drug.indication_status in (indications.SUPPORTED, indications.SYMPTOMATIC):
            # The indication gate already grounded this drug, with the quote;
            # a second, weaker verdict beside it only contradicts it (paracetamol
            # for pain was shown "NOT VERIFIED" under a quoted FUKKM indication).
            drug.indication_supported = True
            continue
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
        # Appended, never overwritten: the indication gate wrote its WITHHELD
        # note first, and overwriting it made a withheld drug vanish from the
        # report without a trace (2026-09-30).
        diagnostic.drug_indication_warning = f"{diagnostic.drug_indication_warning} {note}".strip()
        for n in notes:
            log.warning("GROUNDING  %s", n)


# ------------------------------------------------- indication gate
_NO_DIAGNOSIS = {"", "unspecified diagnosis", "unknown"}
_ROUTE_WORDS = frozenset("intravenous intramuscular subcutaneous sublingual oral nebulised "
                         "nebulized inhaled topical rectal injection infusion tablet".split())
_ADMINISTER = re.compile(r"\b(?:administer\w*|give|giving|start\w*|commence\w*|infus\w*|"
                         r"prescrib\w*|bolus|load\w*)\b", re.I)


def _where(meta: dict) -> str:
    """How a passage is cited in a server-written note: title and page for a
    PDF, title and chapter for a web-only guideline (the NAG 2024)."""
    title = meta.get("cpg_title", "?")
    if meta.get("url"):
        return f"{title}, {meta.get('section', 'web page')}"
    return f"{title} p{meta.get('page_number', '?')}"


def _gate_drug_indications(
    diagnostic: DiagnosticSchema, req: TriageRequest, chunks: list[Retrieved]
) -> None:
    """Withhold every drug that no source links to THIS patient.

    See indications.py for the verdicts. Runs BEFORE the category, indication-
    wording and dose checks, so a withheld drug is never dose-"verified" - the
    heparin-for-rhabdomyolysis report carried DOSE VERIFIED under a drug the
    indication check had already flagged."""
    if not diagnostic.drug_recommendations:
        return
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    working = indications.concepts(primary) if primary.lower() not in _NO_DIAGNOSIS else set()
    differential: set[str] = set()
    for dd in diagnostic.differential_diagnoses:
        differential |= indications.concepts(dd.condition)
    differential -= working
    guideline = [
        (_where(c.metadata), c.text)
        for c in chunks if c.metadata.get("doc_type") in config.CLINICAL_DOC_TYPES
    ]
    patient = indications.signals(req, diagnostic)
    paediatric = req.age < config.PAEDIATRIC_AGE_YEARS

    kept, withheld, notes, conditional = [], [], [], []
    for drug in diagnostic.drug_recommendations:
        name = str(drug.drug_name or "")
        stated = str((drug.paediatric_dose if paediatric else drug.adult_dose) or drug.adult_dose or "")
        aliases = _drug_aliases(name)
        pattern = _alias_pattern(aliases) if aliases else re.compile(r"(?!)")
        match = formulary.lookup(name, str(drug.route or ""), stated)
        entries = list(match.entries) if match else []
        if not working:
            verdict = indications.Verdict(indications.WITHHELD,
                                          "the report has no working diagnosis to check it against")
        else:
            verdict = indications.judge(pattern, entries, working, differential, guideline, patient,
                                        drug_is_generic_fluid=bool(indications.GENERIC_FLUID_RE.match(name)))
        drug.indication_status = verdict.status
        drug.indication_basis = verdict.basis
        if verdict.status == indications.WITHHELD:
            withheld.append(WithheldDrug(
                drug_name=name, route=str(drug.route or ""), stated_dose=stated,
                reason=(f"No source links {name} to the working diagnosis "
                        f"({primary or 'none given'})."),
                basis=verdict.basis,
            ))
            continue
        if verdict.status == indications.CONDITIONAL:
            conditional.append(f"{name} (indicated for {verdict.concept}, a differential - "
                               "give only if that is confirmed)")
        kept.append(drug)

    diagnostic.drug_recommendations = kept
    diagnostic.withheld_drugs = withheld
    if withheld:
        notes.append(
            "WITHHELD - no source links these to this patient's working diagnosis, so they "
            "have been removed from the drug list: "
            + "; ".join(f"{w.drug_name} ({w.basis})" for w in withheld) + "."
        )
        # Scan actions too: the drug check alone missed GTN-with-sildenafil when
        # it arrived as an immediate action.
        for w in withheld:
            p = _alias_pattern(_drug_aliases(w.drug_name)) if _drug_aliases(w.drug_name) else None
            # Generic words ("calcium") are not aliases, but "Administer
            # intravenous calcium to prevent hyperkalaemia" still orders it.
            words = [x for x in re.findall(r"[a-z]{4,}", w.drug_name.lower())
                     if x not in _ROUTE_WORDS and not formulary._FORM.fullmatch(x)]
            generic = re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.I) if words else None
            keep = []
            for a in diagnostic.immediate_actions:
                text = a.action or ""
                if (p and p.search(text)) or (generic and generic.search(text) and _ADMINISTER.search(text)):
                    # Removed, not just noted: on 2026-10-01 atropine and
                    # dopamine stayed in the action list while the drug list
                    # said both were withheld - the report contradicted itself.
                    notes.append(f"The action \"{text[:80]}\" orders {w.drug_name} and was removed with it.")
                    continue
                keep.append(a)
            diagnostic.immediate_actions = keep
    if conditional:
        notes.append("CONDITIONAL - " + "; ".join(conditional) + ".")
    if notes:
        note = " ".join(notes)
        diagnostic.drug_indication_warning = f"{note} {diagnostic.drug_indication_warning}".strip()
        for n in notes:
            log.warning("GROUNDING  %s", n)


def _check_avoid_statements(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Quote any source sentence that advises against a recommended drug for
    this diagnosis (negatives.py). Warns; never removes.

    A drug the indication gate found SUPPORTED for this diagnosis only draws a
    statement whose own sentence names the diagnosis - aspirin is the first
    drug in STEMI, and a document-wide "not recommended" about enteric-coated
    loading doses is not a reason to alarm on it."""
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if primary.lower() in _NO_DIAGNOSIS:
        return
    working = indications.concepts(primary)
    paediatric = req.age < config.PAEDIATRIC_AGE_YEARS
    lines = []
    for drug in diagnostic.drug_recommendations:
        hits = negatives.check(str(drug.drug_name or ""), working, paediatric=paediatric)
        if drug.indication_status == indications.SUPPORTED:
            hits = [h for h in hits if h["matched_in"] == "sentence"]
        for h in hits:
            title = re.sub(r"\.pdf$", "", h["source"], flags=re.I)
            lines.append(f"{drug.drug_name}: \"{h['sentence'][:260]}\" [{title}, p{h['page']}]")
    if lines:
        diagnostic.avoid_warning = (
            "A SOURCE ADVISES AGAINST a recommended drug for this diagnosis - read the "
            "quoted sentence before prescribing: " + " | ".join(lines)
        )
        log.warning("GROUNDING  %s", diagnostic.avoid_warning)


# ------------------------------------------------- truncated / looping output
def _was_truncated(raw: str) -> bool:
    """True when the model's JSON never closed - it hit the token cap mid-object
    and `_extract_json` had to repair it."""
    text = _THINK_RE.sub("", raw or "")
    start = text.find("{")
    if start == -1:
        return False
    depth, in_str, esc = 0, False, False
    for ch in text[start:]:
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
                return False
    return True


def _repetition(raw: str, window: int = 60, times: int = 4) -> str:
    """A passage the model repeated `times`+ times, else "". Measured on the
    2026-09-30 baseline: one sentence about SpO2 repeated for 15,418 characters
    until the token cap."""
    text = re.sub(r"\s+", " ", raw or "")
    if len(text) < window * times:
        return ""
    counts: dict[str, int] = {}
    for i in range(0, len(text) - window, window // 2):
        piece = text[i:i + window]
        counts[piece] = counts.get(piece, 0) + 1
        if counts[piece] >= times:
            return piece.strip()
    return ""


def _flag_truncation(diagnostic: DiagnosticSchema, raw: str, cleaned: str) -> None:
    """Say so when the answer was cut off, and which sections never arrived.

    Without this the repaired object validates, the missing sections take their
    schema defaults ("Unspecified Diagnosis", no actions, no drugs) and the
    report LOOKS complete - the failure a reader cannot see."""
    if not _was_truncated(raw):
        return
    try:
        present = set(json.loads(cleaned).keys())
    except Exception:
        present = set()
    authored = ["primary_diagnosis", "red_flags", "differential_diagnoses", "immediate_actions",
                "investigations", "drug_recommendations", "disposition", "citations"]
    missing = [f.replace("_", " ") for f in authored if f not in present]
    loop = _repetition(raw)
    why = (f"the model repeated itself (\"{loop[:80]}...\") until it ran out of space"
           if loop else "the model's answer reached the length limit before it finished")
    note = (f"INCOMPLETE - {why}. These sections were never written and show empty defaults, "
            f"not findings: {', '.join(missing) or 'the final fields'}. Do not read absence here as "
            "absence of need; re-run the case or assess directly.")
    diagnostic.parse_warning = f"{note} {diagnostic.parse_warning}".strip()
    log.warning("GROUNDING  %s", note)


# ------------------------------------------------- population gate
def _filter_population(chunks: list[Retrieved], age: float) -> tuple[list[Retrieved], list[str]]:
    """Drop chunks written for a different age group (population.py).

    Applied once, to the final context, so it covers every way a chunk can
    arrive - ranked retrieval, the red-flag recall floor, the discriminator
    cells. Measured 2026-09-30: the rhabdomyolysis report for a 55-year-old
    cited the Paediatric Protocols' sea-snake envenomation page."""
    kept, dropped = [], {}
    for c in chunks:
        if population.allowed(c.metadata, age):
            kept.append(c)
        else:
            t = str(c.metadata.get("cpg_title", "?"))
            dropped[t] = dropped.get(t, 0) + 1
    notes = []
    if dropped:
        notes.append(
            f"Population filter: left out {sum(dropped.values())} passage(s) written for a different "
            f"age group than {population.label(age)} - " + "; ".join(f"{t} ({n})" for t, n in dropped.items())
        )
    return kept, notes


# Words too general to tie a diagnosis to a guideline TITLE on their own.
# Narrower than negatives._AMBIGUOUS: body-system words such as "coronary" or
# "hypertension" are exactly what a title is named for.
_TITLE_AMBIGUOUS = frozenset({"stroke", "attack", "arrest", "failure", "infection", "injury",
                              "fever", "emergency", "urgency", "acute", "severe"})


def _check_governing_topic(diagnostic: DiagnosticSchema) -> None:
    """The governing guideline must be ABOUT the working diagnosis.

    Measured 2026-09-30 (second run of the rhabdomyolysis case): with the
    paediatric source filtered out, the model named "MOH Guideline Management
    of Snakebite" as governing - retrieved because sea-snake envenoming causes
    myoglobinuria. Right age group, wrong condition. A guideline is accepted
    when its title names the diagnosis (or a synonym - "STEMI" meets "Acute
    Coronary Syndromes"), or, for the NAG, when one of its chapters covers the
    diagnosis. Otherwise the name is removed and the report says that no
    indexed guideline governs this diagnosis (references.py then offers the
    MaHTAS CPG list)."""
    named = str(diagnostic.primary_diagnosis.supporting_cpg or "").strip()
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if not named or primary.lower() in _NO_DIAGNOSIS:
        return
    concepts = indications.concepts(primary)
    # Single words count unless ambiguous: "heat" must meet "Heat Related
    # Illness", but "stroke" alone must not join heat stroke to a stroke CPG.
    strong = {c for c in concepts if " " in c or (len(c) >= 4 and c not in _TITLE_AMBIGUOUS)}
    if indications.mentions(strong, named):
        return
    if "antimicrobial" in named.lower() and references.select(primary, 30, need_guideline=False):
        return
    diagnostic.primary_diagnosis.supporting_cpg = ""
    note = (f"The model named '{named}' as the governing guideline, but it is not a guideline for "
            f"{primary}; it has been removed. No indexed KKM guideline governs this diagnosis - "
            "see the references to consult.")
    diagnostic.citation_warning = f"{note} {diagnostic.citation_warning}".strip()
    log.warning("GROUNDING  %s", note)


def _check_population_sources(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """The governing guideline must be one written for this patient's age group.

    The context no longer contains wrong-population chunks, but the model can
    still NAME such a document from memory in `supporting_cpg`."""
    named = str(diagnostic.primary_diagnosis.supporting_cpg or "").strip()
    if not named:
        return
    pop = population.title_population(named)
    wrong = (pop == population.PAEDIATRIC and req.age >= population.ADULT_MIN_AGE) or (
        pop == population.ADULT and req.age < population.CHILD_MAX_AGE)
    if not wrong:
        # Right age group - now it must also be about the right condition.
        _check_governing_topic(diagnostic)
        return
    diagnostic.primary_diagnosis.supporting_cpg = ""
    note = (f"The model named '{named}' as the governing guideline, which is written for a different "
            f"age group than this patient ({population.label(req.age)}); it has been removed.")
    diagnostic.citation_warning = f"{note} {diagnostic.citation_warning}".strip()
    log.warning("GROUNDING  %s", note)


# ------------------------------------------------- citation support
_SUPPORT_STOP = frozenset("""
administer administered provide ensure monitor monitoring assess assessment check start
started consider patient patients immediately continue maintain obtain perform review
repeat urgent urgently routine should their within every other which where there these
those about after before during using given giving based level closely clinical signs
symptoms severe acute management treatment treat treated require required requires
recommended recommend guideline guidelines appropriate prevent risk potential possible
intravenous""".split())


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in _SUPPORT_STOP}


# ---------------------------------------------------- A4.2 support scoring
# "Does this passage support this item?" asked at SENTENCE level with rare
# terms weighing more. Calibrated 2026-09-30 on runs 6-7: the old rule (two
# shared 5-letter stems anywhere on the page) passed "Administer IV fluids" on
# a pain page and matched "compartment" to "compared". Now:
#   - terms: words of 5+ letters cut to 7 characters ("compart" != "compare"),
#     plus capitalised abbreviations in the item (ECG, AKI, CK);
#   - weight: inverse document frequency over the whole index;
#   - score: the share of the item's term weight found in one sentence, or two
#     adjacent sentences, of the passage. Wrong pairs scored <= 0.20, right
#     ones >= 0.40; CITATION_SUPPORT_MIN sits between.
_SUPPORT_ABBR_STOP = frozenset("iv im po sc od bd tds qid prn stat the and for mts cpg moh kkm etd".split())
_IDF: dict[str, float] = {}
_IDF_DEFAULT = 8.0  # an unseen term is rare


def _support_terms(text: str, item: bool = False) -> set[str]:
    t = text or ""
    out = {w[:7] for w in re.findall(r"[a-z]{5,}", t.lower()) if w not in _SUPPORT_STOP}
    abbr = (re.findall(r"\b[A-Z][A-Z0-9+]{1,4}\b", t) if item
            else re.findall(r"\b[a-zA-Z][a-zA-Z0-9+]{1,4}\b", t))
    out |= {a.lower() for a in abbr if a.lower() not in _SUPPORT_ABBR_STOP}
    return out


def set_support_idf(docs: list[str]) -> None:
    """Term weights from the whole index (called once by the engine)."""
    import math  # noqa: PLC0415
    df: dict[str, int] = {}
    for d in docs:
        for t in _support_terms(d):
            df[t] = df.get(t, 0) + 1
    n = max(len(docs), 1)
    _IDF.clear()
    _IDF.update({t: math.log(n / (1 + c)) for t, c in df.items()})


def _support(item: str, passage: str) -> tuple[float, set[str]]:
    """(share of the item's term weight in the best sentence window, shared terms)."""
    mine = _support_terms(item, item=True)
    if not mine:
        return 0.0, set()
    total = sum(_IDF.get(t, _IDF_DEFAULT) for t in mine) or 1.0
    _, sents = distill.sentences(passage)
    windows = sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    best, best_shared = 0.0, set()
    for w in windows or [passage or ""]:
        shared = mine & _support_terms(w)
        score = sum(_IDF.get(t, _IDF_DEFAULT) for t in shared) / total
        if score > best:
            best, best_shared = score, shared
    return best, best_shared


def _supported(item: str, passage: str) -> bool:
    mine = _support_terms(item, item=True)
    if not mine:
        return True  # nothing to judge by; other checks own it
    score, shared = _support(item, passage)
    # Two terms shared, or all of a one-term item: "metabolic syndrome" must
    # not support "compartment syndrome".
    return score >= config.CITATION_SUPPORT_MIN and len(shared) >= min(2, len(mine))


def _red_flag_supported(flag, passage: str) -> bool:
    # A long "why" dilutes the share; the flag's own name is judged too.
    return _supported(str(flag.flag or ""), passage) or _supported(f"{flag.flag} {flag.why_it_matters}", passage)


def _align_citations(
    diagnostic: DiagnosticSchema, chunks: list[Retrieved], model_saw: int | None = None,
) -> int:
    """A4.2: move a citation to the passage that supports the item.

    For every action, red flag and drug whose cited [S#] does not support it,
    the given passage that supports it best takes its place, and the move is
    named. Nothing is invented: an item without a citation is left alone, and a
    passage outside what the model was given is never used.

    Nothing is ever REMOVED here. Run 8 (2026-09-30) deleted the second pass's
    "Administer IV isotonic fluid" and "12-lead ECG" because their cited page
    (MSIC p19, a shock algorithm) held the terms in separate sentences - and a
    rhabdomyolysis plan without IV fluids is the unsafe direction. An item no
    passage supports keeps its place; _check_citation_support strips the
    citation and names it as a CITATION GAP.

    Returns the number of moves. Runs before _check_citation_support."""
    by_sid = {f"S{i}": c for i, c in enumerate(chunks, start=1)}
    not_guidance = {config.DOC_TYPE_TRIAGE, *config.ADJUNCT_DOC_TYPES}
    moves: list[str] = []

    # Only passages the model was shown: a page A4.3 appended afterwards was
    # never in front of it, so it cannot be what the model meant.
    seen = set(list(by_sid)[:model_saw]) if model_saw is not None else set(by_sid)

    def candidates(kind: str):
        for sid, c in by_sid.items():
            if sid not in seen:
                continue
            dt = c.metadata.get("doc_type")
            if kind != "red_flag" and dt in not_guidance:
                continue
            if kind != "drug" and dt == config.DOC_TYPE_DRUG:
                continue
            yield sid, c

    def ok(kind: str, obj, c: Retrieved) -> bool:
        if kind == "drug":
            aliases = _drug_aliases(str(obj.drug_name or ""))
            return bool(aliases) and bool(_alias_pattern(aliases).search(c.text or ""))
        if kind == "red_flag":
            return _red_flag_supported(obj, c.text)
        return _supported(obj.action, c.text)

    def score(kind: str, obj, c: Retrieved) -> float:
        if kind == "drug":
            # A guideline sentence with the dose beats a formulary row.
            return 1.0 if c.metadata.get("doc_type") != config.DOC_TYPE_DRUG else 0.5
        text = f"{obj.flag} {obj.why_it_matters}" if kind == "red_flag" else obj.action
        return _support(text, c.text)[0]

    def align(obj, kind: str, where: str) -> bool:
        """False when nothing given supports the item (it is left as it is)."""
        ids = _SID.findall(str(obj.source_id or ""))
        if not ids:
            return True
        good = [sid for sid in ids if sid in by_sid and ok(kind, obj, by_sid[sid])]
        if good:
            if len(good) < len(ids):
                obj.source_id = ", ".join(f"[{g}]" for g in good)
            return True
        best = max(((score(kind, obj, c), sid) for sid, c in candidates(kind) if ok(kind, obj, c)),
                   default=None)
        if best is None:
            return False
        old = ", ".join(f"[{i}]" for i in ids)
        obj.source_id = f"[{best[1]}]"
        c = by_sid[best[1]]
        moves.append(f"{where}: {old} -> [{best[1]}] ({c.metadata.get('cpg_title', '?')} "
                     f"p{c.metadata.get('page_number', '?')})")
        return True

    for a in diagnostic.immediate_actions:
        align(a, "action", f"action {a.sequence}")
    for d in diagnostic.drug_recommendations:
        align(d, "drug", f"drug {d.drug_name}")
    for i, f in enumerate(diagnostic.red_flags, start=1):
        align(f, "red_flag", f"red flag {i}")

    if moves:
        diagnostic.citation_alignment_note = (
            "Citation moved to the passage that supports the item - " + "; ".join(moves[:6]) + ".")
        log.warning("GROUNDING  %s", diagnostic.citation_alignment_note)
    return len(moves)


def _check_citation_support(
    diagnostic: DiagnosticSchema, chunks: list[Retrieved]
) -> tuple[list[str], set[str]]:
    """Every [S#] must actually support the claim it is attached to.

    Existing checks prove a citation EXISTS; this asks whether it SUPPORTS.
    Measured 2026-09-30: "Administer intravenous fluids" and "Monitor urine
    output and renal function" both cited MTS 2022 page 7 - the adult vital-signs
    table, which says nothing about treatment.

      - a triage table, an HTA report or a flow policy never supports an action
        or a drug (they set levels and pathways, not treatment);
      - a drug's cited passage must name the drug;
      - an action or red flag's cited passage must share at least one clinical
        term with it.

    An unsupported id is REMOVED from that item, so the derived citation table
    stops listing it as support, and the removal is named. Returns (notes, the
    items left with no citation at all) so the uncited count is not doubled."""
    by_sid = {f"S{i}": c for i, c in enumerate(chunks, start=1)}
    notes: list[str] = []
    stripped: set[str] = set()
    not_guidance = {config.DOC_TYPE_TRIAGE, *config.ADJUNCT_DOC_TYPES}

    def judge(obj, where: str, text: str, kind: str, drug: str = "") -> None:
        ids = _SID.findall(str(obj.source_id or ""))
        if not ids:
            return
        keep = []
        for sid in ids:
            c = by_sid.get(sid)
            if c is None:
                keep.append(sid)  # a non-existent id is _check_citations' finding
                continue
            dt = c.metadata.get("doc_type")
            reason = ""
            if kind in ("action", "drug") and dt in not_guidance:
                reason = ("a triage table sets the level, not the treatment" if dt == config.DOC_TYPE_TRIAGE
                          else "an HTA report or flow policy is not clinical guidance")
            elif kind == "drug":
                aliases = _drug_aliases(drug)
                if aliases and not _alias_pattern(aliases).search(c.text or ""):
                    reason = "the passage does not name this drug"
            else:
                # A4.2 sentence-level, rarity-weighted support (_supported).
                # The page-level "two shared stems" it replaces let
                # "Administer intravenous fluids" stand on a pain page.
                judged = (_red_flag_supported(obj, c.text) if kind == "red_flag"
                          else _supported(text, c.text))
                if not judged:
                    reason = "the passage does not discuss it"
            if reason:
                notes.append(f"{where} cited [{sid}] ({c.metadata.get('cpg_title', '?')}, "
                             f"p{c.metadata.get('page_number', '?')}) but {reason}")
            else:
                keep.append(sid)
        if not keep:
            stripped.add(where)
        obj.source_id = ", ".join(f"[{k}]" for k in keep)

    # An item added from source cites the chunk its verbatim quote came from;
    # judging its short label against that chunk stripped the citation off
    # "consider transvenous pacing" on 2026-10-01.
    for a in diagnostic.immediate_actions:
        if a.origin != "source":
            judge(a, f"action {a.sequence}", a.action, "action")
    for d in diagnostic.drug_recommendations:
        judge(d, f"drug {d.drug_name}", d.drug_name, "drug", str(d.drug_name or ""))
    for i, f in enumerate(diagnostic.red_flags, start=1):
        if f.origin != "source":
            judge(f, f"red flag {i}", f"{f.flag} {f.why_it_matters}", "red_flag")
    if notes:
        notes = ["unsupported citation removed - " + "; ".join(notes[:6])]
    return notes, stripped


# ------------------------------------------------- vitals, provisional triage, onset
_VITAL_PARAM = [
    (re.compile(r"blood pressure|\bbp\b|systolic|diastolic", re.I), "BP"),
    (re.compile(r"heart rate|pulse|\bhr\b", re.I), "HR"),
    (re.compile(r"resp|\brr\b", re.I), "RR"),
    (re.compile(r"spo2|sp02|oxygen|saturation", re.I), "SpO2"),
    (re.compile(r"temp", re.I), "Temp"),
    (re.compile(r"gcs|glasgow|conscious", re.I), "GCS"),
    (re.compile(r"pain", re.I), "Pain"),
    (re.compile(r"glucose|cbg|sugar", re.I), "Glucose"),
]


def _derive_vitals_levels(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """`mts_level_triggered` comes from the MTS table, never from the model.

    Measured 2026-09-30: the model printed "GCS 15 -> MTS level triggered 5".
    GCS 15 triggers nothing; page 7 prints no Level 5 GCS cell. A value the
    server can derive is never asked for."""
    findings = mts_table.evaluate(req.age, req.vitals, _intake_text(req), req.observations())
    best: dict[str, int] = {}
    for f in findings:
        best[f.parameter] = min(best.get(f.parameter, 9), f.level)
    for row in diagnostic.vitals_interpretation:
        param = next((p for rx, p in _VITAL_PARAM if rx.search(row.parameter or "")), None)
        row.mts_level_triggered = str(best[param]) if param and param in best else ""


_VITAL_ROWS = (
    ("BP", "Blood pressure"), ("HR", "Heart rate"), ("RR", "Respiratory rate"), ("Temp", "Temperature"),
    ("SpO2", "SpO2"), ("GCS", "GCS"), ("Glucose", "Capillary blood glucose"), ("Pain", "Pain score"),
)


def _vitals_rows(req: TriageRequest) -> list[VitalInterpretation]:
    """A4.4: one row per SUPPLIED vital sign, written from the MTS 2022 table -
    the model no longer spends output tokens on it. A value that meets no cell
    says so; it is never called normal."""
    v = req.vitals
    values = {
        "BP": (f"{v.systolic_bp:g}/{v.diastolic_bp:g} mmHg" if v.systolic_bp is not None and v.diastolic_bp is not None
               else f"{v.systolic_bp:g} mmHg systolic" if v.systolic_bp is not None else None),
        "HR": f"{v.heart_rate:g} bpm" if v.heart_rate is not None else None,
        "RR": f"{v.respiratory_rate:g} /min" if v.respiratory_rate is not None else None,
        "Temp": f"{v.temperature:g} °C" if v.temperature is not None else None,
        "SpO2": f"{v.spo2:g}%" if v.spo2 is not None else None,
        "GCS": f"{v.gcs}/15" if v.gcs is not None else None,
        "Glucose": f"{v.capillary_blood_glucose:g} mmol/L" if v.capillary_blood_glucose is not None else None,
        "Pain": f"{v.pain_score}/10" if v.pain_score is not None else None,
    }
    best: dict[str, mts_table.Finding] = {}
    for f in mts_table.evaluate(req.age, v, _intake_text(req), req.observations()):
        if f.parameter not in best or f.level < best[f.parameter].level:
            best[f.parameter] = f
    rows = []
    for key, label in _VITAL_ROWS:
        if values.get(key) is None:
            continue
        f = best.get(key)
        if f is not None and f.level <= 4:
            text = f'MTS 2022 p{f.page}, Level {f.level}: "{f.verbatim}"'
            level = str(f.level)
        else:
            text = "Meets no MTS 2022 criterion above Level 5 on its own"
            level = ""
        rows.append(VitalInterpretation(parameter=label, value=values[key], interpretation=text,
                                        mts_level_triggered=level))
    return rows


_CORE_VITALS = [("systolic_bp", "BP", "BP"), ("heart_rate", "HR", "HR"), ("respiratory_rate", "RR", "RR"),
                ("spo2", "SpO2", "SpO2"), ("temperature", "temperature", "Temp")]


def _flag_provisional_triage(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """A level set without the vital signs can only be a floor.

    MTS 2022 p7: the Secondary Triage Officer "Measures vital signs (BP, HR,
    RR, SpO2, Temp, GCS, Pain Score)". The rhabdomyolysis baseline was triaged
    on GCS and pain alone and the report did not say so."""
    v = req.vitals
    missing = [(label, param) for field, label, param in _CORE_VITALS if getattr(v, field) is None]
    # Page 7 also has the Secondary Triage Officer perform "initial tests for
    # ECG, and Glucose estimation". An ECG is how hyperkalaemia shows first.
    tests = [name for name, absent in (("ECG", not req.ecg_findings),
                                       ("capillary glucose", v.capillary_blood_glucose is None)) if absent]
    if not missing and not tests:
        return
    cells = []
    if req.age >= config.PAEDIATRIC_AGE_YEARS:
        wanted = {p for _, p in missing}
        # The "(ECG row)" tag is the audit record's; a clinician reads the cell.
        cells = [re.sub(r"\s*\([^)]*row\)", "", r.verbatim) for r in mts_table.ADULT_RULES
                 if r.parameter in wanted and r.level <= 2 and r.page == 7]
    escalate = (f" Re-triage at once to Level 2 if: {'; '.join(cells)} (MTS 2022 p7)." if cells else
                " Re-triage against the paediatric vital-sign bands (MTS 2022 p14) once measured.")
    tests_note = (f" No {' or '.join(tests)} recorded - MTS 2022 p7 lists both as initial tests "
                  "at secondary triage." if tests else "")
    diagnostic.triage_provisional = True
    if not missing:
        diagnostic.triage_provisional_note = f"PROVISIONAL -{tests_note[:-1]}; the level can still rise."
        return
    diagnostic.triage_provisional_note = (
        f"PROVISIONAL - not measured: {', '.join(l for l, _ in missing)}. MTS secondary triage measures "
        f"all vital signs (MTS 2022 p7); this level rests on what was recorded and can only rise."
        + escalate + tests_note
    )


_NUM = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a|an|few|several|a few)"
_UNIT = r"(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)"
_ONSET_RE = re.compile(
    rf"\b(?:for|over|since|in the (?:last|past))\s+(?:the\s+)?(?:last\s+|past\s+)?({_NUM}\s+{_UNIT})\b"
    rf"|\b({_NUM}\s+{_UNIT}\s+ago)\b"
    r"|\bsince\s+(this morning|this afternoon|this evening|last night|yesterday)\b",
    re.IGNORECASE,
)


def _derive_onset(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Read the onset from the complaint when the onset field was left blank.
    The rhabdomyolysis report printed "Onset: -" under a complaint that says
    "for two days"."""
    if (req.onset or "").strip():
        return
    for text in (req.complaint, req.history):
        m = _ONSET_RE.search(text or "")
        if m:
            found = next(g for g in m.groups() if g)
            diagnostic.onset_derived = f"{found} (read from the {'complaint' if text is req.complaint else 'history'})"
            return


# ------------------------------------------------- external references
def _attach_references(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Verified links for what the index cannot answer (references.py).

    Chosen by code from the registry - the model never writes a URL - and only
    for a real diagnosis. The MaHTAS CPG list is added only when no indexed
    guideline governs the diagnosis."""
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if primary.lower() in _NO_DIAGNOSIS:
        return
    need = not str(diagnostic.primary_diagnosis.supporting_cpg or "").strip()
    diagnostic.external_references = [
        ExternalReference(title=e["title"], publisher=e.get("publisher", ""), url=e["url"],
                          reason=e.get("reason", ""), checked=e.get("checked"))
        for e in references.select(primary, req.age, need)
    ]


_ACUTE_CUE = re.compile(r"\b(?:loading|stat|immediate\w*|initial\w*|emergen\w*|bolus|iv|"
                        r"intravenous\w*|nebuli[sz]\w*|first[- ]line|resuscitat\w*|urgent\w*)\b", re.I)
_NOT_SYSTEMIC = re.compile(r"\b(?:eye|ear|nasal|topical|cream|ointment|gel|lotion|shampoo|mouth ?wash|"
                           r"lozenge|gargle|paint|paste|enema|pessary|vaginal|dialys\w*|irrigation|patch)\b", re.I)
_CHILD_FORM = re.compile(r"\b(?:syrup|suspension|elixir|drops|paediatric|granules|dispersible)\b", re.I)


_MODIFIED_RELEASE = re.compile(
    r"prolonged|sustained|extended|modified|controlled|slow[- ]release|\b(?:SR|XL|XR|MR|CR|ER|LA)\b", re.I)


_ANALGESICS = frozenset("paracetamol acetaminophen morphine fentanyl tramadol oxycodone pethidine "
                        "ibuprofen diclofenac naproxen mefenamic etoricoxib celecoxib ketorolac".split()
                        + ["glyceryl trinitrate", "gtn", "nitroglycerin"])
_INJECTABLE = re.compile(r"injection|infusion|\bamp(?:oule)?s?\b|for (?:iv|im)\b|solution for (?:injection|infusion)", re.I)
_ORAL_FORM = re.compile(r"tablet|capsule|syrup|suspension|oral|elixir|mixture|granule", re.I)


def _route_cue(sentence: str) -> str:
    """The route a treatment sentence states: "iv", "im", "oral" or ""."""
    if re.search(r"\bIM\b|intramuscular", sentence, re.I):
        return "im"
    if re.search(r"\bIV\b|intravenous|infusion|\bbolus", sentence, re.I):
        return "iv"
    if re.search(r"\boral(?:ly)?\b|\bPO\b|by mouth|tablet", sentence, re.I):
        return "oral"
    return ""


def _fitting_formulations(entries: list, child: bool, route: str = "") -> list:
    """Formulations of one drug a clinician would reach for with THIS patient:
    systemic ones only, and syrups first for a child, last (never) for an adult."""
    systemic = [e for e in entries if not _NOT_SYSTEMIC.search(e.drug_name)]
    # Plain 0.9% is what "saline" / "sodium chloride" means at the bedside, and
    # an emergency is treated with immediate-release forms: a prolonged-release
    # morphine tablet was offered for acute pain on 2026-09-30.
    # F3: the guideline's route first - rapid tranquillisation is "IM
    # haloperidol plus lorazepam" (Schizophrenia CPG p32), and the tablets were
    # offered (2026-09-30).
    def route_miss(e) -> int:
        if route in ("iv", "im"):
            return 0 if _INJECTABLE.search(e.drug_name) else 1
        if route == "oral":
            return 0 if _ORAL_FORM.search(e.drug_name) else 1
        return 0
    systemic.sort(key=lambda e: (route_miss(e), 0 if "0.9%" in e.drug_name else 1,
                                 1 if _MODIFIED_RELEASE.search(e.drug_name) else 0,
                                 1 if re.search(r"supposit|pessar|enema", e.drug_name, re.I) else 0))
    if child:
        return sorted(systemic, key=lambda e: 0 if _CHILD_FORM.search(e.drug_name) else 1)
    return [e for e in systemic if not _CHILD_FORM.search(e.drug_name)] or systemic


# ------------------------------------------------- attention layer A2
@dataclass
class Hypotheses:
    """Working differential used to steer retrieval (not a diagnosis)."""
    likely: list[str]
    exclude: list[str]
    source: str  # "model+rules" | "rules"
    # What the MODEL itself proposed (likely + exclude), before the rule
    # labels were merged in - see _model_hypotheses.
    proposed: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        return list(dict.fromkeys(self.likely + self.exclude))

    def describe(self) -> str:
        bits = [f"likely: {', '.join(self.likely) or '-'}"]
        if self.exclude:
            bits.append(f"to exclude: {', '.join(self.exclude)}")
        return f"Working hypotheses ({self.source}) - " + "; ".join(bits)


def _no_refs(text: str) -> str:
    """Drop "(CHAMP 2020)" / "(MOH 2016 s6.1)" from a reason shown to the model:
    it copied one into a citation field as "[CHAMP 2020]" on 2026-09-30."""
    return re.sub(r"\s*\((?=[^)]*\b(?:19|20)\d\d\b)[^)]*\)", "", text or "").strip()


def _rule_hypotheses(req: TriageRequest) -> list[str]:
    """Conditions the red-flag rules name - a floor under the model's list, so
    a hypothesis the model omits can never lose its guideline."""
    return [flag.label for flag, _ in red_flags.match(focus.strip_negated(_intake_text(req)))]


def _parse_hypotheses(raw: str) -> tuple[list[str], list[str]]:
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        return [], []

    def clean(v) -> list[str]:
        out = []
        for x in (v if isinstance(v, list) else []):
            t = re.sub(r"\s+", " ", str(x)).strip(" .")
            if 2 < len(t) <= 80 and not t.startswith("<"):
                out.append(t)
        return out

    return clean(data.get("hypotheses"))[:3], clean(data.get("must_exclude"))[:2]


# Documents that serve many presentations and name no single condition.
_GENERAL_TITLE_MARKS = ("pain management in emergency", "icu management protocols",
                        "paediatric protocols", "malaysian triage scale",
                        # MOH SPG for AMOs in EMTS 2023: 20 chapters of ED approach.
                        "standard practice guidelines for assistant medical officer")
_INFECTION_SIGNS = re.compile(
    r"fever|febrile|infect|sepsis|septic|\bpus\b|purulent|cellulitis|abscess|pneumonia|"
    r"\buti\b|urinary tract|meningitis|antibiotic|antimicrobial|pharyngitis|tonsillitis", re.I)


def _strong(texts: list[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= {c for c in indications.concepts(t)
                if " " in c or (len(c) >= 4 and c not in _TITLE_AMBIGUOUS)}
    return out


# Checklist elements that name a treatment or a disposition - what the first
# pass must find passages FOR, not just mention. Analgesia is left out: the
# complaint search already reaches pain pages, and a dedicated one pulled
# opioid rows into a rhabdomyolysis prompt.
_PLAN_ELEMENT = re.compile(
    r"\b(?:IV|fluid|bolus|resuscitat\w*|admi\w*|cool\w*|oxygen|antibiot\w*|nebuli\w*|"
    r"insulin|antiplatelet|aspirin|anticoag\w*|heparin|transfus\w*|reperfusion|thromboly\w*|"
    r"steroid\w*|bronchodilat\w*|antivenom|decontaminat\w*)\b", re.I)


def _checklist_queries(req: TriageRequest, hyp: Hypotheses, limit: int = 3) -> list[str]:
    """Searches for what the completeness checklist says a plan for this
    presentation must contain (treatment and disposition elements only).

    Measured 2026-09-30: with CHAMP 2025 indexed, the rhabdomyolysis prompt
    carried its ALPE and lab pages but not the fluid-rate page (p24) - no
    first-pass query asked about fluids - so fluids waited for a second model
    call. The checklist already names them."""
    out: list[str] = []
    label, gaps = completeness.check(" ".join([req.complaint or "", req.history or ""] + hyp.all()), "")
    if label:
        out += [f"{label} {g.element}" for g in gaps if _PLAN_ELEMENT.search(g.element)][:limit]
    # F2: pain always gets an analgesia search, in the Pain Management
    # document's own bands (1-3 mild, 4-6 moderate, 7-10 severe). Run 10 gave
    # no analgesic for pain 7/10: six CHAMP passages had crowded the pain page
    # out. LAST, after the condition's own searches: first, its page ranked
    # above the HDP manual and opioid rows displaced MgSO4 for eclampsia.
    pain = req.vitals.pain_score
    if pain is not None and pain >= 4:
        band = "severe" if pain >= 7 else "moderate"
        out.append(f"{band} pain analgesia in the emergency department paracetamol opioid")
    return out


def _drop_bibliography(chunks: list[Retrieved]) -> tuple[list[Retrieved], list[str]]:
    """Reference-list pages out (distill.is_bibliography). A red-flag document
    whose only passage here was its reference list loses nothing: the list
    supports no recommendation for this patient."""
    kept, gone = [], []
    for c in chunks:
        if c.metadata.get("doc_type") in config.CLINICAL_DOC_TYPES and distill.is_bibliography(c.text):
            gone.append(f"{c.metadata.get('cpg_title', '?')} p{c.metadata.get('page_number', '?')}")
        else:
            kept.append(c)
    notes = ([f"Left out {len(gone)} reference-list page(s): " + "; ".join(dict.fromkeys(gone))]
             if gone else [])
    return kept, notes


def _model_hypotheses(req: TriageRequest, hyp: Hypotheses) -> set[str]:
    """Concepts of the hypotheses the MODEL proposed - the rule labels merged
    into `hyp` left out. A rule vouches for its own documents through
    _primary_forced; its label must not vouch again for its secondary ones
    ("obstructive airway disease" matched the COPD title for a known asthmatic)."""
    # A rule label the model ALSO proposed is the model's hypothesis too:
    # "Rhabdomyolysis" is both, and stripping it capped CHAMP 2025 - the only
    # guideline written for rhabdomyolysis - at two passages (2026-09-30).
    proposed = {h.lower() for h in hyp.proposed}
    rules = {r.lower() for r in _rule_hypotheses(req)} - proposed
    return _strong([h for h in hyp.all() if h.lower() not in rules])


def _primary_forced(req: TriageRequest) -> set[str]:
    """The FIRST title of every red-flag rule the intake fires - the rule's own
    guideline. The rest of a rule's titles are there for a cause or a
    complication (Dyslipidaemia for statin myopathy under rhabdomyolysis, COPD
    beside asthma under wheeze): read, but not prescribed from."""
    return {flag.titles[0] for flag, _ in red_flags.match(focus.strip_negated(_intake_text(req)))
            if flag.titles}


def _treatment_titles(chunks: list[Retrieved], req: TriageRequest, hyp: Hypotheses) -> set[str]:
    """Documents formulary-by-name may take drugs from: those about a working
    hypothesis, general-purpose ones, and each fired rule's own guideline.
    Measured 2026-09-30: without this, a rhabdomyolysis prompt carried
    ezetimibe and simvastatin (Dyslipidaemia), an asthma prompt heparin (COPD),
    and a chest-pain prompt perindopril/indapamide (Hypertension, background)."""
    hyp_c = _model_hypotheses(req, hyp)
    primary = _primary_forced(req)
    out = set()
    for c in chunks:
        title = str(c.metadata.get("cpg_title", ""))
        low = title.lower()
        if (title in primary or indications.mentions(hyp_c, title)
                or c.metadata.get("doc_type") == config.DOC_TYPE_PAEDS
                or any(g in low for g in _GENERAL_TITLE_MARKS)
                or ("antimicrobial" in low and _INFECTION_SIGNS.search(
                    " ".join([req.complaint or "", req.history or ""] + hyp.all())))):
            out.add(title)
    return out


def _topic_gate(
    chunks: list[Retrieved], req: TriageRequest, hyp: Hypotheses, forced: set[str]
) -> tuple[list[Retrieved], list[str]]:
    """Keep a clinical document only when this patient gives a reason to read it.

    Measured 2026-09-30: no relevance SCORE can do this. Both the current
    reranker (bge) and a medical one (MedCPT) rank the snakebite guideline high
    for exertional rhabdomyolysis, because its text says "myalgia ... dark
    coloured urine" - the scorers reward shared symptoms and cannot know there
    was no bite. What separates the two is the CAUSE, so the gate asks about
    the cause. A document stays when:

      - a red-flag rule forced it in (the rule is the reason) - in full when
        it is the rule's own guideline or a hypothesis, otherwise at most
        FORCED_SECONDARY_PASSAGES passages;
      - it serves many presentations (pain management, ICU protocols, the
        paediatric protocols, MTS) - or it is the NAG and there are signs of
        infection;
      - its title names a working hypothesis;
      - its title names something in the intake itself (a stated condition or
        comorbidity) - then ONE passage, as background.

    Everything else is left out, and named. Non-clinical types (the formulary,
    the triage table) pass through untouched."""
    hyp_c = _strong(hyp.all())
    intake_c = _strong([req.complaint or "", req.history or ""]
                       + [c.value.replace("_", " ") for c in req.comorbidities])
    infection = bool(_INFECTION_SIGNS.search(" ".join(
        [req.complaint or "", req.history or ""] + hyp.all())))
    primary = _primary_forced(req)
    model_c = _model_hypotheses(req, hyp)
    kept: list[Retrieved] = []
    dropped: dict[str, int] = {}
    background: dict[str, int] = {}
    secondary: dict[str, int] = {}
    for c in chunks:
        meta = c.metadata
        title = str(meta.get("cpg_title", ""))
        low = title.lower()
        if meta.get("doc_type") not in config.CLINICAL_DOC_TYPES:
            kept.append(c)
            continue
        if title in forced:
            # A rule's secondary document (a cause, a complication) is read,
            # but it does not get to fill the context.
            if title in primary or indications.mentions(model_c, title):
                kept.append(c)
            elif secondary.get(title, 0) < config.FORCED_SECONDARY_PASSAGES:
                secondary[title] = secondary.get(title, 0) + 1
                kept.append(c)
            else:
                dropped[title] = dropped.get(title, 0) + 1
            continue
        if meta.get("doc_type") == config.DOC_TYPE_PAEDS or any(g in low for g in _GENERAL_TITLE_MARKS):
            kept.append(c)
            continue
        if "antimicrobial" in low:
            (kept.append(c) if infection else dropped.__setitem__(title, dropped.get(title, 0) + 1))
            continue
        if indications.mentions(hyp_c, title):
            kept.append(c)
            continue
        if indications.mentions(intake_c, title):
            if background.get(title, 0) < 1:
                background[title] = 1
                kept.append(c)
            else:
                dropped[title] = dropped.get(title, 0) + 1
            continue
        dropped[title] = dropped.get(title, 0) + 1
    notes = []
    if dropped:
        notes.append(
            "Topic gate: left out " + "; ".join(f"{t} ({n})" for t, n in dropped.items())
            + " - about conditions neither in the intake nor among the working hypotheses,"
              " or passages beyond the cap for a supporting document")
    return kept, notes


# --------------------------------------------- action quantities grounding
# 2026-10-02 (MedGemma 1.5 4B trial, rhabdo 55M 80 kg): the only immediate
# action read "IV 0.9% NaCl at 30-50 ml/kg/hr (or 1 ml/kg/hr) [S1]" - 2.4-4 L
# an hour, and 1 ml/kg/hr is a urine-output target, not a rate. Neither figure
# was in the cited page or any retrieved passage, and nothing caught it: the
# dose checks read drug_recommendations, and _check_dose_completeness only asks
# whether a number is PRESENT. A number in an action must now be written in a
# passage the model read (or the FUKKM entry of a drug it recommends); one that
# is not is taken out of the action and named in the safety block.

_ACTION_QTY_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:-|to|–)\s*\d+(?:\.\d+)?)?)\s*"
    r"(?P<unit>(?:ml|mls|millilitres?|l|litres?|liters?|mg|mcg|micrograms?|g|grams?|"
    r"units?|iu|mmol|meq)\b(?:\s*/\s*(?:kg|hr|hour|h|min|minute|day|24\s*h(?:ours?)?)\b)*)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_UNIT_BASE = {"mls": "ml", "millilitre": "ml", "millilitres": "ml", "liter": "l", "liters": "l",
              "litre": "l", "litres": "l", "microgram": "mcg", "micrograms": "mcg", "gram": "g",
              "grams": "g", "units": "unit", "iu": "unit"}


def _quantities(text: str) -> set[tuple[str, str, bool]]:
    """(number, unit, per-kg) for every quantity in `text`; a range gives one
    entry per end. Unit-aware, because a bare '1' is in every passage and
    '30 ml' is not '30 ml/kg'."""
    out: set[tuple[str, str, bool]] = set()
    for m in _ACTION_QTY_RE.finditer(text):
        unit = m.group("unit").lower()
        base = re.match(r"[a-z]+", unit).group(0)
        key = (_UNIT_BASE.get(base, base), "/kg" in re.sub(r"\s+", "", unit))
        out.update((str(float(n)), *key) for n in _NUM_RE.findall(m.group("num")))
    return out


def _ground_action_quantities(
    diagnostic: DiagnosticSchema, chunks: list[Retrieved]
) -> None:
    """Strip any dose, volume or rate in a model-written immediate action whose
    numbers appear in no retrieved passage and no FUKKM entry of a recommended
    drug. Quantities are checked as a whole: '30-50 ml/kg/hr' needs both 30 and
    50 in the same passage. Server-added actions (origin 'source') quote their
    passage verbatim and are skipped."""
    texts = [c.text for c in chunks]
    for drug in diagnostic.drug_recommendations:
        match = formulary.lookup(str(drug.drug_name or ""), str(drug.route or ""), "")
        if match:
            texts.extend(e.dosage for e in match.entries if e.dosage)
    known = [_quantities(t) for t in texts]
    removed: list[str] = []
    for a in diagnostic.immediate_actions:
        if a.origin == "source":
            continue
        text = str(a.action or "")
        out = text
        for m in _ACTION_QTY_RE.finditer(text):
            want = _quantities(m.group(0))
            if any(want <= k for k in known):
                continue
            out = out.replace(m.group(0), "[dose/rate removed]", 1)
            removed.append(f"action {a.sequence}: '{m.group(0)}'")
        if out != text:
            # "(or [dose/rate removed])" and similar leftovers read as noise.
            a.action = re.sub(r"\(\s*or\s*\[dose/rate removed\]\s*\)", "", out).strip()
    if removed:
        note = (
            "UNSOURCED QUANTITY REMOVED - written by the model but found in no retrieved "
            "KKM passage or FUKKM entry: " + "; ".join(removed)
            + ". Take the dose or rate from the cited guideline, not from this report."
        )
        prev = diagnostic.dose_completeness_warning
        diagnostic.dose_completeness_warning = f"{prev} {note}".strip() if prev else note
        log.warning("GROUNDING  %s", note)


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

    adult = req.age >= config.PAEDIATRIC_AGE_YEARS

    def dosed(text: str) -> bool:
        # F3: an adult fluid order is written as a volume or a rate ("2-6 L
        # bolus then 250-300 mL/hr", CHAMP 2025 p26); per-kg is the paediatric
        # form. Run 10 flagged a correct adult rate for having no mL/kg.
        return bool(_DOSE_PRESENT_RE.search(text) or (adult and _ADULT_VOLUME_RE.search(text)))

    for action in diagnostic.immediate_actions:
        text = str(action.action or "")
        if _DOSE_REQUIRED_RE.search(text) and not dosed(text):
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
        ) and not dosed(dose_field):
            missing.append(f"{drug.drug_name}: dose {dose_field[:40]!r} is "
                           + ("neither a rate nor weight-based" if adult else "not weight-based"))

    if missing:
        form = "a volume or rate (adult) or a per-kg dose" if adult else "a per-kg dose"
        note = (
            f"DOSE INCOMPLETE for a {req.weight_kg:g} kg patient - instructions given without "
            f"{form}: " + "; ".join(missing)
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


_NEGATED = re.compile(
    r"\b(?:avoid\w*|no|not|without|never|withh\w*|contraindicated|instead of|rather than|"
    r"do not (?:give|use|administer|prescribe))\b[^.;,]{0,60}",
    re.IGNORECASE,
)


def _drop_negated(text: str) -> str:
    """Remove clauses that say NOT to do something before matching rules.

    Measured 2026-09-30: paracetamol's own caution, "Avoid NSAIDs due to risk
    of kidney injury", matched the NSAID-in-renal-impairment rule and printed
    an ABSOLUTE contraindication against the correct drug. A guardrail that
    fires on the advice to avoid the harm is crying wolf."""
    return _NEGATED.sub(" ", text or "")


def _check_contraindications(
    diagnostic: DiagnosticSchema, req: TriageRequest
) -> None:
    """Test every recommendation against this patient's own picture.

    Actions, investigations AND drugs are all scanned: the GTN that prompted
    this check arrived as an immediate action, so inspecting drug entries alone
    would have missed it entirely.

    A CAUTION annotates. An ABSOLUTE contraindication takes the item out of
    the plan and lists it, with the reason, in the safety block - changed on
    2026-10-01, when an IV nitroglycerin infusion at SBP 86 stayed in the action
    list with its warning in a separate box, where a reader skimming the plan
    would act on it. Moving it keeps the model's failure visible without
    leaving the order where it can be followed."""
    items: list[tuple[str, str]] = []
    for a in diagnostic.immediate_actions:
        items.append((f"action {a.sequence}", _drop_negated(str(a.action or ""))))
    for x in diagnostic.investigations:
        items.append((f"investigation: {x.test}", _drop_negated(f"{x.test} {x.rationale}")))
    for d in diagnostic.drug_recommendations:
        items.append((f"drug: {d.drug_name}", _drop_negated(
                      " ".join(str(getattr(d, f, "") or "") for f in
                               ("drug_name", "indication", "adult_dose",
                                "paediatric_dose", "route", "cautions")))))

    found = contraindications.check(_clinical_context(req, diagnostic), items)
    if not found:
        return

    absolute = [f for f in found if f.severity == "ABSOLUTE"]
    gone = {f.where for f in absolute}
    # One row per (rule, recommendation): nitroglycerin as action 6 AND as a
    # drug entry printed two identical rows on 2026-10-01.
    merged: dict[tuple[str, str], Contraindication] = {}
    for f in found:
        key = (f.rule, f.item.strip().lower())
        where = f.where.replace("drug: ", "drug list: ")
        if key in merged:
            if where not in merged[key].where:
                merged[key].where += f" and {where}"
            continue
        merged[key] = Contraindication(rule=f.rule, severity=f.severity, where=where,
                                       item=f.item, trigger=f.trigger, reason=f.reason)
    for c in merged.values():
        if c.severity == "ABSOLUTE":
            c.where = f"removed from the plan ({c.where})"
    diagnostic.contraindications = list(merged.values())
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
    removed = _remove_absolute(diagnostic, absolute)
    if removed:
        parts.append("REMOVED FROM THE PLAN (absolute contraindication): " + "; ".join(removed) + ".")
    note = " ".join(parts)
    diagnostic.contraindication_warning = note
    for f in found:
        log.warning("CONTRAINDICATION  [%s] %s in %s <- %r", f.severity, f.item, f.where, f.trigger)


def _remove_absolute(diagnostic: DiagnosticSchema, absolute: list) -> list[str]:
    """Take every drug and action an ABSOLUTE finding names out of the plan,
    into withheld_drugs with the reason. Investigations stay - a test is never
    the harm these rules describe."""
    if not absolute:
        return []
    why = {}
    for f in absolute:
        why.setdefault(f.where, f"{f.reason} (triggered by \"{f.trigger}\")")
    removed: list[str] = []
    keep_drugs = []
    for d in diagnostic.drug_recommendations:
        w = why.get(f"drug: {d.drug_name}")
        if w is None:
            keep_drugs.append(d)
            continue
        diagnostic.withheld_drugs.append(WithheldDrug(
            drug_name=str(d.drug_name), route=str(d.route or ""), stated_dose=str(d.adult_dose or d.paediatric_dose or ""),
            reason=f"ABSOLUTELY CONTRAINDICATED for this patient: {w}", basis="contraindications.py"))
        removed.append(f"drug {d.drug_name}")
    diagnostic.drug_recommendations = keep_drugs
    keep_actions = []
    for a in diagnostic.immediate_actions:
        w = why.get(f"action {a.sequence}")
        if w is None:
            keep_actions.append(a)
            continue
        diagnostic.withheld_drugs.append(WithheldDrug(
            drug_name=str(a.action)[:160], reason=f"ABSOLUTELY CONTRAINDICATED for this patient: {w}",
            basis="contraindications.py"))
        removed.append(f"action {a.sequence} ({str(a.action)[:60]})")
    diagnostic.immediate_actions = keep_actions
    return removed


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


def _quoted_disposition_floor(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """F1 (disposition_rules.py): raise the destination when the intake meets a
    source's stated admission criterion for the working diagnosis. Raise only."""
    primary = str(diagnostic.primary_diagnosis.condition or "")
    hit = disposition_rules.applicable(primary, focus.strip_negated(_intake_text(req)))
    if hit is None:
        _card_disposition_floor(diagnostic, req)
        return
    rule, finding = hit
    current = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if disposition_rules.RANK.get(current, 0) >= disposition_rules.RANK[rule.target]:
        return
    diagnostic.disposition = Disposition(rule.target)
    label = rule.target.replace("_", " ").lower().replace("icu hdu", "ICU/HDU")
    note = (f"[Disposition raised from {current.replace('_', ' ').lower()} to {label}: "
            f"{rule.source}{' - a non-KKM source' if rule.external else ''}: \"{rule.quote}\". "
            f"The intake meets it: '{finding}'.]")
    diagnostic.disposition_justification = f"{diagnostic.disposition_justification} {note}".strip()
    log.warning("GROUNDING  %s", note)


# 2026-10-02: KKM states no clinical admission criterion for many diagnoses
# (exertional rhabdomyolysis is the case that showed it; AskCPG reached the same
# gap). The national patient-flow guideline does state WHO decides and against
# WHAT - the hospital's own criteria, with the specialist. Quoted verbatim from
# the OCR text (tests/test_action_quantities.py checks both against it); it is a
# process statement, never a clinical threshold, so it never moves the level.
ADMISSION_PROCESS_QUOTES = (
    ("MOH Patient Flow Management Guideline 2022 p19",
     "Kes akan dinilai mengikut kriteria yang telah ditetapkan untuk kemasukan ke wad"),
    ("MOH Patient Flow Management Guideline 2022 p4",
     "Jururawat di BMU akan membuat penilaian dan akan menempatkan pesakit mengikut "
     "disiplin serta kritikaliti pesakit selepas berbincang dengan pakar"),
)


def _admission_process_note(diagnostic: DiagnosticSchema) -> None:
    """ED observation with no KKM admission criterion to test against: say so,
    and name the process KKM does set, instead of leaving the choice looking
    settled."""
    current = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if current != "ED_OBSERVATION":
        return
    quotes = "; ".join(f'{where}: "{q}"' for where, q in ADMISSION_PROCESS_QUOTES)
    note = ("[No KKM admission criterion found for this diagnosis. Admit-or-observe is decided "
            "against the hospital's own admission criteria, with the specialist - " + quotes + ".]")
    diagnostic.disposition_justification = f"{diagnostic.disposition_justification} {note}".strip()


def _card_disposition_floor(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """F1 for any condition with a card: raise to admission when the patient
    meets an admission criterion its KKM source states. Raise only; the note
    quotes the criterion and names the finding that met it."""
    hit = cards.admission(_cards_for(diagnostic, req), _patient_state(req), req.age, bool(req.pregnant))
    if hit is None:
        _admission_process_note(diagnostic)
        return
    item, target, finding = hit
    current = str(getattr(diagnostic.disposition, "value", diagnostic.disposition))
    if disposition_rules.RANK.get(current, 0) >= disposition_rules.RANK[target]:
        return
    diagnostic.disposition = Disposition(target)
    label = target.replace("_", " ").lower().replace("icu hdu", "ICU/HDU")
    note = (f"[Disposition raised from {current.replace('_', ' ').lower()} to {label}: {item.where}"
            f"{' - a non-KKM source cited by KKM' if item.external else ''}: \"{item.quote}\". "
            f"The patient meets it: '{finding}'.]")
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
def _answer_text(diagnostic: DiagnosticSchema) -> str:
    """Everything the model wrote that a completeness element can be found in."""
    return " \n".join(
        [diagnostic.triage_rationale or "",
         getattr(diagnostic.primary_diagnosis, "reasoning", "") or ""]
        + [f"{a.action} {a.timeframe}" for a in diagnostic.immediate_actions]
        + [f"{x.test} {x.rationale}" for x in diagnostic.investigations]
        + [" ".join(str(getattr(d, f, "") or "") for f in
                    ("drug_name", "indication", "adult_dose", "paediatric_dose",
                     "route", "cautions"))
           for d in diagnostic.drug_recommendations]
        + [f"{d.condition} {d.discriminating_feature}" for d in diagnostic.differential_diagnoses]
        + [f"{diagnostic.disposition.value} {diagnostic.disposition_justification} {diagnostic.referral_to}"]
    )


# ------------------------------------------------------- condition cards
def _patient_state(req: TriageRequest) -> str:
    """The facts a condition card's `when` / `unless` is tested against: the
    intake with its negated clauses removed, plus derived observations in
    words. Never the model's prose - a state the model asserted is not a
    finding, and "the patient is stable" written about a shocked patient is
    exactly the error a state-conditioned quote must not inherit."""
    bits = [focus.strip_negated(_intake_text(req))]
    v = req.vitals
    hypotensive = v.systolic_bp is not None and v.systolic_bp < _hypotension_threshold(req.age)
    if hypotensive:
        bits.append("hypotension hypotensive")
    if v.systolic_bp is not None and v.systolic_bp >= 180:
        bits.append("severe hypertension")
    adult = req.age >= 12
    if v.heart_rate is not None:
        if v.heart_rate < 50:
            bits.append("bradycardia")
        if adult and v.heart_rate > 100:
            bits.append("tachycardia")
    if adult and v.respiratory_rate is not None and v.respiratory_rate > 24:
        bits.append("tachypnoea")
    if v.spo2 is not None and v.spo2 < 94:
        bits.append("hypoxia hypoxaemia")
    if v.temperature is not None:
        if v.temperature >= 38:
            bits.append("fever pyrexia")
        if v.temperature >= 40:
            bits.append("hyperthermia")
        if v.temperature < 35:
            bits.append("hypothermia")
    if v.capillary_blood_glucose is not None:
        if v.capillary_blood_glucose < 4:
            bits.append("hypoglycaemia")
        if v.capillary_blood_glucose > 11:
            bits.append("hyperglycaemia")
    if v.gcs is not None and v.gcs < 15:
        bits.append("reduced consciousness altered consciousness")
        if v.gcs <= 8:
            bits.append("unconscious coma")
    poor_perfusion = req.perfusion is not None and req.perfusion.name in (
        "WEAK_PULSES", "COLD_OR_CYANOSED", "ABSENT_RADIAL")
    if poor_perfusion or (hypotensive and v.heart_rate is not None and v.heart_rate > 100):
        bits.append("shock haemodynamically unstable")
    if (v.pain_score or 0) >= 7:
        bits.append("severe pain")
    if req.egfr is not None and req.egfr < 60:
        bits.append("renal impairment" + (" severe renal impairment" if req.egfr < 30 else ""))
    if req.age < 18:
        bits.append("paediatric child")
    if req.pregnant:
        bits.append("pregnant pregnancy")
    return " \n".join(b for b in bits if b)


def _cards_for(diagnostic: DiagnosticSchema, req: TriageRequest) -> list:
    """The working diagnosis's condition card and its broader condition, or []."""
    if not config.CARDS:
        return []
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if primary.lower() in _NO_DIAGNOSIS:
        return []
    card = cards.lookup(primary, req.age, bool(req.pregnant))
    return cards.lineage(card, req.age, bool(req.pregnant)) if card else []


@dataclass
class _Gap:
    """A checklist element the answer left out - from completeness.py (the
    hand-written presentations) or from the diagnosis's condition card, in
    which case `item` carries its verbatim sentence."""
    presentation: str
    element: str
    why: str
    guideline: str
    item: "cards.Item | None" = None


def _pos(corpus: dict) -> dict:
    """chunk id -> row, built once if the corpus dict does not carry it."""
    if "pos" not in corpus:
        corpus["pos"] = {cid: i for i, cid in enumerate(corpus["ids"])}
    return corpus["pos"]


def _card_context(corpus: dict | None):
    """item -> the text around its quote in its chunk (for cards.checklist)."""
    if not corpus:
        return None

    def around(item) -> str:
        pos = _pos(corpus).get(item.chunk_id)
        if pos is None:
            return ""
        text = re.sub(r"\s+", " ", corpus["docs"][pos] or "")
        i = text.lower().find(item.quote[:40].lower())
        # The page's head as well as the neighbourhood: an algorithm names its
        # state once, in its title ("Bradycardia algorithm"), not beside each line.
        near = text[max(0, i - 500): i + len(item.quote) + 500] if i >= 0 else ""
        return text[:300] + " " + near
    return around


def _other_subtype(primary: str, text: str) -> bool:
    """`text` is about the exclusive sibling of an UNAMBIGUOUS primary subtype
    (a UA/NSTEMI risk score for a STEMI)."""
    for a, b, _ in _EXCLUSIVE:
        for first, second in ((a, b), (b, a)):
            if re.search(first, primary, re.I) and not re.search(second, primary, re.I):
                return bool(re.search(second, text, re.I))
    return False


def _gaps(req: TriageRequest, diagnostic: DiagnosticSchema, corpus: dict | None = None
          ) -> tuple[str, list[_Gap]]:
    """The hand-written presentation when one matches (each was tuned on a
    real failure and is tested against its page); otherwise the condition
    card, which exists for every condition the corpus gives management for."""
    label, hand = completeness.check(_clinical_context(req, diagnostic), _answer_text(diagnostic))
    out = [_Gap(g.presentation, g.element, g.why, g.guideline) for g in hand]
    cs = _cards_for(diagnostic, req)
    if not cs:
        return label, out
    name = cs[0].name
    # Both, not either: on 2026-10-01 the shocked-STEMI answer satisfied the
    # hand ACS checklist, so the card's anticoagulant, investigations and
    # monitoring were never checked. A card item already named by a hand gap
    # is not listed twice.
    named = " ".join(f"{g.element} {g.why}" for g in hand)
    primary = str(diagnostic.primary_diagnosis.condition or "")
    items = [i for i in cards.gaps(cs, _answer_text(diagnostic), _patient_state(req), req.age,
                                   bool(req.pregnant), context=_card_context(corpus))
             if not cards.present(i, named) and not _other_subtype(primary, f"{i.label} {i.quote}")]
    return label or name, out + [_Gap(name, i.label[:1].upper() + i.label[1:],
                       f"{cards.ELEMENT_LABEL[i.element].capitalize()} for {name}"
                       + (f" (applies because: {', '.join(i.when)})" if i.when else "")
                       + (" - a non-KKM source cited by KKM" if i.external else ""),
                       i.where, i) for i in items]


_MONITOR_WORD = re.compile(r"monitor\w*|hourly|urine output|input[- ]?output|serial|repeat", re.I)


def _kkm_covered(diagnostic: DiagnosticSchema, req: TriageRequest) -> set[str]:
    """Elements this report already answers from KKM text: items added from a
    verbatim source sentence, and the elements of a hand-written checklist
    (each of which names its KKM passage)."""
    out: set[str] = set()
    if any(f.origin == "source" for f in diagnostic.red_flags):
        out |= {"red_flag", "complication"}
    if any(x.origin == "source" for x in diagnostic.investigations):
        out.add("investigation")
    if any(a.origin == "source" for a in diagnostic.immediate_actions):
        out.add("treatment")
    if diagnostic.source_cautions:
        out.add("avoid")
    if any(_MONITOR_WORD.search(f"{x.test} {x.source_quote}") for x in diagnostic.investigations
           if x.origin == "source"):
        out.add("monitoring")
    label, _ = completeness.check(_clinical_context(req, diagnostic), _answer_text(diagnostic))
    pres = next((p for p in completeness.PRESENTATIONS if p.label == label), None)
    for el in (pres.elements if pres else ()):
        if RagEngine._INVESTIGATION_ELEMENT.search(el.name):
            out.add("investigation")
        elif _MONITOR_WORD.search(el.name):
            out.add("monitoring")
        elif re.search(r"\badmi", el.name, re.I):
            out.add("admission")
        else:
            out.add("treatment")
    return out


def _state_knowledge_gaps(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Plan A2: where KKM is silent, SAY so - with verified links - instead of
    letting the model fill the gap or importing a foreign guideline."""
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if not config.CARDS or primary.lower() in _NO_DIAGNOSIS:
        return
    cs = _cards_for(diagnostic, req)
    links = [ExternalReference(title=e["title"], publisher=e.get("publisher", ""), url=e["url"],
                               reason=e.get("reason", ""), checked=e.get("checked"))
             for e in references.select(primary, req.age, need_guideline=True, limit=3)]
    if not cs:
        if cards.load():
            diagnostic.knowledge_gaps.append(KnowledgeGap(
                element="guideline",
                statement=(f"No condition card was extracted for {primary} from the indexed KKM documents, "
                           "so this report's checklist for it is the general one. Check the governing "
                           "guideline directly; the references below may help."),
                references=links))
        return
    # Worded as what the automated extraction FOUND, never as what KKM says:
    # run of 2026-10-01 told an ACS reader "no indexed KKM document states red
    # flags" - the ACS CPG does; the extraction had missed them.
    docs = "; ".join(cs[0].documents[:2])
    covered = _kkm_covered(diagnostic, req)
    for el in cards.silent_elements(cs):
        # Never "none found" beside KKM quotes for the same thing: the
        # rhabdomyolysis report of 2026-10-01 said no red flags were found
        # while showing two red flags quoted from KKM documents.
        if el in covered:
            continue
        diagnostic.knowledge_gaps.append(KnowledgeGap(
            element=el,
            statement=(f"No {cards.ELEMENT_LABEL[el]} for {cs[0].name} were found by the automated "
                       f"extraction of the indexed KKM documents. Read the governing guideline ({docs}) "
                       "directly - this report does not supply them from another source."),
            references=links))


def _check_completeness(
    diagnostic: DiagnosticSchema, req: TriageRequest, corpus: dict | None = None
) -> None:
    """Name what the answer left out. Never GENERATES the gap - a checklist that
    invented the missing dose would be a second unreliable reasoner - but does
    QUOTE it: each gap carries the guideline's own sentence covering that
    element, verbatim with its page, when the indexed guideline has one."""
    label, gaps = _gaps(req, diagnostic, corpus)
    if not gaps:
        return
    diagnostic.completeness_gaps = []
    state = _patient_state(req)
    for g in gaps:
        if g.item is not None:
            quote, where = g.item.quote, g.item.where
        else:
            quote, where = _gap_quote(label, g, corpus, req.age, state) if corpus else ("", "")
        diagnostic.completeness_gaps.append(CompletenessGap(
            element=g.element, why=g.why, guideline=g.guideline, quote=quote, quote_source=where))
    note = (
        f"INCOMPLETE for {label}: the answer does not address "
        + "; ".join(g.element for g in gaps)
        + f". Read {gaps[0].guideline} before acting on this report."
    )
    diagnostic.completeness_warning = note
    log.warning("GROUNDING  %s", note)


_GAP_NAME_STOP = frozenset({"other", "considered", "assessment", "assessed", "with", "and",
                            "avoided", "withheld", "sought", "decision", "indication", "serial"})


# D2 (run 11, 2026-09-30): repeat labs were "covered" by "once CK is clearly
# downtrending ... not warranted" and compartment syndrome by an admission-
# criteria list. A sentence saying something is NOT needed never answers a
# required element; a recommendation outranks a list.
_NOT_NEEDED = re.compile(
    r"\bnot (?:be )?(?:warranted|required|necessary|needed|indicated|recommended|routinely|useful)\b|"
    r"\bno (?:need|role|benefit)\b|\bunnecessary\b", re.I)
_RECOMMENDS = re.compile(
    r"\b(?:should|must|recommended|monitor\w*|repeat\w*|every \d|hourly|assess\w*|check\w*|give|"
    r"administer\w*|start|initiate|refer|admit)\b", re.I)


_INSTRUCTION = re.compile(
    r"\b(?:give|giving|administer\w*|start|initiate|commence|consider|perform|repeat|monitor\w*|use|check|"
    r"send|obtain|refer|admit|treat|insert|apply|should|must|recommended|undergo|titrate)\b", re.I)


_FOREIGN_BODY = re.compile(
    r"American (?:College|Heart|Academy|Society)|European Society|\bNICE\b|\bACEP\b|\bAHA\b|\bESC\b|"
    r"Royal College|National Institute for Health", re.I)


def _listy(sentence: str) -> bool:
    return sentence.count(";") + sentence.count(" - ") >= 3 or bool(re.search(r"\bcriteria\b", sentence, re.I))


def _gap_quote(label: str, gap, corpus: dict, age: float, state: str = "") -> tuple[str, str]:
    quote, where, _ = _gap_quote_hit(label, gap, corpus, age, state)
    return quote, where


def _gap_quote_hit(label: str, gap, corpus: dict, age: float, state: str = "") -> tuple[str, str, int | None]:
    """The sentence in the named guideline that best covers a missing element.

    Found by the element's own evidence patterns (the same regexes that decided
    it was missing), within the documents whose significant title words the
    gap's guideline name shares. Population-filtered like the context itself.
    Returns ("", "") when no sentence qualifies - silence, never a guess."""
    el = completeness.element(label, gap.element)
    if el is None:
        return "", "", None
    # Each document the checklist names, matched on its own: "...; MOH
    # Snakebite 2017 s4.5 (serial CK)" pooled with two other titles never shared
    # two words with the Snakebite guideline's title (2026-10-01). Section refs
    # and notes in brackets are not title words.
    parts = [_title_words(re.sub(r"\([^)]*\)|\bs\d[\d.]*", " ", p)) for p in re.split(r";|\s-\s", gap.guideline)]
    parts = [p for p in parts if p]
    if not parts:
        return "", "", None
    patterns = [re.compile(p, re.I) for p in el.present]
    # Fix 2026-10-03: the CK row quoted Snakebite p99 "Serial blood results
    # every 4 - 6 hours" over s4.5.4 "Creatine kinase: ... Serial monitoring to
    # monitor trend". A sentence that names the test the element is about fits
    # it better than one about bloods in general.
    test_rx = [re.compile(rx, re.I) for rx, _ in _TEST_SYNONYMS if re.search(rx, gap.element, re.I)]
    name_words = {w for w in re.findall(r"[a-z]{4,}", gap.element.lower())} - _GAP_NAME_STOP
    best: tuple[tuple, str, str, int] | None = None
    for idx, (text, meta) in enumerate(zip(corpus["docs"], corpus["metas"])):
        if meta.get("doc_type") not in config.CLINICAL_DOC_TYPES:
            continue
        title_w = _title_words(str(meta.get("cpg_title", "")))
        if not any(len(p & title_w) >= min(2, len(p)) for p in parts):
            continue
        title_fit = max(len(p & title_w) for p in parts)
        if not population.allowed(meta, age):
            continue
        # distill.sentences, not a split on every newline: PDF line wraps cut
        # "6.1.7. Renal Function Test / Acute kidney injury due to ..." into
        # pieces too short to quote, so the renal-profile gap had no quote and
        # was never filled (2026-10-01). It re-joins wraps and keeps bullets apart.
        # A unit is cut at each numbered heading glued into it first: "... sea
        # snake bites. 4.5.4 Creatine kinase: For early detection of ..." kept
        # only its snakebite head, so the CK line was never a candidate
        # (2026-10-03). A piece that opens a heading starts a new section.
        pieces: list[tuple[str, bool]] = []
        for u in distill.sentences(text or "")[1]:
            for j, piece in enumerate(re.split(r"\s+(?=\d+(?:\.\d+)+\.?\s)", re.sub(r"\s+", " ", u).strip())):
                pieces.append((piece, j > 0))
        units = [p for p, _ in pieces]
        # A pair of adjacent sentences as well: "Creatine kinase: For early
        # detection of rhabdomyolysis. Serial monitoring to monitor trend."
        # states one instruction across two (Snakebite s4.5.4) - never across
        # a heading.
        pairs = [f"{x} {y}" for (x, _), (y, opens) in zip(pieces, pieces[1:]) if not opens]
        for sent in units + pairs:
            sent = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", re.sub(r"\s+", " ", sent).strip())
            # ...and stop at the next numbered heading ("... parenchyma. 6.1.8."),
            # with stray page numbers in front removed ("85 90 Serial blood ...").
            sent = re.split(r"\s+\d+(?:\.\d+)+\.?(?=\s|$)", sent)[0].strip()
            sent = re.sub(r"^(?:\d+\s+)+(?=[A-Za-z])", "", sent)
            # A list's lead-in cut from its items ("Indication for observation
            # and admission: i.") says nothing on its own.
            if not 30 <= len(sent) <= 320 or re.search(r":\s*(?:[ivx]{1,4}|[a-z]|\d{1,2})[.)]?$", sent):
                continue
            # Guards found 2026-10-01 once the Snakebite guideline could answer
            # rhabdomyolysis gaps: a table row is not a sentence; a line about a
            # cause the patient does not have (snakebite, dengue ...) does not
            # speak for this patient; and a foreign body's recommendation quoted
            # inside a Malaysian document is not KKM guidance.
            if (len(sent) > 100 and not complications.PROSE.search(sent)) or _FOREIGN_BODY.search(sent):
                continue
            cause = complications.CAUSE_SPECIFIC.search(sent)
            if cause and cause.group(0).lower() not in (state or "").lower():
                continue
            # Evidence is looked for AFTER "do not / avoid" clauses are removed:
            # the heat-stroke rule "DO NOT administer Paracetamol" was quoted as
            # support for analgesia in rhabdomyolysis on 2026-09-30.
            positive = _drop_negated(sent)
            hits = sum(1 for p in patterns if p.search(positive))
            if not hits or _NOT_NEEDED.search(sent) or cards.contradicts_state(sent, state):
                continue
            overlap = len(name_words & set(re.findall(r"[a-z]{4,}", positive.lower())))
            # The sentence must also be ABOUT the element. "heat stroke" in the
            # guideline's scope statement matched "other causes of dark urine"
            # on 2026-09-30 and was quoted as though it answered it.
            names_test = any(rx.search(positive) for rx in test_rx)
            if name_words and not overlap and hits < 2 and not names_test:
                continue
            # KKM before non-KKM, then a sentence naming the element's test, then
            # the best-covering sentence - a recommendation over a description,
            # anything over a list.
            score = (meta.get("doc_type") != config.DOC_TYPE_EXTERNAL, names_test,
                     hits * 3 + overlap + (2 if _RECOMMENDS.search(sent) else 0) - (2 if _listy(sent) else 0)
                     + title_fit)
            if best is None or score > best[0]:
                where = _where(meta)
                best = (score, _clean_quote(sent), where, idx)
    return (best[1], best[2], best[3]) if best else ("", "", None)


_CONSEQUENCE = re.compile(
    r"(?:lead(?:s|ing)? to|result(?:s|ing)? in|caus(?:e|es|ing)|risk of|complicat\w*|"
    r"sign of|feature of|hallmark|indicat\w* of|due to|secondary to|release of|progress\w* to)"
    r"[^.;,]{0,40}$", re.I)


def _named_as_consequence(concepts: set[str], text: str) -> str | None:
    """A concept the report names AS a consequence or feature of the working
    diagnosis ("...which can lead to acute kidney injury"). Merely listing it
    beside the diagnosis ("such as myopathy, rhabdomyolysis") is not enough -
    found 2026-09-30, when "Myopathy" was wrongly removed on that basis."""
    for sentence in re.split(r"(?<=[.;])\s+", text or ""):
        for c in sorted(concepts, key=len, reverse=True):
            words = indications._words(sentence)
            hay = " " + " ".join(words) + " "
            if f" {c} " not in hay:
                continue
            first = c.split()[0]
            idx = sentence.lower().find(first[:5])
            if idx > 0 and _CONSEQUENCE.search(sentence[:idx]):
                return c
    return None


# F4: a differential set aside on a finding whose vital sign was never measured.
# Run 10: "Heat-related illness - the symptoms are more consistent with muscle
# damage ... no mention of hyperthermia" with no temperature recorded.
_UNMEASURED_CUES = (
    ("temperature", "temperature", r"hyperthermi\w*|fever\w*|febrile|pyrexi\w*|temperature|\bheat\b|heat[- ]related"),
    ("spo2", "SpO2", r"hypox\w*|desaturat\w*|oxygen saturation|\bSpO2\b"),
    ("systolic_bp", "blood pressure", r"hypotens\w*|hypertens\w*|\bshock\b|blood pressure"),
    ("heart_rate", "heart rate", r"tachycard\w*|bradycard\w*|heart rate|pulse"),
    ("capillary_blood_glucose", "capillary glucose", r"hypoglyc\w*|hyperglyc\w*|glucose"),
)
_ABSENCE = re.compile(r"\b(?:no|not|without|absence of|lack of|normal|unlikely|less likely|rather than|"
                      r"more consistent with|no evidence|no mention)\b", re.I)


def _cap_confidence(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """F7: a diagnosis its source defines by a test cannot be HIGH confidence
    before that test has a result. Run 10 stated "Exertional Rhabdomyolysis,
    HIGH" with no CK measured; the KKM definition is CK > 10x ULN."""
    prof = complications.profile_for(str(diagnostic.primary_diagnosis.condition or ""))
    if prof is None or not prof.defining_test:
        return
    if str(getattr(diagnostic.primary_diagnosis.confidence, "value", diagnostic.primary_diagnosis.confidence)) != "HIGH":
        return
    result_rx, label, definition, source = prof.defining_test
    if re.search(result_rx, _intake_text(req), re.I):
        return
    diagnostic.primary_diagnosis.confidence = Confidence.MODERATE
    note = (f"[Confidence capped at MODERATE until the {label} result is known - {source}: "
            f"\"{definition}\"]")
    diagnostic.primary_diagnosis.reasoning = f"{diagnostic.primary_diagnosis.reasoning} {note}".strip()
    log.warning("GROUNDING  %s", note)


def _check_unmeasured_exclusions(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    notes = []
    for dd in diagnostic.differential_diagnoses:
        text = f"{dd.condition} {dd.discriminating_feature}"
        if not _ABSENCE.search(str(dd.discriminating_feature or "")):
            continue
        for field, label, cue in _UNMEASURED_CUES:
            if getattr(req.vitals, field, None) is None and re.search(cue, text, re.I):
                notes.append(f"'{dd.condition}' is set aside on a finding that needs the {label}, which was not "
                             f"measured - it cannot be excluded until the {label} is recorded")
                break
    if notes:
        note = "NOT EXCLUDED - " + "; ".join(notes) + "."
        diagnostic.differential_warning = f"{diagnostic.differential_warning} {note}".strip()
        log.warning("GROUNDING  %s", note)


# G7 (2026-09-30): "tamponade ... does not have hypotension with elevated JVP"
# for a patient who WAS hypotensive and whose JVP was never examined. F4 above
# covers the vital signs; these are the examination findings a report can lean
# on without anyone having recorded them.
_EXAM_FINDINGS = (
    ("JVP", r"\bJVP\b|jugular venous|raised neck veins|distended neck veins"),
    ("heart sounds / murmur", r"\bmurmur\w*|heart sounds|muffled heart|gallop"),
    ("chest auscultation", r"\bcrepitation\w*|\bcrackles\b|\brales\b|breath sounds|air entry"),
    ("abdominal examination", r"\bguarding\b|\brigidity\b|rebound tenderness|peritonism|abdominal distension"),
    ("neck stiffness / meningism", r"neck stiffness|meningism|kernig|brudzinski"),
    ("neurological examination", r"focal (?:neurolog\w* )?deficit|hemipares\w*|pupil\w*|plantar\w*|\breflexes\b"),
    ("peripheral pulses", r"pulse deficit|absent (?:peripheral )?pulses|radio-femoral|pulses? (?:equal|unequal)"),
    ("skin / rash", r"\brash\b|petechia\w*|purpura\w*|urticaria"),
    ("peripheral oedema", r"(?:pedal|leg|peripheral|ankle) o?edema"),
)
# Findings the report may state as absent that the recorded state contradicts.
# A negation reaches up to 40 characters within its clause: "No evidence of
# dyspnea or hypoxia" (SpO2 91, RR 28) slipped a two-word window on 2026-10-01.
_NEG = r"\b(?:no|not|without|absence of|absent)\b[^.;:]{0,40}?"
_STATE_CONTRADICTIONS = (
    (_NEG + r"\bhypotens\w*|\bnormotensive\b|haemodynamically stable|hemodynamically stable",
     ("hypotension", "shock"), "hypotension"),
    (_NEG + r"\bhypox\w*|\bnot hypoxic\b|saturating well|normal saturation", ("hypoxia",), "hypoxia"),
    (_NEG + r"\b(?:dyspno?ea|breathless\w*|shortness of breath|respiratory distress)",
     ("tachypnoea", "hypoxia"), "tachypnoea or hypoxia"),
    (r"\bafebrile\b|" + _NEG + r"\bfever\b", ("fever",), "fever"),
    (_NEG + r"\btachycardi\w*", ("tachycardia",), "tachycardia"),
    (_NEG + r"\bshock\b", ("shock",), "shock"),
    (r"\bfully conscious\b|\bGCS (?:of )?15\b|\balert and orientated\b",
     ("reduced consciousness",), "reduced consciousness"),
)
# G8: mutually exclusive subtypes of one diagnosis, named in the same report.
_EXCLUSIVE = (
    (r"\bSTEMI\b|ST[- ]elevation (?:myocardial infarction|MI|ACS)", r"\bNSTE[- ]?ACS\b|\bNSTEMI\b|non[- ]ST[- ]elevation",
     "ST-elevation and non-ST-elevation ACS"),
    (r"isch(?:a)?emic stroke|cerebral infarct", r"ha?emorrhagic stroke|intracerebral ha?emorrhage|\bICH\b",
     "ischaemic and haemorrhagic stroke"),
    (r"\btype 1 diabetes|\bT1DM\b", r"\btype 2 diabetes|\bT2DM\b", "type 1 and type 2 diabetes"),
)


def _check_finding_consistency(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """G7 + G8: statements that contradict the recorded findings, lean on an
    examination finding nobody recorded, or name two exclusive subtypes of the
    diagnosis. Advisory: names the sentence; changes nothing."""
    state = _patient_state(req)
    intake = _intake_text(req)
    notes: list[str] = []
    claims = ([(f"differential '{dd.condition}'", str(dd.discriminating_feature or ""))
               for dd in diagnostic.differential_diagnoses]
              + [("diagnostic reasoning", str(diagnostic.primary_diagnosis.reasoning or "")),
                 ("triage rationale", str(diagnostic.triage_rationale or ""))])
    for where, text in claims:
        for rx, states, label in _STATE_CONTRADICTIONS:
            m = re.search(rx, text, re.I)
            if m and cards.state_hit(states, state):
                notes.append(f"the {where} says \"{m.group(0)}\" but the recorded findings show {label}")
        if where.startswith("differential") and _ABSENCE.search(text):
            for label, rx in _EXAM_FINDINGS:
                m = re.search(rx, text, re.I)
                if m and not re.search(rx, intake, re.I):
                    notes.append(f"the {where} is set aside on the {label} (\"{m.group(0)}\"), which was not "
                                 "recorded - it cannot be excluded on that basis")
                    break
    # A differential that names a state the patient is already in ("Cardiogenic
    # shock - presence of hypotension and signs of shock", 2026-10-01) is the
    # patient's condition, not an alternative to exclude.
    kept_dd = []
    for dd in diagnostic.differential_diagnoses:
        cond = str(dd.condition or "")
        hit = next((s for s in ("shock", "hypotension", "hypoxia", "sepsis") if re.search(rf"\b{s}", cond, re.I)
                    and cards.state_hit((s,), state)), None)
        if hit:
            notes.append(f"'{cond}' is listed as a differential, but the recorded findings show {hit} - it is "
                         "this patient's current state, to be managed, not an alternative to exclude")
            continue
        kept_dd.append(dd)
    diagnostic.differential_diagnoses = kept_dd
    primary = str(diagnostic.primary_diagnosis.condition or "")
    # The plan, not the differentials: listing NSTEMI as a differential of a
    # STEMI is exactly what a differential is for.
    rest = " \n".join(
        [diagnostic.triage_rationale or "", diagnostic.primary_diagnosis.reasoning or "",
         diagnostic.disposition_justification or ""]
        + [a.action for a in diagnostic.immediate_actions if a.origin != "source"]
        + [f"{d.drug_name} {d.indication}" for d in diagnostic.drug_recommendations])
    for a, b, label in _EXCLUSIVE:
        for first, second in ((a, b), (b, a)):
            if re.search(first, primary, re.I) and not re.search(second, primary, re.I):
                m = re.search(second, rest, re.I)
                if m:
                    notes.append(f"the diagnosis is \"{primary}\" but the plan also says \"{m.group(0)}\" - "
                                 f"{label} are managed differently; confirm which applies")
                break
    if notes:
        note = "INCONSISTENT - " + "; ".join(dict.fromkeys(notes)) + "."
        diagnostic.consistency_warning = f"{diagnostic.consistency_warning} {note}".strip()
        log.warning("GROUNDING  %s", note)


# 2026-10-03 (rhabdo run, 82%): ice packs and a mist fan were recommended while
# the report's own differential set heat stroke aside. An action that treats a
# condition the report set aside is not this patient's plan. Removed when the
# finding that decides it was measured; when it was not (the temperature was
# never taken), kept but made conditional on measuring it - the exclusion is
# unproven (F4) and a real heat stroke must not lose its cooling.
# (condition, regex on the condition, regex on an action, vitals field, what to measure)
_CONDITION_ACTIONS = (
    ("heat stroke", r"heat\s*stroke|hyperthermi\w*|heat[- ]related",
     r"\bice\b|evaporative|mist fan|cold water|immersion|tepid spong\w*|\bcool(?:ing)?\b",
     "temperature", "core temperature"),
    ("hypoglycaemia", r"hypoglyc\w*", r"\bdextrose\b|\bD(?:10|50)\b|glucagon",
     "capillary_blood_glucose", "capillary glucose"),
)


def _gate_set_aside_actions(diagnostic: DiagnosticSchema, req: TriageRequest) -> None:
    """Actions that treat a differential the report set aside: removed, or made
    conditional when the deciding finding was never measured."""
    primary = str(diagnostic.primary_diagnosis.condition or "")
    notes: list[str] = []
    for label, cond_rx, act_rx, field, measure in _CONDITION_ACTIONS:
        if re.search(cond_rx, primary, re.I):
            continue
        if not any(re.search(cond_rx, str(dd.condition or ""), re.I)
                   and _ABSENCE.search(str(dd.discriminating_feature or ""))
                   for dd in diagnostic.differential_diagnoses):
            continue
        measured = getattr(req.vitals, field, None) is not None
        kept = []
        for a in diagnostic.immediate_actions:
            if not re.search(act_rx, a.action or "", re.I):
                kept.append(a)
                continue
            if measured:
                notes.append(f"removed \"{a.action}\" - it treats {label}, which the report set aside")
                continue
            if not a.action.startswith("Only if"):
                notes.append(f"\"{a.action}\" treats {label}, which the report sets aside without a {measure} "
                             f"- made conditional on measuring it")
                a.action = f"Only if {label} is confirmed ({measure} not yet recorded - measure it first): {a.action}"
            kept.append(a)
        diagnostic.immediate_actions = kept
    if notes:
        note = "SET-ASIDE CONDITION - " + "; ".join(notes) + "."
        diagnostic.consistency_warning = f"{diagnostic.consistency_warning} {note}".strip()
        log.warning("GROUNDING  %s", note)


# D1 (run 11): "renal function" was listed twice. Test names are compared on a
# canonical form, the way a clinician would read them as the same order.
_TEST_SYNONYMS = (
    (r"renal (?:profile|function|panel)|\bRFT\b|\bRP\b|\bBUSE\b|urea (?:and|&) (?:serum )?creatinine|"
     r"\bU ?& ?E\b|serum (?:urea|creatinine)", "renal profile"),
    (r"full blood (?:count|picture)|\bFBC\b|\bCBC\b|complete blood count", "full blood count"),
    (r"liver function|\bLFTs?\b", "liver function"),
    (r"\b12[- ]lead\b|\bECG\b|electrocardiogra\w*", "ecg"),
    (r"creatine (?:phospho)?kinase|creatinine kinase|\bCK\b|\bCPK\b", "creatine kinase"),
    (r"capillary (?:blood )?glucose|\bCBG\b|\bdextrostix\b|blood (?:sugar|glucose)", "glucose"),
    (r"serum electrolytes|\belectrolytes\b", "electrolytes"),
    (r"troponin\w*|\bhs-?c?Tn\w*", "troponin"),
    (r"chest x-?ray|\bCXR\b", "chest x-ray"),
    (r"coagulation (?:profile|screen)|\bPT\b.{0,10}\bAPTT\b|\bINR\b", "coagulation"),
    (r"blood gas\w*|\bABG\b|\bVBG\b", "blood gas"),
    (r"urine (?:analysis|FEME|dipstick)|urinalysis|\bUFEME\b", "urinalysis"),
    (r"group(?:ing)? and cross[- ]?match|\bGXM\b|cross[- ]?match", "cross-match"),
)


def _canonical_test(name: str) -> str:
    hits = sorted({canon for rx, canon in _TEST_SYNONYMS if re.search(rx, name or "", re.I)})
    if hits:
        return " + ".join(hits)
    return re.sub(r"\W+", " ", (name or "").lower()).strip()


def _dedupe_plan(diagnostic: DiagnosticSchema) -> None:
    """D1: one row per investigation and per action. The first row is kept;
    a later one only lends it the source it lacked."""
    kept: list[Investigation] = []
    index: dict[str, Investigation] = {}
    dropped = []
    for x in diagnostic.investigations:
        key = _canonical_test(x.test)
        if key and key in index:
            first = index[key]
            if not first.source_id and x.source_id:
                first.source_id = x.source_id
            if not first.source_quote and x.source_quote:
                first.source_quote, first.source_where = x.source_quote, x.source_where
            dropped.append(x.test)
            continue
        index[key] = x
        kept.append(x)
    diagnostic.investigations = kept
    seen: set[str] = set()
    actions = []
    for a in diagnostic.immediate_actions:
        key = re.sub(r"\W+", " ", (a.action or "").lower()).strip()
        if key and key in seen:
            dropped.append(a.action)
            continue
        seen.add(key)
        actions.append(a)
    diagnostic.immediate_actions = actions
    if dropped:
        log.info("Deduplicated plan rows: %s", "; ".join(dropped))


def _renumber_actions(diagnostic: DiagnosticSchema) -> None:
    """1..n after removals (the list read 1-5, 7-15 on 2026-10-01), with the
    contraindication rows that name an action kept pointing at the same one."""
    remap = {}
    for n, a in enumerate(diagnostic.immediate_actions, start=1):
        remap[f"action {a.sequence}"] = f"action {n}"
        a.sequence = n
    for c in diagnostic.contraindications:
        c.where = re.sub(r"action \d+", lambda m: remap.get(m.group(0), m.group(0)), c.where)


def _check_differentials(diagnostic: DiagnosticSchema) -> None:
    """A differential must be a different explanation, not the working
    diagnosis again, and not one of its own features or complications.

    Measured 2026-09-30: the rhabdomyolysis report listed "Acute Kidney Injury"
    (the complication its own red flag named) and "Myoglobinuria" (the finding
    its own reasoning named) as differentials to exclude. They are moved out of
    the differential list and named, so the list that remains is one a clinician
    can actually work through."""
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()
    if primary.lower() in _NO_DIAGNOSIS or not diagnostic.differential_diagnoses:
        return
    working = indications.concepts(primary, expand=False)
    # Joined as separate sentences, so "leading to" in the reasoning cannot
    # reach across into a red flag's text.
    own_text = ". ".join(
        [diagnostic.primary_diagnosis.reasoning or ""]
        + [f"{f.flag}: {f.why_it_matters}" for f in diagnostic.red_flags]
    )
    kept, moved = [], []
    for dd in diagnostic.differential_diagnoses:
        mine = indications.concepts(dd.condition, expand=False)
        if not mine:
            kept.append(dd)
            continue
        if mine <= working:
            moved.append(f"'{dd.condition}' restates the working diagnosis")
        elif _named_as_consequence({c for c in mine if " " in c or len(c) >= 6}, own_text):
            moved.append(f"'{dd.condition}' is named in this report as a feature or complication of "
                         f"{primary}, not an alternative to it")
        else:
            kept.append(dd)
    if moved:
        diagnostic.differential_diagnoses = kept
        diagnostic.differential_warning = (
            "Removed from the differentials - " + "; ".join(moved)
            + ". Differentials should be alternative explanations to exclude.")
        log.warning("GROUNDING  %s", diagnostic.differential_warning)


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
        note(getattr(x, "source_id", ""))
    for c in diagnostic.source_cautions:
        note(c.source_id)

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
    stripped: set[str] | None = None,
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
            if where not in (stripped or set()):  # already named as unsupported
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


# F3: an adult fluid volume or rate - "2 - 6 l", "250 - 300 ml / hr", "1-2 L/hr".
_ADULT_VOLUME_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:-|\u2013|to)?\s*\d*(?:[.,]\d+)?\s*(?:ml|l|litres?|liters?)\s*(?:/|per)\s*(?:h|hr|hour)\b"
    r"|\d+(?:[.,]\d+)?\s*(?:-|\u2013|to)?\s*\d*(?:[.,]\d+)?\s*(?:l|litres?|liters?)\b(?!\s*/\s*(?:kg|min))",
    re.I)
_FLUID_DRUG = re.compile(r"sodium chloride|saline|ringer|hartmann|dextrose|crystalloid|isotonic|\bfluids?\b", re.I)
_FLUID_WORD = re.compile(r"fluid|isotonic|crystalloid|saline|ringer|hartmann|\bns\b|\blr\b", re.I)

_DOSE_IN_TEXT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:-|–|to)?\s*\d*(?:[.,]\d+)?\s*"
    r"(?:mcg|microgram|mg|g|ml|units?|iu)\b(?:\s*/\s*kg)?", re.IGNORECASE)


def _cpg_dose_sentence(drug_name: str, chunks: list[Retrieved], stated: str = "") -> tuple[str, str]:
    """A verbatim sentence from the retrieved guideline that names this drug AND
    a dose. FUKKM carries general dosing; an indication-specific loading dose
    lives only in the CPG, so aspirin 300mg in ACS can be verified from nowhere
    else."""
    clinical = set(config.CLINICAL_DOC_TYPES) | {config.DOC_TYPE_TRIAGE}
    if _FLUID_DRUG.search(drug_name or ""):
        # F3: a fluid regimen names "isotonic fluids" or "NS / LR", rarely the
        # FUKKM product, and is written as a volume and a rate. Rates first.
        best: tuple[int, str, str] | None = None
        for chunk in chunks:
            if chunk.metadata.get("doc_type") not in clinical:
                continue
            flat = re.sub(r"\s+", " ", _clean_quote(chunk.text or ""))
            for sentence in re.split(r"(?<=[.;])\s+", flat):
                if not 20 <= len(sentence) <= 320 or not _FLUID_WORD.search(sentence):
                    continue
                if not _ADULT_VOLUME_RE.search(sentence):
                    continue
                score = (2 if re.search(r"/\s*(?:h|hr|hour)\b", sentence, re.I) else 0) + \
                        (1 if re.search(r"bolus|initial", sentence, re.I) else 0)
                if best is None or score > best[0]:
                    page = chunk.metadata.get("page_number")
                    title = str(chunk.metadata.get("cpg_title", "") or "")
                    best = (score, sentence.strip(), f"{title} p{page}" if page else title)
        if best:
            return best[1], best[2]
    aliases = _drug_aliases(drug_name)
    if not aliases:
        return "", ""
    pattern = _alias_pattern(aliases)
    for chunk in chunks:
        if chunk.metadata.get("doc_type") not in clinical:
            continue
        for sentence in re.split(r"(?<=[.;])\s+|\n", chunk.text or ""):
            text = re.sub(r"\s+", " ", sentence).strip()
            if len(text) < 12 or len(text) > 320:
                continue
            if pattern.search(text) and _DOSE_IN_TEXT.search(text) and not cards.purpose_mismatch(text, stated):
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
    regimen: list[str] = []
    no_regimen: list[str] = []
    cs = _cards_for(diagnostic, req)
    primary = str(diagnostic.primary_diagnosis.condition or "").strip()

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

        purpose = " ".join(str(getattr(drug, f, "") or "") for f in ("indication", "adult_dose", "paediatric_dose",
                                                                      "frequency", "duration"))
        cpg_text, cpg_ref = _cpg_dose_sentence(str(drug.drug_name or ""), chunks, purpose)
        if not cpg_text and cs:
            # The diagnosis's card may hold the regimen even when the page with
            # it was not retrieved.
            hit = cards.has_dose(cs, _drug_aliases(str(drug.drug_name or "")) | {
                str(drug.drug_name or "").lower()}, purpose)
            if hit is not None:
                cpg_text, cpg_ref = hit.quote, hit.where
        if cpg_text:
            sources.append((cpg_ref or "CPG", cpg_text))
            drug.dose_source_cpg = f"{cpg_text}  [{cpg_ref}]"

        verdict = dosing.check(stated, sources, req.weight_kg, paediatric)
        drug.dose_verdict = verdict.status
        drug.dose_verdict_detail = verdict.detail
        # F3: a stated dose that is only FUKKM's general range, beside a
        # condition-specific regimen in the guideline ("100 - 1000 ml IV, as
        # needed" for rhabdomyolysis, whose CHAMP regimen is litres per hour).
        nums = set(re.findall(r"\d+(?:\.\d+)?", stated))
        if match and cpg_text and nums:
            in_fukkm = nums <= set(re.findall(r"\d+(?:\.\d+)?", match.best.dosage or ""))
            in_cpg = bool(nums & set(re.findall(r"\d+(?:\.\d+)?", cpg_text)))
            if in_fukkm and not in_cpg:
                regimen.append(f"{drug.drug_name}: the stated dose '{stated[:40]}' is FUKKM's general range; "
                               f"the guideline regimen for this presentation is \"{cpg_text[:220]}\" [{cpg_ref}]")
        # D3: a fluid order whose figures are FUKKM's general range, and no KKM
        # sentence for this diagnosis gives a regimen - said plainly, so the
        # general range is never read as the condition's regimen.
        elif (match and not cpg_text and nums and _FLUID_DRUG.search(str(drug.drug_name or ""))
              and nums <= set(re.findall(r"\d+(?:\.\d+)?", match.best.dosage or ""))
              and primary.lower() not in _NO_DIAGNOSIS):
            no_regimen.append(f"{drug.drug_name}: '{stated[:40]}' is FUKKM's general range - no indexed KKM "
                              f"document gives a fluid regimen for {primary}")
        if verdict.status == dosing.EXCEEDS_MAXIMUM:
            alarms.append(f"{drug.drug_name}: {verdict.detail}")
        elif verdict.status != dosing.VERIFIED:
            unverified.append(f"{drug.drug_name} ({verdict.status.replace('_', ' ').lower()})")

    notes: list[str] = []
    if alarms:
        notes.append("DOSE EXCEEDS A STATED MAXIMUM - " + "; ".join(alarms) + ".")
    if regimen:
        notes.append("DOSE IS THE GENERAL RANGE, NOT THIS CONDITION'S REGIMEN - " + "; ".join(regimen) + ".")
    if no_regimen:
        notes.append("NO KKM REGIMEN - " + "; ".join(no_regimen)
                     + ". The volume and rate are the clinician's decision.")
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


def triage_preview(req: TriageRequest) -> TriagePreview:
    """The deterministic triage, without the model: ~1 s instead of minutes.

    Uses exactly the functions the full run uses (`_mts_floor`, `MTS_TIME`,
    `_reassess`, the red-flag rules, the provisional check), so the preview and
    the report cannot disagree about anything the code decides."""
    level, reasons = _mts_floor(req)
    shell = DiagnosticSchema(mts_triage_level=TriageLevel(level))
    _flag_provisional_triage(shell, req)
    return TriagePreview(
        mts_triage_level=level,
        triage_colour=TRIAGE_COLOUR[level],
        mts_triage_label=TriageLevel(level).name,
        time_to_treatment=MTS_TIME[level],
        reassessment=_reassess(level),
        reasons=reasons,
        red_flags=[f"{flag.label} (\"{hit}\")"
                   for flag, hit in red_flags.match(focus.strip_negated(_intake_text(req)))],
        provisional_note=shell.triage_provisional_note,
    )


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


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _static_prefix_len(engine, prompt_ids: list[int], system: str) -> int:
    """How many leading tokens of this prompt are the fixed system block.

    Found by rendering the same chat template with an EMPTY user turn and taking
    the common token prefix - no assumption about the template's markup. Cached
    per system prompt: it is computed once per process."""
    key = hash(system)
    cached = engine._static_prefix.get(key)
    if cached is None:
        shell = _build_prompt(engine._tokenizer, [{"role": "system", "content": system},
                                                   {"role": "user", "content": ""}])
        cached = engine._tokenizer.encode(shell)
        engine._static_prefix[key] = cached
    return _common_prefix(prompt_ids, cached)


def _generate(engine, prompt: str, max_tokens: int, job_id: str = "", system: str = "",
              stats: dict | None = None) -> str:
    """Streamed so progress reflects real work: prefill is exact (mlx-lm reports
    tokens processed against the total) and decode is counted per token.

    Two savings, both measured motives:

      Prompt-prefix cache. The ~2,000-token system prompt is identical on every
      request; its KV cache is kept between requests and only the rest of the
      prompt is prefilled (~15 s at this machine's 135 tok/s). After each run the
      cache is trimmed back to that fixed prefix, so it holds ~300 MB, not the
      ~2 GB of a whole case. Reuse is decided by exact token equality, so a
      changed system prompt simply rebuilds it.

      Loop stop. 2026-09-30, ahf-72f: the model repeated one sentence for 15,418
      characters and ran for 502 s until the token cap. Generation now stops as
      soon as a passage repeats (`_repetition`), and `_flag_truncation` marks
      the report INCOMPLETE.

    `stats`, when given, is filled with what A4.1 records: prompt tokens, how
    many came from the prefix cache, output tokens, and prefill / decode time
    (prefill ends at the first generated token)."""
    kw = _gen_kwargs(engine)
    if job_id:
        kw["prompt_progress_callback"] = lambda done, total: progress.prefill(job_id, done, total)
    ids = engine._tokenizer.encode(prompt)
    feed: list[int] = ids
    use_cache = config.PROMPT_CACHE and system and "draft_model" not in kw and not config.KV_BITS
    static_len = 0
    if use_cache:
        try:
            static_len = _static_prefix_len(engine, ids, system)
            cache = engine._prompt_cache
            reuse = _common_prefix(ids, engine._prompt_cache_tokens) if cache is not None else 0
            reuse = min(reuse, static_len, len(ids) - 1)
            if cache is not None and reuse >= 256 and cache_mod.can_trim_prompt_cache(cache):
                extra = cache[0].offset - reuse
                if extra > 0:
                    cache_mod.trim_prompt_cache(cache, extra)
                feed = ids[reuse:]
                log.info("Prompt cache: reused %d of %d prompt tokens.", reuse, len(ids))
            else:
                cache = cache_mod.make_prompt_cache(engine._model)
                reuse = 0
            engine._prompt_cache = cache
            kw["prompt_cache"] = cache
        except Exception as exc:  # the cache is an optimisation, never a failure
            log.warning("Prompt cache unavailable (%s); prefilling the whole prompt.", exc)
            feed, use_cache, static_len = ids, False, 0
            engine._prompt_cache, engine._prompt_cache_tokens = None, []
            kw.pop("prompt_cache", None)
    while True:
        try:
            out, tokens, n = [], [], 0
            t_start = time.perf_counter()
            t_first = None
            for resp in stream_generate(
                engine._model, engine._tokenizer, prompt=feed,
                max_tokens=max_tokens, **kw,
            ):
                if t_first is None:
                    t_first = time.perf_counter()
                out.append(resp.text)
                tokens.append(resp.token)
                n += 1
                if job_id and n % 8 == 0:  # ~8 tokens between updates keeps the lock cheap
                    progress.decode(job_id, n)
                # 8 repeats, not 4: JSON legitimately repeats short runs of
                # structure (several drugs with the same empty fields).
                if n % 64 == 0 and _repetition("".join(out)[-3000:], times=8):
                    log.warning("Generation stopped at %d tokens: the output is repeating itself.", n)
                    break
            if job_id:
                progress.decode(job_id, n)
            if stats is not None:
                t_end = time.perf_counter()
                t_first = t_first or t_end
                stats.update(prompt_tokens=len(ids), reused_tokens=len(ids) - len(feed),
                             output_tokens=n, prefill_ms=int((t_first - t_start) * 1000),
                             decode_ms=int((t_end - t_first) * 1000))
            if use_cache:
                engine._prompt_cache_tokens = ids + tokens
                keep = static_len if static_len >= 256 else 0
                cache = engine._prompt_cache
                if keep and cache_mod.can_trim_prompt_cache(cache):
                    cache_mod.trim_prompt_cache(cache, cache[0].offset - keep)
                    engine._prompt_cache_tokens = ids[:keep]
                else:
                    engine._prompt_cache, engine._prompt_cache_tokens = None, []
            return "".join(out)
        except TypeError as exc:
            # Drop whichever accelerator this build will not take, and retry.
            dropped = next((k for k in ("draft_model", "kv_bits", "prompt_cache")
                            if k in kw and k in str(exc)), None)
            if dropped is None:
                dropped = next(iter(kw), None)
            if dropped is None:
                raise
            log.warning("Generation option %r unsupported here (%s); continuing without it.", dropped, exc)
            kw.pop(dropped)
            if dropped == "prompt_cache":
                feed, use_cache = ids, False
                engine._prompt_cache, engine._prompt_cache_tokens = None, []


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
        # Prompt-prefix KV cache (see _generate) and the tokens it holds.
        self._prompt_cache = None
        self._prompt_cache_tokens: list[int] = []
        self._static_prefix: dict[int, list[int]] = {}

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------ warm-up
    warm_state: str = "off"

    def warm_up(self) -> str:
        """Build, before the first report, everything the first report would
        otherwise build: the model, the BM25 index and A4.2 term weights, the
        reference-list filter, the embedder and reranker, the formulary index,
        and the KV cache of the fixed system prompt.

        Measured 2026-09-30: the first report after a (re)start spent 28 s in
        retrieval (8 s warm) and read its prompt with nothing cached. Holds the
        GPU slot, so a report submitted meanwhile waits rather than competing.
        Any failure leaves the engine as lazy as it was - never unusable."""
        t0 = time.perf_counter()
        self.warm_state = "warming"
        try:
            with gpu_queue.slot():
                self._ensure_model_loaded()
                _ = self._corpus
                _ = self._bibliography_ids
                self.retrieve("exertional rhabdomyolysis dark urine management", 3, config.CLINICAL_DOC_TYPES)
                self._embed(["warm-up"])
                formulary.mentioned("aspirin 300 mg")
                negatives.check("aspirin", "warm-up", False)
                if config.PROMPT_CACHE:
                    # One token: the prefill is the point. _generate keeps the
                    # system prompt's KV cache for the next request.
                    prompt = _build_prompt(self._tokenizer, [{"role": "system", "content": TRIAGE_SYSTEM},
                                                             {"role": "user", "content": "warm-up"}])
                    _generate(self, prompt, 1, system=TRIAGE_SYSTEM)
            self.warm_state = f"ready ({time.perf_counter() - t0:.0f} s)"
        except Exception as exc:  # noqa: BLE001
            self.warm_state = f"failed: {type(exc).__name__}: {exc}"[:200]
            log.warning("Warm-up incomplete (%s); the first report will build what is missing.", exc)
        log.info("Warm-up %s", self.warm_state)
        return self.warm_state

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
        set_support_idf(docs)  # A4.2 term weights, from the same corpus
        log.info("BM25 index ready in %.1fs.", time.time() - t0)
        return {
            "ids": ids,
            "docs": docs,
            "metas": [m or {} for m in metas],
            "bm25": bm25,
            "pos": {cid: i for i, cid in enumerate(ids)},
        }

    @cached_property
    def _bibliography_ids(self) -> frozenset[str]:
        """Chunks that are reference-list pages (distill.is_bibliography) -
        17.5% of the clinical index, measured 2026-09-30 (1,456 of 8,306). They
        are skipped at retrieval, so a search slot goes to guidance instead."""
        if not config.DROP_BIBLIOGRAPHY:
            return frozenset()
        c = self._corpus
        ids = frozenset(cid for cid, d, m in zip(c["ids"], c["docs"], c["metas"])
                        if m.get("doc_type") in config.CLINICAL_DOC_TYPES and distill.is_bibliography(d))
        log.info("Skipping %d reference-list chunks at retrieval.", len(ids))
        return ids

    @cached_property
    def indexed_titles(self) -> set[str]:
        return {m.get("cpg_title", "") for m in self._corpus["metas"]}

    @cached_property
    def _reranker(self):
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        log.info("Loading cross-encoder %s...", config.RERANKER_MODEL)
        return CrossEncoder(config.RERANKER_MODEL, max_length=config.RERANK_MAX_LENGTH)

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
            if c["ids"][i] in self._bibliography_ids:
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
        bib = self._bibliography_ids
        fused = [c for c in fused if c.chunk_id not in bib]

        if not use_rerank:
            return fused[:k]
        try:
            return self._rerank(query, fused[: max(k, config.RERANK_POOL)], k)
        except Exception as exc:
            log.warning("Reranker failed (%s: %s); returning fused order.", type(exc).__name__, exc)
            return fused[:k]

    # --------------------------------------------------- red-flag recall floor
    def _hypotheses(self, req: TriageRequest, attention, triage_line: str, job_id: str = "",
                    trace: "attention_audit.Trace | None" = None) -> Hypotheses:
        """A2.1: a short model call on the KEY FINDINGS alone, merged with the
        conditions the red-flag rules name. Any failure falls back to the rules:
        this steers retrieval, it must never stop a report."""
        rules = _rule_hypotheses(req)
        if not config.HYPOTHESES:
            return Hypotheses(rules, [], "rules")
        progress.stage(job_id, "retrieval", "Forming a working differential", 0.2)
        user = (f"Patient: {req.age:g} y, {req.gender.value}.\n"
                f"{focus.render(attention, triage_line)}\n\n"
                'Respond ONLY with JSON: {"hypotheses": ["<most likely>", "<second>", "<third>"], '
                '"must_exclude": ["<dangerous alternative>"]}')
        try:
            prompt = _build_prompt(self._tokenizer, [{"role": "system", "content": HYPOTHESIS_SYSTEM},
                                                     {"role": "user", "content": user}])
            # No prefix cache: the one cache slot holds the triage prompt's prefix.
            likely, exclude = _parse_hypotheses(_generate(
                self, prompt, config.MAX_TOKENS_HYPOTHESES,
                stats=trace.generation("hypotheses") if trace else None))
        except Exception as exc:
            log.warning("Hypothesis step failed (%s); using the red-flag rules only.", exc)
            likely, exclude = [], []
        merged = list(dict.fromkeys(likely + [r for r in rules
                                              if not any(r.lower() in x.lower() for x in likely)]))
        return Hypotheses(merged[:5], exclude, "model+rules" if likely else "rules",
                          proposed=likely + exclude)

    def _hypothesis_retrieval(self, req: TriageRequest, attention, hyp: Hypotheses) -> list[Retrieved]:
        """A2.2: one search for the complaint, one per working hypothesis, one per
        red-flag finding - fused by rank, at most MAX_PER_DOC passages per
        document so no single guideline crowds out the rest.

        Weak findings ("sweating", "smoker", "diabetic") are NOT searched on
        their own: measured 2026-09-30, they pulled in depression and diabetes
        CPGs for a chest-pain patient."""
        queries = [f"{req.complaint} {req.history} {_retrieval_terms(req)} diagnosis and management"]
        queries += [f"{h} emergency department management" for h in hyp.all()[:4]]
        # "Management" alone reaches diagnosis and pathway pages; the drug
        # chapters need their own query (ACS CPG p172 - aspirin, clopidogrel -
        # was only found this way, measured 2026-09-30).
        queries += [f"{h} initial treatment drugs and doses" for h in hyp.likely[:2]]
        # Measured best at reaching acute drug pages across STEMI, asthma,
        # cellulitis and dengue (2026-09-30 phrasing probe).
        queries += [f"{h} emergency drug therapy loading dose" for h in hyp.likely[:1]]
        queries += [f.text for f in attention.ranked() if f.red_flag][:2]
        plan = _checklist_queries(req, hyp)
        queries += plan
        treatment = {q for q in queries if q.endswith(("drugs and doses", "loading dose"))} | set(plan)
        unique = list(dict.fromkeys(queries))
        lists = [self.retrieve(q, config.HYPOTHESIS_QUERY_K, config.CLINICAL_DOC_TYPES) for q in unique]
        fused = self._rrf(*lists) if len(lists) > 1 else lists[0]
        # Every search keeps its best passage first (a round of heads), then
        # the fused order fills the rest - so a must-exclude hypothesis such
        # as DVT keeps a passage even when the likelier ones dominate the
        # fused ranking (the swollen-leg probe lost its VTE passage).
        # Treatment searches keep their top TWO: the fused ranking favours the
        # diagnosis pages every query shares, and pushed the STEMI loading-
        # dose page (aspirin + clopidogrel) out of a 12-passage context.
        heads = [c for q, lst in zip(unique, lists) for c in lst[: 2 if q in treatment else 1]]
        # Checklist heads first: the per-document cap filled with CHAMP's
        # diagnosis pages before its fluid-rate page was reached (2026-09-30).
        # The analgesia search is guaranteed a head too, but in the normal
        # order: at the front, its pain page out-ranked the HDP manual.
        plan_heads = ([c for q, lst in zip(unique, lists) if q in plan and "analgesia" not in q for c in lst[:2]]
                      + [c for q, lst in zip(unique, lists) if q in plan and "analgesia" in q for c in lst[:1]])
        heads = plan_heads + heads
        seen_ids: set[str] = set()
        ordered = [c for c in heads + fused if not (c.chunk_id in seen_ids or seen_ids.add(c.chunk_id))]
        out: list[Retrieved] = []
        per: dict[str, int] = {}
        for c in ordered:
            t = str(c.metadata.get("cpg_title", ""))
            if per.get(t, 0) >= config.MAX_PER_DOC:
                continue
            per[t] = per.get(t, 0) + 1
            out.append(c)
            if len(out) >= config.CLINICAL_TOTAL:
                break
        return out

    @cached_property
    def _embed(self):
        """The index's own medical embedder (MedEmbed), reused for sentences."""
        fn = getattr(self.collection, "_embedding_function", None)
        if fn is None:
            from .ingest import build_embedding_function  # noqa: PLC0415
            fn = build_embedding_function()
        return lambda texts: fn(list(texts))

    def _distill(self, chunks: list[Retrieved], queries: list[str],
                 edge: bool = True) -> tuple[list[Retrieved], str]:
        """A3 (distill.py): keep each clinical chunk's most relevant sentences,
        trim formulary rows, order edge-first, and fit a token budget.

        Any failure returns the chunks unchanged - distillation saves time, it
        must never cost a report."""
        try:
            clinical = [c for c in chunks if c.metadata.get("doc_type") in config.CLINICAL_DOC_TYPES]
            other = [distill.with_text(c, distill.trim_formulary(c.text))
                     if c.metadata.get("doc_type") == config.DOC_TYPE_DRUG else c
                     for c in chunks if c.metadata.get("doc_type") not in config.CLINICAL_DOC_TYPES]
            before = sum(len(c.text) for c in chunks)

            def dose_sentence(t: str) -> bool:
                return bool(_TREATMENT_HINT.search(t) and formulary.mentioned(t)
                            and distill._QUANTITY.search(t))

            scored = distill.distill(clinical, queries, self._embed,
                                     per_chunk=config.DISTILL_SENTENCES, always_keep=dose_sentence)
            # Budget: drop whole low-relevance chunks until the clinical text
            # fits, but never a document's last passage.
            # ~3.5 characters per token for this text (lowercased, numeric);
            # 4 under-counted - ACS measured 3.9k tokens against a 3.5k budget.
            budget = int(config.CONTEXT_BUDGET_TOKENS * 3.5)
            ranked = sorted(scored, key=lambda x: -x.relevance)
            kept, used, per_doc = [], 0, {}
            for sc in ranked:
                per_doc[sc.chunk.metadata.get("cpg_title")] = per_doc.get(sc.chunk.metadata.get("cpg_title"), 0) + 1
            for sc in ranked:
                size = len(sc.text())
                title = sc.chunk.metadata.get("cpg_title")
                if used + size > budget and per_doc[title] > 1:
                    per_doc[title] -= 1
                    continue
                kept.append(sc)
                used += size
            ordered = distill.edge_order(kept) if edge else kept
            half = (len(ordered) + 1) // 2
            front = [distill.with_text(sc.chunk, sc.text()) for sc in ordered[:half]]
            back = [distill.with_text(sc.chunk, sc.text()) for sc in ordered[half:]]
            # Formulary rows are look-up tables: the middle, not an edge.
            result = front + other + back
            after = sum(len(c.text) for c in result)
            note = (f"Distilled context: {len(result)} passage(s), ~{int(after / 3.5):,} tokens "
                    f"(from ~{int(before / 3.5):,}); most relevant sentences kept, best at the start and end")
            return result, note
        except Exception as exc:
            log.warning("Distillation skipped (%s); using whole passages.", exc)
            return chunks, "Distillation skipped - whole passages used"

    @cached_property
    def _fukkm_rows(self) -> dict[str, int]:
        """FUKKM listing number -> index of its chunk in the corpus."""
        c = self._corpus
        return {str(m.get("fukkm_no")): i for i, m in enumerate(c["metas"])
                if m.get("doc_type") == config.DOC_TYPE_DRUG and m.get("fukkm_no")}

    def _formulary_by_name(self, chunks: list[Retrieved], age: float = 30.0,
                           titles: set[str] | None = None, pain: int | None = None) -> list[Retrieved]:
        """Attention layer A1/A2: the FUKKM rows for drugs the kept guideline text
        NAMES IN TREATMENT SENTENCES, at most DRUG_K.

        - Clauses that say NOT to give a drug are removed first.
        - Only sentences with a dose, route or "give/commence" count; PDF line
          breaks are joined first so a name and its dose meet.
        - Ranked by the relevance of the passages that give the drug (earlier
          passages weigh more), with acute-phase wording ("loading", "stat",
          "IV", "bolus", "initial") counting double - otherwise a smoking-
          cessation page put nicotine gum above aspirin for a STEMI.
        - The formulation must fit the patient: no eye or ear drops, creams or
          lozenges; syrups only for children, tablets and injections for adults
          (0.9% saline EYE DROPS were chosen for a rhabdomyolysis patient).
        - `titles`: only these documents are read (_treatment_titles)."""
        weight: dict[str, float] = {}
        first: dict[str, int] = {}
        routes: dict[str, str] = {}  # F3: the route the guideline sentence states
        specific: set[str] = set()   # named by a condition's own guideline, not a general one
        pos = 0
        rank = 0
        for c in chunks:
            if c.metadata.get("doc_type") not in config.CLINICAL_DOC_TYPES:
                continue
            if titles is not None and c.metadata.get("cpg_title") not in titles:
                continue
            rank += 1
            # Once per passage, at its strongest mention: a pain page that names
            # morphine in six sentences out-voted MgSO4 for eclampsia.
            per_chunk: dict[str, float] = {}
            flat = re.sub(r"(?<![.;:])\s*\n\s*(?!\s*[-\u2022\uf0b7\d])", " ", c.text or "")
            for sent in re.split(r"(?<=[.;])\s+|\n", _drop_negated(flat)):
                # A paper title is not a treatment sentence ("...benazepril and
                # losartan in chronic renal insufficiency").
                if not _TREATMENT_HINT.search(sent) or distill.is_citation(sent):
                    continue
                boost = 2.0 if _ACUTE_CUE.search(sent) else 1.0
                # A general-purpose document (Pain Management) names drugs for
                # every patient; its opioid rows displaced MgSO4 for eclampsia
                # and oxytocin for PPH (2026-09-30). Half weight.
                if any(g in str(c.metadata.get("cpg_title", "")).lower() for g in _GENERAL_TITLE_MARKS):
                    boost *= 0.5
                general_doc = boost < 1.0 or any(
                    g in str(c.metadata.get("cpg_title", "")).lower() for g in _GENERAL_TITLE_MARKS)
                route = _route_cue(sent)
                for g in dict.fromkeys(formulary.mentioned(sent)):
                    per_chunk[g] = max(per_chunk.get(g, 0.0), boost / rank)
                    if not general_doc:
                        specific.add(g)
                    first.setdefault(g, pos)
                    # A parenteral mention anywhere wins: "IV phenytoin" may
                    # come after a sentence naming it without a route.
                    if route and (g not in routes or route in ("iv", "im")):
                        routes[g] = route if routes.get(g) not in ("iv", "im") else routes[g]
                    pos += 1
            for g, w in per_chunk.items():
                weight[g] = weight.get(g, 0.0) + w
        # Names that resolve to the same FUKKM products are one drug: "MgSO4"
        # and "magnesium sulphate" split its weight in half (2026-09-30).
        canon: dict[tuple, str] = {}
        for g in sorted(weight, key=lambda g: (-weight[g], first[g])):
            key = tuple(sorted(str(e.fukkm_no) for e in formulary.entries_for(g))) or (g,)
            if key in canon:
                keep = canon[key]
                weight[keep] += weight.pop(g)
                first[keep] = min(first[keep], first[g])
                if g in routes and routes.get(keep) not in ("iv", "im"):
                    routes[keep] = routes[g]
            else:
                canon[key] = g
        # The condition's own guideline's drugs before a general document's
        # (gout: colchicine before the pain page's paracetamol and morphine).
        ranked = sorted(weight, key=lambda g: (g not in specific, -weight[g], first[g]))
        # A drug named once, late, is not what this presentation is treated
        # with (nicotine spray ranked sixth for a STEMI).
        top = weight[ranked[0]] if ranked else 0.0
        kept_ranked = [g for g in ranked if weight[g] >= config.FORMULARY_MIN_SHARE * top]
        # F2: pain 4+ keeps one analgesic row whatever its weight - paracetamol
        # when the context names it (run 10 offered none for pain 7/10).
        if pain is not None and pain >= 4 and not any(g in _ANALGESICS for g in kept_ranked):
            # The condition's own guideline first (ACS: morphine, not
            # paracetamol), then paracetamol, then any analgesic named.
            pick = next((g for g in ranked if g in _ANALGESICS and g in specific), None) or \
                next((g for g in ranked if g == "paracetamol"), None) or \
                next((g for g in ranked if g in _ANALGESICS), None)
            # A free row only: never displaces a drug the condition's own
            # guideline names (ACS keeps its antithrombotics).
            if pick and len(kept_ranked) < config.FORMULARY_ROWS:
                kept_ranked = kept_ranked + [pick]
        ranked = kept_ranked
        rows, out = self._fukkm_rows, []
        have = {c.chunk_id for c in chunks}
        corpus = self._corpus
        child = age < config.PAEDIATRIC_AGE_YEARS
        for generic in ranked:
            for e in _fitting_formulations(formulary.entries_for(generic), child, routes.get(generic, "")):
                i = rows.get(str(e.fukkm_no))
                if i is None or corpus["ids"][i] in have:
                    continue
                out.append(Retrieved(chunk_id=corpus["ids"][i], text=corpus["docs"][i],
                                     metadata=corpus["metas"][i]))
                have.add(corpus["ids"][i])
                break
            if len(out) >= config.FORMULARY_ROWS:
                break
        return out

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
    def _context(self, chunks: list[Retrieved], start: int = 1) -> tuple[str, list[RetrievedSource]]:
        blocks = []
        sources = []
        for i, chunk in enumerate(chunks, start=start):
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
                    url=meta.get("url"),
                )
            )
        return "\n\n---\n\n".join(blocks), sources

    # ============================================================== MODE A
    @cached_property
    def provenance(self) -> dict:
        """Code, corpus and model fingerprints for this process. Cached: none of
        them can change without a restart (see versioning.py)."""
        sha, dirty = versioning.git_state()
        corpus, docs, chunks = versioning.corpus_fingerprint(self.collection)
        return versioning.Provenance(
            git_sha=sha, git_dirty=dirty,
            code_fingerprint=versioning.code_fingerprint(),
            corpus_fingerprint=corpus, corpus_documents=docs, corpus_chunks=chunks,
            model=self.model_name,
        ).as_dict()

    def triage(self, req: TriageRequest, job_id: str = "") -> TriageResponse:
        """One run at a time on the reasoning model; see gpu_queue.py."""
        with gpu_queue.slot(job_id) as ticket:
            result = self._triage(req, job_id)
        result.provenance.queue_wait_ms = ticket.waited_ms
        return result

    def _triage(self, req: TriageRequest, job_id: str = "") -> TriageResponse:
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
        trace = attention_audit.Trace()
        floor_level, floor_reasons = _mts_floor(req)
        attention = focus.extract(req)
        if config.TRIAGE_CONTEXT == "chunks":
            triage_chunks = self.retrieve(
                f"MTS 2022 triage {req.complaint} {vitals}",
                config.TRIAGE_K,
                [config.DOC_TYPE_TRIAGE],
            )
            triage_chunks = self._dedupe(
                triage_chunks,
                self.discriminator_cells(
                    f"{req.complaint} {vitals}", floor_level, cohort, config.MTS_CELL_K
                ),
            )
        else:
            # Attention layer A1: the level is decided and enforced by code
            # (mts_table + _enforce_mts_level), so the MTS cells are replaced by
            # a two-line TRIAGE block in the prompt. Measured 2026-09-30: ten
            # cells were 1,745 of 8,165 context tokens, and they were what the
            # model kept citing for IV fluids and urine output.
            triage_chunks = []

        # 2. Clinical guidance. CLINICAL_DOC_TYPES includes the Paediatric
        #    Protocols (1,190 chunks) - previously excluded here, so the richest
        #    paediatric source in the corpus was never retrieved for Mode A.
        triage_line = (f"MTS {floor_level} {TriageLevel(floor_level).name} - "
                       + ("; ".join(floor_reasons) if floor_reasons else "no discriminator above Level 5 matched"))
        with trace.stage("hypotheses"):
            hyp = self._hypotheses(req, attention, triage_line, job_id, trace)
        t_retrieval = time.perf_counter()
        progress.stage(job_id, "retrieval", "Searching clinical practice guidelines", 0.4)
        clinical_chunks = self._hypothesis_retrieval(req, attention, hyp)
        if cohort == "paediatric":
            clinical_chunks = self._dedupe(
                clinical_chunks,
                self.retrieve(
                    f"child paediatric management of {req.complaint}",
                    config.PAEDS_K,
                    [config.DOC_TYPE_PAEDS],
                ),
            )

        # 3. Formulary. By NAME from the guideline text (A1) - or, with
        #    FORMULARY_BY_NAME=0, the older embedding search over treatment
        #    sentences, which returned antivenoms for a rhabdomyolysis patient.
        progress.stage(job_id, "retrieval", "Searching the FUKKM formulary", 0.75)
        drug_chunks = [] if config.FORMULARY_BY_NAME else self.retrieve(
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
            # Negated clauses removed first: "no chest pain" must not force the
            # ACS guideline in.
            chunks, red_flag_notes = self.apply_red_flags(
                focus.strip_negated(_intake_text(req)), chunks, config.CLINICAL_DOC_TYPES
            )

        chunks, population_notes = _filter_population(chunks, req.age)
        red_flag_notes += population_notes
        chunks, bib_notes = _drop_bibliography(chunks)
        red_flag_notes += bib_notes
        if config.TOPIC_GATE:
            forced = set(red_flags.titles_for(focus.strip_negated(_intake_text(req))))
            chunks, gate_notes = _topic_gate(chunks, req, hyp, forced)
            red_flag_notes += [hyp.describe()] + gate_notes
        card_chunks, card_note = self._card_passages(req, hyp, chunks)
        if card_chunks:
            chunks = chunks + card_chunks
            red_flag_notes.append(card_note)
        if config.FORMULARY_BY_NAME:
            chunks = chunks + self._formulary_by_name(
                chunks, req.age, _treatment_titles(chunks, req, hyp) if config.TOPIC_GATE else None,
                req.vitals.pain_score)
        if config.DISTILL:
            queries = ([f.text for f in attention.ranked()[:5]] + hyp.all()[:4]
                       + [f"{h} treatment and dose" for h in hyp.likely[:1]])
            chunks, distil_note = self._distill(chunks, queries)
            red_flag_notes.append(distil_note)

        context, sources = self._context(chunks)
        weight = f"{req.weight_kg:g} kg" if req.weight_kg else "NOT SUPPLIED"
        focus_block = focus.render(attention, triage_line) if config.FOCUS_BLOCK else ""

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
  "immediate_actions": [{"sequence": 1, "action": "<action; a dose, volume or rate ONLY as written in the cited passage, else no number>", "timeframe": "<when>", "source_id": "S1"}],
  "investigations": [{"test": "<test>", "rationale": "<why>", "urgency": "STAT"}],
  "drug_recommendations": [{"drug_name": "<drug>", "indication": "<indication>", "adult_dose": "<dose>", "paediatric_dose": "<mg/kg or ml/kg dose>", "route": "<route>", "frequency": "<frequency>", "duration": "<duration>", "prescriber_category": "<FUKKM category from a retrieved chunk, else NOT_IN_RETRIEVED_SOURCES>", "prescriber_category_meaning": "<what that category means>", "cautions": "<cautions>", "source_id": "S1"}],
  "prescriber_category_warning": "<name any drug whose category was NOT in the retrieved chunks, else empty>",
  "disposition": "ED_OBSERVATION",
  "disposition_justification": "<why this destination>",
  "referral_required": false,
  "referral_to": "<team, or empty>",
  "evidence_gaps": "<vitals not supplied, and anything the retrieved chunks did not cover>"
}"""
        if config.OUTPUT_DIET:
            # A4.4: rows and fields the server derives exactly are not asked
            # for (_vitals_rows, _ground_prescriber_categories). Measured on
            # run 7: ~1,550 output tokens at ~10 tokens/s were the bulk of 167 s.
            json_template = re.sub(r'\n  "vitals_interpretation": [^\n]*', "", json_template)
            json_template = re.sub(r'\n  "prescriber_category_warning": [^\n]*', "", json_template)
            json_template = re.sub(r', "prescriber_category": "[^"]*", "prescriber_category_meaning": "[^"]*"',
                                   "", json_template)
            # Plan E (run 11: writing was 70 of 116 s). An adult report has no
            # use for a per-kg paediatric dose, and the free-text slots ask for
            # the length a clinician reads, not an essay.
            if cohort == "adult":
                json_template = json_template.replace(
                    ', "paediatric_dose": "<mg/kg or ml/kg dose>"', "")
            for slot, short in (
                    ("<why THIS patient is that level, naming the worst criterion>",
                     "<one sentence: the worst criterion for THIS patient>"),
                    ("<reasoning from THIS patient's findings>", "<two sentences from THIS patient's findings>"),
                    ('"why_it_matters": "<why>"', '"why_it_matters": "<one short clause>"'),
                    ("<what separates it>", "<one short clause>"),
                    ('"rationale": "<why>"', '"rationale": "<a few words>"'),
                    ("<why this destination>", "<one sentence>"),
                    ('"cautions": "<cautions>"', '"cautions": "<patient-specific caution, or empty>"')):
                json_template = json_template.replace(slot, short)

        user_prompt = f"""PATIENT
Age: {req.age:g} years ({cohort})
Gender: {req.gender.value}
Weight: {weight}
Vitals: {vitals}
Complaint: {req.complaint}
Pain Score: {getattr(req.vitals, 'pain_score', 'Not specified')}
History: {req.history or "None"}{_patient_extras(req)}

{focus_block}

SOURCES ({len(chunks)} chunks)
{context}

{focus.reminder(attention) if config.FOCUS_BLOCK else ""}
TASK: Evaluate based on the Evaluation Sequence.
REMINDER 1: every <angle-bracket> below is a SLOT to fill with THIS patient's
data. Never copy a slot marker into your answer. The numbers and enum values
shown are placeholders illustrating the type - they are NOT recommendations.
{"" if config.OUTPUT_DIET else _REMINDER_CATEGORY}Respond ONLY with a complete JSON object matching this exact structure:
{json_template}"""

        messages = [{"role": "system", "content": TRIAGE_SYSTEM}, {"role": "user", "content": user_prompt}]
        
        prompt = _build_prompt(self._tokenizer, messages)
        trace.add("retrieval", int((time.perf_counter() - t_retrieval) * 1000))
        first_pass = len(sources)
        progress.stage(job_id, "prefill", "Reading the retrieved guideline context", 0.0)
        raw_response = _generate(self, prompt, config.MAX_TOKENS_TRIAGE, job_id, system=TRIAGE_SYSTEM,
                                 stats=trace.generation("triage"))
        
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

        _flag_truncation(diagnostic, raw_response, cleaned)
        model_saw = len(chunks)  # before F5 appends pages the model never read
        if config.TWO_STAGE:
            with trace.stage("second_pass"):
                if config.STAGE2_MODE == "llm":
                    self._second_stage(req, diagnostic, chunks, sources, job_id, hyp, trace)
                else:
                    self._quoted_gap_items(req, diagnostic, chunks, sources, trace)

        t_checks = time.perf_counter()
        progress.stage(job_id, "checks", "Running triage, grounding and safety checks", 0.3)
        if config.STAGE2_MODE == "llm":
            model_saw = len(chunks)  # the LLM second pass read its passages
        complications_added = self._anchor_complications(req, diagnostic, chunks, sources)
        self._source_cautions(req, diagnostic, chunks, sources)
        # Enforced in code, not asked for in the prompt.
        _enforce_mts_level(diagnostic, req)
        _gate_drug_indications(diagnostic, req, chunks)
        _ground_prescriber_categories(diagnostic, sources)
        _check_drug_indications(diagnostic, chunks, self._corpus)
        _check_avoid_statements(diagnostic, req)
        _ground_action_quantities(diagnostic, chunks)
        _check_dose_completeness(diagnostic, req)
        _check_dosing(diagnostic, req, chunks)
        _check_contraindications(diagnostic, req)
        _gate_set_aside_actions(diagnostic, req)
        _dedupe_plan(diagnostic)
        _renumber_actions(diagnostic)
        _enforce_disposition(diagnostic, req)
        _quoted_disposition_floor(diagnostic, req)
        _check_redirection(diagnostic, req)
        _check_urgency_sanity(diagnostic, req)
        # Last of the checks that can move the level, so it reads the final one.
        _ground_triage_timing(diagnostic)
        _check_differentials(diagnostic)
        _check_unmeasured_exclusions(diagnostic, req)
        _cap_confidence(diagnostic, req)
        _check_finding_consistency(diagnostic, req)
        _check_completeness(diagnostic, req, self._corpus)
        _state_knowledge_gaps(diagnostic, req)
        _check_population_sources(diagnostic, req)
        moves = _align_citations(diagnostic, chunks, model_saw)
        support_notes, stripped = _check_citation_support(diagnostic, chunks)
        _check_citations(diagnostic, sources, _ground_citations(diagnostic, sources) + support_notes,
                         stripped)
        if config.OUTPUT_DIET:
            diagnostic.vitals_interpretation = _vitals_rows(req)
        else:
            _derive_vitals_levels(diagnostic, req)
        _flag_provisional_triage(diagnostic, req)
        _derive_onset(diagnostic, req)
        _attach_references(diagnostic, req)
        _flag_leaked_slots(diagnostic)

        trace.add("checks", int((time.perf_counter() - t_checks) * 1000))
        attention_report = attention_audit.build(trace, sources, first_pass, diagnostic)
        attention_report.citation_moves = moves
        attention_report.citations_unsupported = len(stripped)
        attention_report.complications_added = complications_added
        red_flag_notes.append(attention_audit.summary(attention_report))

        return TriageResponse(
            diagnostic=diagnostic,
            triage_colour=TRIAGE_COLOUR[int(diagnostic.mts_triage_level)],
            request=req,
            sources=sources,
            model=self.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            corpus_warnings=self.corpus_warnings,
            retrieval_notes=red_flag_notes,
            attention=attention_report,
        )

    # ------------------------------------------------ A4.3 complications
    def _complication_sentence(self, prof, comp, chunks: list[Retrieved], req: TriageRequest):
        """(sid or None, sentence, chunk) for the first sentence that may speak
        for this complication - in the given passages first, then the index."""
        intake = _intake_text(req)

        def sentences_of(text: str, title: str = ""):
            for sent in distill.sentences(text)[1]:
                sent = _clean_quote(sent)
                if distill.is_citation(sent):
                    continue
                # Section numbers glue headings to sentences ("... cardiac
                # arrest. 6.3.6 Cardiopulmonary ..."): quote the piece that
                # carries the statement.
                # ...and superscript citations glue on the next one ("late
                # complication of er. 15, 44 in the proper").
                for piece in re.split(r"\s*(?:^|\s)\d+(?:\.\d+)+\.?(?=\s|$)"
                                      r"|(?<=[a-z)\]])\.\s+\d{1,3}(?:\s*[,\u2013-]\s*\d{1,3})*(?=\s+[a-z])", sent):
                    piece = piece.strip(" .;:")
                    if complications.qualifies(_drop_negated(piece), prof, comp, intake, title):
                        yield piece + "."
                        break

        # One ranking over every candidate: a KKM sentence beats a non-KKM
        # one; among equals, a passage the model was given beats the rest of
        # the index; then prose, then length. Taking the first given match let
        # CHAMP p5's admission-criteria list beat the KKM Heat p10 sentence for
        # AKI (2026-09-30).
        cands: list[tuple[tuple, str | None, str, Retrieved]] = []
        for i, c in enumerate(chunks, start=1):
            if c.metadata.get("doc_type") not in config.CLINICAL_DOC_TYPES:
                continue
            title = str(c.metadata.get("cpg_title", ""))
            ext = c.metadata.get("doc_type") == config.DOC_TYPE_EXTERNAL
            for sent in sentences_of(c.text, title):
                pref = complications.preference(sent, title, ext, complications.on_topic(prof, title))
                cands.append(((pref[0], 0) + pref[1:] + (i,), f"S{i}", sent, c))
        corpus = self._corpus
        bib = self._bibliography_ids
        given_ids = {c.chunk_id for c in chunks}
        for cid, text, meta in zip(corpus["ids"], corpus["docs"], corpus["metas"]):
            if cid in given_ids or meta.get("doc_type") not in config.CLINICAL_DOC_TYPES or cid in bib:
                continue
            if meta.get("status", "active") != "active" or not population.allowed(meta, req.age):
                continue
            if not re.search(comp.evidence, text or "", re.I):
                continue
            title = str(meta.get("cpg_title", ""))
            ext = meta.get("doc_type") == config.DOC_TYPE_EXTERNAL
            for sent in sentences_of(text, title):
                pref = complications.preference(sent, title, ext, complications.on_topic(prof, title))
                cands.append(((pref[0], 1) + pref[1:] + (0,), None, sent,
                              Retrieved(chunk_id=cid, text=text, metadata=meta)))
        if not cands:
            return None
        _, sid, sent, chunk = min(cands, key=lambda x: x[0])
        return sid, sent, chunk

    def _source_cautions(self, req: TriageRequest, diagnostic: DiagnosticSchema,
                         chunks: list[Retrieved], sources: list[RetrievedSource]) -> int:
        """F2: the "do not" statements the source makes for the working
        diagnosis (complications.Caution), quoted verbatim and cited - only
        from documents whose title is about the condition; KKM before non-KKM,
        a given passage before the rest of the index."""
        prof = complications.profile_for(str(diagnostic.primary_diagnosis.condition or ""))
        if prof is None or not prof.cautions:
            return self._card_cautions(req, diagnostic, chunks, sources)
        intake = focus.strip_negated(_intake_text(req))
        corpus, bib = self._corpus, self._bibliography_ids
        given_ids = {c.chunk_id for c in chunks}
        added = 0
        try:
            for cau in prof.cautions:
                if cau.when and not re.search(cau.when, intake, re.I):
                    continue
                rx = re.compile(cau.evidence, re.I)
                cands = []
                pool = [(i, c.chunk_id, c.text, c.metadata) for i, c in enumerate(chunks, start=1)] + [
                    (None, cid, t, m) for cid, t, m in zip(corpus["ids"], corpus["docs"], corpus["metas"])
                    if cid not in given_ids]
                for i, cid, text, meta in pool:
                    title = str(meta.get("cpg_title", ""))
                    if (meta.get("doc_type") not in config.CLINICAL_DOC_TYPES or cid in bib
                            or not complications.on_topic(prof, title)):
                        continue
                    # Flattened first: a PDF line break sits inside most phrases.
                    flat = re.sub(r"\s+", " ", _clean_quote(text))
                    m = rx.search(flat)
                    if not m:
                        continue
                    sent = complications.quote_around(flat, m)
                    if distill.is_citation(sent):
                        continue
                    ext = meta.get("doc_type") == config.DOC_TYPE_EXTERNAL
                    cands.append(((ext, i is None, len(sent)), i, sent,
                                  Retrieved(chunk_id=cid, text=text, metadata=meta)))
                if not cands:
                    continue
                _, i, sent, chunk = min(cands, key=lambda x: x[0])
                if i is None:
                    _, new = self._context([chunk], start=len(chunks) + 1)
                    chunks.append(chunk)
                    sources.extend(new)
                    sid = new[0].source_id
                else:
                    sid = f"S{i}"
                diagnostic.source_cautions.append(SourceCaution(
                    label=cau.label, quote=sent + complications.gloss_for(prof, sent), source_id=f"[{sid}]",
                    external=chunk.metadata.get("doc_type") == config.DOC_TYPE_EXTERNAL))
                added += 1
        except Exception as exc:  # a caution must never cost the report
            log.warning("Source cautions skipped (%s).", exc)
        return added

    def _anchor_complications(self, req: TriageRequest, diagnostic: DiagnosticSchema,
                              chunks: list[Retrieved], sources: list[RetrievedSource]) -> int:
        """A4.3 (complications.py): add the diagnosis's complications the red
        flags miss, each from a verbatim sentence; then drop red flags that only
        restate the diagnosis. Appends to chunks/sources when the sentence came
        from outside the prompt. Returns how many were added."""
        prof = complications.profile_for(str(diagnostic.primary_diagnosis.condition or ""))
        if prof is None:
            return self._card_complications(req, diagnostic, chunks, sources)
        try:
            # A flag that only restates the diagnosis does not count as naming a
            # complication - it is removed below, and its "can lead to acute
            # kidney injury" would leave AKI unlisted.
            texts = [f"{f.flag} {f.why_it_matters}" for f in diagnostic.red_flags
                     if not complications.restates(prof, f.flag)]
            added: list[str] = []
            for comp in complications.missing(prof, texts):
                hit = self._complication_sentence(prof, comp, chunks, req)
                if hit is None:
                    continue
                sid, sentence, chunk = hit
                if sid is None:
                    _, new = self._context([chunk], start=len(chunks) + 1)
                    chunks.append(chunk)
                    sources.extend(new)
                    sid = new[0].source_id
                diagnostic.red_flags.append(RedFlag(
                    flag=comp.label,
                    why_it_matters=f'Source: "{sentence}"' + complications.gloss_for(prof, sentence),
                    source_id=f"[{sid}]",
                    origin="source"))
                added.append(f"{comp.label} [{sid}] ({_where(chunk.metadata)})")
            has_complication = any(
                not complications.restates(prof, f.flag) and any(
                    re.search(c.present, f"{f.flag} {f.why_it_matters}", re.I) for c in prof.complications)
                for f in diagnostic.red_flags)
            removed = []
            if has_complication:
                keep = []
                for f in diagnostic.red_flags:
                    if f.origin != "source" and complications.restates(prof, f.flag):
                        removed.append(f.flag)
                    else:
                        keep.append(f)
                diagnostic.red_flags = keep
            # A complication is not an alternative diagnosis: run 9 listed "Acute
            # kidney injury" as a differential of rhabdomyolysis while it sat in
            # the red flags. Moved only when the red flags carry it, so it is
            # never lost from the report.
            flags_text = " ".join(f"{f.flag} {f.why_it_matters}" for f in diagnostic.red_flags)
            moved, keep_dd = [], []
            for dd in diagnostic.differential_diagnoses:
                cond = str(dd.condition or "")
                comp = next((c for c in prof.complications if re.search(c.present, cond, re.I)), None)
                if (comp and not re.search(prof.diagnosis, cond, re.I)
                        and re.search(comp.present, flags_text, re.I)):
                    moved.append(cond)
                    continue
                keep_dd.append(dd)
            diagnostic.differential_diagnoses = keep_dd
            parts = []
            if added:
                parts.append(f"Red flag(s) added for {prof.name}, each quoting its source verbatim: "
                             + "; ".join(added))
            if moved:
                parts.append(f"Removed from the differentials because each is a complication of "
                             f"{prof.name}, listed under red flags: " + "; ".join(moved))
            if removed:
                parts.append("Removed as red flags because they restate the diagnosis or the complaint "
                             "rather than a danger: " + "; ".join(removed))
            if parts:
                diagnostic.complication_note = ". ".join(parts) + "."
                log.warning("GROUNDING  %s", diagnostic.complication_note)
            return len(added)
        except Exception as exc:  # an addition must never cost the report
            log.warning("Complication anchoring skipped (%s).", exc)
            return 0

    # ------------------------------------------------ condition cards
    def _cite_item(self, item, chunks: list[Retrieved], sources: list[RetrievedSource]) -> str | None:
        """The [S#] of a card item's chunk: its place in the prompt, or a new
        source appended after the ones the model read."""
        for i, c in enumerate(chunks, start=1):
            if c.chunk_id == item.chunk_id:
                return f"S{i}"
        corpus = self._corpus
        pos = _pos(corpus).get(item.chunk_id)
        if pos is None:
            return None
        chunk = Retrieved(chunk_id=item.chunk_id, text=corpus["docs"][pos], metadata=corpus["metas"][pos])
        _, new = self._context([chunk], start=len(chunks) + 1)
        chunks.append(chunk)
        sources.extend(new)
        return new[0].source_id

    def _card_passages(self, req: TriageRequest, hyp: "Hypotheses", chunks: list[Retrieved]
                       ) -> tuple[list[Retrieved], str]:
        """Plan B1, for every condition: the chunks that hold the likeliest
        hypothesis's core first-hours items, from its condition card. Retrieval
        searches the complaint's words; the card knows where the management for
        the DIAGNOSIS is written - the SPG's appendix protocols, a CPG's drug
        chapter - wherever the wording differs."""
        if not config.CARDS or config.CARD_PASSAGES <= 0 or not hyp.likely:
            return [], ""
        card = cards.lookup(hyp.likely[0], req.age, bool(req.pregnant))
        if card is None:
            return [], ""
        items = cards.checklist(cards.lineage(card, req.age, bool(req.pregnant)), _patient_state(req),
                                req.age, bool(req.pregnant), limit=40, context=_card_context(self._corpus))
        have = {c.chunk_id for c in chunks}
        corpus, bib = self._corpus, self._bibliography_ids
        out: list[Retrieved] = []
        for it in items:
            if it.chunk_id in have or it.chunk_id in bib:
                continue
            pos = _pos(corpus).get(it.chunk_id)
            if pos is None or not population.allowed(corpus["metas"][pos], req.age):
                continue
            have.add(it.chunk_id)
            out.append(Retrieved(chunk_id=it.chunk_id, text=corpus["docs"][pos], metadata=corpus["metas"][pos]))
            if len(out) >= config.CARD_PASSAGES:
                break
        note = (f"Condition card '{card.name}': {len(out)} passage(s) holding its core first-hours items added "
                f"({'; '.join(sorted({_where(c.metadata) for c in out}))})") if out else ""
        return out, note

    def _card_complications(self, req: TriageRequest, diagnostic: DiagnosticSchema,
                            chunks: list[Retrieved], sources: list[RetrievedSource], limit: int = 3) -> int:
        """A4.3 for any condition with a card: the complications its KKM
        sources name that the red flags miss, each quoted; then red flags that
        only restate the diagnosis or complaint are removed, and differentials
        that are this condition's complications are moved to the red flags'
        side. Same contract as the hand profiles - silence when nothing
        qualifies."""
        cs = _cards_for(diagnostic, req)
        if not cs:
            return 0
        try:
            state = _patient_state(req)
            comps = cards.complications(cs, state, req.age, bool(req.pregnant))
            if not comps:
                return 0
            name = cs[0].name
            own = (indications.concepts(str(diagnostic.primary_diagnosis.condition or ""), expand=False)
                   | indications.concepts(req.complaint or "", expand=False)
                   | {cards._key(n) for c in cs for n in (c.name,) + c.aliases})

            def restates(flag: RedFlag) -> bool:
                mine = indications.concepts(flag.flag, expand=False)
                return bool(mine) and mine <= own and not any(cards.present(c, flag.flag) for c in comps)

            flags_text = " ".join(f"{f.flag} {f.why_it_matters}" for f in diagnostic.red_flags if not restates(f))
            added: list[str] = []
            for comp in comps:
                if len(added) >= limit:
                    break
                if cards.present(comp, flags_text):
                    continue
                sid = self._cite_item(comp, chunks, sources)
                if sid is None:
                    continue
                diagnostic.red_flags.append(RedFlag(
                    flag=comp.label, why_it_matters=f'Source: "{comp.quote}"', source_id=f"[{sid}]",
                    origin="source"))
                flags_text += f" {comp.label} {comp.quote}"
                added.append(f"{comp.label} [{sid}] ({comp.where})")
            removed = []
            if any(any(cards.present(c, f"{f.flag} {f.why_it_matters}") for c in comps) for f in diagnostic.red_flags):
                keep = []
                for f in diagnostic.red_flags:
                    if f.origin != "source" and restates(f):
                        removed.append(f.flag)
                    else:
                        keep.append(f)
                diagnostic.red_flags = keep
            flags_text = " ".join(f"{f.flag} {f.why_it_matters}" for f in diagnostic.red_flags)
            moved, keep_dd = [], []
            for dd in diagnostic.differential_diagnoses:
                comp = next((c for c in comps if cards.present(c, str(dd.condition or ""))), None)
                if comp and cards.present(comp, flags_text):
                    moved.append(str(dd.condition))
                    continue
                keep_dd.append(dd)
            diagnostic.differential_diagnoses = keep_dd
            parts = []
            if added:
                parts.append(f"Red flag(s) added for {name}, each quoting its source verbatim: " + "; ".join(added))
            if moved:
                parts.append(f"Removed from the differentials because each is a complication of {name}, "
                             "listed under red flags: " + "; ".join(moved))
            if removed:
                parts.append("Removed as red flags because they restate the diagnosis or the complaint "
                             "rather than a danger: " + "; ".join(removed))
            if parts:
                diagnostic.complication_note = ". ".join(parts) + "."
                log.warning("GROUNDING  %s", diagnostic.complication_note)
            return len(added)
        except Exception as exc:  # an addition must never cost the report
            log.warning("Card complications skipped (%s).", exc)
            return 0

    def _card_cautions(self, req: TriageRequest, diagnostic: DiagnosticSchema,
                       chunks: list[Retrieved], sources: list[RetrievedSource], limit: int = 3) -> int:
        """F2 for any condition with a card: its KKM "avoid" sentences that
        apply to this patient - first those naming something the report
        recommends, then the rest - quoted and cited."""
        cs = _cards_for(diagnostic, req)
        if not cs:
            return 0
        try:
            items = cards.cautions(cs, _patient_state(req), req.age, bool(req.pregnant))
            plan = _answer_text(diagnostic)
            items.sort(key=lambda i: not cards.present(i, plan))
            added = 0
            for it in items[:limit]:
                sid = self._cite_item(it, chunks, sources)
                if sid is None:
                    continue
                diagnostic.source_cautions.append(SourceCaution(
                    label=it.label, quote=it.quote, source_id=f"[{sid}]", external=it.external))
                added += 1
            return added
        except Exception as exc:  # a caution must never cost the report
            log.warning("Card cautions skipped (%s).", exc)
            return 0

    # ------------------------------------------------ F5 quoted gap items
    _INVESTIGATION_ELEMENT = re.compile(
        r"\bCK\b|creatine|creatinine|urea|potassium|\bECG\b|bloods?|\bFBC\b|platelet|liver|\bLFT\b|"
        r"level\b|cross[- ]?match|group and|coagulation|urin(?:e|alysis) (?:analysis|output)?|glucose|"
        r"renal profile|monitoring", re.I)

    def _quoted_gap_items(self, req: TriageRequest, diagnostic: DiagnosticSchema, chunks: list[Retrieved],
                          sources: list[RetrievedSource], trace: "attention_audit.Trace | None" = None) -> int:
        """F5: for each checklist element the answer left out, add the source
        sentence that covers it as an action or investigation - verbatim,
        cited, marked "from source". No model call.

        Left to other layers: admission (F1 decides disposition). Never used:
        a quote that recommends what the element says to avoid ("Analgesia,
        with NSAIDs avoided" must not be filled with "a short course of ...
        NSAIDs")."""
        label, gaps = _gaps(req, diagnostic, self._corpus)
        if not gaps:
            if trace:
                trace.second_pass = "not needed (checklist complete)"
            return 0
        corpus = self._corpus
        state = _patient_state(req)
        added: list[str] = []
        merged: list[str] = []
        seq = max((a.sequence for a in diagnostic.immediate_actions), default=0)
        by_id = {c.chunk_id: i for i, c in enumerate(chunks, start=1)}
        for gap in gaps:
            # At most QUOTED_GAP_MAX additions: seven on 2026-10-01 turned a
            # resuscitation plan into a list of algorithm lines.
            if len(added) >= config.QUOTED_GAP_MAX:
                break
            if re.search(r"\badmi", gap.element, re.I):
                continue
            if gap.item is not None:
                quote, where, idx = gap.item.quote, gap.item.where, _pos(corpus).get(gap.item.chunk_id)
            else:
                quote, where, idx = _gap_quote_hit(label, gap, corpus, req.age, state)
            if not quote or idx is None:
                continue
            avoided = re.search(r"(\w+?)s? avoided", gap.element, re.I)
            if avoided and re.search(re.escape(avoided.group(1)), _drop_negated(quote), re.I):
                continue
            cid = corpus["ids"][idx]
            if cid in by_id:
                sid = f"S{by_id[cid]}"
            else:
                chunk = Retrieved(chunk_id=cid, text=corpus["docs"][idx], metadata=corpus["metas"][idx])
                _, new = self._context([chunk], start=len(chunks) + 1)
                chunks.append(chunk)
                sources.extend(new)
                sid = new[0].source_id
                by_id[cid] = len(chunks)
            # D4: the element is the action; the sentence it rests on is shown
            # beneath it, not run into it.
            is_test = (gap.item.element == "investigation" if gap.item is not None
                       else bool(self._INVESTIGATION_ELEMENT.search(gap.element)))
            # An action needs a sentence that says to DO something; a heading
            # ("TIMI RISK SCORE FOR UA/NSTEMI") is not one. A test list may be.
            if not is_test and not (_INSTRUCTION.search(quote) or _DOSE_IN_TEXT.search(quote)):
                continue
            # Fix 2026-10-03: "Serial CK and bloods" was appended beside the
            # model's "Creatine kinase (CK)" row; D1 then merged the two, so the
            # serial instruction vanished, the CK row wore the serial-bloods
            # quote, and the merge spent a slot of the cap that compartment
            # syndrome needed. A test the plan already orders is completed in
            # place and costs no slot.
            same = next((x for x in diagnostic.investigations
                         if _canonical_test(x.test) == _canonical_test(gap.element)), None) if is_test else None
            if same is not None:
                if same.source_quote:
                    continue
                same.test = f"{same.test} - {gap.element}"
                same.source_id, same.source_quote, same.source_where = f"[{sid}]", quote, where
                merged.append(f"investigation: {same.test} [{sid}]")
                continue
            if is_test:
                diagnostic.investigations.append(Investigation(
                    test=gap.element, urgency="Urgent", rationale="",
                    source_id=f"[{sid}]", origin="source", source_quote=quote, source_where=where))
                added.append(f"investigation: {gap.element} [{sid}]")
            else:
                seq += 1
                diagnostic.immediate_actions.append(ImmediateAction(
                    sequence=seq, action=gap.element, timeframe="",
                    source_id=f"[{sid}]", origin="source", source_quote=quote, source_where=where))
                added.append(f"action: {gap.element} [{sid}]")
        if trace:
            trace.second_pass = (f"quoted {len(added) + len(merged)} of {len(gaps)} gap(s) from source "
                                 "(no model call)")
        if added or merged:
            diagnostic.second_pass_note = (
                "Added from source for checklist elements the answer left out, each quoting its guideline "
                "sentence: " + "; ".join(added + merged) + ".")
            log.info("QUOTED GAPS  %s", diagnostic.second_pass_note)
        return len(added) + len(merged)

    def _second_stage(
        self,
        req: TriageRequest,
        diagnostic: DiagnosticSchema,
        chunks: list[Retrieved],
        sources: list[RetrievedSource],
        job_id: str = "",
        hyp: "Hypotheses | None" = None,
        trace: "attention_audit.Trace | None" = None,
    ) -> None:
        """Second pass: retrieve for the DIAGNOSIS, then ask only for what is missing.

        The first pass retrieves for the complaint, before any diagnosis
        exists - so "bilateral leg pain" retrieves the VTE CPG, and management
        for the diagnosis the model then reaches may never have been in front
        of it. When the completeness checklist finds at least
        TWO_STAGE_MIN_GAPS required elements missing, this retrieves again with
        the working diagnosis and the missing elements as the query and asks for
        those items only. Appends to `chunks` and `sources` in place (new ids
        continue the numbering), so every later check - the indication gate,
        dosing, citation support - covers the new items exactly as the old.

        Any failure keeps the first-pass answer unchanged: this can only add."""
        primary = str(diagnostic.primary_diagnosis.condition or "").strip()
        if primary.lower() in _NO_DIAGNOSIS:
            return
        label, gaps = _gaps(req, diagnostic, self._corpus)
        if len(gaps) < config.TWO_STAGE_MIN_GAPS:
            if trace:
                trace.second_pass = f"not needed ({len(gaps)} gap(s))"
            return
        # A4.4: a gap no KKM sentence answers cannot be filled from KKM
        # sources, so it does not buy a second model call. Run 7's
        # "Admission for IV fluids and serial CK" rested on CHAMP, which was
        # not in the index until 2026-09-30.
        answerable = [g for g in gaps
                      if (g.item is not None and _pos(self._corpus).get(g.item.chunk_id) is not None)
                      or (g.item is None and _gap_quote(label, g, self._corpus, req.age, _patient_state(req))[0])]
        if not answerable:
            if trace:
                trace.second_pass = (f"skipped - none of the {len(gaps)} gap(s) is answered by an "
                                     "indexed KKM sentence")
            log.info("Second pass skipped: no indexed sentence answers %s", [g.element for g in gaps])
            return
        gaps = answerable
        if trace:
            trace.second_pass = f"ran for {len(gaps)} answerable gap(s)"
        try:
            progress.stage(job_id, "checks", f"Second pass: retrieving guidance for {primary}", 0.05)
            query = f"{primary} " + " ".join(g.element for g in gaps) + " management treatment"
            have = {c.chunk_id for c in chunks}
            extra = [c for c in self.retrieve(query, config.CPG_K, config.CLINICAL_DOC_TYPES)
                     if c.chunk_id not in have]
            extra, _ = _filter_population(extra, req.age)
            extra, _ = _drop_bibliography(extra)
            if config.TOPIC_GATE:
                # Same gate as the first pass, with the diagnosis the first
                # pass reached added: the second pass drew from snakebite, ACS
                # and dengue pages for rhabdomyolysis on 2026-09-30.
                gate_h = Hypotheses([primary] + (hyp.likely if hyp else []), hyp.exclude if hyp else [], "",
                                    proposed=[primary] + (hyp.proposed if hyp else []))
                forced = set(red_flags.titles_for(focus.strip_negated(_intake_text(req))))
                extra, _ = _topic_gate(extra, req, gate_h, forced)
            extra = extra[: config.TWO_STAGE_CHUNKS]
            if not extra:
                return
            if config.DISTILL:
                extra, _ = self._distill(extra, [primary] + [g.element for g in gaps], edge=False)
            context, new_sources = self._context(extra, start=len(chunks) + 1)
            already = "\n".join(
                [f"- action: {a.action}" for a in diagnostic.immediate_actions]
                + [f"- investigation: {x.test}" for x in diagnostic.investigations]
                + [f"- drug: {d.drug_name}" for d in diagnostic.drug_recommendations]
            ) or "- (nothing)"
            user = f"""PATIENT: {req.age:g} y, {req.gender.value}. {req.complaint}
Vitals: {req.vitals.as_clinical_text()}
WORKING DIAGNOSIS: {primary}

MISSING (address these only):
{chr(10).join(f"- {g.element}: {_no_refs(g.why)}" for g in gaps)}

ALREADY IN THE PLAN:
{already}

SOURCES:
{context}

Respond ONLY with JSON of this shape (omit a list you cannot fill from the SOURCES):
{{"immediate_actions": [{{"action": "<action>", "timeframe": "<when>", "source_id": "[S#]"}}],
 "investigations": [{{"test": "<test>", "urgency": "STAT|Urgent|Routine", "rationale": "<why>"}}],
 "drug_recommendations": [{{"drug_name": "<drug>", "indication": "<why>", "adult_dose": "<dose from source or empty>", "paediatric_dose": "", "route": "<route>", "frequency": "", "duration": "", "cautions": "", "source_id": "[S#]"}}]}}"""
            prompt = _build_prompt(self._tokenizer, [{"role": "system", "content": STAGE2_SYSTEM},
                                                     {"role": "user", "content": user}])
            progress.stage(job_id, "checks", f"Second pass: writing the missing items for {primary}", 0.1)
            raw = _generate(self, prompt, config.MAX_TOKENS_STAGE2,  # no prefix cache: keep TRIAGE's
                            stats=trace.generation("stage2") if trace else None)
            payload = json.loads(_extract_json(raw))
        except Exception as exc:
            log.warning("Second pass skipped (%s); keeping the first-pass answer.", exc)
            return

        def norm(text: str) -> str:
            return re.sub(r"\W+", " ", (text or "").lower()).strip()

        def seen(text: str, pool: list[str]) -> bool:
            # Both sides normalised: "Serum creatine kinase (CK) level" was
            # added twice on 2026-09-30 because only one side lost its brackets.
            t = norm(text)
            return not t or any(t in p or p in t for p in (norm(x) for x in pool))

        added: list[str] = []
        pool = [re.sub(r"\W+", " ", a.action.lower()).strip() for a in diagnostic.immediate_actions]
        seq = max((a.sequence for a in diagnostic.immediate_actions), default=0)
        for item in payload.get("immediate_actions") or []:
            try:
                a = ImmediateAction.model_validate({**item, "sequence": seq + 1})
            except Exception:
                continue
            if seen(a.action, pool):
                continue
            seq += 1
            diagnostic.immediate_actions.append(a)
            pool.append(a.action.lower())
            added.append(f"action: {a.action}")
        pool = [x.test.lower() for x in diagnostic.investigations]
        for item in payload.get("investigations") or []:
            try:
                x = Investigation.model_validate(item)
            except Exception:
                continue
            if seen(x.test, pool):
                continue
            diagnostic.investigations.append(x)
            pool.append(x.test.lower())
            added.append(f"investigation: {x.test}")
        pool = [d.drug_name.lower() for d in diagnostic.drug_recommendations]
        for item in payload.get("drug_recommendations") or []:
            try:
                d = DrugRecommendation.model_validate(item)
            except Exception:
                continue
            if seen(d.drug_name, pool):
                continue
            diagnostic.drug_recommendations.append(d)
            pool.append(d.drug_name.lower())
            added.append(f"drug: {d.drug_name}")
        if not added:
            return
        chunks.extend(extra)
        sources.extend(new_sources)
        titles = sorted({str(c.metadata.get("cpg_title", "?")) for c in extra})
        diagnostic.second_pass_note = (
            f"Second pass for {primary}: {len(added)} item(s) added for elements the first answer "
            f"missed, from {'; '.join(titles)} - " + "; ".join(added[:8])
            + ". Every added item went through the same checks as the rest.")
        log.info("SECOND PASS  %s", diagnostic.second_pass_note)

    # ============================================================== MODE B
    def inquire(self, req: InquiryRequest) -> InquiryResponse:
        with gpu_queue.slot() as ticket:
            result = self._inquire(req)
        result.provenance.queue_wait_ms = ticket.waited_ms
        return result

    def _inquire(self, req: InquiryRequest) -> InquiryResponse:
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
        
        markdown = _generate(self, prompt, config.MAX_TOKENS_INQUIRY, system=INQUIRY_SYSTEM).strip()
        
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