"""Malaysian Triage Scale 2022 (Appendix 8, New Revised Version) - the
discriminator tables transcribed as data.

Why this file exists
--------------------
The MTS discriminator grid is a TABLE. The same principle that sends FUKKM to a
name lookup instead of a vector search applies here: a table is queried, not
embedded. `MTS 2022.pdf` is 0.3% of the index and its page 7 vital-signs table
survives ingestion as four level-columns concatenated with the same separator
that divides bullets *within* a column - nothing recoverable says "No Pain" is
Level 5. So the grid lives here, in code, where it can be read and audited.

Every rule carries the VERBATIM cell text it came from and the page it is
printed on. Where this module deviates from or extends the printed table, the
rule carries a `note` and a `status`, and is listed by `deviations()`. Run

    python -m app.mts_table

to print the full review table, and

    python -m app.mts_table --audit

to re-prove that the transcribed paediatric bands leave no value unclassified.

Sources, all from MTS 2022 Appendix 8:
  page 4  CRITICAL FIRST LOOK / RAPID ASSESSMENT   (Levels 1-3, adult)
  page 5  RAPID ASSESSMENT continued                (bleeding)
  page 6  the triage-away pathway
  page 7  VITAL SIGNS (ADULT)                       (Levels 2-5)
  page 13 paediatric triage-away policy
  page 14 VITAL SIGNS (PAEDIATRICS)                 (Levels 1-4 only)
  pages 15-18 COMPLAINTS LIST (PAEDIATRICS)         (Levels 1-4, header printed)
  page 18 the MyTriage App note                     (Levels 2-5, all ages)

One threshold is not MTS's to set. Whether a 0 - 10 pain score can be obtained
from a child at all, and by which instrument, comes from PAED_PAIN_SOURCE
below - Paediatric Protocols for Malaysian Hospitals, already in the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import config

MTS_SOURCE = "MTS 2022 Appendix 8 (New Revised Version)"

# MTS 2022 page 3. The single source for both - rag_engine.MTS_TIME is an alias,
# so the prompt, the report field and this module cannot state different times.
# The Level 5 standard is UNDER 90 MINUTES; the hand-written prompt block said
# 120 until 2026-09-11.
LEVEL_NAMES = {
    1: "RESUSCITATION", 2: "EMERGENCY", 3: "URGENT",
    4: "EARLY CARE", 5: "ROUTINE",
}
TIME_TO_TREATMENT = {
    1: "Immediate (0 minutes)",
    2: "under 10 minutes",
    3: "under 30 minutes",
    4: "under 60 minutes",
    5: "under 90 minutes",
}

# The date the clinical thresholds below were reviewed and signed off by the
# user. Rules marked STATUS_SIGNED_OFF are settled: they are not to be
# "tidied" toward the printed grid by a later pass without a new decision.
SIGN_OFF_DATE = "2026-09-09"

STATUS_SIGNED_OFF = "SIGNED OFF"  # a departure from the grid, reviewed and kept
STATUS_POLICY = "POLICY"  # a default the document does not supply
STATUS_CORRECTED = "CORRECTED"  # a transcription error found and fixed
STATUS_CONFLICT = "CONFLICT"  # the document disagrees with itself
STATUS_OPEN = "OPEN"  # known gap, no decision yet


# ===========================================================================
# The leaning layer
#
# Several printed cells split on a clinical qualifier the structured intake
# does not carry: "with symptoms", "No symptoms", a shock state. The split is
# in the document. The decision about WHICH WAY TO LEAN when nothing is
# recorded is not - and while that decision sat inside each rule it was
# invisible: two rules leaned in opposite directions and only a comment said
# why.
#
# So the leaning is one layer, declared once per qualifier. A Qualifier names
# the split it resolves, the positive evidence that settles it from the intake
# text, the structured vitals that settle it with no text at all, and the
# direction it leans when neither is present TOGETHER WITH the reason that
# direction is the right one. The resolution is spelled into the reason line,
# so a clinician reading an escalation can see whether the qualifier was
# DOCUMENTED, OBSERVED in the vitals, or ASSUMED by this layer.
#
# The governing principle: lean toward the direction the evidence is more
# likely to be genuinely missing from, and argue it by how often the rule
# fires. A qualifier that is reliably written down when present leans ABSENT.
# One that is dangerous to miss, cheap to act on, and already gated behind an
# abnormal number leans PRESENT.
# ===========================================================================

DOCUMENTED = "documented"
OBSERVED = "observed in the vitals"
ASSUMED = "not recorded"


@dataclass(frozen=True)
class Resolution:
    """How one qualifier came out, and on what basis."""

    present: bool
    how: str
    qualifier: str

    def tag(self) -> str:
        """The parenthetical that goes into the clinician-facing reason line."""
        if self.how == ASSUMED:
            return f"(assumed {'present' if self.present else 'absent'})"
        return f"({self.how})"


@dataclass(frozen=True)
class Qualifier:
    """One clinical split the printed table makes and the intake does not."""

    name: str
    split: str  # the printed split, so the reviewer can find it
    pattern: re.Pattern  # positive evidence in the complaint / history
    lean: bool  # the direction taken when nothing settles it
    because: str  # why that direction, argued by how often the rule fires
    from_vitals: Callable[["Ctx"], bool] | None = None
    vitals_meaning: str = ""

    def resolve(self, ctx: "Ctx", text: str) -> Resolution:
        # Measured beats written. A vital sign that establishes the qualifier
        # is stronger evidence than a word in a complaint field, and it is
        # present in far more real intakes.
        if self.from_vitals is not None:
            try:
                if self.from_vitals(ctx):
                    return Resolution(True, OBSERVED, self.name)
            except TypeError:  # a vital the test needs was not recorded
                pass
        if self.pattern.search(text or ""):
            return Resolution(True, DOCUMENTED, self.name)
        # A written NEGATIVE settles it too. Found 2026-09-30 by the evaluation
        # harness: "CBG 22, feels well" was leaned symptoms-PRESENT and raised
        # to Level 2, although page 7 prints "> 18 mmol/L no symptoms" as
        # Level 3 and the intake says, in words, that there are none. Leaning
        # is for when nothing is recorded - not for overriding what is.
        # Positive evidence is tested first, so "feels well apart from a
        # headache" still resolves present.
        if DOCUMENTED_NEGATIVE_RE.search(text or ""):
            return Resolution(False, DOCUMENTED, self.name)
        return Resolution(self.lean, ASSUMED, self.name)


# An intake that says, in words, that the patient has no symptoms.
DOCUMENTED_NEGATIVE_RE = re.compile(
    r"\b(?:feels? (?:well|fine|ok|okay)|feeling (?:well|fine)|asymptomatic|symptom[- ]?free"
    r"|no (?:other )?(?:complaints?|symptoms?)|otherwise well|well in himself|well in herself)\b",
    re.IGNORECASE,
)


# Every alternative below is either an exact word or carries an explicit \w*.
# The pattern used to be written `\b(headache|...|confus|...)\b`, which
# silently matched nothing for the prefix alternatives - `\bconfus\b` cannot
# match "confused", and `\bvomit\b` cannot match "vomiting". A hypertensive
# patient with documented confusion was therefore never escalated. Fixed
# 2026-09-09; the trailing \b is kept so `fit\b` does not match "fitness".
HTN_SYMPTOM_RE = re.compile(
    r"\b(?:headache\w*|chest pain|blurred vision|visual disturbance\w*|vision"
    r"|shortness of breath|breathless\w*|dyspnoea|dyspnea|palpitation\w*"
    r"|slurred speech|weakness|numbness|seizure\w*|fit|fits|confus\w*"
    r"|epistaxis|nose ?bleed\w*|vomit\w*|nausea|dizz\w*|giddy|giddiness"
    r"|collapse\w*)\b",
    re.I,
)

GLUCOSE_SYMPTOM_RE = re.compile(
    r"\b(?:sweat\w*|diaphore\w*|tremor\w*|shak\w*|palpitation\w*|confus\w*"
    r"|drowsy|drowsiness|lethargic|lethargy|unconscious\w*|unresponsive"
    r"|seizure\w*|fit|fits|collapse\w*|dizz\w*|giddy|giddiness"
    r"|polyuria|polydipsia|thirst\w*|vomit\w*|abdominal pain|kussmaul"
    r"|hyperventil\w*)\b",
    re.I,
)

# Page 4's SHOCK STATE and APPEARANCE columns, Level 1: the signs that separate
# a shock state from a bare low reading. Deliberately does NOT include
# "confused" or "weak pulses" - page 4 prints those in the Level 2 column, and
# the signed-off ruling of 2026-09-09 puts a hypotensive but perfusing patient
# at Level 2.
SHOCK_SIGN_RE = re.compile(
    r"cardiac arrest|absent radial pulse|pulseless|unrecordable"
    r"|unobtainable (?:blood )?pressure|cold (?:and )?clammy|clammy and cold"
    r"|cold peripher|mottl|cyanos|cyanotic|thready"
    r"|(?:crt|capillary refill)[^0-9]{0,14}(?:[5-9]|1[0-9])"
    r"|major trauma|unresponsive|obtunded|anuri|oliguri"
    r"|decompensat|peri-?arrest|\bin shock\b|\bshocked\b",
    re.I,
)

# Symptomatic bradycardia. Syncope, presyncope, chest pain and breathlessness
# are what a patient with a genuinely inadequate rate reports; an athletic or
# beta-blocked patient at 48 reports none of them.
BRADY_SYMPTOM_RE = re.compile(
    r"\b(?:syncope|syncopal|faint\w*|near-?faint\w*|pre-?syncop\w*"
    r"|black(?:ed)? out|collapse\w*|dizz\w*|giddy|giddiness|light-?head\w*"
    r"|chest pain|breathless\w*|dyspnoea|dyspnea|shortness of breath"
    r"|confus\w*|drowsy|drowsiness|palpitation\w*)\b",
    re.I,
)


QUALIFIERS: dict[str, Qualifier] = {
    q.name: q
    for q in [
        Qualifier(
            name="htn_symptoms",
            split="page 7 splits BP > 220/130 and BP > 180/110 on "
                  "\"with symptoms\" / \"Mild symptoms\" / \"No symptoms\"",
            pattern=HTN_SYMPTOM_RE,
            lean=False,
            because="Asymptomatic hypertension is the single commonest abnormal "
                    "reading in a Malaysian ED or Klinik Kesihatan. Leaning "
                    "present would escalate a large share of all attendances, "
                    "which is how alarm fatigue is built - and hypertensive "
                    "symptoms are usually the reason the patient came, so they "
                    "are the kind of thing that does get written down. Leaning "
                    "absent still lands the patient at Level 3 or 4, never 5.",
        ),
        Qualifier(
            name="glucose_symptoms",
            split="page 7 splits CBG < 2.5 and CBG > 18 on "
                  "\"and symptoms\" / \"no symptoms\"",
            pattern=GLUCOSE_SYMPTOM_RE,
            lean=True,
            because="A CBG below 2.5 or above 18 is dangerous whether or not "
                    "anyone wrote a symptom down. The rule fires only behind an "
                    "already-abnormal number, so leaning present costs one level "
                    "on a rare reading, and leaning absent costs a missed "
                    "hypoglycaemic. It also preserves the behaviour the guardrail "
                    "had before the table was transcribed.",
        ),
        Qualifier(
            name="shock_signs",
            split="page 7 prints SBP < 90 at LEVEL 2; page 4 reserves LEVEL 1 "
                  "for \"Major Trauma in Shock\", \"Absent Radial Pulse\", "
                  "\"Pale, cyanosed, cold peripheries\" and \"Severe "
                  "Tachycardia / Bradycardia\"",
            pattern=SHOCK_SIGN_RE,
            lean=False,
            because="Hypotension alone is common in the well elderly, the "
                    "dehydrated and the small-framed, and page 7 puts it at "
                    "Level 2. Leaning present would make every low reading a "
                    "resuscitation call. The findings that actually separate the "
                    "two - a profoundly low pressure, tachycardia, bradycardia, "
                    "hypoxia - are MEASURED, so in almost every real intake this "
                    "qualifier is settled from the vitals and never leaned at all.",
            from_vitals=lambda c: (
                _lt(c.systolic_bp, 70)
                or (
                    _lt(c.systolic_bp, 90)
                    and (
                        _gt(c.heart_rate, 120)
                        or _lt(c.heart_rate, 50)
                        or _lt(c.spo2, 92)
                    )
                )
            ),
            vitals_meaning="SBP < 70, or SBP < 90 together with HR > 120 "
                           "(page 4 \"Severe Tachycardia\"), HR < 50 (page 4 "
                           "\"Bradycardia\") or SpO2 < 92% (page 4 respiratory "
                           "distress). GCS is deliberately absent: page 4 prints "
                           "\"Confused\" in the LEVEL 2 column, and GCS has its "
                           "own rules.",
        ),
        Qualifier(
            name="bradycardia_symptoms",
            split="nothing in MTS 2022 prints a 40 - 50 bradycardia band; "
                  "page 4's SHOCK STATE row prints \"Bradycardia\" at LEVEL 1 "
                  "and \"Cannot stand / walk unsupported\" at LEVEL 3",
            pattern=BRADY_SYMPTOM_RE,
            lean=False,
            because="This is the clearest cry-wolf risk in the module. HR 40 - 50 "
                    "is not a printed MTS band at all; it is carried over from the "
                    "previous _mts_floor, and an athletic or beta-blocked patient "
                    "sits there all day. Escalating on the number alone escalates "
                    "those patients on nothing else. Symptomatic bradycardia is a "
                    "different patient, and the symptoms are exactly what such a "
                    "patient reports, so leaning absent loses very little.",
            from_vitals=lambda c: _lt(c.systolic_bp, 90) or _lt(c.gcs, 15),
            vitals_meaning="SBP < 90 or GCS < 15 alongside the low rate - "
                           "haemodynamic or cerebral consequence, measured rather "
                           "than reported.",
        ),
    ]
}


# ===========================================================================
# PRIMARY TRIAGE and INITIAL TESTS - the structured observations
#
# Page 7 states that the final level "takes into consideration all of the
# selected parameters ie. Primary Triage, Vital Signs, Complaints List and
# Initial Tests". Until 2026-09-10 this module read two of those four: the
# vital signs, and the complaints list through free text. Everything the
# PRIMARY TRIAGE OFFICER sees on pages 4 - 5 - the Critical First Look and the
# Rapid Assessment of breathing, perfusion and bleeding - and the ECG half of
# page 7's Initial Tests had no field to arrive in, so every cell in them sat
# in UNIMPLEMENTED_ADULT_CELLS and a patient matching only those cells was
# under-triaged.
#
# These vocabularies are that missing field set. Each value is one printed
# cell, or a group of cells printed in the SAME column of the SAME row - the
# only bundling that cannot change a level. Values are plain strings, not
# enums, so this module keeps its single dependency on `config`; the API enums
# in schemas.py are built from the same literals and
# tests/test_triage_modifiers.py asserts the two sets are equal, so a value
# renamed on one side cannot silently stop matching on the other.
#
# EVERY FIELD IS OPTIONAL. An unset observation contributes nothing - it never
# asserts the reassuring value. That is the same contract the vitals already
# have, and it is why adding these fields cannot lower an existing patient's
# level.
#
# What is deliberately NOT here matters as much. Page 4 prints "Peripheries
# warm, CRT normal" and "Not fully conscious" in its LEVEL 3 column, and page
# 7 prints "No documented fever" in its LEVEL 4 column. Read as discriminators
# those would escalate every carefully assessed well patient to Level 3 or 4.
# See deviations() for each one and the reading taken instead.
# ===========================================================================

# ---- APPEARANCE: page 4 CRITICAL FIRST LOOK + page 7's two impression cells.
# A global impression is not additive, so this is one value, not a set.
APPEARANCE_NOT_DISTRESSED = "walking_talking_not_distressed"
APPEARANCE_UNWELL = "appears_unwell"
APPEARANCE_CANNOT_SIT_STAND = "cannot_sit_or_stand_unsupported"
APPEARANCE_NOT_RESPONDING = "not_responding_to_call"
APPEARANCE_SEPTIC_ILL = "appears_septic_or_critically_ill"
APPEARANCE_VALUES: tuple[str, ...] = (
    APPEARANCE_NOT_DISTRESSED, APPEARANCE_UNWELL, APPEARANCE_CANNOT_SIT_STAND,
    APPEARANCE_NOT_RESPONDING, APPEARANCE_SEPTIC_ILL,
)

# ---- BREATHING: page 4 RAPID ASSESSMENT / RESPIRATORY DISTRESS.
# The speech-and-effort ladder. SpO2 is NOT here - it is already a vital and
# already has rules from both page 4 and page 7.
BREATHING_NOT_BREATHLESS = "not_breathless"
BREATHING_WHEEZE_AIRWAY_INTACT = "wheeze_expiratory_rhonchi_airway_intact"
BREATHING_NEEDS_OXYGEN = "needs_oxygen_support"
BREATHING_SHORT_PHRASES = "difficulty_breathing_short_phrases_only"
BREATHING_ABNORMAL_SOUNDS = "abnormal_airway_sounds"
BREATHING_EXCESSIVE_WORK = "excessive_work_of_breathing_sweating"
BREATHING_ONE_WORD = "cannot_speak_one_word_reply"
BREATHING_ASSISTED = "requires_assisted_breathing"
BREATHING_VALUES: tuple[str, ...] = (
    BREATHING_NOT_BREATHLESS, BREATHING_WHEEZE_AIRWAY_INTACT,
    BREATHING_NEEDS_OXYGEN, BREATHING_SHORT_PHRASES,
    BREATHING_ABNORMAL_SOUNDS, BREATHING_EXCESSIVE_WORK,
    BREATHING_ONE_WORD, BREATHING_ASSISTED,
)

# ---- PERFUSION: page 4 RAPID ASSESSMENT / SHOCK STATE.
# This is the field the paediatric path most needed. A child compensates by
# vasoconstricting long before the blood pressure moves, so page 14's SBP
# bands are a LATE sign and capillary refill is the early one.
PERFUSION_WARM_NORMAL = "warm_pink_pulses_normal"
PERFUSION_CRT_OVER_2 = "crt_over_2_seconds"
PERFUSION_WEAK_PULSES = "tachycardia_weak_pulses"
PERFUSION_COLD_OR_CYANOSED = "pale_cyanosed_cold_peripheries"
PERFUSION_ABSENT_RADIAL = "absent_radial_pulse"
PERFUSION_VALUES: tuple[str, ...] = (
    PERFUSION_WARM_NORMAL, PERFUSION_CRT_OVER_2, PERFUSION_WEAK_PULSES,
    PERFUSION_COLD_OR_CYANOSED, PERFUSION_ABSENT_RADIAL,
)

# ---- BLEEDING: page 5 RAPID ASSESSMENT / BLEEDING, a whole printed row.
BLEEDING_NONE = "minimal_or_no_active_bleeding"
BLEEDING_WOUND_OR_ENT = "bleeding_from_fracture_joint_wound_ent_or_menorrhagia"
BLEEDING_HAEMATOMA_OR_DISORDER = "expanding_haematoma_or_bleeding_disorder"
BLEEDING_VOMIT_OR_COUGH_BLOOD = "active_vomiting_or_coughing_blood"
BLEEDING_SUSPECTED_INTERNAL = "suspected_internal_bleeding_ectopic_or_aaa"
BLEEDING_ARTERIAL_OR_UNCONTROLLED = "arterial_uncontrolled_or_massive_bleeding"
BLEEDING_VALUES: tuple[str, ...] = (
    BLEEDING_NONE, BLEEDING_WOUND_OR_ENT, BLEEDING_HAEMATOMA_OR_DISORDER,
    BLEEDING_VOMIT_OR_COUGH_BLOOD, BLEEDING_SUSPECTED_INTERNAL,
    BLEEDING_ARTERIAL_OR_UNCONTROLLED,
)

# ---- ECG: page 7's INITIAL TESTS row, all eleven printed findings.
# A set, not one value: one tracing carries several. An EMPTY set means no ECG
# was done and asserts nothing - "Normal ECG" is a value you must choose.
ECG_NORMAL = "normal_ecg"
ECG_NO_ST_T_CHANGES = "no_st_t_wave_changes"
ECG_NO_FINDINGS_ONGOING_PAIN = "no_ecg_findings_continuing_chest_pain"
ECG_AF_OVER_100 = "atrial_fibrillation_over_100"
ECG_FREQUENT_ECTOPICS = "frequent_ectopics"
ECG_BLOCK_OR_PAUSE = "blocks_or_sinus_pauses"
ECG_TALL_TENTED_T = "tall_tented_t_waves"
ECG_ST_ELEVATION_OR_DEPRESSION = "st_elevations_or_depressions"
ECG_WIDE_COMPLEX_TACHY = "wide_complex_tachycardia"
ECG_NARROW_COMPLEX_TACHY_OVER_150 = "narrow_complex_tachycardia_over_150"
ECG_VALUES: tuple[str, ...] = (
    ECG_NORMAL, ECG_NO_ST_T_CHANGES, ECG_NO_FINDINGS_ONGOING_PAIN,
    ECG_AF_OVER_100, ECG_FREQUENT_ECTOPICS, ECG_BLOCK_OR_PAUSE,
    ECG_TALL_TENTED_T, ECG_ST_ELEVATION_OR_DEPRESSION,
    ECG_WIDE_COMPLEX_TACHY, ECG_NARROW_COMPLEX_TACHY_OVER_150,
)

# ---- FEVER HISTORY: page 7's Temp row, Level 4 and Level 5 columns.
FEVER_REPORTED = "fever_reported_before_arrival"
FEVER_NONE_REPORTED = "no_fever_reported"
FEVER_HISTORY_VALUES: tuple[str, ...] = (FEVER_REPORTED, FEVER_NONE_REPORTED)

# ---- COMORBIDITIES. Only one of these is a printed MTS cell
# ("Immunocompromised", page 7 Level 2). The rest carry no MTS rule at all and
# exist so the drug guardrails in contraindications.py fire on a recorded fact
# instead of on whether the word happened to be typed into the history box.
# Anticoagulant therapy is the pointed example: page 5 lists it under BLEEDING
# "CHECK FOR", NOT as a level cell, so it must not escalate on its own - every
# warfarin patient would become Level 3.
COMORBID_IMMUNOCOMPROMISED = "immunocompromised"
COMORBID_ANTICOAGULANT = "on_anticoagulant"
COMORBID_CKD = "chronic_kidney_disease_or_dialysis"
COMORBID_AIRWAY_DISEASE = "copd_or_asthma"
COMORBID_DIABETES = "diabetes"
COMORBID_LIVER_DISEASE = "liver_disease"
COMORBIDITY_VALUES: tuple[str, ...] = (
    COMORBID_IMMUNOCOMPROMISED, COMORBID_ANTICOAGULANT, COMORBID_CKD,
    COMORBID_AIRWAY_DISEASE, COMORBID_DIABETES, COMORBID_LIVER_DISEASE,
)

# ---- PAEDIATRIC ASSESSMENT TRIANGLE: page 13, for the under-12s.
# Page 13's own sentence is the reason this belongs in the floor: "Children
# without danger signs identified by the Paediatric Assessment Triangle at
# Primary Triage should proceed to Registration and Secondary Triage." The
# danger signs ARE a triage decision, and page 14's vital-sign bands do not
# contain them. Cells already covered by PERFUSION (cold peripheries, CRT) and
# by the RR band (tachypnoea alone) are not repeated here.
PAED_SNORING_MUFFLED = "snoring_muffled_or_hoarse_speech"
PAED_STRIDOR_GRUNTING = "stridor_grunting_or_wheezing"
PAED_SNIFFING_TRIPOD = "sniffing_or_tripod_position"
PAED_HEAD_BOBBING = "head_bobbing"
PAED_MOTTLING_CYANOSIS = "patchy_or_bluish_skin_discolouration"
PAED_DIFFICULTY_SWALLOWING = "difficulty_in_swallowing"
PAED_DROOLING = "drooling"
PAED_POSITIONAL_DISTRESS = "unable_to_walk_or_refusal_to_lie_down"
PAED_RETRACTIONS = "supraclavicular_intercostal_or_substernal_retractions"
PAED_FLARING = "nasal_flaring_or_accessory_muscles"
PAED_PALLOR = "pale_mucous_membranes_sole_or_palm"
PAEDIATRIC_SIGN_VALUES: tuple[str, ...] = (
    PAED_SNORING_MUFFLED, PAED_STRIDOR_GRUNTING, PAED_SNIFFING_TRIPOD,
    PAED_HEAD_BOBBING, PAED_MOTTLING_CYANOSIS, PAED_DIFFICULTY_SWALLOWING,
    PAED_DROOLING, PAED_POSITIONAL_DISTRESS, PAED_RETRACTIONS,
    PAED_FLARING, PAED_PALLOR,
)


@dataclass(frozen=True)
class Observations:
    """The structured Primary Triage and Initial Test findings, all optional.

    Duck-typed the same way `vitals` is: rag_engine builds one from the
    TriageRequest and this module never imports the schema.
    """

    appearance: str | None = None
    breathing: str | None = None
    perfusion: str | None = None
    bleeding: str | None = None
    fever_history: str | None = None
    ecg: frozenset[str] = frozenset()
    comorbidities: frozenset[str] = frozenset()
    paediatric_signs: frozenset[str] = frozenset()


NO_OBSERVATIONS = Observations()


# ===========================================================================
# Evaluation context
# ===========================================================================


@dataclass
class Ctx:
    """Everything a rule may read. Derived facts are resolved here, once, and
    spelled out - a rule never re-reads free text."""

    age: float
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    heart_rate: float | None = None
    respiratory_rate: float | None = None
    temperature: float | None = None
    spo2: float | None = None
    gcs: int | None = None
    glucose: float | None = None
    pain_score: int | None = None
    # Primary Triage and Initial Tests. Every one is optional; an unset value
    # matches no rule and never asserts the reassuring finding.
    obs: Observations = NO_OBSERVATIONS
    # Filled by _resolve_qualifiers, one entry per QUALIFIERS key.
    qualifiers: dict[str, Resolution] = field(default_factory=dict)

    @property
    def is_paediatric(self) -> bool:
        return self.age < config.PAEDIATRIC_AGE_YEARS

    @property
    def has_measured_fever(self) -> bool:
        """A temperature was taken AND it is at or above page 7's lowest
        printed fever band ("Temp 37.5 - 39 C", Level 3). Distinct from "no
        fever": an unmeasured temperature is not a normal one."""
        return self.temperature is not None and self.temperature >= 37.5

    def is_(self, field_name: str, value: str) -> bool:
        """Does a single-valued observation hold this printed cell?"""
        return getattr(self.obs, field_name) == value

    def any_(self, field_name: str, *values: str) -> bool:
        """Does a multi-valued observation carry any of these printed cells?"""
        return bool(getattr(self.obs, field_name) & frozenset(values))

    def q(self, name: str) -> bool:
        """Is the qualifier present, however it was settled?"""
        return self.qualifiers[name].present

    def tag(self, name: str) -> str:
        """The basis on which it was settled, for the reason line."""
        return self.qualifiers[name].tag()


def _resolve_qualifiers(ctx: Ctx, text: str) -> Ctx:
    ctx.qualifiers = {name: q.resolve(ctx, text) for name, q in QUALIFIERS.items()}
    return ctx


@dataclass(frozen=True)
class Rule:
    """One cell of the printed grid."""

    level: int
    parameter: str
    verbatim: str  # the cell text, exactly as printed
    page: int
    test: Callable[[Ctx], bool]
    describe: Callable[[Ctx], str]
    note: str = ""  # non-empty => this rule deviates from or extends the source
    status: str = ""  # one of the STATUS_* constants, when note is set
    # How this cell reads in the model prompt, when `verbatim` would mislead
    # there. `verbatim` is the audit record and must stay exactly as printed:
    # the Pain Level 4 cell is recorded as "No Pain (Level 5 column)" because
    # that is the cell it was DERIVED from, but a prompt line reading
    # "LEVEL 4 - Pain: No Pain" teaches the opposite of what the rule tests.
    # Empty means verbatim is already the right wording. See prompt_rules().
    prompt_text: str = ""

    def evaluate(self, ctx: Ctx) -> "Finding | None":
        try:
            if not self.test(ctx):
                return None
            observed = self.describe(ctx)
        except TypeError:  # a vital this rule needs was not recorded
            return None
        return Finding(
            level=self.level,
            parameter=self.parameter,
            verbatim=self.verbatim,
            page=self.page,
            observed=observed,
            note=self.note,
        )


@dataclass(frozen=True)
class Finding:
    level: int
    parameter: str
    verbatim: str
    page: int
    observed: str
    note: str = ""

    def reason(self) -> str:
        """The clinician-facing line. Names the observation and the MTS cell it
        matched, so the escalation can be checked against the appendix."""
        return f"{self.observed} - MTS p{self.page} L{self.level} \"{self.verbatim}\""


# ---------------------------------------------------------------- helpers
def _between(value: float | None, lo: float, hi: float) -> bool:
    """Printed ranges such as "37.5 - 39 C" read as inclusive at both ends.
    Boundary overlaps between adjacent columns are intentional: the floor takes
    the most acute match, which is page 7's own tie-break rule -
    "if modifiers do not differentiate between two triage levels, the higher
    triage level modifier should be selected"."""
    return value is not None and lo <= value <= hi


def _gt(value: float | None, limit: float) -> bool:
    """A printed "> X" cell. Exclusive, as printed."""
    return value is not None and value > limit


def _lt(value: float | None, limit: float) -> bool:
    """A printed "< X" cell. Exclusive, as printed."""
    return value is not None and value < limit


# ===========================================================================
# ADULT - page 7 VITAL SIGNS (ADULT), with the Level 1 cells page 7 does not
# carry taken from page 4 RAPID ASSESSMENT.
# ===========================================================================

ADULT_RULES: list[Rule] = [
    # ------------------------------------------------------------ BP
    #
    # The two hypotension rules are one printed cell read as two. Page 7 prints
    # "SBP < 90" at LEVEL 2 and page 4 prints a shock state at LEVEL 1; which
    # applies depends on whether the low pressure comes with decompensation.
    # Signed off 2026-09-09: Level 1 when there are signs of decompensated
    # shock or organ hypoperfusion, Level 2 when the patient is high-risk,
    # confused or unstable but still breathing and perfusing.
    Rule(
        1, "BP", "Major Trauma in Shock / Absent Radial Pulse / Pale, cyanosed, "
                 "cold peripheries / Severe Tachycardia / Bradycardia", 4,
        lambda c: _lt(c.systolic_bp, 90) and c.q("shock_signs"),
        lambda c: f"systolic BP {c.systolic_bp:g} mmHg with signs of "
                  f"decompensated shock {c.tag('shock_signs')}",
        note="Page 7 prints SBP < 90 as LEVEL 2, without qualification. Level 1 "
             "is taken from page 4's SHOCK STATE column and applies only when "
             "the hypotension is accompanied by decompensation or organ "
             "hypoperfusion. Until 2026-09-09 every SBP < 90 returned Level 1; "
             "narrowing it keeps the shocked-STEMI escalation (SBP 78, HR 124) "
             "and stops an isolated low reading from being a resuscitation call. "
             "See the shock_signs qualifier for what counts.",
        status=STATUS_SIGNED_OFF,
    ),
    Rule(
        2, "BP", "SBP < 90", 7,
        lambda c: _lt(c.systolic_bp, 90) and not c.q("shock_signs"),
        lambda c: f"systolic BP {c.systolic_bp:g} mmHg, no signs of "
                  f"decompensated shock {c.tag('shock_signs')}",
    ),
    Rule(
        2, "BP", "BP > 220/130 with symptoms", 7,
        lambda c: (_gt(c.systolic_bp, 220) or _gt(c.diastolic_bp, 130)) and c.q("htn_symptoms"),
        lambda c: f"BP {c.systolic_bp:g}/{c.diastolic_bp or 0:g} mmHg with "
                  f"symptoms {c.tag('htn_symptoms')}",
    ),
    Rule(
        3, "BP", "BP > 220/130 No symptoms", 7,
        lambda c: (_gt(c.systolic_bp, 220) or _gt(c.diastolic_bp, 130)) and not c.q("htn_symptoms"),
        lambda c: f"BP {c.systolic_bp:g}/{c.diastolic_bp or 0:g} mmHg, no "
                  f"symptoms {c.tag('htn_symptoms')}",
    ),
    Rule(
        3, "BP", "BP > 180/110 Mild symptoms", 7,
        lambda c: (_gt(c.systolic_bp, 180) or _gt(c.diastolic_bp, 110)) and c.q("htn_symptoms"),
        lambda c: f"BP {c.systolic_bp:g}/{c.diastolic_bp or 0:g} mmHg with "
                  f"symptoms {c.tag('htn_symptoms')}",
    ),
    Rule(
        4, "BP", "BP > 180/110 No symptoms", 7,
        lambda c: (_gt(c.systolic_bp, 180) or _gt(c.diastolic_bp, 110)) and not c.q("htn_symptoms"),
        lambda c: f"BP {c.systolic_bp:g}/{c.diastolic_bp or 0:g} mmHg, no "
                  f"symptoms {c.tag('htn_symptoms')}",
    ),
    # ------------------------------------------------------------ HR
    Rule(
        2, "HR", "HR > 120", 7,
        lambda c: _gt(c.heart_rate, 120),
        lambda c: f"heart rate {c.heart_rate:g} bpm",
    ),
    Rule(
        3, "HR", "HR 100 - 120", 7,
        lambda c: _between(c.heart_rate, 100, 120),
        lambda c: f"heart rate {c.heart_rate:g} bpm",
    ),
    Rule(
        2, "HR", "Bradycardia < 40 (ECG row)", 7,
        lambda c: _lt(c.heart_rate, 40),
        lambda c: f"heart rate {c.heart_rate:g} bpm (bradycardia)",
    ),
    Rule(
        3, "HR", "Cannot stand / walk unsupported (SHOCK STATE, L3)", 4,
        lambda c: _between(c.heart_rate, 40, 50) and c.q("bradycardia_symptoms"),
        lambda c: f"heart rate {c.heart_rate:g} bpm with symptoms of inadequate "
                  f"rate {c.tag('bradycardia_symptoms')}",
        note="MTS 2022 prints no 40 - 50 bradycardia band anywhere; the band is "
             "carried over from the previous _mts_floor, which escalated every "
             "HR <= 50 to Level 3. Narrowed 2026-09-09 to SYMPTOMATIC "
             "bradycardia, so a beta-blocked or athletic patient at 48 with "
             "nothing else wrong is no longer escalated on the number alone. "
             "The level is unchanged; only how often it fires. A bradycardic "
             "patient who is genuinely shocked is caught by the LEVEL 1 "
             "hypotension rule instead, which reads HR < 50 as page 4's "
             "\"Bradycardia\" in the shock column.",
        status=STATUS_SIGNED_OFF,
        prompt_text="HR 40 - 50 WITH symptoms of an inadequate rate",
    ),
    # ------------------------------------------------------------ RR
    Rule(
        2, "RR", "RR > 30", 7,
        lambda c: _gt(c.respiratory_rate, 30),
        lambda c: f"respiratory rate {c.respiratory_rate:g} /min",
    ),
    Rule(
        3, "RR", "RR 20 - 30", 7,
        lambda c: _between(c.respiratory_rate, 20, 30),
        lambda c: f"respiratory rate {c.respiratory_rate:g} /min",
    ),
    # ------------------------------------------------------------ SpO2
    Rule(
        1, "SpO2", "SpO2 < 90% room air", 4,
        lambda c: _lt(c.spo2, 90),
        lambda c: f"SpO2 {c.spo2:g}% (room air)",
    ),
    Rule(
        2, "SpO2", "SpO2 90 - 92% room air", 4,
        lambda c: _between(c.spo2, 90, 92),
        lambda c: f"SpO2 {c.spo2:g}% (room air)",
    ),
    Rule(
        2, "SpO2", "SpO2 < 92%", 7,
        lambda c: _lt(c.spo2, 92),
        lambda c: f"SpO2 {c.spo2:g}%",
    ),
    Rule(
        3, "SpO2", "SpO2 92 - 94%", 7,
        lambda c: _between(c.spo2, 92, 94),
        lambda c: f"SpO2 {c.spo2:g}%",
    ),
    # ------------------------------------------------------------ Temp
    Rule(
        2, "Temp", "Temp > 39 or < 36", 7,
        lambda c: _gt(c.temperature, 39) or _lt(c.temperature, 36),
        lambda c: f"temperature {c.temperature:g} C",
    ),
    Rule(
        3, "Temp", "Temp 37.5 - 39 C", 7,
        lambda c: _between(c.temperature, 37.5, 39),
        lambda c: f"temperature {c.temperature:g} C",
    ),
    # ------------------------------------------------------------ GCS
    Rule(
        1, "GCS", "Unresponsive / Airway unprotected", 4,
        lambda c: _lt(c.gcs, 9),
        lambda c: f"GCS {c.gcs}",
        note="Page 4 states Level 1 as \"Unresponsive\", without a GCS number. "
             "GCS < 9 is this module's mapping of that cell - the conventional "
             "threshold at which the airway is no longer reliably protected.",
        status=STATUS_SIGNED_OFF,
    ),
    Rule(
        2, "GCS", "GCS < 13 or drop > 2", 7,
        lambda c: _lt(c.gcs, 13),
        lambda c: f"GCS {c.gcs}",
    ),
    Rule(
        2, "GCS", "GCS 13", 7,
        lambda c: c.gcs == 13,
        lambda c: f"GCS {c.gcs}",
        note="Page 7 prints \"GCS < 13\", which excludes 13. Carried over from "
             "the previous _mts_floor, which escalated GCS <= 13 to Level 2, "
             "and kept: a GCS of 13 is a measurable deficit, and escalating it "
             "one level is the safer direction on a value this close to the "
             "printed cut-off.",
        status=STATUS_SIGNED_OFF,
    ),
    Rule(
        3, "GCS", "Altered mental state", 4,
        lambda c: c.gcs == 14,
        lambda c: f"GCS {c.gcs}",
        note="Not a printed GCS threshold. Carried over from the previous "
             "_mts_floor and mapped onto page 4's Level 3 \"Altered mental "
             "state\", which is the cell a GCS of 14 describes.",
        status=STATUS_SIGNED_OFF,
    ),
    # ------------------------------------------------------------ Pain
    Rule(
        2, "Pain", "Severe Pain (8 - 10)", 7,
        lambda c: _between(c.pain_score, 8, 10),
        lambda c: f"pain score {c.pain_score}/10 (severe)",
    ),
    Rule(
        3, "Pain", "Pain Score 4 - 7", 7,
        lambda c: _between(c.pain_score, 4, 7),
        lambda c: f"pain score {c.pain_score}/10 (moderate)",
    ),
    Rule(
        4, "Pain", "No Pain (Level 5 column)", 7,
        lambda c: _between(c.pain_score, 1, 3),
        lambda c: f"pain score {c.pain_score}/10 (mild)",
        note="Page 7 prints no adult cell for a pain score of 1 - 3: Level 5 "
             "requires literally \"No Pain\" and Level 3 starts at 4. Any pain "
             "therefore excludes Level 5, and 1 - 3 falls below Level 3, "
             "leaving Level 4. The paediatric table prints this cell "
             "explicitly as \"Little pain\", which is the corroboration. This "
             "is why a negative-control test must use pain_score: 0.",
        status=STATUS_SIGNED_OFF,
        prompt_text="Pain Score 1 - 3 (any pain below the Level 3 band)",
    ),
    # ------------------------------------------------------------ Glucose
    Rule(
        2, "Glucose", "< 2.5 mmol/L and symptoms", 7,
        lambda c: _lt(c.glucose, 2.5) and c.q("glucose_symptoms"),
        lambda c: f"CBG {c.glucose:g} mmol/L with symptoms "
                  f"{c.tag('glucose_symptoms')}",
    ),
    Rule(
        3, "Glucose", "< 2.5 mmol/L no symptoms", 7,
        lambda c: _lt(c.glucose, 2.5) and not c.q("glucose_symptoms"),
        lambda c: f"CBG {c.glucose:g} mmol/L, no symptoms "
                  f"{c.tag('glucose_symptoms')}",
    ),
    Rule(
        2, "Glucose", "> 18 mmol/L and symptoms", 7,
        lambda c: _gt(c.glucose, 18) and c.q("glucose_symptoms"),
        lambda c: f"CBG {c.glucose:g} mmol/L with symptoms "
                  f"{c.tag('glucose_symptoms')}",
    ),
    Rule(
        3, "Glucose", "> 18 mmol/L no symptoms", 7,
        lambda c: _gt(c.glucose, 18) and not c.q("glucose_symptoms"),
        lambda c: f"CBG {c.glucose:g} mmol/L, no symptoms "
                  f"{c.tag('glucose_symptoms')}",
    ),
    Rule(
        4, "Glucose", "2.5 - 4.0 mmol/L", 7,
        lambda c: _between(c.glucose, 2.5, 4.0),
        lambda c: f"CBG {c.glucose:g} mmol/L",
    ),
    Rule(
        4, "Glucose", "12 - 18 mmol/L", 7,
        lambda c: _between(c.glucose, 12, 18),
        lambda c: f"CBG {c.glucose:g} mmol/L",
    ),
    # ---------------------------------------------------- Appearance (p4, p7)
    #
    # Page 4's CRITICAL FIRST LOOK and page 7's two global-impression cells.
    # Column membership below was read from the PDF's word coordinates, not
    # from the extracted text: the text layer emits each column top-to-bottom
    # and then the next, so a cell printed low in the Level 2 column reads as
    # though it followed the Level 3 items. That is how "Not responding to
    # call" (page 4, x=246 = LEVEL 2) had been noted as a Level 1 finding.
    Rule(
        2, "Appearance", "Appears Septic, Ill", 7,
        lambda c: c.is_("appearance", APPEARANCE_SEPTIC_ILL),
        lambda c: "triage impression: appears septic or critically ill",
    ),
    Rule(
        2, "Appearance", "Not responding to call", 4,
        lambda c: c.is_("appearance", APPEARANCE_NOT_RESPONDING),
        lambda c: "triage impression: not responding to call",
    ),
    Rule(
        3, "Appearance", "Appears unwell", 7,
        lambda c: c.is_("appearance", APPEARANCE_UNWELL),
        lambda c: "triage impression: appears unwell",
    ),
    Rule(
        3, "Appearance", "Cannot sit / stand unsupported", 4,
        lambda c: c.is_("appearance", APPEARANCE_CANNOT_SIT_STAND),
        lambda c: "triage impression: cannot sit or stand unsupported",
    ),
    # ----------------------------------------------------- Breathing (p4)
    #
    # RAPID ASSESSMENT / RESPIRATORY DISTRESS. The speech-and-effort ladder is
    # what SpO2 alone misses: a compensating asthmatic holds a normal
    # saturation while speaking in short phrases, and page 4 puts that patient
    # at Level 2 on the speech cell alone.
    Rule(
        1, "Breathing", "Require assisted breathing", 4,
        lambda c: c.is_("breathing", BREATHING_ASSISTED),
        lambda c: "requires assisted breathing",
    ),
    Rule(
        1, "Breathing", "Cannot speak; one-word reply", 4,
        lambda c: c.is_("breathing", BREATHING_ONE_WORD),
        lambda c: "cannot speak, one-word replies only",
    ),
    Rule(
        1, "Breathing", "Excessive work of breathing, sweating", 4,
        lambda c: c.is_("breathing", BREATHING_EXCESSIVE_WORK),
        lambda c: "excessive work of breathing with sweating",
    ),
    Rule(
        1, "Breathing", "Abnormal sounds", 4,
        lambda c: c.is_("breathing", BREATHING_ABNORMAL_SOUNDS),
        lambda c: "abnormal airway sounds",
    ),
    Rule(
        2, "Breathing", "Difficulty to breathe / Short phrases only", 4,
        lambda c: c.is_("breathing", BREATHING_SHORT_PHRASES),
        lambda c: "difficulty breathing, speaking in short phrases only",
        note="Two cells printed in the SAME Level 2 column of the same page 4 "
             "row, offered as one option because a patient in short phrases is "
             "by definition finding it difficult to breathe. Bundling within "
             "one column cannot change a level.",
        status=STATUS_POLICY,
    ),
    Rule(
        3, "Breathing", "Need O2 support", 4,
        lambda c: c.is_("breathing", BREATHING_NEEDS_OXYGEN),
        lambda c: "needs oxygen support",
    ),
    Rule(
        3, "Breathing", "Wheeze, expiratory rhonchi; airway intact", 4,
        lambda c: c.is_("breathing", BREATHING_WHEEZE_AIRWAY_INTACT),
        lambda c: "wheeze or expiratory rhonchi, airway intact",
    ),
    # ----------------------------------------------------- Perfusion (p4)
    Rule(
        1, "Perfusion", "Pale, cyanosed, cold peripheries", 4,
        lambda c: c.is_("perfusion", PERFUSION_COLD_OR_CYANOSED),
        lambda c: "pale, cyanosed or cold peripheries",
    ),
    Rule(
        1, "Perfusion", "Absent Radial Pulse", 4,
        lambda c: c.is_("perfusion", PERFUSION_ABSENT_RADIAL),
        lambda c: "absent radial pulse",
    ),
    Rule(
        2, "Perfusion", "Tachycardia, Weak Pulses", 4,
        lambda c: c.is_("perfusion", PERFUSION_WEAK_PULSES),
        lambda c: "weak pulses with tachycardia",
    ),
    Rule(
        2, "Perfusion", "CRT > 2 seconds", 4,
        lambda c: c.is_("perfusion", PERFUSION_CRT_OVER_2),
        lambda c: "capillary refill time over 2 seconds",
    ),
    # ------------------------------------------------------ Bleeding (p5)
    #
    # A whole printed row that had no field at all. Each option below groups
    # only cells printed in the SAME column, so the grouping cannot move a
    # level; the verbatim names the first cell and the note names the rest.
    Rule(
        1, "Bleeding", "Arterial Limb Bleeding", 5,
        lambda c: c.is_("bleeding", BLEEDING_ARTERIAL_OR_UNCONTROLLED),
        lambda c: "arterial, uncontrolled or massive bleeding",
        note="Groups the page 5 LEVEL 1 column: \"Arterial Limb Bleeding\", "
             "\"Active uncontrolled bleeding\", \"Massive Vaginal bleed\", "
             "\"Severe Facial Injury\", \"Severe Pelvic Injury\".",
        status=STATUS_POLICY,
    ),
    Rule(
        2, "Bleeding", "Active Vomit / Cough Blood", 5,
        lambda c: c.is_("bleeding", BLEEDING_VOMIT_OR_COUGH_BLOOD),
        lambda c: "actively vomiting or coughing blood",
    ),
    Rule(
        2, "Bleeding", "Suspected Intra-Abdominal Bleeding / Ectopic / AAA", 5,
        lambda c: c.is_("bleeding", BLEEDING_SUSPECTED_INTERNAL),
        lambda c: "suspected internal bleeding",
        note="Groups the page 5 LEVEL 2 column: \"Suspected Intra-Abdominal "
             "Bleeding / Ectopic / AAA\", \"Suspected vascular injury\", "
             "\"Compartment syndrome\".",
        status=STATUS_POLICY,
    ),
    Rule(
        3, "Bleeding", "Bleeding from Fractures/ Dislocations / Joints / Wounds", 5,
        lambda c: c.is_("bleeding", BLEEDING_WOUND_OR_ENT),
        lambda c: "bleeding from a wound, fracture, joint, ENT site or menorrhagia",
        note="Groups the page 5 LEVEL 3 column: \"Bleeding from Fractures/ "
             "Dislocations / Joints / Wounds\", \"Menorrhagia\", "
             "\"ENT Bleeding\".",
        status=STATUS_POLICY,
    ),
    Rule(
        3, "Bleeding", "Expanding haematoma", 5,
        lambda c: c.is_("bleeding", BLEEDING_HAEMATOMA_OR_DISORDER),
        lambda c: "expanding haematoma or bleeding in a known bleeding disorder",
        note="Groups \"Expanding haematoma\" and \"Bleeding Disorders\", both "
             "page 5 LEVEL 3. Note what this rule does NOT read: page 5 lists "
             "\"Anti-coagulant therapy\" in the CHECK FOR column, not in any "
             "level column, so the on_anticoagulant comorbidity escalates "
             "nothing on its own. Reading it as a level cell would put every "
             "warfarin and DOAC patient in the department at Level 3.",
        status=STATUS_SIGNED_OFF,
    ),
    # ----------------------------------------------------------- ECG (p7)
    #
    # Page 7's INITIAL TESTS row, which page 7 names as one of the four inputs
    # to the final level. An empty ECG set means no tracing was taken and
    # matches nothing; "Normal ECG" is a value the user must choose.
    Rule(
        2, "ECG", "Wide Complex Tachycardia", 7,
        lambda c: c.any_("ecg", ECG_WIDE_COMPLEX_TACHY),
        lambda c: "ECG: wide complex tachycardia",
    ),
    Rule(
        2, "ECG", "Narrow Complex Tachycardia > 150 / min", 7,
        lambda c: c.any_("ecg", ECG_NARROW_COMPLEX_TACHY_OVER_150),
        lambda c: "ECG: narrow complex tachycardia over 150 / min",
    ),
    Rule(
        2, "ECG", "ST elevations or depressions", 7,
        lambda c: c.any_("ecg", ECG_ST_ELEVATION_OR_DEPRESSION),
        lambda c: "ECG: ST elevation or depression",
        note="LEVEL 2, not Level 3. UNIMPLEMENTED_ADULT_CELLS recorded this "
             "cell at Level 3 together with the Atrial Fibrillation group; the "
             "PDF word coordinates put \"ST elevations or depressions\" at "
             "x=131, which is page 7's LEVEL 2 column, alongside \"Bradycardia "
             "< 40\" (already read correctly by the HR rule). The extracted "
             "text ordered it after the Level 3 items because it is printed "
             "lower on the page. The error never reached a patient - the cell "
             "was unimplemented - but it would have under-triaged every "
             "ischaemic ECG by one level the moment it was implemented.",
        status=STATUS_CORRECTED,
    ),
    Rule(
        3, "ECG", "Atrial Fibrillation > 100", 7,
        lambda c: c.any_("ecg", ECG_AF_OVER_100),
        lambda c: "ECG: atrial fibrillation over 100 / min",
    ),
    Rule(
        3, "ECG", "Frequent ectopics", 7,
        lambda c: c.any_("ecg", ECG_FREQUENT_ECTOPICS),
        lambda c: "ECG: frequent ectopics",
    ),
    Rule(
        3, "ECG", "Blocks / Sinus Pauses", 7,
        lambda c: c.any_("ecg", ECG_BLOCK_OR_PAUSE),
        lambda c: "ECG: block or sinus pause",
    ),
    Rule(
        3, "ECG", "Tall Tented T waves", 7,
        lambda c: c.any_("ecg", ECG_TALL_TENTED_T),
        lambda c: "ECG: tall tented T waves",
    ),
    Rule(
        4, "ECG", "No ECG ﬁndings; continuing chest pain", 7,
        lambda c: c.any_("ecg", ECG_NO_FINDINGS_ONGOING_PAIN),
        lambda c: "ECG: no findings, but chest pain continuing",
    ),
    Rule(
        5, "ECG", "Normal ECG", 7,
        lambda c: c.any_("ecg", ECG_NORMAL),
        lambda c: "ECG: normal",
    ),
    Rule(
        5, "ECG", "No ST-T wave changes", 7,
        lambda c: c.any_("ecg", ECG_NO_ST_T_CHANGES),
        lambda c: "ECG: no ST-T wave changes",
    ),
    # --------------------------------------------------- Comorbidity (p7)
    Rule(
        2, "Comorbidity", "Immunocompromised", 7,
        lambda c: c.any_("comorbidities", COMORBID_IMMUNOCOMPROMISED),
        lambda c: "immunocompromised",
    ),
    # -------------------------------------------------- Fever history (p7)
    Rule(
        4, "Fever history", "History of Fever", 7,
        lambda c: (c.is_("fever_history", FEVER_REPORTED)
                   and not c.has_measured_fever),
        lambda c: (
            "fever reported before arrival, none documented at triage "
            + (f"(temperature {c.temperature:g} C)" if c.temperature is not None
               else "(temperature not measured)")
        ),
        note="Page 7 prints \"History of Fever\" and \"No documented fever\" as "
             "two bullets in the SAME LEVEL 4 column (x=358), directly beneath "
             "\"Temp 37.5 - 39 C\" at Level 3 and beside \"No Fever\" at Level "
             "5. They are read here as ONE modifier - fever reported, none "
             "found now - because the alternative reading, in which \"No "
             "documented fever\" stands alone, would place every afebrile "
             "patient at Level 4 and contradict the Level 5 cell next to it. "
             "This also settles the question recorded as OPEN until "
             "2026-09-10, which assumed the cell was LEVEL 3 and rejected it "
             "as certain to escalate every febrile presentation. At Level 4 it "
             "cannot: a patient with a measured fever is already at Level 3 or "
             "2 on the Temp row, so this rule only ever bites for someone "
             "afebrile now who reports fever earlier - which is exactly the "
             "patient the cell describes.",
        status=STATUS_SIGNED_OFF,
    ),
    Rule(
        5, "Fever history", "No Fever", 7,
        lambda c: (c.is_("fever_history", FEVER_NONE_REPORTED)
                   and not c.has_measured_fever),
        lambda c: "no fever reported and none documented",
    ),
]


# Page 4 and page 5 carry no age qualification: they are the front door, and
# page 13 sends children through Primary Triage explicitly. Page 7's ECG,
# glucose, GCS, impression and fever rows have no paediatric counterpart on
# page 14, and the argument the module already makes for GCS applies to all of
# them - a child in a wide complex tachycardia is not Level 4 because page 14
# is silent on rhythm.
PAEDIATRIC_ADULT_PARAMETERS: tuple[str, ...] = (
    "GCS", "Glucose", "Appearance", "Breathing", "Perfusion", "Bleeding",
    "ECG", "Comorbidity", "Fever history",
)


# ===========================================================================
# PAEDIATRIC - page 13 WORK OF BREATHING and CIRCULATION, the danger signs of
# the Paediatric Assessment Triangle.
#
# Page 13's own sentence is why these belong in the floor: "Children without
# danger signs identified by the Paediatric Assessment Triangle at Primary
# Triage should proceed to Registration and Secondary Triage." The danger
# signs decide whether a child is triaged at all - and not one of them appears
# in page 14's vital-sign bands, which were until now the only paediatric
# input this module had.
#
# Cells already carried by another field are not repeated: "Cold Peripheries"
# and "CRT > 5 secs" (L1) and "CRT > 2 secs" (L2) are the PERFUSION field, and
# "Tachypnoea alone" (L3) is the RR band.
# ===========================================================================

PAEDIATRIC_SIGN_RULES: list[Rule] = [
    Rule(
        1, "Paediatric signs", "Snoring, muffled or hoarseness in speech", 13,
        lambda c: c.any_("paediatric_signs", PAED_SNORING_MUFFLED),
        lambda c: "snoring, muffled or hoarse speech",
    ),
    Rule(
        1, "Paediatric signs", "Stridor, grunting or wheezing", 13,
        lambda c: c.any_("paediatric_signs", PAED_STRIDOR_GRUNTING),
        lambda c: "stridor, grunting or wheezing",
    ),
    Rule(
        1, "Paediatric signs", "Sniffing position, tripod position", 13,
        lambda c: c.any_("paediatric_signs", PAED_SNIFFING_TRIPOD),
        lambda c: "sniffing or tripod position",
    ),
    Rule(
        1, "Paediatric signs", "Head bobbing for infants", 13,
        lambda c: c.any_("paediatric_signs", PAED_HEAD_BOBBING),
        lambda c: "head bobbing",
    ),
    Rule(
        1, "Paediatric signs", "Patchy or bluish skin discolouration", 13,
        lambda c: c.any_("paediatric_signs", PAED_MOTTLING_CYANOSIS),
        lambda c: "mottled or bluish skin discolouration",
    ),
    Rule(
        2, "Paediatric signs", "Difficulty in swallowing", 13,
        lambda c: c.any_("paediatric_signs", PAED_DIFFICULTY_SWALLOWING),
        lambda c: "difficulty swallowing",
    ),
    Rule(
        2, "Paediatric signs", "Drooling", 13,
        lambda c: c.any_("paediatric_signs", PAED_DROOLING),
        lambda c: "drooling",
    ),
    Rule(
        2, "Paediatric signs", "Unable to walk", 13,
        lambda c: c.any_("paediatric_signs", PAED_POSITIONAL_DISTRESS),
        lambda c: "unable to walk or refusing to lie down",
        note="Groups \"Unable to walk\" and \"Refusal to lie down\", both "
             "page 13 LEVEL 2, Abnormal Positioning.",
        status=STATUS_POLICY,
    ),
    Rule(
        2, "Paediatric signs",
        "Supraclavicular, intercostal or substernal retractions", 13,
        lambda c: c.any_("paediatric_signs", PAED_RETRACTIONS),
        lambda c: "supraclavicular, intercostal or substernal retractions",
    ),
    Rule(
        2, "Paediatric signs", "Nasal flaring on inspiration", 13,
        lambda c: c.any_("paediatric_signs", PAED_FLARING),
        lambda c: "nasal flaring or accessory muscle use",
        note="Groups \"Nasal flaring on inspiration\" and \"Accessory "
             "muscles\", both page 13 LEVEL 2, Flaring.",
        status=STATUS_POLICY,
    ),
    Rule(
        2, "Paediatric signs", "Pale mucous membranes / sole / palm", 13,
        lambda c: c.any_("paediatric_signs", PAED_PALLOR),
        lambda c: "pale mucous membranes, soles or palms",
    ),
]


# Page 7 cells this module cannot evaluate, because the structured intake
# carries no field for them. Listed so the gap is a recorded fact rather than
# a silent omission, and reported by deviations() as OPEN.
UNIMPLEMENTED_ADULT_CELLS: list[tuple[int, str, str]] = [
    (1, "Cardiac Arrest / Not Breathing / Major Trauma in Shock / Severe Resp Distress",
        "page 4 CRITICAL FIRST LOOK, Level 1. No structured field, and none is "
        "proposed: a patient who matches these is in the resuscitation bay "
        "before anyone opens a form. The complaint text and red_flags.py are "
        "the route"),
    (2, "Severe Chest Pain / Ongoing Seizures",
        "page 4 CRITICAL FIRST LOOK, Level 2. Derivable from the complaint "
        "text only; \"Severe Pain\", printed beside them in the same column, "
        "is already carried by the pain score"),
    (2, "Obvious Neuro deficits / Abnormal posturing / Confused, agitated, disoriented",
        "page 4 CONSCIOUS LEVELS, Level 2. The GCS rules reach the same level "
        "for a measured drop, but the named signs themselves have no field"),
]


# Printed cells this module could evaluate and deliberately does NOT, because
# each describes a NORMAL patient in a column above Level 5. Implementing them
# literally would escalate every patient who was assessed carefully - the
# examination itself would become the risk factor. Reported by deviations() as
# a decision, not as a gap.
DELIBERATELY_NOT_IMPLEMENTED: list[tuple[int, int, str, str]] = [
    (4, 3, "Peripheries warm, CRT normal",
        "A normal perfusion finding printed in page 4's LEVEL 3 column. The "
        "SECONDARY TRIAGE column of the same row prints \"Warm, pink, pulses "
        "normal\" - nearly the same words - and that is the reading taken: "
        "normal perfusion sends the patient on to Secondary Triage, it does "
        "not make them Urgent. The PERFUSION field therefore has a "
        "\"warm, pulses normal\" value that matches no rule"),
    (7, 4, "No documented fever",
        "Read as the second half of \"History of Fever\", the bullet printed "
        "directly above it in the same LEVEL 4 column, rather than as a cell "
        "standing alone. Alone it would place every afebrile patient at Level "
        "4 and contradict \"No Fever\" at Level 5 beside it. See the Fever "
        "history rule"),
    (4, 0, "Walking / Talking / Not distressed / Not aggressive",
        "Page 4's SECONDARY TRIAGE column is a DESTINATION, not a level: it "
        "means the Primary Triage Officer found nothing and the patient "
        "proceeds to registration. The reassuring values in the APPEARANCE, "
        "BREATHING, PERFUSION and BLEEDING fields therefore assert no level "
        "at all - they are recorded so the report can say the observation was "
        "made, which is not the same as it being absent"),
]


# ===========================================================================
# PAEDIATRIC - page 14 VITAL SIGNS (PAEDIATRICS).
#
# The printed table has FOUR level-columns, 1 to 4. There is no Level 5 column
# for children anywhere in the document. Note what that means, because it is
# stronger than the page 13 policy sentence: page 14's LEAST ACUTE paediatric
# column IS Level 4, so a child with entirely normal observations matches
# "> 96%", "36.5 - 37.5 C", "No pain" and lands at Level 4 by the table
# itself. Level 5 is not reachable for a child through any printed cell.
# ===========================================================================

# (upper age bound exclusive, printed label)
_RR_HR_BANDS = [(0.25, "< 3 months"), (1, "< 1 year"), (8, "< 8 years"), (12, "< 12 years")]
_SBP_BANDS = [(1, "< 1 year"), (5, "< 5 years"), (12, "< 12 years")]

# parameter -> age label -> level -> list of (lo, hi) ranges, transcribed cell
# by cell from page 14. The bound convention is the document's own notation and
# is NOT symmetric - see _paed_match:
#     "20 - 25 bpm" -> (20, 25)     inclusive at both ends
#     "> 60 bpm"    -> (60, None)   exclusive at 60
#     "< 20 bpm"    -> (None, 20)   exclusive at 20
_PAED_RANGES: dict[str, dict[str, dict[int, list[tuple[float | None, float | None]]]]] = {
    "RR": {
        "< 3 months": {1: [(60, None), (None, 20)], 2: [(50, None)],
                       3: [(40, 50), (20, 25)], 4: [(25, 40)]},
        "< 1 year":   {1: [(60, None), (None, 15)], 2: [(50, None)],
                       3: [(40, 50), (15, 20)], 4: [(20, 40)]},
        "< 8 years":  {1: [(60, None), (None, 12)], 2: [(45, None)],
                       3: [(30, 45), (12, 20)], 4: [(20, 30)]},
        "< 12 years": {1: [(60, None), (None, 12)], 2: [(40, None)],
                       3: [(25, 40), (12, 15)], 4: [(15, 25)]},
    },
    "HR": {
        "< 3 months": {1: [(180, None), (None, 80)], 2: [(80, 100)],
                       3: [(160, 180), (100, 120)], 4: [(120, 160)]},
        "< 1 year":   {1: [(180, None), (None, 80)], 2: [(80, 100)],
                       3: [(150, 180), (100, 110)], 4: [(110, 150)]},
        "< 8 years":  {1: [(180, None), (None, 60)], 2: [(150, 180), (60, 80)],
                       3: [(130, 150)], 4: [(80, 130)]},
        "< 12 years": {1: [(180, None), (None, 60)], 2: [(150, 180)],
                       3: [(120, 150), (60, 70)], 4: [(70, 120)]},
    },
    "SBP": {
        "< 1 year":   {1: [(None, 60)], 2: [(60, 70)], 3: [(100, None)], 4: [(70, 100)]},
        "< 5 years":  {1: [(None, 70)], 2: [(70, 90)], 3: [(110, None)], 4: [(90, 110)]},
        "< 12 years": {1: [(None, 80)], 2: [(80, 90)], 3: [(130, None)], 4: [(90, 130)]},
    },
    "SpO2": {
        "All ages":   {1: [(None, 92)], 2: [(92, 94)], 3: [(94, 96)], 4: [(96, None)]},
    },
    "Temp": {
        "All ages":   {1: [(40, None), (None, 36)], 2: [(39, 40)],
                       3: [(37.5, 39), (36.0, 36.5)], 4: [(36.5, 37.5)]},
    },
}

# The printed cell text, kept beside the numbers so a reviewer can compare.
_PAED_VERBATIM: dict[tuple[str, str, int], str] = {
    ("RR", "< 3 months", 1): "> 60 bpm / < 20 bpm",
    ("RR", "< 3 months", 2): "> 50 bpm",
    ("RR", "< 3 months", 3): "45 - 50 bpm / 20 - 25 bpm",
    ("RR", "< 3 months", 4): "25 - 40 bpm",
    ("RR", "< 1 year", 1): "> 60 bpm / < 15 bpm",
    ("RR", "< 1 year", 2): "> 50 bpm",
    ("RR", "< 1 year", 3): "40 - 50 bpm / 15 - 20 bpm",
    ("RR", "< 1 year", 4): "20 - 40 bpm",
    ("RR", "< 8 years", 1): "> 60 bpm / < 12 bpm",
    ("RR", "< 8 years", 2): "> 45 bpm",
    ("RR", "< 8 years", 3): "30 - 45 bpm / 12 - 20 bpm",
    ("RR", "< 8 years", 4): "20 - 30 bpm",
    ("RR", "< 12 years", 1): "> 60 bpm / < 12 bpm",
    ("RR", "< 12 years", 2): "> 40 bpm",
    ("RR", "< 12 years", 3): "25 - 40 bpm / 12 - 15 bpm",
    ("RR", "< 12 years", 4): "15 - 25 bpm",
    ("HR", "< 3 months", 1): "> 180 bpm / < 80 bpm",
    ("HR", "< 3 months", 2): "80 - 100 bpm",
    ("HR", "< 3 months", 3): "160 - 180 bpm / 100 - 120 bpm",
    ("HR", "< 3 months", 4): "120 - 160 bpm",
    ("HR", "< 1 year", 1): "> 180 bpm / < 80 bpm",
    ("HR", "< 1 year", 2): "80 - 100 bpm",
    ("HR", "< 1 year", 3): "150 - 180 bpm / 100 - 110 bpm",
    ("HR", "< 1 year", 4): "110 - 150 bpm",
    ("HR", "< 8 years", 1): "> 180 bpm / < 60 bpm",
    ("HR", "< 8 years", 2): "150 - 180 bpm / 60 - 80 bpm",
    ("HR", "< 8 years", 3): "130 - 150 bpm",
    ("HR", "< 8 years", 4): "80 - 130 bpm",
    ("HR", "< 12 years", 1): "> 180 bpm / < 60 bpm",
    ("HR", "< 12 years", 2): "150 - 180 bpm",
    ("HR", "< 12 years", 3): "120 - 150 bpm / 60 - 70 bpm",
    ("HR", "< 12 years", 4): "70 - 120 bpm",
    ("SBP", "< 1 year", 1): "SBP < 60 mmHg",
    ("SBP", "< 1 year", 2): "60 - 70 mmHg",
    ("SBP", "< 1 year", 3): "> 100 mmHg",
    ("SBP", "< 1 year", 4): "70 - 100 mmHg",
    ("SBP", "< 5 years", 1): "SBP < 70 mmHg",
    ("SBP", "< 5 years", 2): "70 - 90 mmHg",
    ("SBP", "< 5 years", 3): "> 110 mmHg",
    ("SBP", "< 5 years", 4): "90 - 110 mmHg",
    ("SBP", "< 12 years", 1): "SBP < 80 mmHg",
    ("SBP", "< 12 years", 2): "80 - 90 mmHg",
    ("SBP", "< 12 years", 3): "> 130 mmHg",
    ("SBP", "< 12 years", 4): "90 - 130 mmHg",
    ("SpO2", "All ages", 1): "< 92%",
    ("SpO2", "All ages", 2): "92 - 94%",
    ("SpO2", "All ages", 3): "94 - 96%",
    ("SpO2", "All ages", 4): "> 96%",
    ("Temp", "All ages", 1): "> 40 / < 36",
    ("Temp", "All ages", 2): "39 - 40C",
    ("Temp", "All ages", 3): "37.5 - 39 C / 36.0 - 36.5",
    ("Temp", "All ages", 4): "36.5 - 37.5 C",
}

# Ranges this module widened to close a gap the printed table leaves open. The
# value falls between two printed bands, so page 7's tie-break applies and the
# more acute band takes it.
PAED_GAP_CLOSURES = {
    ("RR", "< 3 months", 3): "printed \"45 - 50 bpm\"; widened to 40 - 50 because "
                             "40 - 45 is unspecified between the Level 4 band "
                             "(25 - 40) and this one, and page 7's tie-break "
                             "gives an undifferentiated value to the higher level",
    ("RR", "< 1 year", 3): "printed \"40 - 50 bpm\"; UNCHANGED - listed only to "
                           "record that it was checked. Contiguous with the "
                           "Level 4 band (20 - 40), and re-proved contiguous by "
                           f"audit() on {SIGN_OFF_DATE}",
}

_PAED_UNITS = {"RR": "/min", "HR": " bpm", "SBP": " mmHg", "SpO2": "%", "Temp": " C"}
_PAED_LABEL = {"RR": "respiratory rate", "HR": "heart rate", "SBP": "systolic BP",
               "SpO2": "SpO2", "Temp": "temperature"}

# ===========================================================================
# The paediatric pain ladder
#
# MTS 2022 prints this row twice and does not agree with itself. Page 14's
# VITAL SIGNS (PAEDIATRICS) reads > 7 -> L1, > 4 -> L2, "Little pain" -> L3,
# "No pain" -> L4. The COMPLAINTS LIST (PAEDIATRICS), pp15-18, reads "Severe
# Pain > 7" -> L2 and "Pain Score > 4" -> L3 with mild presentations at L4, in
# ABDOMINAL PAIN, ALLERGY / ANAPHYLAXIS, BURNS SCALDS, EAR / ENT, FALL and
# HEADACHE - one level less acute at every rung. Page 7's ADULT row agrees with
# the complaints list, not with page 14: "Severe Pain (8 - 10)" is L2 there.
#
# So page 14's pain row is the outlier: five other tables in the same document,
# including the adult one, put severe pain at Level 2, and page 14 is the only
# place in the whole appendix that puts ANY pain score at Level 1.
#
# RESOLVED 2026-09-09 in favour of pp15-18, the whole row from one table:
#
#     0 - 4  -> Level 4      > 4 (5 - 7) -> Level 3      > 7 -> Level 2
#
# The deciding argument is operational, not textual. MTS Level 1 is
# RESUSCITATION, and _MIN_DISPOSITION floors an MTS 1 patient at
# RESUSCITATION_BAY - so page 14's reading dispatched every child with a limb
# fracture to the resuscitation bay, displacing children in airway or
# circulatory failure. It bought nothing in exchange: a child who is genuinely
# critical still reaches Level 1 through the page 14 vitals bands, page 13's
# Paediatric Assessment Triangle, the GCS and glucose rules that apply at every
# age, and pp15-18's own Level 1 cells - "Mangled Limb", "Arterial bleeding",
# "Cold, painful, dusky limb", "CRT > 2 sec, pulse not felt", "Penetrating
# type", "Lethargy / Toxic appearance". Level 2 is Emergency, seen within 15
# minutes, which is when a child in severe pain should have analgesia.
#
# The ladder now runs parallel to the adult row at every rung, differing only
# in that a child floors at Level 4 where an adult reaches 5. Pain no longer
# makes a child MORE acute than an adult with the identical score, which was
# the anomaly that started this.
#
# Each rung is one constant. PAED_SEVERE_PAIN_LEVEL = 1,
# PAED_MODERATE_PAIN_LEVEL = 2 and PAED_LITTLE_PAIN_LEVEL = 3 together restore
# the literal page 14 reading.
# ===========================================================================

PAED_SEVERE_PAIN_LEVEL = 2  # pp15-18 "Severe Pain > 7"; p14 prints Level 1
PAED_MODERATE_PAIN_LEVEL = 3  # pp15-18 "Pain Score > 4"; p14 prints Level 2
PAED_LITTLE_PAIN_LEVEL = 4  # pp15-18 mild presentations; p14 prints Level 3

# Where the number itself comes from. MTS prints its paediatric pain row for
# "Child > 5 y" and says nothing about younger children, which used to mean a
# 3-year-old with a pain score of 9 produced NO pain finding at all - the
# single largest hole in the paediatric path. MTS is not the authority on that
# question, and the corpus already holds the one that is.
PAED_PAIN_SOURCE = (
    "Paediatric Protocols for Malaysian Hospitals, 5th Edition 2025, "
    "Section 21 Paediatric Emergency, chapter 117 \"Recognition and Assessment "
    "of Pain\" (p740)"
)

# (upper age bound exclusive, compact tag for the reason line, the instruments
# chapter 117 offers at that age). The Alder Hey Triage Pain Score is a
# five-item observational score, range 0 - 10, named by the chapter FOR TRIAGE,
# and it carries no lower age bound - so a recorded score is honoured at every
# paediatric age, and the reason line says whether it could have been
# self-reported or must have been observed. A self-reported 9 and an observed 9
# are not the same evidence, and a clinician re-checking the triage needs to
# see which one they have.
PAED_PAIN_INSTRUMENTS: list[tuple[float, str, str]] = [
    (3.0, "observational",
     "At or below 3 years chapter 117 offers only the Alder Hey Triage Pain "
     "Score (observational, 5 items, range 0 - 10). Both self-report "
     "instruments start above 3."),
    (8.0, "FACES or observational",
     "Above 3 years: FACES Pain Scale (Wong & Baker) - \"the child ... is asked "
     "to choose a face on the scale which best describes his / her level of "
     "pain. Score is 2, 4, 6, 8, or 10\" - so a FACES score is always even and "
     "never lands between rungs. Alder Hey also applies."),
    (float("inf"), "self-reported",
     "Above 8 years: Verbal Pain Assessment Scale (Likert) - \"asked to rate "
     "his or her pain by circling on any number on the scale of 0 to 10\". "
     "FACES and Alder Hey also apply."),
]


def paed_pain_instrument(age: float) -> tuple[str, str]:
    """(compact tag, full note) for how a child of this age was scored."""
    for bound, tag, note in PAED_PAIN_INSTRUMENTS:
        if age <= bound:
            return tag, note
    return PAED_PAIN_INSTRUMENTS[-1][1], PAED_PAIN_INSTRUMENTS[-1][2]


# (level, the printed cell, the page it is printed on, the score test). Three
# rungs cite pp15-18 because that is where their LEVEL is printed; "No pain" is
# page 14's, which pp15-18 has no cell for.
_PAED_PAIN_RUNGS: list[tuple[int, str, int, Callable[[int], bool]]] = [
    (PAED_SEVERE_PAIN_LEVEL, "Severe Pain > 7", 15, lambda s: s > 7),
    (PAED_MODERATE_PAIN_LEVEL, "Pain Score > 4", 15, lambda s: s > 4),
    (PAED_LITTLE_PAIN_LEVEL, "Mild symptoms", 15, lambda s: 1 <= s <= 4),
    (4, "No pain", 14, lambda s: s == 0),
]


def _paed_match(value: float, lo: float | None, hi: float | None) -> bool:
    """Does `value` fall in one printed paediatric cell?

    The bound convention is the document's own notation, and it is not the same
    at both ends: an "a - b" range is inclusive, a "> a" or "< b" cell is
    exclusive.

    CORRECTED 2026-09-09. Both open-ended forms used to be read inclusively,
    which put every boundary value exactly one level too acute: an SpO2 of 92%
    read as Level 1 where page 14 prints "< 92%" at Level 1 and "92 - 94%" at
    Level 2; a temperature of 40.0 C read as Level 1 against a printed
    "39 - 40C" at Level 2; an infant SBP of 60 read as Level 1 against
    "60 - 70 mmHg" at Level 2; an RR of exactly 60 read as Level 1 against
    "> 50 bpm" at Level 2. Under this convention no band is left with a gap -
    audit() re-proves that - and every boundary value lands in the cell the
    document prints it in.
    """
    if lo is None:
        return value < hi
    if hi is None:
        return value > lo
    return lo <= value <= hi


def _paed_band(parameter: str, age: float) -> str:
    bands = _SBP_BANDS if parameter == "SBP" else _RR_HR_BANDS
    for bound, label in bands:
        if age < bound:
            return label
    return bands[-1][1]


def _paed_value(ctx: Ctx, parameter: str) -> float | None:
    return {
        "RR": ctx.respiratory_rate, "HR": ctx.heart_rate, "SBP": ctx.systolic_bp,
        "SpO2": ctx.spo2, "Temp": ctx.temperature,
    }[parameter]


def _paed_findings(ctx: Ctx) -> list[Finding]:
    out: list[Finding] = []
    for parameter, by_band in _PAED_RANGES.items():
        band = "All ages" if "All ages" in by_band else _paed_band(parameter, ctx.age)
        value = _paed_value(ctx, parameter)
        if value is None:
            continue
        for level, ranges in sorted(by_band[band].items()):
            for lo, hi in ranges:
                if _paed_match(value, lo, hi):
                    key = (parameter, band, level)
                    out.append(Finding(
                        level=level, parameter=parameter,
                        verbatim=f"{_PAED_VERBATIM[key]} ({band})", page=14,
                        observed=f"{_PAED_LABEL[parameter]} "
                                 f"{value:g}{_PAED_UNITS[parameter]}",
                        note=PAED_GAP_CLOSURES.get(key, ""),
                    ))
                    break
    # Pain Score. Honoured at EVERY paediatric age - see PAED_PAIN_SOURCE for
    # why page 14's "Child > 5 y" is not a limit on the score, only on that
    # table's own row.
    if ctx.pain_score is not None:
        tag, instrument = paed_pain_instrument(ctx.age)
        for level, verbatim, page, hit in _PAED_PAIN_RUNGS:
            if hit(ctx.pain_score):
                out.append(Finding(
                    level=level, parameter="Pain", verbatim=verbatim, page=page,
                    observed=f"pain score {ctx.pain_score}/10 ({tag})",
                    note=(f"{instrument} Source: {PAED_PAIN_SOURCE}."
                          if ctx.age <= 5 else ""),
                ))
                break
    return out


# ===========================================================================
# Page 13, verbatim, and what it does and does not settle.
#
# The wording is GRADUATED, not one flat rule, and this module now follows the
# gradient rather than flattening it:
#
#   "They may be triaged to Levels 1 - 4 as needed. Generally, children,
#    especially those below 8 years old should not be triaged at Level 5 -
#    Routine. Infants, below 1 years old should be at Level 4 or higher."
#
# absolute below 1 year; near-absolute below 8 ("especially"); a general
# preference from 8 to 12 ("generally"). The floor is Level 4 at every age
# regardless, because page 14's own least-acute column IS Level 4 - but the
# reason a clinician is shown should be the sentence that actually applies to
# the child in front of them.
# ===========================================================================

PAEDIATRIC_FLOOR_AGE = float(config.PAEDIATRIC_AGE_YEARS)
PAEDIATRIC_NO_ROUTINE = (
    "Children should not be routinely triaged-away. ... Generally, children, "
    "especially those below 8 years old should not be triaged at Level 5 - "
    "Routine. Infants, below 1 years old should be at Level 4 or higher."
)
PAEDIATRIC_LEVEL5_STRICT_AGE = 8.0  # below this, page 13 says "especially"

# (upper age bound exclusive, the sentence that governs that age)
PAEDIATRIC_POLICY: list[tuple[float, str]] = [
    (1.0,
     "MTS p13: \"Infants, below 1 years old should be at Level 4 or higher.\" "
     "Stated without qualification."),
    (PAEDIATRIC_LEVEL5_STRICT_AGE,
     "MTS p13: \"Generally, children, especially those below 8 years old "
     "should not be triaged at Level 5 - Routine.\""),
    (PAEDIATRIC_FLOOR_AGE,
     "MTS p13: \"They may be triaged to Levels 1 - 4 as needed. Generally, "
     "children ... should not be triaged at Level 5 - Routine.\" Above 8 this "
     "is a general preference rather than the explicit rule that applies below "
     "8, but page 14's least acute paediatric column IS Level 4, so no printed "
     "cell returns 5 for a child of any age."),
]


def paediatric_policy(age: float) -> str:
    """The page 13 sentence that actually governs a child of this age."""
    for bound, text in PAEDIATRIC_POLICY:
        if age < bound:
            return text
    return PAEDIATRIC_POLICY[-1][1]


# ===========================================================================
# The floor
# ===========================================================================


def evaluate(
    age: float,
    vitals,
    clinical_text: str = "",
    observations: Observations | None = None,
) -> list[Finding]:
    """Every discriminator cell the observations match, most acute first."""
    ctx = _resolve_qualifiers(
        Ctx(
            age=age,
            obs=observations or NO_OBSERVATIONS,
            systolic_bp=vitals.systolic_bp,
            diastolic_bp=vitals.diastolic_bp,
            heart_rate=vitals.heart_rate,
            respiratory_rate=vitals.respiratory_rate,
            temperature=vitals.temperature,
            spo2=vitals.spo2,
            gcs=vitals.gcs,
            glucose=vitals.capillary_blood_glucose,
            pain_score=vitals.pain_score,
        ),
        clinical_text,
    )
    if ctx.is_paediatric:
        findings = _paed_findings(ctx)
        findings += [f for r in PAEDIATRIC_SIGN_RULES if (f := r.evaluate(ctx))]
        # Page 14 has no ECG, GCS or glucose row. Those adult cells still apply
        # to a child - a child with GCS 7 is not Level 4 because the paediatric
        # table is silent on consciousness. The Primary Triage rows extend to
        # children for a stronger reason than analogy: pages 4 - 5 are the
        # front door and carry no age qualification at all, and page 13 sends
        # children through Primary Triage explicitly.
        findings += [
            f for r in ADULT_RULES
            if r.parameter in PAEDIATRIC_ADULT_PARAMETERS and (f := r.evaluate(ctx))
        ]
    else:
        findings = [f for r in ADULT_RULES if (f := r.evaluate(ctx))]
    return sorted(findings, key=lambda f: f.level)


def floor(
    age: float,
    vitals,
    clinical_text: str = "",
    observations: Observations | None = None,
) -> tuple[int, list[str]]:
    """The most acute level the observations alone justify, with its reasons.

    Unlike the level it replaces, this ASSERTS Level 5 rather than returning it
    as the empty-reasons default: page 7's Level 5 column is "No Fever / No Pain
    / Vital Signs within normal limits / Normal Limits", and a patient who
    matches it is a documented Routine, not an absence of evidence.
    """
    findings = evaluate(age, vitals, clinical_text, observations)
    paed_floor = 4 if age < PAEDIATRIC_FLOOR_AGE else 5

    if not findings:
        if paed_floor == 4:
            return 4, [
                "no paediatric discriminator matched, and " + paediatric_policy(age)
            ]
        return 5, []

    level = min(f.level for f in findings)
    reasons = [f.reason() for f in findings if f.level == level]

    if level > paed_floor:
        return paed_floor, reasons + [
            f"held at Level {paed_floor}: {paediatric_policy(age)}"
        ]
    return level, reasons


# ===========================================================================
# Self-audit and the deviation record
# ===========================================================================


def audit() -> list[str]:
    """Re-prove that the transcribed paediatric bands classify every value.

    Every boundary in every band, the point either side of it, and the midpoint
    of every interval between boundaries must match at least one printed cell.
    A value that matches none would silently drop out of _paed_findings and
    lower the floor. Returns the problems found; an empty list is a pass.
    """
    problems: list[str] = []
    for parameter, by_band in _PAED_RANGES.items():
        step = 0.1 if parameter == "Temp" else 1.0
        for band, by_level in by_band.items():
            edges = sorted({
                b for ranges in by_level.values() for lo, hi in ranges
                for b in (lo, hi) if b is not None
            })
            probes = [edges[0] - step]
            for i, edge in enumerate(edges):
                probes.append(edge)
                nxt = edges[i + 1] if i + 1 < len(edges) else None
                if nxt is not None and nxt - edge > step:
                    probes.append(round((edge + nxt) / 2, 1))
            probes.append(edges[-1] + step)
            for value in probes:
                hits = [
                    level for level, ranges in by_level.items()
                    if any(_paed_match(value, lo, hi) for lo, hi in ranges)
                ]
                if not hits:
                    problems.append(
                        f"{parameter} {band}: {value:g} matches no printed cell"
                    )
    for name, q in QUALIFIERS.items():
        if not q.because:
            problems.append(f"qualifier {name}: leans {q.lean} with no reason given")
    return problems


@dataclass(frozen=True)
class Deviation:
    what: str
    status: str
    why: str


# The adult mapping rules as they are shown to the model.
#
# They used to be a hand-written block in rag_engine.MTS_2022_GUARDRAILS, and by
# 2026-09-11 it had drifted from this module in four places, every one of them
# in the under-triage direction:
#
#   printed here          the prompt said      consequence
#   Pain 8-10 -> L2       "Pain Score 7-10"    pain 7 taught as Emergency
#   Pain 4-7  -> L3       "Pain Score 4-6"     pain 7 taught as neither
#   HR > 120  -> L2       "HR 101-130" at L3   HR 121-130 taught as Urgent
#   Level 5 < 90 mins     "< 120 mins"         contradicted MTS_SCALE above it
#
# `_enforce_mts_level` corrected the OUTCOME every time, so no patient was
# mis-triaged - but the model was reasoning from wrong numbers and then being
# overridden, which is visible to the clinician as a guardrail note on a report
# whose rationale argues for the wrong level. Generating the block from
# ADULT_RULES is the same move already made for mts_triage_label, the citation
# table and time_to_treatment: a value the server can derive is never asked for
# and never re-typed.
#
# Only measurable parameters are rendered. Appearance, Bleeding, Breathing,
# Perfusion and ECG reach the model as structured intake lines and as retrieved
# discriminator cells, and listing all 64 cells here would crowd the prompt.
PROMPT_PARAMETERS = ("GCS", "Pain", "HR", "RR", "SpO2", "Temp", "BP", "Glucose")

# The Primary Triage observations. These are not numbers - they are what the
# triage officer SEES - and they reach the model as structured intake lines and
# as retrieved discriminator cells. Only Levels 1-2 are rendered: a patient in
# arrest or in shock has no useful vital signs to match on, so the observational
# route to an acute level has to be stated. The less acute observational cells
# would only crowd the prompt.
#
# Rendered from ADULT_RULES for the same reason PROMPT_PARAMETERS is. The first
# version of this block, written the same day the drifted one was deleted, was
# hand-typed prose - which is how the drift happens.
PROMPT_OBSERVATIONAL = ("Appearance", "Breathing", "Perfusion", "Bleeding")
PROMPT_OBSERVATIONAL_MAX_LEVEL = 2


def prompt_rules() -> str:
    """The adult grid, rendered from the cells `floor()` actually applies."""
    by_level: dict[int, list[Rule]] = {}
    for rule in ADULT_RULES:
        if rule.parameter in PROMPT_PARAMETERS:
            by_level.setdefault(rule.level, []).append(rule)

    lines = [
        "MALAYSIAN TRIAGE SCALE (MTS 2022) ADULT MAPPING RULES",
        f"Source: {MTS_SOURCE}, pages 4 and 7, transcribed cell by cell.",
        "Every line below is the cell text as printed. The server re-applies",
        "these rules in code after you answer and will OVERRIDE a level that",
        "contradicts them, so reason from these numbers and no others.",
    ]
    for level in sorted(by_level):
        lines.append(
            f"\n  LEVEL {level} - {LEVEL_NAMES[level]}, "
            f"{TIME_TO_TREATMENT[level]}"
        )
        for rule in by_level[level]:
            lines.append(
                f"    {rule.parameter}: {rule.prompt_text or rule.verbatim}"
                f" (p{rule.page})"
            )
    # Level 5 has no measurable cell of its own. `floor()` reaches it by
    # elimination, and a model shown Levels 1-4 and nothing else will not.
    lines.append(
        f"\n  LEVEL 5 - {LEVEL_NAMES[5]}, {TIME_TO_TREATMENT[5]}"
        "\n    Reached by ELIMINATION: no criterion above is met, the patient"
        "\n    reports NO pain at all, and the complaint is routine or chronic."
        "\n    Any pain score of 1 or more excludes Level 5 (p7 prints the"
        "\n    Level 5 pain cell as literally 'No Pain')."
    )
    obs: dict[int, list[Rule]] = {}
    for rule in ADULT_RULES:
        if (
            rule.parameter in PROMPT_OBSERVATIONAL
            and rule.level <= PROMPT_OBSERVATIONAL_MAX_LEVEL
        ):
            obs.setdefault(rule.level, []).append(rule)
    if obs:
        lines.append(
            "\nPRIMARY TRIAGE OBSERVATIONS (MTS 2022 pages 4-5). Not numbers -"
            "\nwhat the triage officer SEES. A patient meeting one of these is"
            "\nthat level whatever the vital signs say."
        )
        for level in sorted(obs):
            lines.append(f"\n  LEVEL {level} - {LEVEL_NAMES[level]}")
            for rule in obs[level]:
                lines.append(
                    f"    {rule.parameter}: {rule.prompt_text or rule.verbatim}"
                    f" (p{rule.page})"
                )
        lines.append(
            "\n  CLEARING PRIMARY TRIAGE - page 4's fourth column, headed"
            "\n  SECONDARY TRIAGE, is the negative case: walking, talking, not"
            "\n  distressed, SpO2 > 94% on air, warm peripheries with normal"
            "\n  pulses, alert and sitting upright. It is NOT a triage level. A"
            "\n  patient matching it has no Primary Triage escalation and is"
            "\n  levelled on the Secondary Triage criteria alone."
        )
    lines.append(
        "\nA patient meets the MOST ACUTE level of any single criterion above."
        "\nRanges are inclusive as printed: 'Pain Score 4 - 7' includes 7, and"
        "\n'GCS < 13' excludes 13 (GCS 13 is its own Level 2 cell)."
        "\nNEVER assign Level 1 or Level 2 on a pain score of 1-3 unless an"
        "\nairway, breathing or circulatory criterion above is also met."
    )
    return "\n".join(lines)


def deviations() -> list[Deviation]:
    """Every place this module departs from, extends, or corrects the printed
    tables, and every default the document does not supply.

    STATUS_SIGNED_OFF entries were reviewed clinically on SIGN_OFF_DATE and are
    settled - a later pass must not "tidy" them back toward the printed grid
    without a new decision. STATUS_OPEN entries still need one.
    """
    out = [
        Deviation(
            f"{r.parameter} L{r.level} \"{r.verbatim}\" (p{r.page})",
            r.status or STATUS_POLICY,
            r.note,
        )
        for r in ADULT_RULES if r.note
    ]
    out += [
        Deviation(
            f"paediatric {p} L{lvl} ({band}) (p14)",
            STATUS_SIGNED_OFF,
            why,
        )
        for (p, band, lvl), why in PAED_GAP_CLOSURES.items()
    ]
    out.append(Deviation(
        "paediatric open-ended bounds (p14)",
        STATUS_CORRECTED,
        "Every \"> x\" and \"< x\" paediatric cell used to be read INCLUSIVELY, "
        "putting each boundary value one level too acute - SpO2 92% at Level 1 "
        "against a printed \"92 - 94%\" at Level 2, Temp 40.0 C at Level 1 "
        f"against \"39 - 40C\", infant SBP 60 at Level 1 against \"60 - 70 mmHg\", "
        f"RR 60 at Level 1 against \"> 50 bpm\". Corrected {SIGN_OFF_DATE}; "
        "audit() proves the corrected bands still leave no value unclassified.",
    ))
    out.append(Deviation(
        "paediatric pain ladder: the whole row moved to pp15-18 (RESOLVED)",
        STATUS_CONFLICT,
        f"Page 14 prints > 7 -> L1, > 4 -> L2, \"Little pain\" -> L3. The "
        "COMPLAINTS LIST (PAEDIATRICS) pp15-18 prints \"Severe Pain > 7\" -> L2 "
        "and \"Pain Score > 4\" -> L3 with mild presentations at L4, in ABDOMINAL "
        "PAIN, ALLERGY / ANAPHYLAXIS, BURNS SCALDS, EAR / ENT, FALL and HEADACHE. "
        "Page 7's ADULT row agrees with the complaints list, not page 14. Page 14 "
        "is the only place in the entire appendix that puts any pain score at "
        f"Level 1. Resolved {SIGN_OFF_DATE} in favour of pp15-18, all three rungs "
        f"from one table: 0-4 -> L{PAED_LITTLE_PAIN_LEVEL}, 5-7 -> "
        f"L{PAED_MODERATE_PAIN_LEVEL}, > 7 -> L{PAED_SEVERE_PAIN_LEVEL}. "
        "THE DECIDING ARGUMENT IS OPERATIONAL: MTS 1 is Resuscitation and "
        "_MIN_DISPOSITION floors it at RESUSCITATION_BAY, so page 14's reading "
        "sent every child with a limb fracture to the resuscitation bay and "
        "displaced children in airway or circulatory failure - while buying "
        "nothing, because a critical child still reaches Level 1 through the p14 "
        "vitals bands, p13's Paediatric Assessment Triangle, the GCS and glucose "
        "rules, and pp15-18's own Level 1 cells (\"Mangled Limb\", \"Arterial "
        "bleeding\", \"Cold, painful, dusky limb\", \"CRT > 2 sec, pulse not "
        "felt\", \"Lethargy / Toxic appearance\"). The ladder now runs parallel "
        "to the adult row, differing only in the Level 4 paediatric floor, so "
        "pain no longer makes a child MORE acute than an adult with the same "
        "score. Set all three constants back (1 / 2 / 3) to restore page 14.",
    ))
    out.append(Deviation(
        "paediatric pain score is read at EVERY age (p14 prints \"Child > 5 y\")",
        STATUS_SIGNED_OFF,
        "The largest hole in the paediatric path, closed "
        f"{SIGN_OFF_DATE}: the pain row was gated on age > 5, so a 3-year-old "
        "with a pain score of 9 produced no pain finding at all and could floor "
        "at Level 4 on a fracture. MTS is not the authority on whether a child "
        f"can be scored - {PAED_PAIN_SOURCE} is, and it is already in the "
        "corpus. It gives the Alder Hey Triage Pain Score: five items, range "
        "0 - 10, named for triage, with NO lower age bound. \"Child > 5 y\" is a "
        "property of page 14's own row, not of the score. Self-report has its own "
        "floors - FACES above 3 (scores 2/4/6/8/10, always even), Verbal Pain "
        "Assessment above 8 - so the reason line now carries whether the number "
        "could have been self-reported or must have been observed, because those "
        "are not the same evidence.",
    ))
    out.append(Deviation(
        "child at pain 4 is Level 4, adult at pain 4 is Level 3 - CHECKED",
        STATUS_CONFLICT,
        "Recorded so it cannot later look like a bug. Both readings are literal "
        "and the tables genuinely differ on where the mild band ends: page 7 "
        "prints the ADULT band as \"Pain Score 4 - 7\" (Level 3, inclusive of 4), "
        "while pp15-18 print the paediatric cell as \"Pain Score > 4\" (Level 3, "
        "exclusive of 4) and page 14 prints \"Little pain\" over 1 - 4. A score "
        "of exactly 4 therefore lands one level less acute for a child - and 4 "
        "is a real FACES score, so it does occur. Left as printed: inventing a "
        "paediatric \"4 -> Level 3\" would contradict pp15-18's explicit \"> 4\", "
        "and the gap is one level at one point on the scale.",
    ))
    out.append(Deviation(
        "paediatric GCS uses the adult thresholds - CHECKED, no change",
        STATUS_POLICY,
        "Recorded because it looks like a gap and is not. Paediatric Protocols "
        "5th Ed 2025 Section 9 (p298) prints a Modified Glasgow Coma Scale for "
        "infants, whose verbal and motor descriptors differ from the adult "
        "scale - but it scores 4 + 5 + 6 on the same 3 - 15 range, so the GCS "
        "thresholds in ADULT_RULES (< 9, < 13, 13, 14) carry over unchanged. The "
        "descriptors matter to whoever assigns the number, not to this module.",
    ))
    out.append(Deviation(
        "Primary Triage and Initial Tests are now read (ADDED 2026-09-10)",
        STATUS_SIGNED_OFF,
        "Page 7: the final level \"takes into consideration all of the "
        "selected parameters ie. Primary Triage, Vital Signs, Complaints List "
        "and Initial Tests\". This module read two of those four. It now reads "
        "all four: pages 4 - 5 arrive as the APPEARANCE, BREATHING, PERFUSION "
        "and BLEEDING observations, page 13's Paediatric Assessment Triangle "
        "as PAEDIATRIC SIGNS, and page 7's ECG row and Immunocompromised cell "
        "as the ECG and COMORBIDITY observations. Every field is OPTIONAL and "
        "an unset one matches nothing, so no existing patient's level can move "
        "because the fields exist. Column membership for every new cell was "
        "read from the PDF word coordinates rather than the extracted text, "
        "which flattens each column in turn and had already produced two "
        "recorded level errors - see the ST-elevation and Fever history rules.",
    ))
    out.append(Deviation(
        "paediatric danger signs now reach the floor (ADDED 2026-09-10)",
        STATUS_SIGNED_OFF,
        "The largest remaining paediatric hole. Page 13 says \"Children "
        "without danger signs identified by the Paediatric Assessment Triangle "
        "at Primary Triage should proceed to Registration and Secondary "
        "Triage\" - so the danger signs decide whether a child is triaged at "
        "all - and not one of them appears in page 14's vital-sign bands, "
        "which were the only paediatric input this module had. A child with "
        "stridor, retractions and a capillary refill of 3 seconds could floor "
        "at Level 4 on normal numbers, because a compensating child holds its "
        "blood pressure until it does not. PERFUSION carries the circulation "
        "cells for every age; PAEDIATRIC_SIGN_RULES carries the rest for the "
        "under-12s.",
    ))
    for name, q in QUALIFIERS.items():
        out.append(Deviation(
            f"lean: {name} -> {'PRESENT' if q.lean else 'ABSENT'} when unrecorded",
            STATUS_POLICY,
            f"Split: {q.split}. Settled from the vitals where possible"
            + (f" ({q.vitals_meaning.rstrip('.')})" if q.vitals_meaning else "")
            + f". Leaned {'present' if q.lean else 'absent'} otherwise, because "
            + q.because[0].lower() + q.because[1:],
        ))
    out.append(Deviation(
        "paediatric Level 5",
        STATUS_POLICY,
        "Every patient under "
        f"{PAEDIATRIC_FLOOR_AGE:g} is floored at Level 4. The binding reason is "
        "structural, not policy: page 14's least acute column IS Level 4 "
        "(\"> 96%\", \"36.5 - 37.5 C\", \"No pain\"), and the paediatric "
        "COMPLAINTS LIST (pp15-18) has no Level 5 column either, so a child with "
        "wholly normal observations lands at 4 by the table before page 13 is "
        "consulted. Page 13 then confirms it, in graduated wording this module "
        "follows: absolute below 1 year, \"especially\" below 8, \"generally\" "
        "from 8 to 12. NOTE THE CONFLICT: page 18 says the MyTriage App at "
        "Secondary Triage \"will provide a triage recommendation (to MTC Levels "
        "2 - 5)\" and is \"designed to be used for all age groups\", naming "
        "\"school-going children ages 6 - 12\" - so the document's own tool can "
        "output Level 5 for a child its tables cannot. Level 4 is kept because "
        "no printed cell supports 5, and because Level 4 still PERMITS discharge "
        "with follow-up: the ceiling in _apply_disposition_ceiling is a "
        "permitted set, so a child treated and sent home is not blocked by it.",
    ))
    out += [
        Deviation(
            f"NOT IMPLEMENTED: {cell} (L{lvl})",
            STATUS_OPEN,
            f"{why}. The cell is printed and this module cannot evaluate it, so "
            "a patient who matches only that cell is under-triaged by it.",
        )
        for lvl, cell, why in UNIMPLEMENTED_ADULT_CELLS
    ]
    out += [
        Deviation(
            f"NOT IMPLEMENTED ON PURPOSE: {cell} (p{page}"
            + (f" L{lvl})" if lvl else " Secondary Triage column)"),
            STATUS_SIGNED_OFF,
            f"{why}.",
        )
        for page, lvl, cell, why in DELIBERATELY_NOT_IMPLEMENTED
    ]
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    problems = audit()
    if "--audit" in sys.argv:
        print(f"{MTS_SOURCE} - paediatric band audit")
        for p in problems:
            print(f"  PROBLEM  {p}")
        print("  no unclassified value in any band" if not problems
              else f"  {len(problems)} problem(s)")
        sys.exit(1 if problems else 0)

    print(f"{MTS_SOURCE}\n")
    print("ADULT RULES (page 4 / page 7)")
    print(f"{'LEVEL':6} {'PARAM':9} {'PAGE':5} VERBATIM CELL")
    for r in sorted(ADULT_RULES, key=lambda x: (x.parameter, x.level)):
        mark = f"  [{r.status or STATUS_POLICY}]" if r.note else ""
        print(f"  L{r.level:<4} {r.parameter:9} p{r.page:<4} {r.verbatim}{mark}")
    print("\nPAEDIATRIC RULES (page 14)")
    for parameter, by_band in _PAED_RANGES.items():
        for band in by_band:
            cells = " | ".join(
                f"L{lvl}: {_PAED_VERBATIM[(parameter, band, lvl)]}"
                for lvl in sorted(by_band[band])
            )
            print(f"  {parameter:5} {band:12} {cells}")
    print(f"  Pain  all paed ages  L{PAED_SEVERE_PAIN_LEVEL}: Severe Pain > 7 "
          f"(p15) | L{PAED_MODERATE_PAIN_LEVEL}: Pain Score > 4 (p15) | "
          f"L{PAED_LITTLE_PAIN_LEVEL}: Mild symptoms (p15) | L4: No pain (p14)")
    print("\n  PAEDIATRIC PAIN INSTRUMENT BY AGE")
    for age in (2, 5, 10):
        print(f"    age {age:<3g} {paed_pain_instrument(age)[0]}")
    print("\nPAEDIATRIC LEVEL 5 POLICY BY AGE (page 13)")
    for age in (0.5, 4, 9):
        print(f"  age {age:<5g} {paediatric_policy(age)}")

    by_status: dict[str, list[Deviation]] = {}
    for d in deviations():
        by_status.setdefault(d.status, []).append(d)
    print("\nDEVIATIONS, CORRECTIONS AND POLICY CHOICES")
    for status in (STATUS_OPEN, STATUS_CONFLICT, STATUS_CORRECTED,
                   STATUS_SIGNED_OFF, STATUS_POLICY):
        items = by_status.get(status, [])
        if not items:
            continue
        print(f"\n  === {status} ({len(items)}) ===")
        for d in items:
            print(f"  - {d.what}\n      {d.why}")
    print(f"\nAUDIT: {'clean' if not problems else problems}")
