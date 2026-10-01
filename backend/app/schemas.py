"""Pydantic models.

Mode A  -> DiagnosticSchema is handed to the local LLM. 
           We use forgiving defaults so smaller MLX models don't crash 
           the parser if they skip a nested key.
Mode B  -> free-form markdown answer plus machine-readable citations.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Literal, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import mts_table as _mts


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _LLMOut(_Strict):
    """Base for everything the local model authors.

    A small model types loosely — it emits `"mts_level_triggered": 3` where the
    schema wants `"3"`, and Pydantic v2 rejects int-for-str by default, turning
    a usable answer into a 502. Coercing numbers to strings on the OUTPUT models
    only keeps the API strict about what clients send while staying tolerant of
    how the model spells its own values."""

    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


# ------------------------------------------------- enum spelling tolerance
# Same class of problem as the int-for-str coercion above, and solved the same
# way. The model writes enum values the way a clinician READS them: MTS 4 came
# back as "EARLY CARE", the schema wanted "EARLY_CARE", and a whole Mode A
# generation - minutes of local compute - was thrown away as a 502 over one
# missing underscore. Note the frontend renders these with `.replace("_", " ")`,
# so the model was emitting precisely the display form.
#
# Normalising is not the same as guessing: only separators and case are touched,
# so a value that means something else still fails. It stays strict about what
# CLIENTS send (those models are _Strict, not _LLMOut).
_ENUM_SEP = re.compile(r"[\s\-/]+")


def _enum_token(value: Any) -> Any:
    """'EARLY CARE' -> 'EARLY_CARE'; 'Admit ICU/HDU' -> 'ADMIT_ICU_HDU'."""
    if isinstance(value, str):
        return _ENUM_SEP.sub("_", value.strip()).upper()
    return value


_MTS_DIGIT = re.compile(r"[1-5]")


def _triage_int(value: Any) -> Any:
    """Accept 'Level 4', 'MTS 4', '4' or 4. Anything else is left to fail."""
    if isinstance(value, str) and (m := _MTS_DIGIT.search(value)):
        return int(m.group(0))
    return value


# ============================================================== Mode A input

class Vitals(_Strict):
    systolic_bp: float | None = Field(None, description="mmHg", ge=0, le=350)
    diastolic_bp: float | None = Field(None, description="mmHg", ge=0, le=250)
    heart_rate: float | None = Field(None, description="beats/min", ge=0, le=350)
    respiratory_rate: float | None = Field(None, description="breaths/min", ge=0, le=120)
    temperature: float | None = Field(None, description="deg C", ge=20, le=45)
    spo2: float | None = Field(None, description="% on room air unless stated", ge=0, le=100)
    gcs: int | None = Field(None, description="Glasgow Coma Scale 3-15", ge=3, le=15)
    capillary_blood_glucose: float | None = Field(None, description="mmol/L", ge=0, le=60)
    pain_score: int | None = Field(None, description="0-10 numeric rating", ge=0, le=10)

    def as_clinical_text(self) -> str:
        parts: list[str] = []
        if self.systolic_bp is not None or self.diastolic_bp is not None:
            sbp = f"{self.systolic_bp:g}" if self.systolic_bp is not None else "?"
            dbp = f"{self.diastolic_bp:g}" if self.diastolic_bp is not None else "?"
            parts.append(f"BP {sbp}/{dbp} mmHg")
        if self.heart_rate is not None:
            parts.append(f"HR {self.heart_rate:g} bpm")
        if self.respiratory_rate is not None:
            parts.append(f"RR {self.respiratory_rate:g} breaths/min")
        if self.temperature is not None:
            parts.append(f"Temp {self.temperature:g} degC")
        if self.spo2 is not None:
            parts.append(f"SpO2 {self.spo2:g}%")
        if self.gcs is not None:
            parts.append(f"GCS {self.gcs}")
        if self.capillary_blood_glucose is not None:
            parts.append(f"CBG {self.capillary_blood_glucose:g} mmol/L")
        if self.pain_score is not None:
            parts.append(f"Pain score {self.pain_score}/10")
        return "; ".join(parts) if parts else "No vital signs recorded."


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


# ------------------------------------------------- triage modifiers
#
# MTS 2022 page 7: the final triage level "takes into consideration all of the
# selected parameters ie. Primary Triage, Vital Signs, Complaints List and
# Initial Tests". The intake used to carry two of those four. The enums below
# are the other two, plus the background a prescribing check needs.
#
# EVERY ONE IS OPTIONAL. Unset means not assessed, never "normal" - the same
# contract the vitals already have, and the reason adding these fields cannot
# change the level of a patient entered the old way.
#
# The MTS-derived values are taken from `mts_table`, which holds the printed
# grid, so the two cannot drift apart. tests/test_triage_modifiers.py asserts
# set equality in both directions.


class GeneralAppearance(str, Enum):
    """MTS p4 CRITICAL FIRST LOOK, plus p7's two global-impression cells."""
    NOT_DISTRESSED = _mts.APPEARANCE_NOT_DISTRESSED
    UNWELL = _mts.APPEARANCE_UNWELL
    CANNOT_SIT_STAND = _mts.APPEARANCE_CANNOT_SIT_STAND
    NOT_RESPONDING = _mts.APPEARANCE_NOT_RESPONDING
    SEPTIC_ILL = _mts.APPEARANCE_SEPTIC_ILL


class Breathing(str, Enum):
    """MTS p4 RAPID ASSESSMENT / RESPIRATORY DISTRESS - the speech ladder."""
    NOT_BREATHLESS = _mts.BREATHING_NOT_BREATHLESS
    WHEEZE_AIRWAY_INTACT = _mts.BREATHING_WHEEZE_AIRWAY_INTACT
    NEEDS_OXYGEN = _mts.BREATHING_NEEDS_OXYGEN
    SHORT_PHRASES = _mts.BREATHING_SHORT_PHRASES
    ABNORMAL_SOUNDS = _mts.BREATHING_ABNORMAL_SOUNDS
    EXCESSIVE_WORK = _mts.BREATHING_EXCESSIVE_WORK
    ONE_WORD = _mts.BREATHING_ONE_WORD
    ASSISTED = _mts.BREATHING_ASSISTED


class Perfusion(str, Enum):
    """MTS p4 RAPID ASSESSMENT / SHOCK STATE, and p13 CIRCULATION."""
    WARM_NORMAL = _mts.PERFUSION_WARM_NORMAL
    CRT_OVER_2 = _mts.PERFUSION_CRT_OVER_2
    WEAK_PULSES = _mts.PERFUSION_WEAK_PULSES
    COLD_OR_CYANOSED = _mts.PERFUSION_COLD_OR_CYANOSED
    ABSENT_RADIAL = _mts.PERFUSION_ABSENT_RADIAL


class Bleeding(str, Enum):
    """MTS p5 RAPID ASSESSMENT / BLEEDING."""
    NONE = _mts.BLEEDING_NONE
    WOUND_OR_ENT = _mts.BLEEDING_WOUND_OR_ENT
    HAEMATOMA_OR_DISORDER = _mts.BLEEDING_HAEMATOMA_OR_DISORDER
    VOMIT_OR_COUGH_BLOOD = _mts.BLEEDING_VOMIT_OR_COUGH_BLOOD
    SUSPECTED_INTERNAL = _mts.BLEEDING_SUSPECTED_INTERNAL
    ARTERIAL_OR_UNCONTROLLED = _mts.BLEEDING_ARTERIAL_OR_UNCONTROLLED


class EcgFinding(str, Enum):
    """MTS p7 INITIAL TESTS. An EMPTY list means no ECG was taken."""
    NORMAL = _mts.ECG_NORMAL
    NO_ST_T_CHANGES = _mts.ECG_NO_ST_T_CHANGES
    NO_FINDINGS_ONGOING_PAIN = _mts.ECG_NO_FINDINGS_ONGOING_PAIN
    AF_OVER_100 = _mts.ECG_AF_OVER_100
    FREQUENT_ECTOPICS = _mts.ECG_FREQUENT_ECTOPICS
    BLOCK_OR_PAUSE = _mts.ECG_BLOCK_OR_PAUSE
    TALL_TENTED_T = _mts.ECG_TALL_TENTED_T
    ST_ELEVATION_OR_DEPRESSION = _mts.ECG_ST_ELEVATION_OR_DEPRESSION
    WIDE_COMPLEX_TACHY = _mts.ECG_WIDE_COMPLEX_TACHY
    NARROW_COMPLEX_TACHY_OVER_150 = _mts.ECG_NARROW_COMPLEX_TACHY_OVER_150


class FeverHistory(str, Enum):
    """MTS p7 Temp row, Levels 4 and 5. Unset means the question was not put."""
    REPORTED = _mts.FEVER_REPORTED
    NONE_REPORTED = _mts.FEVER_NONE_REPORTED


class Comorbidity(str, Enum):
    """Only IMMUNOCOMPROMISED is an MTS cell (p7 Level 2). The rest carry no
    triage rule at all and exist so the drug guardrails fire on a recorded
    fact instead of on whether the word was typed into the history box."""
    IMMUNOCOMPROMISED = _mts.COMORBID_IMMUNOCOMPROMISED
    ANTICOAGULANT = _mts.COMORBID_ANTICOAGULANT
    CKD = _mts.COMORBID_CKD
    AIRWAY_DISEASE = _mts.COMORBID_AIRWAY_DISEASE
    DIABETES = _mts.COMORBID_DIABETES
    LIVER_DISEASE = _mts.COMORBID_LIVER_DISEASE


class PaediatricSign(str, Enum):
    """MTS p13 - the danger signs of the Paediatric Assessment Triangle."""
    SNORING_MUFFLED = _mts.PAED_SNORING_MUFFLED
    STRIDOR_GRUNTING = _mts.PAED_STRIDOR_GRUNTING
    SNIFFING_TRIPOD = _mts.PAED_SNIFFING_TRIPOD
    HEAD_BOBBING = _mts.PAED_HEAD_BOBBING
    MOTTLING_CYANOSIS = _mts.PAED_MOTTLING_CYANOSIS
    DIFFICULTY_SWALLOWING = _mts.PAED_DIFFICULTY_SWALLOWING
    DROOLING = _mts.PAED_DROOLING
    POSITIONAL_DISTRESS = _mts.PAED_POSITIONAL_DISTRESS
    RETRACTIONS = _mts.PAED_RETRACTIONS
    FLARING = _mts.PAED_FLARING
    PALLOR = _mts.PAED_PALLOR


# ---- not MTS cells: context that steers retrieval and the drug checks -----

class ArrivalMode(str, Enum):
    """MTS p6 names these placements explicitly. Not a level modifier."""
    WALK_IN = "walk_in"
    AMBULANCE = "ambulance"
    POLICE_OKT = "police_okt"
    REFERRED_FROM_CLINIC = "referred_from_clinic"
    OSCC = "oscc"


class ExposureRisk(str, Enum):
    """MTS p5 INFECTIOUS DISEASES / HAZMAT. Drives PLACEMENT, not level: the
    printed table's output column is "TO BE PLACED AT", not a triage level."""
    SUSPECTED_TB = "suspected_tb"
    FEBRILE_RESPIRATORY = "febrile_respiratory_illness"
    OUTBREAK_OR_TRAVEL = "outbreak_or_travel_contact"
    KNOWN_MDRO = "known_mdro"
    CHEMICAL_HAZMAT = "chemical_or_hazmat"


class BehaviouralRisk(str, Enum):
    """MTS p6 AGGRESSIVE / POTENTIALLY VIOLENT PERSONS - Code GREY."""
    AGITATED_OR_AGGRESSIVE = "agitated_or_aggressive"
    WEAPON_OR_POLICE_ESCORT = "weapon_or_police_escort"


class PregnancyStatus(str, Enum):
    """Replaces the `pregnant: bool` this model used to carry.

    A boolean has no way to say "not established", so an unasked question and
    a negative answer were the same value. They are not the same clinically:
    before a teratogen, an unestablished status in a woman of childbearing age
    is a finding to raise, and a documented negative is a finding to rely on.
    """
    PREGNANT = "pregnant"
    NOT_PREGNANT = "not_pregnant"
    UNKNOWN = "unknown"


class Facility(str, Enum):
    """Where this assessment is happening. Set once in the UI and remembered,
    not asked per patient.

    It changes the RECOMMENDATION, never the level. A reperfusion decision is
    unanswerable without knowing whether there is a cath lab, and "refer to
    specialist" means different things from a Klinik Kesihatan and from a
    state hospital.
    """
    KLINIK_KESIHATAN = "klinik_kesihatan"
    DISTRICT_NO_SPECIALIST = "district_hospital_no_specialist"
    DISTRICT_WITH_SPECIALIST = "district_hospital_with_specialist"
    TERTIARY_PCI_CT = "state_or_tertiary_with_pci_and_ct"


_FACILITY_MEANING: dict[str, str] = {
    Facility.KLINIK_KESIHATAN: (
        "a primary-care Klinik Kesihatan: no inpatient beds, no on-site "
        "specialist, no CT and no cath lab. Definitive care means stabilise "
        "and transfer, and the transfer decision is part of the answer."
    ),
    Facility.DISTRICT_NO_SPECIALIST: (
        "a district hospital without resident specialists: inpatient beds and "
        "basic imaging, no PCI. Fibrinolysis and inter-hospital transfer are "
        "the reperfusion options, not primary PCI."
    ),
    Facility.DISTRICT_WITH_SPECIALIST: (
        "a district hospital with resident specialists: inpatient and HDU "
        "care and on-site specialist referral, but assume no cath lab unless "
        "a retrieved source says otherwise."
    ),
    Facility.TERTIARY_PCI_CT: (
        "a state or tertiary hospital with CT and a cath lab on site: primary "
        "PCI and thrombolysis are both available, and specialist teams can be "
        "referred to directly."
    ),
}


# Every modifier value, in the words a clinician would write.
#
# This map is load-bearing, not cosmetic. `red_flags.py`, `contraindications.py`
# and `completeness.py` all match REGULAR EXPRESSIONS against a prose picture of
# the patient. An enum name reaches none of them: a rule hunting for "asthma"
# will never see `Comorbidity.AIRWAY_DISEASE`. Spelling each value out here is
# what connects a checkbox to a guardrail, and it is the same rule already
# applied to the derived vitals in `_clinical_context` - fire on the
# observation, not on the model's wording.
_MODIFIER_PROSE: dict[str, str] = {
    # appearance (MTS p4, p7)
    _mts.APPEARANCE_NOT_DISTRESSED: "appearance: walking, talking, not distressed",
    _mts.APPEARANCE_UNWELL: "appears unwell",
    _mts.APPEARANCE_CANNOT_SIT_STAND: "cannot sit or stand unsupported",
    _mts.APPEARANCE_NOT_RESPONDING: "not responding to call",
    _mts.APPEARANCE_SEPTIC_ILL: "appears septic, toxic or critically ill",
    # breathing (MTS p4)
    _mts.BREATHING_NOT_BREATHLESS: "not breathless, no oxygen needed",
    _mts.BREATHING_WHEEZE_AIRWAY_INTACT: "wheeze with expiratory rhonchi, airway intact",
    _mts.BREATHING_NEEDS_OXYGEN: "needs oxygen support",
    _mts.BREATHING_SHORT_PHRASES: "difficulty breathing, speaking in short phrases only",
    _mts.BREATHING_ABNORMAL_SOUNDS: "abnormal airway sounds",
    _mts.BREATHING_EXCESSIVE_WORK: "excessive work of breathing with sweating",
    _mts.BREATHING_ONE_WORD: "cannot speak, one-word replies only",
    _mts.BREATHING_ASSISTED: "requires assisted breathing",
    # perfusion (MTS p4, p13)
    _mts.PERFUSION_WARM_NORMAL: "peripheries warm and pink, pulses normal",
    _mts.PERFUSION_CRT_OVER_2: "capillary refill time over 2 seconds",
    _mts.PERFUSION_WEAK_PULSES: "weak peripheral pulses with tachycardia",
    _mts.PERFUSION_COLD_OR_CYANOSED: "pale, cyanosed, cold peripheries",
    _mts.PERFUSION_ABSENT_RADIAL: "absent radial pulse",
    # bleeding (MTS p5)
    _mts.BLEEDING_NONE: "minimal or no active bleeding",
    _mts.BLEEDING_WOUND_OR_ENT:
        "active bleeding from a wound, fracture, joint, ENT site or menorrhagia",
    _mts.BLEEDING_HAEMATOMA_OR_DISORDER:
        "expanding haematoma, or bleeding in a known bleeding disorder",
    _mts.BLEEDING_VOMIT_OR_COUGH_BLOOD:
        "active bleeding: vomiting or coughing blood (haematemesis / haemoptysis)",
    _mts.BLEEDING_SUSPECTED_INTERNAL:
        "suspected internal bleeding - intra-abdominal, ectopic pregnancy, "
        "abdominal aortic aneurysm, vascular injury or compartment syndrome",
    _mts.BLEEDING_ARTERIAL_OR_UNCONTROLLED:
        "arterial, uncontrolled or massive active bleeding",
    # fever history (MTS p7)
    _mts.FEVER_REPORTED: "history of fever before arrival",
    _mts.FEVER_NONE_REPORTED: "no fever reported",
    # ECG (MTS p7). Spelled with the diagnoses each finding implies, because
    # that is the vocabulary the CPGs and the red-flag rules use - "ST
    # elevation" appears in the ACS guideline, "st_elevations_or_depressions"
    # appears nowhere.
    _mts.ECG_NORMAL: "ECG normal",
    _mts.ECG_NO_ST_T_CHANGES: "ECG shows no ST-T wave changes",
    _mts.ECG_NO_FINDINGS_ONGOING_PAIN:
        "ECG shows no acute findings but chest pain is continuing",
    _mts.ECG_AF_OVER_100: "ECG shows atrial fibrillation with a rate over 100",
    _mts.ECG_FREQUENT_ECTOPICS: "ECG shows frequent ectopics",
    _mts.ECG_BLOCK_OR_PAUSE: "ECG shows heart block or sinus pauses",
    _mts.ECG_TALL_TENTED_T:
        "ECG shows tall tented T waves, suggesting hyperkalaemia",
    _mts.ECG_ST_ELEVATION_OR_DEPRESSION:
        "ECG shows ST elevation or ST depression - acute coronary syndrome, "
        "STEMI or NSTEMI",
    _mts.ECG_WIDE_COMPLEX_TACHY:
        "ECG shows a wide complex tachycardia (ventricular tachycardia until "
        "proven otherwise)",
    _mts.ECG_NARROW_COMPLEX_TACHY_OVER_150:
        "ECG shows a narrow complex tachycardia over 150 per minute (SVT)",
    # comorbidities
    _mts.COMORBID_IMMUNOCOMPROMISED: "immunocompromised",
    _mts.COMORBID_ANTICOAGULANT: "on anticoagulant therapy",
    _mts.COMORBID_CKD: "chronic kidney disease or on dialysis (renal impairment)",
    _mts.COMORBID_AIRWAY_DISEASE:
        "obstructive airway disease - COPD or asthma",
    _mts.COMORBID_DIABETES: "diabetes mellitus",
    _mts.COMORBID_LIVER_DISEASE: "chronic liver disease (hepatic impairment)",
    # paediatric assessment triangle (MTS p13)
    _mts.PAED_SNORING_MUFFLED: "snoring, muffled or hoarse speech",
    _mts.PAED_STRIDOR_GRUNTING: "stridor, grunting or audible wheezing",
    _mts.PAED_SNIFFING_TRIPOD: "sniffing or tripod position",
    _mts.PAED_HEAD_BOBBING: "head bobbing",
    _mts.PAED_MOTTLING_CYANOSIS: "patchy, mottled or bluish skin discolouration",
    _mts.PAED_DIFFICULTY_SWALLOWING: "difficulty swallowing",
    _mts.PAED_DROOLING: "drooling",
    _mts.PAED_POSITIONAL_DISTRESS: "unable to walk, or refusing to lie down",
    _mts.PAED_RETRACTIONS:
        "supraclavicular, intercostal or substernal retractions",
    _mts.PAED_FLARING: "nasal flaring and accessory muscle use",
    _mts.PAED_PALLOR: "pale mucous membranes, soles or palms",
    # arrival (MTS p6)
    ArrivalMode.WALK_IN: "arrived as a walk-in",
    ArrivalMode.AMBULANCE: "arrived by ambulance",
    ArrivalMode.POLICE_OKT: "brought in by police (OKT)",
    ArrivalMode.REFERRED_FROM_CLINIC:
        "referred from a Klinik Kesihatan or another facility - the referral "
        "question must be answered explicitly",
    ArrivalMode.OSCC: "presented through the One-Stop Crisis Centre (OSCC)",
    # exposure (MTS p5). Placement, not level.
    ExposureRisk.SUSPECTED_TB:
        "suspected active tuberculosis - MTS p5 requires negative-pressure "
        "isolation and a surgical mask for the patient",
    ExposureRisk.FEBRILE_RESPIRATORY:
        "febrile respiratory illness - MTS p5 requires negative-pressure "
        "isolation and a surgical mask for the patient and relatives",
    ExposureRisk.OUTBREAK_OR_TRAVEL:
        "outbreak contact or recent travel to an affected area",
    ExposureRisk.KNOWN_MDRO:
        "known multi-drug resistant organism (CRE, MRSA) - MTS p5 requires an "
        "isolation room",
    ExposureRisk.CHEMICAL_HAZMAT:
        "chemical or HAZMAT exposure - MTS p5 requires decontamination and "
        "removal of clothing before entry to the department",
    # behaviour (MTS p6)
    BehaviouralRisk.AGITATED_OR_AGGRESSIVE:
        "agitated or aggressive - MTS p6 calls for verbal de-escalation and "
        "trained personnel before the patient enters the department",
    BehaviouralRisk.WEAPON_OR_POLICE_ESCORT:
        "weapon involved or under police escort - MTS p6 Code GREY",
}


# The same facts, reworded for the ONE consumer that pattern-matches them.
#
# `contraindications.py` decides a teratogen is absolutely contraindicated on
# the regex `\bpregnan`. Two display strings above contain that substring
# while describing a patient who is NOT known to be pregnant, and both would
# have produced a false ABSOLUTE:
#
#   * "suspected ... ectopic pregnancy" - and METHOTREXATE, on the teratogen
#     list, is the correct medical treatment for an ectopic. The check would
#     have alarmed on the right answer, which the guardrail rules forbid.
#   * "pregnancy status NOT established ..." - a prompt to ask the question,
#     turned by one substring into a hard contraindication.
#
# Rewording only the matching copy keeps the report readable AND the regex
# honest. Anything absent from this map matches on its display text.
_MATCHING_PROSE: dict[str, str] = {
    _mts.BLEEDING_SUSPECTED_INTERNAL:
        "suspected internal bleeding - intra-abdominal, ruptured ectopic, "
        "abdominal aortic aneurysm, vascular injury or compartment syndrome",
}

# What an unestablished pregnancy status looks like to a matching rule. It
# carries no "pregnan" substring on purpose; `contraindications.py` has a
# CAUTION rule keyed on this exact phrase, so the finding is deliberate and
# graded rather than an accident of wording.
UNESTABLISHED_GRAVID_PHRASE = (
    "patient of childbearing age, gravid status not established"
)

# And the same problem for a documented NEGATIVE. "pregnancy excluded" is the
# natural wording and it too contains "pregnan", so it fired the ABSOLUTE
# teratogen rule on the one patient who had definitively answered the
# question - the exact opposite of the intended behaviour. The matching copy
# says it without the substring.
EXCLUDED_GRAVID_PHRASE = "gravid status excluded, not gestating"


def modifier_prose(value: str) -> str:
    """The clinician-facing wording for one modifier value, or "" if unknown.

    Public because retrieval needs a few of these to build a query, and
    reaching into the map from another module would let a rename break
    silently.
    """
    return _MODIFIER_PROSE.get(value, "")


class TriageRequest(_Strict):
    age: float = Field(..., description="Age in years", ge=0, le=130)
    gender: Gender = Gender.UNKNOWN
    weight_kg: float | None = Field(None, description="Required for paediatric", ge=0, le=400)
    vitals: Vitals = Field(default_factory=Vitals)
    complaint: str = Field(..., min_length=3, max_length=4000)
    history: str = Field("", max_length=4000)

    # ---- Primary Triage (MTS pp4-5, p13) and Initial Tests (p7). Optional.
    appearance: GeneralAppearance | None = None
    breathing: Breathing | None = None
    perfusion: Perfusion | None = None
    bleeding: Bleeding | None = None
    fever_history: FeverHistory | None = None
    ecg_findings: List[EcgFinding] = Field(
        default_factory=list,
        description="MTS p7 initial test. Empty means no ECG was taken.",
    )
    paediatric_signs: List[PaediatricSign] = Field(
        default_factory=list,
        description="MTS p13 Paediatric Assessment Triangle danger signs.",
    )

    # ---- timing and context. Not MTS cells; these steer retrieval.
    onset: str = Field(
        "", max_length=200,
        description=(
            "Symptom onset, last known well, or day of illness - free text as "
            "the clinician says it (\"45 min ago\", \"day 5 of fever\"). The "
            "single strongest field for selecting the right SECTION of a CPG: "
            "the stroke thrombolysis window, the ACS reperfusion clock and the "
            "dengue phase are all read off it."
        ),
    )
    trauma_mechanism: str = Field("", max_length=300)
    arrival_mode: ArrivalMode | None = None

    # ---- background the prescribing checks need.
    allergies: str = Field(
        "", max_length=500,
        description="Free text, or NKDA. Parsed by contraindications.stated_allergies.",
    )
    current_medications: str = Field("", max_length=1000)
    comorbidities: List[Comorbidity] = Field(default_factory=list)
    egfr: float | None = Field(
        None, ge=0, le=200,
        description="eGFR ml/min/1.73m2, for renal dose adjustment.",
    )

    # ---- pregnancy. Tri-state; see PregnancyStatus.
    pregnancy: PregnancyStatus = PregnancyStatus.UNKNOWN
    gestation_weeks: float | None = Field(None, ge=0, le=45)
    breastfeeding: bool = False

    # ---- safety and placement (MTS pp5-6). Never change the level.
    exposure_risk: List[ExposureRisk] = Field(default_factory=list)
    behavioural_risk: List[BehaviouralRisk] = Field(default_factory=list)

    # ---- where this is happening. A remembered setting, not a per-patient field.
    facility: Facility | None = None

    @property
    def pregnant(self) -> bool:
        """Kept so every existing `req.pregnant` call site reads the same. It
        is now DERIVED: only an explicit PREGNANT counts, so UNKNOWN no longer
        silently means "no"."""
        return self.pregnancy is PregnancyStatus.PREGNANT

    @property
    def pregnancy_unestablished(self) -> bool:
        """A patient of childbearing age whose status was never established.

        Not a contraindication on its own - a prompt to establish it, graded
        as CAUTION by teratogen_when_gravid_status_unknown and only when a
        teratogen is actually recommended.

        Gender.UNKNOWN is deliberately EXCLUDED. Including it fired this on
        every intake that left the gender selector alone, which is most of
        them - a flag that appears on every patient is one nobody reads. An
        unrecorded gender means the intake is too thin to say anything here;
        an explicitly recorded FEMALE or OTHER is a real answer.
        """
        return (
            self.pregnancy is PregnancyStatus.UNKNOWN
            and self.gender in (Gender.FEMALE, Gender.OTHER)
            and 11 <= self.age <= 55
        )

    def observations(self) -> "_mts.Observations":
        """The intake as the MTS grid reads it."""
        return _mts.Observations(
            appearance=self.appearance.value if self.appearance else None,
            breathing=self.breathing.value if self.breathing else None,
            perfusion=self.perfusion.value if self.perfusion else None,
            bleeding=self.bleeding.value if self.bleeding else None,
            fever_history=self.fever_history.value if self.fever_history else None,
            ecg=frozenset(e.value for e in self.ecg_findings),
            comorbidities=frozenset(c.value for c in self.comorbidities),
            paediatric_signs=frozenset(s.value for s in self.paediatric_signs),
        )

    def facility_meaning(self) -> str:
        """What the recommendation may assume about this place, in words."""
        return _FACILITY_MEANING.get(self.facility, "") if self.facility else ""

    def modifier_lines(self, for_matching: bool = False) -> list[str]:
        """The optional intake, spelled into clinical prose.

        One string per recorded fact, phrased the way a clinician writes it -
        because everything downstream matches on the OBSERVATION, not on an
        enum name. This is the same lesson as the GTN patient whose
        hypotension was in the vitals and never in anyone's prose: a
        contraindication rule looking for "asthma" must find the word, not
        `Comorbidity.AIRWAY_DISEASE`.
        """
        def say(key: str) -> str:
            if for_matching and key in _MATCHING_PROSE:
                return _MATCHING_PROSE[key]
            return _MODIFIER_PROSE.get(key, "")

        out: list[str] = []
        for name in ("appearance", "breathing", "perfusion", "bleeding",
                     "fever_history", "arrival_mode"):
            value = getattr(self, name)
            if value is not None and (text := say(value.value)):
                out.append(text)
        for item in (*self.ecg_findings, *self.comorbidities,
                     *self.paediatric_signs, *self.exposure_risk,
                     *self.behavioural_risk):
            if text := say(item.value):
                out.append(text)
        if self.onset.strip():
            out.append(f"symptom onset / day of illness: {self.onset.strip()}")
        if self.trauma_mechanism.strip():
            out.append(f"mechanism of injury: {self.trauma_mechanism.strip()}")
        if self.allergies.strip():
            out.append(f"allergies: {self.allergies.strip()}")
        if self.current_medications.strip():
            out.append(f"current medications: {self.current_medications.strip()}")
        if self.egfr is not None:
            out.append(f"eGFR {self.egfr:g} ml/min/1.73m2")
            if self.egfr < 60:
                # Spelled in words for the same reason as everything else here:
                # three contraindication rules look for "renal impairment".
                out.append("renal impairment (eGFR below 60)")
            if self.egfr < 30:
                out.append("severe renal impairment (eGFR below 30)")
        if self.pregnancy is PregnancyStatus.PREGNANT:
            weeks = (f" at {self.gestation_weeks:g} weeks gestation"
                     if self.gestation_weeks is not None else
                     " (gestation not stated)")
            out.append(f"pregnant{weeks}")
        elif self.pregnancy is PregnancyStatus.NOT_PREGNANT:
            out.append(EXCLUDED_GRAVID_PHRASE if for_matching
                       else "pregnancy excluded")
        elif self.pregnancy_unestablished:
            out.append(
                UNESTABLISHED_GRAVID_PHRASE if for_matching else
                "pregnancy status NOT established in a patient of childbearing "
                "age - establish it before any teratogen or ionising imaging"
            )
        if self.breastfeeding:
            out.append("breastfeeding")
        return out

    def modifiers_text(self, for_matching: bool = False) -> str:
        lines = self.modifier_lines(for_matching)
        return "; ".join(lines) if lines else ""


# ====================================================== Mode A output (LLM)

class TriageLevel(int, Enum):
    RESUSCITATION = 1
    EMERGENCY = 2
    URGENT = 3
    EARLY_CARE = 4
    ROUTINE = 5

class TriageLabel(str, Enum):
    RESUSCITATION = "RESUSCITATION"
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    EARLY_CARE = "EARLY_CARE"
    ROUTINE = "ROUTINE"

class Confidence(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Disposition(str, Enum):
    RESUSCITATION_BAY = "RESUSCITATION_BAY"
    ADMIT_ICU_HDU = "ADMIT_ICU_HDU"
    ADMIT_WARD = "ADMIT_WARD"
    ED_OBSERVATION = "ED_OBSERVATION"
    REFER_SPECIALIST = "REFER_SPECIALIST"
    DISCHARGE_WITH_FOLLOW_UP = "DISCHARGE_WITH_FOLLOW_UP"


class Citation(_LLMOut):
    source_id: str = Field(default="[S1]")
    document: str = Field(default="KKM Document")
    page: str = Field(default="N/A")
    edition_year: str = Field(default="N/A")


class VitalInterpretation(_LLMOut):
    parameter: str = Field(default="Vitals")
    value: str = Field(default="N/A")
    interpretation: str = Field(default="N/A")
    mts_level_triggered: str = Field(default="")


class SourceCaution(BaseModel):
    """Server-set (F2): a "do not" quoted verbatim from a document about the
    working diagnosis."""
    label: str
    quote: str
    source_id: str = ""
    external: bool = False


class RedFlag(_LLMOut):
    flag: str = Field(default="Unspecified Red Flag")
    why_it_matters: str = Field(default="")
    source_id: str = Field(default="[S1]")
    # "" when the model wrote it; "source" when the server added it from a
    # verbatim guideline sentence (complications.py) - never model text.
    origin: str = Field(default="")


class DifferentialDiagnosis(_LLMOut):
    condition: str = Field(default="Unspecified Differential")
    discriminating_feature: str = Field(default="")


class PrimaryDiagnosis(_LLMOut):
    condition: str = Field(default="Unspecified Diagnosis")
    confidence: Confidence = Field(default=Confidence.MODERATE)
    reasoning: str = Field(default="")
    supporting_cpg: str = Field(default="")

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_confidence(cls, v: Any) -> Any:
        return _enum_token(v)


class ImmediateAction(_LLMOut):
    sequence: int = Field(default=1)
    action: str = Field(default="Assess ABCs")
    timeframe: str = Field(default="Immediately")
    source_id: str = Field(default="[S1]")
    origin: str = Field(default="")  # "source": added by the server from a verbatim sentence (F5)
    # Server-set with origin "source": the guideline sentence the action rests
    # on, shown as a sub-line under a short action (D4) - never model text.
    source_quote: str = Field(default="")
    source_where: str = Field(default="")


class Investigation(_LLMOut):
    test: str = Field(default="Standard Bloods")
    rationale: str = Field(default="")
    urgency: str = Field(default="Routine")
    source_id: str = Field(default="")
    origin: str = Field(default="")
    source_quote: str = Field(default="")
    source_where: str = Field(default="")


class DrugRecommendation(_LLMOut):
    drug_name: str = Field(default="Unspecified Drug")
    indication: str = Field(default="")
    adult_dose: str = Field(default="")
    paediatric_dose: str = Field(default="")
    route: str = Field(default="")
    frequency: str = Field(default="")
    duration: str = Field(default="")
    prescriber_category: str = Field(default="NOT_IN_RETRIEVED_SOURCES")
    prescriber_category_meaning: str = Field(default="")
    dose_verdict: str | None = Field(
        default=None,
        description=(
            "Server-set: VERIFIED / EXCEEDS_MAXIMUM / DIFFERS_FROM_SOURCE / "
            "NOT_COMPARABLE. Never model-authored."
        ),
    )
    dose_verdict_detail: str = Field(default="")
    dose_source_fukkm: str = Field(
        default="",
        description="FUKKM dosage text quoted VERBATIM, with its listing number.",
    )
    dose_source_cpg: str = Field(
        default="",
        description="CPG dose sentence quoted VERBATIM, with its page.",
    )
    indication_supported: bool | None = Field(
        default=None,
        description=(
            "Server-set, never model-authored. True when this drug is named in a "
            "clinical guideline chunk actually retrieved for THIS presentation. "
            "False means it was found only in the formulary catalogue, which "
            "lists every drug for every condition - the signature of a "
            "plausible-but-wrong recommendation."
        ),
    )
    indication_status: str | None = Field(
        default=None,
        description=(
            "Server-set by indications.py: SUPPORTED (a source links the drug to "
            "the working diagnosis), SYMPTOMATIC (it treats a symptom this patient "
            "has) or CONDITIONAL (linked only to a differential). WITHHELD drugs "
            "are moved to `withheld_drugs` and never appear here."
        ),
    )
    indication_basis: str = Field(
        default="",
        description="Server-set: the FUKKM field or guideline sentence the verdict rests on, quoted.",
    )
    cautions: str = Field(default="")
    source_id: str = Field(default="[S1]")


class ExternalReference(BaseModel):
    """Server-set. A verified link for the clinician to consult - chosen by code
    from data/reference_links.json, never written by the model."""
    title: str
    publisher: str = ""
    url: str
    reason: str = ""
    checked: str | None = None


class WithheldDrug(BaseModel):
    """Server-set. A drug the model proposed that no source links to this patient."""
    drug_name: str
    route: str = ""
    stated_dose: str = ""
    reason: str
    basis: str = ""


class Contraindication(BaseModel):
    """Server-set. A conflict between a recommendation and this patient."""
    rule: str
    severity: Literal["ABSOLUTE", "CAUTION"]
    where: str
    item: str
    trigger: str
    reason: str


class KnowledgeGap(BaseModel):
    """Server-set. Something this diagnosis needs that NO indexed KKM (or
    KKM-cited) document states - said plainly, with verified links to read
    instead. The model is never asked to fill it."""
    element: str
    statement: str
    references: List["ExternalReference"] = Field(default_factory=list)


class CompletenessGap(BaseModel):
    """Server-set. A required element of this presentation that the answer omitted.
    Never generated content - only the statement that something is missing."""
    element: str
    why: str
    guideline: str
    quote: str = Field(
        default="",
        description=(
            "Server-set: the guideline's own sentence that covers this element, "
            "quoted VERBATIM from the indexed document - never generated."
        ),
    )
    quote_source: str = Field(default="", description="Document title and page of `quote`.")


class DiagnosticSchema(_LLMOut):
    mts_triage_level: TriageLevel = Field(default=TriageLevel.URGENT)
    mts_triage_label: TriageLabel = Field(default=TriageLabel.URGENT)
    time_to_treatment: str = Field(
        default="under 30 minutes",
        description=(
            "Server-set from the final MTS level, never model-authored - it is a "
            "lookup against the five-row table on MTS 2022 page 3."
        ),
    )
    reassessment: str = Field(
        default="",
        description=(
            "Server-set. The MTS 2022 page 3 re-triage rule. Empty at Levels 1-2, "
            "where the patient is not waiting."
        ),
    )
    triage_rationale: str = Field(default="Based on clinical presentation.")
    vitals_interpretation: List[VitalInterpretation] = Field(default_factory=list)
    red_flags: List[RedFlag] = Field(default_factory=list)
    primary_diagnosis: PrimaryDiagnosis = Field(default_factory=lambda: PrimaryDiagnosis())
    differential_diagnoses: List[DifferentialDiagnosis] = Field(default_factory=list)
    immediate_actions: List[ImmediateAction] = Field(default_factory=list)
    investigations: List[Investigation] = Field(default_factory=list)
    drug_recommendations: List[DrugRecommendation] = Field(default_factory=list)
    prescriber_category_warning: str = Field(default="")
    drug_indication_warning: str = Field(
        default="",
        description="Server-set. Drugs not supported by any retrieved guideline.",
    )
    dose_completeness_warning: str = Field(
        default="",
        description="Server-set. Weight-based instructions missing an actual dose.",
    )
    contraindication_warning: str = Field(
        default="",
        description=(
            "Server-set, never model-authored. Drug-condition and drug-drug conflicts "
            "between a recommendation and this patient's own intake."
        ),
    )
    contraindications: List["Contraindication"] = Field(default_factory=list)
    withheld_drugs: List["WithheldDrug"] = Field(
        default_factory=list,
        description="Server-set: drugs removed by the indication gate, with the reason.",
    )
    second_pass_note: str = Field(
        default="",
        description="Server-set: what the diagnosis-conditioned second pass added, and from where.",
    )
    differential_warning: str = Field(
        default="",
        description="Server-set: differentials that restate or belong to the working diagnosis.",
    )
    source_cautions: List[SourceCaution] = Field(
        default_factory=list,
        description="Server-set: cautions for the working diagnosis, each quoted from its source.",
    )
    complication_note: str = Field(
        default="",
        description="Server-set: red flags added from a verbatim source sentence, or dropped "
                    "because they only restated the diagnosis.",
    )
    citation_alignment_note: str = Field(
        default="",
        description="Server-set: citations moved to the passage that supports the item.",
    )
    external_references: List["ExternalReference"] = Field(
        default_factory=list,
        description="Server-set: verified links for what the indexed corpus does not cover.",
    )
    triage_provisional: bool = Field(
        default=False,
        description="Server-set: a core vital sign was not measured, so the level can still rise.",
    )
    triage_provisional_note: str = Field(
        default="",
        description="Server-set: which vitals are missing and the printed cells that would escalate.",
    )
    onset_derived: str = Field(
        default="",
        description="Server-set: onset read from the complaint text when the onset field was left blank.",
    )
    avoid_warning: str = Field(
        default="",
        description=(
            "Server-set: a recommended drug that a source document says to avoid "
            "for this diagnosis, with the quoted sentence and its page."
        ),
    )
    completeness_warning: str = Field(
        default="",
        description="Server-set. Required elements of this presentation not addressed.",
    )
    completeness_gaps: List["CompletenessGap"] = Field(default_factory=list)
    knowledge_gaps: List["KnowledgeGap"] = Field(
        default_factory=list,
        description="Server-set. What KKM sources do not state for this diagnosis, with links.",
    )
    consistency_warning: str = Field(
        default="",
        description=(
            "Server-set. Statements in the report that contradict the recorded "
            "findings, rest on a finding that was never recorded, or contradict "
            "each other."
        ),
    )
    redirection_warning: str = Field(
        default="",
        description=(
            "Server-set. The report sends this patient out of the ETD, but the "
            "JKN Selangor Redirection Policy 2024 section 4.2 names them as one "
            "who must be seen in the ETD. Advisory: that policy is state-level."
        ),
    )
    citation_warning: str = Field(
        default="", description="Server-set. Uncited or invalid [S#] references."
    )
    urgency_warning: str = Field(
        default="",
        description=(
            "Server-set. Urgency the report claimed but the triage level does not "
            "support - red flags on a routine patient, STAT orders at MTS 4-5."
        ),
    )
    parse_warning: str = Field(
        default="",
        description=(
            "Server-set. Fields the model returned unusably, which therefore show "
            "their schema default rather than any model judgement."
        ),
    )
    disposition: Disposition = Field(default=Disposition.ED_OBSERVATION)
    disposition_justification: str = Field(default="")
    referral_required: bool = Field(default=False)
    referral_to: str = Field(default="")
    evidence_gaps: str = Field(default="")
    citations: List[Citation] = Field(default_factory=list)

    @field_validator("mts_triage_level", mode="before")
    @classmethod
    def _norm_level(cls, v: Any) -> Any:
        return _triage_int(v)

    @field_validator("disposition", mode="before")
    @classmethod
    def _norm_disposition(cls, v: Any) -> Any:
        return _enum_token(v)

    @field_validator("mts_triage_label", mode="before")
    @classmethod
    def _norm_label(cls, v: Any) -> Any:
        """Never fails: the label is derived from the level below, so an
        unrecognisable one costs no information and is not worth a 502."""
        token = _enum_token(v)
        return token if token in TriageLabel.__members__ else TriageLabel.URGENT

    @model_validator(mode="after")
    def _label_follows_level(self) -> "DiagnosticSchema":
        """The label is a rendering of the level, not an independent claim.

        Deriving it closes the only way the two could disagree - a report
        badged ROUTINE beside level 1 - and removes one more value the model
        can misspell. `_enforce_mts_level` already recomputes it this way when
        it moves the level, so this only makes the same rule apply at parse."""
        want = TriageLabel[TriageLevel(int(self.mts_triage_level)).name]
        if self.mts_triage_label is not want:
            self.mts_triage_label = want
        return self


# ---- server-side additions (not part of the LLM contract) ------------------

class RetrievedSource(BaseModel):
    source_id: str
    filename: str
    cpg_title: str
    edition_year: str
    doc_type: str
    page_number: int | None = None
    drug_name: str | None = None
    prescriber_category: str | None = None
    score: float | None = None
    retrieval: str | None = Field(
        default=None,
        description=(
            "How this source reached the answer: 'semantic', 'term match', "
            "'semantic+term match', or 'red-flag rule'. Lets a reader tell an "
            "embedding match from a literal clinical-term match from a "
            "deterministic safety rule."
        ),
    )
    excerpt: str
    url: str | None = Field(
        default=None,
        description="For web-only guidelines (the NAG 2024): the live page this passage came from.",
    )
    version_date: str | None = None
    status: str | None = "active"

class TriagePreview(BaseModel):
    """The deterministic part of a triage, available in about a second.

    Everything here is computed by code from the intake - the MTS 2022 table,
    the red-flag rules - with no model call, so it can be shown while the full
    report is still generating. It is a FLOOR: the full report can only keep or
    raise this level (bar the narrow de-escalation `_enforce_mts_level`
    documents), and it says so."""
    mts_triage_level: int
    triage_colour: Literal["RED", "YELLOW", "GREEN"]
    mts_triage_label: str
    time_to_treatment: str
    reassessment: str = ""
    reasons: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    provisional_note: str = ""
    note: str = (
        "Provisional, from the MTS 2022 table and red-flag rules alone. The full "
        "report may raise this level; verify before acting."
    )


class ReportFlag(BaseModel):
    """A clinician's one-click objection to an audited report."""
    reason: Literal["wrong_triage", "wrong_diagnosis", "wrong_drug", "missing_item",
                    "wrong_source", "other"]
    note: str = Field("", max_length=2000)
    flagged_by: str = Field("", max_length=200)


class ReportProvenance(BaseModel):
    """What produced this report, so it can be looked up and reproduced.

    Filled by the API layer after generation (see audit.py / versioning.py).
    `audit_id` is empty when auditing is off or the write failed."""
    audit_id: str = ""
    git_sha: str = ""
    git_dirty: bool = False
    code_fingerprint: str = ""
    corpus_fingerprint: str = ""
    queue_wait_ms: int = 0


class AttentionReport(BaseModel):
    """Server-set (A4.1): where the time went and what the model used.

    Stored with the report in the audit log, so `python -m app.attention_audit`
    can aggregate it across reports."""
    stages_ms: dict[str, int] = Field(default_factory=dict)
    prompt_tokens: int = 0
    prompt_tokens_reused: int = 0
    output_tokens: int = 0
    stage2_output_tokens: int = 0
    hypothesis_output_tokens: int = 0
    passages_given: int = 0
    passages_cited: int = 0
    cited: List[str] = Field(default_factory=list)
    uncited_documents: List[str] = Field(default_factory=list)
    cited_by_position: dict[str, int] = Field(default_factory=dict)
    given_by_position: dict[str, int] = Field(default_factory=dict)
    citation_moves: int = 0
    citations_unsupported: int = 0
    complications_added: int = 0
    second_pass: str = ""


class TriageResponse(BaseModel):
    diagnostic: DiagnosticSchema
    triage_colour: Literal["RED", "YELLOW", "GREEN"]
    request: TriageRequest = Field(
        ...,
        description=(
            "The intake this result was derived from, echoed verbatim so an "
            "exported or archived report carries its own inputs."
        ),
    )
    sources: List[RetrievedSource]
    model: str
    latency_ms: int
    corpus_warnings: List[str] = Field(default_factory=list)
    retrieval_notes: List[str] = Field(
        default_factory=list,
        description=(
            "Guidelines forced into context by a red-flag rule, each with the "
            "rule and the phrase in the intake that fired it."
        ),
    )
    provenance: ReportProvenance = Field(default_factory=ReportProvenance)
    attention: AttentionReport | None = None
    disclaimer: str = (
        "Decision-support prototype over KKM CPG/MTS/FUKKM documents. "
        "Not a medical device. Every recommendation must be verified against the "
        "cited source and the treating clinician's judgement before acting."
    )


# ============================================================== Mode B

class InquiryScope(str, Enum):
    ALL = "all"
    CPG_ONLY = "cpg_only"
    QUICK_REFERENCE_ONLY = "quick_reference_only"
    PAEDIATRIC_ONLY = "paediatric_only"
    TRIAGE_ONLY = "triage_only"
    FORMULARY_ONLY = "formulary_only"

class InquiryRequest(_Strict):
    query: str = Field(..., min_length=3, max_length=4000)
    scope: InquiryScope = InquiryScope.ALL
    max_sources: int = Field(24, ge=4, le=60)

class InquiryResponse(BaseModel):
    query: str
    answer_markdown: str
    sources: List[RetrievedSource]
    documents_scanned: List[str]
    model: str
    latency_ms: int
    corpus_warnings: List[str] = Field(default_factory=list)
    provenance: ReportProvenance = Field(default_factory=ReportProvenance)

# ============================================================== ops

class CorpusStats(BaseModel):
    collection: str
    total_chunks: int
    documents: List[dict]
    embedding_model: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    warnings: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    chroma_ready: bool
    total_chunks: int
    model: str
    model_loaded: bool = Field(
        False,
        description=(
            "True once the local MLX weights are resident in memory. The first "
            "request after boot loads them and is correspondingly slow."
        ),
    )
    gpu_busy: bool = Field(False, description="A report is being generated right now.")
    gpu_waiting: int = Field(0, description="Reports queued behind the one running.")
    warm: str = Field("off", description="Start-up warm-up: off | warming | ready (N s) | failed: ...")


class ProgressResponse(BaseModel):
    """Live progress of an in-flight Mode A run.

    `percent` is exact during retrieval and prefill. During decode it is an
    estimate against a calibrated typical output length, because the model's
    output length is not knowable in advance - `estimated` flags that so the UI
    can say so rather than imply a precision it does not have.
    """
    job_id: str
    stage: Literal["queued", "retrieval", "prefill", "decode", "checks", "done",
                   "error", "unknown"]
    percent: float
    detail: str
    done: bool
    estimated: bool = False
    error: str = ""
    elapsed_s: float = 0.0
