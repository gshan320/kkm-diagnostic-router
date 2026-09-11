"""Regression tests for the optional triage modifiers added 2026-09-10.

MTS 2022 page 7: the final level "takes into consideration all of the selected
parameters ie. Primary Triage, Vital Signs, Complaints List and Initial
Tests". The intake read two of those four. These tests cover the other two and
the background fields that go with them.

Three properties matter more than any individual rule, and each has a section
below:

    BLANK-SAFE      an unset field must match nothing, so no patient entered
                    the old way can change level because the fields exist
    NO CRY-WOLF     the reassuring values, and the comorbidities that are not
                    MTS cells, must escalate NOTHING
    ONE VOCABULARY  mts_table, schemas and the frontend each hold a copy of
                    every value string; a rename in one must fail here

Run from the backend directory, no pytest needed:

    PYTHONPATH=$PWD ../.venv/bin/python tests/test_triage_modifiers.py

Exits non-zero on any failure.
"""
import pathlib
import re
import sys

from app import contraindications, mts_table
from app.schemas import (
    ArrivalMode, BehaviouralRisk, Bleeding, Breathing, Comorbidity,
    DiagnosticSchema, DrugRecommendation, EcgFinding, ExposureRisk, Facility,
    FeverHistory, GeneralAppearance, PaediatricSign, Perfusion,
    PregnancyStatus, TriageRequest, Vitals, modifier_prose,
)
from app import rag_engine as R

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def req(age=40, complaint="unwell", **kw):
    vitals = kw.pop("vitals", None)
    return TriageRequest(age=age, complaint=complaint,
                         vitals=vitals or Vitals(), **kw)


def level(**kw):
    return R._mts_floor(req(**kw))[0]


# ===========================================================================
print("\n1. ONE VOCABULARY. The value strings live in three files; a rename")
print("   in any one of them must fail here, not in production.")
# ===========================================================================
PAIRS = [
    ("appearance", GeneralAppearance, mts_table.APPEARANCE_VALUES),
    ("breathing", Breathing, mts_table.BREATHING_VALUES),
    ("perfusion", Perfusion, mts_table.PERFUSION_VALUES),
    ("bleeding", Bleeding, mts_table.BLEEDING_VALUES),
    ("ecg", EcgFinding, mts_table.ECG_VALUES),
    ("fever history", FeverHistory, mts_table.FEVER_HISTORY_VALUES),
    ("comorbidity", Comorbidity, mts_table.COMORBIDITY_VALUES),
    ("paediatric signs", PaediatricSign, mts_table.PAEDIATRIC_SIGN_VALUES),
]
for name, enum, values in PAIRS:
    check(f"{name}: schema enum and mts_table agree, both directions",
          {m.value for m in enum} == set(values),
          f"schema={sorted(m.value for m in enum)} table={sorted(values)}")

ALL_ENUMS = [e for _, e, _ in PAIRS] + [ArrivalMode, ExposureRisk, BehaviouralRisk]
missing_prose = [m.value for enum in ALL_ENUMS for m in enum if not modifier_prose(m.value)]
check("every modifier value has clinician-facing prose", not missing_prose,
      str(missing_prose))

# The frontend is the third copy. It is plain text here on purpose: importing
# it is not possible, and a value that exists in Python but not in the form is
# a field the user can never set.
ts = pathlib.Path(__file__).resolve().parents[2] / "frontend/src/lib/modifiers.ts"
if ts.exists():
    in_ts = set(re.findall(r'value:\s*"([a-z0-9_]+)"', ts.read_text()))
    py = {m.value for enum in ALL_ENUMS for m in enum}
    py |= {m.value for m in PregnancyStatus} | {m.value for m in Facility}
    check("every backend modifier value is offered by the frontend form",
          py <= in_ts, f"missing from modifiers.ts: {sorted(py - in_ts)}")
    check("the frontend offers no value the backend would reject",
          in_ts <= py, f"unknown to the backend: {sorted(in_ts - py)}")
else:  # pragma: no cover
    check("frontend modifiers.ts found", False, str(ts))


# ===========================================================================
print("\n2. BLANK-SAFE. This is the whole promise of the feature: the fields")
print("   are optional, so an intake that ignores them must be unchanged.")
# ===========================================================================
check("empty adult intake is still Level 5", level() == 5)
check("empty child intake is still Level 4", level(age=6) == 4)
check("no observations -> no findings at all",
      mts_table.evaluate(40, Vitals()) == [])
shocked = Vitals(systolic_bp=86, heart_rate=124, respiratory_rate=28, spo2=91,
                 gcs=15, pain_score=9)
check("the shocked-STEMI case is still Level 1 with no modifiers",
      level(age=58, complaint="crushing chest pain, clammy", vitals=shocked) == 1)
check("an unset field is not the reassuring value",
      mts_table.Observations().appearance is None
      and mts_table.Observations().ecg == frozenset())


# ===========================================================================
print("\n3. NO CRY-WOLF. A correct, reassuring or merely-contextual answer")
print("   must stay silent - the rule the guardrails are judged by.")
# ===========================================================================
for name, kw in [
    ("appearance 'walking, talking, not distressed'",
     dict(appearance=GeneralAppearance.NOT_DISTRESSED)),
    ("breathing 'not breathless'", dict(breathing=Breathing.NOT_BREATHLESS)),
    ("perfusion 'warm, pulses normal'", dict(perfusion=Perfusion.WARM_NORMAL)),
    ("bleeding 'minimal or none'", dict(bleeding=Bleeding.NONE)),
    ("all four reassuring at once",
     dict(appearance=GeneralAppearance.NOT_DISTRESSED,
          breathing=Breathing.NOT_BREATHLESS,
          perfusion=Perfusion.WARM_NORMAL, bleeding=Bleeding.NONE)),
]:
    check(f"{name} -> still Level 5", level(**kw) == 5, f"got {level(**kw)}")

check("page 4's L3 'Peripheries warm, CRT normal' is NOT implemented",
      any("Peripheries warm" in cell
          for _, _, cell, _ in mts_table.DELIBERATELY_NOT_IMPLEMENTED))

# Page 5 lists anticoagulant therapy under BLEEDING "CHECK FOR", not in a
# level column. Reading it as a cell would make every warfarin patient Level 3.
check("on_anticoagulant alone escalates nothing",
      level(age=70, comorbidities=[Comorbidity.ANTICOAGULANT]) == 5)
for c in (Comorbidity.CKD, Comorbidity.AIRWAY_DISEASE, Comorbidity.DIABETES,
          Comorbidity.LIVER_DISEASE):
    check(f"{c.value} alone escalates nothing", level(comorbidities=[c]) == 5)
check("immunocompromised IS a printed cell and does escalate",
      level(comorbidities=[Comorbidity.IMMUNOCOMPROMISED]) == 2)

# Placement and context fields must never move the level.
check("exposure risk does not change the level",
      level(exposure_risk=[ExposureRisk.SUSPECTED_TB]) == 5)
check("behavioural risk does not change the level",
      level(behavioural_risk=[BehaviouralRisk.AGITATED_OR_AGGRESSIVE]) == 5)
check("arrival by ambulance does not change the level",
      level(arrival_mode=ArrivalMode.AMBULANCE) == 5)
check("the facility does not change the level",
      level(facility=Facility.KLINIK_KESIHATAN) == 5)


# ===========================================================================
print("\n4. Each new cell fires at the level page 4, 5, 7 or 13 prints for it.")
print("   Column membership was read from the PDF word coordinates.")
# ===========================================================================
CELLS = [
    # appearance - page 4 CRITICAL FIRST LOOK and page 7
    (dict(appearance=GeneralAppearance.SEPTIC_ILL), 2, "Appears Septic, Ill (p7)"),
    (dict(appearance=GeneralAppearance.NOT_RESPONDING), 2, "Not responding to call (p4)"),
    (dict(appearance=GeneralAppearance.UNWELL), 3, "Appears unwell (p7)"),
    (dict(appearance=GeneralAppearance.CANNOT_SIT_STAND), 3, "Cannot sit / stand unsupported (p4)"),
    # breathing - page 4 RESPIRATORY DISTRESS
    (dict(breathing=Breathing.ASSISTED), 1, "Require assisted breathing"),
    (dict(breathing=Breathing.ONE_WORD), 1, "Cannot speak; one-word reply"),
    (dict(breathing=Breathing.EXCESSIVE_WORK), 1, "Excessive work of breathing"),
    (dict(breathing=Breathing.ABNORMAL_SOUNDS), 1, "Abnormal sounds"),
    (dict(breathing=Breathing.SHORT_PHRASES), 2, "Short phrases only"),
    (dict(breathing=Breathing.NEEDS_OXYGEN), 3, "Need O2 support"),
    (dict(breathing=Breathing.WHEEZE_AIRWAY_INTACT), 3, "Wheeze, airway intact"),
    # perfusion - page 4 SHOCK STATE
    (dict(perfusion=Perfusion.COLD_OR_CYANOSED), 1, "Pale, cyanosed, cold peripheries"),
    (dict(perfusion=Perfusion.ABSENT_RADIAL), 1, "Absent Radial Pulse"),
    (dict(perfusion=Perfusion.WEAK_PULSES), 2, "Tachycardia, Weak Pulses"),
    (dict(perfusion=Perfusion.CRT_OVER_2), 2, "CRT > 2 seconds"),
    # bleeding - page 5
    (dict(bleeding=Bleeding.ARTERIAL_OR_UNCONTROLLED), 1, "Arterial Limb Bleeding"),
    (dict(bleeding=Bleeding.VOMIT_OR_COUGH_BLOOD), 2, "Active Vomit / Cough Blood"),
    (dict(bleeding=Bleeding.SUSPECTED_INTERNAL), 2, "Suspected Intra-Abdominal / Ectopic / AAA"),
    (dict(bleeding=Bleeding.WOUND_OR_ENT), 3, "Bleeding from Fractures / Wounds"),
    (dict(bleeding=Bleeding.HAEMATOMA_OR_DISORDER), 3, "Expanding haematoma"),
    # ECG - page 7 INITIAL TESTS
    (dict(ecg_findings=[EcgFinding.WIDE_COMPLEX_TACHY]), 2, "Wide Complex Tachycardia"),
    (dict(ecg_findings=[EcgFinding.NARROW_COMPLEX_TACHY_OVER_150]), 2, "Narrow Complex Tachycardia > 150"),
    (dict(ecg_findings=[EcgFinding.ST_ELEVATION_OR_DEPRESSION]), 2, "ST elevations or depressions"),
    (dict(ecg_findings=[EcgFinding.AF_OVER_100]), 3, "Atrial Fibrillation > 100"),
    (dict(ecg_findings=[EcgFinding.FREQUENT_ECTOPICS]), 3, "Frequent ectopics"),
    (dict(ecg_findings=[EcgFinding.BLOCK_OR_PAUSE]), 3, "Blocks / Sinus Pauses"),
    (dict(ecg_findings=[EcgFinding.TALL_TENTED_T]), 3, "Tall Tented T waves"),
    (dict(ecg_findings=[EcgFinding.NO_FINDINGS_ONGOING_PAIN]), 4, "No ECG findings; continuing chest pain"),
    (dict(ecg_findings=[EcgFinding.NORMAL]), 5, "Normal ECG"),
    (dict(ecg_findings=[EcgFinding.NO_ST_T_CHANGES]), 5, "No ST-T wave changes"),
]
for kw, want, cell in CELLS:
    got = level(**kw)
    check(f"{cell} -> Level {want}", got == want, f"got {got}")

check("the most acute cell wins across fields",
      level(appearance=GeneralAppearance.UNWELL,
            perfusion=Perfusion.ABSENT_RADIAL,
            ecg_findings=[EcgFinding.AF_OVER_100]) == 1)


# ===========================================================================
print("\n5. The two levels the old record had WRONG. Both were transcribed")
print("   from the flattened text layer, which prints each column in turn.")
# ===========================================================================
# "ST elevations or depressions" sits at x=131 on page 7 - the LEVEL 2 column,
# beside "Bradycardia < 40" - not with the Level 3 group it was listed under.
check("ST elevation is Level 2, not the Level 3 it was recorded as",
      level(ecg_findings=[EcgFinding.ST_ELEVATION_OR_DEPRESSION]) == 2)
check("  and AF > 100, from the same old bundle, really is Level 3",
      level(ecg_findings=[EcgFinding.AF_OVER_100]) == 3)
check("  the correction is on the deviation record",
      any(d.status == mts_table.STATUS_CORRECTED and "ST elevation" in d.what
          for d in mts_table.deviations()))

# "History of Fever" sits at x=358 - the LEVEL 4 column, not Level 3. That is
# why it was safe to implement: it cannot escalate a febrile patient, because
# a measured fever is already Level 3 or 2 on the temperature row above it.
check("fever reported, afebrile now -> Level 4",
      level(vitals=Vitals(temperature=36.9),
            fever_history=FeverHistory.REPORTED) == 4)
check("fever reported AND measured 38.2 -> Level 3 on the temperature",
      level(vitals=Vitals(temperature=38.2),
            fever_history=FeverHistory.REPORTED) == 3)
check("fever reported, temperature never taken -> Level 4",
      level(fever_history=FeverHistory.REPORTED) == 4)
check("no fever reported and none measured -> Level 5",
      level(vitals=Vitals(temperature=36.8),
            fever_history=FeverHistory.NONE_REPORTED) == 5)


# ===========================================================================
print("\n6. The paediatric hole. A compensating child holds its blood")
print("   pressure until it does not; page 14 has no danger-sign row at all.")
# ===========================================================================
well_child = Vitals(spo2=97, heart_rate=110, respiratory_rate=26, temperature=37.0)
check("child with normal-for-age vitals and no signs -> Level 4 floor",
      level(age=4, vitals=well_child) == 4)
check("  the SAME child with stridor -> Level 1",
      level(age=4, vitals=well_child,
            paediatric_signs=[PaediatricSign.STRIDOR_GRUNTING]) == 1)
check("  with retractions -> Level 2",
      level(age=4, vitals=well_child,
            paediatric_signs=[PaediatricSign.RETRACTIONS]) == 2)
check("  with mottled skin -> Level 1",
      level(age=4, vitals=well_child,
            paediatric_signs=[PaediatricSign.MOTTLING_CYANOSIS]) == 1)
check("  with CRT > 2 s -> Level 2 (perfusion, every age)",
      level(age=4, vitals=well_child, perfusion=Perfusion.CRT_OVER_2) == 2)
check("paediatric signs on an ADULT intake are ignored",
      level(age=40, paediatric_signs=[PaediatricSign.STRIDOR_GRUNTING]) == 5)
check("page 7 rows a child has no counterpart for still apply (ECG)",
      level(age=4, vitals=well_child,
            ecg_findings=[EcgFinding.WIDE_COMPLEX_TACHY]) == 2)


# ===========================================================================
print("\n7. The modifiers reach the layers that match on TEXT. A tick-box")
print("   the guardrails cannot see is not a safety feature.")
# ===========================================================================
r = req(age=58, complaint="chest pain",
        perfusion=Perfusion.COLD_OR_CYANOSED,
        comorbidities=[Comorbidity.AIRWAY_DISEASE],
        allergies="penicillin", current_medications="sildenafil PRN",
        egfr=38, onset="45 minutes ago")
intake = R._intake_text(r)
check("perfusion reaches the intake text", "cold peripheries" in intake)
check("a comorbidity reaches it as a WORD the rules look for",
      "asthma" in intake.lower())
check("an allergy reaches it", "penicillin" in intake)
check("a current medication reaches it", "sildenafil" in intake)
check("eGFR 38 is spelled as 'renal impairment', which three rules match",
      "renal impairment" in intake)
check("onset reaches it", "45 minutes ago" in intake)
check("onset and ECG reach the RETRIEVAL query", "45 minutes ago" in R._retrieval_terms(r))
check("arrival mode does NOT pad the retrieval query",
      "ambulance" not in R._retrieval_terms(
          req(arrival_mode=ArrivalMode.AMBULANCE)))

# The perfusion tick must settle the shock_signs qualifier the same way a
# typed "cold peripheries" always did.
findings = mts_table.evaluate(
    58, Vitals(systolic_bp=86), intake,
    req(age=58, perfusion=Perfusion.COLD_OR_CYANOSED).observations())
check("a ticked perfusion box settles the shock_signs qualifier",
      any(f.level == 1 and f.parameter == "BP" for f in findings),
      str([f.reason() for f in findings]))


# ===========================================================================
print("\n8. Pregnancy is tri-state, and 'not asked' must not become a")
print("   contraindication - methotrexate TREATS an ectopic.")
# ===========================================================================
check("the bool is gone; UNKNOWN is the default",
      req().pregnancy is PregnancyStatus.UNKNOWN)
check("`pregnant` is now derived, and UNKNOWN is not 'no'",
      req().pregnant is False
      and req(pregnancy=PregnancyStatus.PREGNANT).pregnant is True)
check("unestablished fires for a woman of childbearing age",
      req(age=28, gender="female").pregnancy_unestablished)
check("  not once it is answered",
      not req(age=28, gender="female",
              pregnancy=PregnancyStatus.NOT_PREGNANT).pregnancy_unestablished)
check("  not for a man", not req(age=28, gender="male").pregnancy_unestablished)
check("  not outside childbearing age",
      not req(age=70, gender="female").pregnancy_unestablished)
# Gender.UNKNOWN is excluded on purpose: it is the default selector value, so
# including it fired this flag on almost every intake.
check("  not when the gender selector was never touched",
      not req(age=28).pregnancy_unestablished)

not_pregnant = req(age=28, gender="female",
                   bleeding=Bleeding.SUSPECTED_INTERNAL)
matching = not_pregnant.modifiers_text(for_matching=True)
display = "; ".join(not_pregnant.modifier_lines())
check("the DISPLAY text says 'ectopic pregnancy' and 'pregnancy status'",
      "ectopic pregnancy" in display and "pregnancy status" in display.lower())
check("the MATCHING text carries no 'pregnan' substring at all",
      not re.search(r"\bpregnan", matching, re.I), matching)


def findings_for(request, drug):
    """The same path rag_engine takes: build the patient picture, then test one
    recommendation against it."""
    d = DiagnosticSchema(drug_recommendations=[DrugRecommendation(drug_name=drug)])
    return contraindications.check(
        R._clinical_context(request, d), [("drug_recommendations", drug)]
    )


names = {f.rule for f in findings_for(not_pregnant, "methotrexate")}
check("methotrexate for a suspected ectopic is NOT an ABSOLUTE teratogen alert",
      "teratogen_in_pregnancy" not in names, str(names))
check("  it is a graded CAUTION instead, naming the unanswered question",
      "teratogen_when_gravid_status_unknown" in names, str(names))
check("  and that finding is severity CAUTION",
      all(f.severity == "CAUTION" for f in findings_for(not_pregnant, "methotrexate")
          if f.rule == "teratogen_when_gravid_status_unknown"))

really_pregnant = req(age=28, gender="female", pregnancy=PregnancyStatus.PREGNANT,
                      gestation_weeks=12)
names = {f.rule for f in findings_for(really_pregnant, "warfarin")}
check("a DOCUMENTED pregnancy still raises the ABSOLUTE teratogen rule",
      "teratogen_in_pregnancy" in names, str(names))
check("gestation is carried into the picture",
      "12 weeks gestation" in "; ".join(really_pregnant.modifier_lines()))

known_negative = req(age=28, gender="female",
                     pregnancy=PregnancyStatus.NOT_PREGNANT)
names = {f.rule for f in findings_for(known_negative, "warfarin")}
check("a documented negative raises NEITHER pregnancy rule",
      not ({"teratogen_in_pregnancy", "teratogen_when_gravid_status_unknown"}
           & names), str(names))


# ===========================================================================
print("\n9. The facility shapes the recommendation and nothing else.")
# ===========================================================================
kk = req(age=58, complaint="chest pain", facility=Facility.KLINIK_KESIHATAN)
check("the facility is explained to the model in words",
      "no cath lab" in kk.facility_meaning())
check("  and reaches the prompt", "Setting:" in R._patient_extras(kk))
check("an unset facility says nothing at all",
      req(complaint="chest pain").facility_meaning() == "")
check("a bare intake adds no PATIENT lines",
      R._patient_extras(req(complaint="sore throat")) == "")


print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("All checks passed.")
