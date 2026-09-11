"""Regression test for the 2026-09-08 'EARLY CARE' parse failure and its class.

The model returned the DISPLAY form of MTS level 4 - "EARLY CARE", which is what
the frontend renders - where the enum wanted "EARLY_CARE", and a 140-second
generation was discarded as a 502. Each check below is tied to a real failure or
to a right answer that must stay silent.

Run from the backend directory, no pytest needed:

    source ../.venv/bin/activate
    PYTHONPATH=$PWD python tests/test_enum_tolerance.py

Exits non-zero on any failure.
"""
import json, sys, pathlib
from pydantic import ValidationError
from app.schemas import DiagnosticSchema, TriageLabel, TriageLevel, Disposition, Confidence
from app import rag_engine as R

fails = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond: fails.append(name)

print("\n1. The exact payload that produced the 502")
d = DiagnosticSchema.model_validate({"mts_triage_level": 4, "mts_triage_label": "EARLY CARE"})
check("'EARLY CARE' now parses", d.mts_triage_label is TriageLabel.EARLY_CARE, str(d.mts_triage_label))

print("\n2. Label is derived from level, never trusted")
d = DiagnosticSchema.model_validate({"mts_triage_level": 1, "mts_triage_label": "ROUTINE"})
check("level 1 + label ROUTINE -> RESUSCITATION", d.mts_triage_label is TriageLabel.RESUSCITATION, str(d.mts_triage_label))
d = DiagnosticSchema.model_validate({"mts_triage_level": 5, "mts_triage_label": "total nonsense"})
check("unrecognisable label still parses, derived", d.mts_triage_label is TriageLabel.ROUTINE, str(d.mts_triage_label))
d = DiagnosticSchema.model_validate({"mts_triage_level": 2})
check("label absent from payload -> derived", d.mts_triage_label is TriageLabel.EMERGENCY, str(d.mts_triage_label))

print("\n3. Same class, other spellings and other enums")
for raw, want in [("early-care", 4), ("Early Care", 4), ("early_care", 4), ("  EARLY   CARE  ", 4)]:
    d = DiagnosticSchema.model_validate({"mts_triage_level": want, "mts_triage_label": raw})
    check(f"label {raw!r}", d.mts_triage_label is TriageLabel.EARLY_CARE)
for raw in ["Level 4", "MTS 4", "4", 4]:
    d = DiagnosticSchema.model_validate({"mts_triage_level": raw})
    check(f"level {raw!r} -> 4", int(d.mts_triage_level) == 4, str(d.mts_triage_level))
for raw, want in [("Discharge with follow-up", Disposition.DISCHARGE_WITH_FOLLOW_UP),
                  ("Admit ICU/HDU", Disposition.ADMIT_ICU_HDU),
                  ("resuscitation bay", Disposition.RESUSCITATION_BAY)]:
    d = DiagnosticSchema.model_validate({"mts_triage_level": 3, "disposition": raw})
    check(f"disposition {raw!r}", d.disposition is want, str(d.disposition))
d = DiagnosticSchema.model_validate({"mts_triage_level": 3, "primary_diagnosis": {"condition": "x", "confidence": "Moderate"}})
check("confidence 'Moderate'", d.primary_diagnosis.confidence is Confidence.MODERATE, str(d.primary_diagnosis.confidence))

print("\n4. Normalising is not guessing - a wrong value still fails")
for bad in ["VERY_URGENT", "TRIAGE_LEVEL_SIX", "MAYBE"]:
    try:
        DiagnosticSchema.model_validate({"mts_triage_level": 3, "disposition": bad}); ok = False
    except ValidationError: ok = True
    check(f"disposition {bad!r} rejected", ok)

print("\n5. Salvage: one bad field no longer costs the run")
payload = {"mts_triage_level": 5, "disposition": "SEND_HOME_MAYBE",
           "primary_diagnosis": {"condition": "Allergic rhinitis", "confidence": "HIGH"},
           "triage_rationale": "kept"}
try:
    DiagnosticSchema.model_validate(payload); exc = None
except ValidationError as e: exc = e
sal, dropped = R._salvage(json.dumps(payload), exc)
check("salvaged an object", sal is not None)
check("dropped names reported", dropped == ["disposition"], str(dropped))
check("good fields survived", sal and sal.primary_diagnosis.condition == "Allergic rhinitis" and sal.triage_rationale == "kept")
check("bad field on its default", sal and sal.disposition is Disposition.ED_OBSERVATION)

print("\n6. The triage level is never salvaged")
payload = {"mts_triage_level": "critically unwell", "triage_rationale": "x"}
try:
    DiagnosticSchema.model_validate(payload); exc = None
except ValidationError as e: exc = e
sal, dropped = R._salvage(json.dumps(payload), exc)
check("unparseable level refuses salvage", sal is None, f"{sal} {dropped}")

print("\n7. Guardrails still all present after the edits")
expected = ["_enforce_mts_level","_mts_floor","_ground_prescriber_categories","_check_drug_indications",
            "_check_dose_completeness","_check_dosing","_check_contraindications","_enforce_disposition",
            "_check_completeness","_check_citations","_flag_leaked_slots","_extract_json","_salvage"]
missing = [f for f in expected if not hasattr(R, f)]
check("all 13 functions exist", not missing, f"missing {missing}")

print("\n8. Your test case still floors at MTS 5 (no escalation)")
# Updated 2026-09-09. This block used to submit pain_score=1 and assert a floor
# of 5. That was wrong about the document, not about the code: MTS 2022 p7 makes
# Level 5 "No Fever / No Pain", and p7 also says the higher modifier wins where
# two do not differentiate - so ANY pain puts this patient at Level 4. A
# negative control must therefore record pain_score 0, or leave it blank.
from app.schemas import TriageRequest, Vitals
vitals = dict(systolic_bp=118, diastolic_bp=76, heart_rate=72, respiratory_rate=16,
              temperature=36.8, spo2=99, gcs=15)
req = TriageRequest(age=28, gender="male", complaint="Sneezing and a clear runny nose for three days.",
    vitals=Vitals(pain_score=0, **vitals))
floor, reasons = R._mts_floor(req)
check("floor is 5 with no reasons", floor == 5 and reasons == [], f"{floor} {reasons}")

req_mild = TriageRequest(age=28, gender="male", complaint="Sneezing and a clear runny nose for three days.",
    vitals=Vitals(pain_score=1, **vitals))
floor_mild, reasons_mild = R._mts_floor(req_mild)
check("the same patient with pain 1/10 floors at 4, not 5",
      floor_mild == 4 and reasons_mild, f"{floor_mild} {reasons_mild}")
d = DiagnosticSchema.model_validate({"mts_triage_level": 4, "mts_triage_label": "EARLY CARE"})
R._enforce_mts_level(d, req)
check("model's 4 kept, label consistent", int(d.mts_triage_level) == 4 and d.mts_triage_label is TriageLabel.EARLY_CARE, str(d.mts_triage_label))
d2 = DiagnosticSchema.model_validate({"mts_triage_level": 1, "mts_triage_label": "EARLY CARE"})
R._enforce_mts_level(d2, req)
check("over-triage 1 corrected to 4 + label follows", int(d2.mts_triage_level) == 4 and d2.mts_triage_label is TriageLabel.EARLY_CARE, f"{d2.mts_triage_level} {d2.mts_triage_label}")
check("correction is disclosed in rationale", "System Guardrail" in d2.triage_rationale, d2.triage_rationale)

print("\n9. Label removed from the prompt contract")
src = pathlib.Path(R.__file__).read_text()
tmpl = src.split('json_template = """', 1)[1].split('"""', 1)[0]
check("template no longer asks for the label", '"mts_triage_label"' not in tmpl)
check("template still asks for the level", '"mts_triage_level"' in tmpl)

print()
print(f"{len(fails)} failure(s)" if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
