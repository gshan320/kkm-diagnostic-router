"""Regression tests for Phase 2 (2026-09-30): the right source for the right
patient, and nothing on the report that the code can derive left to the model.

    population     adult never sees paediatric-only sources, and the reverse
    citations      a [S#] must support the claim it is attached to
    vitals         "MTS level triggered" comes from the MTS table
    provisional    a level set without vital signs says so, with the cells
                   that would escalate it
    onset          read from the complaint when the field is blank
    leaning        a documented negative ("feels well") settles a qualifier
    preview        the code-only triage, same answer as the full run's floor

    ../.venv/bin/python -m tests.test_phase2
"""
import sys

from app import population as P
from app import rag_engine as R
from app.rag_engine import Retrieved
from app.schemas import (
    DiagnosticSchema, DrugRecommendation, ImmediateAction, PrimaryDiagnosis, RedFlag,
    TriageRequest, VitalInterpretation,
)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def chunk(title, doc_type="CPG_FULL", text="", page=1):
    return Retrieved(chunk_id=f"{title}-{page}", text=text,
                     metadata={"cpg_title": title, "doc_type": doc_type, "page_number": page,
                               "filename": f"{title}.pdf"})


PAEDS = chunk("Paediatric Protocols for Malaysian Hospitals", "PAEDIATRIC_PROTOCOL", "sea snake")
DENGUE_KIDS = chunk("Management of Dengue Fever in Children")
DENGUE_ADULTS = chunk("Management of Dengue Infection in Adults")
RHINO = chunk("Management of Rhinosinusitis in Adolescents and Adults")
MTS = chunk("Malaysian Triage Scale", "TRIAGE_PROTOCOL", "Pain Score 4 - 7 Temp 37.5 - 39 C")

# ===========================================================================
print("1. Population: which documents a patient may be grounded in.")
# ===========================================================================
check("Paediatric Protocols are paediatric", P.of(PAEDS.metadata) == P.PAEDIATRIC)
check("'Dengue Fever in Children' is paediatric", P.of(DENGUE_KIDS.metadata) == P.PAEDIATRIC)
check("'Dengue Infection in Adults' is adult", P.of(DENGUE_ADULTS.metadata) == P.ADULT)
check("'Adolescents and Adults' is for both", P.of(RHINO.metadata) == P.ALL)
check("MTS 2022 is for all ages", P.of(MTS.metadata) == P.ALL)
kept, notes = R._filter_population([PAEDS, DENGUE_KIDS, DENGUE_ADULTS, RHINO, MTS], 55)
titles = [c.metadata["cpg_title"] for c in kept]
check("a 55-year-old loses both paediatric sources",
      "Paediatric Protocols for Malaysian Hospitals" not in titles
      and "Management of Dengue Fever in Children" not in titles, str(titles))
check("and keeps adult, all-ages and MTS sources", len(kept) == 3, str(titles))
check("the removal is named", notes and "Population filter" in notes[0], str(notes))
kept, _ = R._filter_population([PAEDS, DENGUE_KIDS, DENGUE_ADULTS, RHINO, MTS], 8)
check("an 8-year-old loses the adult-only CPG and keeps the rest",
      [c.metadata["cpg_title"] for c in kept].count("Management of Dengue Infection in Adults") == 0
      and len(kept) == 4)
kept, _ = R._filter_population([PAEDS, DENGUE_KIDS, DENGUE_ADULTS], 15)
check("a 15-year-old may use both", len(kept) == 3)

d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(
    condition="Rhabdomyolysis", supporting_cpg="Paediatric Protocols for Malaysian Hospitals"))
R._check_population_sources(d, TriageRequest(age=55, complaint="leg pain"))
check("a paediatric governing guideline named for an adult is removed",
      d.primary_diagnosis.supporting_cpg == "" and "different age group" in d.citation_warning)
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(
    condition="Dengue", supporting_cpg="Management of Dengue Fever in Children"))
R._check_population_sources(d, TriageRequest(age=8, complaint="fever"))
check("the same guideline for a child is kept", d.primary_diagnosis.supporting_cpg != "")

# ===========================================================================
print("2. A citation must support its claim.")
# ===========================================================================
CPG = chunk("Management of Dengue Infection in Adults",
            text="Commence intravenous crystalloid fluid therapy and monitor urine output hourly.")
FUKKM = chunk("FUKKM", "DRUG_FORMULARY", "Paracetamol 500 mg Tablet. Mild to moderate pain and pyrexia.")
d = DiagnosticSchema(
    immediate_actions=[
        ImmediateAction(sequence=1, action="Administer intravenous fluids", source_id="[S1]"),
        ImmediateAction(sequence=2, action="Start intravenous crystalloid fluid therapy", source_id="[S2]"),
    ],
    drug_recommendations=[DrugRecommendation(drug_name="Heparin", source_id="[S3]"),
                          DrugRecommendation(drug_name="Paracetamol", source_id="[S3]")],
    red_flags=[RedFlag(flag="Reduced urine output", why_it_matters="oliguria", source_id="[S2]")],
)
notes, stripped = R._check_citation_support(d, [MTS, CPG, FUKKM])
check("an action citing the MTS table loses that citation",
      d.immediate_actions[0].source_id == "" and "action 1" in stripped, d.immediate_actions[0].source_id)
check("an action whose passage says the same thing keeps it", d.immediate_actions[1].source_id == "[S2]")
check("a drug cited to a passage that does not name it loses it",
      d.drug_recommendations[0].source_id == "")
check("a drug cited to its own FUKKM row keeps it", d.drug_recommendations[1].source_id == "[S3]")
check("a red flag sharing a clinical term with its passage keeps it", d.red_flags[0].source_id == "[S2]")
check("the removals are named", notes and "MTS" not in "" and "triage table" in notes[0], str(notes)[:200])

# ===========================================================================
print("3. 'MTS level triggered' is derived, never model-written.")
# ===========================================================================
req = TriageRequest(age=55, complaint="leg pain", vitals={"gcs": 15, "pain_score": 7})
d = DiagnosticSchema(vitals_interpretation=[
    VitalInterpretation(parameter="GCS", value="15", mts_level_triggered="5"),
    VitalInterpretation(parameter="Pain Score", value="7", mts_level_triggered="2"),
])
R._derive_vitals_levels(d, req)
check("GCS 15 triggers nothing (the model wrote 5)", d.vitals_interpretation[0].mts_level_triggered == "")
check("pain 7 triggers Level 3 (whatever the model wrote)", d.vitals_interpretation[1].mts_level_triggered == "3")

# ===========================================================================
print("4. A level without vital signs is marked provisional.")
# ===========================================================================
d = DiagnosticSchema()
R._flag_provisional_triage(d, req)
check("the rhabdomyolysis intake is provisional", d.triage_provisional)
check("it names what is missing", "not measured: BP, HR, RR, SpO2, temperature" in d.triage_provisional_note,
      d.triage_provisional_note[:120])
check("and quotes the Level 2 cells that would escalate it",
      "SBP < 90" in d.triage_provisional_note and "HR > 120" in d.triage_provisional_note)
vitals_only = {"systolic_bp": 120, "diastolic_bp": 80, "heart_rate": 80, "respiratory_rate": 16,
               "spo2": 98, "temperature": 36.8}
d = DiagnosticSchema()
R._flag_provisional_triage(d, TriageRequest(age=55, complaint="leg pain", vitals=vitals_only))
check("vitals measured but no ECG or glucose -> still provisional, naming the tests",
      d.triage_provisional and "No ECG or capillary glucose" in d.triage_provisional_note, d.triage_provisional_note)
full = TriageRequest(age=55, complaint="leg pain", ecg_findings=["normal_ecg"],
                     vitals={**vitals_only, "capillary_blood_glucose": 5.5})
d = DiagnosticSchema()
R._flag_provisional_triage(d, full)
check("a fully assessed intake (vitals, ECG, glucose) is not provisional", not d.triage_provisional)

# ===========================================================================
print("5. Onset is read from the complaint when the field is blank.")
# ===========================================================================
for text, want in [
    ("dark tea-coloured urine for two days.", "two days"),
    ("chest pain since this morning", "this morning"),
    ("weakness noticed 1 hour ago", "1 hour ago"),
]:
    d = DiagnosticSchema()
    R._derive_onset(d, TriageRequest(age=55, complaint=text))
    check(f"'{text}' -> {want}", d.onset_derived.startswith(want), d.onset_derived)
d = DiagnosticSchema()
R._derive_onset(d, TriageRequest(age=55, complaint="pain for two days", onset="3 days"))
check("a filled onset field is never overridden", d.onset_derived == "")

# ===========================================================================
print("6. A documented negative settles a leaning qualifier.")
# ===========================================================================
lvl, why = R._mts_floor(TriageRequest(age=40, complaint="Came for a sugar check; feels well.",
                                      vitals={"capillary_blood_glucose": 22}))
check("CBG 22 + 'feels well' -> Level 3 (printed '> 18 mmol/L no symptoms')", lvl == 3, str(why))
lvl, _ = R._mts_floor(TriageRequest(age=40, complaint="Came for a sugar check.",
                                    vitals={"capillary_blood_glucose": 22}))
check("CBG 22 with nothing written still leans Level 2", lvl == 2)
lvl, _ = R._mts_floor(TriageRequest(age=40, complaint="Feels well apart from a severe headache.",
                                    vitals={"systolic_bp": 230, "diastolic_bp": 135}))
check("positive evidence outranks 'feels well' (BP 230/135 + headache -> Level 2)", lvl == 2)

# ===========================================================================
print("7. The preview is the full run's floor, instantly.")
# ===========================================================================
p = R.triage_preview(req)
check("rhabdomyolysis preview: Level 3, under 30 minutes", p.mts_triage_level == 3
      and p.time_to_treatment == "under 30 minutes", p.model_dump_json()[:160])
check("the preview carries the provisional note", p.provisional_note.startswith("PROVISIONAL"))
check("it matches _mts_floor exactly", p.mts_triage_level == R._mts_floor(req)[0])

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
