"""Regression tests for Phases 3-4 (2026-09-30) that need neither the model
nor the index: the new red-flag and completeness rules, the verified
reference links, gap quotes, the differential check, and the prompt-cache
prefix logic.

The second pass and the prompt cache themselves need the model and are tested
in the end-to-end evaluation run.

    ../.venv/bin/python -m tests.test_phase34
"""
import sys

from app import completeness as C
from app import rag_engine as R
from app import red_flags as F
from app import references as Rf
from app.schemas import (
    DiagnosticSchema, DifferentialDiagnosis, PrimaryDiagnosis, RedFlag, TriageRequest,
)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def fired(text):
    return {f.name for f, _ in F.match(text)}


HEAT = "MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency and Trauma Department"

# ===========================================================================
print("1. New red-flag rules fire on the presentations that motivated them - and not otherwise.")
# ===========================================================================
rhabdo = ("Severe bilateral lower extremity pain, generalised muscle weakness, and dark "
          "tea-coloured urine for two days. High-intensity workouts and marathons.")
check("rhabdomyolysis intake fires the rhabdomyolysis rule", "rhabdomyolysis" in fired(rhabdo), str(fired(rhabdo)))
check("and forces the heat-illness guideline in", HEAT in F.titles_for(rhabdo))
check("exertional heat stroke fires heat_illness",
      "heat_illness" in fired("Collapsed while running a 10 km race in the afternoon heat; confused."))
check("a snake bite fires snakebite", "snakebite" in fired("Bitten on the right foot by a snake 2 hours ago"))
check("tall tented T waves fire hyperkalaemia", "hyperkalaemia" in fired("ECG: tall tented T waves"))
check("'leg pain after a fall' fires none of the new rules",
      not fired("Leg pain after a fall") & {"rhabdomyolysis", "exertional_muscle_injury", "heat_illness"})
check("muscle ache without exertion or heat does not fire exertional_muscle_injury",
      "exertional_muscle_injury" not in fired("Muscle ache with a runny nose for two days"))
check("'fitness' does not trigger anything as 'fit'", "exertional_muscle_injury" not in fired("keeps fit"))

# ===========================================================================
print("2. Completeness checklists for the new presentations.")
# ===========================================================================
label, gaps = C.check("Rhabdomyolysis", "Administer IV fluids. Check CK.")
names = {g.element for g in gaps}
check("rhabdomyolysis is recognised", label == "Rhabdomyolysis", label)
check("the ECG gap is named", "12-lead ECG / cardiac monitoring" in names, str(names))
check("CK and fluids count as addressed",
      "Creatine kinase" not in names and not any(n.startswith("IV isotonic fluid") for n in names))
label, gaps = C.check("Exertional heat stroke", "Give paracetamol 1 g. Active cooling with ice packs.")
names = {g.element for g in gaps}
check("heat stroke flags antipyretics not withheld", "Antipyretics withheld" in names, str(names))
label, gaps = C.check("Exertional heat stroke", "Active cooling. Do not give paracetamol or NSAIDs.")
check("...and is satisfied when they are withheld", "Antipyretics withheld" not in {g.element for g in gaps})
el = C.element("Rhabdomyolysis", "Creatine kinase")
check("element() finds the Element behind a gap", el is not None and el.present)

# ===========================================================================
print("3. Gap quotes come verbatim from the named guideline, or not at all.")
# ===========================================================================
corpus = {
    "docs": ["6.1.8. Muscle enzymes (Creatinine kinase) should be taken for all suspected heat stroke.",
             "Aspirin should be given to all patients with ACS."],
    "metas": [{"cpg_title": HEAT, "doc_type": "CPG_FULL", "page_number": 10},
              {"cpg_title": "Management of Acute Coronary Syndromes", "doc_type": "CPG_FULL", "page_number": 5}],
}
gap = C.Gap("Heat stroke / heat-related illness", "End-organ work-up (CK, renal, liver, coagulation, glucose)",
            "why", "MOH Clinical Guidelines on Management of Heat Related Illness (2016)")
q, where = R._gap_quote(gap.presentation, gap, corpus, 21)
check("the heat guideline's own sentence is quoted", "Creatinine kinase" in q and "p10" in where, f"{q} | {where}")
gap2 = C.Gap("Rhabdomyolysis", "Compartment syndrome assessment", "why",
             "No KKM rhabdomyolysis guideline - MOH Heat Related Illness 2016")
q, _ = R._gap_quote(gap2.presentation, gap2, corpus, 55)
check("no qualifying sentence -> no quote, never a guess", q == "", q)

# ===========================================================================
print("4. Differentials must be alternatives, not the diagnosis again.")
# ===========================================================================
d = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis",
                                       reasoning="muscle breakdown after exercise, leading to myoglobinuria"),
    red_flags=[RedFlag(flag="Dark urine", why_it_matters="risk of acute kidney injury")],
    differential_diagnoses=[DifferentialDiagnosis(condition="Acute Kidney Injury"),
                            DifferentialDiagnosis(condition="Myoglobinuria"),
                            DifferentialDiagnosis(condition="Haemoglobinuria from haemolysis"),
                            DifferentialDiagnosis(condition="Rhabdomyolysis")],
)
R._check_differentials(d)
left = [x.condition for x in d.differential_diagnoses]
check("the complication named in the report's own red flag is moved out", "Acute Kidney Injury" not in left, str(left))
check("the finding named in its own reasoning is moved out", "Myoglobinuria" not in left, str(left))
check("the primary restated is moved out", "Rhabdomyolysis" not in left)
check("a real alternative stays", left == ["Haemoglobinuria from haemolysis"], str(left))
check("the removals are named", "Removed from the differentials" in d.differential_warning)
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="STEMI"),
                     differential_diagnoses=[DifferentialDiagnosis(condition="Aortic dissection")])
R._check_differentials(d)
check("an unrelated alternative is untouched", len(d.differential_diagnoses) == 1 and not d.differential_warning)
d = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
    red_flags=[RedFlag(flag="Weakness", why_it_matters="a systemic issue such as myopathy, rhabdomyolysis")],
    differential_diagnoses=[DifferentialDiagnosis(condition="Myopathy or neuromuscular disorder")])
R._check_differentials(d)
check("an alternative merely LISTED beside the diagnosis stays (2026-09-30 regression)",
      len(d.differential_diagnoses) == 1, d.differential_warning)
gap3 = C.Gap("Rhabdomyolysis", "Other causes of dark urine considered", "why", "MOH Heat Related Illness 2016")
corpus3 = {"docs": ["This guideline is a guide in initial management of heat exhaustion / heat stroke."],
           "metas": [{"cpg_title": HEAT, "doc_type": "CPG_FULL", "page_number": 2}]}
q, _ = R._gap_quote(gap3.presentation, gap3, corpus3, 55)
check("a scope sentence that merely names heat stroke is not quoted as 'other causes' (2026-09-30)", q == "", q)

# ===========================================================================
print("5. Verified links: chosen by code from the registry, never model-written.")
# ===========================================================================
check("the registry exists (python -m app.references --build)", len(Rf.registry()) > 10, str(len(Rf.registry())))
check("every registered URL is an https MOH/MaHTAS address",
      all(e["url"].startswith(("https://sites.google.com/moh.gov.my/", "https://mymahtas.moh.gov.my/"))
          for e in Rf.registry()))
sel = Rf.select("Community-acquired pneumonia", 60, need_guideline=False)
check("pneumonia links to NAG A11 respiratory infections", sel and sel[0]["id"] == "nag-a11-respiratory-infections",
      str([e["id"] for e in sel]))
sel = Rf.select("Sepsis", 50, need_guideline=False)
check("sepsis links to NAG A12 sepsis first", sel and sel[0]["id"] == "nag-a12-sepsis", str([e["id"] for e in sel]))
sel = Rf.select("Urinary tract infection", 6, need_guideline=False)
check("a child's UTI gets the paediatric section, never the adult one",
      sel and sel[0]["id"].startswith("nag-b") and all(not e["id"].startswith("nag-a") for e in sel),
      str([e["id"] for e in sel]))
check("STEMI gets no antimicrobial link", Rf.select("ST-elevation myocardial infarction", 58, False) == [])
sel = Rf.select("Exertional rhabdomyolysis", 55, need_guideline=True)
check("no governing guideline -> the MaHTAS CPG list is offered", sel and sel[0]["id"] == "mahtas-cpg-list")
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Community-acquired pneumonia",
                                                        supporting_cpg="Management of X"))
R._attach_references(d, TriageRequest(age=60, complaint="cough and fever"))
check("_attach_references puts the links on the report", d.external_references
      and "Antimicrobial" in d.external_references[0].title)

# ===========================================================================
print("6. Prompt cache: prefix arithmetic.")
# ===========================================================================
check("common prefix of identical lists is their length", R._common_prefix([1, 2, 3], [1, 2, 3]) == 3)
check("common prefix stops at the first difference", R._common_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2)
check("common prefix with an empty list is 0", R._common_prefix([], [1]) == 0)
check("the loop stop does not fire on ordinary JSON",
      not R._repetition('{"a": 1, "b": 2, "c": [' + ", ".join(f'"item {i}"' for i in range(80)) + "]}", times=8))

# ===========================================================================
print("7. The governing guideline must be about the diagnosis (second rhabdomyolysis run).")
# ===========================================================================
for dx, title, keep in [
    ("Rhabdomyolysis", "MOH Guideline Management of Snakebite", False),
    ("Exertional heat stroke", HEAT, True),
    ("Exertional heat stroke", "Management of Ischaemic Stroke", False),
    ("STEMI", "Management of Acute Coronary Syndromes", True),
    ("Hypertensive emergency", "Management of Hypertension", True),
    ("Pit viper envenoming", "MOH Guideline Management of Snakebite", True),
    ("Community-acquired pneumonia", "National Antimicrobial Guideline 2024 (Adults)", True),
    ("Rhabdomyolysis", "National Antimicrobial Guideline 2024 (Adults)", False),
    # The corpus has no allergic-rhinitis CPG; the rhinosinusitis CPG is not one.
    ("Allergic rhinitis", "Management of Rhinosinusitis in Adolescents and Adults", False),
]:
    d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition=dx, supporting_cpg=title))
    R._check_governing_topic(d)
    check(f"{dx} / {title[:38]} -> {'kept' if keep else 'removed'}",
          bool(d.primary_diagnosis.supporting_cpg) == keep)

# ===========================================================================
print("8. Rhabdomyolysis checklist: analgesia, alternatives, admission.")
# ===========================================================================
label, gaps = C.check("Rhabdomyolysis. Severe leg pain, pain score 7",
                      "IV fluids. CK. ECG. Urine output. Compartment. ED_OBSERVATION")
names = {g.element for g in gaps}
check("no analgesia -> gap", "Analgesia" in names, str(names))
check("no alternative causes -> gap", "Other causes of dark urine considered" in names)
check("ED observation -> admission gap", "Admission for IV fluids and serial CK" in names)
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                     differential_diagnoses=[DifferentialDiagnosis(condition="Haemoglobinuria from haemolysis")])
check("a differential now counts toward the answer text", "Haemoglobinuria" in R._answer_text(d))

# ===========================================================================
print("9. Instant triage card wording.")
# ===========================================================================
req = TriageRequest(age=55, complaint="Severe bilateral lower extremity pain, generalised muscle weakness, "
                    "and dark tea-coloured urine for two days.", history="Marathons.",
                    vitals={"gcs": 15, "pain_score": 7})
p = R.triage_preview(req)
check("red-flag rules are shown as words, not identifiers",
      all("_" not in f for f in p.red_flags), str(p.red_flags))
check("the '(ECG row)' audit tag is gone", "(ECG row)" not in p.provisional_note)
check("missing ECG and glucose are named", "No ECG or capillary glucose recorded" in p.provisional_note)
req2 = TriageRequest(age=40, complaint="unwell", ecg_findings=["normal_ecg"], vitals={
    "systolic_bp": 120, "diastolic_bp": 80, "heart_rate": 80, "respiratory_rate": 16, "spo2": 98,
    "temperature": 36.8, "capillary_blood_glucose": 5.5})
d = DiagnosticSchema()
R._flag_provisional_triage(d, req2)
check("a fully assessed intake is not provisional", not d.triage_provisional, d.triage_provisional_note)

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
