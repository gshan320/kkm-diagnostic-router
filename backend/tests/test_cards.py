"""Regression tests for the generic, all-conditions layer (2026-09-30 plan):
condition cards (cards_build.py verification, cards.py run time), the source
policy, the symmetric indication vocabulary (G1), the oral-anticoagulant
triple-therapy rule (G2), state-conditioned quotes (D2/G4), finding
consistency (G7/G8), plan de-duplication (D1) and the KKM-cited reference
registry. Sections 1-6 need neither the model nor the index; section 7 checks
the live cards and qualifications against the index when both are present.

    ../.venv/bin/python -m tests.test_cards
"""
import json
import sys

from app import cards as K
from app import cards_build as B
from app import contraindications as CI
from app import indications as I
from app import rag_engine as R
from app import references as REF
from app import source_policy as SP
from app.schemas import (DiagnosticSchema, DifferentialDiagnosis, DrugRecommendation, ImmediateAction,
                         Investigation, PrimaryDiagnosis, TriageRequest)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def item(element, label, terms, quote, when=(), unless=(), core=True, setting="ed", external=False,
         chunk="c1", population=""):
    return K.Item(element=element, label=label, terms=tuple(terms), when=tuple(when), unless=tuple(unless),
                  core=core, setting=setting, population=population, quote=quote, chunk_id=chunk,
                  title="Test CPG", doc_type="CPG_FULL", page=12, url="", external=external)


ACS = K.Card(
    id="acs", name="Acute coronary syndrome", aliases=("ACS",), parents=(), population="adult",
    acuity="emergency", documents=("Test CPG",), items=(
        item("treatment", "Aspirin 300 mg loading", ["aspirin"], "Give aspirin 300 mg stat."),
        item("treatment", "Fluid bolus in RV infarct", ["fluid", "saline"],
             "Give a fluid bolus if there is no pulmonary congestion.", when=["shock"],
             unless=["pulmonary oedema"]),
        item("treatment", "Pharmaco-invasive strategy", ["fibrinolysis"],
             "In stable patients a pharmaco-invasive strategy with fibrinolysis is reasonable."),
        item("investigation", "Serial troponin", ["troponin"], "Repeat troponin at 3 hours."),
        item("admission", "Admit to CCU", ["ccu"], "All patients with ACS should be admitted to the CCU."),
        item("complication", "Cardiogenic shock", ["cardiogenic shock"],
             "Cardiogenic shock complicates 5% of cases."),
        item("avoid", "Avoid nitrates in hypotension", ["nitrate", "gtn"],
             "Nitrates should be avoided when systolic BP is below 90.", when=["hypotension"]),
    ))
STEMI = K.Card(id="stemi", name="ST-elevation myocardial infarction", aliases=("STEMI",),
               parents=("Acute coronary syndrome",), population="adult", acuity="emergency",
               documents=("Test CPG",), items=(
                   item("treatment", "Primary PCI", ["pci"], "Primary PCI is the preferred reperfusion."),))
PAED = K.Card(id="bronchiolitis", name="Bronchiolitis", aliases=(), parents=(), population="paediatric",
              acuity="urgent", documents=("Paeds",), items=(item("treatment", "Oxygen", ["oxygen"], "Give oxygen."),))


def use_cards(cs):
    if hasattr(K.load, "cache_clear"):
        K.load.cache_clear()
    K._names.cache_clear()
    K._max_words.cache_clear()
    K.load = lambda: tuple(cs)  # noqa: E731
    K._names.cache_clear()


_real_load = K.load
use_cards([ACS, STEMI, PAED])

# ===========================================================================
print("1. Build: packet filter and quote verification.")
# ===========================================================================
check("a recommendation is kept", B.keep_sentence("Aspirin 300 mg should be given to all patients."))
check("a study result is dropped", not B.keep_sentence("In a meta-analysis of 12 RCTs mortality fell (OR 0.8)."))
check("a study sentence that also recommends is kept",
      B.keep_sentence("Based on RCT evidence, aspirin is recommended in all patients."))
check("committee lists and disclaimers are dropped",
      not B.keep_sentence("External reviewers Dr Ali Consultant Physician Dr Tan Hospital Kuala Lumpur")
      and not B.keep_sentence("These guidelines are not legally binding and must be read with judgement."))
check("normalisation is whitespace-, case- and dash-blind",
      B.norm("Give  0.9%\nsaline – 20 ML/kg") == B.norm("give 0.9% saline - 20 ml/kg"))
check("a short term must stand alone", not B._term_in("ck", B.norm("known ckd stage 4"))
      or B._term_in("ck", B.norm("serum ck level")))

# ===========================================================================
print("2. Run time: lookup, lineage, population.")
# ===========================================================================
check("an abbreviation finds its card", K.lookup("ACS", 58).id == "acs")
check("the most specific card wins", K.lookup("Inferior STEMI with shock", 58).id == "stemi")
check("lineage adds the broader condition", [c.id for c in K.lineage(STEMI, 58)] == ["stemi", "acs"])
check("a paediatric card never serves an adult", K.lookup("Bronchiolitis", 40) is None
      and K.lookup("Bronchiolitis", 1).id == "bronchiolitis")
check("an unknown diagnosis has no card", K.lookup("Migraine", 30) is None)
check("a shorter form of a card's name finds it ('Myocardial infarction' -> STEMI card)",
      K.lookup("Myocardial infarction", 58) is not None and K.lookup("Myocardial infarction", 58).id == "stemi")
check("...but a generic one-word diagnosis does not", K.lookup("Pain", 58) is None)
check("card vocabulary expands a passage to its parent (G1)",
      "coronary" in K.synonyms("Give ticagrelor 180mg in STEMI", parents=True))

# ===========================================================================
print("3. Patient state: when / unless, stable-only quotes (G4), admission.")
# ===========================================================================
shock = "chest pain; hypotension hypotensive; shock haemodynamically unstable; tachycardia"
stable = "chest pain"
fluid = ACS.of("treatment")[1]
check("a 'when' item applies only in that state", K.applies(fluid, shock) and not K.applies(fluid, stable))
check("'unless' removes it", not K.applies(fluid, shock + " pulmonary oedema"))
check("stemmed state words match ('hypotensive' meets 'hypotension')", K.state_hit(("hypotension",), "hypotensive"))
pi = ACS.of("treatment")[2]
check("a stable-patients sentence never applies to a shocked patient (G4)",
      K.applies(pi, stable) and not K.applies(pi, shock))
answer = "Aspirin 300 mg. ECG. Troponin."
gaps = K.gaps([ACS], answer, shock, 58)
check("gaps are the core ED items not in the answer", [g.label for g in gaps] == ["Fluid bolus in RV infarct"],
      str([g.label for g in gaps]))
check("a term must appear as a word ('ck' is not 'ckd')",
      not K.present(item("investigation", "CK", ["ck"], "CK."), "known CKD")
      and K.present(item("investigation", "CK", ["ck"], "CK."), "serum CK"))
adm = K.admission([ACS], stable, 58)
check("an unconditional, core admission sentence raises to ICU/CCU",
      adm is not None and adm[1] == "ADMIT_ICU_HDU", str(adm))
check("cautions follow the state", [c.label for c in K.cautions([ACS], shock, 58)] == ["Avoid nitrates in hypotension"]
      and K.cautions([ACS], stable, 58) == [])
check("KKM-silent elements are named", "monitoring" in K.silent_elements([ACS]) and
      "treatment" not in K.silent_elements([ACS]))

# ===========================================================================
print("4. G1 symmetric indications, G2 oral anticoagulants.")
# ===========================================================================
working = I.concepts("Acute coronary syndrome")
sentence = "In STEMI give ticagrelor 180mg loading dose."
check("the passage's own abbreviation is expanded (ticagrelor for ACS)",
      I.mentions(working, sentence, symmetric=True) is not None)
check("...and one-sided matching is unchanged", I.mentions(I.concepts("Gout"), sentence, symmetric=True) is None)
ctx = "STEMI with cardiogenic shock"
hep = CI.check(ctx, [("drug: Heparin", "Heparin 60 units/kg IV")])
war = CI.check(ctx, [("drug: Warfarin", "Warfarin 5 mg")])
check("heparin in ACS is not called triple therapy",
      not any(f.rule == "anticoagulant_added_to_antiplatelets" for f in hep))
check("warfarin still is", any(f.rule == "anticoagulant_added_to_antiplatelets" for f in war))

# ===========================================================================
print("5. G7/G8 consistency, D1 de-duplication, D2 quote filters.")
# ===========================================================================
req = TriageRequest(age=58, vitals={"systolic_bp": 86, "heart_rate": 124, "spo2": 91}, complaint="chest pain")
d = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="STEMI", reasoning="Haemodynamically stable inferior STEMI."),
    differential_diagnoses=[DifferentialDiagnosis(condition="Cardiac tamponade",
                                                  discriminating_feature="does not have elevated JVP")],
    drug_recommendations=[DrugRecommendation(drug_name="Enoxaparin", indication="NSTE-ACS loading")])
R._check_finding_consistency(d, req)
w = d.consistency_warning
check("a stated finding the vitals contradict is named (G7)", "hypotension" in w, w)
check("an unrecorded examination finding cannot exclude (G7)", "JVP" in w, w)
check("two exclusive subtypes in one report are named (G8)", "NSTE-ACS" in w, w)
ok = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="STEMI", reasoning="Inferior STEMI."),
                      differential_diagnoses=[DifferentialDiagnosis(condition="NSTEMI", discriminating_feature="")])
R._check_finding_consistency(ok, TriageRequest(age=58, complaint="chest pain"))
check("a differential of the other subtype is not an inconsistency", ok.consistency_warning == "", ok.consistency_warning)
d = DiagnosticSchema(investigations=[Investigation(test="Renal function"), Investigation(test="Full blood count"),
                                     Investigation(test="Renal profile (BUSE)", source_id="[S3]",
                                                   source_quote="Send RP.", origin="source")],
                     immediate_actions=[ImmediateAction(sequence=1, action="IV access"),
                                        ImmediateAction(sequence=2, action="IV access.")])
R._dedupe_plan(d)
check("renal function listed twice is one row (D1)", [x.test for x in d.investigations]
      == ["Renal function", "Full blood count"], str([x.test for x in d.investigations]))
check("...which keeps the source the duplicate had", d.investigations[0].source_id == "[S3]")
check("a repeated action is one row", len(d.immediate_actions) == 1)
check("'not warranted' never answers a required element (D2)",
      R._NOT_NEEDED.search("once CK is clearly downtrending, repeat testing is not warranted") is not None)
check("a list of criteria ranks below a recommendation (D2)",
      R._listy("Admission criteria: CK > 20,000; AKI; dark urine; acidosis"))

# ===========================================================================
print("5b. Absolute contraindications leave the plan; hand + card checklists combine.")
# ===========================================================================
shocked = TriageRequest(age=58, vitals={"systolic_bp": 86, "heart_rate": 124}, complaint="chest pain")
d = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="STEMI"),
    immediate_actions=[ImmediateAction(sequence=1, action="Aspirin 300 mg"),
                       ImmediateAction(sequence=2, action="IV nitroglycerin 5-10 mcg/min")],
    drug_recommendations=[DrugRecommendation(drug_name="Nitroglycerin", adult_dose="5-10 mcg/min"),
                          DrugRecommendation(drug_name="Aspirin", adult_dose="300 mg")])
R._check_contraindications(d, shocked)
check("an absolutely contraindicated drug leaves the drug list",
      [x.drug_name for x in d.drug_recommendations] == ["Aspirin"], str([x.drug_name for x in d.drug_recommendations]))
check("...and the action that orders it leaves the actions",
      [a.action for a in d.immediate_actions] == ["Aspirin 300 mg"])
check("...both stay visible, with the reason", len(d.withheld_drugs) == 2
      and all("ABSOLUTELY CONTRAINDICATED" in w.reason for w in d.withheld_drugs)
      and "REMOVED FROM THE PLAN" in d.contraindication_warning)
use_cards([ACS, STEMI, PAED])
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Acute coronary syndrome"),
                     immediate_actions=[ImmediateAction(sequence=1, action="Aspirin 300 mg")])
label, gaps = R._gaps(TriageRequest(age=58, complaint="chest pain, ECG ST elevation"), d)
check("the card's items are checked even when a hand checklist matches",
      any(g.item is not None and g.element == "Serial troponin" for g in gaps), str([g.element for g in gaps]))
check("...alongside the hand checklist's own", any(g.item is None for g in gaps), str([g.element for g in gaps]))
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Acute coronary syndrome"))
R._state_knowledge_gaps(d, TriageRequest(age=58, complaint="chest pain"))
check("a missing element is stated as not FOUND, never as KKM silence",
      d.knowledge_gaps and all("found by the automated extraction" in g.statement
                               and "no indexed KKM document states" not in g.statement for g in d.knowledge_gaps))

# ===========================================================================
print("5c. Conditional and foreign-state items, merged rows, renumbering, withheld actions.")
# ===========================================================================
brady = item("treatment", "Consider transvenous pacing", ["pacing"], "Consider transvenous pacing.")
cond = item("treatment", "Glucagon", ["glucagon"], "Glucagon if calcium channel blocker overdose.")
CARD2 = K.Card(id="x", name="Acute coronary syndrome", aliases=("ACS",), parents=(), population="adult",
               acuity="emergency", documents=("Test CPG",), items=(brady, cond, ACS.items[3]))
ctx = lambda i: "BRADYCARDIA ALGORITHM atropine 500 mcg" if i is brady else ""  # noqa: E731
got = [i.label for i in K.checklist([CARD2], shock, 58, context=ctx)]
check("a quote stating its own condition is not required for everyone", "Glucagon" not in got, str(got))
check("an item from another state's algorithm is not required (bradycardia, HR 124)",
      "Consider transvenous pacing" not in got and "Serial troponin" in got, str(got))
check("...but is when the patient has that state",
      "Consider transvenous pacing" in [i.label for i in K.checklist([CARD2], "bradycardia", 58, context=ctx)])
check("a UA/NSTEMI line is the other subtype for a STEMI", R._other_subtype("STEMI", "TIMI RISK SCORE FOR UA/NSTEMI")
      and not R._other_subtype("ACS (STEMI/NSTEMI)", "TIMI RISK SCORE FOR UA/NSTEMI"))
d = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="STEMI"),
    immediate_actions=[ImmediateAction(sequence=1, action="Aspirin 300 mg"),
                       ImmediateAction(sequence=2, action="IV nitroglycerin 5-10 mcg/min"),
                       ImmediateAction(sequence=3, action="IV access")],
    drug_recommendations=[DrugRecommendation(drug_name="Nitroglycerin", adult_dose="5-10 mcg/min")])
R._check_contraindications(d, shocked)
R._renumber_actions(d)
check("one contraindication row per recommendation", len([c for c in d.contraindications
                                                          if c.item.lower() == "nitroglycerin"]) == 1,
      str([(c.item, c.where) for c in d.contraindications]))
check("...marked removed from the plan", all("removed from the plan" in c.where for c in d.contraindications))
check("actions are renumbered 1..n after a removal", [a.sequence for a in d.immediate_actions] == [1, 2])

# ===========================================================================
print("5d. Avoid beats recommend, dose purpose, wider negation, state as differential.")
# ===========================================================================
dop = item("treatment", "Initiate dopamine infusion", ["dopamine"], "Dopamine 5 - 20ug/kg/min IV infusion.")
dav = item("avoid", "Avoid dopamine in cardiogenic shock", ["dopamine"],
           "Dopamine should be avoided as it has been associated with a higher mortality.")
CARD3 = K.Card(id="y", name="Acute coronary syndrome", aliases=(), parents=(), population="adult",
               acuity="emergency", documents=("Test CPG",), items=(dop, dav, ACS.items[3]))
check("an item a KKM avoid sentence contradicts is never listed as missing",
      "Initiate dopamine infusion" not in [i.label for i in K.checklist([CARD3], shock, 58)])
check("a prophylaxis dose is not compared with a treatment dose",
      K.purpose_mismatch("enoxaparin 40mg OD may be considered until the patient is ambulant.", "1 mg/kg SC")
      and not K.purpose_mismatch("loading dose 300mg, maintenance 75mg daily", "300 mg"))
ckd = item("treatment", "Primary PCI in CKD", ["pci"], "In CKD patients presenting with STEMI, Primary PCI should be the strategy of choice.")
check("a subgroup line (In CKD patients ...) is conditional",
      "Primary PCI in CKD" not in [i.label for i in K.checklist([K.Card(id="z", name="ACS", aliases=(), parents=(),
          population="adult", acuity="emergency", documents=(), items=(ckd,))], shock, 58)])
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="ACS"),
                     differential_diagnoses=[DifferentialDiagnosis(condition="Pulmonary embolism",
                                                                   discriminating_feature="No evidence of dyspnea or hypoxia"),
                                             DifferentialDiagnosis(condition="Cardiogenic shock",
                                                                   discriminating_feature="hypotension")])
R._check_finding_consistency(d, TriageRequest(age=58, vitals={"systolic_bp": 86, "heart_rate": 124, "spo2": 91,
                                                              "respiratory_rate": 28}, complaint="chest pain"))
check("'no evidence of dyspnea or hypoxia' with SpO2 91 is caught", "hypoxia" in d.consistency_warning, d.consistency_warning)
check("a differential that is the patient's own state is moved out",
      [x.condition for x in d.differential_diagnoses] == ["Pulmonary embolism"])

from app.schemas import RedFlag  # noqa: E402
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Acute coronary syndrome"),
                     red_flags=[RedFlag(flag="Cardiogenic shock", why_it_matters='Source: "..."', origin="source")])
R._state_knowledge_gaps(d, TriageRequest(age=58, complaint="chest pain"))
check("no 'none found' statement for an element the report already quotes from KKM",
      not any(g.element in ("red_flag", "complication") for g in d.knowledge_gaps),
      str([g.element for g in d.knowledge_gaps]))

from app import completeness as CM  # noqa: E402
heat = [("so haemo concentration shown by\nelevated PCV and Hb.\n6.1.7. Renal Function Test\nAcute kidney injury due to "
         "inadequacy of volume, dehydration and\nmay also due to rhadomyolysis, or direct thermal injury to renal\n"
         "parenchyma.\n6.1.8. Muscle enzymes",
         {"cpg_title": "MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency "
                       "and Trauma Department", "doc_type": "CPG_FULL", "page_number": 10})]
lab, gs = CM.check("Rhabdomyolysis dark urine", "IV saline. CK. potassium.")
g = next(x for x in gs if "renal profile" in x.element)
q, w = R._gap_quote(lab, g, {"ids": ["h"], "docs": [heat[0][0]], "metas": [heat[0][1]]}, 55)
check("a heading split from its sentence by a PDF line wrap is still quoted",
      q.startswith("Renal Function Test Acute kidney injury") and q.endswith("parenchyma."), q)

# ===========================================================================
print("6. Source policy and the cited-reference registry.")
# ===========================================================================
champ = ("EXT CHAMP Clinical Practice Guideline for the Management of Exertional Rhabdomyolysis in "
         "Warfighters (US DoD, not KKM) 2025.pdf")
check("CHAMP is not allowed: no KKM document cites it", not SP.allowed(champ))
check("KKM and Malaysian documents are", SP.basis("CPG Management of Gout 2021.pdf")[0] == SP.KKM
      and SP.basis("MSN Consensus Guidelines on the Management of Epilepsy 2024.pdf")[0] == SP.MALAYSIAN_BODY)
check("WAO and GINA qualify by KKM citation",
      SP.basis("EXT WAO World Allergy Organization Anaphylaxis Guidance 2020.pdf")[0] == SP.KKM_CITATION
      and SP.basis("EXT GINA Global Strategy for Asthma Management and Prevention 2024.pdf")[0] == SP.KKM_CITATION)
check("WHO CRE stays by the user's decision",
      SP.basis("WHO Guidelines for the Prevention and Control of CRE, Acinetobacter and Pseudomonas in "
               "Health Care Facilities 2017.pdf")[0] == SP.USER_APPROVED)
check("every qualification names its basis and, if cited, where",
      all(q["basis"] in (SP.KKM_CITATION, SP.USER_APPROVED) and
          (q["basis"] != SP.KKM_CITATION or (q.get("cited_by") and q.get("page") and q.get("quote")))
          for q in SP.qualifications()))
check("a DOI entry reads back to its list number",
      REF._entry_text("12. Smith J. Guideline for X. BMJ 2020. doi:", 44).startswith("Smith J."))

# ===========================================================================
print("7. Live data: cards and qualifications against the index.")
# ===========================================================================
K.load = _real_load
K.load.cache_clear()
K._names.cache_clear()
K._max_words.cache_clear()
try:
    index = B.load_index()
except Exception as exc:  # no index on this machine
    index = None
    print(f"  (skipped: no index - {exc})")
if index and index["ids"]:
    pos = {cid: i for i, cid in enumerate(index["ids"])}
    for q in SP.qualifications():
        if q["basis"] != SP.KKM_CITATION:
            continue
        hit = any(B.norm(q["quote"]) in B.norm(d) for d, m in zip(index["docs"], index["metas"])
                  if m.get("cpg_title") == q["cited_by"] and m.get("page_number") == q["page"])
        check(f"qualification quote is on {q['cited_by'][:40]} p{q['page']}", hit, q["quote"])
    check("no chunk in the index comes from a document the policy rejects",
          all(SP.allowed(m.get("filename", "")) for m in index["metas"]))
    live = K.load()
    if live:
        bad = [(c.name, i.label) for c in live for i in c.items
               if i.chunk_id not in pos or B.norm(i.quote) not in B.norm(index["docs"][pos[i.chunk_id]])]
        check(f"every card quote ({sum(len(c.items) for c in live)}) is verbatim in its chunk", not bad, str(bad[:5]))
        check("every card item names at least one term found in its quote",
              all(any(B._term_in(t, B.norm(i.quote)) for t in i.terms) for c in live for i in c.items))
        check("no card item comes from a document outside the policy",
              all(SP.allowed(index["metas"][pos[i.chunk_id]].get("filename", ""))
                  for c in live for i in c.items if i.chunk_id in pos))
        acs = K.lookup("Acute coronary syndrome", 58)
        check("the live cards know ACS", acs is not None, "no card")
    else:
        print("  (skipped: data/condition_cards.json not built)")

print()
if fails:
    print(f"{len(fails)} FAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
