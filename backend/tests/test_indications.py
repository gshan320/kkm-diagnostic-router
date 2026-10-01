"""Regression tests for Phase 1 (2026-09-30): the drug indication gate, the
mined avoid-statements, and the truncated-answer flag.

Every check is tested against a RIGHT answer as well as the wrong one - a gate
that withholds heparin for rhabdomyolysis but also for a DVT is not a fix.

    ../.venv/bin/python -m tests.test_indications
"""
import sys

from app import indications as I
from app import negatives as N
from app import rag_engine as R
from app.schemas import (
    DiagnosticSchema, DifferentialDiagnosis, DrugRecommendation, ImmediateAction,
    PrimaryDiagnosis, TriageRequest,
)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def report(dx, drugs, differentials=(), actions=()):
    return DiagnosticSchema(
        primary_diagnosis=PrimaryDiagnosis(condition=dx),
        differential_diagnoses=[DifferentialDiagnosis(condition=d) for d in differentials],
        drug_recommendations=[DrugRecommendation(drug_name=n, route=r, adult_dose=dose)
                              for n, r, dose in drugs],
        immediate_actions=[ImmediateAction(sequence=i + 1, action=a) for i, a in enumerate(actions)],
    )


def gate(dx, drugs, differentials=(), actions=(), age=55, pain=0, temp=None, complaint="unwell"):
    req = TriageRequest(age=age, complaint=complaint, vitals={"pain_score": pain, "temperature": temp})
    d = report(dx, drugs, differentials, actions)
    R._gate_drug_indications(d, req, [])
    return d


def kept(d):
    return {x.drug_name: x.indication_status for x in d.drug_recommendations}


def withheld(d):
    return [w.drug_name for w in d.withheld_drugs]


HEPARIN = ("Heparin", "IV", "5000 units")

# ===========================================================================
print("1. The motivating failure: heparin for exertional rhabdomyolysis.")
# ===========================================================================
d = gate("Rhabdomyolysis", [HEPARIN], differentials=["Acute Kidney Injury", "Myoglobinuria"],
         actions=["Administer intravenous fluids to maintain hydration"], pain=7,
         complaint="Severe bilateral lower extremity pain and dark tea-coloured urine")
check("heparin is withheld", withheld(d) == ["Heparin"], str(withheld(d)))
check("and removed from the drug list", "Heparin" not in kept(d))
check("the warning names heparin and quotes FUKKM",
      "WITHHELD" in d.drug_indication_warning and "Heparin" in d.drug_indication_warning
      and "FUKKM" in d.drug_indication_warning, d.drug_indication_warning[:200])
R._check_dosing(d, TriageRequest(age=55, complaint="unwell"), [])
check("a withheld drug is never dose-checked (no DOSE VERIFIED)",
      all(x.drug_name != "Heparin" for x in d.drug_recommendations))

# ===========================================================================
print("2. Right answers stay silent.")
# ===========================================================================
d = gate("Deep vein thrombosis of the left leg", [HEPARIN])
check("heparin passes for a DVT", kept(d).get("Heparin") == I.SUPPORTED, str(kept(d)))
d = gate("Acute coronary syndrome", [HEPARIN, ("Aspirin", "oral", "300 mg")])
check("heparin passes for ACS (FUKKM: myocardial infarction)", kept(d).get("Heparin") == I.SUPPORTED, str(kept(d)))
check("aspirin passes for ACS", "Aspirin" in kept(d), str(kept(d)))
d = gate("Pulmonary embolism", [HEPARIN])
check("heparin passes for a PE", kept(d).get("Heparin") == I.SUPPORTED)
d = gate("Rhabdomyolysis", [("Normal saline", "IV", "1 L")],
         actions=["IV fluid resuscitation with 0.9% saline"])
check("IV saline passes for rhabdomyolysis as fluid replacement",
      kept(d).get("Normal saline") == I.SYMPTOMATIC, str(kept(d)) + str(withheld(d)))
d = gate("Rhabdomyolysis", [("Paracetamol", "oral", "1 g")], pain=7)
check("paracetamol passes when the patient has pain", kept(d).get("Paracetamol") == I.SYMPTOMATIC, str(kept(d)))
d = gate("Acute severe asthma exacerbation", [("Salbutamol", "nebulised", "5 mg")])
check("salbutamol passes for asthma", kept(d).get("Salbutamol") == I.SUPPORTED, str(kept(d)))

# ===========================================================================
print("3. Words must not cross concepts.")
# ===========================================================================
d = gate("Acute decompensated heart failure", [HEPARIN])
check("heparin's 'heart surgery' does not license it for heart FAILURE",
      withheld(d) == ["Heparin"], str(kept(d)))
d = gate("Acute ischaemic stroke", [("Salbutamol", "nebulised", "5 mg")])
check("salbutamol is withheld for a stroke", withheld(d) == ["Salbutamol"], str(kept(d)))
c = I.concepts("Acute ischaemic stroke")
check("a stroke's concepts do not include the bare word 'infarction'", "infarction" not in c, str(c))

# ===========================================================================
print("4. A differential licenses a drug only conditionally.")
# ===========================================================================
d = gate("Cellulitis of the right leg", [HEPARIN], differentials=["Deep vein thrombosis"])
check("heparin is CONDITIONAL when DVT is only a differential",
      kept(d).get("Heparin") == I.CONDITIONAL, str(kept(d)))
check("and the warning says to give it only if confirmed", "only if" in d.drug_indication_warning)

# ===========================================================================
print("5. No working diagnosis -> nothing can be judged, so nothing is shown.")
# ===========================================================================
d = gate("Unspecified Diagnosis", [("Furosemide", "IV", "40 mg")])
check("a drug in a report with no diagnosis is withheld", withheld(d) == ["Furosemide"], str(kept(d)))

# ===========================================================================
print("6. Withheld drugs named in actions are flagged too.")
# ===========================================================================
d = gate("Rhabdomyolysis", [HEPARIN], actions=["Start heparin infusion"])
check("the action ordering heparin is removed with it, and named",
      "orders Heparin and was removed" in d.drug_indication_warning
      and not any("heparin" in a.action.lower() for a in d.immediate_actions),
      d.drug_indication_warning[-200:])

# ===========================================================================
print("7. Avoid-statements: quoted from the source, only for this diagnosis.")
# ===========================================================================
have_table = bool(N.statements())
check("the mined table exists (python -m app.negatives)", have_table)
if have_table:
    def avoid(dx, drugs, **kw):
        d = gate(dx, drugs, **kw)
        R._check_avoid_statements(d, TriageRequest(age=kw.get("age", 30), complaint="unwell"))
        return d.avoid_warning
    w = avoid("Exertional heat stroke", [("Paracetamol", "oral", "1 g")], temp=41.0, age=21)
    check("paracetamol in heat stroke quotes the MOH heat-illness guideline",
          "Heat Related Illness" in w and "Paracetamol" in w, w[:200])
    w = avoid("Dengue fever with warning signs", [("Diclofenac", "oral", "50 mg")], pain=5)
    check("an NSAID in dengue quotes a dengue CPG", "Dengue" in w, w[:200])
    w = avoid("ST-elevation myocardial infarction", [("Aspirin", "oral", "300 mg")])
    check("aspirin in STEMI draws NO avoid-warning (it is the first drug given)", w == "", w[:200])
    w = avoid("Acute severe asthma exacerbation", [("Salbutamol", "nebulised", "5 mg")])
    check("salbutamol in asthma draws no avoid-warning", w == "", w[:200])
    w = avoid("Dengue fever with warning signs", [("Paracetamol", "oral", "1 g")], pain=5, temp=38.5)
    check("paracetamol in dengue draws no avoid-warning", w == "", w[:200])
    hits = N.check("Aspirin", I.concepts("Exertional heat stroke"))
    check("stroke-CPG aspirin statements do not attach to HEAT stroke",
          all("Ischaemic Stroke" not in h["source"] for h in hits), str([h["source"] for h in hits]))

# ===========================================================================
print("8. A truncated, looping answer is flagged, with what is missing.")
# ===========================================================================
loop = "the patient's SpO2 is 89%, which is less than 92%, which would trigger Level 2, but "
raw = '{"mts_triage_level": 3, "triage_rationale": "' + loop * 40
cleaned = R._extract_json(raw)
d = DiagnosticSchema.model_validate_json(cleaned)
R._flag_truncation(d, raw, cleaned)
check("a cut-off answer is detected", R._was_truncated(raw))
check("the report says INCOMPLETE", d.parse_warning.startswith("INCOMPLETE"), d.parse_warning[:120])
check("it names the loop", "repeated itself" in d.parse_warning)
check("it lists the sections that never arrived",
      "primary diagnosis" in d.parse_warning and "drug recommendations" in d.parse_warning)
ok_raw = '{"mts_triage_level": 3, "triage_rationale": "fine"}'
d2 = DiagnosticSchema.model_validate_json(R._extract_json(ok_raw))
R._flag_truncation(d2, ok_raw, R._extract_json(ok_raw))
check("a complete answer is not flagged", d2.parse_warning == "")

# ===========================================================================
print("9. Run-4 regressions (2026-09-30 11:36).")
# ===========================================================================
aki = TriageRequest(age=55, complaint="Dark tea-coloured urine and leg pain",
                    history="acute kidney injury suspected", vitals={"pain_score": 7})
d = report("Rhabdomyolysis with acute kidney injury", [("Acetaminophen", "oral", "1 g")])
d.drug_recommendations[0].cautions = "Avoid NSAIDs due to risk of kidney injury."
R._check_contraindications(d, aki)
check("paracetamol whose caution says 'avoid NSAIDs' is NOT flagged as an NSAID",
      not d.contraindications, str([(c.rule, c.severity) for c in d.contraindications]))
d = report("Rhabdomyolysis with acute kidney injury", [("Ibuprofen", "oral", "400 mg")])
R._check_contraindications(d, aki)
check("ibuprofen in kidney injury still is", any(c.severity == "ABSOLUTE" for c in d.contraindications))
d = gate("Rhabdomyolysis", [("Intravenous calcium", "IV", ""), ("Paracetamol", "oral", "1 g")], pain=7)
R._check_drug_indications(d, [], None)
check("a later check does not overwrite the WITHHELD note",
      "WITHHELD" in d.drug_indication_warning and "Intravenous calcium" in d.drug_indication_warning,
      d.drug_indication_warning[:160])
check("a drug the gate grounded is not re-flagged 'NOT VERIFIED'",
      "Paracetamol" not in d.drug_indication_warning.split("WITHHELD")[0]
      and "NOT FOUND IN THE GUIDELINE FOR THIS PRESENTATION: Paracetamol" not in d.drug_indication_warning)

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
