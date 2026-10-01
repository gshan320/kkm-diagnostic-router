"""Regression tests for the 2026-09-30 corpus expansion and fixes R1-R3, F1-F7:
routing rules for the new MOH obstetric / psychiatric / dental / andrology
documents, their checklists and complication profiles, the quoted admission
floor, source cautions, dose precision, unmeasured exclusions, code-quoted gap
items and the confidence cap. Sections 1-6 need neither the model nor the
index; section 7 checks quotes against the PDFs and the live index.

    ../.venv/bin/python -m tests.test_expansion
"""
import re
import sys

from app import completeness as C
from app import complications as K
from app import disposition_rules as D
from app import focus, formulary as F
from app import rag_engine as R
from app import red_flags as RF
from app.schemas import (DiagnosticSchema, DifferentialDiagnosis, DrugRecommendation, ImmediateAction,
                         PrimaryDiagnosis, TriageRequest)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def fired(complaint, **kw):
    req = TriageRequest(age=kw.pop("age", 30), complaint=complaint, **kw)
    return {f.name for f, _ in RF.match(focus.strip_negated(R._intake_text(req)))}


# ===========================================================================
print("1. R1 routing rules and synonyms.")
# ===========================================================================
check("bleeding after delivery is PPH, not menorrhagia",
      fired("Heavy vaginal bleeding 2 hours after delivering her baby, soaking pads", gender="female")
      == {"postpartum_haemorrhage"})
check("heavy menstrual bleeding is still menorrhagia",
      "heavy_vaginal_bleeding" in fired("Heavy menstrual bleeding for 10 days, soaking pads", gender="female"))
check("a fit at 35 weeks is a hypertensive disorder of pregnancy",
      "hypertensive_disorder_of_pregnancy" in fired("Fitted once at home, 35 weeks pregnant", gender="female"))
check("...and so is one with pregnancy recorded in the field",
      "hypertensive_disorder_of_pregnancy" in fired("Fitted once with severe headache", gender="female",
                                                    pregnancy="pregnant", gestation_weeks=35))
check("a fit in a man is not", "hypertensive_disorder_of_pregnancy" not in fired("Had a fit at work", gender="male"))
check("'pregnancy excluded' does not make a fit obstetric",
      "hypertensive_disorder_of_pregnancy" not in fired("Had a fit at work", gender="female",
                                                        pregnancy="not_pregnant"))
check("agitation with voices routes to the psychiatric CPGs",
      "acute_behavioural_disturbance" in fired("Shouting, threatening staff, hearing voices"))
check("a knocked-out tooth is a dental avulsion", "dental_avulsion" in fired("Front tooth knocked out playing football"))
check("'knocked out' for two minutes is NOT a dental avulsion",
      "dental_avulsion" not in fired("Knocked out for 2 minutes after a fall, vomiting"))
check("priapism routes to the ED CPG", "priapism" in fired("Painful erection lasting 6 hours"))
check("an adult fit routes to the MSN epilepsy guideline",
      "seizure_or_status_epilepticus" in fired("Had a fit at work lasting 10 minutes", gender="male"))
check("...but a fit in pregnancy goes to the eclampsia rule instead",
      fired("Fitted once, 35 weeks pregnant", gender="female") & {"seizure_or_status_epilepticus"} == set())
check("an overdose routes to the MOH antidote guide", "poisoning_or_overdose" in fired("Took 30 paracetamol tablets"))
check("eclampsia meets the HDP manual's title",
      bool(R.indications.mentions(R._strong(["Eclampsia"]), "MOH Training Manual Hypertensive Disorders in Pregnancy")))
check("psychosis meets the Schizophrenia CPG title",
      bool(R.indications.mentions(R._strong(["Acute psychosis"]), "Management of Schizophrenia")))

# ===========================================================================
print("2. R2 checklists, and F6 tightening.")
# ===========================================================================
lab, gaps = C.check("Exertional rhabdomyolysis dark urine",
                    "Urine analysis for myoglobin and creatinine - to assess renal function; potassium; CK")
names = [g.element for g in gaps]
check("urine creatinine / 'assess renal function' is not a renal profile",
      "Serum urea and creatinine (renal profile)" in names, str(names))
check("a named serum test is", "Serum urea and creatinine (renal profile)" not in
      [g.element for g in C.check("rhabdomyolysis", "BUSE and serum creatinine")[1]])
check("haemolysis / G6PD is its own element", "Haemolysis / G6PD deficiency considered" in names)
check("serial CK and bloods is required", "Serial CK and bloods" in names)
check("'laboratory results every 6-24 hours' satisfies it",
      "Serial CK and bloods" not in
      [g.element for g in C.check("rhabdomyolysis", "monitor laboratory results every 6-24 hours")[1]])
for ctx, ans, want_label, must_gap in [
    ("Primary postpartum haemorrhage", "oxytocin, two large-bore IV cannula, crossmatch", "Postpartum haemorrhage",
     "Tranexamic acid"),
    ("Eclampsia 35 weeks pregnant BP 172/112", "MgSO4 4 g IV", "Eclampsia / severe pre-eclampsia",
     "Antihypertensive for severe blood pressure"),
    ("Acute agitation, schizophrenia relapse", "IM haloperidol", "Acute agitation / psychosis",
     "Organic cause and intoxication excluded"),
    ("Acute gout flare", "colchicine", "Acute gout flare", "Septic arthritis excluded"),
    ("Avulsed permanent upper incisor", "replant", "Avulsed permanent tooth", "Dental referral"),
]:
    lab, gaps = C.check(ctx, ans)
    check(f"{want_label}: checklist matches and reports '{must_gap}'",
          lab == want_label and must_gap in [g.element for g in gaps], f"{lab} {[g.element for g in gaps]}")
check("ergometrine caution only matters with hypertension",
      "Ergometrine avoided in hypertension or cardiac disease" not in
      [g.element for g in C.check("postpartum haemorrhage", "oxytocin")[1]])

# ===========================================================================
print("3. R3 profiles and F2 cautions (data).")
# ===========================================================================
for dx, name in [("Primary postpartum haemorrhage", "postpartum haemorrhage"),
                 ("Eclampsia", "eclampsia / severe pre-eclampsia"),
                 ("Acute agitation with psychosis", "acute agitation / psychosis"),
                 ("Lithium toxicity", "lithium toxicity"), ("Acute gout flare", "gout flare")]:
    p = K.profile_for(dx)
    check(f"'{dx}' has the {name} profile", p is not None and p.name == name, p.name if p else "none")
hdp = K.profile_for("Eclampsia")
po = hdp.complications[1]  # pulmonary oedema
check("'PE' counts as pre-eclampsia only inside the HDP manual",
      K.qualifies("severe PE is complicated by acute pulmonary oedema in some women", hdp, po, "",
                  "MOH Training Manual Hypertensive Disorders in Pregnancy")
      and not K.qualifies("abnormal features caused by pe include effusion and pulmonary oedema", hdp, po, "",
                          "Prevention and Treatment of Venous Thromboembolism (VTE)"))
check("an on-topic document outranks another KKM document",
      K.preference("x shock is common", "PPH guide", False, True) < K.preference("x shock", "UGIB", False, False))
check("KKM still outranks non-KKM",
      K.preference("a", "KKM", False, False) < K.preference("a", "CHAMP", True, True))
m = re.search("contraindicated in patients with cardiac disease or hypertension",
              "within 60 minutes SYNTOMETRINE (Oxytocin 5IU + Ergometrine 0.5mg) IM 1 ampoule (1ml) "
              "Contraindicated in patients with cardiac disease or hypertension Reduces blood loss below 500mls", re.I)
q = K.quote_around("within 60 minutes SYNTOMETRINE (Oxytocin 5IU + Ergometrine 0.5mg) IM 1 ampoule (1ml) "
                   "Contraindicated in patients with cardiac disease or hypertension Reduces blood loss below 500mls", m)
check("a table cell is quoted without the rest of the row",
      q.endswith("hypertension.") and "Reduces blood loss" not in q, q)
check("'e.g.' does not end a quote",
      K.quote_around("Caution use during periods of dehydration e.g. acute gastroenteritis, fasting. Next.",
                     re.search("caution use", "Caution use during periods of dehydration e.g. acute gastroenteritis, "
                               "fasting. Next.", re.I)).startswith("Caution use during periods of dehydration e.g. acute"))
check("lithium's document is the Bipolar CPG", K.on_topic(K.profile_for("Lithium toxicity"), "Management of Bipolar Disorder"))

# ===========================================================================
print("4. F1 quoted admission floor.")
# ===========================================================================
check("no hand-written rhabdomyolysis rule: CHAMP (not KKM) was its only source",
      D.applicable("Exertional rhabdomyolysis", "dark urine") is None)
check("eclampsia -> ICU/HDU", D.applicable("Eclampsia", "fitted once")[0].target == "ADMIT_ICU_HDU")
check("pre-eclampsia (either spelling) is not eclampsia",
      D.applicable("Pre-eclampsia", "fit") is None and D.applicable("pre eclampsia", "fit") is None)
check("severe pre-eclampsia / HELLP -> HDU", D.applicable("HELLP syndrome", "")[0].target == "ADMIT_ICU_HDU")
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Eclampsia"), disposition="RESUSCITATION_BAY")
R._quoted_disposition_floor(d, TriageRequest(age=24, complaint="fitted", gender="female"))
check("the floor never lowers", d.disposition.value == "RESUSCITATION_BAY")

# ===========================================================================
print("5. F3 dose precision, F4 unmeasured exclusions, F7 confidence.")
# ===========================================================================
adult = TriageRequest(age=55, weight_kg=80, complaint="dark urine")
d = DiagnosticSchema(immediate_actions=[ImmediateAction(sequence=1, action="IV isotonic fluids 2-6 L bolus then "
                                                                            "250-300 ml/hr")])
R._check_dose_completeness(d, adult)
check("an adult fluid volume / rate is a dose", d.dose_completeness_warning == "", d.dose_completeness_warning)
child = TriageRequest(age=6, weight_kg=20, complaint="dehydrated")
d = DiagnosticSchema(immediate_actions=[ImmediateAction(sequence=1, action="IV fluid bolus 500 ml")])
R._check_dose_completeness(d, child)
check("...but a child still needs ml/kg", "per-kg" in d.dose_completeness_warning)
check("IM in the guideline sentence picks the injection",
      R._fitting_formulations(F.entries_for("haloperidol"), False, "im")[0].drug_name.endswith("Injection"))
check("no route stated keeps the old order",
      "Tablet" in R._fitting_formulations(F.entries_for("haloperidol"), False, "")[0].drug_name)
check("route cues", R._route_cue("IM haloperidol plus lorazepam") == "im" and R._route_cue("give orally") == "oral")
d = DiagnosticSchema(differential_diagnoses=[DifferentialDiagnosis(
    condition="Heat-related illness", discriminating_feature="more consistent with muscle damage than heat illness")])
R._check_unmeasured_exclusions(d, TriageRequest(age=55, vitals={"pain_score": 7}, complaint="dark urine"))
check("heat illness cannot be excluded without a temperature", "NOT EXCLUDED" in d.differential_warning)
d = DiagnosticSchema(differential_diagnoses=[DifferentialDiagnosis(
    condition="Heat stroke", discriminating_feature="temperature normal")])
R._check_unmeasured_exclusions(d, TriageRequest(age=55, vitals={"temperature": 37.0}, complaint="dark urine"))
check("...but can once it is measured", d.differential_warning == "")
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Exertional rhabdomyolysis", confidence="HIGH"))
R._cap_confidence(d, TriageRequest(age=55, complaint="dark urine"))
check("HIGH is capped to MODERATE before a CK result", d.primary_diagnosis.confidence.value == "MODERATE"
      and "10x of uln" in d.primary_diagnosis.reasoning and "Dyslipidaemia" in d.primary_diagnosis.reasoning)
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Exertional rhabdomyolysis", confidence="HIGH"))
R._cap_confidence(d, TriageRequest(age=55, complaint="dark urine", history="CK 45,000 U/L at the clinic"))
check("...and not once the CK is reported", d.primary_diagnosis.confidence.value == "HIGH")
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Migraine", confidence="HIGH"))
R._cap_confidence(d, TriageRequest(age=35, complaint="headache"))
check("a diagnosis without a defining test is untouched", d.primary_diagnosis.confidence.value == "HIGH")

# ===========================================================================
print("6. F2 analgesia search, F5 quoted gap items (config).")
# ===========================================================================
qs = R._checklist_queries(TriageRequest(age=55, vitals={"pain_score": 7}, complaint="leg pain, dark urine"),
                          R.Hypotheses(["Rhabdomyolysis"], [], "t"))
check("pain 7 gets a 'severe pain' analgesia search, after the condition's own", qs and qs[-1].startswith("severe pain analgesia")
      and not qs[0].startswith("severe pain"), str(qs))
check("no pain, no analgesia search", not any("analgesia" in q for q in R._checklist_queries(
    TriageRequest(age=30, vitals={"pain_score": 0}, complaint="wheeze"), R.Hypotheses(["asthma"], [], "t"))))
check("the second pass is code-quoted by default", R.config.STAGE2_MODE == "quote")

# ===========================================================================
print("7. Quotes against the PDFs and the live index.")
# ===========================================================================
sys.path.insert(0, "tests/eval")
import harness as H  # noqa: E402
for rule in D.RULES:
    fn = next((p.name for p in H.RAW_PDFS.glob("*.pdf") if R.ingest_title(p.name) == rule.title), None) \
        if hasattr(R, "ingest_title") else None
from app import ingest  # noqa: E402
titles = {ingest.parse_title(p.name): p.name for p in H.RAW_PDFS.glob("*.pdf")}
for rule in D.RULES:
    fn = titles.get(rule.title)
    parts = [x.strip() for x in re.split(r"[;:]", rule.quote.replace(">=", "≥")) if len(x.strip()) > 12]
    ok = fn is not None and all(H.quote_status(fn, rule.page, x)[0] in ("ok", "ok-spacing") for x in parts)
    check(f"admission rule '{rule.name}': every quoted criterion is on {rule.source}", ok, fn or "no file")
try:
    engine = R.get_engine()
    engine.count()
    have_index = True
except Exception as exc:  # noqa: BLE001
    print(f"  SKIP  index unavailable ({exc})")
    have_index = False
if have_index:
    missing = RF.validate_titles(engine.indexed_titles)
    check("every routing rule's title is indexed", not missing, str(missing))
    want = {
        ("Primary postpartum haemorrhage", "female", 29, "bleeding after delivering her baby"):
            {"Ergometrine / Syntometrine contraindicated in hypertension or cardiac disease"},
        ("Eclampsia", "female", 24, "fitted, 35 weeks pregnant"):
            {"Give the MgSO4 loading dose slowly", "Magnesium toxicity - antidote is calcium gluconate"},
        ("Lithium toxicity", "male", 45, "tremor, on lithium"): {"Lithium level rises with dehydration"},
    }
    for (dx, sex, age, complaint), labels in want.items():
        d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition=dx))
        engine._source_cautions(TriageRequest(age=age, gender=sex, complaint=complaint), d, [], [])
        got = {c.label for c in d.source_cautions}
        check(f"{dx}: cautions quoted from source", labels <= got, str(got))
    d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Exertional rhabdomyolysis"))
    engine._source_cautions(TriageRequest(age=55, gender="male", complaint="dark urine after marathon"), d, [], [])
    check("rhabdomyolysis: no caution from a non-KKM source (CHAMP left the index)",
          not any(c.external for c in d.source_cautions), str([c.label for c in d.source_cautions]))
    p = K.profile_for("Eclampsia")
    hit = engine._complication_sentence(p, p.complications[2], [], TriageRequest(age=24, gender="female",
                                                                                 complaint="fitted, pregnant"))
    check("eclampsia's stroke risk is quoted from the HDP manual",
          hit is not None and "Hypertensive Disorders" in hit[2].metadata.get("cpg_title", ""), hit[1] if hit else "")
    p = K.profile_for("Primary postpartum haemorrhage")
    hit = engine._complication_sentence(p, p.complications[1], [], TriageRequest(age=29, gender="female",
                                                                                 complaint="bleeding after delivery"))
    check("PPH's DIC is quoted from the PPH guide",
          hit is not None and "Postpartum Haemorrhage" in hit[2].metadata.get("cpg_title", ""), hit[1] if hit else "")

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
