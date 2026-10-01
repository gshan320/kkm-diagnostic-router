"""Regression tests for attention layers A1-A3 (2026-09-30): key findings with
negation, the negation-aware red-flag floor, formulary by name, the topic gate,
and sentence distillation with bibliography removal. None need the model or
the index (section 8 uses a stub embedder).

    ../.venv/bin/python -m tests.test_attention
"""
import sys

from app import focus as Fo
from app import formulary as F
from app import red_flags
from app.schemas import TriageRequest

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


RHABDO = TriageRequest(age=55, vitals={"gcs": 15, "pain_score": 7},
                       complaint="Severe bilateral lower extremity pain, generalised muscle weakness, "
                                 "and dark tea-coloured urine for two days.",
                       history="High-intensity workouts and marathons.")

# ===========================================================================
print("1. Key findings: what the intake says, weighted, with red flags tagged.")
# ===========================================================================
f = Fo.extract(RHABDO)
texts = [x.text for x in f.findings]
check("each complaint clause becomes a finding",
      {"Severe bilateral lower extremity pain", "generalised muscle weakness",
       "dark tea-coloured urine for two days"} <= set(texts), str(texts))
tags = {x.text: x.red_flag for x in f.findings}
check("dark urine is tagged with the rhabdomyolysis rule", "rhabdomyolysis" in tags["dark tea-coloured urine for two days"])
check("weakness is tagged via the history (a rule combining clauses)",
      "exertional muscle injury" in tags["generalised muscle weakness"], str(tags))
check("red-flag findings rank first", f.ranked()[0].red_flag != "")
check("the abnormal vital (pain 7) is a finding", any(x.kind == "vital" for x in f.findings))
check("missing vitals, ECG and glucose are listed",
      f.missing == ["BP", "HR", "RR", "SpO2", "temperature", "ECG", "capillary glucose"], str(f.missing))
block = Fo.render(f, "MTS 3 URGENT")
check("the rendered block carries the code-decided triage line", "TRIAGE (decided by code" in block)
check("the reminder names the top findings", "dark tea-coloured urine" in Fo.reminder(f))

# ===========================================================================
print("2. Negation: stated-absent findings are never treated as present.")
# ===========================================================================
neg = TriageRequest(age=40, complaint="Palpitations since this morning, no chest pain",
                    history="Denies fever. No known illnesses.")
f = Fo.extract(neg)
check("'no chest pain' is a negative, not a finding",
      "no chest pain" in [n.lower() for n in f.negatives] and all("chest pain" not in x.text.lower() for x in f.findings),
      f"{f.negatives} | {[x.text for x in f.findings]}")
check("'Denies fever' is a negative", any("fever" in n.lower() for n in f.negatives))
check("a pseudo-negation is not a negation", not Fo.is_negated("no improvement with paracetamol"))
fired = {fl.name for fl, _ in red_flags.match(Fo.strip_negated("Palpitations. No chest pain."))}
check("'no chest pain' does not fire the ACS red flag", "acute_coronary_syndrome" not in fired, str(fired))
fired = {fl.name for fl, _ in red_flags.match(Fo.strip_negated("Chest pain radiating to the arm"))}
check("real chest pain still does", "acute_coronary_syndrome" in fired)
check("negation stops at the clause boundary",
      "palpitations" in Fo.strip_negated("No chest pain, but palpitations").lower())

# ===========================================================================
print("3. Formulary by name: drugs the text names, not words that look like drugs.")
# ===========================================================================
found = F.mentioned("Commence normal saline 0.9% 5-7 ml/kg; paracetamol 15 mg/kg for fever; "
                    "IV furosemide 40 mg; aspirin 300 mg chewed.")
check("drug names and synonyms are found", {"normal saline", "paracetamol", "furosemide", "aspirin"} <= set(found),
      str(found))
check("ordinary words are not drugs",
      F.mentioned("a major risk factor; continuous monitoring of essential protein intake") == [],
      str(F.mentioned("a major risk factor; continuous monitoring of essential protein intake")))
check("lab analytes are not drugs", F.mentioned("check serum potassium, sodium and blood glucose") == [])
check("aspirin resolves to FUKKM's acetylsalicylic acid",
      any("acetylsalicylic" in e.drug_name.lower() for e in F.entries_for("aspirin")))

# ===========================================================================
print("4. A2 topic gate: a cause-specific document needs a reason to be read.")
# ===========================================================================
from app import rag_engine as R, population as P, formulary as F2
from app.rag_engine import Retrieved
from app.schemas import (DiagnosticSchema, DrugRecommendation, ImmediateAction,
                         PrimaryDiagnosis)


def doc(title, dt="CPG_FULL", i=0):
    return Retrieved(chunk_id=f"{title}-{i}", text="x", metadata={"cpg_title": title, "doc_type": dt})


HEAT = "MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency and Trauma Department"
pool = [doc("MOH Guideline Management of Snakebite"), doc("Management of Haemophilia"),
        doc("Management of Obesity"), doc(HEAT), doc("Management of Dyslipidaemia"),
        doc("MOH Pain Management in Emergency and Trauma Department"),
        doc("FUKKM", "DRUG_FORMULARY")]
hyp = R.Hypotheses(["Exertional rhabdomyolysis", "Heat-related illness"], ["Haemolysis"], "t")
kept, notes = R._topic_gate(pool, RHABDO, hyp, forced={"Management of Dyslipidaemia"})
titles = [c.metadata["cpg_title"] for c in kept]
check("snakebite, haemophilia and obesity are left out for exertional rhabdomyolysis",
      not {"MOH Guideline Management of Snakebite", "Management of Haemophilia", "Management of Obesity"} & set(titles),
      str(titles))
check("the hypothesis-matched heat guideline stays", HEAT in titles)
check("a red-flag-forced document stays", "Management of Dyslipidaemia" in titles)
check("a general-purpose document stays", "MOH Pain Management in Emergency and Trauma Department" in titles)
check("the formulary passes through untouched", "FUKKM" in titles)
check("what was left out is named", notes and "Snakebite" in notes[0], str(notes))
bite = TriageRequest(age=40, complaint="Bitten on the foot by a snake 2 hours ago")
kept, _ = R._topic_gate([doc("MOH Guideline Management of Snakebite")], bite,
                        R.Hypotheses(["Pit viper envenoming"], [], "t"), set())
check("...but the snakebite guideline stays for a snake bite", len(kept) == 1)
htn = TriageRequest(age=58, complaint="Chest pain", history="Hypertension on amlodipine")
kept, _ = R._topic_gate([doc("Management of Hypertension", i=1), doc("Management of Hypertension", i=2)], htn,
                        R.Hypotheses(["Acute coronary syndrome"], [], "t"), set())
check("a comorbidity-only document keeps ONE passage, as background", len(kept) == 1)
nag = doc("National Antimicrobial Guideline 2024 (Adults)")
check("the NAG stays when there are signs of infection",
      len(R._topic_gate([nag], TriageRequest(age=45, complaint="Red swollen leg with fever"),
                        R.Hypotheses(["Cellulitis"], [], "t"), set())[0]) == 1)
check("...and is left out when there are none",
      R._topic_gate([nag], RHABDO, hyp, set())[0] == [])

# ===========================================================================
print("5. Hypotheses: parsed defensively, merged with the rules, never blocking.")
# ===========================================================================
likely, exclude = R._parse_hypotheses('{"hypotheses": ["Exertional rhabdomyolysis", "<most likely>", "Heat stroke"], '
                                      '"must_exclude": ["Haemolysis"]}')
check("slot markers are discarded", likely == ["Exertional rhabdomyolysis", "Heat stroke"], str(likely))
check("must_exclude is read", exclude == ["Haemolysis"])
check("unparseable output yields nothing (the rules take over)", R._parse_hypotheses("not json") == ([], []))
check("the red-flag rules name hypotheses on their own", "rhabdomyolysis" in R._rule_hypotheses(RHABDO))

# ===========================================================================
print("6. Formulations fit the patient; paediatric titles are recognised.")
# ===========================================================================
saline = F2.entries_for("sodium chloride")
adult = R._fitting_formulations(saline, child=False)
check("no eye drops for systemic use", all("eye" not in e.drug_name.lower() for e in adult))
check("0.9% saline comes first", adult and "0.9%" in adult[0].drug_name, adult[0].drug_name if adult else "")
para = F2.entries_for("paracetamol")
check("no syrup for an adult", all("syrup" not in e.drug_name.lower() for e in R._fitting_formulations(para, False)))
kid = R._fitting_formulations(para, child=True)
check("a child gets a liquid form first", kid and any(w in kid[0].drug_name.lower() for w in ("syrup", "suspension", "drops")),
      kid[0].drug_name if kid else "")
check("'(Paediatrics)' is paediatric (the plural was missed)",
      P.of({"cpg_title": "National Antimicrobial Guideline 2024 (Paediatrics)"}) == P.PAEDIATRIC)
check("ipratropium is found by its plain name", "ipratropium" in F2.mentioned("nebulised ipratropium 0.5 mg"))

# ===========================================================================
print("7. Run-5 fixes.")
# ===========================================================================
from app import completeness as C
corpus = {"docs": ["DO NOT administer Paracetamol or Aspirin or other NSAIDS."],
          "metas": [{"cpg_title": HEAT, "doc_type": "CPG_FULL", "page_number": 14}]}
gap = C.Gap("Rhabdomyolysis", "Analgesia, with NSAIDs avoided", "why", "MOH Heat Related Illness 2016")
check("'DO NOT administer paracetamol' is never quoted as support for analgesia",
      R._gap_quote(gap.presentation, gap, corpus, 55) == ("", ""))
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                     immediate_actions=[ImmediateAction(sequence=3, action="Administer intravenous calcium to prevent hyperkalemia.")],
                     drug_recommendations=[DrugRecommendation(drug_name="Calcium gluconate", route="IV")])
R._gate_drug_indications(d, TriageRequest(age=55, complaint="dark urine and leg pain"), [])
check("an action ordering a withheld drug by its generic word is removed with it",
      "orders Calcium gluconate and was removed" in d.drug_indication_warning
      and not any("calcium" in a.action.lower() for a in d.immediate_actions), d.drug_indication_warning[-160:])
block = Fo.render(Fo.extract(RHABDO), "MTS 3")
check("the rule name follows the finding, not precedes it",
      "dark tea-coloured urine for two days  (red-flag rule: rhabdomyolysis)" in block, block[:300])

# ===========================================================================
print("8. A3 distillation: the right sentences, never a reference list.")
# ===========================================================================
from app import distill as D
import numpy as np

h, ss = D.sentences("[Management of Dyslipidaemia — 2023 | page 59]\nstatin - associated muscle symptoms. "
                    "rhabdomyolysis is defined as ck > 10x uln. check renal function.")
check("the header line is kept apart", h.startswith("[Management of Dyslipidaemia"))
check("lower-case text is still split into sentences", len(ss) == 3, str(ss))
check("a decimal is not a sentence boundary", len(D.sentences("Give 0.9% saline 1 L. Then reassess.")[1]) == 2)
check("a flattened table is cut into scoreable pieces",
      all(len(x) <= D.MAX_SENTENCE for x in D.sentences("dose " * 200)[1]))
REF = ("1. davidson mh, clark ja, glass lm, kanumalla a. statin safety. am j cardiol 2006 ; 97 ( 8a ) : 32c - 43c. "
       "2. parker ba, capizzi ja, grimaldi as, et al. effect of statins on skeletal muscle. circulation 2013 ; 127 : 96 - 103. "
       "3. moriarty pm, jacobson ta, bruckert e, et al. alirocumab in statin intolerance. j clin lipidol 2014 ; 8 : 554 - 61. "
       "4. doi : 10. 1161 / circulationaha. 121. 057736. accessed 1 april 2017.")
check("a reference-list page is recognised", D.is_bibliography(REF), str(D.citation_marks(REF)))
CLIN = ("Commence 0.9% sodium chloride 1-2 L/hour, aiming for urine output 200-300 ml/hour. Check CK, "
        "potassium and creatinine every 6-12 hours. A meta-analysis (Fedele et al, 1991; level 5) found benefit.")
check("guidance with an inline citation is not", not D.is_bibliography(CLIN), str(D.citation_marks(CLIN)))
check("a paper title is a citation sentence",
      D.is_citation("parker ba, capizzi ja, grimaldi as. effect of statins. circulation 2013 ; 127 : 96 - 103."))
check("a dose sentence is not", not D.is_citation("Give aspirin 300 mg stat, then 75-150 mg daily."))


class C:
    def __init__(self, text, title="T", page=1, dt="CPG_FULL"):
        self.text = f"[{title} — 2020 | page {page}]\n{text}"
        self.metadata = {"cpg_title": title, "doc_type": dt}


def stub_embed(texts):
    # Bag-of-keywords embedder: enough to rank "fluid" sentences above others.
    keys = ["fluid", "saline", "ck", "potassium", "statin", "cooling"]
    v = np.array([[t.lower().count(k) + 0.01 for k in keys] for t in texts], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


chunk = C("Heat stroke needs cooling. Give saline fluid 1 L. Statin history matters. "
          "The weather was hot. Check CK and potassium. " + REF)
sc = D.distill([chunk], ["saline fluid", "CK potassium"], stub_embed, per_chunk=2)[0]
kept = [sc.sentences[i] for i in sc.keep]
check("the best sentences for the queries are kept", any("saline" in k for k in kept) and any("CK" in k for k in kept),
      str(kept))
check("no reference entry is ever kept", not any(D.is_citation(k) for k in kept), str(kept))
check("kept sentences stay under their header, gaps marked",
      sc.text().startswith("[T — 2020 | page 1]") and "..." in sc.text(), sc.text())
check("dose extras are capped (a dosing table is all dose sentences)",
      len(D.distill([C(" ".join(f"Drug{i} {i} mg daily." for i in range(12)))], ["x"], stub_embed,
                    per_chunk=2, always_keep=lambda t: "mg" in t, max_extra=2)[0].keep) == 4)
items = [D.Scored(C(str(i)), "", ["s"], [r], [0]) for i, r in enumerate([0.9, 0.8, 0.5, 0.3])]
order = [it.relevance for it in D.edge_order(items)]
check("edge order: best first, second-best last", order[0] == 0.9 and order[-1] == 0.8, str(order))
check("formulary rows lose their codes",
      "MDC" not in D.trim_formulary("Generic name: X\nMDC Code: 1\nWHO ATC Code: A\nDosage: 1 g"))
check("a year-bearing reference is stripped from a reason",
      R._no_refs("IV fluids for rhabdomyolysis (CHAMP 2020)") == "IV fluids for rhabdomyolysis")

# ===========================================================================
print("9. A3 source discipline: secondary documents are read, not prescribed from.")
# ===========================================================================
ASTHMA = TriageRequest(age=32, vitals={"respiratory_rate": 30, "spo2": 91},
                       complaint="Worsening shortness of breath and wheeze", history="Known asthmatic.")
COPD = "Management of Chronic Obstructive Pulmonary Disease (COPD)"
check("the rule's own guideline is primary, its secondary one is not",
      R._primary_forced(ASTHMA) == {"Management of Asthma in Adults"}, str(R._primary_forced(ASTHMA)))
ahyp = R.Hypotheses(["Acute severe asthma exacerbation", "obstructive airway disease"], [], "model+rules")
kept, _ = R._topic_gate([doc(COPD, i=i) for i in range(4)], ASTHMA, ahyp,
                        {COPD, "Management of Asthma in Adults"})
check("a secondary document keeps at most FORCED_SECONDARY_PASSAGES passages",
      len(kept) == R.config.FORCED_SECONDARY_PASSAGES, str(len(kept)))
tt = R._treatment_titles([doc(COPD), doc("Management of Asthma in Adults"),
                          doc("MOH Pain Management in Emergency and Trauma Department")], ASTHMA, ahyp)
check("formulary-by-name reads the asthma CPG and general documents, not COPD",
      tt == {"Management of Asthma in Adults", "MOH Pain Management in Emergency and Trauma Department"}, str(tt))
tt = R._treatment_titles([doc("Management of Dyslipidaemia"), doc(HEAT)], RHABDO,
                         R.Hypotheses(["Exertional rhabdomyolysis"], [], "t"))
check("Dyslipidaemia (statin myopathy, a cause) supplies no drugs for rhabdomyolysis",
      "Management of Dyslipidaemia" not in tt and HEAT in tt, str(tt))
bib_chunk = Retrieved(chunk_id="b", text=REF, metadata={"cpg_title": "Management of Dyslipidaemia",
                                                        "doc_type": "CPG_FULL", "page_number": 108})
kept, notes = R._drop_bibliography([bib_chunk, doc(HEAT)])
check("a reference-list page is dropped and named", len(kept) == 1 and "p108" in notes[0], str(notes))
check("'SABA' finds salbutamol in the formulary", "saba" in F.mentioned("initial treatment with SABA")
      and "salbutamol" in F.entries_for("saba")[0].drug_name.lower())

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
