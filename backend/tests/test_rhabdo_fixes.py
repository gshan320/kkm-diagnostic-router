"""Regression tests for the 2026-10-03 rhabdo fixes (harness run 20261002-013457,
82%): actions for a differential the report set aside, a checklist test
completed in its existing row, and the quote chosen for it. No model, no index.

    ../.venv/bin/python -m tests.test_rhabdo_fixes
"""
import sys

from app import rag_engine as R
from app.rag_engine import Retrieved
from app.schemas import (DiagnosticSchema, DifferentialDiagnosis, ImmediateAction, Investigation,
                         PrimaryDiagnosis, RetrievedSource, TriageRequest)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def rhabdo_req(**vitals):
    return TriageRequest(age=55, vitals={"gcs": 15, "pain_score": 7, **vitals},
                         complaint="Severe bilateral lower extremity pain, generalised muscle weakness, "
                                   "and dark tea-coloured urine for two days.",
                         history="High-intensity workouts and marathons.")


def report(primary="Rhabdomyolysis"):
    return DiagnosticSchema(
        primary_diagnosis=PrimaryDiagnosis(condition=primary),
        differential_diagnoses=[DifferentialDiagnosis(
            condition="Heat Stroke", discriminating_feature="Absence of core temperature elevation.")],
        immediate_actions=[ImmediateAction(sequence=1, action="Administer 0.9% normal saline IV"),
                           ImmediateAction(sequence=2, action="Apply ice packs to axillary, neck, and groin regions"),
                           ImmediateAction(sequence=3, action="Initiate evaporative cooling with mist fan")])


print("1. actions for a set-aside condition")
d = report()
R._gate_set_aside_actions(d, rhabdo_req())
acts = [a.action for a in d.immediate_actions]
check("temperature not taken: cooling kept, made conditional",
      len(acts) == 3 and all(a.startswith("Only if heat stroke") for a in acts[1:]), acts)
check("...saline untouched", acts[0] == "Administer 0.9% normal saline IV")
check("...and named in the consistency warning", "SET-ASIDE CONDITION" in d.consistency_warning)
R._gate_set_aside_actions(d, rhabdo_req())
check("...not prefixed twice", not any(a.action.count("Only if") > 1 for a in d.immediate_actions))
d = report()
R._gate_set_aside_actions(d, rhabdo_req(temperature=37.2))
check("temperature measured: cooling removed",
      [a.action for a in d.immediate_actions] == ["Administer 0.9% normal saline IV"])
d = report("Exertional heat stroke")
R._gate_set_aside_actions(d, rhabdo_req())
check("heat stroke is the diagnosis: nothing changes",
      len(d.immediate_actions) == 3 and not d.consistency_warning)

print("2. the quote for a checklist test, and where it goes")
SNAKE = "MOH Guideline Management of Snakebite"
docs = [
    ("p99", 99, "Serial blood results every 4 - 6 hours for the first 24 to 48 hours."),
    ("p47", 47, "Early hyperkalemia may be seen following extensive rhabdomyolysis in sea snake bites. "
                "4.5.4 Creatine kinase: For early detection of rhabdomyolysis.\n"
                "Serial monitoring to monitor trend. 4.5.5 Urinalysis: To assess for myoglobinuria."),
    ("p53", 53, "4.6 Indication for observation and admission: i. All patients with a history of snakebite."),
]
corpus = {"ids": [i for i, _, _ in docs], "docs": [t for _, _, t in docs],
          "metas": [{"cpg_title": SNAKE, "doc_type": "CPG_FULL", "page_number": p} for _, p, _ in docs]}
corpus["pos"] = {c: i for i, c in enumerate(corpus["ids"])}
R.set_support_idf(corpus["docs"])
gap = R._Gap(presentation="Rhabdomyolysis", element="Serial CK and bloods", why="",
             guideline="MOH Snakebite 2017 s4.5 (serial CK)")
quote, where = R._gap_quote("Rhabdomyolysis", gap, corpus, 55)
check("the sentence naming CK wins over serial bloods in general",
      quote.startswith("Creatine kinase: For early detection") and "Serial monitoring" in quote, quote)
check("...cut at its heading, not swallowed by the snakebite line before it", "snake" not in quote.lower(), quote)
check("a list lead-in cut from its items is never a quote",
      R._gap_quote("Rhabdomyolysis", R._Gap("Rhabdomyolysis", "Admission for IV fluids and serial CK", "",
                                            "MOH Snakebite 2017"), corpus, 55)[0] != "Indication for observation and admission: i.")

eng = object.__new__(R.RagEngine)
eng.__dict__["_corpus"] = corpus
d = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                     investigations=[Investigation(test="Creatine kinase (CK)", source_id="")])
chunks = [Retrieved(chunk_id=i, text=t, metadata=m) for i, t, m in zip(corpus["ids"], corpus["docs"], corpus["metas"])]
sources = [RetrievedSource(source_id=f"S{n}", filename="f.pdf", cpg_title=SNAKE, edition_year="2017",
                           doc_type="CPG_FULL", page_number=0, excerpt="") for n in range(1, 4)]
eng._quoted_gap_items(rhabdo_req(), d, chunks, sources)
ck = [x for x in d.investigations if "creatine" in x.test.lower() or "CK" in x.test]
check("serial CK completes the model's CK row instead of a second row", len(ck) == 1, [x.test for x in ck])
check("...which now says serial and carries the CK quote",
      "Serial CK" in ck[0].test and ck[0].source_quote.startswith("Creatine kinase"), (ck[0].test, ck[0].source_quote))

print(f"\n{len(fails)} fail(s)")
sys.exit(1 if fails else 0)
