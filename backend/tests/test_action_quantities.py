"""Regression test for _ground_action_quantities (2026-10-02): a dose, volume
or rate in a model-written immediate action must be written in a retrieved
passage. Born of the MedGemma rhabdo run that ordered '30-50 ml/kg/hr' saline
for an 80 kg man with nothing in the corpus saying so. No model, no index.

    ../.venv/bin/python -m tests.test_action_quantities
"""
import sys

from app import rag_engine as R
from app.rag_engine import Retrieved
from app.schemas import DiagnosticSchema, ImmediateAction

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


HRI = Retrieved(chunk_id="hri-19", metadata={},
                text="Set IV lines and administer 0.9% normal saline, rate and volume depend on "
                     "patient's premorbid condition. Target urine output above 0.5 ml/kg/hr. Give 250-500 ml/hr, then 250 ml/hr.")


def run(*actions):
    d = DiagnosticSchema(immediate_actions=[ImmediateAction(sequence=i, action=a)
                                            for i, a in enumerate(actions, start=1)])
    R._ground_action_quantities(d, [HRI])
    return d


d = run("IV 0.9% NaCl at 30-50 ml/kg/hr (or 1 ml/kg/hr) [S1]")
a = d.immediate_actions[0].action
check("an invented range is removed", "30-50" not in a, a)
check("the bracket leftover is tidied", "(or" not in a, a)
check("the removal is named in the safety block", "UNSOURCED QUANTITY REMOVED" in d.dose_completeness_warning)

d = run("IV 0.9% saline at 250 ml/hr [S1]")
check("a rate written in a passage stays", "250 ml/hr" in d.immediate_actions[0].action)
check("...and raises no warning", not d.dose_completeness_warning)

d = run("Saline 250-500 ml/hr")
check("a range is kept only when both ends are in one passage", "250-500 ml/hr" in d.immediate_actions[0].action)

d = run("Give 25 ml/hr")
check("'25' is not matched inside '250'", "[dose/rate removed]" in d.immediate_actions[0].action)

d = DiagnosticSchema(immediate_actions=[ImmediateAction(action="Give 999 ml stat", origin="source")])
R._ground_action_quantities(d, [HRI])
check("server-quoted actions are left alone", "999 ml" in d.immediate_actions[0].action)

d = run("Start IV fluids, 0.9% saline")
check("a percentage concentration is not treated as a dose", d.immediate_actions[0].action.endswith("saline"))

d = run("Give 250 ml/kg")
check("the same number per kg is not the same dose", "[dose/rate removed]" in d.immediate_actions[0].action)

d = run("Give 2 ml/kg/hr")
check("a bare number elsewhere in the passage does not ground a dose", "[dose/rate removed]" in d.immediate_actions[0].action)

# The admission-process quotes must be in the indexed OCR text, word for word.
import json, pathlib, re as _re
ocr = pathlib.Path("data/raw_pdfs/MOH Patient Flow Management Guideline (Garis Panduan Pengurusan Aliran Pesakit) 2022.pdf.ocr.json")
if ocr.exists():
    pages = json.loads(ocr.read_text(encoding="utf-8"))["pages"]
    flat = lambda x: _re.sub(r"\s+", " ", x).strip()
    for where, q in R.ADMISSION_PROCESS_QUOTES:
        n = int(where.rsplit("p", 1)[1])
        check(f"quote verbatim on {where}", flat(q) in flat(pages[n - 1]))
from app.schemas import Disposition
d = DiagnosticSchema(disposition=Disposition.ED_OBSERVATION)
R._admission_process_note(d)
check("ED observation without a criterion names the process", "hospital's own admission criteria" in d.disposition_justification)
d = DiagnosticSchema(disposition=Disposition.ADMIT_WARD)
R._admission_process_note(d)
check("...and an admission is left alone", "admission criteria" not in d.disposition_justification)

print(f"\n{len(fails)} fail(s)")
sys.exit(1 if fails else 0)
