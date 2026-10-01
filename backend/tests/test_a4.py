"""Regression tests for attention layer A4 (2026-09-30): the attention audit
(A4.1), sentence-level citation support and alignment (A4.2), complications
anchored to verbatim source sentences (A4.3), and the output diet (A4.4).
Sections 1-5 need neither the model nor the index; section 6 checks the
curated complication profiles against the live index when it is present.

    ../.venv/bin/python -m tests.test_a4
"""
import sys

from app import attention_audit as AA
from app import complications as C
from app import config
from app import rag_engine as R
from app.rag_engine import Retrieved
from app.schemas import (Citation, DiagnosticSchema, DrugRecommendation, ImmediateAction,
                         PrimaryDiagnosis, RedFlag, RetrievedSource, TriageRequest)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def chunk(title, page, text, dt="CPG_FULL", i=0):
    return Retrieved(chunk_id=f"{title}-{page}-{i}", text=f"[{title} — 2020 | page {page}]\n{text}",
                     metadata={"cpg_title": title, "doc_type": dt, "page_number": page})


def src(i, title):
    return RetrievedSource(source_id=f"S{i}", filename="f.pdf", cpg_title=title, edition_year="2020",
                           doc_type="CPG_FULL", page_number=i, excerpt="")


RHABDO = TriageRequest(age=55, vitals={"gcs": 15, "pain_score": 7},
                       complaint="Severe bilateral lower extremity pain, generalised muscle weakness, "
                                 "and dark tea-coloured urine for two days.",
                       history="High-intensity workouts and marathons.")
HEAT = "MOH Clinical Guidelines on Management of Heat Related Illness"
PAIN = "MOH Pain Management in Emergency and Trauma Department"
HK = "Malaysian Consensus on the Management of Acute and Persistent Hyperkalaemia"

# ===========================================================================
print("1. A4.1 attention audit: stage times, positions, what was cited.")
# ===========================================================================
t = AA.Trace()
t.add("retrieval", 1200)
t.add("retrieval", 300)
with t.stage("checks"):
    pass
t.generation("triage").update(prompt_tokens=4000, reused_tokens=2900, output_tokens=900,
                              prefill_ms=8000, decode_ms=90000)
check("stage times accumulate", t.stages["retrieval"] == 1500)
sources = [src(i, title) for i, title in enumerate([HEAT, HEAT, HK, PAIN, HEAT, HK], start=1)]
d = DiagnosticSchema(citations=[Citation(source_id="[S1]"), Citation(source_id="[S6]")])
rep = AA.build(t, sources, 5, d)
check("reading and writing come from the generation stats",
      rep.stages_ms["reading"] == 8000 and rep.stages_ms["writing"] == 90000)
check("positions are thirds of the first prompt; later sources are 'added later'",
      rep.given_by_position == {"front": 1, "middle": 2, "back": 2, "added later": 1}, str(rep.given_by_position))
check("cited by position", rep.cited_by_position == {"front": 1, "added later": 1}, str(rep.cited_by_position))
check("a document cited through a later passage is not 'never cited'",
      rep.uncited_documents == [PAIN], str(rep.uncited_documents))
line = AA.summary(rep)
check("the summary line says what was cited and where the time went",
      "cites 2 of 6 passages" in line and "writing 90 s" in line and "2,900 cached" in line, line)
check("the aggregate reads the records", "citations moved" in AA.aggregate([rep, rep]))
check("an empty audit log is said plainly", "No report" in AA.aggregate([]))

# ===========================================================================
print("2. A4.2 support: sentence level, rare terms weigh more.")
# ===========================================================================
check("'compartment' and 'compared' are different terms",
      not (R._support_terms("compartment syndrome", item=True) & R._support_terms("compared with placebo")))
check("capitalised abbreviations count in an item", "ecg" in R._support_terms("Perform 12-lead ECG", item=True))
pain_p22 = ("For moderate pain use: Paracetamol 1g IV (if available) or oral 4 hourly prn. "
            "Severe pain: consider morphine titration. Reassess pain score after each dose.")
heat_p10 = ("Renal Function Test Acute kidney injury due to inadequacy of volume, dehydration and may also "
            "due to rhabdomyolysis. Urine analysis for protein, cast and myoglobin.")
check("IV fluids for AKI is NOT supported by a pain page",
      not R._supported("Administer intravenous fluids to prevent acute kidney injury", pain_p22))
check("paracetamol for moderate pain IS supported by it",
      R._supported("Administer intravenous paracetamol 1g for moderate pain", pain_p22))
check("urine output / AKI monitoring is supported by the renal sentence",
      R._supported("Monitor urine output and assess for signs of acute kidney injury", heat_p10))
check("terms scattered across a page do not add up",
      not R._supported("Assess kidney perfusion with pain relief",
                       "Kidney stones are rare. " + "Filler sentence here. " * 5 + "Pain relief follows."))

# ===========================================================================
print("3. A4.2 alignment: move to the supporting passage, never invent.")
# ===========================================================================
chunks = [chunk(PAIN, 22, pain_p22), chunk(HEAT, 10, heat_p10),
          chunk("Management of Dyslipidaemia", 81, "Metabolic syndrome in schizophrenia is common. Statins help.")]
a1 = ImmediateAction(sequence=1, action="Monitor urine output and assess for acute kidney injury", source_id="[S1]")
a2 = ImmediateAction(sequence=2, action="Administer intravenous paracetamol 1g for moderate pain", source_id="[S1]")
a3 = ImmediateAction(sequence=3, action="Assess for signs of compartment syndrome", source_id="[S3]")
a4 = ImmediateAction(sequence=4, action="Arrange physiotherapy review for gait", source_id="[S3]")
dx = DiagnosticSchema(immediate_actions=[a1, a2, a3, a4])
moves = R._align_citations(dx, chunks, model_saw=3)
check("an unsupported citation moves to the passage that supports it", a1.source_id == "[S2]", a1.source_id)
check("a supported citation stays", a2.source_id == "[S1]")
check("an item no passage supports is NEVER removed (run 8 lost IV fluids that way)",
      a3 in dx.immediate_actions and a4 in dx.immediate_actions and moves == 1)
notes, stripped = R._check_citation_support(dx, chunks)
check("...its unsupported citation is stripped and named instead",
      a3.source_id == "" and "action 3" in stripped and "does not discuss it" in notes[0], str(notes))
check("the moves are named", "action 1: [S1] -> [S2]" in dx.citation_alignment_note, dx.citation_alignment_note)
b = ImmediateAction(sequence=1, action="Monitor urine output and assess for acute kidney injury", source_id="[S1]")
R._align_citations(DiagnosticSchema(immediate_actions=[b]), chunks[:1] + [chunks[2], chunks[1]], model_saw=2)
check("a passage the model never saw is never used", b.source_id == "[S1]", b.source_id)
drug = DrugRecommendation(drug_name="Paracetamol", route="IV", source_id="[S2]")
R._align_citations(DiagnosticSchema(drug_recommendations=[drug]), chunks, model_saw=3)
check("a drug's citation moves to a passage that names it", drug.source_id == "[S1]", drug.source_id)

# ===========================================================================
print("4. A4.3 complications: the dangers, from verbatim sentences only.")
# ===========================================================================
prof = C.profile_for("Rhabdomyolysis")
check("rhabdomyolysis has a profile", prof is not None and prof.name == "rhabdomyolysis")
check("'Rhabdomyolysis' and 'Exertional muscle injury' only restate",
      C.restates(prof, "Rhabdomyolysis") and C.restates(prof, "Exertional muscle injury"))
check("'Hyperkalaemia' is not a restatement", not C.restates(prof, "Hyperkalaemia with arrhythmia"))
check("'Rhabdomyolysis with acute kidney injury' is not a mere restatement",
      not C.restates(prof, "Rhabdomyolysis with acute kidney injury"))
general = "In skeletal muscle breakdown (rhabdomyolysis), hyperkalemia can also lead to cardiac arrest."
snake = "Early hyperkalemia may be seen following extensive rhabdomyolysis in sea snake bites."
hk = prof.complications[0]
check("a general sentence may speak for a runner", C.qualifies(general, prof, hk, RHABDO.complaint))
check("a sea-snake sentence may not", not C.qualifies(snake, prof, hk, RHABDO.complaint + RHABDO.history))
check("...unless the intake names the snake", C.qualifies(snake, prof, hk, "bitten by a sea snake"))
check("prose is preferred to a table fragment",
      C.preference(general, "MOH Guideline Management of Snakebite")
      < C.preference("Rhabdomyolysis causing renal failure & hyperkalaemia Paralysis 11", "X"))
aki = prof.complications[1]
check("renal failure as a RISK FACTOR is not the complication",
      not C.qualifies("patients with chronic renal failure and rhabdomyolysis history are susceptible",
                      prof, aki, ""))


class Stub:
    """Just enough engine for the A4.3 and second-pass methods."""
    _context = R.RagEngine._context
    _complication_sentence = R.RagEngine._complication_sentence
    _anchor_complications = R.RagEngine._anchor_complications
    _card_complications = R.RagEngine._card_complications
    _second_stage = R.RagEngine._second_stage
    _bibliography_ids = frozenset()

    def __init__(self, docs):
        self._corpus = {"ids": [f"c{i}" for i in range(len(docs))], "docs": [d for d, _ in docs],
                        "metas": [m for _, m in docs]}


stub = Stub([(general, {"cpg_title": "MOH Guideline Management of Snakebite", "doc_type": "CPG_FULL",
                        "page_number": 63, "status": "active"}),
             (snake, {"cpg_title": "MOH Guideline Management of Snakebite", "doc_type": "CPG_FULL",
                      "page_number": 47, "status": "active"})])
given = [chunk(HEAT, 10, heat_p10)]
srcs = [src(1, HEAT)]
dx = DiagnosticSchema(
    primary_diagnosis=PrimaryDiagnosis(condition="Exertional rhabdomyolysis"),
    red_flags=[RedFlag(flag="Rhabdomyolysis", why_it_matters="Dark urine can lead to acute kidney injury",
                       source_id="[S1]"),
               RedFlag(flag="Exertional muscle injury", why_it_matters="marathons", source_id="[S1]")])
n = stub._anchor_complications(RHABDO, dx, given, srcs)
names = [f.flag for f in dx.red_flags]
check("hyperkalaemia and AKI are added (no compartment-syndrome sentence here: silence)",
      n == 2 and names == ["Hyperkalaemia - risk of cardiac arrhythmia or arrest", "Acute kidney injury"], str(names))
check("AKI is quoted from the passage the model was given", dx.red_flags[1].source_id == "[S1]")
check("hyperkalaemia's page is appended as a new source",
      dx.red_flags[0].source_id == "[S2]" and len(srcs) == 2 and srcs[1].page_number == 63)
check("the quote is the verbatim sentence", general.rstrip(".") in dx.red_flags[0].why_it_matters)
check("added flags are marked as from source", all(f.origin == "source" for f in dx.red_flags))
check("the note names what was added and what was removed",
      "Hyperkalaemia" in dx.complication_note and "Exertional muscle injury" in dx.complication_note)
from app.schemas import DifferentialDiagnosis  # noqa: E402
dx3 = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                       red_flags=[RedFlag(flag="Rhabdomyolysis", why_it_matters="x")],
                       differential_diagnoses=[DifferentialDiagnosis(condition="Heat stroke"),
                                               DifferentialDiagnosis(condition="Acute kidney injury")])
stub._anchor_complications(RHABDO, dx3, [chunk(HEAT, 10, heat_p10)], [src(1, HEAT)])
check("a complication listed as a differential moves out once the red flags carry it",
      [x.condition for x in dx3.differential_diagnoses] == ["Heat stroke"]
      and "Acute kidney injury" in dx3.complication_note, dx3.complication_note)
dx4 = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                       differential_diagnoses=[DifferentialDiagnosis(condition="Compartment syndrome")])
Stub([])._anchor_complications(RHABDO, dx4, [], [])
check("...but stays when no red flag carries it (never lost)",
      [x.condition for x in dx4.differential_diagnoses] == ["Compartment syndrome"])
dx2 = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"),
                       red_flags=[RedFlag(flag="Rhabdomyolysis", why_it_matters="x")])
Stub([])._anchor_complications(RHABDO, dx2, [], [])
check("nothing found -> nothing removed (the report is never left with no red flags)",
      [f.flag for f in dx2.red_flags] == ["Rhabdomyolysis"])
check("no profile -> untouched",
      Stub([])._anchor_complications(RHABDO, DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(
          condition="Migraine")), [], []) == 0)

# ===========================================================================
print("5. A4.4 output diet: what code knows, the model no longer writes.")
# ===========================================================================
full = TriageRequest(age=58, vitals={"systolic_bp": 85, "diastolic_bp": 50, "heart_rate": 125, "gcs": 15,
                                     "pain_score": 2}, complaint="feels unwell today")
rows = {r.parameter: r for r in R._vitals_rows(full)}
check("one row per supplied vital, none for absent ones",
      set(rows) == {"Blood pressure", "Heart rate", "GCS", "Pain score"}, str(set(rows)))
check("a triggering value cites its MTS cell and level",
      rows["Heart rate"].mts_level_triggered == "2" and "HR > 120" in rows["Heart rate"].interpretation)
check("a value that meets no cell is not called normal",
      rows["GCS"].mts_level_triggered == "" and "normal" not in rows["GCS"].interpretation.lower())
check("the diet system prompt asks for no prescriber category",
      "prescriber_category" not in R._DRUG_GROUNDING_DIET and "prescriber_category" in R._DRUG_GROUNDING_FULL)
d = DiagnosticSchema(drug_recommendations=[DrugRecommendation(drug_name="Paracetamol", route="IV")])
R._ground_prescriber_categories(d, [])
check("an unstated category is filled from FUKKM without a 'corrected' note",
      d.drug_recommendations[0].prescriber_category not in ("", R.UNVERIFIED)
      and "corrected" not in d.prescriber_category_warning, d.prescriber_category_warning)
d = DiagnosticSchema(drug_recommendations=[DrugRecommendation(drug_name="Paracetamol", route="IV",
                                                              prescriber_category="C+")])
R._ground_prescriber_categories(d, [])
check("...but a WRONG stated category is still corrected and named", "corrected" in d.prescriber_category_warning)
from app import formulary as F  # noqa: E402
check("immediate-release before prolonged-release, suppositories last",
      [e.drug_name for e in R._fitting_formulations(F.entries_for("morphine"), False)][0].endswith("Injection"))
tr = AA.Trace()
dx = DiagnosticSchema(primary_diagnosis=PrimaryDiagnosis(condition="Rhabdomyolysis"))
Stub([])._second_stage(RHABDO, dx, [], [], trace=tr)
check("the second pass is skipped when no indexed sentence answers any gap",
      tr.second_pass.startswith("skipped") or tr.second_pass.startswith("not needed"), tr.second_pass)
check("rerank defaults are the measured ones", config.RERANK_POOL == 15 and config.RERANK_MAX_LENGTH == 320)

# ===========================================================================
print("6. The curated profiles against the live index (skipped without it).")
# ===========================================================================
try:
    engine = R.get_engine()
    engine.count()
    have_index = True
except Exception as exc:  # noqa: BLE001
    print(f"  SKIP  index unavailable ({exc})")
    have_index = False
if have_index:
    expect = {("rhabdomyolysis", "Hyperkalaemia - risk of cardiac arrhythmia or arrest"): "Snakebite",
              ("rhabdomyolysis", "Acute kidney injury"): "Heat Related",
              ("heat stroke", "Rhabdomyolysis"): "Heat Related",
              ("hyperkalaemia", "Cardiac arrhythmia or arrest"): "Hyperkalaemia"}
    for p in C.PROFILES:
        for comp in p.complications:
            hit = engine._complication_sentence(p, comp, [], RHABDO)
            want = expect.get((p.name, comp.label))
            if want:
                check(f"{p.name}: '{comp.label}' is quoted from {want}",
                      hit is not None and want in hit[2].metadata.get("cpg_title", ""),
                      hit[1] if hit else "none")
            if hit:
                check(f"{p.name}: '{comp.label}' quote names no absent cause",
                      not C.CAUSE_SPECIFIC.search(hit[1]), hit[1])
    hit = engine._complication_sentence(prof, prof.complications[2], [], RHABDO)
    check("rhabdomyolysis: no KKM sentence states compartment syndrome - silence, not a foreign source",
          hit is None or hit[2].metadata.get("doc_type") != config.DOC_TYPE_EXTERNAL, hit[1] if hit else "none")

# ===========================================================================
print("7. Non-KKM source, text repair, warm-up, reference filter.")
# ===========================================================================
from app import distill as D, ingest as I  # noqa: E402
from app import red_flags as RF  # noqa: E402
ext = ("EXT CHAMP Clinical Practice Guideline for the Management of Exertional Rhabdomyolysis in "
       "Warfighters (US DoD, not KKM) 2025.pdf")
check("an 'EXT ' file is a non-KKM guideline, never a KKM CPG",
      I.classify_doc(ext) == config.DOC_TYPE_EXTERNAL
      and I.classify_doc("CPG Management of Asthma in Adults 2024.pdf") == config.DOC_TYPE_CPG_FULL)
check("its title names it as non-KKM", I.parse_title(ext).endswith("(US DoD, not KKM)"))
from app import source_policy as SP  # noqa: E402
check("...and the source policy keeps it out of the index: no KKM document cites it", not SP.allowed(ext))
check("no red-flag rule points at a non-KKM title", not any("not KKM" in t for t in RF.all_titles()))
check("it may ground a drug or a dose", config.DOC_TYPE_EXTERNAL in config.CLINICAL_DOC_TYPES)
check("the prompt tells the model the KKM source wins", "the KKM source wins" in R.SOURCE_TYPES)
check("soft hyphens are removed", I.clean_text("Mo\xad\nnitor urine") == "Monitor urine")
pages, n = I.rejoin_broken_words(["Initial diagno\nstic evaluation", "a diagnostic test", "in\nformation",
                                  "information", "heat\nstroke"])
check("a word broken across lines is re-joined when it appears whole elsewhere",
      pages[0] == "Initial diagnostic evaluation" and n == 1, str(pages))
check("two real words at a line break stay apart", pages[2] == "in\nformation" and pages[4] == "heat\nstroke")
apa = ("54. mingels, a., jacobs, l., michielsen, e., swaanenburg, j. ( 2009 ). reference population and "
       "marathon runner sera. clinical chemistry, 55 ( 1 ), 101 - 108. doi : 10. 1373 / clinchem. 55. "
       "o'hanlon, r., wilson, m., wage, r., smith, g. ( 2010 ). troponin release following endurance "
       "exercise. journal of cardiovascular magnetic resonance, 12 ( 1 ), article 38. doi : 10. 1186")
check("an APA-style reference page is recognised", D.is_bibliography(apa), str(D.citation_marks(apa)))
check("...but not a dosing chunk with a reference tail",
      not D.is_bibliography(apa + " amoxicillin 500mg po q8h plus metronidazole 400mg po q8h"))
check("warm-up is on by default and reported by /health",
      config.WARMUP and "warm" in __import__("app.schemas", fromlist=["HealthResponse"]).HealthResponse.model_fields)

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
