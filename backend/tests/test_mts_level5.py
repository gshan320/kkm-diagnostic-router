"""Regression tests for the MTS Level 5 work of 2026-09-09.

Four changes, each tied below to the real failure that motivated it AND to a
right answer that must stay silent - a guardrail that cries wolf is worse than
none.

    _mts_floor / mts_table   the adult and paediatric grids as data
    _apply_disposition_ceiling  a routine patient cannot be sent to ED observation
    _check_urgency_sanity    red flags and STAT orders a routine report invented
    ingest.extract_table_cells  one chunk per discriminator cell

Run from the backend directory, no pytest needed:

    PYTHONPATH=$PWD ../.venv/bin/python tests/test_mts_level5.py

Exits non-zero on any failure.
"""
import re, sys, pathlib


def pathlib_read_template():
    src = pathlib.Path(R.__file__).read_text()
    return src.split('json_template = """', 1)[1].split('"""', 1)[0]

from app.schemas import (
    DiagnosticSchema, Disposition, Investigation, RedFlag, TriageLevel,
    TriageRequest, Vitals,
)
from app import mts_table
from app import rag_engine as R

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def req(age=40, complaint="unwell", history="", **vitals):
    return TriageRequest(age=age, complaint=complaint, history=history, vitals=Vitals(**vitals))


# ===========================================================================
print("\n1. The missed escalations. Every one of these returned 'floor 5, no")
print("   reasons' from the half-table the old _mts_floor implemented.")
# ===========================================================================
level, reasons = R._mts_floor(req(temperature=39.5, respiratory_rate=28,
                                  systolic_bp=230, diastolic_bp=135))
check("39.5 C + RR 28 + BP 230/135 -> Level 2", level == 2, f"got {level}")
check("  and names why", bool(reasons), str(reasons))

for name, kw, want in [
    ("Temp 39.5 (>39)", dict(temperature=39.5), 2),
    ("Temp 35.5 (<36)", dict(temperature=35.5), 2),
    ("Temp 38.0 (37.5-39)", dict(temperature=38.0), 3),
    ("RR 34 (>30)", dict(respiratory_rate=34), 2),
    ("RR 24 (20-30)", dict(respiratory_rate=24), 3),
    ("BP 230/135 no symptoms", dict(systolic_bp=230, diastolic_bp=135), 3),
    ("BP 190/115 no symptoms", dict(systolic_bp=190, diastolic_bp=115), 4),
    ("CBG 22 with symptoms", dict(capillary_blood_glucose=22.0), 2),
    ("CBG 14 (12-18)", dict(capillary_blood_glucose=14.0), 4),
    ("HR 125 (>120)", dict(heart_rate=125), 2),
    ("SpO2 88 (<90, page 4)", dict(spo2=88), 1),
]:
    got, _ = R._mts_floor(req(**kw))
    check(f"{name} -> Level {want}", got == want, f"got {got}")

# ===========================================================================
print("\n2. Right answers that must stay silent, and the escalations that")
print("   were already load-bearing and must not regress.")
# ===========================================================================
normal = dict(systolic_bp=120, diastolic_bp=78, heart_rate=78, respiratory_rate=16,
              temperature=36.8, spo2=99, gcs=15, capillary_blood_glucose=5.4,
              pain_score=0)
level, reasons = R._mts_floor(req(complaint="blocked nose and sneezing", **normal))
check("wholly normal adult -> Level 5", level == 5, f"got {level}")
check("  with no alarm raised", reasons == [], str(reasons))

# MTS 2022 p7 Level 5 requires literally "No Pain"; p7 also says the higher
# modifier wins when two do not differentiate. pain_score 1 is therefore NOT
# Level 5, and an earlier claim that the model over-triaged this case was wrong.
level, _ = R._mts_floor(req(complaint="runny nose", **{**normal, "pain_score": 1}))
check("same patient with pain 1/10 -> Level 4, not 5", level == 4, f"got {level}")

level, _ = R._mts_floor(req(age=62, complaint="crushing chest pain",
                            systolic_bp=78, heart_rate=124, spo2=93))
check("shocked STEMI still Level 1", level == 1, f"got {level}")

level, _ = R._mts_floor(req(gcs=7))
check("GCS 7 still Level 1", level == 1, f"got {level}")

# ===========================================================================
print("\n3. Paediatrics. The old floor had no paediatric bands at all and")
print("   explicitly declined to hard-code them; page 14 supplies them.")
# ===========================================================================
level, reasons = R._mts_floor(req(age=6, complaint="cough", respiratory_rate=50,
                                  heart_rate=120, spo2=95, temperature=38.2))
check("6-year-old, RR 50 -> Level 2", level == 2, f"got {level}")

level, reasons = R._mts_floor(req(age=0.5, complaint="mild rash", heart_rate=130,
                                  respiratory_rate=30, systolic_bp=85, spo2=98,
                                  temperature=37.0))
check("well infant floors at Level 4, never 5", level == 4, f"got {level}")
check("  and cites p13/p14 for it", any("p1" in r for r in reasons), str(reasons))

level, _ = R._mts_floor(req(age=9, complaint="sore throat", **normal))
check("well 9-year-old floors at Level 4", level == 4, f"got {level}")

# The paediatric SpO2 column is stricter than the adult one: < 92% is Level 1
# for a child where an adult at 91% is Level 2.
level, _ = R._mts_floor(req(age=4, complaint="wheeze", spo2=91))
check("child SpO2 91% -> Level 1 (adult would be 2)", level == 1, f"got {level}")

check("mts_table lists its own deviations", len(mts_table.deviations()) >= 5,
      str(len(mts_table.deviations())))

# ===========================================================================
print("\n4. Disposition ceiling. _enforce_disposition floored MTS 1 and 2")
print("   upward but capped nothing, which sent a Level 4 rhinitis patient")
print("   to ED observation.")
# ===========================================================================
def diag(level, disposition, **kw):
    return DiagnosticSchema.model_validate(
        {"mts_triage_level": level, "disposition": disposition, **kw}
    )

d = diag(5, "ED_OBSERVATION")
R._enforce_disposition(d, req(complaint="runny nose", **normal))
check("MTS 5 + ED observation -> discharge",
      d.disposition is Disposition.DISCHARGE_WITH_FOLLOW_UP, str(d.disposition))
check("  and the triage-away requirements are stated",
      "note recording down their complaints" in d.disposition_justification,
      d.disposition_justification)

d = diag(4, "ADMIT_WARD")
R._enforce_disposition(d, req(complaint="earache"))
check("MTS 4 + admit ward -> ED observation",
      d.disposition is Disposition.ED_OBSERVATION, str(d.disposition))

# The ceiling is a permitted SET, not a point on the intensity ladder:
# REFER_SPECIALIST outranks ED_OBSERVATION there, and capping by rank would
# block the outpatient ORL referral the rhinosinusitis algorithm asks for.
d = diag(5, "REFER_SPECIALIST")
R._enforce_disposition(d, req(complaint="chronic nasal blockage"))
check("MTS 5 + specialist referral is left alone",
      d.disposition is Disposition.REFER_SPECIALIST, str(d.disposition))

d = diag(4, "DISCHARGE_WITH_FOLLOW_UP")
R._enforce_disposition(d, req(age=5, complaint="mild cough"))
check("child sent home carries the p13 caution",
      "should not be routinely triaged-away" in d.disposition_justification,
      d.disposition_justification)

d = diag(1, "DISCHARGE_WITH_FOLLOW_UP")
R._enforce_disposition(d, req(complaint="cardiac arrest"))
check("MTS 1 floor still raises to resuscitation bay",
      d.disposition is Disposition.RESUSCITATION_BAY, str(d.disposition))

d = diag(2, "ADMIT_ICU_HDU")
R._enforce_disposition(d, req(complaint="severe chest pain"))
check("MTS 2 already at ICU is not 'corrected' down",
      d.disposition is Disposition.ADMIT_ICU_HDU, str(d.disposition))

# ===========================================================================
print("\n5. Urgency sanity. The rhinitis report listed 'Mild pain (Pain Score")
print("   1/10)' as a RED FLAG and ordered 'Allergy testing - STAT'.")
# ===========================================================================
d = diag(5, "DISCHARGE_WITH_FOLLOW_UP",
         red_flags=[{"flag": "Mild pain (Pain Score 1/10)"}])
R._check_urgency_sanity(d, req(complaint="runny nose and sneezing for 3 days"))
check("red flag nothing supports is removed at MTS 5", d.red_flags == [], str(d.red_flags))
check("  and the removal is named", "Mild pain" in d.urgency_warning, d.urgency_warning)

d = diag(4, "ED_OBSERVATION",
         investigations=[{"test": "Allergy testing", "urgency": "STAT"}])
R._check_urgency_sanity(d, req(complaint="itchy nose"))
check("STAT order at MTS 4 is flagged", "STAT" in d.urgency_warning, d.urgency_warning)
check("  but the investigation is not deleted", len(d.investigations) == 1)

# Right answers that must stay silent.
d = diag(5, "DISCHARGE_WITH_FOLLOW_UP",
         red_flags=[{"flag": "Chest pain radiating to the jaw"}])
R._check_urgency_sanity(d, req(complaint="central chest pain radiating to jaw, sweating"))
check("red flags are KEPT when a red_flags.py rule fires", len(d.red_flags) == 1,
      str(d.red_flags))
check("  and nothing is warned about", d.urgency_warning == "", d.urgency_warning)

d = diag(2, "ADMIT_WARD",
         red_flags=[{"flag": "Hypotension"}],
         investigations=[{"test": "Troponin", "urgency": "STAT"}])
R._check_urgency_sanity(d, req(complaint="chest pain"))
check("a genuinely urgent report is untouched",
      d.urgency_warning == "" and len(d.red_flags) == 1, d.urgency_warning)

# ===========================================================================
print("\n6. Table-aware ingest. The triage grid was 25 unusable prose chunks.")
# ===========================================================================
import pymupdf
from pathlib import Path
from collections import Counter
from app import config
from app.ingest import (
    extract_table_cells, extract_action_rows, describe_pdf, clean_text, _doc_key,
    MTS_LEVEL_CLEARANCE, IMPLICIT_LEVEL_COLUMNS,
)

path = config.RAW_PDF_DIR / "MTS 2022.pdf"
if not path.exists():
    check("MTS 2022.pdf present", False, str(path))
else:
    doc = pymupdf.open(path)
    pages = [clean_text(p.get_text()) for p in doc]
    meta = describe_pdf(path, "\n".join(pages[:4]), "\n".join(pages))
    cells = list(extract_table_cells(doc, meta, _doc_key(meta.filename)))
    by_level = Counter(c.metadata["mts_level"] for c in cells)
    check("discriminator cells extracted", len(cells) > 300, str(len(cells)))
    check("all five levels represented",
          set(by_level) - {MTS_LEVEL_CLEARANCE} == {1, 2, 3, 4, 5}, str(dict(by_level)))
    check("Level 5 cells exist", by_level[5] >= 30, str(by_level[5]))

    # --- 2026-09-11: the SECONDARY TRIAGE column of pages 4-5 -------------
    # It is not a level. It used to be swept into key_cols, so its criteria
    # were concatenated onto the row label of the Level 1/2/3 cells beside it
    # and a RESUSCITATION cell was labelled "Warm, pink, pulses normal".
    clearance = [c for c in cells if c.metadata["mts_level"] == MTS_LEVEL_CLEARANCE]
    check("SECONDARY TRIAGE column is captured, not discarded",
          len(clearance) == 5, str(len(clearance)))
    check("clearance cells come only from the Primary Triage pages",
          {c.metadata["page_number"] for c in clearance} == {4, 5},
          str({c.metadata["page_number"] for c in clearance}))
    check("clearance cells state they are not a triage level",
          all("NOT a triage level" in c.text for c in clearance))
    check("clearance level is outside every discriminator_cells fetch range",
          all(MTS_LEVEL_CLEARANCE not in range(max(1, lvl - 1), min(5, lvl + 1) + 1)
              for lvl in (1, 2, 3, 4, 5)))
    p45 = [c for c in cells
           if c.metadata["page_number"] in (4, 5) and c.metadata["mts_level"] > 0]
    check("clearance criteria no longer contaminate Level 1-3 row labels",
          not any(
              needle in c.metadata["row_label"]
              for c in p45
              for needle in ("Walking", "Warm, pink", "Sit upright", "Minimal /")
          ),
          str(sorted({c.metadata["row_label"] for c in p45})[:2]))
    check("Primary Triage row labels survive intact",
          {c.metadata["row_label"] for c in p45} == {
              "APPEARANCE",
              "RESPIRATORY DISTRESS - Airway breathing - SpO2",
              "SHOCK STATE - Peripheries - Pulses - AVPU",
              "CONSCIOUS LEVELS - Airway - AVPU - Brief Neuro",
              "BLEEDING - Seen External - Suspect Internal - Bleeding Disorders"
              " - Anticoagulant therapy",
          },
          str(sorted({c.metadata["row_label"] for c in p45})))

    # --- 2026-09-11: the two tables on pages 5-6 with no level columns ----
    actions = list(extract_action_rows(doc, meta, _doc_key(meta.filename)))
    check("action rows extracted from the non-level tables",
          len(actions) == 15, str(len(actions)))
    check("action rows can never reach discriminator_cells()",
          all(a.metadata["is_table_cell"] is False for a in actions))
    check("the adult complaints list is not double-indexed as action rows",
          not any(a.metadata["table_caption"].upper() in IMPLICIT_LEVEL_COLUMNS
                  for a in actions),
          str({a.metadata["table_caption"] for a in actions}))
    check("TB placement is now backed by an indexed row",
          any("Active Tuberculosis" in a.metadata["row_label"]
              and "Negative Pressure" in a.text for a in actions))
    check("HAZMAT decontamination is now backed by an indexed row",
          any("HAZMAT" in a.metadata["row_label"]
              and "Decontamination" in a.text for a in actions))
    check("Code GREY behavioural actions are indexed",
          any("Code GREY" in a.text for a in actions))
    check("the table-wide PPE note is kept, not dropped with its empty row",
          any(a.metadata["row_kind"] == "note" and "PPE" in a.text for a in actions))
    check("action-row and cell ids never collide",
          len({c.id for c in cells + actions}) == len(cells) + len(actions))

    l5 = [c for c in cells if c.metadata["mts_level"] == 5]
    check("Level 5 is adult-only, as the document intends",
          {c.metadata["cohort"] for c in l5} == {"ADULT"},
          str({c.metadata["cohort"] for c in l5}))
    check("'No Pain' is attached to Level 5, not floating in prose",
          any("No Pain" in c.text for c in l5))
    check("ENT routine presentations reachable at Level 5",
          any("EAR / ENT" in c.metadata["row_label"] for c in l5))
    check("inferred headers are marked as such",
          all(c.metadata["header_inferred"] is True
              for c in cells if c.metadata["page_number"] in (8, 9, 10, 11)))
    check("printed headers are not marked inferred",
          all(c.metadata["header_inferred"] is False
              for c in cells if c.metadata["page_number"] == 7))
    check("mid-word wraps are rejoined",
          any("DEHYDRATION" in c.metadata["row_label"] for c in cells),
          str([c.metadata["row_label"] for c in cells if "DEHYDRA" in c.metadata["row_label"]][:3]))

# ===========================================================================
print("\n7. Citations. The 2026-09-09 rhinitis run listed FUKKM alone while")
print("   supporting_cpg correctly named the rhinosinusitis CPG, so the report")
print("   quoted a guideline it never cited.")
# ===========================================================================
from app.schemas import RetrievedSource

def src(sid, title, page, year="2016", doc_type="CPG_FULL", filename=""):
    return RetrievedSource(source_id=sid, filename=filename or f"{title}.pdf",
                           cpg_title=title, edition_year=year, doc_type=doc_type,
                           page_number=page, excerpt=f"excerpt from {title}")

SOURCES = [
    src("S1", "Malaysian Triage Scale", 7, "2022", "TRIAGE_PROTOCOL"),
    src("S2", "Management of Rhinosinusitis in Adolescents and Adults", 12),
    src("S3", "Management of Rhinosinusitis in Adolescents and Adults", 28),
    src("S4", "Management of Asthma in Adults", 74, "2024"),
    src("S5", "FUKKM \u2014 MOH Medicines Formulary (Blue Book)", None, "2024", "DRUG_FORMULARY"),
]

# The exact failure: the model cites only the formulary, but names the CPG.
d = DiagnosticSchema.model_validate({
    "mts_triage_level": 5,
    "primary_diagnosis": {"condition": "Allergic rhinitis",
                          "supporting_cpg": "Management of Rhinosinusitis in Adolescents and Adults"},
    "drug_recommendations": [{"drug_name": "Cetirizine", "source_id": "S5"}],
    "citations": [{"source_id": "S5", "document": "FUKKM", "page": "N/A", "edition_year": "2024"}],
})
notes = R._ground_citations(d, SOURCES)
cited = [c.source_id for c in d.citations]
check("the named CPG is now cited, not just FUKKM", "S2" in cited, str(cited))
check("the formulary is still cited", "S5" in cited, str(cited))
check("  and nothing is warned about", notes == [], str(notes))
check("page and year come from the index, not the model",
      any(c.source_id == "S2" and c.page == "12" and c.edition_year == "2016"
          for c in d.citations), str([(c.source_id, c.page, c.edition_year) for c in d.citations]))

# Compactness: only what is cited, never the whole retrieved set.
check("uncited sources are NOT listed", "S4" not in cited and "S1" not in cited, str(cited))
check("citation count stays small", len(d.citations) == 2, str(len(d.citations)))
check("citations are in retrieval order", cited == sorted(cited, key=lambda i: int(i[1:])),
      str(cited))
check("only the first chunk of a guideline is cited, not every one",
      "S3" not in cited, str(cited))

# Inline [S#] in prose counts as a citation.
d = DiagnosticSchema.model_validate({
    "mts_triage_level": 3,
    "triage_rationale": "Tachypnoea meets the Level 3 band [S1].",
    "primary_diagnosis": {"condition": "Asthma exacerbation", "supporting_cpg": "Asthma in Adults"},
})
notes = R._ground_citations(d, SOURCES)
cited = [c.source_id for c in d.citations]
check("an [S#] cited only in the rationale is picked up", "S1" in cited, str(cited))
check("a loosely-named guideline still resolves", "S4" in cited, str(cited))

# A guideline named but never retrieved is a grounding problem, not a citation.
d = DiagnosticSchema.model_validate({
    "mts_triage_level": 4,
    "primary_diagnosis": {"condition": "Gout", "supporting_cpg": "Management of Gout"},
    "immediate_actions": [{"sequence": 1, "action": "Rest the joint", "source_id": "S1"}],
})
notes = R._ground_citations(d, SOURCES)
check("a guideline that was never retrieved is flagged",
      any("not in the retrieved sources" in n for n in notes), str(notes))
R._check_citations(d, SOURCES, notes)
check("  and the flag reaches citation_warning",
      "Management of Gout" in d.citation_warning, d.citation_warning)

# A report that cites nothing at all must say so.
d = DiagnosticSchema.model_validate({"mts_triage_level": 5})
notes = R._ground_citations(d, SOURCES)
check("a report citing nothing is flagged", any("traceable" in n for n in notes),
      str(notes))
check("  and no citations are invented", d.citations == [], str(d.citations))

# An id the model invented must not reach the table.
d = DiagnosticSchema.model_validate({
    "mts_triage_level": 3,
    "immediate_actions": [{"sequence": 1, "action": "Give oxygen", "source_id": "S99"}],
})
R._ground_citations(d, SOURCES)
check("a fabricated source id is not cited", d.citations == [], str(d.citations))

check("the prompt no longer asks for a citations array",
      "citations" not in pathlib_read_template())

# ===========================================================================
print("\n8. The 2026-09-09 clinical sign-off. Eleven thresholds were put to")
print("   the user; these lock in the answers. A SIGNED OFF rule is settled -")
print("   a later pass must not tidy it back toward the printed grid.")
# ===========================================================================

# --- item 4: SBP < 90 is one printed cell read as two -----------------------
# Level 1 when the hypotension comes with decompensated shock or organ
# hypoperfusion (p4 SHOCK STATE); Level 2 when the patient is high-risk,
# confused or unstable but still breathing and perfusing (p7, as printed).
for name, kw, complaint, want in [
    ("SBP 78 + HR 124 (shocked STEMI)", dict(systolic_bp=78, heart_rate=124, spo2=93),
     "crushing chest pain", 1),
    ("SBP 65 (profound, alone)", dict(systolic_bp=65), "feeling faint", 1),
    ("SBP 85 + HR 44 (p4 bradycardia in shock)", dict(systolic_bp=85, heart_rate=44),
     "unwell", 1),
    ("SBP 85 + SpO2 91 (hypoxia + hypotension)", dict(systolic_bp=85, spo2=91),
     "unwell", 1),
    ("SBP 88 + cold mottled peripheries", dict(systolic_bp=88), "cold peripheries, mottled skin", 1),
    ("SBP 78 alone, perfusing", dict(systolic_bp=78), "dizzy on standing", 2),
    ("SBP 88 + confused (p4 puts 'Confused' at L2)", dict(systolic_bp=88),
     "confused since this morning", 2),
]:
    got, reasons = R._mts_floor(req(complaint=complaint, **kw))
    check(f"{name} -> Level {want}", got == want, f"got {got}: {reasons}")

# --- item 9: bradycardia 40 - 50 is not printed in MTS at all ---------------
# Narrowed to SYMPTOMATIC bradycardia. This is the cry-wolf fix: the number
# alone used to escalate every beta-blocked and athletic patient.
level, reasons = R._mts_floor(req(complaint="routine medication review",
                                  **{**normal, "heart_rate": 48}))
check("HR 48, nothing else wrong -> Level 5, not 3", level == 5, f"got {level}: {reasons}")
check("  and no alarm is raised", reasons == [], str(reasons))

level, _ = R._mts_floor(req(complaint="syncope while standing at the bus stop",
                            **{**normal, "heart_rate": 48}))
check("HR 48 WITH syncope -> Level 3", level == 3, f"got {level}")

level, _ = R._mts_floor(req(complaint="unwell", **{**normal, "heart_rate": 38}))
check("HR 38 -> Level 2 (p7 'Bradycardia < 40', printed)", level == 2, f"got {level}")

# --- item 3: the leaning layer ---------------------------------------------
level, reasons = R._mts_floor(req(complaint="headache and blurred vision",
                                  systolic_bp=230, diastolic_bp=135))
check("BP 230/135 + documented symptoms -> Level 2", level == 2, f"got {level}")
check("  and the reason says the symptoms were documented",
      any("(documented)" in r for r in reasons), str(reasons))

level, reasons = R._mts_floor(req(complaint="came for a BP check",
                                  systolic_bp=230, diastolic_bp=135))
check("BP 230/135 + nothing recorded -> Level 3 (leans absent)", level == 3, f"got {level}")
check("  and the reason says the lean was assumed, not observed",
      any("(assumed absent)" in r for r in reasons), str(reasons))

# The regex bug this layer fixed: `\b(...|confus|...)\b` can never match
# "confused", so a hypertensive patient with documented confusion was leaned
# ASYMPTOMATIC and held at Level 3.
level, _ = R._mts_floor(req(complaint="confused and vomiting since morning",
                            systolic_bp=230, diastolic_bp=135))
check("BP 230/135 + 'confused and vomiting' -> Level 2", level == 2, f"got {level}")

level, reasons = R._mts_floor(req(complaint="found drowsy", capillary_blood_glucose=2.0))
check("CBG 2.0 -> Level 2 (glucose leans present)", level == 2, f"got {level}")
level, reasons = R._mts_floor(req(complaint="routine screening", capillary_blood_glucose=2.0))
check("CBG 2.0 with nothing recorded is STILL Level 2", level == 2, f"got {level}")

check("every qualifier states which way it leans and why",
      mts_table.audit() == [], str(mts_table.audit()))

# --- item 2: the paediatric floor, and the thorn-in-the-knee case ----------
# Page 14's LEAST ACUTE paediatric column is Level 4, so a wholly normal child
# lands at 4 by the table itself, before p13's policy sentence is consulted.
paed_normal = dict(systolic_bp=100, heart_rate=95, respiratory_rate=22,
                   temperature=36.8, spo2=99, gcs=15, capillary_blood_glucose=5.4,
                   pain_score=0)
level, reasons = R._mts_floor(req(age=7, complaint="thorn pricks to the knee",
                                  **paed_normal))
check("7-year-old, thorns, no pain -> Level 4 by p14's own table", level == 4, f"got {level}")

# The thorn question, resolved. Page 14 prints "Little pain" at LEVEL 3 for a
# child over 5, which made a mildly upset child Urgent where an adult with the
# same score is Level 4. Set to 4 on 2026-09-09, following the paediatric
# COMPLAINTS LIST ("Mild symptoms" in WOUNDS / SKIN) over page 14 - the one
# place in this module that reads pp15-18 in preference to p14.
level, reasons = R._mts_floor(req(age=7, complaint="thorn pricks to the knee, mildly upset",
                                  **{**paed_normal, "pain_score": 2}))
check("same child, mildly upset (pain 2) -> Level 4, not p14's Level 3",
      level == 4, f"got {level}: {reasons}")
check("  and the constant is what says so",
      mts_table.PAED_LITTLE_PAIN_LEVEL == 4, str(mts_table.PAED_LITTLE_PAIN_LEVEL))
level, _ = R._mts_floor(req(complaint="thorn pricks to the knee, mildly upset",
                            **{**normal, "pain_score": 2}))
check("  and an ADULT with the same pain score agrees at Level 4", level == 4,
      f"got {level}")

# All three rungs now read from pp15-18, so the ladder is continuous and runs
# parallel to the adult row. Page 14's reading (1 / 2 / 3) put a child with a
# limb fracture in the RESUSCITATION BAY, because _MIN_DISPOSITION floors an
# MTS 1 patient there.
check("all three rungs read from pp15-18, not p14",
      (mts_table.PAED_SEVERE_PAIN_LEVEL,
       mts_table.PAED_MODERATE_PAIN_LEVEL,
       mts_table.PAED_LITTLE_PAIN_LEVEL) == (2, 3, 4),
      str((mts_table.PAED_SEVERE_PAIN_LEVEL, mts_table.PAED_MODERATE_PAIN_LEVEL,
           mts_table.PAED_LITTLE_PAIN_LEVEL)))

# Age-appropriate normals, so PAIN is the only discriminator that can fire.
PAED_NORMALS = {
    2: dict(heart_rate=110, respiratory_rate=25, systolic_bp=100, spo2=98, temperature=37.0),
    4: dict(heart_rate=110, respiratory_rate=25, systolic_bp=100, spo2=98, temperature=37.0),
    7: dict(heart_rate=95, respiratory_rate=22, systolic_bp=100, spo2=99, temperature=36.8),
    11: dict(heart_rate=95, respiratory_rate=22, systolic_bp=110, spo2=99, temperature=36.8),
}

def paed_pain(age, score):
    return R._mts_floor(req(age=age, complaint="fell off a swing", gcs=15,
                            capillary_blood_glucose=5.4, pain_score=score,
                            **PAED_NORMALS[age]))

ladder = {score: paed_pain(7, score)[0] for score in range(11)}
for score, want in [(0, 4), (1, 4), (4, 4), (5, 3), (7, 3), (8, 2), (10, 2)]:
    check(f"child pain {score}/10 -> Level {want}", ladder[score] == want,
          f"got {ladder[score]}")
check("  the ladder is continuous - no step wider than one level",
      all(ladder[i] - ladder[i + 1] in (0, 1) for i in range(10)), str(ladder))
check("  a child in severe pain is Emergency, not Resuscitation",
      1 not in set(ladder.values()), str(ladder))

# THE HOLE THIS CLOSED: the row was gated on age > 5, so a young child in
# severe pain produced no pain finding at all and floored at 4 on a fracture.
for age in (2, 4, 7, 11):
    check(f"age {age} is scored on the SAME ladder",
          [paed_pain(age, s)[0] for s in (0, 4, 5, 8)] == [4, 4, 3, 2],
          str([paed_pain(age, s)[0] for s in (0, 4, 5, 8)]))
level, reasons = paed_pain(2, 9)
check("2-year-old at pain 9/10 -> Level 2 (was Level 4, no pain finding at all)",
      level == 2, f"got {level}: {reasons}")
check("  and the reason says the score must have been OBSERVED",
      any("(observational)" in r for r in reasons), str(reasons))
level, reasons = paed_pain(11, 9)
check("  where an 11-year-old's could be self-reported",
      any("(self-reported)" in r for r in reasons), str(reasons))
level, reasons = paed_pain(7, 9)
check("  and a 7-year-old's by FACES or observation",
      any("(FACES or observational)" in r for r in reasons), str(reasons))
check("  the instrument source is the Paediatric Protocols, not MTS",
      "Alder Hey" in str(mts_table.deviations()) and
      "Paediatric Protocols" in mts_table.PAED_PAIN_SOURCE,
      mts_table.PAED_PAIN_SOURCE)
for age, tag in [(1, "observational"), (3, "observational"),
                 (5, "FACES or observational"), (9, "self-reported")]:
    check(f"  age {age} -> {tag}", mts_table.paed_pain_instrument(age)[0] == tag,
          mts_table.paed_pain_instrument(age)[0])

# The paediatric ladder runs parallel to the adult one and no longer above it.
# The single permitted exception is pain 0, where the child is held at Level 4
# by the paediatric floor while the adult reaches Level 5 - that is p13's rule,
# not the pain row.
for score in range(11):
    child, _ = paed_pain(11, score)
    adult, _ = R._mts_floor(req(complaint="fell over", **{**normal, "pain_score": score}))
    floor_case = score == 0 and (child, adult) == (4, 5)
    check(f"  pain {score}: child L{child} vs adult L{adult}",
          child >= adult or floor_case, f"child {child} vs adult {adult}")
child0, _ = paed_pain(11, 0)
adult0, _ = R._mts_floor(req(complaint="fell over", **{**normal, "pain_score": 0}))
check("  and the one difference left at pain 0 is the p13 Level 4 floor",
      (child0, adult0) == (4, 5), f"child {child0} vs adult {adult0}")

# p13's wording is graduated, so the sentence a clinician is shown must be the
# one that actually governs the child in front of them.
for age, phrase in [(0.5, "Infants, below 1 years old"),
                    (4, "especially those below 8"),
                    (10, "Generally")]:
    level, reasons = R._mts_floor(req(age=age, complaint="mild rash"))
    check(f"age {age:g}, no vitals -> Level 4", level == 4, f"got {level}")
    check(f"  and cites the p13 sentence for that age ({phrase!r})",
          any(phrase in r for r in reasons), str(reasons))

check("the three p13 sentences are actually different",
      len({mts_table.paediatric_policy(a) for a in (0.5, 4, 10)}) == 3)

# --- item 11 / boundary corrections: "> x" and "< x" cells are EXCLUSIVE ----
# Each of these returned one level too acute before 2026-09-09.
for name, kw, want in [
    ("child SpO2 92% (p14 L1 is '< 92%', L2 is '92 - 94%')", dict(age=6, spo2=92), 2),
    ("child Temp 40.0 C (p14 L2 is '39 - 40C')", dict(age=6, temperature=40.0), 2),
    ("infant SBP 60 (p14 L1 is 'SBP < 60', L2 is '60 - 70')",
     dict(age=0.5, systolic_bp=60), 2),
    ("neonate RR 60 (p14 L1 is '> 60 bpm', L2 is '> 50 bpm')",
     dict(age=0.1, respiratory_rate=60), 2),
    ("child SpO2 91% is still Level 1", dict(age=6, spo2=91), 1),
]:
    got, reasons = R._mts_floor(req(complaint="unwell", **kw))
    check(f"{name} -> Level {want}", got == want, f"got {got}: {reasons}")

check("no paediatric value falls outside every printed cell",
      mts_table.audit() == [], str(mts_table.audit()))

# --- items 1, 5 - 8: kept as transcribed, now locked ----------------------
for name, kw, want in [
    ("paediatric pain 8 -> L2 (pp15-18 'Severe Pain > 7')", dict(age=8, pain_score=8), 2),
    ("adult pain 8 -> L2 (p7)", dict(pain_score=8), 2),
    ("adult pain 2 -> L4 (no printed cell; by elimination)", dict(pain_score=2), 4),
    ("GCS 13 -> L2 (p7 prints '< 13', which excludes 13)", dict(gcs=13), 2),
    ("GCS 14 -> L3 (p4 'Altered mental state')", dict(gcs=14), 3),
    ("GCS 8 -> L1 (p4 'Unresponsive', no number printed)", dict(gcs=8), 1),
]:
    got, reasons = R._mts_floor(req(complaint="unwell", **kw))
    check(f"{name} -> Level {want}", got == want, f"got {got}: {reasons}")

# The deviation record is the audit trail for all of the above.
devs = mts_table.deviations()
known = {mts_table.STATUS_SIGNED_OFF, mts_table.STATUS_POLICY,
         mts_table.STATUS_CORRECTED, mts_table.STATUS_CONFLICT,
         mts_table.STATUS_OPEN}
check("every deviation carries a status and a reason",
      all(d.status in known and d.why.strip() for d in devs),
      str([d.what for d in devs if d.status not in known or not d.why.strip()]))
by_status = {}
for d in devs:
    by_status.setdefault(d.status, []).append(d.what)
check("the signed-off thresholds are recorded as signed off",
      len(by_status.get(mts_table.STATUS_SIGNED_OFF, [])) >= 6,
      str(by_status.get(mts_table.STATUS_SIGNED_OFF)))
check("the p14-vs-pp15-18 pain conflict is recorded WITH its resolution",
      any("pain ladder" in d.what and "RESOLVED" in d.what and "pp15-18" in d.why
          for d in devs),
      str(by_status.get(mts_table.STATUS_CONFLICT)))
check("the resuscitation-bay consequence is on the record",
      any("RESUSCITATION_BAY" in d.why for d in devs))
check("the age-gate removal cites the Paediatric Protocols",
      any("Alder Hey" in d.why and "no lower age bound" in d.why.lower()
          for d in devs))
# Reduced from >= 5 on 2026-09-10: five of the seven cells this assertion was
# written to guard - Appears Septic/Ill, Immunocompromised, Appears unwell, the
# fever row and the whole ECG row - are now IMPLEMENTED, so they are correctly
# no longer OPEN. What must stay true is that the remaining gaps are still
# named rather than quietly dropped.
check("the cells this module cannot evaluate are still recorded as OPEN",
      len(by_status.get(mts_table.STATUS_OPEN, [])) == len(
          mts_table.UNIMPLEMENTED_ADULT_CELLS) >= 1,
      str(by_status.get(mts_table.STATUS_OPEN)))
check("cells left out ON PURPOSE are recorded as a decision, not as a gap",
      len(mts_table.DELIBERATELY_NOT_IMPLEMENTED) >= 3
      and all(any(cell in d.what for d in devs)
              for _, _, cell, _ in mts_table.DELIBERATELY_NOT_IMPLEMENTED))
check("no cell is recorded as both an accidental and a deliberate omission",
      not ({c for _, c, _ in mts_table.UNIMPLEMENTED_ADULT_CELLS}
           & {c for _, _, c, _ in mts_table.DELIBERATELY_NOT_IMPLEMENTED}))
check("page 18's Levels 2-5 conflict is recorded against the paediatric floor",
      any("Levels" in d.why and "2 - 5" in d.why for d in devs))


# ===========================================================================
print("\n9. The prompt block is GENERATED from mts_table, not re-typed beside it.")
# ===========================================================================
from app import mts_table as M
from app.schemas import Vitals as _V

block = M.prompt_rules()

# The four places the hand-written MTS_2022_GUARDRAILS had drifted by
# 2026-09-11. Every one of them taught the model a less acute level than the
# grid applies, and _enforce_mts_level then overrode the answer.
check("Level 5 time is the page 3 standard, not the 120 minutes the prompt said",
      "under 90 minutes" in block and "120 min" not in block)
check("severe adult pain starts at 8, not 7",
      "Severe Pain (8 - 10)" in block and "Pain Score 7-10" not in block)
check("the adult moderate-pain band runs to 7, not 6",
      "Pain Score 4 - 7" in block and "Pain Score 4-6" not in block)
check("the HR bands are the printed ones",
      "HR > 120" in block and "HR 100 - 120" in block and "101-130" not in block)

# Drift cannot come back: every rendered line is a cell floor() applies.
# Only the cell lines - the Level 5 block is prose, not a cell.
_CELL_LINE = re.compile(r"^ {4}([A-Za-z][A-Za-z0-9 ]*): (.+) \(p(\d+)\)$")
rendered = [m.group(0).strip() for m in
            (_CELL_LINE.match(l) for l in block.splitlines()) if m]
def _rendered(rule):
    return f"{rule.parameter}: {rule.prompt_text or rule.verbatim} (p{rule.page})"

measurable = {_rendered(r) for r in M.ADULT_RULES
              if r.parameter in M.PROMPT_PARAMETERS}
# The Primary Triage observations, Levels 1-2 only - what the officer SEES.
# These were hand-typed prose on the morning of 2026-09-11, the same day the
# drifted hand-typed block was deleted, which is exactly how drift arrives.
observational = {_rendered(r) for r in M.ADULT_RULES
                 if r.parameter in M.PROMPT_OBSERVATIONAL
                 and r.level <= M.PROMPT_OBSERVATIONAL_MAX_LEVEL}
cells = measurable | observational
check("every prompt line is a real cell from ADULT_RULES",
      all(line in cells for line in rendered),
      str([l for l in rendered if l not in cells][:3]))
check("every measurable cell reaches the prompt",
      len(rendered) == len(cells), f"{len(rendered)} rendered vs {len(cells)} cells")
check("the two groups do not overlap",
      not (measurable & observational))
check("the observational group is stated, and only at Levels 1-2",
      len(observational) >= 10
      and all(r.level <= 2 for r in M.ADULT_RULES
              if _rendered(r) in observational))
check("nothing in the prompt is hand-typed prose passing as a cell",
      "PRIMARY TRIAGE OBSERVATIONS" in block
      and not hasattr(R, "MTS_PRIMARY_TRIAGE")
      and not hasattr(R, "MTS_2022_GUARDRAILS")
      and not hasattr(R, "MTS_SCALE"))
check("Level 5 is stated even though no measurable cell carries it",
      "LEVEL 5 - ROUTINE" in block and "ELIMINATION" in block)
check("the audit record keeps the verbatim cell the prompt rewords",
      any(r.verbatim == "No Pain (Level 5 column)"
          and r.prompt_text.startswith("Pain Score 1 - 3") for r in M.ADULT_RULES))

# The prompt and the report field read one table.
check("rag_engine's time table IS mts_table's, not a copy",
      R.MTS_TIME is M.TIME_TO_TREATMENT)
for lvl in (1, 2, 3, 4, 5):
    d = DiagnosticSchema(mts_triage_level=lvl)
    d.time_to_treatment = "whatever the model typed"
    R._ground_triage_timing(d)
    check(f"  L{lvl} time is server-derived", d.time_to_treatment == M.TIME_TO_TREATMENT[lvl],
          d.time_to_treatment)
check("the page 3 re-triage rule reaches Levels 3-5",
      all(R._reassess(l) for l in (3, 4, 5)))
check("and not Levels 1-2, where nobody is waiting",
      not any(R._reassess(l) for l in (1, 2)))

# ===========================================================================
print("\n10. Document typing. TRIAGE_PROTOCOL is privileged, not a topic label.")
# ===========================================================================
from app.ingest import classify_doc
from app import ingest as ingest_module

check("MTS 2022 is the triage protocol",
      classify_doc("MTS 2022.pdf") == config.DOC_TYPE_TRIAGE)
check("a MaHTAS HTA report is not a clinical guideline",
      classify_doc("MaHTAS Health Technology Assessment National Early Warning "
                   "Score (NEWS) 2020.pdf") == config.DOC_TYPE_HTA)
check("a state redirection policy is not a clinical guideline",
      classify_doc("JKN Selangor Emergency Medicine and Trauma Services "
                   "Redirection Policy 2024.pdf") == config.DOC_TYPE_POLICY)
check("an MOH clinical guideline still types as one",
      classify_doc("MOH Pain Management in Emergency and Trauma Department "
                   "(2nd Edition) 2020.pdf") == config.DOC_TYPE_CPG_FULL)
check("a future triage guideline book would type as the triage protocol",
      classify_doc("Buku Garis Panduan Triage 2026.pdf") == config.DOC_TYPE_TRIAGE)
# Nearly every MOH clinical guideline is published BY MaHTAS and says so on its
# cover. A CPG whose filename carries that word must stay a CPG, or it silently
# leaves CLINICAL_DOC_TYPES and stops being able to ground a drug or a dose.
check("the CPG prefix beats the MaHTAS publisher name",
      classify_doc("CPG Early Management of Head Injury in Adults (MaHTAS) "
                   "2015.pdf") == config.DOC_TYPE_CPG_FULL)
check("the QR prefix beats it too",
      classify_doc("QR Management of Tuberculosis MaHTAS 2021.pdf")
      == config.DOC_TYPE_CPG_QR)
check("every adjunct type carries a source-type caution on every chunk",
      set(config.ADJUNCT_DOC_TYPES) <= set(ingest_module.DOC_TYPE_CAUTION))
check("the NEWS dual-version chart page is called out by page",
      ("National Early Warning Score", 23) in ingest_module.PAGE_CAUTIONS)

# Pain Management in ETD 2020 grades pain for ANALGESIA and disagrees with the
# MTS grid at a score of exactly 7 - it calls 7-10 severe, MTS prints
# "Severe Pain (8 - 10)". Both are current MOH documents answering different
# questions. The hand-typed prompt block replaced on 2026-09-11 carried THIS
# document's bands (7-10 and 4-6), which is how the drift arrived.
pain_caution = next(
    (v for k, v in ingest_module.DOC_CAUTIONS.items() if "Pain Management" in k), ""
)
check("the pain guideline carries a band-conflict caution", bool(pain_caution))
check("  it states the MTS band verbatim", "Severe Pain (8 - 10)" in pain_caution)
check("  it says the zone colours are not MTS levels",
      "NOT" in pain_caution and "ANALGESIA zones" in pain_caution)
check("the code is unaffected either way - floor() reads MTS alone",
      M.floor(40, _V(pain_score=7), "", [])[0] == 3
      and M.floor(40, _V(pain_score=8), "", [])[0] == 2)
check("neither adjunct type can ground a drug or a dose",
      not set(config.ADJUNCT_DOC_TYPES) & set(config.CLINICAL_DOC_TYPES))
check("the triage protocol is not a drug-grounding type either",
      config.DOC_TYPE_TRIAGE not in config.CLINICAL_DOC_TYPES)

print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("All checks passed.")
