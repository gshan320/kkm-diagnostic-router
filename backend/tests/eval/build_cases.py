"""Generate the evaluation cases that come from structured sources.

    mts_cell    15 adult intakes, each isolating ONE printed MTS 2022 cell, with
                the cell's verbatim text and page as evidence. Answer key: the
                cell's level and the page 3 time-to-treatment.
    ktas        5 real adult ED records (Moon et al. 2019), one per expert KTAS
                level, sampled with a fixed seed. Answer key: the EXPERT level as
                a +/-1 band (KTAS is not MTS), the real disposition, and the ED
                diagnosis where the complaint could reveal it.
    mimic       5 real ED stays from the open MIMIC-IV-ED demo, spread across ESI
                acuity. Same answer-key shape as KTAS.
    published   2 published case reports (MedCaseReasoning, CC BY 4.0), trimmed
                to what was known on arrival. Answer key: the final diagnosis.

The management cases (rhabdomyolysis, ACS, asthma, ...) are NOT generated here:
their answer keys are extracted from guideline text by two independent passes
and adjudicated - see README.md.

Run from the backend directory; it rewrites only the files it owns:

    ../.venv/bin/python -m tests.eval.build_cases
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from app.schemas import TriageRequest

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
EVAL_SOURCES = HERE.parent.parent / "data" / "eval_sources"
SEED = 20260930
MTS = "MTS 2022.pdf"

TTT = {
    1: ("Level 1 - Resuscitation - 0 minutes", "0 minutes"),
    2: ("Level 2 - Emergency - under 10 minutes", "under 10 minutes"),
    3: ("Level 3 - Urgent - under 30 minutes", "under 30 minutes"),
    4: ("Level 4 - Early Care - under 60 minutes", "under 60 minutes"),
    5: ("Level 5 - Routine - under 90 minutes", "under 90 minutes"),
}

NORMAL = dict(systolic_bp=122, diastolic_bp=78, heart_rate=82, respiratory_rate=16,
              temperature=36.8, spo2=98, gcs=15, pain_score=0)


def _triage(level: int, acceptable: list[int], cells: list[tuple[str, int]], note: str = "") -> dict:
    quote, ttt = TTT[level]
    return {
        "level": level,
        "acceptable": acceptable,
        "time_to_treatment": ttt,
        "note": note,
        "evidence": [{"source": MTS, "page": p, "quote": q} for q, p in cells]
                    + [{"source": MTS, "page": 3, "quote": quote}],
    }


def _write(case: dict) -> None:
    TriageRequest.model_validate(case["intake"])  # fail loudly on a bad intake
    path = CASES / f"{case['id']}.json"
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- MTS cells
# (id, level, cell verbatim, page, complaint, vitals overrides, extra intake)
# Complaints are chosen so that no Complaints-List cell outranks the vital sign
# under test; "other vitals normal" is what isolates the cell.
MTS_CELLS = [
    ("mts-sbp-under-90", 2, "SBP < 90", 7, "Dizzy and tired since this morning.",
     dict(systolic_bp=84, diastolic_bp=56, heart_rate=96), {}),
    ("mts-bp-over-220-symptoms", 2, "BP > 220/130 with", 7,
     "Headache and blurred vision since this morning.",
     dict(systolic_bp=232, diastolic_bp=134, pain_score=3), {}),
    ("mts-hr-over-120", 2, "HR > 120", 7, "Heart racing for the past hour.",
     dict(heart_rate=136), {}),
    ("mts-spo2-under-92", 2, "SpO2 < 92%", 7, "Cough and breathlessness for two days.",
     dict(spo2=90), {}),
    ("mts-temp-over-39", 2, "Temp > 39 or < 36", 7, "Fever and chills since last night.",
     dict(temperature=39.6, heart_rate=98), {"fever_history": "fever_reported_before_arrival"}),
    ("mts-gcs-under-13", 2, "GCS < 13 or drop > 2", 7, "Found drowsy at home by family this morning.",
     dict(gcs=11), {}),
    ("mts-pain-severe", 2, "Severe Pain (8 - 10)", 7,
     "Right loin pain coming in waves and spreading to the groin.",
     dict(pain_score=9), {}),
    ("mts-hr-100-120", 3, "HR 100 - 120", 7, "Feeling weak and tired for three days.",
     dict(heart_rate=110), {}),
    ("mts-rr-20-30", 3, "RR 20 - 30", 7, "Breathless when climbing stairs for a week.",
     dict(respiratory_rate=24, spo2=96), {}),
    ("mts-spo2-92-94", 3, "SpO2 92 - 94%", 7, "Cough for five days.",
     dict(spo2=93), {}),
    ("mts-temp-37-5-39", 3, "Temp 37.5 - 39 C", 7, "Fever for two days with body aches.",
     dict(temperature=38.4), {"fever_history": "fever_reported_before_arrival"}),
    ("mts-glucose-over-18-no-symptoms", 3, "> 18 mmol/L no", 7,
     "Came for a sugar check; feels well.",
     dict(capillary_blood_glucose=22.0), {"comorbidities": ["diabetes"]}),
    ("mts-ecg-af-over-100", 3, "Atrial Fibrillation > 100", 7,
     "Irregular heartbeat noticed on a home blood pressure machine.",
     dict(heart_rate=112), {"ecg_findings": ["atrial_fibrillation_over_100"]}),
    ("mts-bp-over-180-no-symptoms", 4, "BP > 180/110 No", 7,
     "Ran out of blood pressure tablets; no complaints.",
     dict(systolic_bp=186, diastolic_bp=112), {}),
    ("mts-level-5-routine", 5, "Vital Signs within", 7,
     "Mild sore throat and runny nose for two days.",
     {}, {"fever_history": "no_fever_reported"}),
]


NOTES = {
    "mts-glucose-over-18-no-symptoms":
        "The intake DOCUMENTS the absence of symptoms ('feels well'), so the printed cell is "
        "'> 18 mmol/L no symptoms' -> Level 3. Found 2026-09-30: the leaning layer "
        "(mts_table glucose_symptoms, lean PRESENT) only looks for positive evidence and "
        "escalates this to Level 2 - an over-triage, logged for Phase 2.",
}


def build_mts() -> int:
    for cid, level, cell, page, complaint, over, extra in MTS_CELLS:
        vitals = {**NORMAL, **over}
        intake = {"age": 40, "gender": "male", "weight_kg": 70, "vitals": vitals,
                  "complaint": complaint, "history": "No known illnesses.",
                  "facility": "state_or_tertiary_with_pci_and_ct", **extra}
        if cid == "mts-glucose-over-18-no-symptoms":
            intake["history"] = "Type 2 diabetes on metformin."
        _write({
            "id": cid,
            "title": f"MTS 2022 cell isolation: '{cell}' (p{page}) -> Level {level}",
            "tier": "mts_cell",
            "provenance": "Generated by tests/eval/build_cases.py from the MTS 2022 cell named in the title.",
            "intake": intake,
            "scored_sections": ["triage"],
            "expected": {"triage": _triage(level, [level], [(cell, page)], NOTES.get(
                cid, "Every other vital sign is normal, so this cell alone sets the level."))},
            "disputed": [],
        })
    return len(MTS_CELLS)


# ---------------------------------------------------------------- KTAS
KTAS_FILE = EVAL_SOURCES / "ktas" / "ktas_moon2019_records.xlsx"
KTAS_ARRIVAL = {"1": "walk_in", "2": "ambulance", "4": "ambulance"}
KTAS_DISPOSITION = {
    "1": ["DISCHARGE_WITH_FOLLOW_UP", "REFER_SPECIALIST"],
    "2": ["ADMIT_WARD", "ADMIT_ICU_HDU"],
    "3": ["ADMIT_ICU_HDU", "RESUSCITATION_BAY"],
}
KTAS_MENTAL = {"1": "Alert on arrival.", "2": "Responds to voice only on arrival.",
               "3": "Responds to pain only on arrival.", "4": "Unconscious on arrival."}
# Hand-written for the records the seed selects: the ED diagnosis text is free
# text, so what counts as a match has to be said explicitly. None = the
# complaint cannot reveal the diagnosis, so it is not scored.
KTAS_DIAGNOSIS = {
    1008: ("Cardiac arrest with successful resuscitation",
           [r"cardiac arrest", r"post[- ]?(cardiac[- ]arrest|CPR|resuscitation)", r"\bROSC\b",
            r"return of spontaneous circulation"]),
    755: ("Non-ST elevation myocardial infarction",
          [r"\bNSTEMI\b", r"non[- ]?ST[- ]?(segment[- ])?elevation", r"myocardial infarction",
           r"acute coronary syndrome", r"\bACS\b"]),
    978: ("Gastrointestinal haemorrhage",
          [r"gastrointestinal (haemorrhage|hemorrhage|bleed)", r"\bU?GIB\b", r"\bGI bleed",
           r"(peptic|gastric|duodenal) ulcer", r"variceal"]),
    966: ("Supraventricular tachycardia",
          [r"supraventricular tachycardia", r"\bP?SVT\b", r"\bAVN?RT\b", r"narrow[- ]complex tachycardia"]),
    360: None,
}


def _xlsx_records(path: Path) -> list[dict]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            strings.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall(".//m:row", ns):
        vals = {}
        for c in r.findall("m:c", ns):
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            v = c.find("m:v", ns)
            x = v.text if v is not None else ""
            if c.get("t") == "s" and x:
                x = strings[int(x)]
            vals[col] = x
        rows.append(vals)
    header = rows[0]
    return [{h.strip(): r.get(col, "") for col, h in header.items()} | {"_row": i + 2}
            for i, r in enumerate(rows[1:])]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _band(level: int) -> list[int]:
    return [x for x in (level - 1, level, level + 1) if 1 <= x <= 5]


def build_ktas() -> int:
    if not KTAS_FILE.exists():
        print(f"  skip KTAS: {KTAS_FILE} not found")
        return 0
    recs = _xlsx_records(KTAS_FILE)
    pool = [r for r in recs if (_num(r["Age"]) or 0) >= 18 and r["Disposition"] in KTAS_DISPOSITION
            and _num(r["SBP"]) and _num(r["HR"]) and _num(r["BT"])]
    rng = random.Random(SEED)
    n = 0
    for lvl in "12345":
        r = rng.choice([x for x in pool if x["KTAS_expert"] == lvl])
        row = r["_row"]
        if row not in KTAS_DIAGNOSIS:
            raise SystemExit(f"KTAS row {row} was selected but has no hand-written diagnosis entry")
        vitals = {k: v for k, v in {
            "systolic_bp": _num(r["SBP"]), "diastolic_bp": _num(r["DBP"]), "heart_rate": _num(r["HR"]),
            "respiratory_rate": _num(r["RR"]), "temperature": _num(r["BT"]), "spo2": _num(r["Saturation"]),
            "pain_score": int(_num(r["NRS_pain"])) if _num(r["NRS_pain"]) is not None else None,
        }.items() if v is not None}
        intake = {"age": round(_num(r["Age"])), "gender": "female" if r["Sex"] == "1" else "male",
                  "vitals": vitals, "complaint": r["Chief_complain"].strip().capitalize() + ".",
                  "history": KTAS_MENTAL.get(r["Mental"], "") + (" Injury-related attendance." if r["Injury"] == "2" else "")}
        if r["Arrival mode"] in KTAS_ARRIVAL:
            intake["arrival_mode"] = KTAS_ARRIVAL[r["Arrival mode"]]
        if r["Mental"] == "4":
            intake["appearance"] = "not_responding_to_call"
        expert = int(r["KTAS_expert"])
        expected = {
            "triage": {"level": expert, "acceptable": _band(expert), "reference_scale": "KTAS",
                       "time_to_treatment": None,
                       "note": f"KTAS expert level {expert} (triage nurse gave {r['KTAS_RN']}). KTAS is not MTS: "
                               "scored as a +/-1 band; a level LESS urgent than the band is the safety failure.",
                       "evidence": []},
            "disposition": {"acceptable": KTAS_DISPOSITION[r["Disposition"]],
                            "preferred": KTAS_DISPOSITION[r["Disposition"]][0],
                            "note": "The disposition this patient actually received.", "evidence": []},
        }
        scored = ["triage", "disposition"]
        dx = KTAS_DIAGNOSIS[row]
        if dx:
            expected["diagnosis"] = {"primary": {"label": dx[0], "match": dx[1], "evidence": []},
                                     "acceptable_alternatives": []}
            scored.append("diagnosis")
        _write({
            "id": f"ktas-row{row}-expert{expert}",
            "title": f"KTAS record {row}: {r['Chief_complain']} ({r['Diagnosis in ED']})",
            "tier": "ktas",
            "provenance": "Moon SH et al. PLoS ONE 2019;14(9):e0216972, supporting data S1 (CC BY 4.0), "
                          f"spreadsheet row {row}.",
            "intake": intake, "scored_sections": scored, "expected": expected, "disputed": [],
        })
        n += 1
    return n


# ---------------------------------------------------------------- MIMIC-IV-ED demo
MIMIC = EVAL_SOURCES / "mimic_ed_demo"
MIMIC_DISPOSITION = {"HOME": ["DISCHARGE_WITH_FOLLOW_UP", "REFER_SPECIALIST"],
                     "ADMITTED": ["ADMIT_WARD", "ADMIT_ICU_HDU"]}
MIMIC_ARRIVAL = {"WALK IN": "walk_in", "AMBULANCE": "ambulance"}
MIMIC_DIAGNOSIS = {
    "39467106": ("Pneumonia", [r"pneumonia", r"\bCAP\b", r"lower respiratory tract infection", r"\bLRTI\b"]),
    "36428628": None,  # "Transfer, CVA" with an ED diagnosis of "Weakness" - not recoverable from the complaint
    "39394858": None,  # "Other chest pain" - non-specific
    "33258284": ("Ascites from chronic liver disease",
                 [r"ascites", r"cirrho", r"chronic liver disease", r"(hepatic|liver) decompensation",
                  r"decompensated (liver|cirrhosis)", r"portal hypertension"]),
    "38074673": ("Blocked catheter / catheter complication with UTI",
                 [r"catheter (blockage|obstruct|block)", r"blocked (urinary |foley |suprapubic )?catheter",
                  r"cystostomy", r"catheter[- ]associated", r"\bCAUTI\b", r"urinary tract infection", r"\bUTI\b"]),
}


def build_mimic() -> int:
    if not (MIMIC / "triage.csv.gz").exists():
        print(f"  skip MIMIC: {MIMIC} not populated")
        return 0
    rd = lambda f: list(csv.DictReader(gzip.open(MIMIC / f, "rt")))
    tri = {r["stay_id"]: r for r in rd("triage.csv.gz")}
    stays = {r["stay_id"]: r for r in rd("edstays.csv.gz")}
    pts = {r["subject_id"]: r for r in rd("patients.csv.gz")}

    def full(r):
        return all(_num(r[k]) is not None for k in
                   ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"))

    ok = [s for s, r in tri.items() if full(r) and s in stays
          and stays[s]["disposition"] in MIMIC_DISPOSITION and r["subject_id"] in pts]
    rng = random.Random(SEED)
    picked = []
    for acuity, k in (("1", 1), ("2", 2), ("3", 1), ("4", 1)):
        picked += rng.sample(sorted(s for s in ok if tri[s]["acuity"] == acuity), k)
    n = 0
    for s in picked:
        r, st = tri[s], stays[s]
        if s not in MIMIC_DIAGNOSIS:
            raise SystemExit(f"MIMIC stay {s} was selected but has no hand-written diagnosis entry")
        temp_c = round((_num(r["temperature"]) - 32) * 5 / 9, 1)
        vitals = {"systolic_bp": _num(r["sbp"]), "diastolic_bp": _num(r["dbp"]),
                  "heart_rate": _num(r["heartrate"]), "respiratory_rate": _num(r["resprate"]),
                  "temperature": temp_c, "spo2": _num(r["o2sat"])}
        if _num(r["pain"]) is not None and 0 <= _num(r["pain"]) <= 10:
            vitals["pain_score"] = int(_num(r["pain"]))
        intake = {"age": int(pts[r["subject_id"]]["anchor_age"]),
                  "gender": "female" if st["gender"] == "F" else "male",
                  "vitals": vitals, "complaint": r["chiefcomplaint"].strip() + ".", "history": ""}
        if st["arrival_transport"] in MIMIC_ARRIVAL:
            intake["arrival_mode"] = MIMIC_ARRIVAL[st["arrival_transport"]]
        acuity = int(float(r["acuity"]))
        expected = {
            "triage": {"level": acuity, "acceptable": _band(acuity), "reference_scale": "ESI",
                       "time_to_treatment": None,
                       "note": f"ESI acuity {acuity} assigned at triage. ESI is not MTS: scored as a +/-1 band.",
                       "evidence": []},
            "disposition": {"acceptable": MIMIC_DISPOSITION[st["disposition"]],
                            "preferred": MIMIC_DISPOSITION[st["disposition"]][0],
                            "note": "The disposition this patient actually received.", "evidence": []},
        }
        scored = ["triage", "disposition"]
        dx = MIMIC_DIAGNOSIS[s]
        if dx:
            expected["diagnosis"] = {"primary": {"label": dx[0], "match": dx[1], "evidence": []},
                                     "acceptable_alternatives": []}
            scored.append("diagnosis")
        _write({
            "id": f"mimic-stay{s}-esi{acuity}",
            "title": f"MIMIC-IV-ED demo stay {s}: {r['chiefcomplaint']}",
            "tier": "mimic",
            "provenance": f"MIMIC-IV-ED Demo v2.2 (ODbL), stay_id {s}; age from MIMIC-IV Demo hosp/patients.",
            "intake": intake, "scored_sections": scored, "expected": expected, "disputed": [],
        })
        n += 1
    return n


# ---------------------------------------------------------------- published cases
# MedCaseReasoning (zou-lab, CC BY 4.0), test split. Trimmed to what was known
# on arrival: later laboratory results that name the answer are left out, so
# the case tests reasoning rather than reading.
PUBLISHED = [
    {
        "id": "published-pmc10622404-rhabdomyolysis",
        "title": "Published case: fatigue, myalgia and confusion in a 65-year-old man (PMC10622404)",
        "provenance": "Clinical Case Reports 2023; PMC10622404 via MedCaseReasoning (CC BY 4.0). "
                      "Arrival findings only - the CK and myoglobin results are withheld.",
        "intake": {
            "age": 65, "gender": "male",
            "complaint": "Progressive fatigue, diffuse muscle pain and confusion; several days of worsening "
                         "leg and shoulder pain and now unable to walk independently.",
            "history": "Chronic ischaemic heart disease after coronary bypass grafting, hypertension, diabetes, "
                       "COPD and stage 3 chronic kidney disease. Hypotensive on arrival. ECG: atrial "
                       "fibrillation, left bundle-branch block, ST-segment depression and peaked, high T waves.",
            "ecg_findings": ["st_elevations_or_depressions", "tall_tented_t_waves"],
            "comorbidities": ["diabetes", "copd_or_asthma", "chronic_kidney_disease_or_dialysis"],
            "appearance": "appears_unwell", "arrival_mode": "ambulance",
            "facility": "state_or_tertiary_with_pci_and_ct",
        },
        "diagnosis": ("Rhabdomyolysis", [r"rhabdomyoly"]),
    },
    {
        "id": "published-pmc9676116-biliary-pancreatitis",
        "title": "Published case: severe right upper-quadrant and epigastric pain in a 62-year-old man (PMC9676116)",
        "provenance": "Clinical Case Reports 2022; PMC9676116 via MedCaseReasoning (CC BY 4.0). "
                      "Arrival findings only - the amylase, lipase and liver results are withheld.",
        "intake": {
            "age": 62, "gender": "male",
            "vitals": {"systolic_bp": 130, "diastolic_bp": 80, "heart_rate": 80, "respiratory_rate": 20,
                       "temperature": 37.0, "spo2": 99},
            "complaint": "Severe right upper-quadrant abdominal pain, moderate epigastric pain and chest pain, "
                         "with nausea and vomiting.",
            "history": "COVID-19 infection 14 days ago, then mild right upper-quadrant discomfort that became "
                       "severe over 3 days, worse after eating and not relieved by changing position. Chest pain "
                       "atypical for a cardiac cause. No alcohol, no trauma, no known chronic disease. Mother and "
                       "sisters had cholecystectomies. ECG: bradyarrhythmia, T-wave inversion in V1 and ST "
                       "elevation V2-V6. Chest and abdominal X-rays unremarkable apart from gallbladder sludge.",
            "ecg_findings": ["st_elevations_or_depressions"],
            "facility": "state_or_tertiary_with_pci_and_ct",
        },
        "diagnosis": ("Biliary (gallstone) pancreatitis",
                      [r"pancreatitis"]),
    },
]


def build_published() -> int:
    for p in PUBLISHED:
        label, match = p["diagnosis"]
        _write({
            "id": p["id"], "title": p["title"], "tier": "published", "provenance": p["provenance"],
            "intake": p["intake"], "scored_sections": ["diagnosis"],
            "expected": {"diagnosis": {"primary": {"label": label, "match": match, "evidence": []},
                                       "acceptable_alternatives": []}},
            "disputed": [],
        })
    return len(PUBLISHED)


def main() -> int:
    CASES.mkdir(parents=True, exist_ok=True)
    counts = {"mts_cell": build_mts(), "ktas": build_ktas(), "mimic": build_mimic(),
              "published": build_published()}
    for k, v in counts.items():
        print(f"  {k:10s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
