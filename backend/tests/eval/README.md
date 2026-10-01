# Evaluation harness

Scores the system's triage reports against **evidence-anchored answer keys**, so
"better" is a number rather than an impression, and a regression is caught
before a clinician sees it.

There is no clinician on the development team. The substitute is not "trust the
model": every answer-key item carries a **verbatim quote and page** from a
KKM/MOH document (or, where KKM is silent, a named international guideline kept
in `data/eval_sources/references/`), and a script checks that the quote is really
on that page. What this measures is **agreement with published guidance and with
expert-labelled data** - strong, auditable, but not clinical validation. Reports
keep saying "not clinically validated".

## Commands (from `backend/`)

    ../.venv/bin/python -m tests.eval.run --verify            # cases valid, every quote found
    ../.venv/bin/python -m tests.eval.run                     # run all cases via the API, then score
    ../.venv/bin/python -m tests.eval.run --tier mts_cell     # a subset
    ../.venv/bin/python -m tests.eval.run --only rhabdo-exertional-55m
    ../.venv/bin/python -m tests.eval.run --rescore data/eval_runs/<run-id>   # no model needed
    ../.venv/bin/python -m tests.eval.build_cases             # regenerate the structured-source cases

The API must be running (`./run.sh`). Cases go through HTTP, so they share the
GPU queue with the UI and are audited with origin `eval`. A run is resumable;
outputs land in `data/eval_runs/<run-id>/` (git-ignored): raw responses,
`scorecard.json`, `scorecard.md`.

**Do not edit Python under `backend/` while a run is in progress**: the dev
server runs with `--reload`, and a reload kills the case being generated.

## The 40 cases (v1, 2026-09-30)

| tier | n | where the answer key comes from | what is scored |
|---|---:|---|---|
| `management` | 13 | Two independent extraction passes over the guideline PDFs, adjudicated; only items both passes support are scored | every section |
| `mts_cell` | 15 | One printed MTS 2022 cell per case, other vitals normal (`build_cases.py`) | triage level + time to treatment |
| `ktas` | 5 | Moon et al. 2019, expert KTAS level, real disposition and ED diagnosis | triage (+/-1 band), disposition, diagnosis where recoverable |
| `mimic` | 5 | MIMIC-IV-ED demo: ESI acuity, disposition, ICD diagnosis | same as KTAS |
| `published` | 2 | MedCaseReasoning case reports (CC BY 4.0), trimmed to arrival findings | final diagnosis |

KTAS and ESI are 5-level scales like MTS but not the same scale: they are a
**band check**, and the safety signal is a level *less urgent* than the band.

## What the numbers mean

- **Section score** = F1 of precision and recall for list sections (red flags,
  differentials, actions, investigations, drugs); 0/1 for triage and
  disposition; 0/0.5/1 for diagnosis (primary / only in differentials / absent).
- **Case score** = weighted mean of the sections the case grades (triage 25,
  diagnosis 15, actions 15, red flags 10, investigations 10, drugs 10,
  differentials 5, disposition 5, grounding 5).
- **Safety counts are separate from the score**: `under_triage`,
  `forbidden_drug` (a source says avoid it here), `unsupported_drug` (no source
  in the key supports it - the heparin failure), `forbidden_source` (e.g. the
  Paediatric Protocols cited for an adult), `under_disposition`. A release must
  not increase any of them.

## Answer-key format

Each `cases/<id>.json` holds `intake` (a `TriageRequest`), `scored_sections`,
`expected` and `disputed` (items the two extraction passes did not agree on -
kept for the record, never scored). A list item is

    {"id": "ecg", "label": "12-lead ECG", "match": ["\\bECG\\b", "electrocardiogra"],
     "required": true,
     "evidence": [{"source": "<file in raw_pdfs/ or references/>", "page": 13, "quote": "..."}]}

`page` is the 1-based PDF page index. `match` regexes recognise the item in
free text; `required: false` items count toward precision but not recall.

## Found while building it

- **MTS glucose leaning ignores documented negatives.** "CBG 22, feels well" is
  printed as Level 3 (`> 18 mmol/L no symptoms`), but `_mts_floor` returns
  Level 2 because the `glucose_symptoms` qualifier leans PRESENT unless it finds
  positive evidence of symptoms, and "feels well" is not positive evidence of
  anything. An over-triage, not a safety failure; logged for Phase 2.
- **A truncated model answer is not flagged.** Baseline `ahf-72f`: the model
  looped on one sentence inside `triage_rationale` for 15,418 characters until
  the 4,000-token budget ran out (502 s). `_extract_json`'s brace repair closed
  the truncated JSON, the result validated, and every later section silently
  took its schema default - "Unspecified Diagnosis", no actions, no drugs - with
  an empty `parse_warning`. Only the completeness check hinted at it. The MTS
  floor still escalated it to Level 1 correctly. Needed: detect a generation
  that hit the token cap or needed brace repair and mark the report
  "INCOMPLETE - model output truncated"; a repetition guard or constrained
  decoding (Phase 4) to stop the loop itself.
