# KKM Diagnostic Router & CPG Comparative Intelligence — Phase 1

Local, **fully offline** RAG prototype over official Malaysian Ministry of Health
(KKM) clinical documents. Retrieval, embeddings and generation all run on this
machine — no clinical text leaves it. Two modes:

| Mode | Endpoint | Input | Output |
|---|---|---|---|
| **A — Clinical Triage Router** | `POST /api/v1/triage` | age, gender, weight, vitals, complaint, history | `DiagnosticSchema` JSON: MTS 2022 level, red flags, primary diagnosis + differentials, ordered immediate actions, investigations, KKM drug dosing with FUKKM prescriber category, disposition, evidence gaps, citations — plus the intake echoed back |
| **B — CPG & Formulary Explorer** | `POST /api/v1/compare-inquire` | freeform question, scope | Comparative markdown (answer / comparison / key differences / caveats / sources) with `[S#]` citations |

Supporting endpoints: `GET /health`, `GET /api/v1/stats`.

## Stack

- **Backend** FastAPI + Pydantic v2, ChromaDB (persistent, cosine), PyMuPDF,
  `sentence-transformers/all-MiniLM-L6-v2` (local embeddings, 384-dim).
- **LLM** A **local MLX model on Apple Silicon** via `mlx-lm` — default
  `mlx-community/Llama-3.2-3B-Instruct-4bit`, overridable with
  `LOCAL_MLX_MODEL`. Weights are loaded once into unified memory on the first
  request (`_ensure_model_loaded`) and reused for the process lifetime, so the
  first call after boot is slow and later ones are not.
  **No API key, no network call at inference time.**
- **Frontend** Next.js 16 App Router, React 19, Tailwind v4, react-markdown +
  remark-gfm (comparative tables).

### How Mode A gets structured JSON

A small local model cannot be constrained by a server-side schema, so the
pipeline earns its structure in three steps (`rag_engine.triage`):

1. The prompt carries an explicit **JSON template** of every field the schema
   expects, alongside the MTS 2022 mapping rules and a worked ACS example.
2. The reply is scraped for its first `{...}` block (`re.search(r'\{.*\}')`) and
   handed to `DiagnosticSchema.model_validate_json`. Invalid JSON raises
   `RagError` and the endpoint returns 502 — nothing half-parsed is served.
3. **Deterministic guardrails overrule the model afterwards** — see below.

Every value in the JSON template is an obvious `<slot marker>`, never sample
clinical prose. An earlier template carried plausible text
("Reason based on worst criterion (e.g., Pain Score 8)") and the model copied it
verbatim into a real patient's `triage_rationale`; `_flag_leaked_slots` now
detects any marker that survives into the answer and appends a note to
`evidence_gaps`. The LLM-authored models also inherit `_LLMOut`
(`coerce_numbers_to_str=True`), because a small model writes
`"mts_level_triggered": 3` where the schema wants `"3"` — without coercion that
is a 502 on an otherwise usable answer.

### The intake: all four MTS inputs

MTS 2022 page 7 states that the final level "takes into consideration all of the
selected parameters ie. **Primary Triage, Vital Signs, Complaints List and
Initial Tests**". Until 2026-09-10 `TriageRequest` carried two of those four —
the vital signs, and the complaints list as free text. Everything the Primary
Triage Officer records on pages 4–5, and the ECG half of the Initial Tests on
page 7, had no field to arrive in, so every cell in them sat unevaluated in
`UNIMPLEMENTED_ADULT_CELLS`.

The optional half of the intake closes that. It is optional in the strict
sense: **an unset field matches no rule and never asserts the reassuring
value**, which is the same contract the vitals already had and the reason a
patient entered the old way cannot change level because the fields now exist.
`tests/test_triage_modifiers.py` §2 is that property.

| Field | Source | What it unlocks |
|---|---|---|
| `appearance` | p4 Critical First Look, p7 | `Appears Septic, Ill` (L2), `Not responding to call` (L2), `Appears unwell` (L3), `Cannot sit / stand unsupported` (L3) |
| `breathing` | p4 Respiratory Distress | The speech-and-effort ladder, L1–L3. This is what SpO₂ alone misses: a compensating asthmatic holds a normal saturation while speaking in short phrases, and p4 puts that patient at Level 2 on the speech cell alone |
| `perfusion` | p4 Shock State, p13 Circulation | `CRT > 2 seconds` (L2) through `Absent Radial Pulse` (L1). The early sign of paediatric shock — page 14's SBP bands are a late one |
| `bleeding` | p5 Bleeding | A whole printed row, L1–L3, previously unreachable |
| `ecg_findings` | p7 Initial Tests | All eleven printed ECG findings across four levels. Also the sharpest retrieval lever in the intake: the corpus holds *ACS (1st Ed, 2026)* and *NSTE-ACS (3rd Ed, 2021)* as separate documents and nothing else distinguishes them |
| `paediatric_signs` | p13 | The Paediatric Assessment Triangle danger signs. Page 13 makes them the decision on whether a child is triaged at all, and not one appears in page 14's bands |
| `fever_history` | p7 Temp row | `History of Fever` / `No documented fever` (L4), read as one modifier |
| `comorbidities` | p7 (`Immunocompromised`, L2) | One MTS cell; the rest change no level and exist so the drug checks fire on a recorded fact |
| `onset` | not MTS | The strongest CPG-**section** selector available: the stroke thrombolysis window, the ACS reperfusion clock, the dengue phase |
| `allergies`, `current_medications`, `egfr` | not MTS | Make `contraindications.py` fire on a recorded fact rather than on whether the word was typed into the history box |
| `exposure_risk`, `behavioural_risk` | p5–p6 | Placement and Code GREY. The printed output column is "TO BE PLACED AT", so these change no level |
| `arrival_mode`, `trauma_mechanism` | p6, not MTS | Context; `trauma_mechanism` steers the head-injury and abdominal-trauma pathways |
| `pregnancy`, `gestation_weeks`, `breastfeeding` | not MTS | Replaces `pregnant: bool`, which had no way to say "not established" |
| `facility` | not MTS | A remembered setting, not a per-patient field. Changes the recommendation, never the level: a reperfusion decision is unanswerable without knowing whether there is a cath lab |

**Column membership was read from the PDF word coordinates, not from the
extracted text.** The text layer emits each column top-to-bottom and then the
next, so a cell printed low in the Level 2 column reads as though it followed
the Level 3 items. That had already produced two wrong levels in the
`UNIMPLEMENTED_ADULT_CELLS` record — `ST elevations or depressions` (x=131, so
**Level 2**, not the Level 3 it was listed under) and `History of Fever`
(x=358, so **Level 4**, not Level 3). Neither had reached a patient, because
neither cell was implemented; both would have on the day they were.

Three printed cells are **deliberately** not implemented, and
`mts_table.DELIBERATELY_NOT_IMPLEMENTED` records each with its reasoning. Each
describes a normal patient in a column above Level 5 — page 4's L3 "Peripheries
warm, CRT normal" is the clearest — so reading them literally would escalate
every patient who was examined carefully, making the examination itself the
risk factor.

One more trap is worth recording, because it recurred three times while this
was built. `contraindications.py` decides a teratogen is absolutely
contraindicated on the regex `\bpregnan`. Three natural phrasings of the new
intake contain that substring while describing a patient who is *not* known to
be pregnant — "suspected ectopic pregnancy", "pregnancy status not
established", "pregnancy excluded" — and each would have produced a false
ABSOLUTE, including on **methotrexate, which is the correct treatment for an
ectopic**. The intake therefore renders twice: `modifier_lines()` for the
report and the prompt, and `modifier_lines(for_matching=True)` for the one
consumer that pattern-matches it. The unestablished case is a real, graded
check instead — `teratogen_when_gravid_status_unknown`, severity CAUTION,
keyed on a phrase carrying no `pregnan` substring at all.

### Deterministic guardrails

Prompt instructions do not hold on a 4-bit 3B model, so anything that must be
true is enforced in code after validation, in `rag_engine.py`:

The prompt itself is now **generated** from the same table the guardrails apply.
`MTS_2022_GUARDRAILS` used to be a hand-typed restatement of the grid beside
`mts_table.py`, and by 2026-09-11 it had drifted from it in four places, every
one in the under-triage direction: severe pain taught as 7-10 rather than 8-10,
the moderate band as 4-6 rather than 4-7, an "HR 101-130" Level 3 band the
document does not print, and a Level 5 time of 120 minutes where page 3 says 90.
`_enforce_mts_level` corrected the *outcome* every time, so no patient was
mis-triaged - but the model was reasoning from wrong numbers and then being
overridden, which a clinician reads as a guardrail note on a report whose
rationale argues for the wrong level. `mts_table.prompt_rules()` renders the
block from `ADULT_RULES` in two groups - `PROMPT_PARAMETERS` for the measurable
cells and `PROMPT_OBSERVATIONAL` for the Primary Triage findings at Levels 1-2,
which is the route to an acute level for a patient whose vital signs are not
informative. A test asserts every rendered line is a cell the code actually
applies, that the two groups do not overlap, and that none of the three old
hand-typed constants still exists.

Worth recording how easily this recurs: the first version of the Primary Triage
half was itself written as hand-typed prose, on the same morning the drifted
block was deleted. It was replaced with generated output before shipping.

| Guardrail | What it does |
|---|---|
| `_enforce_mts_level` | Computes the most acute level the **vitals alone** justify (`_mts_floor`) and escalates the model to it — age-specific hypotension (APLS: `70 + 2×age` for 1–10 y, 90 above), GCS bands, SpO₂ bands, pain score, hypoglycaemia, adult heart-rate bands. De-escalates only when nothing objective supports an acute level. Every change appends `[System Guardrail: …]` to the rationale, naming the observation that forced it |
| `_ground_prescriber_categories` | Matches each recommended drug against the **retrieved** FUKKM chunks by base name. No match → `prescriber_category` is forced to `NOT_IN_RETRIEVED_SOURCES`. Match with a different letter → corrected to the chunk's value. `prescriber_category_warning` is then **rewritten from what was actually found** — the model does not get to author that field |
| `_flag_leaked_slots` | Appends a note to `evidence_gaps` when a `<template slot>` was copied instead of filled |
| `_ground_triage_timing` | Sets `time_to_treatment` from the **final** level rather than trusting the model, and attaches the MTS p3 re-triage rule at Levels 3-5. Runs last of the checks that can move a level. Before 2026-09-11 the field was only written when `_enforce_mts_level` changed something, so a report the guardrail agreed with kept whatever the model typed - including the schema default of "under 30 minutes" on an MTS 5 patient |
| `_check_redirection` | Warns when the report sends a patient out of the ETD whom the **JKN Selangor Redirection Policy 2024 s4.2** says must be seen there - 14 of its 16 clauses in code, the other 2 named as uncheckable. Advisory, never an override: the policy is state-level and MTS 2022 carries no redirection criteria at all. See `app/redirection.py` |

Observed working: on an adult ACS case the model claimed `Aspirin` was category
`A`; no FUKKM Aspirin chunk had been retrieved, so the category was forced to
`NOT_IN_RETRIEVED_SOURCES` and the warning named the drug. That is what makes
the UI's red "CATEGORY UNVERIFIED" chip trustworthy rather than aspirational.

Mode B is a single non-streaming `generate()` call returning markdown.

## Setup

```bash
./init_project.sh                 # venv, pip, npm, directories
source .venv/bin/activate
cd backend && python -m app.ingest --reset   # build the index
./backend/run.sh                             # API on :8000  (/docs for OpenAPI)
cd frontend && npm run dev                   # UI  on :3000
```

`./init_project.sh --ingest` does the ingestion step for you.

`backend/.env` is **optional** — every setting has a working default. The one
thing that does need the network is the very first inference: `mlx-lm` downloads
the model weights from Hugging Face and caches them under `~/.cache/huggingface`.
Pre-pull them if the deployment machine is air-gapped.

## Ingestion

```bash
cd backend
python -m app.ingest --dry-run    # parse + chunk + report, nothing written
python -m app.ingest --reset      # rebuild from scratch
python -m app.ingest              # incremental upsert (IDs are deterministic)
python -m app.ingest --stats      # what is indexed right now
```

Per-page PDF extraction keeps `page_number` citations exact. Metadata attached
to every chunk: `filename`, `cpg_title`, `edition_year`, `edition`, `doc_type`,
`page_number` / drug fields, `version_date`, `status`, `chunk_index`. `doc_type`
is one of `TRIAGE_PROTOCOL`, `CPG_FULL`, `CPG_QUICK_REFERENCE`,
`PAEDIATRIC_PROTOCOL`, `DRUG_FORMULARY`, `HTA_REPORT`, `PATIENT_FLOW_POLICY`,
and Mode A retrieves against it three times rather than once, so a drug question
cannot crowd out the triage criteria. Every query also filters
`status = "active"`, so superseded chunks can be retired without deleting them.

#### Not every authoritative document is a clinical guideline

Added 2026-09-11 with the NEWS HTA report and the Selangor redirection policy.
Both are MOH documents and both belong in the corpus; neither is a guideline,
and typing them as one would have been a safety problem rather than a tidiness
one.

`config.CLINICAL_DOC_TYPES` is the list a drug recommendation may be grounded
in. `_check_drug_indications` and `_dose_sentence` both read it and skip any
chunk outside it, so a doc_type left out of that list **cannot support a drug or
supply a dose** - which is the entire reason `HTA_REPORT` exists. A Health
Technology Assessment is an appraisal *of* a tool: 122 pages of systematic
review in the NEWS case, dense with sentences like "NEWS >= 5, 30-day mortality
OR 11.8 (95%CI 4.26, 32.6)". Those are study findings that read like clinical
thresholds, and `_DOSE_IN_TEXT` would happily have matched some of them.

Two further protections, because metadata alone does not travel into the prompt:

- **Every chunk of an adjunct type carries a `SOURCE TYPE` line** in its own
  text (`ingest.DOC_TYPE_CAUTION`), so the model is told what it is reading at
  the point it reads it rather than being expected to infer it from a doc_type
  string in the header.
- **`PAGE_CAUTIONS`** annotates a page whose text layer is actively misleading.
  One entry so far: page 23 of the NEWS report prints the **NEWS (2012) and
  NEWS2 (2017) parameter charts side by side**, and the extractor flattens them
  into a single run with the two version labels emitted *after* both tables. The
  chunk therefore offers two different threshold sets for the same six vital
  signs - a respiratory rate of 22 scores 0 on one chart and 2 on the other -
  with nothing in the text attributing a row to a version. The caution says so
  and forbids computing a score from that page.

**`DOC_CAUTIONS`** is the third of the three, for a fact about one document
rather than a whole type. One entry, and it is the most useful thing the
2026-09-11 work found. *Pain Management in Emergency and Trauma Department, 2nd
Ed 2020* grades pain differently from MTS 2022, and they disagree at exactly one
score:

| score | Pain Management in ETD 2020 (p10) | MTS 2022 (p7) |
|---|---|---|
| 1-3 | MILD, green zone | Level 4 |
| 4-6 | MODERATE, yellow zone | Level 3 |
| **7** | **SEVERE, red zone** | **Level 3** |
| 8-10 | SEVERE, red zone | Level 2 |

Both are current MOH documents and **both are right** - MTS grades how long a
patient may *wait*, the pain guideline grades which *analgesia* they get. A
patient at 7 is Level 3 and needs the severe-pain protocol, with no
contradiction. What makes it dangerous is that the pain guideline states its
bands as **zone colours**, and MTS levels carry colours too: "SEVERE PAIN, Pain
Score 7-10, RED ZONE" reads like an instruction to triage a 7 as MTS red. It is
an instruction to give morphine.

This also explains the prompt drift described under *Deterministic guardrails*.
The hand-typed block said "Pain Score 7-10" at Level 2 and "4-6" at Level 3 -
**this document's bands**, not the MTS grid's. The drift was never random; it
was another MOH document's numbers, which is how the error arrived and why it
went unnoticed. `_mts_floor` was never at risk - it reads `ADULT_RULES` only.

Classification (`ingest.classify_doc`) is filename-keyed and **order matters**.
The `CPG ` and `QR ` prefixes are tested first, because nearly every MOH
guideline is published by MaHTAS and says so on its cover; without that order a
file named `CPG ... (MaHTAS) ....pdf` would type as an HTA report and silently
lose the ability to ground a drug. `HTA_REPORT` and `PATIENT_FLOW_POLICY` are
then tested **before** `TRIAGE_PROTOCOL`, because both documents are *about*
triage decisions and `TRIAGE_PROTOCOL` is privileged: it is the only type whose
tables are mined into discriminator cells, and `discriminator_cells()` now
filters on it explicitly. Only MTS 2022 may hold that type.

### Four-layer retrieval

`all-MiniLM-L6-v2` could not bridge symptom text to disease vocabulary. Measured:
the TB CPG sat at cosine **0.306** for "pulmonary tuberculosis sputum smear" but
**0.564** for "chronic cough haemoptysis night sweats", and fell out of the top 60
entirely. A textbook TB presentation silently returned asthma chunks. Four layers
now sit between an intake and the context, each catching what the one before it
structurally cannot:

1. **MedEmbed-large-v0.1** — medical-domain bi-encoder, 1024 dims, 512 tokens.
2. **BM25 lexical channel**, fused with the dense list by Reciprocal Rank Fusion.
   Fusion is by *rank*, so a BM25 relevance score and a cosine distance never have
   to be made commensurable. This channel matches `haemoptysis` as a string —
   deterministic, and explainable to a clinician in a way a cosine value is not.
   Its tokenizer preserves `2g`, `0.9%`, `mg/kg`, because a dose is exactly what
   it exists to catch.
3. **`BAAI/bge-reranker-base` cross-encoder** over the fused pool. It reads query
   and chunk *together*, so it can judge interactions ("calf swelling" + "pleuritic
   chest pain" -> PE) that a bi-encoder cannot represent.
4. **Deterministic red-flag floor** (`app/red_flags.py`) — 19 auditable rules, each
   with a written clinical rationale. Layers 1-3 make a miss rare; only this makes
   it impossible for the listed presentations. Rules only ever ADD sources, so a
   firing rule can never mask a correct retrieval, and `validate_titles()` runs at
   start-up so a renamed file cannot leave a dead rule masquerading as a safety net.

Measured on 12 clinical presentations (retrieval hit = the correct guideline is
returned):

| configuration | hits | mean rank | speed |
|---|---|---|---|
| MiniLM baseline (previous stack) | 6/8 | — | 0.07 s |
| L1 MedEmbed dense only | 11/12 | 2.5 | 0.07 s |
| L1+L2 +BM25 hybrid | **12/12** | 2.2 | 0.08 s |
| L1+L2+L3 +cross-encoder | **12/12** | 1.8 | 3.6 s |
| L1+L2+L3+L4 (full) | **12/12** | **1.3** | 2.0 s |

Layer 4 is not decoration: with layers 2-3 disabled, L1 alone misses head injury
and the red-flag floor recovers it (11/12 -> 12/12).

`CANDIDATE_K` was tuned, not guessed. A *deeper* pool is both slower and worse,
because it feeds the cross-encoder more distractors: 60 -> mean rank 1.8 at 3.3 s;
30 -> 1.3 at 2.0 s; 20 degrades again to 2.1. Held at 30.

Because mean rank is now 1.3, wide retrieval became pure context cost — recall
stays 12/12 down to `CPG_K=6`. `CPG_K` was cut 18 -> 12, kept above the minimum so
the model still receives several chunks *of* the right guideline (dose tables,
algorithm steps), not merely the one chunk that proves it was found.

Every source carries a `retrieval` field — `semantic`, `term match`,
`semantic+term match`, or `red-flag rule` — so a reader can tell an embedding
match from a literal clinical-term match from a safety rule. `retrieval_notes`
records which rule fired and the phrase that fired it, including when the rule
fired but retrieval had already found the guideline.

Mode A's three retrievals (`rag_engine.triage`):

1. **Triage criteria** — MTS 2022 only, queried with complaint + vitals.
2. **Clinical guidance** — `CLINICAL_DOC_TYPES`, which includes the Paediatric
   Protocols; a paediatric patient gets a second, protocol-only pass
   (`PAEDS_K`). This list was previously hardcoded to CPG types only, so the
   1,190-chunk Paediatric Protocols — a third of the corpus — was never
   retrieved for Mode A at all.
3. **Formulary** — queried with the **treatment sentences found in step 2**
   (`_formulary_query`), not with the patient's symptoms. Querying FUKKM with
   "fever, bleeding, vomiting" returned snake antivenom and antihistamines for a
   dengue child; querying it with the dengue algorithm's own words ("Commence
   Normal Saline at 5–7 ml/kg") returns crystalloids, colloids and ORS.

Current corpus, after the clean rebuild of **2026-09-11**: **8,973 chunks** —
7,277 from **53 PDFs**, 1,696 from 1,686 FUKKM drug records. Zero ingest
warnings, one collection segment, 147 MB on disk. Every document carries an
edition year and a real text layer; there are no scanned PDFs.

| chunks | doc_type |
|---:|---|
| 5,244 | `CPG_FULL` |
| 1,696 | `DRUG_FORMULARY` |
| 1,174 | `PAEDIATRIC_PROTOCOL` |
| 389 | `TRIAGE_PROTOCOL` |
| 227 | `CPG_QUICK_REFERENCE` |
| 225 | `HTA_REPORT` |
| 18 | `PATIENT_FLOW_POLICY` |

`TRIAGE_PROTOCOL`'s 389 is MTS 2022 alone and breaks down as 25 prose chunks +
**344 discriminator cells** + **5 clearance cells** + **15 action rows**. The
document was 25 chunks — 0.3% of the index — before any of it was extracted as
structure.

## Printable reports

Every Mode A result carries **Download report** and **Print / PDF** buttons
(`frontend/src/lib/report.ts`). Both render one self-contained HTML document —
no external CSS, fonts or scripts — that opens on any machine and paginates
cleanly to A4. It reproduces the intake (blank vitals printed as *not measured*),
the full findings, a citations table, a blank cross-check sign-off block, and a
one-line evidence-base footer naming the documents the findings were drawn from.
"Print / PDF" opens the browser print dialog against the same document, so
"Save as PDF" produces the archival copy.

The report reads its patient details from `TriageResponse.request`, which the API
echoes back verbatim — an exported file therefore carries its own inputs and does
not depend on browser state surviving a reload.

## Data quality

Both issues that shipped with the first cut of this prototype have since been
**resolved in the current index**. Kept here because the recovery paths still
matter if the corpus is rebuilt.

1. **FUKKM column mis-alignment — fixed.** The original scrape read the listing
   table by fixed column index and was off by one, leaving no generic drug name
   anywhere in `fukkm_database.json`. `scrape_fukkm.py` was rewritten to map
   columns by **header text** and to refuse to overwrite a good file when
   validation fails; `ingest.py` still detects the old `column_shifted` shape and
   re-maps what is recoverable, so a stale file degrades rather than corrupts.
   The indexed data now carries real names (`Abacavir Sulphate 300mg tablet`),
   real prescriber categories (`A*`, `B`, `A`, `A/KK`, `C`, `C+`), indications
   (1,676/1,697) and **real dosage text (1,675/1,697 — 99%)**, including
   paediatric weight bands (*"Children: i. Weighing 14 to <20kg: one-half of…"*).
   To rebuild:

   ```bash
   cd backend
   python -m app.scrape_fukkm --max-pages 2   # sample, inspect the output
   python -m app.scrape_fukkm                 # full re-scrape
   python -m app.ingest --reset
   ```

   The rewritten scraper already follows each drug's detail page, so dosing is
   captured — the earlier "listing page has no dose column" limitation no longer
   applies.

2. **Missing edition years — fixed.** Every indexed PDF now resolves an
   `edition_year`, so comparative answers can date their sources.
   One value is worth a second look: `CPG Management of Acute Coronary
   Syndromes (1st Edition).pdf` resolves to **2026**, which is almost certainly a
   stray year picked up from the document body rather than the edition date.
   Rename the file to carry the correct year (the parser prefers the filename)
   and re-ingest.

3. **MTS 2022 Primary Triage cells were self-contradicting - fixed 2026-09-11.**
   A re-scan of the triage document already in the corpus, prompted by the
   question "is there anything in this file we are still not extracting?",
   found three things. The first was not a gap but a live defect.

   `CRITICAL FIRST LOOK` (p4) and `RAPID ASSESSMENT` (pp4-5) print **five**
   columns: `LEVEL 1 / LEVEL 2 / LEVEL 3 / SECONDARY TRIAGE`. The last is not a
   triage level - it is the clearance criteria that let a patient leave Primary
   Triage with no escalation. `extract_table_cells` treated every non-level
   column as part of the row label, so those criteria were **concatenated onto
   the discriminator of the Level 1-3 cells beside them**:

   ```
   Discriminator: SHOCK STATE - Peripheries - Pulses - AVPU
                  / - Warm, pink, pulses normal - Alert, walking
   MTS Level 1 (RESUSCITATION) criteria: - Pale, cyanosed, cold peripheries ...
   ```

   Twelve cells read like that, and `discriminator_cells()` force-fetches them
   with `forced=True` on every Level 1-2 case, so the model was handed a chunk
   labelling a resuscitation criterion "warm, pink, alert, walking". Row labels
   now come from the columns that *lead* the table, and a trailing non-level
   column becomes its own chunk at `MTS_LEVEL_CLEARANCE` - level **0**, chosen
   because `discriminator_cells()` filters on `range(level-1, level+2)` bounded
   at 1, so a clearance cell can never be served as a triage criterion.

   The other two were genuine gaps. **Fifteen rows on pp5-6** - `INFECTIOUS
   DISEASES / HAZMAT` and `AGGRESSIVE / POTENTIALLY VIOLENT PERSONS` - have no
   level columns at all and were skipped outright, surviving only as flattened
   prose. Their output column is a *placement* (`Isolation (Negative Pressure)`,
   `Decontamination`, Code GREY), the intake has collected the inputs for them
   since 2026-09-10 (`exposure_risk`, `behavioural_risk`), and nothing indexed
   backed those enum values. `extract_action_rows` now emits one chunk per row
   with `is_table_cell=False`, which is what keeps a placement rule out of the
   triage-level evidence. And **page 3's re-triage rule** - reassess every hour
   until seen, immediately on any change - was prose-only and never reached a
   report; `_ground_triage_timing` now attaches it.

   MTS 2022 yields **344 discriminator cells + 5 clearance cells + 15 action
   rows**. `tests/test_mts_level5.py` sections 6 and 9 hold the regressions.

Note that `RagEngine.corpus_warnings` currently returns warnings **only** when
the index is empty. Ingestion-time findings — mis-aligned columns, unparseable
PDFs, missing years — are logged by `python -m app.ingest` but are no longer
surfaced through `GET /api/v1/stats` or the UI banner. Read the ingest log after
rebuilding the corpus; do not treat a clean `/stats` as a clean corpus.

## Chunk size caveat

The spec asks for 500-token chunks with 50-token overlap, which is what ships.
But `all-MiniLM-L6-v2` truncates its input at **256 tokens**, so the tail of each
chunk does not influence its embedding vector (the full chunk is still handed to
the LLM). For lossless retrieval:

```bash
echo 'CHUNK_TOKENS=256' >> backend/.env
cd backend && python -m app.ingest --reset
```

## Configuration (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LOCAL_MLX_MODEL` | `mlx-community/Qwen3-8B-4bit` | local MLX model for both modes. Benchmarked — see Model selection |
| `MAX_TOKENS_TRIAGE` | `4000` | generation cap, Mode A |
| `MAX_TOKENS_INQUIRY` | `32000` | generation cap, Mode B |
| `EMBEDDING_MODEL` | `abhinand/MedEmbed-large-v0.1` | 1024-dim medical retrieval embedder. **Changing this requires `ingest --reset`** |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `512` / `64` | chunking (lossless: the embedder takes 512) |
| `CANDIDATE_K` | `30` | pool size the cross-encoder reranks. Measured optimum — see Retrieval |
| `HYBRID_ENABLED` / `RERANK_ENABLED` | `1` / `1` | A/B a retrieval layer out without touching code |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `RED_FLAGS_ENABLED` / `RED_FLAG_CHUNKS_PER_RULE` / `RED_FLAG_MAX_CHUNKS` | `1` / `3` / `15` | deterministic recall floor |
| `TRIAGE_K` / `CPG_K` / `DRUG_K` | `8` / `12` / `8` | Mode A retrieval depth |
| `PAEDS_K` | `6` | extra Paediatric Protocols pass for children |
| `PAEDIATRIC_AGE_YEARS` | `12` | paediatric/adult cut-off |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS |

`backend/.env` is optional; `.env.example` lists every override that exists.
Mode B takes its retrieval depth from the request's `max_sources`, not from
config. `GET /health` reports the model actually in use and a `model_loaded`
flag — false until the first request has pulled the weights into memory.

## Grounding

The Mode A system prompt requires a `[S#]` citation for every statement, pins
the MTS level to the **worst single criterion** across pain score, complaint
modifiers and vitals, and forbids Level 1/2 for mild pain absent ABC collapse —
with the deterministic guardrail above enforcing that last rule in code rather
than trusting the model. Blank vitals are reported as gaps, never as normal.

`DRUG_GROUNDING` (rules 7-8 of `TRIAGE_SYSTEM`) asks for the right behaviour,
and `_ground_prescriber_categories` **enforces** it — the prompt alone was
measured not to hold. Prescriber categories in the output are therefore either
traceable to a retrieved FUKKM chunk or explicitly marked
`NOT_IN_RETRIEVED_SOURCES`; there is no third possibility.

### Model selection

Benchmarked 2026-09-04 on both saved cases, two runs each, through an identical
retrieval stack (one engine; only the LLM varied):

| model | valid JSON | clinical rubric | median latency | actions |
|---|---|---|---|---|
| **Qwen3-8B-4bit** | **4/4** | **0.875** | 345 s | 3.5 |
| Qwen3-14B-4bit | 4/4 | 0.75 | 600 s | 3.5 |
| Llama-3.2-3B-4bit | 2/4 | 0.25 | 114 s | 1.0 |

The 3B is **disqualified, not merely weaker**: it emitted unparseable JSON on
both shocked-ACS runs, and averaged one action, one drug and one differential.
The 14B is *slower and worse* than the 8B — bigger did not help here. Peak RSS is
4.4 GB with the full retrieval stack loaded, on a 16 GB machine.

#### Regression run, 2026-09-11

Both saved cases re-run twice against the rebuilt 8,973-chunk index, to confirm
the corpus expansion and the new guardrails changed nothing clinical:

| case | level | time to treatment | disposition | latency |
|---|---|---|---|---|
| chest pain, pain 7, ST elevation | **2** EMERGENCY (RED) | under 10 minutes | ADMIT_WARD | 156 s / 169 s |
| allergic rhinitis, pain 0 | **5** ROUTINE (GREEN) | under 90 minutes | DISCHARGE_WITH_FOLLOW_UP | 136 s / 128 s |

Identical on every clinical field across both runs; only red-flag wording
varied. Three things worth reading off it:

- The chest-pain model answer was **Level 3** both times, escalated to 2 by
  `_enforce_mts_level` on the ST-elevation field. Pain 7 is Level 3 on the MTS
  grid, so this is the corrected pain band being reasoned from, and the ECG
  modifier doing the work.
- `time_to_treatment` on the routine case reads **under 90 minutes**. It is now
  derived from the final level; the schema default it would otherwise have shown
  is "under 30 minutes".
- **No `HTA_REPORT` or `PATIENT_FLOW_POLICY` chunk reached either report.** The
  two adjunct documents did not crowd the clinical slots, which was the stated
  worry when they were proposed.

The two latency figures per case are the two runs. The higher chest-pain figure
is **contended, not a regression** - test suites were running on the same GPU.
The uncontended rhinitis figure, 128 s, is faster than the 154 s recorded on
2026-09-10 against a smaller index.

Qwen3 emits a `<think>` reasoning trace by default. `_build_prompt` disables it:
for Mode A the trace is discarded and a 4000-token budget can be spent without
ever reaching the closing brace.

### Drug guardrails

`_ground_prescriber_categories` verifies **provenance, not indication**. Measured
consequence: the 3B recommended snake antivenom for dengue, and Qwen3-8B
recommended **Artesunate — an antimalarial — for dengue shock**, with a genuine
FUKKM category `B` that passed the provenance check. A larger model did not fix
this, so two further checks run in code:

- **`_check_drug_indications`** — FUKKM is a catalogue of every drug for every
  condition, so presence in it proves nothing about this patient. A drug is
  marked `indication_supported=False` unless it is named in a clinical guideline
  chunk retrieved *for this presentation*. The formulary and the CPGs name drugs
  differently ("Sodium Chloride Injection 0.9%" vs "normal saline"), so a curated
  synonym table bridges them — deliberately auditable, because a fuzzy matcher
  here would trade a caught false negative for a silent false positive on a drug.
- **`_check_dose_completeness`** — "administer intravenous fluids" is not an
  instruction for a 20 kg shocked child when the retrieved chunk says 5-7 ml/kg.
  Weight-based orders lacking a per-kg figure are flagged. Runs only when the
  weight is known, since otherwise the gap is a missing input, not a model fault.

Both populate `drug_indication_warning` / `dose_completeness_warning` and append
to `evidence_gaps`. Neither edits the recommendation — they mark it.

### Live progress (`app/progress.py`, `GET /api/v1/progress/{job_id}`)

A case takes 3-4 minutes. Pass an `X-Job-Id` header to `POST /api/v1/triage` and
poll that id for real progress. Retrieval and prefill are **exact** - mlx-lm
reports prompt tokens processed against the total. Decode is **estimated** and
labelled as such, because the model's output length is unknown in advance; the
curve is linear to a calibrated typical length then asymptotic, so it keeps
advancing for any output length and only `finish()` ever reports 100%.

A hard clamp was tried first and measured badly: a shocked-ACS answer ran to
1,144 tokens against a 520-token estimate and the bar sat frozen at 96% for a
minute, which reads as a hang.

### Contraindication checking (`app/contraindications.py`)

On a shocked inferior STEMI with V4R elevation and sildenafil taken 12 hours
earlier, the model recommended **sublingual GTN**. Both contraindications were
stated in the intake *and* present in the retrieved guideline text - retrieval
worked, nothing read it. The recommendation also arrived as an **immediate
action**, not a drug entry, so a check that inspected only
`drug_recommendations` would have missed it.

`_check_contraindications` tests every action, investigation and drug against the
patient's own picture. Derived observations are spelled out in words
("hypotension", "bradycardia", "paediatric patient") so a rule fires on the
**observation**, not on whether the model happened to use that word - the GTN
patient's hypotension was in the vitals and never in the model's prose.

24 rules span the loaded categories: nitrate/PDE5 and nitrate/RV-infarct,
beta-blockade in shock or airway disease, antithrombotics in active bleeding,
thrombolysis when haemorrhage is possible, NSAIDs and aspirin in dengue,
metformin in renal impairment or shock, teratogens in pregnancy, aspirin and
codeine in children, sedation masking a head-injury GCS, plus any drug the intake
records an allergy to. Findings **annotate, never delete** - silently removing a
recommendation would hide the model failure instead of exposing it.

Both `ABSOLUTE` and `CAUTION` findings render **above** the clinical sections in
the on-screen card and the printed report. An absolute contraindication buried in
"evidence gaps" at section 10 is not a safety control.

### Completeness, disposition and citation checks

Three further checks, all in code, all reported inside the single **Safety
checks** block so the report stays compact enough to peer-review:

- **`completeness.py`** - per-presentation required elements for 11 presentations
  (ACS, dengue, stroke, ICH, asthma/COPD, diabetic emergency, heart failure, head
  injury, TB, hip fracture, neonatal jaundice). Elements can be conditional, so
  volume loading is required only for an RV infarct and atropine only for a
  bradycardia actually present in the vitals. **These checks never generate
  clinical content** - a checklist that invented the missing dose would be a
  second unreliable reasoner rather than a check on the first. Measured on the
  shocked STEMI: correctly flags the missing P2Y12, RV volume loading and
  atropine, and stays silent on aspirin, ECG and reperfusion which were present.
- **`_enforce_disposition`** - the model twice sent an MTS 1 cardiogenic-shock
  STEMI to "ED observation" with no referral. MTS 1 floors at critical care, MTS 2
  at admission. `ADMIT_ICU_HDU` and `RESUSCITATION_BAY` are the **same tier**:
  ranking ICU below the resuscitation bay made the guardrail "correct" an ICU
  answer to an ED bay, which a unit test caught.
- **`_check_citations`** - flags recommendations carrying no `[S#]` and citations
  naming a source that is not in the retrieved set. Reported as **one line**; the
  citations table is deliberately left compact.

### Formulary grounding is a database lookup, not a search (`app/formulary.py`)

Prescriber categories used to be matched only against FUKKM chunks that
embedding search happened to retrieve. Measured on a shocked STEMI, that search
returned a **pneumococcal vaccine, secukinumab, rituximab and fluconazole**;
clopidogrel ranked 16th and acetylsalicylic acid was not in the top 25. FUKKM is
~1,700 identically-formatted rows, so the drug name - the only discriminating
signal - is a sliver of each chunk. It is now queried as the database it is.

Three faults, all measured:

1. **Wrong tool** - semantic search over a structured catalogue.
2. **Vocabulary** - FUKKM files aspirin as *Acetylsalicylic Acid* and GTN as
   *Glyceryl Trinitrate*, so both read "CATEGORY UNVERIFIED" while in the file.
3. **Category varies by formulation** - glyceryl trinitrate is `C` sublingual,
   `A, A/KK` as an injection, `B` as an aerosol; paracetamol is `A` IV and `C+`
   as a syrup. A single category per drug NAME is wrong; the route decides.

Matching is scored, not first-hit, and route-aware. Near-misses caught in
testing, each worse than the original bug: `Atropine` resolved to **eye drops**
rather than the IV injection, `Paracetamol` oral to a **paediatric syrup**, and
`Normal Saline` to **0.45%** - hypotonic, not the resuscitation fluid. A minimum
match score means a weak hit yields "unverified" rather than a confidently wrong
category, and where the route is unknown and formulations disagree BOTH
categories are reported. Provenance is the **FUKKM listing number**, which
identifies the row in the published book.

### Dose grounding (`app/dosing.py`)

Prescriber category is a lookup; the DOSE used to be the model's own text. Three
layers now close that:

1. **The stated dose selects the formulation.** "Aspirin 300mg" resolves to the
   *300mg Soluble Tablet*, not the *150mg Dispersible*. Without this a dose check
   flags a correct ACS loading dose - the fastest way to teach a clinician to
   ignore dose warnings.
2. **The authoritative dose is quoted VERBATIM** under each drug: the FUKKM
   dosage field with its listing number, and the CPG dose sentence with its page.
   The reviewer reads the source, not the model's paraphrase.
3. **Numeric cross-check** against every sibling formulation AND the CPG, with
   four honest verdicts:

   | verdict | meaning |
   |---|---|
   | `VERIFIED` | inside a range quoted by FUKKM or the CPG |
   | `EXCEEDS_MAXIMUM` | breaches an explicit stated ceiling - the only alarm |
   | `DIFFERS_FROM_SOURCE` | outside the formulary's general dose; an
     indication-specific loading dose may legitimately differ |
   | `NOT_COMPARABLE` | "titrate to effect", incompatible units, or no number |

**Cohort segmentation is a safety requirement, not a refinement.** A single FUKKM
field carries both regimens ("ADULT: 5 to 20 mg ... CHILD: 0.1 - 0.2 mg/kg"), and
matching against the adult sentence VERIFIED **morphine 20 mg for a 20 kg child** -
five times the paediatric maximum. Doses for a paediatric patient are now checked
only against paediatric text, and a per-kg source range is expanded using the
patient's weight. A cohort word only starts a section if that section states a
dose, so "use in children under 16 is not recommended" is read as the caution it
is rather than a paediatric regimen.

**What this guarantees:** no dose appears without a named source or an explicit
mark that it could not be verified. **What it does not:** that the dose is right
for this patient - indication, renal function and interactions remain clinical
judgements the CPG expresses in prose.

### Why indication support is deliberately conservative

Only a drug named in the **excerpt cited for this patient** counts as supported.
Presence elsewhere in the guideline does not: dabigatran is genuinely in the ACS
CPG (for AF patients needing a DOAC alongside DAPT) and passed an earlier
"anywhere in the guideline" check on a shocked STEMI.

Prevalence cannot separate the two either, and this was measured rather than
assumed: **paracetamol appears in just 1 chunk (0.72%) of the Dengue CPG - the
same band as dabigatran in ACS (0.74%)** - and is the correct antipyretic. A
frequency threshold would flag a correct drug and train clinicians to ignore the
warning. So the two cases share a verdict but get different wording:
"NOT FOUND IN THE GUIDELINE" versus "NOT VERIFIED FOR THIS PRESENTATION".

What is **not** guaranteed, and must be checked by the reading clinician:

- **The triage level is now enforced, not trusted.** The 3B model under-triaged
  both live cases to Level 3; `_enforce_mts_level` escalated both to Level 1 on
  age-specific hypotension, with the reason printed in the rationale. What is
  still not enforced is escalation on *narrative* red flags (active haemorrhage,
  chest pain) when the vitals look normal — the model alone decides those.
- **Drug choice.** A grounded category does not make the drug appropriate. Before
  the retrieval fix, this case produced *Haemato Polyvalent Snake Antivenom* —
  category `B`, correctly traced to a genuinely retrieved chunk, and clinically
  absurd. The guardrail verifies provenance, not indication. Retrieval now offers
  the right shelf (paracetamol, ORS, colloid, electrolyte solution), but the
  model still recommends too few drugs and may omit the crystalloid bolus that
  the retrieved algorithm calls for.
- **Doses.** `paediatric_dose` has been observed wrong (an infusion rate where a
  weight-based bolus was required).
- **`evidence_gaps`.** Observed claiming vitals were "not supplied" when all
  nine were.

The report's cross-check sign-off block exists to record that this verification
actually happened.

## Scope and safety

Clinical decision-support prototype for registered clinicians. Not a medical
device, not validated, not submitted to any regulatory authority, not for
patient-facing use. Every recommendation must be checked against the cited
source document, at the cited page, before acting — the reports carry
machine-authored warnings naming what could not be verified, and those warnings
are part of the output, not decoration. `drug_recommendations` in particular is
**not usable prescribing output**; see §Why indication support is deliberately
conservative.

The MTS 2022 file in `raw_pdfs/` is a third-party re-host with page watermarks
(stripped at ingest); **replace it with the official KKM copy before any real
use.** It is the document that sets every triage level.

Full statement in [`NOTICE`](NOTICE).

## Licence and corpus

The source code is Apache-2.0 — see [`LICENSE`](LICENSE).

That licence covers **only the code**. It does not and cannot cover the MOH
guidelines, protocols and formulary data the software reads; those remain the
property of their owners under their own terms, and are deliberately not
distributed in this repository. [`CORPUS.md`](CORPUS.md) documents every
document, its owner, its reproduction terms and where to obtain it, and
[`NOTICE`](NOTICE) carries the copyright acknowledgement those terms require.

A clone will therefore **not run as-is**: there is no corpus to retrieve from
until `backend/data/raw_pdfs/` is populated. Each empty data folder carries a
README with the rebuild procedure.
