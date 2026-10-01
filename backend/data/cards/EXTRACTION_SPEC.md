# Condition-card extraction spec

You turn one or more **packets** (filtered text of a Malaysian Ministry of Health
guideline) into **condition cards**: for every condition the document gives
management guidance for, the sentences that answer each clinical question, quoted
exactly. Code verifies every quote afterwards (`python -m app.cards_build --verify`).
An item whose quote is not found verbatim in the cited chunk is thrown away, so
copy exactly.

## Input format

```
# DOC <title> (<year>) | <doc_type> | population: adult|paediatric|all [| part k of n]
## p<page> [r<ref>]          <- one chunk; "r123" is its reference id
<sentence>
§ <heading>                  <- a heading kept for context (never quote a "§" line)
```

Sentences between chunks were removed by a filter (study results, background,
front matter). Text is sometimes lowercased or has spaced punctuation
("( acs )", "0. 9 %"). Copy it as it appears.

## Output

Write ONE JSON file per packet: `backend/data/cards/raw/<packet file stem>.json`
(for example, the packet `management-of-gout.txt` becomes `raw/management-of-gout.json`).
Write it with the Write tool; valid JSON only, UTF-8.

```json
{
  "packet": "<packet file stem>",
  "document": "<title from the DOC line>",
  "conditions": [
    {
      "name": "Acute gout flare",
      "aliases": ["acute gouty arthritis", "gout attack", "gout flare"],
      "parent": "Gout",
      "population": "adult",
      "acuity": "urgent",
      "items": [
        {
          "element": "treatment",
          "label": "Colchicine for acute flare",
          "terms": ["colchicine"],
          "when": [],
          "unless": ["severe renal impairment"],
          "core": true,
          "setting": "ed",
          "population": "",
          "ref": "r4411",
          "quote": "<exact text copied from the chunk r4411>"
        }
      ]
    }
  ]
}
```

### Condition fields
- **name**: the condition, as a clinician names it: "Acute exacerbation of asthma",
  "Diabetic ketoacidosis", "Paracetamol poisoning", "Burns", "Bradycardia", or a
  presentation the document handles as one ("Undifferentiated chest pain"). Use the
  English medical name with British spelling (haemorrhage, oedema, paediatric).
- **aliases**: abbreviations, synonyms, US spellings and common phrasings a
  clinician would type as a diagnosis ("DKA", "STEMI", "ST-elevation myocardial
  infarction"). Never put a broader or different condition here (no "shock" on
  "Septic shock", no "chest pain" on "STEMI").
- **parent**: the broader condition this is a form of, if any ("Acute coronary
  syndrome" for "STEMI"; "Dengue infection" for "Severe dengue"). Else omit.
- **population**: adult | paediatric | obstetric | all. Paediatric Protocols are
  paediatric; the Perinatal Care Manual and pregnancy documents are obstetric.
- **acuity**: emergency (threat to life within hours) | urgent (needs same-day care)
  | routine | chronic (long-term management).

Granularity: one card per condition that has its own management guidance. A state
or complication handled as a section of a larger condition (cardiogenic shock in
ACS, hyperkalaemia in rhabdomyolysis) stays on the parent card as items with
`when`; give it its own card only when the document gives it its own chapter.
Skip conditions with no management content. For a split document (part k of n),
extract what is in your part only; cards with the same name are merged later.

### Item fields
- **element**, one of:
  - `definition`: what defines or diagnoses the condition (criteria, thresholds).
  - `red_flag`: findings that mark severity, danger or need to escalate (warning
    signs, danger signs, "features of severe ...").
  - `complication`: dangerous consequences to anticipate or look for.
  - `investigation`: a test to do (bloods, ECG, imaging, bedside test).
  - `treatment`: an action or drug to give, including doses, fluids, procedures
    and resuscitation steps.
  - `avoid`: do-not-give / contraindicated / use-with-caution statements specific
    to this condition.
  - `admission`: when to admit (ward, HDU, ICU).
  - `referral`: when to refer, consult or transfer (specialist, higher centre).
  - `discharge`: when a patient may go home, and the follow-up to arrange.
  - `monitoring`: what to monitor and how often.
- **label**: at most 8 words, the item as a clinician would write it in a plan
  ("IV hydrocortisone 200 mg", "Serum potassium and ECG", "Avoid NSAIDs",
  "Admit if warning signs"). For `avoid`, start with "Avoid" or "Caution".
- **terms**: 1-5 lowercase words or short phrases whose presence in a clinician's
  written plan shows the item was addressed: the drug, test, procedure or finding
  name and its common abbreviations (["potassium", "k+", "serum electrolytes"],
  ["adrenaline", "epinephrine"], ["ecg", "electrocardiogram"]). **At least one term
  must appear in the quote.** No generic words (patient, treatment, management,
  dose, assess). For `avoid`, the terms are the thing to avoid (["nsaid", "ibuprofen",
  "diclofenac"]).
- **when**: the patient state in which the item applies, as short lowercase phrases
  a triage note or vital-sign reading would contain: "hypotension", "shock",
  "hypoxia", "tachycardia", "bradycardia", "fever", "reduced consciousness",
  "seizure", "pregnant", "paediatric", "renal impairment", "hyperkalaemia",
  "st elevation", "warning signs", "bleeding". Empty list when the item applies to
  every patient with the condition. Put the condition of an `admission` or
  `referral` criterion here ("admit if hypotensive" → when ["hypotension"]).
- **unless**: states in which the item must NOT be applied, using the same
  vocabulary ("in stable patients" → unless ["shock", "hypotension"]; "if no
  pulmonary oedema" → unless ["pulmonary oedema"]). Empty list if none.
- **core**: true when every patient with this condition (meeting `when`) should get
  it in the first hours of care, i.e. the standard of care a reviewer would mark as
  missing. False for second-line, optional, specialist-only or long-term items.
- **setting**: ed (emergency department / first hours) | inpatient | outpatient
  (clinic, long-term, secondary prevention) | any.
- **population**: only when this item is narrower than the card
  (a paediatric dose in an adult card): adult | paediatric | obstetric. Else "".
- **ref**: the `[r...]` id of the chunk the quote comes from.
- **quote**: 1-2 sentences copied **exactly** from that one chunk: same words,
  same order, same spelling and odd spacing. You may start and end mid-sentence,
  but never skip words inside the quote, never join text from two chunks, never
  include a "§" heading, and keep it under 400 characters. If a dose table row is
  the only source, quote the row text as it appears.

## What to extract, and what not to
- Only guidance for handling a patient: what to look for, do, give, avoid, and when
  to admit, refer or discharge. Not epidemiology, not study results, not
  pathophysiology unless it is a definition or a complication statement.
- Prefer the document's recommendation sentences and algorithm/table text
  ("should", "is recommended", "give", "refer", "admit", dose lines) over narrative.
- Every item's quote must actually say the thing its label claims. A list of
  admission criteria answers `admission`, not `treatment`. A sentence saying a
  test is "not warranted" is not an `investigation`; if it is clinically
  important, it is an `avoid` item.
- Keep the patient state the sentence depends on. "In patients with cardiogenic
  shock, fluids ... only if no pulmonary congestion" is `treatment` with when
  ["shock"] and unless ["pulmonary oedema", "pulmonary congestion"]. "For stable
  patients, pharmaco-invasive strategy ..." gets unless ["shock", "hypotension"].
- Aim for completeness over brevity: an emergency condition typically has 10-40
  items. Cover every element the document addresses for it. One item per
  distinct action (aspirin and ticagrelor are two items even if one sentence names
  both; both may quote that same sentence).
- Non-KKM documents (the DOC line says "NOT KKM"): extract the same way; code
  labels them.

Before finishing each packet, re-check a sample of your quotes against the packet
text character by character. When all your packets are written, reply with one
line per packet: `<packet>: <conditions> conditions, <items> items`.
