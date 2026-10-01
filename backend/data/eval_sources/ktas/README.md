# KTAS triage accuracy dataset (evaluation only - not ingested)

Source: Moon SH, Shim JL, Park KS, Park CS. *Triage accuracy and causes of
mistriage using the Korean Triage and Acuity Scale.* PLoS ONE 2019;14(9):e0216972.
https://doi.org/10.1371/journal.pone.0216972
Data: PLOS figshare, https://plos.figshare.com/articles/dataset/Triage_accuracy_and_causes_of_mistriage_using_the_Korean_Triage_and_Acuity_Scale/9779267
Licence: PLOS supporting information is published under CC BY 4.0 - cite the paper
above in any derived scorecard.

| file here | original name | contents |
|---|---|---|
| `ktas_moon2019_records.xlsx` | `pone.0216972.s001.xlsx` | 1,267 adult ED records, 24 columns |
| `ktas_moon2019_codebook.xlsx` | `pone.0216972.s002.xlsx` | variable coding (e.g. Sex 1 = female, 2 = male) |

Columns used by the harness: `Age`, `Sex`, `Chief_complain`, `Mental`, `NRS_pain`,
`SBP`, `DBP`, `HR`, `RR`, `BT`, `Saturation`, `KTAS_RN` (triage nurse),
`KTAS_expert` (reference level), `Diagnosis in ED`, `Disposition`, `mistriage`.

**KTAS is not MTS.** Both are 5-level scales with 1 as most urgent, but their
discriminators differ. Use `KTAS_expert` as a band check (within one level, and
the direction of any disagreement - under-triage is the safety metric), never as
an exact MTS answer key.
