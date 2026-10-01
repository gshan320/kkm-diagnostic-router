/** Standalone, printable HTML report for a Mode A triage result.
 *
 *  The output is one self-contained file — no external CSS, fonts or scripts —
 *  so it opens identically on any machine, prints to A4 without reflowing, and
 *  can be saved as PDF straight from the browser's print dialog. Everything the
 *  screen card shows is here, plus the patient intake that produced it, so the
 *  clinician cross-checking it sees the inputs and not just the conclusions. */

import type {
  CompletenessGap,
  DrugRecommendation,
  Contraindication,
  RetrievedSource,
  TriageColour,
  TriageRequest,
  TriageResponse,
  Vitals,
} from "./types";
import { sourcePdfUrl } from "./api";
import { labelFor } from "./modifiers";

/** Older API builds did not echo the intake back; the report degrades to a
 *  stated gap rather than printing a patient block that is silently empty. */
const intakeOf = (result: TriageResponse): TriageRequest | null =>
  result.request ?? null;

const DISPOSITION_LABEL: Record<string, string> = {
  RESUSCITATION_BAY: "Resuscitation bay",
  ADMIT_ICU_HDU: "Admit — ICU / HDU",
  ADMIT_WARD: "Admit — ward",
  ED_OBSERVATION: "ED observation",
  REFER_SPECIALIST: "Refer to specialist",
  DISCHARGE_WITH_FOLLOW_UP: "Discharge with follow-up",
};

const DOC_TYPE_LABEL: Record<string, string> = {
  TRIAGE_PROTOCOL: "MTS 2022",
  CPG_FULL: "CPG",
  CPG_QUICK_REFERENCE: "Quick Reference",
  PAEDIATRIC_PROTOCOL: "Paediatric Protocol",
  DRUG_FORMULARY: "FUKKM",
  // The label is where a reviewer checks provenance, so it carries the caveat.
  // Neither of these is a clinical guideline: see config.CLINICAL_DOC_TYPES.
  HTA_REPORT: "HTA report — not a guideline",
  PATIENT_FLOW_POLICY: "State policy — disposition only",
  // Indexed only where no KKM document covers the condition (config.py).
  EXTERNAL_GUIDELINE: "Non-KKM guideline — international",
};

const COLOUR_HEX: Record<TriageColour, { bg: string; fg: string }> = {
  RED: { bg: "#c81e1e", fg: "#ffffff" },
  YELLOW: { bg: "#c27803", fg: "#ffffff" },
  GREEN: { bg: "#046c4e", fg: "#ffffff" },
};

const VITAL_ROWS: Array<{ key: keyof Vitals; label: string; unit: string }> = [
  { key: "systolic_bp", label: "Systolic BP", unit: "mmHg" },
  { key: "diastolic_bp", label: "Diastolic BP", unit: "mmHg" },
  { key: "heart_rate", label: "Heart rate", unit: "bpm" },
  { key: "respiratory_rate", label: "Respiratory rate", unit: "/min" },
  { key: "temperature", label: "Temperature", unit: "°C" },
  { key: "spo2", label: "SpO₂", unit: "%" },
  { key: "gcs", label: "GCS", unit: "/15" },
  { key: "capillary_blood_glucose", label: "Capillary blood glucose", unit: "mmol/L" },
  { key: "pain_score", label: "Pain score", unit: "/10" },
];

const GENDER_LABEL: Record<string, string> = {
  male: "Male",
  female: "Female",
  other: "Other",
  unknown: "Not stated",
};

/** HTML-escape. Every value below is model or user text, so nothing is trusted. */
/** Machine sentinels must never reach a printed clinical report. Rendering is
 *  the last line of defence: any code path that forgets to humanise a value is
 *  caught here rather than printing NOT_IN_RETRIEVED_SOURCES at a clinician. */
const MACHINE_LABELS: Array<[RegExp, string]> = [
  [/NOT_IN_RETRIEVED_SOURCES/g, "not listed in the formulary"],
  [/EXCEEDS_MAXIMUM/g, "exceeds stated maximum"],
  [/DIFFERS_FROM_SOURCE/g, "differs from source"],
  [/NOT_COMPARABLE/g, "not comparable"],
  // Deliberately NOT matching bare "VERIFIED": it is a real English word and a
  // badge label this report prints on purpose.
];

function humanise(text: string): string {
  return MACHINE_LABELS.reduce((acc, [re, word]) => acc.replace(re, word), text);
}

function esc(value: unknown): string {
  if (value == null) return "";
  return humanise(String(value))
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Empty cells read as "—", never as blank — a blank cell is ambiguous on paper. */
const dash = (value: unknown): string => {
  const text = esc(value).trim();
  return text === "" ? "—" : text;
};

/** The prompt template shows source_id as "[S1]", so the model emits it with
 *  or without brackets. Normalise before we add our own. */
const sid = (id: string): string => String(id ?? "").trim().replace(/^\[|\]$/g, "");

const cite = (id: string): string => {
  const clean = sid(id);
  return clean ? ` <span class="cite">[${esc(clean)}]</span>` : "";
};

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

/** Local time, written out in full — the recipient may be in another timezone. */
function stamp(date: Date): string {
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`
  );
}

function fileStamp(date: Date): string {
  return (
    `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-` +
    `${pad(date.getHours())}${pad(date.getMinutes())}`
  );
}

function section(title: string, body: string): string {
  return `<section class="sec">
  <h2>${esc(title)}</h2>
  ${body}
</section>`;
}

const NONE = `<p class="none">None identified from the retrieved sources.</p>`;

/** The optional intake, printed only where it was actually recorded.
 *
 *  Rows are OMITTED, never rendered as "no" or "none". A blank field means
 *  the question was not put, and printing it as a negative would turn an
 *  unasked question into a documented finding - the same reason the vitals
 *  table marks a missing value "not measured" rather than showing a normal
 *  range. A section with nothing recorded disappears entirely, so a minimal
 *  intake still prints a short report.
 */
function modifierBlock(request: TriageRequest): string {
  const rows: Array<[string, string]> = [];
  const one = (label: string, value: string | null | undefined) => {
    const text = labelFor(value);
    if (text) rows.push([label, text]);
  };
  const many = (label: string, values: string[] | undefined) => {
    const text = (values ?? []).map(labelFor).filter(Boolean).join(" · ");
    if (text) rows.push([label, text]);
  };
  const free = (label: string, value: string | null | undefined) => {
    if (value && value.trim()) rows.push([label, value.trim()]);
  };

  one("General appearance", request.appearance);
  one("Breathing and speech", request.breathing);
  one("Perfusion", request.perfusion);
  one("Active bleeding", request.bleeding);
  many("Paediatric danger signs", request.paediatric_signs);
  many("12-lead ECG", request.ecg_findings);
  one("Fever history", request.fever_history);
  one("Mode of arrival", request.arrival_mode);
  free("Mechanism of injury", request.trauma_mechanism);
  free("Drug allergies", request.allergies);
  free("Current medications", request.current_medications);
  many("Comorbidities", request.comorbidities);
  if (request.egfr != null) {
    rows.push(["eGFR", `${esc(request.egfr)} ml/min/1.73m²`]);
  }

  // Pregnancy prints only when a real answer was given, or when the answer is
  // missing on a patient it could matter for. "Not established" is itself a
  // finding worth showing; on a male or a child it is noise, so it is absent.
  const childbearing =
    (request.gender === "female" || request.gender === "other") &&
    request.age >= 11 &&
    request.age <= 55;
  const pregnancy = request.pregnancy ?? "unknown";
  if (pregnancy === "pregnant") {
    rows.push([
      "Pregnancy",
      request.gestation_weeks != null
        ? `Pregnant, ${esc(request.gestation_weeks)} weeks gestation`
        : "Pregnant, gestation not stated",
    ]);
  } else if (pregnancy === "not_pregnant") {
    rows.push(["Pregnancy", "Excluded"]);
  } else if (childbearing) {
    rows.push([
      "Pregnancy",
      "<strong>Not established</strong> — confirm before any teratogen or ionising imaging",
    ]);
  }
  if (request.breastfeeding) rows.push(["Breastfeeding", "Yes"]);

  many("Infection / exposure risk", request.exposure_risk);
  many("Behavioural risk", request.behavioural_risk);
  one("Assessed at", request.facility);

  if (rows.length === 0) return "";
  return `<table class="kv stacked">
  <colgroup><col style="width:26%"><col style="width:74%"></colgroup>
  <tbody>
    ${rows
      .map(([k, v]) => `<tr><th>${esc(k)}</th><td class="wrap">${v}</td></tr>`)
      .join("\n    ")}
  </tbody>
</table>
<p class="note">Optional intake. A row that is absent was <em>not assessed</em>,
which is not the same as normal.</p>`;
}

function patientBlock(request: TriageRequest | null, derivedOnset = ""): string {
  if (!request) {
    return section(
      "1 · Patient",
      `<p class="none">The patient input for this result is not available in this
       session (the report was generated after a page reload). Re-run the
       assessment to capture it.</p>`,
    );
  }

  const identity = `<table class="kv">
  <colgroup><col style="width:26%"><col style="width:24%"><col style="width:26%"><col style="width:24%"></colgroup>
  <tbody>
    <tr>
      <th>Age</th><td>${dash(`${request.age} years`)}</td>
      <th>Gender</th><td>${dash(GENDER_LABEL[request.gender] ?? request.gender)}</td>
    </tr>
    <tr>
      <th>Weight</th><td>${request.weight_kg != null ? `${esc(request.weight_kg)} kg` : "not recorded"}</td>
      <th>Onset</th><td>${request.onset ? esc(request.onset) : dash(derivedOnset)}</td>
    </tr>
  </tbody>
</table>`;

  const vitals = `<table class="grid vitals">
  <colgroup><col style="width:46%"><col style="width:27%"><col style="width:27%"></colgroup>
  <thead><tr><th>Parameter</th><th class="num">Value</th><th>Unit</th></tr></thead>
  <tbody>
    ${VITAL_ROWS.map((row) => {
      const value = request.vitals[row.key];
      const missing = value == null;
      return `<tr${missing ? ' class="missing"' : ""}>
      <th scope="row">${esc(row.label)}</th>
      <td class="num">${missing ? "not measured" : esc(value)}</td>
      <td>${missing ? "" : esc(row.unit)}</td>
    </tr>`;
    }).join("\n    ")}
  </tbody>
</table>
<p class="note">A parameter marked <em>not measured</em> was left blank at intake.
It is reported as a gap, never as normal.</p>`;

  const prose = `<table class="kv stacked">
  <colgroup><col style="width:26%"><col style="width:74%"></colgroup>
  <tbody>
    <tr><th>Presenting complaint</th><td class="wrap">${dash(request.complaint)}</td></tr>
    <tr><th>Relevant history</th><td class="wrap">${dash(request.history)}</td></tr>
  </tbody>
</table>`;

  return section("1 · Patient", `${identity}\n${vitals}\n${prose}\n${modifierBlock(request)}`);
}

function triageBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const colour = COLOUR_HEX[result.triage_colour];
  return `<section class="sec keep">
  <h2>2 · Triage outcome</h2>
  <div class="triage">
    <div class="badge" style="background:${colour.bg};color:${colour.fg}">
      <span class="badge-top">MTS 2022</span>
      <span class="badge-level">${esc(d.mts_triage_level)}</span>
      <span class="badge-label">${esc(d.mts_triage_label.replace(/_/g, " "))}</span>
      <span class="badge-colour">${esc(result.triage_colour)}</span>
    </div>
    <div class="triage-body">
      <p class="ttt">Time to treatment: <strong>${dash(d.time_to_treatment)}</strong></p>
      <p class="wrap">${dash(d.triage_rationale)}</p>
      ${d.triage_provisional_note ? `<p class="warn"><strong>${esc(d.triage_provisional_note)}</strong></p>` : ""}
      ${d.reassessment ? `<p class="reassess">${esc(d.reassessment)}</p>` : ""}
    </div>
  </div>
</section>`;
}

function cautionsBlock(result: TriageResponse): string {
  const items = result.diagnostic.source_cautions ?? [];
  if (items.length === 0) return "";
  return section(
    `Cautions from the guidelines (${items.length})`,
    `<table class="grid">
  <colgroup><col style="width:38%"><col style="width:62%"></colgroup>
  <thead><tr><th>Caution</th><th>Source, verbatim</th></tr></thead>
  <tbody>
    ${items
      .map(
        (c) => `<tr class="flag">
      <th scope="row" class="wrap">${dash(c.label)}${cite(c.source_id)}${
        c.external ? ` <span class="muted">(non-KKM source)</span>` : ""
      }</th>
      <td class="wrap">&ldquo;${dash(c.quote)}&rdquo;</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

function redFlagsBlock(result: TriageResponse): string {
  const flags = result.diagnostic.red_flags;
  if (flags.length === 0) return section("3 · Red flags", NONE);
  return section(
    `3 · Red flags (${flags.length})`,
    `<table class="grid">
  <colgroup><col style="width:38%"><col style="width:62%"></colgroup>
  <thead><tr><th>Flag</th><th>Why it matters</th></tr></thead>
  <tbody>
    ${flags
      .map(
        (flag) => `<tr class="flag">
      <th scope="row" class="wrap">${dash(flag.flag)}${cite(flag.source_id)}${
        flag.origin === "source" ? ` <span class="muted">(added from source)</span>` : ""
      }</th>
      <td class="wrap">${dash(flag.why_it_matters)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

function diagnosisBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const p = d.primary_diagnosis;
  const differentials =
    d.differential_diagnoses.length === 0
      ? ""
      : `<h3>Differentials to exclude</h3>
<table class="grid">
  <colgroup><col style="width:38%"><col style="width:62%"></colgroup>
  <thead><tr><th>Condition</th><th>Discriminating feature</th></tr></thead>
  <tbody>
    ${d.differential_diagnoses
      .map(
        (diff) => `<tr>
      <th scope="row" class="wrap">${dash(diff.condition)}</th>
      <td class="wrap">${dash(diff.discriminating_feature)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`;

  return section(
    "4 · Primary diagnosis",
    `<p class="dx">${dash(p.condition)}
   <span class="chip">${esc(p.confidence)} CONFIDENCE</span></p>
<p class="wrap">${dash(p.reasoning)}</p>
<p class="note">Governing guideline: ${dash(p.supporting_cpg.replace(/^\[|\]$/g, ""))}</p>
${differentials}`,
  );
}

function actionsBlock(result: TriageResponse): string {
  const actions = [...result.diagnostic.immediate_actions].sort(
    (a, b) => a.sequence - b.sequence,
  );
  if (actions.length === 0) return section("5 · Immediate actions", NONE);
  return section(
    `5 · Immediate actions (${actions.length}, in order)`,
    `<table class="grid">
  <colgroup><col style="width:6%"><col style="width:62%"><col style="width:32%"></colgroup>
  <thead><tr><th class="num">#</th><th>Action</th><th>Timeframe</th></tr></thead>
  <tbody>
    ${actions
      .map(
        (action) => `<tr>
      <td class="num seq">${esc(action.sequence)}</td>
      <td class="wrap">${dash(action.action)}${cite(action.source_id)}${
        action.origin === "source" ? ` <span class="muted">(added from source)</span>` : ""
      }${
        action.source_quote ? `<br><span class="muted">&ldquo;${esc(action.source_quote)}&rdquo; [${esc(action.source_where)}]</span>` : ""
      }</td>
      <td class="wrap when">${dash(action.timeframe)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

function investigationsBlock(result: TriageResponse): string {
  const items = result.diagnostic.investigations;
  if (items.length === 0) return section("6 · Investigations", NONE);
  return section(
    `6 · Investigations (${items.length})`,
    `<table class="grid">
  <colgroup><col style="width:30%"><col style="width:18%"><col style="width:52%"></colgroup>
  <thead><tr><th>Test</th><th>Urgency</th><th>Rationale</th></tr></thead>
  <tbody>
    ${items
      .map(
        (item) => `<tr>
      <th scope="row" class="wrap">${dash(item.test)}${item.source_id ? cite(item.source_id) : ""}${
        item.origin === "source" ? ` <span class="muted">(added from source)</span>` : ""
      }${
        item.source_quote ? `<br><span class="muted">&ldquo;${esc(item.source_quote)}&rdquo; [${esc(item.source_where)}]</span>` : ""
      }</th>
      <td class="wrap when">${dash(item.urgency)}</td>
      <td class="wrap">${dash(item.rationale)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

const DOSE_LABEL: Record<string, [string, string]> = {
  VERIFIED: ["dose-ok", "DOSE VERIFIED"],
  EXCEEDS_MAXIMUM: ["dose-bad", "EXCEEDS STATED MAXIMUM"],
  DIFFERS_FROM_SOURCE: ["dose-warn", "DIFFERS FROM SOURCE"],
  NOT_COMPARABLE: ["dose-grey", "DOSE NOT VERIFIED"],
};

/** The authoritative dose, quoted verbatim. The figure in the table above is
 *  the model's; these lines are copied from the formulary and the guideline, so
 *  a reviewer can check the prescription against the source without leaving the
 *  page. Kept to at most two short lines per drug to stay reviewable. */
function doseSources(drug: DrugRecommendation): string {
  const verdict = drug.dose_verdict ?? "";
  const [cls, label] = DOSE_LABEL[verdict] ?? ["", ""];
  const badge = label
    ? `<span class="chip ${cls}">${esc(label)}</span>`
    : "";
  const lines: string[] = [];
  if (drug.dose_source_fukkm?.trim())
    lines.push(`<div class="src"><em>Formulary:</em> ${esc(drug.dose_source_fukkm)}</div>`);
  if (drug.dose_source_cpg?.trim())
    lines.push(`<div class="src"><em>Guideline:</em> ${esc(drug.dose_source_cpg)}</div>`);
  if (verdict && verdict !== "VERIFIED" && drug.dose_verdict_detail?.trim())
    lines.push(`<div class="src"><em>Check:</em> ${esc(drug.dose_verdict_detail)}</div>`);
  if (!badge && lines.length === 0) return "";
  return `<div class="dosebox">${badge}${lines.join("")}</div>`;
}

/** The indication gate's verdict and the sentence it rests on. CONDITIONAL is
 *  spelled out, because "indicated for a differential" is easy to misread as
 *  "indicated". */
const INDICATION_LABEL: Record<string, string> = {
  SUPPORTED: "Supported for the working diagnosis",
  SYMPTOMATIC: "For a symptom this patient has",
  CONDITIONAL: "Only if the differential it treats is confirmed",
};
function indicationLine(drug: TriageResponse["diagnostic"]["drug_recommendations"][number]): string {
  const status = drug.indication_status ?? "";
  if (!status) return "";
  const label = INDICATION_LABEL[status] ?? status;
  return drug.indication_basis ? `${label} — ${drug.indication_basis}` : label;
}

function drugsBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const warning = d.prescriber_category_warning
    ? `<div class="warn danger">
  <strong>Prescriber-category warning</strong>
  <p class="wrap">${esc(d.prescriber_category_warning)}</p>
</div>`
    : "";

  if (d.drug_recommendations.length === 0) {
    return section("7 · KKM drug dosing", `${warning}${NONE}`);
  }

  const cards = d.drug_recommendations
    .map((drug) => {
      const unverified = drug.prescriber_category === "NOT_IN_RETRIEVED_SOURCES";
      const rows: Array<[string, string]> = [
        ["Indication", drug.indication],
        ["Indication check", indicationLine(drug)],
        ["Adult dose", drug.adult_dose],
        ["Paediatric dose", drug.paediatric_dose],
        ["Route", drug.route],
        ["Frequency", drug.frequency],
        ["Duration", drug.duration],
        ["Cautions", drug.cautions],
      ];
      return `<div class="drug keep">
  <p class="drug-name">${dash(drug.drug_name)}
    <span class="chip ${unverified ? "chip-grey" : "chip-cat"}">${
      unverified ? "NOT IN FORMULARY" : `FUKKM CAT. ${esc(drug.prescriber_category)}`
    }</span>${cite(drug.source_id)}</p>
  <table class="kv stacked">
    <colgroup><col style="width:22%"><col style="width:78%"></colgroup>
    <tbody>
      ${rows
        .filter(([, value]) => String(value ?? "").trim() !== "")
        .map(
          ([label, value]) =>
            `<tr><th>${esc(label)}</th><td class="wrap">${esc(value)}</td></tr>`,
        )
        .join("\n      ")}
    </tbody>
  </table>
  ${doseSources(drug)}
  <p class="note">${dash(drug.prescriber_category_meaning)}</p>
</div>`;
    })
    .join("\n");

  return section(
    `7 · KKM drug dosing (${d.drug_recommendations.length})`,
    `${warning}${cards}`,
  );
}

function dispositionBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const referral =
    d.referral_required && d.referral_to
      ? `<tr><th>Referral</th><td class="wrap">${esc(d.referral_to)}</td></tr>`
      : `<tr><th>Referral</th><td>${d.referral_required ? "Required — destination not stated" : "Not required"}</td></tr>`;
  return section(
    "8 · Disposition",
    `<table class="kv stacked">
  <colgroup><col style="width:26%"><col style="width:74%"></colgroup>
  <tbody>
    <tr><th>Destination</th><td><strong>${dash(
      DISPOSITION_LABEL[d.disposition] ?? d.disposition,
    )}</strong></td></tr>
    ${referral}
    <tr><th>Justification</th><td class="wrap">${dash(d.disposition_justification)}</td></tr>
  </tbody>
</table>`,
  );
}

function vitalsInterpretationBlock(result: TriageResponse): string {
  const items = result.diagnostic.vitals_interpretation;
  if (items.length === 0) return "";
  return section(
    `9 · Vital-sign interpretation (${items.length})`,
    `<table class="grid">
  <colgroup><col style="width:24%"><col style="width:14%"><col style="width:42%"><col style="width:20%"></colgroup>
  <thead><tr><th>Parameter</th><th class="num">Value</th><th>Interpretation</th><th>MTS level triggered</th></tr></thead>
  <tbody>
    ${items
      .map(
        (vital) => `<tr>
      <th scope="row">${dash(vital.parameter)}</th>
      <td class="num">${dash(vital.value)}</td>
      <td class="wrap">${dash(vital.interpretation)}</td>
      <td>${dash(vital.mts_level_triggered)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

function gapsBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const gaps = d.evidence_gaps?.trim()
    ? `<p class="wrap">${esc(d.evidence_gaps)}</p>`
    : `<p class="none">None stated.</p>`;
  const corpus =
    result.corpus_warnings.length > 0
      ? `<h3>Corpus caveats — read before acting</h3>
<ul class="bullets">
  ${result.corpus_warnings.map((warning) => `<li class="wrap">${esc(warning)}</li>`).join("\n  ")}
</ul>`
      : "";
  return section("10 · Evidence gaps and caveats", `${gaps}${corpus}`);
}

/** The document name, linked to the PDF at the cited page when the backend
 *  serves it. The link works while the API is running on this machine; the
 *  printed name is always there. */
function citationDocument(
  result: TriageResponse,
  citation: TriageResponse["diagnostic"]["citations"][number],
): string {
  const id = sid(citation.source_id);
  const src = result.sources.find((s) => s.source_id === id);
  if (src?.url) {
    return `<a href="${esc(src.url)}" target="_blank" rel="noopener">${dash(citation.document)}</a>`;
  }
  if (!src?.filename || src.doc_type === "DRUG_FORMULARY") return dash(citation.document);
  const href = sourcePdfUrl(src.filename, src.page_number ?? citation.page);
  return `<a href="${esc(href)}" target="_blank" rel="noopener">${dash(citation.document)}</a>`;
}

/** Verified links to consult for what the indexed guidelines do not cover. */
function referencesBlock(result: TriageResponse): string {
  const refs = result.diagnostic.external_references ?? [];
  if (refs.length === 0) return "";
  return `<section class="sec keep">
  <h2>References to consult</h2>
  <p class="note">Not used to generate this report. Chosen by the system from a verified list of MOH pages, never written by the model.</p>
  <ul>
    ${refs
      .map(
        (r) =>
          `<li><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a> — ${esc(r.reason)}${
            r.checked ? ` <span class="note">(link checked ${esc(r.checked)})</span>` : ""
          }</li>`,
      )
      .join("\n    ")}
  </ul>
</section>`;
}

/** One muted paragraph: why these sources and not others. Compact on purpose -
 *  it answers "why was X left out?" without becoming a second citation table. */
function selectionBlock(result: TriageResponse): string {
  const notes = (result.retrieval_notes ?? []).filter((n) => n.trim());
  if (notes.length === 0) return "";
  return `<details class="appendix"><summary>How the sources were chosen</summary>
  <ul class="bullets note">${notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
</details>`;
}

function citationsBlock(result: TriageResponse): string {
  const items = result.diagnostic.citations;
  if (items.length === 0) return section("11 · Citations", NONE);
  return section(
    `11 · Citations (${items.length})`,
    `<table class="grid">
  <colgroup><col style="width:10%"><col style="width:58%"><col style="width:16%"><col style="width:16%"></colgroup>
  <thead><tr><th>ID</th><th>Document</th><th>Page</th><th>Edition year</th></tr></thead>
  <tbody>
    ${items
      .map(
        (citation) => `<tr>
      <td class="mono">${sid(citation.source_id) ? `[${esc(sid(citation.source_id))}]` : "—"}</td>
      <td class="wrap">${citationDocument(result, citation)}</td>
      <td>${dash(citation.page)}</td>
      <td>${dash(citation.edition_year)}</td>
    </tr>`,
      )
      .join("\n    ")}
  </tbody>
</table>`,
  );
}

/** One or two lines naming the documents the findings actually came from —
 *  enough provenance to back the report without reprinting the corpus. */
function provenanceLine(sources: RetrievedSource[]): string {
  if (sources.length === 0) {
    return `<p class="prov"><strong>Evidence base:</strong> no source passages
    were retrieved for this assessment — treat every statement above as
    ungrounded and verify it directly against the guideline.</p>`;
  }

  // Retrieval order is relevance order, so first mention wins and the list
  // reads most-relevant-first. Titles are deduplicated across chunks.
  const seen = new Set<string>();
  const documents: string[] = [];
  for (const source of sources) {
    const title = (source.cpg_title || source.filename || "").trim();
    if (!title || seen.has(title)) continue;
    seen.add(title);
    const year = source.edition_year?.trim();
    // Comma, not parentheses — most titles already end in "(3rd Edition)".
    documents.push(`${esc(title)}${year ? `, ${esc(year)}` : ", edition year not stated"}`);
  }

  const shown = documents.slice(0, 4);
  const rest = documents.length - shown.length;
  const tail = rest > 0 ? ` and ${rest} further document${rest === 1 ? "" : "s"}` : "";

  return `<p class="prov"><strong>Evidence base:</strong> drawn from
  ${esc(sources.length)} passage${sources.length === 1 ? "" : "s"} retrieved from
  ${shown.join("; ")}${tail}. Section 11 gives the page for each [S#] cited above.</p>`;
}

function signOff(): string {
  return `<section class="sec keep">
  <h2>Cross-check sign-off</h2>
  <table class="grid signoff">
    <colgroup><col style="width:25%"><col style="width:25%"><col style="width:25%"><col style="width:25%"></colgroup>
    <thead><tr><th>Name</th><th>Designation / MMC no.</th><th>Date &amp; time</th><th>Signature</th></tr></thead>
    <tbody>
      <tr><td></td><td></td><td></td><td></td></tr>
      <tr><td></td><td></td><td></td><td></td></tr>
    </tbody>
  </table>
</section>`;
}

const STYLES = `
:root { color-scheme: light; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; padding: 24px 28px 48px; background: #f1f5f9; color: #0f172a;
  font: 11pt/1.45 "Helvetica Neue", Helvetica, Arial, "Segoe UI", sans-serif;
}
.sheet {
  max-width: 210mm; margin: 0 auto; padding: 16mm 14mm; background: #fff;
  border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(15,23,42,.12);
}
h1 { margin: 0 0 2px; font-size: 15pt; letter-spacing: -.01em; }
h2 {
  margin: 0 0 8px; padding-bottom: 4px; font-size: 10.5pt;
  text-transform: uppercase; letter-spacing: .06em; color: #334155;
  border-bottom: 1.5px solid #0f172a;
}
h3 { margin: 14px 0 6px; font-size: 10pt; color: #334155; }
p { margin: 0 0 6px; }
.sub { margin: 0; font-size: 9.5pt; color: #475569; }
.sec { margin-top: 20px; }
.sec:first-of-type { margin-top: 16px; }

/* Meta strip */
.meta {
  margin-top: 12px; width: 100%; border-collapse: collapse;
  font-size: 8.5pt; color: #334155;
}
.meta th, .meta td {
  border: 1px solid #cbd5e1; padding: 4px 7px; text-align: left; vertical-align: top;
}
.meta th { width: 15%; background: #f1f5f9; font-weight: 600; white-space: nowrap; }

/* Tables — fixed layout is what keeps columns aligned across renderers. */
table.grid, table.kv {
  width: 100%; border-collapse: collapse; table-layout: fixed;
  margin: 0 0 8px; font-size: 9.5pt;
}
table.grid th, table.grid td, table.kv th, table.kv td {
  border: 1px solid #cbd5e1; padding: 5px 7px; text-align: left;
  vertical-align: top; overflow-wrap: anywhere;
}
table.grid thead th {
  background: #e2e8f0; font-weight: 700; font-size: 8.5pt;
  text-transform: uppercase; letter-spacing: .04em;
}
table.grid tbody th, table.kv th { background: #f8fafc; font-weight: 600; }
table.grid tbody tr:nth-child(even) td { background: #fbfcfd; }
/* Specific enough to beat the table.grid th / table.grid td rules above. */
table.grid .num, table.kv .num {
  text-align: right; font-variant-numeric: tabular-nums;
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
}
.wrap { white-space: pre-wrap; }
.muted { color: #64748b; font-size: 0.86em; }
.dosebox { margin: 6px 0 2px; padding: 7px 9px; background: #f8fafc; border-left: 3px solid #cbd5e1; border-radius: 4px; }
.dosebox .src { font-size: 0.86em; color: #334155; margin-top: 4px; white-space: pre-wrap; }
.dosebox .src em { color: #64748b; font-style: normal; font-weight: 600; }
.chip.dose-ok   { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.chip.dose-bad  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.chip.dose-warn { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.chip.dose-grey { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
.reason + .reason { margin-top: 6px; padding-top: 6px; border-top: 1px dotted #cbd5e1; }
/* Severity badges for the automated safety checks. Red is reserved for an
   ABSOLUTE contraindication so it cannot be confused with a soft caution. */
.sev {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.78em; font-weight: 700; letter-spacing: 0.02em; white-space: nowrap;
}
.sev-abs { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.sev-cau { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.alert {
  background: #fef2f2; border: 1px solid #fca5a5; border-left: 4px solid #dc2626;
  color: #7f1d1d; padding: 10px 12px; border-radius: 6px; margin: 0 0 10px;
}
.mono { font-family: "SFMono-Regular", Menlo, Consolas, monospace; }
.missing td, .missing th { color: #94a3b8; font-style: italic; }
.vitals .num { font-weight: 600; }
.when { color: #0369a1; font-weight: 600; }
.seq { font-weight: 700; }
tr.flag th { border-left: 3px solid #c81e1e; }

/* Triage badge */
.triage { display: flex; gap: 14px; align-items: stretch; }
.badge {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-width: 34mm; padding: 8px 12px; border-radius: 6px; text-align: center;
}
.badge-top { font-size: 7pt; font-weight: 700; letter-spacing: .18em; opacity: .9; }
.badge-level { font-size: 30pt; font-weight: 800; line-height: 1; }
.badge-label { font-size: 8.5pt; font-weight: 700; letter-spacing: .04em; }
.badge-colour { font-size: 7pt; letter-spacing: .18em; opacity: .9; }
.triage-body { flex: 1; }
.ttt { font-size: 11pt; }
.reassess { font-size: 9pt; color: #475569; margin-top: 4px; }

/* Diagnosis + chips */
.dx { font-size: 12pt; font-weight: 700; }
.chip {
  display: inline-block; margin-left: 6px; padding: 2px 6px; border-radius: 3px;
  border: 1px solid #94a3b8; background: #f1f5f9; color: #334155;
  font-size: 7.5pt; font-weight: 700; letter-spacing: .04em; vertical-align: middle;
}
.chip-cat { border-color: #4338ca; background: #e0e7ff; color: #3730a3; }
.chip-grey { border-color: #94a3b8; background: #e2e8f0; color: #475569; }
.cite {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 8pt; color: #0369a1; white-space: nowrap;
}

/* Drugs */
.drug { margin: 0 0 10px; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 4px; }
.drug-name { font-size: 10.5pt; font-weight: 700; margin-bottom: 6px; }
.drug table.kv { margin-bottom: 4px; }

/* Banners */
.warn {
  margin: 0 0 10px; padding: 8px 10px; border-radius: 4px;
  border: 1px solid #f59e0b; background: #fffbeb; font-size: 9.5pt;
}
.warn.danger { border-color: #c81e1e; background: #fef2f2; }
.warn strong { display: block; margin-bottom: 3px; font-size: 9pt; text-transform: uppercase; letter-spacing: .05em; }
.banner {
  margin-top: 12px; padding: 7px 10px; border: 1.5px solid #c81e1e;
  border-radius: 4px; background: #fef2f2; color: #7f1d1d;
  font-size: 8.5pt; font-weight: 600;
}
.note { margin: 4px 0 0; font-size: 8.5pt; color: #64748b; }
.appendix { margin-top: 14px; font-size: 8.5pt; color: #475569; }
.appendix summary { cursor: pointer; font-weight: 600; color: #334155; }
.appendix[open] summary { margin-bottom: 6px; }
.none { margin: 0; font-size: 9.5pt; color: #64748b; font-style: italic; }
.bullets { margin: 4px 0 0; padding-left: 18px; font-size: 9.5pt; }
.bullets li { margin-bottom: 3px; }

/* Provenance line */
.prov {
  margin: 0 0 6px; padding: 6px 8px; border-left: 3px solid #334155;
  background: #f8fafc; font-size: 8.5pt; color: #334155;
}

/* Sign-off + footer */
.signoff td { height: 13mm; }
.footer {
  margin-top: 22px; padding-top: 8px; border-top: 1px solid #cbd5e1;
  font-size: 8pt; line-height: 1.5; color: #475569;
}

/* Print */
@page { size: A4 portrait; margin: 12mm 10mm 14mm; }
@media print {
  body { padding: 0; background: #fff; }
  .sheet { max-width: none; margin: 0; padding: 0; border: 0; box-shadow: none; }
  .no-print { display: none !important; }
  h2 { break-after: avoid; page-break-after: avoid; }
  .sec { break-inside: auto; }
  .keep, .drug, .prov, .triage, .warn, .banner, .alert { break-inside: avoid; page-break-inside: avoid; }
  tr, .signoff { break-inside: avoid; page-break-inside: avoid; }
  thead { display: table-header-group; }
  tfoot { display: table-footer-group; }
}
`;

/** Safety objections, placed BEFORE the actions they object to.
 *  An absolute contraindication buried in "evidence gaps" at section 10 is not
 *  a safety control - the GTN-with-sildenafil recommendation that prompted this
 *  block would have been read and acted on long before a clinician reached it. */
function contraindicationsBlock(result: TriageResponse): string {
  const d = result.diagnostic;
  const items = d.contraindications ?? [];
  const extra = [
    d.parse_warning,
    d.consistency_warning,
    d.avoid_warning,
    d.differential_warning,
    d.second_pass_note,
    d.complication_note,
    d.citation_alignment_note,
    d.dose_completeness_warning,
    d.drug_indication_warning,
    d.urgency_warning,
    d.redirection_warning,
  ].filter((w): w is string => Boolean(w && w.trim()));
  if (
    items.length === 0 &&
    extra.length === 0 &&
    (d.completeness_gaps?.length ?? 0) === 0 &&
    (d.knowledge_gaps?.length ?? 0) === 0 &&
    !d.citation_warning?.trim()
  )
    return "";

  // Grouped by the flagged recommendation, not by rule. Ungrouped, one GTN
  // order flagged on three grounds reads as "6 contraindications" and repeats
  // the same reason text six times - which is how alarm fatigue is built.
  const groups = new Map<string, Contraindication[]>();
  for (const c of items) {
    const key = `${c.item}||${c.where}`;
    groups.set(key, [...(groups.get(key) ?? []), c]);
  }
  const worst = (g: Contraindication[]) =>
    g.some((c) => c.severity === "ABSOLUTE") ? "ABSOLUTE" : "CAUTION";
  const ordered = [...groups.values()].sort((a, b) =>
    worst(a) === worst(b) ? 0 : worst(a) === "ABSOLUTE" ? -1 : 1,
  );

  const row = (g: Contraindication[]) => {
    const sev = worst(g);
    const first = g[0];
    const count = g.length > 1 ? ` <span class="muted">×${g.length}</span>` : "";
    return `
      <tr>
        <td><span class="sev sev-${sev === "ABSOLUTE" ? "abs" : "cau"}">${esc(sev)}</span>${count}</td>
        <td><strong>${esc(first.item)}</strong><br><span class="muted">${esc(first.where)}</span></td>
        <td class="wrap">${g.map((c) => esc(c.trigger)).join("; ")}</td>
        <td class="wrap">${g.map((c) => `<div class="reason">${esc(c.reason)}</div>`).join("")}</td>
      </tr>`;
  };

  const table = ordered.length
    ? `<table class="grid">
      <thead><tr><th>Severity</th><th>Recommendation</th><th>Triggered by (from the intake)</th><th>Why</th></tr></thead>
      <tbody>${ordered.map(row).join("\n")}</tbody>
    </table>`
    : "";
  const nAbs = ordered.filter((g) => worst(g) === "ABSOLUTE").length;
  const nReasons = items.filter((c) => c.severity === "ABSOLUTE").length;
  const lead = nAbs
    ? `<p class="wrap alert"><strong>${nAbs} recommendation${nAbs > 1 ? "s" : ""} the model proposed ${
        nAbs > 1 ? "are" : "is"
      } ABSOLUTELY CONTRAINDICATED for this patient${
        nReasons > nAbs ? ` (${nReasons} independent reasons)` : ""
      } and ${nAbs > 1 ? "were" : "was"} removed from the plan. Listed here so the error stays visible.</strong></p>`
    : "";
  const notes = extra.map((w) => `<p class="wrap">${esc(w)}</p>`).join("\n  ");

  // Omissions, kept to one compact list. The reviewer needs to see WHAT is
  // missing and WHICH guideline settles it - not a second table.
  const gaps: CompletenessGap[] = d.completeness_gaps ?? [];
  const missing = gaps.length
    ? `<p class="wrap alert"><strong>Not addressed in this report:</strong>
      ${gaps.map((g) => esc(g.element)).join(" · ")}.
      <span class="muted">Read ${esc(gaps[0].guideline)} before acting.</span></p>
      ${gaps
        .filter((g) => g.quote)
        .map(
          (g) =>
            `<p class="wrap"><strong>${esc(g.element)} — the guideline says:</strong> &ldquo;${esc(
              g.quote,
            )}&rdquo; <span class="muted">[${esc(g.quote_source)}]</span></p>`,
        )
        .join("\n      ")}`
    : "";

  // Where KKM is silent: one line per element, then the links once.
  const kgaps = d.knowledge_gaps ?? [];
  const kLinks = kgaps.flatMap((g) => g.references ?? [])
    .filter((r, i, all) => all.findIndex((x) => x.url === r.url) === i);
  const silent = kgaps.length
    ? `${kgaps.map((g) => `<p class="wrap"><strong>Not covered by KKM:</strong> ${esc(g.statement)}</p>`).join("\n  ")}
      ${kLinks.length ? `<p class="wrap muted">Consult: ${kLinks
        .map((r) => `<a href="${esc(r.url)}">${esc(r.title)}</a>`)
        .join(" · ")}</p>` : ""}`
    : "";

  // Citation integrity is one line by design: the citation table stays compact.
  const cites = d.citation_warning?.trim()
    ? `<p class="wrap muted">${esc(d.citation_warning)}</p>`
    : "";

  return `<details class="appendix"><summary>Safety checks — full detail</summary>
  ${lead}${missing}${silent}${table}${notes}${cites}
</details>`;
}

/** The glance version of the safety checks, right under the triage outcome.
 *  Each line is short on purpose; the reasoning, quotes and per-check notes are
 *  in the collapsed "full detail" block at the end. An ABSOLUTE contraindication
 *  stays up here: the item was already taken out of the plan, but a reader must
 *  not learn that only at the bottom of the page. */
function safetySummary(result: TriageResponse): string {
  const d = result.diagnostic;
  const lines: string[] = [];
  const abs = new Set((d.contraindications ?? []).filter((c) => c.severity === "ABSOLUTE").map((c) => c.item));
  const cau = new Set((d.contraindications ?? []).filter((c) => c.severity !== "ABSOLUTE").map((c) => c.item));
  if (abs.size)
    lines.push(`<p class="wrap alert"><strong>Removed as absolutely contraindicated:</strong> ${[...abs]
      .map(esc).join(" · ")}</p>`);
  if (cau.size)
    lines.push(`<p class="wrap"><strong>Caution:</strong> ${[...cau].map(esc).join(" · ")}</p>`);
  const gaps = d.completeness_gaps ?? [];
  if (gaps.length)
    lines.push(`<p class="wrap"><strong>Not addressed:</strong> ${gaps.map((g) => esc(g.element)).join(" · ")}</p>`);
  const kg = d.knowledge_gaps ?? [];
  if (kg.length)
    lines.push(`<p class="wrap"><strong>Not covered by KKM:</strong> ${kg.length} element${kg.length > 1 ? "s" : ""} — see full detail</p>`);
  const corrections = [
    d.parse_warning, d.consistency_warning, d.avoid_warning, d.differential_warning,
    d.second_pass_note, d.complication_note, d.citation_alignment_note,
    d.dose_completeness_warning, d.drug_indication_warning, d.urgency_warning, d.redirection_warning,
  ].filter((w) => w && w.trim()).length;
  if (corrections)
    lines.push(`<p class="wrap muted">${corrections} automated correction${corrections > 1 ? "s" : ""} applied — see full detail at the end.</p>`);
  if (d.citation_warning?.trim()) lines.push(`<p class="wrap muted">${esc(d.citation_warning)}</p>`);
  return lines.length ? section("Safety checks", lines.join("\n  ")) : "";
}

/** The backend audit row this report was recorded as. A report with no audit
 *  row says so rather than leaving the cell blank, which would read as "not
 *  applicable" instead of "not recorded". */
function auditCell(result: TriageResponse): string {
  const id = result.provenance?.audit_id;
  return id ? esc(id) : "not recorded";
}

/** Code and corpus fingerprints: enough to tell whether two reports came from
 *  the same system. "uncommitted" marks a build that git alone cannot recreate. */
function buildCell(result: TriageResponse): string {
  const p = result.provenance;
  if (!p || !p.code_fingerprint) return "—";
  const dirty = p.git_dirty ? " (uncommitted)" : "";
  return `code ${esc(p.code_fingerprint)}${dirty} · corpus ${esc(p.corpus_fingerprint)}`;
}

/** Section numbers follow the order the sections are printed in, so a
 *  reorder never leaves "4 · " above "3 · ". Each block's own number is
 *  dropped and the sequence re-applied; unnumbered blocks get none. */
function numbered(blocks: string[]): string {
  let n = 0;
  return blocks
    .filter((b) => b.trim())
    .map((b) => b.replace(/<h2>(\d+ · )?/, (m, num) => (num ? `<h2>${++n} · ` : m)))
    .join("\n  ");
}

export function buildReportHtml(
  result: TriageResponse,
  generatedAt: Date = new Date(),
): string {
  const d = result.diagnostic;
  const request = intakeOf(result);
  const reportId = `KKM-A-${fileStamp(generatedAt)}-L${d.mts_triage_level}`;
  const title = `KKM Triage Report ${reportId}`;

  const patientLine = request
    ? `${request.age} y · ${GENDER_LABEL[request.gender] ?? request.gender}${
        request.weight_kg != null ? ` · ${request.weight_kg} kg` : ""
      }${request.pregnancy === "pregnant" ? " · pregnant" : ""}`
    : "not captured";

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>${STYLES}</style>
</head>
<body>
<div class="sheet">
  <header>
    <h1>KKM Diagnostic Router — Clinical Triage Report</h1>
    <p class="sub">Mode A · Retrieval-augmented decision support over Malaysian MOH
    CPGs, Quick References, Paediatric Protocols, MTS 2022 and the FUKKM formulary.</p>
    <table class="meta">
      <tbody>
        <tr>
          <th>Report ID</th><td class="mono">${esc(reportId)}</td>
          <th>Generated</th><td>${esc(stamp(generatedAt))}</td>
        </tr>
        <tr>
          <th>Patient</th><td>${esc(patientLine)}</td>
          <th>Reasoning model</th><td class="mono">${dash(result.model)}</td>
        </tr>
        <tr>
          <th>Outcome</th>
          <td>MTS ${esc(d.mts_triage_level)} · ${esc(result.triage_colour)} · ${esc(
            d.mts_triage_label.replace(/_/g, " "),
          )}</td>
          <th>Latency</th><td>${esc((result.latency_ms / 1000).toFixed(1))} s</td>
        </tr>
        <tr>
          <th>Audit ID</th><td class="mono">${auditCell(result)}</td>
          <th>Build</th><td class="mono">${buildCell(result)}</td>
        </tr>
      </tbody>
    </table>
  </header>

  ${numbered([
    triageBlock(result),
    safetySummary(result),
    diagnosisBlock(result),
    redFlagsBlock(result),
    actionsBlock(result),
    investigationsBlock(result),
    drugsBlock(result),
    dispositionBlock(result),
    cautionsBlock(result),
    patientBlock(request, d.onset_derived ?? ""),
    vitalsInterpretationBlock(result),
    gapsBlock(result),
    citationsBlock(result),
    referencesBlock(result),
  ])}
  ${contraindicationsBlock(result)}
  ${selectionBlock(result)}
  <p class="banner">Decision-support output for registered clinicians. Not a
  medical device, not clinically validated, not for patient-facing use. Every
  recommendation must be verified against the cited source before acting.</p>
  ${signOff()}

  <footer class="footer">
    ${provenanceLine(result.sources)}
    <p>${dash(result.disclaimer)}</p>
    <p>Report ${esc(reportId)} · generated ${esc(stamp(generatedAt))} ·
    model ${dash(result.model)}. This file is self-contained: open it in any
    browser to read, or print it (Ctrl/Cmd&nbsp;+&nbsp;P) to paper or PDF.</p>
  </footer>
</div>
</body>
</html>`;
}

export function reportFilename(
  result: TriageResponse,
  generatedAt: Date = new Date(),
): string {
  const request = intakeOf(result);
  const age = request ? `${String(request.age).replace(".", "-")}y` : "unknown-age";
  const condition = (result.diagnostic.primary_diagnosis.condition || "triage")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
  return `KKM-triage_MTS${result.diagnostic.mts_triage_level}_${age}_${condition}_${fileStamp(
    generatedAt,
  )}.html`;
}

/** Saves the report as a single .html file next to the browser's downloads. */
export function downloadTriageReport(result: TriageResponse): void {
  const now = new Date();
  const blob = new Blob([buildReportHtml(result, now)], {
    type: "text/html;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = reportFilename(result, now);
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoke late — Safari reads the blob after the click returns.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Opens the browser print dialog on the same report, via an offscreen iframe
 *  so no popup blocker is involved. "Save as PDF" there gives an archival PDF. */
export function printTriageReport(result: TriageResponse): void {
  const html = buildReportHtml(result, new Date());
  const frame = document.createElement("iframe");
  frame.setAttribute("aria-hidden", "true");
  frame.style.cssText =
    "position:fixed;right:0;bottom:0;width:0;height:0;border:0;visibility:hidden";
  frame.srcdoc = html;
  frame.onload = () => {
    const view = frame.contentWindow;
    if (!view) return;
    view.focus();
    view.print();
    // Chrome's print dialog is modal to the tab; the frame can go once it closes.
    setTimeout(() => frame.remove(), 60_000);
  };
  document.body.appendChild(frame);
}
