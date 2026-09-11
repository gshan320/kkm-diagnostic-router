"use client";

import { useEffect, useState } from "react";
import type {
  Facility, ProgressResponse, TriageRequest, Vitals,
} from "@/lib/types";
import {
  APPEARANCE, ARRIVAL, BEHAVIOURAL, BLEEDING, BREATHING, COMORBIDITIES, ECG,
  EXPOSURE, FACILITY, FEVER_HISTORY, PAEDIATRIC_SIGNS, PERFUSION, PREGNANCY,
  type Option,
} from "@/lib/modifiers";

const EXAMPLES: Array<{ label: string; value: TriageRequest }> = [
  {
    label: "Adult chest pain, shocked",
    value: {
      age: 58,
      gender: "male",
      weight_kg: 78,
      vitals: {
        systolic_bp: 86,
        diastolic_bp: 54,
        heart_rate: 124,
        respiratory_rate: 28,
        temperature: 36.8,
        spo2: 91,
        gcs: 15,
        pain_score: 9,
      },
      complaint:
        "Central crushing chest pain for 45 minutes, radiating to the left arm, diaphoretic and clammy.",
      history: "Type 2 diabetes on metformin, hypertension, 20 pack-year smoker.",
      onset: "45 minutes ago, sudden onset at rest",
      appearance: "appears_unwell",
      perfusion: "pale_cyanosed_cold_peripheries",
      ecg_findings: ["st_elevations_or_depressions"],
      comorbidities: ["diabetes"],
      allergies: "NKDA",
      current_medications: "Metformin 1 g BD, amlodipine 5 mg OD",
      arrival_mode: "ambulance",
      pregnancy: "unknown",
    },
  },
  {
    label: "Child, day 5 dengue",
    value: {
      age: 6,
      gender: "female",
      weight_kg: 20,
      vitals: {
        systolic_bp: 82,
        diastolic_bp: 60,
        heart_rate: 148,
        respiratory_rate: 34,
        temperature: 37.2,
        spo2: 96,
        gcs: 14,
        capillary_blood_glucose: 4.8,
      },
      complaint:
        "Day 5 of fever, now afebrile with abdominal pain, persistent vomiting, cold peripheries and reduced urine output.",
      history: "Sibling had dengue two weeks ago. No known allergies.",
      onset: "day 5 of illness, fever settled yesterday",
      appearance: "appears_unwell",
      perfusion: "crt_over_2_seconds",
      fever_history: "fever_reported_before_arrival",
      paediatric_signs: ["pale_mucous_membranes_sole_or_palm"],
      allergies: "NKDA",
      arrival_mode: "walk_in",
      pregnancy: "unknown",
    },
  },
];

const EMPTY: TriageRequest = {
  age: 0,
  gender: "unknown",
  weight_kg: null,
  vitals: {},
  complaint: "",
  history: "",
  appearance: null,
  breathing: null,
  perfusion: null,
  bleeding: null,
  fever_history: null,
  ecg_findings: [],
  paediatric_signs: [],
  onset: "",
  trauma_mechanism: "",
  arrival_mode: null,
  allergies: "",
  current_medications: "",
  comorbidities: [],
  egfr: null,
  pregnancy: "unknown",
  gestation_weeks: null,
  breastfeeding: false,
  exposure_risk: [],
  behavioural_risk: [],
  facility: null,
};

const VITAL_FIELDS: Array<{
  key: keyof Vitals;
  label: string;
  unit: string;
  step?: string;
}> = [
  { key: "systolic_bp", label: "Systolic BP", unit: "mmHg" },
  { key: "diastolic_bp", label: "Diastolic BP", unit: "mmHg" },
  { key: "heart_rate", label: "Heart rate", unit: "bpm" },
  { key: "respiratory_rate", label: "Resp. rate", unit: "/min" },
  { key: "temperature", label: "Temperature", unit: "°C", step: "0.1" },
  { key: "spo2", label: "SpO₂", unit: "%" },
  { key: "gcs", label: "GCS", unit: "3–15" },
  { key: "capillary_blood_glucose", label: "CBG", unit: "mmol/L", step: "0.1" },
  { key: "pain_score", label: "Pain score", unit: "0–10" },
];

const fieldClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100";
const labelClass =
  "mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400";
const legendClass =
  "mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400";

const FACILITY_KEY = "kkm.facility";

const STAGE_LABEL: Record<string, string> = {
  queued: "Starting",
  retrieval: "Retrieving guidelines",
  prefill: "Reading context",
  decode: "Writing assessment",
  checks: "Safety checks",
  done: "Complete",
  error: "Failed",
  unknown: "Working",
};

/** Progress ring driven by real backend work. The decode phase is marked as an
 *  estimate because the model's output length is not knowable in advance -
 *  showing an unqualified percentage there would imply a precision we do not
 *  have. */
function ProgressRing({ progress }: { progress?: ProgressResponse | null }) {
  const pct = Math.max(0, Math.min(100, progress?.percent ?? 0));
  const r = 16;
  const c = 2 * Math.PI * r;
  return (
    <span className="flex items-center gap-3" aria-live="polite">
      <span className="relative inline-flex h-11 w-11 shrink-0">
        <svg viewBox="0 0 40 40" className="h-11 w-11 -rotate-90">
          <circle cx="20" cy="20" r={r} fill="none" strokeWidth="4"
            className="stroke-slate-200 dark:stroke-slate-700" />
          <circle cx="20" cy="20" r={r} fill="none" strokeWidth="4"
            strokeLinecap="round" strokeDasharray={c}
            strokeDashoffset={c * (1 - pct / 100)}
            className="stroke-sky-600 transition-[stroke-dashoffset] duration-500 ease-out" />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold tabular-nums text-slate-700 dark:text-slate-200">
          {Math.round(pct)}%
        </span>
      </span>
      <span className="min-w-0">
        <span className="block text-xs font-medium text-slate-700 dark:text-slate-300">
          {STAGE_LABEL[progress?.stage ?? "queued"] ?? "Working"}
          {progress?.estimated && (
            <span className="ml-1 font-normal text-slate-500">(estimated)</span>
          )}
          {progress?.elapsed_s ? (
            <span className="ml-2 font-normal tabular-nums text-slate-500">
              {Math.round(progress.elapsed_s)}s
            </span>
          ) : null}
        </span>
        <span className="block truncate text-xs text-slate-500">
          {progress?.detail ?? "Contacting the reasoning model…"}
        </span>
      </span>
    </span>
  );
}

/** A single-valued modifier. The empty option is always first and always
 *  available: "Not assessed" must stay reachable after a mis-click, because
 *  an unset field asserts nothing while a wrong one asserts something. */
function ModifierSelect<T extends string>({
  label, options, value, onChange, note,
}: {
  label: string;
  options: Option<T>[];
  value: T | null | undefined;
  onChange: (v: T | null) => void;
  note?: string;
}) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      <select
        className={fieldClass}
        value={value ?? ""}
        onChange={(e) => onChange((e.target.value || null) as T | null)}
      >
        <option value="">Not assessed</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
            {o.hint ? ` — MTS ${o.hint}` : ""}
          </option>
        ))}
      </select>
      {note && <p className="mt-1 text-[11px] text-slate-500">{note}</p>}
    </div>
  );
}

/** A multi-valued modifier. Nothing ticked means NOT ASSESSED, which is why
 *  the reassuring options ("Normal ECG", "No fever reported") are values you
 *  choose rather than what a blank field implies. */
function ModifierChips<T extends string>({
  label, options, values, onChange, note,
}: {
  label: string;
  options: Option<T>[];
  values: T[] | undefined;
  onChange: (v: T[]) => void;
  note?: string;
}) {
  const set = values ?? [];
  const toggle = (v: T) =>
    onChange(set.includes(v) ? set.filter((x) => x !== v) : [...set, v]);
  return (
    <div>
      <label className={labelClass}>{label}</label>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => {
          const on = set.includes(o.value);
          return (
            <button
              key={o.value}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(o.value)}
              title={o.hint ? `MTS ${o.hint}` : undefined}
              className={`rounded-full border px-2.5 py-1 text-xs transition ${
                on
                  ? "border-sky-600 bg-sky-600 text-white"
                  : "border-slate-300 text-slate-600 hover:border-sky-500 hover:text-sky-700 dark:border-slate-700 dark:text-slate-400 dark:hover:text-sky-400"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
      {note && <p className="mt-1 text-[11px] text-slate-500">{note}</p>}
    </div>
  );
}

export default function PatientInputForm({
  onSubmit,
  loading,
  progress,
}: {
  onSubmit: (request: TriageRequest) => void;
  loading: boolean;
  progress?: ProgressResponse | null;
}) {
  const [form, setForm] = useState<TriageRequest>(EMPTY);

  // The facility is a property of WHERE THE USER WORKS, not of the patient, so
  // it is remembered rather than re-asked. It reaches the request like any
  // other field; only its persistence differs.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(FACILITY_KEY) as Facility | null;
      if (saved) setForm((prev) => ({ ...prev, facility: saved }));
    } catch {
      /* private mode, blocked storage - the field simply starts unset */
    }
  }, []);

  const setFacility = (facility: Facility | null) => {
    setForm((prev) => ({ ...prev, facility }));
    try {
      if (facility) localStorage.setItem(FACILITY_KEY, facility);
      else localStorage.removeItem(FACILITY_KEY);
    } catch {
      /* not persisting is survivable; not sending it would not be */
    }
  };

  const set = <K extends keyof TriageRequest>(key: K, value: TriageRequest[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const setVital = (key: keyof Vitals, raw: string) =>
    setForm((prev) => ({
      ...prev,
      vitals: { ...prev.vitals, [key]: raw === "" ? null : Number(raw) },
    }));

  const canSubmit = form.complaint.trim().length >= 3 && !loading;
  const isChild = form.age > 0 && form.age < 12;
  const maybePregnant = form.gender === "female" || form.gender === "other";

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (canSubmit) onSubmit(form);
      }}
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Patient assessment
        </h2>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example.label}
              type="button"
              onClick={() => setForm({ ...EMPTY, ...example.value, facility: form.facility })}
              className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-600 transition hover:border-sky-500 hover:text-sky-700 dark:border-slate-700 dark:text-slate-400 dark:hover:text-sky-400"
            >
              {example.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/50">
        <label className="text-xs font-medium text-slate-600 dark:text-slate-400" htmlFor="facility">
          This facility
        </label>
        <select
          id="facility"
          className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
          value={form.facility ?? ""}
          onChange={(e) => setFacility((e.target.value || null) as Facility | null)}
        >
          <option value="">Not set</option>
          {FACILITY.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <span className="text-[11px] text-slate-500">
          Remembered on this device. Shapes what can be recommended here — a
          reperfusion or transfer decision needs it. Never changes the triage level.
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <label className={labelClass} htmlFor="age">
            Age (years)
          </label>
          <input
            id="age"
            className={fieldClass}
            type="number"
            min={0}
            max={130}
            step="0.1"
            required
            value={form.age === 0 ? "" : form.age}
            onChange={(e) =>
              setForm({ ...form, age: e.target.value === "" ? 0 : Number(e.target.value) })
            }
          />
        </div>
        <div>
          <label className={labelClass} htmlFor="gender">
            Gender
          </label>
          <select
            id="gender"
            className={fieldClass}
            value={form.gender}
            onChange={(e) =>
              setForm({ ...form, gender: e.target.value as TriageRequest["gender"] })
            }
          >
            <option value="unknown">Not stated</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="weight">
            Weight (kg)
          </label>
          <input
            id="weight"
            className={fieldClass}
            type="number"
            min={0}
            max={400}
            step="0.1"
            placeholder="for mg/kg"
            value={form.weight_kg ?? ""}
            onChange={(e) =>
              setForm({
                ...form,
                weight_kg: e.target.value === "" ? null : Number(e.target.value),
              })
            }
          />
        </div>
        <div>
          <label className={labelClass} htmlFor="onset">
            Onset / day of illness
          </label>
          <input
            id="onset"
            className={fieldClass}
            type="text"
            maxLength={200}
            placeholder="45 min ago · day 5 of fever"
            value={form.onset ?? ""}
            onChange={(e) => set("onset", e.target.value)}
          />
        </div>
      </div>

      <fieldset className="mt-5">
        <legend className={labelClass}>
          Vital signs — leave blank if not measured (blank is reported as a gap,
          not as normal)
        </legend>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {VITAL_FIELDS.map((vital) => (
            <div key={vital.key}>
              <label className={labelClass} htmlFor={vital.key}>
                {vital.label}{" "}
                <span className="text-slate-400">({vital.unit})</span>
              </label>
              <input
                id={vital.key}
                className={fieldClass}
                type="number"
                step={vital.step ?? "1"}
                value={form.vitals[vital.key] ?? ""}
                onChange={(e) => setVital(vital.key, e.target.value)}
              />
            </div>
          ))}
        </div>
      </fieldset>

      <div className="mt-5 grid gap-3 lg:grid-cols-2">
        <div>
          <label className={labelClass} htmlFor="complaint">
            Presenting complaint <span className="text-rose-600">*</span>
          </label>
          <textarea
            id="complaint"
            className={`${fieldClass} h-24 resize-y`}
            required
            minLength={3}
            placeholder="Onset, character, severity, associated features…"
            value={form.complaint}
            onChange={(e) => setForm({ ...form, complaint: e.target.value })}
          />
        </div>
        <div>
          <label className={labelClass} htmlFor="history">
            Relevant history
          </label>
          <textarea
            id="history"
            className={`${fieldClass} h-24 resize-y`}
            placeholder="Comorbidities, current medications, allergies…"
            value={form.history}
            onChange={(e) => setForm({ ...form, history: e.target.value })}
          />
        </div>
      </div>

      {/* Everything below is optional. MTS 2022 p7 sets the final level from
          four inputs - Primary Triage, Vital Signs, Complaints List and
          Initial Tests - and the fields above are only the middle two. */}
      <details className="mt-5 rounded-lg border border-slate-200 dark:border-slate-800">
        <summary className="cursor-pointer select-none px-4 py-2.5 text-xs font-medium text-slate-700 dark:text-slate-300">
          Optional detail — Primary Triage, ECG, and background
          <span className="ml-2 font-normal text-slate-500">
            all blank-safe; each one narrows the guideline match
          </span>
        </summary>

        <div className="space-y-5 border-t border-slate-200 px-4 py-4 dark:border-slate-800">
          <fieldset>
            <legend className={legendClass}>
              Primary triage — MTS 2022 pp. 4–5
            </legend>
            <div className="grid gap-3 sm:grid-cols-2">
              <ModifierSelect
                label="General appearance"
                options={APPEARANCE}
                value={form.appearance}
                onChange={(v) => set("appearance", v)}
              />
              <ModifierSelect
                label="Breathing and speech"
                options={BREATHING}
                value={form.breathing}
                onChange={(v) => set("breathing", v)}
                note="Catches the compensating asthmatic that SpO₂ alone misses."
              />
              <ModifierSelect
                label="Perfusion"
                options={PERFUSION}
                value={form.perfusion}
                onChange={(v) => set("perfusion", v)}
                note="In a child this moves before the blood pressure does."
              />
              <ModifierSelect
                label="Active bleeding"
                options={BLEEDING}
                value={form.bleeding}
                onChange={(v) => set("bleeding", v)}
              />
            </div>
          </fieldset>

          {isChild && (
            <fieldset>
              <legend className={legendClass}>
                Paediatric Assessment Triangle — MTS 2022 p. 13
              </legend>
              <ModifierChips
                label="Danger signs present"
                options={PAEDIATRIC_SIGNS}
                values={form.paediatric_signs}
                onChange={(v) => set("paediatric_signs", v)}
                note="Page 13: a child without these proceeds to secondary triage. None of them appears in the page 14 vital-sign bands."
              />
            </fieldset>
          )}

          <fieldset>
            <legend className={legendClass}>
              Initial tests — MTS 2022 p. 7
            </legend>
            <ModifierChips
              label="12-lead ECG findings"
              options={ECG}
              values={form.ecg_findings}
              onChange={(v) => set("ecg_findings", v)}
              note="Leave every box clear if no ECG was taken. This is also what separates the STEMI guideline from the NSTE-ACS one."
            />
          </fieldset>

          <fieldset>
            <legend className={legendClass}>Context</legend>
            <div className="grid gap-3 sm:grid-cols-2">
              <ModifierSelect
                label="Fever history"
                options={FEVER_HISTORY}
                value={form.fever_history}
                onChange={(v) => set("fever_history", v)}
              />
              <ModifierSelect
                label="Mode of arrival"
                options={ARRIVAL}
                value={form.arrival_mode}
                onChange={(v) => set("arrival_mode", v)}
              />
              <div className="sm:col-span-2">
                <label className={labelClass} htmlFor="mechanism">
                  Mechanism of injury
                </label>
                <input
                  id="mechanism"
                  className={fieldClass}
                  type="text"
                  maxLength={300}
                  placeholder="Motorcycle vs car, 60 km/h, helmeted, thrown 5 m…"
                  value={form.trauma_mechanism ?? ""}
                  onChange={(e) => set("trauma_mechanism", e.target.value)}
                />
              </div>
            </div>
          </fieldset>

          <fieldset>
            <legend className={legendClass}>
              Background — read by the prescribing checks
            </legend>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <div className="flex items-center justify-between">
                  <label className={labelClass} htmlFor="allergies">
                    Drug allergies
                  </label>
                  <button
                    type="button"
                    onClick={() => set("allergies", "NKDA")}
                    className="mb-1 rounded border border-slate-300 px-1.5 py-0.5 text-[10px] text-slate-500 hover:border-sky-500 hover:text-sky-700 dark:border-slate-700"
                  >
                    NKDA
                  </button>
                </div>
                <input
                  id="allergies"
                  className={fieldClass}
                  type="text"
                  maxLength={500}
                  placeholder="Penicillin (rash), NSAIDs…"
                  value={form.allergies ?? ""}
                  onChange={(e) => set("allergies", e.target.value)}
                />
              </div>
              <div>
                <label className={labelClass} htmlFor="meds">
                  Current medications
                </label>
                <input
                  id="meds"
                  className={fieldClass}
                  type="text"
                  maxLength={1000}
                  placeholder="Metformin, amlodipine, sildenafil PRN…"
                  value={form.current_medications ?? ""}
                  onChange={(e) => set("current_medications", e.target.value)}
                />
              </div>
              <div>
                <label className={labelClass} htmlFor="egfr">
                  eGFR{" "}
                  <span className="text-slate-400">(ml/min/1.73m²)</span>
                </label>
                <input
                  id="egfr"
                  className={fieldClass}
                  type="number"
                  min={0}
                  max={200}
                  placeholder="for renal dose adjustment"
                  value={form.egfr ?? ""}
                  onChange={(e) =>
                    set("egfr", e.target.value === "" ? null : Number(e.target.value))
                  }
                />
              </div>
              <div className="sm:col-span-2">
                <ModifierChips
                  label="Comorbidities"
                  options={COMORBIDITIES}
                  values={form.comorbidities}
                  onChange={(v) => set("comorbidities", v)}
                  note="Only “Immunocompromised” changes the triage level (MTS p7 L2). The rest let the drug checks fire on a recorded fact."
                />
              </div>
            </div>
          </fieldset>

          {maybePregnant && (
            <fieldset>
              <legend className={legendClass}>Pregnancy</legend>
              <div className="grid gap-3 sm:grid-cols-3">
                <ModifierSelect
                  label="Status"
                  options={PREGNANCY}
                  value={form.pregnancy ?? "unknown"}
                  onChange={(v) => set("pregnancy", v ?? "unknown")}
                  note="“Not established” is not the same as “not pregnant”."
                />
                <div>
                  <label className={labelClass} htmlFor="gestation">
                    Gestation (weeks)
                  </label>
                  <input
                    id="gestation"
                    className={fieldClass}
                    type="number"
                    min={0}
                    max={45}
                    disabled={form.pregnancy !== "pregnant"}
                    value={form.gestation_weeks ?? ""}
                    onChange={(e) =>
                      set(
                        "gestation_weeks",
                        e.target.value === "" ? null : Number(e.target.value),
                      )
                    }
                  />
                </div>
                <div className="flex items-end pb-2">
                  <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
                    <input
                      type="checkbox"
                      className="size-4 rounded border-slate-300"
                      checked={form.breastfeeding ?? false}
                      onChange={(e) => set("breastfeeding", e.target.checked)}
                    />
                    Breastfeeding
                  </label>
                </div>
              </div>
            </fieldset>
          )}

          <fieldset>
            <legend className={legendClass}>
              Safety and placement — MTS 2022 pp. 5–6
            </legend>
            <div className="space-y-3">
              <ModifierChips
                label="Infection or exposure risk"
                options={EXPOSURE}
                values={form.exposure_risk}
                onChange={(v) => set("exposure_risk", v)}
                note="Drives isolation and decontamination. The printed table's output is a placement, not a triage level."
              />
              <ModifierChips
                label="Behavioural risk"
                options={BEHAVIOURAL}
                values={form.behavioural_risk}
                onChange={(v) => set("behavioural_risk", v)}
              />
            </div>
          </fieldset>
        </div>
      </details>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
        >
          {loading ? "Routing…" : "Route patient"}
        </button>
        <button
          type="button"
          onClick={() => setForm({ ...EMPTY, facility: form.facility })}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-600 transition hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
        >
          Clear
        </button>
        {loading && <ProgressRing progress={progress} />}
      </div>
    </form>
  );
}
