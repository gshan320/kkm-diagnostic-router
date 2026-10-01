"use client";

import type { TriagePreview } from "@/lib/types";

/** The code-decided triage, shown about a second after Submit while the model
 *  is still writing the full report. Everything here comes from the MTS 2022
 *  table and the red-flag rules - no model - so it is labelled PROVISIONAL and
 *  replaced by the full report when that arrives. */
const TONE: Record<string, string> = {
  RED: "border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200",
  YELLOW:
    "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200",
  GREEN:
    "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200",
};

export function TriagePreviewCard({ preview }: { preview: TriagePreview }) {
  return (
    <section
      className={`rounded-xl border px-5 py-4 ${TONE[preview.triage_colour] ?? TONE.YELLOW}`}
      aria-live="polite"
    >
      <p className="text-[11px] font-semibold uppercase tracking-wide opacity-80">
        Provisional triage — code only, full report still generating
      </p>
      <p className="mt-1 text-lg font-semibold">
        MTS {preview.mts_triage_level} · {preview.mts_triage_label.replace(/_/g, " ")} ·{" "}
        {preview.triage_colour}
        <span className="ml-2 text-sm font-normal">
          Time to treatment: {preview.time_to_treatment}
        </span>
      </p>
      {preview.reasons.length > 0 && (
        <ul className="mt-2 list-disc pl-5 text-sm">
          {preview.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
      {preview.red_flags.length > 0 && (
        <p className="mt-2 text-sm">
          <span className="font-semibold">Red-flag rules fired: </span>
          {preview.red_flags.join("; ")}
        </p>
      )}
      {preview.provisional_note && (
        <p className="mt-2 text-sm font-medium">{preview.provisional_note}</p>
      )}
      {preview.reassessment && <p className="mt-1 text-xs opacity-80">{preview.reassessment}</p>}
      <p className="mt-2 text-xs opacity-70">{preview.note}</p>
    </section>
  );
}
