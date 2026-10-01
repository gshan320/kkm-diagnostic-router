"use client";

import { useState } from "react";
import type { RetrievedSource } from "@/lib/types";

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

export default function SourceList({ sources }: { sources: RetrievedSource[] }) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
      className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <summary className="cursor-pointer px-5 py-3 text-sm font-semibold text-slate-900 dark:text-slate-100">
        Retrieved sources ({sources.length})
        <span className="ml-2 font-normal text-slate-500">
          — every [S#] the model was given
        </span>
      </summary>
      <div className="max-h-96 overflow-y-auto border-t border-slate-100 px-5 py-3 dark:border-slate-800">
        <ul className="space-y-3">
          {sources.map((source) => (
            <li key={source.source_id} className="text-xs">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono font-semibold text-sky-700 dark:text-sky-400">
                  [{source.source_id}]
                </span>
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  {source.cpg_title || source.filename}
                </span>
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                  {DOC_TYPE_LABEL[source.doc_type] ?? source.doc_type}
                </span>
                {source.edition_year && (
                  <span className="text-slate-500">{source.edition_year}</span>
                )}
                {source.page_number != null && (
                  <span className="text-slate-500">p.{source.page_number}</span>
                )}
                {source.prescriber_category && (
                  <span className="text-slate-500">
                    cat. {source.prescriber_category}
                  </span>
                )}
                {source.score != null && (
                  <span className="ml-auto font-mono text-slate-400">
                    {source.score.toFixed(3)}
                  </span>
                )}
              </div>
              <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-slate-500 dark:text-slate-400">
                {source.excerpt}
              </p>
            </li>
          ))}
        </ul>
      </div>
    </details>
  );
}
