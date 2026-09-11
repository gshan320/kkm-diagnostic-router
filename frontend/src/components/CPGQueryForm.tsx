"use client";

import { useState } from "react";
import type { InquiryRequest, InquiryScope } from "@/lib/types";

const SCOPES: Array<{ value: InquiryScope; label: string }> = [
  { value: "all", label: "All sources" },
  { value: "cpg_only", label: "CPGs (full + QR)" },
  { value: "quick_reference_only", label: "Quick Reference only" },
  { value: "paediatric_only", label: "Paediatric Protocols" },
  { value: "triage_only", label: "MTS 2022 only" },
  { value: "formulary_only", label: "FUKKM formulary" },
];

const EXAMPLE_QUERIES = [
  "How does IV fluid resuscitation differ between the adult and the paediatric dengue guidelines?",
  "Compare the blood-pressure targets in the Hypertension CPG against those in the Ischaemic Stroke QR.",
  "What antiplatelet regimen does the ACS CPG recommend, and what is each drug's FUKKM prescriber category?",
  "What are the MTS 2022 Level 2 vital-sign criteria for a 3-month-old?",
];

const fieldClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100";

export default function CPGQueryForm({
  onSubmit,
  loading,
}: {
  onSubmit: (request: InquiryRequest) => void;
  loading: boolean;
}) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<InquiryScope>("all");
  const [maxSources, setMaxSources] = useState(24);

  const canSubmit = query.trim().length >= 3 && !loading;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (canSubmit) onSubmit({ query: query.trim(), scope, max_sources: maxSources });
      }}
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <h2 className="mb-3 text-sm font-semibold text-slate-900 dark:text-slate-100">
        Ask across the indexed KKM corpus
      </h2>

      <textarea
        className={`${fieldClass} h-24 resize-y`}
        placeholder="e.g. What is the difference in IV fluid resuscitation between the Dengue CPG editions?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        required
        minLength={3}
      />

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div>
          <label
            className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400"
            htmlFor="scope"
          >
            Search scope
          </label>
          <select
            id="scope"
            className={fieldClass}
            value={scope}
            onChange={(e) => setScope(e.target.value as InquiryScope)}
          >
            {SCOPES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label
            className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400"
            htmlFor="max-sources"
          >
            Chunks retrieved: {maxSources}
          </label>
          <input
            id="max-sources"
            type="range"
            min={4}
            max={60}
            step={2}
            value={maxSources}
            onChange={(e) => setMaxSources(Number(e.target.value))}
            className="mt-3 w-full accent-sky-600"
          />
        </div>
      </div>

      <div className="mt-4">
        <p className="mb-2 text-xs font-medium text-slate-500">Try one of these</p>
        <div className="flex flex-col gap-1.5">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setQuery(example)}
              className="rounded-lg border border-slate-200 px-3 py-2 text-left text-xs text-slate-600 transition hover:border-sky-400 hover:text-sky-700 dark:border-slate-800 dark:text-slate-400 dark:hover:text-sky-400"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5 flex items-center gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
        >
          {loading ? "Scanning…" : "Compare & answer"}
        </button>
        {loading && (
          <span className="text-xs text-slate-500">
            Scanning PDFs and the formulary, then composing the comparison…
          </span>
        )}
      </div>
    </form>
  );
}
