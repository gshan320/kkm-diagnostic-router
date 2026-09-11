"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { InquiryResponse } from "@/lib/types";
import SourceList from "./SourceList";

export default function ComparativeResponseCard({
  result,
}: {
  result: InquiryResponse;
}) {
  return (
    <div className="space-y-4">
      {result.corpus_warnings.length > 0 && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 dark:border-amber-800 dark:bg-amber-950/40">
          <p className="mb-1 text-xs font-semibold text-amber-900 dark:text-amber-300">
            Corpus caveats — read before acting
          </p>
          <ul className="list-disc space-y-0.5 pl-4 text-xs text-amber-900 dark:text-amber-300/90">
            {result.corpus_warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div className="border-b border-slate-100 px-5 py-3 dark:border-slate-800">
          <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
            {result.query}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
            <span>{result.sources.length} chunks retrieved</span>
            <span>·</span>
            <span>{result.model}</span>
            <span>·</span>
            <span>{(result.latency_ms / 1000).toFixed(1)}s</span>
          </div>
          {result.documents_scanned.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {result.documents_scanned.map((document) => (
                <span
                  key={document}
                  className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                >
                  {document}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="md px-5 py-4">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Comparative tables are wide; give each its own scroll container
              // so the page body never scrolls sideways.
              table: ({ children }) => (
                <div className="table-scroll">
                  <table>{children}</table>
                </div>
              ),
            }}
          >
            {result.answer_markdown}
          </Markdown>
        </div>
      </div>

      <SourceList sources={result.sources} />
    </div>
  );
}
