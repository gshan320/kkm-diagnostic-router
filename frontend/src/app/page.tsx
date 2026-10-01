"use client";

import { useEffect, useState } from "react";
import CPGQueryForm from "@/components/CPGQueryForm";
import ComparativeResponseCard from "@/components/ComparativeResponseCard";
import DiagnosticCard from "@/components/DiagnosticCard";
import PatientInputForm from "@/components/PatientInputForm";
import {
  API_BASE,
  fetchProgress,
  fetchStats,
  runInquiry,
  runPreview,
  runTriage,
} from "@/lib/api";
import { TriagePreviewCard } from "@/components/TriagePreviewCard";
import type {
  TriagePreview,
  ProgressResponse,
  CorpusStats,
  InquiryRequest,
  InquiryResponse,
  TriageRequest,
  TriageResponse,
} from "@/lib/types";

type Tab = "triage" | "explorer";

const TABS: Array<{ id: Tab; label: string; hint: string }> = [
  { id: "triage", label: "Clinical Triage Router", hint: "MODE A" },
  { id: "explorer", label: "CPG & Formulary Explorer", hint: "MODE B" },
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("triage");
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [statsLoaded, setStatsLoaded] = useState(false);

  const [triageResult, setTriageResult] = useState<TriageResponse | null>(null);
  const [triageError, setTriageError] = useState<string | null>(null);
  const [triageLoading, setTriageLoading] = useState(false);
  const [triageProgress, setTriageProgress] = useState<ProgressResponse | null>(null);
  const [triagePreview, setTriagePreview] = useState<TriagePreview | null>(null);

  const [inquiryResult, setInquiryResult] = useState<InquiryResponse | null>(null);
  const [inquiryError, setInquiryError] = useState<string | null>(null);
  const [inquiryLoading, setInquiryLoading] = useState(false);

  useEffect(() => {
    let active = true;
    fetchStats().then((value) => {
      if (!active) return;
      setStats(value);
      setStatsLoaded(true);
    });
    return () => {
      active = false;
    };
  }, []);

  async function submitTriage(request: TriageRequest) {
    // The backend reports real progress against this id: exact for retrieval
    // and prompt prefill, estimated for generation.
    const jobId =
      globalThis.crypto?.randomUUID?.() ??
      `job-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setTriageLoading(true);
    setTriageError(null);
    setTriageResult(null);
    setTriageProgress(null);
    setTriagePreview(null);
    // The code-decided triage arrives in about a second; show it while the
    // model writes the full report.
    void runPreview(request).then((p) => setTriagePreview(p));

    let polling = true;
    const poll = async () => {
      while (polling) {
        const p = await fetchProgress(jobId);
        if (!polling) break;
        if (p) setTriageProgress(p);
        await new Promise((r) => setTimeout(r, 900));
      }
    };
    void poll();

    try {
      setTriageResult(await runTriage(request, jobId));
    } catch (error) {
      setTriageError(error instanceof Error ? error.message : String(error));
    } finally {
      polling = false;
      setTriageLoading(false);
      setTriageProgress(null);
    }
  }

  async function submitInquiry(request: InquiryRequest) {
    setInquiryLoading(true);
    setInquiryError(null);
    setInquiryResult(null);
    try {
      setInquiryResult(await runInquiry(request));
    } catch (error) {
      setInquiryError(error instanceof Error ? error.message : String(error));
    } finally {
      setInquiryLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          KKM Diagnostic Router &amp; CPG Comparative Intelligence
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
          Retrieval-augmented decision support over official Malaysian MOH
          Clinical Practice Guidelines, Quick Reference guides, Paediatric
          Protocols, the Malaysian Triage Scale 2022 and the FUKKM formulary.
          For registered clinicians — every answer is traceable to a cited page.
        </p>

        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
          {!statsLoaded ? (
            <span>Checking index…</span>
          ) : stats ? (
            <>
              <span>
                <strong className="text-slate-700 dark:text-slate-300">
                  {stats.total_chunks.toLocaleString()}
                </strong>{" "}
                chunks · {stats.documents.length} sources
              </span>
              <span>
                {stats.embedding_model} · {stats.chunk_tokens}/
                {stats.chunk_overlap_tokens} tokens
              </span>
            </>
          ) : (
            <span className="text-rose-600">
              API unreachable at {API_BASE} — start it with ./backend/run.sh
            </span>
          )}
        </div>
      </header>

      <nav className="mb-6 flex gap-1 border-b border-slate-200 dark:border-slate-800">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setTab(item.id)}
            className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium transition ${
              tab === item.id
                ? "border-sky-600 text-sky-700 dark:text-sky-400"
                : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-300"
            }`}
          >
            {item.label}
            <span className="ml-2 font-mono text-[10px] text-slate-400">
              {item.hint}
            </span>
          </button>
        ))}
      </nav>

      {tab === "triage" ? (
        <div className="space-y-5">
          <PatientInputForm
            onSubmit={submitTriage}
            loading={triageLoading}
            progress={triageProgress}
          />
          {triageLoading && triagePreview && <TriagePreviewCard preview={triagePreview} />}
          {triageError && <ErrorBanner message={triageError} />}
          {triageResult && <DiagnosticCard result={triageResult} />}
        </div>
      ) : (
        <div className="space-y-5">
          <CPGQueryForm onSubmit={submitInquiry} loading={inquiryLoading} />
          {inquiryError && <ErrorBanner message={inquiryError} />}
          {inquiryResult && <ComparativeResponseCard result={inquiryResult} />}
        </div>
      )}

      {stats && stats.documents.length > 0 && (
        <details className="mt-8 rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <summary className="cursor-pointer px-5 py-3 text-sm font-semibold text-slate-900 dark:text-slate-100">
            Indexed corpus ({stats.documents.length} documents)
          </summary>
          <div className="w-full overflow-x-auto border-t border-slate-100 dark:border-slate-800">
            <table className="w-full border-collapse text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800">
                <tr>
                  <th className="px-4 py-2 font-semibold">Document</th>
                  <th className="px-4 py-2 font-semibold">Type</th>
                  <th className="px-4 py-2 font-semibold">Edition</th>
                  <th className="px-4 py-2 font-semibold">Year</th>
                  <th className="px-4 py-2 text-right font-semibold">Chunks</th>
                </tr>
              </thead>
              <tbody>
                {stats.documents.map((document) => (
                  <tr
                    key={document.filename}
                    className="border-t border-slate-100 dark:border-slate-800"
                  >
                    <td className="px-4 py-2 text-slate-800 dark:text-slate-200">
                      {document.cpg_title}
                    </td>
                    <td className="px-4 py-2 text-slate-500">{document.doc_type}</td>
                    <td className="px-4 py-2 text-slate-500">
                      {document.edition || "—"}
                    </td>
                    <td className="px-4 py-2 text-slate-500">
                      {document.edition_year || (
                        <span className="text-amber-600">not stated</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-500">
                      {document.chunks.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </main>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-rose-300 bg-rose-50 px-4 py-3 dark:border-rose-800 dark:bg-rose-950/40">
      <p className="text-xs font-semibold text-rose-900 dark:text-rose-300">
        Request failed
      </p>
      <p className="mt-1 text-sm text-rose-900 dark:text-rose-200">{message}</p>
    </div>
  );
}
