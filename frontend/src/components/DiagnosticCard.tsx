"use client";

import { downloadTriageReport, printTriageReport } from "@/lib/report";
import type { Contraindication, TriageColour, TriageResponse } from "@/lib/types";
import SourceList from "./SourceList";

const COLOUR_STYLES: Record<TriageColour, string> = {
  RED: "bg-red-600 text-white",
  YELLOW: "bg-amber-500 text-white",
  GREEN: "bg-emerald-700 text-white",
};

const DISPOSITION_LABEL: Record<string, string> = {
  RESUSCITATION_BAY: "Resuscitation bay",
  ADMIT_ICU_HDU: "Admit — ICU / HDU",
  ADMIT_WARD: "Admit — ward",
  ED_OBSERVATION: "ED observation",
  REFER_SPECIALIST: "Refer to specialist",
  DISCHARGE_WITH_FOLLOW_UP: "Discharge with follow-up",
};

const CONFIDENCE_STYLES: Record<string, string> = {
  HIGH: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  MODERATE: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  LOW: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
};

/** The model may emit source ids already bracketed ("[S1]") because the prompt
 *  template shows them that way — strip before re-bracketing. */
function Cite({ id }: { id: string }) {
  const clean = (id ?? "").trim().replace(/^\[|\]$/g, "");
  if (!clean) return null;
  return (
    <span className="ml-1 font-mono text-[10px] text-sky-700 dark:text-sky-400">
      [{clean}]
    </span>
  );
}

function Section({
  title,
  count,
  children,
}: {
  title: string;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-slate-100 px-5 py-4 dark:border-slate-800">
      <h3 className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase">
        {title}
        {count != null && <span className="ml-1 text-slate-400">({count})</span>}
      </h3>
      {children}
    </section>
  );
}

const EMPTY = (
  <p className="text-sm text-slate-500 italic">
    None identified from the retrieved sources.
  </p>
);

export default function DiagnosticCard({ result }: { result: TriageResponse }) {
  const d = result.diagnostic;

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

      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        {/* MTS 2022 badge */}
        <div className="flex flex-wrap items-center gap-4 px-5 py-4">
          <div
            className={`flex flex-col items-center rounded-xl px-5 py-3 ${COLOUR_STYLES[result.triage_colour]}`}
          >
            <span className="text-[10px] font-semibold tracking-widest uppercase opacity-90">
              MTS 2022
            </span>
            <span className="text-3xl leading-none font-bold">
              {d.mts_triage_level}
            </span>
            <span className="mt-0.5 text-[11px] font-semibold tracking-wide">
              {d.mts_triage_label.replace("_", " ")}
            </span>
          </div>
          <div className="min-w-48 flex-1">
            <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Time to treatment: {d.time_to_treatment}
            </p>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              {d.triage_rationale}
            </p>
            {d.reassessment ? (
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                {d.reassessment}
              </p>
            ) : null}
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="text-right text-[11px] text-slate-400">
              <p>{result.model}</p>
              <p>{(result.latency_ms / 1000).toFixed(1)}s</p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => downloadTriageReport(result)}
                title="Save a self-contained, printable HTML report of these findings"
                className="rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-slate-900 dark:bg-slate-700 dark:hover:bg-slate-600"
              >
                Download report
              </button>
              <button
                type="button"
                onClick={() => printTriageReport(result)}
                title="Open the print dialog — choose a printer, or Save as PDF"
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                Print / PDF
              </button>
            </div>
          </div>
        </div>

        {/* Safety checks sit ABOVE everything clinical. An absolute
            contraindication that a reader has to scroll to is not a safeguard.
            Grouped by flagged recommendation: one GTN order objected to on
            three grounds is ONE problem with three reasons, not six alerts. */}
        {(d.contraindications?.length ?? 0) > 0 &&
          (() => {
            const groups = new Map<string, Contraindication[]>();
            for (const c of d.contraindications!) {
              const key = `${c.item}||${c.where}`;
              groups.set(key, [...(groups.get(key) ?? []), c]);
            }
            const worst = (g: Contraindication[]) =>
              g.some((c) => c.severity === "ABSOLUTE") ? "ABSOLUTE" : "CAUTION";
            const ordered = [...groups.values()].sort((a, b) =>
              worst(a) === worst(b) ? 0 : worst(a) === "ABSOLUTE" ? -1 : 1,
            );
            const nAbs = ordered.filter((g) => worst(g) === "ABSOLUTE").length;
            return (
              <div className="mx-5 mb-4 rounded-lg border-2 border-rose-500 bg-rose-50 px-4 py-3 dark:border-rose-700 dark:bg-rose-950/50">
                <p className="text-xs font-bold uppercase tracking-wide text-rose-900 dark:text-rose-200">
                  Automated safety checks
                </p>
                {nAbs > 0 && (
                  <p className="mt-1 text-sm font-semibold text-rose-900 dark:text-rose-100">
                    {nAbs} recommendation{nAbs > 1 ? "s" : ""} above{" "}
                    {nAbs > 1 ? "are" : "is"} absolutely contraindicated for this
                    patient. Do not act without re-reading the cited guideline.
                  </p>
                )}
                <ul className="mt-2 space-y-2">
                  {ordered.map((g, i) => {
                    const sev = worst(g);
                    return (
                      <li
                        key={`${g[0].item}-${i}`}
                        className="text-sm text-rose-900 dark:text-rose-100"
                      >
                        <span
                          className={`mr-2 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold ${
                            sev === "ABSOLUTE"
                              ? "bg-rose-600 text-white"
                              : "bg-amber-400 text-amber-950"
                          }`}
                        >
                          {sev}
                          {g.length > 1 ? ` ×${g.length}` : ""}
                        </span>
                        <strong>{g[0].item}</strong>{" "}
                        <span className="text-rose-700 dark:text-rose-300">
                          ({g[0].where})
                        </span>
                        <ul className="mt-1 space-y-1">
                          {g.map((c, j) => (
                            <li
                              key={j}
                              className="text-xs text-rose-800 dark:text-rose-200/90"
                            >
                              <em>&ldquo;{c.trigger}&rdquo;</em> — {c.reason}
                            </li>
                          ))}
                        </ul>
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })()}

        {(d.completeness_gaps?.length ?? 0) > 0 && (
          <div className="mx-5 mb-4 rounded-lg border border-orange-300 bg-orange-50 px-4 py-3 dark:border-orange-800 dark:bg-orange-950/40">
            <p className="text-xs font-semibold text-orange-900 dark:text-orange-300">
              Not addressed in this report
            </p>
            <p className="mt-1 text-sm text-orange-900 dark:text-orange-200">
              {d.completeness_gaps!.map((g) => g.element).join(" · ")}
            </p>
            <p className="mt-1 text-xs text-orange-800/80 dark:text-orange-300/80">
              Read {d.completeness_gaps![0].guideline} before acting.
            </p>
          </div>
        )}

        {(d.dose_completeness_warning ||
          d.drug_indication_warning ||
          d.citation_warning ||
          d.urgency_warning ||
          d.redirection_warning ||
          d.parse_warning) && (
          <div className="mx-5 mb-4 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 dark:border-amber-800 dark:bg-amber-950/40">
            <p className="text-xs font-semibold text-amber-900 dark:text-amber-300">
              Grounding checks
            </p>
            {[
              d.parse_warning,
              d.dose_completeness_warning,
              d.drug_indication_warning,
              d.urgency_warning,
              d.redirection_warning,
              d.citation_warning,
            ]
              .filter(Boolean)
              .map((w, i) => (
                <p key={i} className="mt-1 text-sm text-amber-900 dark:text-amber-200">
                  {w}
                </p>
              ))}
          </div>
        )}

        {d.prescriber_category_warning && (
          <div className="mx-5 mb-4 rounded-lg border border-rose-300 bg-rose-50 px-4 py-3 dark:border-rose-800 dark:bg-rose-950/40">
            <p className="text-xs font-semibold text-rose-900 dark:text-rose-300">
              Prescriber-category warning
            </p>
            <p className="mt-1 text-sm text-rose-900 dark:text-rose-200">
              {d.prescriber_category_warning}
            </p>
          </div>
        )}

        <Section title="Red flags" count={d.red_flags.length}>
          {d.red_flags.length === 0 ? (
            EMPTY
          ) : (
            <ul className="space-y-2">
              {d.red_flags.map((flag) => (
                <li
                  key={flag.flag}
                  className="rounded-lg border-l-4 border-rose-500 bg-rose-50/60 px-3 py-2 dark:bg-rose-950/30"
                >
                  <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                    {flag.flag}
                    <Cite id={flag.source_id} />
                  </p>
                  <p className="mt-0.5 text-xs text-slate-600 dark:text-slate-400">
                    {flag.why_it_matters}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Section>

        <Section title="Primary diagnosis">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-base font-semibold text-slate-900 dark:text-slate-100">
              {d.primary_diagnosis.condition}
            </p>
            <span
              className={`rounded px-2 py-0.5 text-[10px] font-semibold ${CONFIDENCE_STYLES[d.primary_diagnosis.confidence]}`}
            >
              {d.primary_diagnosis.confidence} CONFIDENCE
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            {d.primary_diagnosis.reasoning}
          </p>
          {d.primary_diagnosis.supporting_cpg && (
            <p className="mt-1 text-xs text-slate-500">
              Governing guideline: {d.primary_diagnosis.supporting_cpg}
            </p>
          )}
          {d.differential_diagnoses.length > 0 && (
            <div className="mt-3">
              <p className="mb-1 text-xs font-medium text-slate-500">
                Differentials to exclude
              </p>
              <ul className="space-y-1">
                {d.differential_diagnoses.map((diff) => (
                  <li key={diff.condition} className="text-xs text-slate-600 dark:text-slate-400">
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {diff.condition}
                    </span>{" "}
                    — {diff.discriminating_feature}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Section>

        <Section title="Immediate actions" count={d.immediate_actions.length}>
          {d.immediate_actions.length === 0 ? (
            EMPTY
          ) : (
            <ol className="space-y-2">
              {[...d.immediate_actions]
                .sort((a, b) => a.sequence - b.sequence)
                .map((action) => (
                  <li key={action.sequence} className="flex gap-3">
                    <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-sky-600 text-[11px] font-semibold text-white">
                      {action.sequence}
                    </span>
                    <div>
                      <p className="text-sm text-slate-800 dark:text-slate-200">
                        {action.action}
                        <Cite id={action.source_id} />
                      </p>
                      <p className="text-xs font-medium text-sky-700 dark:text-sky-400">
                        {action.timeframe}
                      </p>
                    </div>
                  </li>
                ))}
            </ol>
          )}
        </Section>

        {d.investigations.length > 0 && (
          <Section title="Investigations" count={d.investigations.length}>
            <div className="w-full overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-3 py-2 font-semibold">Test</th>
                    <th className="px-3 py-2 font-semibold">Urgency</th>
                    <th className="px-3 py-2 font-semibold">Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {d.investigations.map((investigation) => (
                    <tr
                      key={investigation.test}
                      className="border-t border-slate-100 dark:border-slate-800"
                    >
                      <td className="px-3 py-2 font-medium text-slate-800 dark:text-slate-200">
                        {investigation.test}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-sky-700 dark:text-sky-400">
                        {investigation.urgency}
                      </td>
                      <td className="px-3 py-2 text-slate-600 dark:text-slate-400">
                        {investigation.rationale}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        )}

        {/* Drug dosing card */}
        <Section title="KKM drug dosing" count={d.drug_recommendations.length}>
          {d.drug_recommendations.length === 0 ? (
            EMPTY
          ) : (
            <ul className="grid gap-3 lg:grid-cols-2">
              {d.drug_recommendations.map((drug) => {
                const unverified =
                  drug.prescriber_category === "NOT_IN_RETRIEVED_SOURCES";
                return (
                  <li
                    key={`${drug.drug_name}-${drug.indication}`}
                    className="rounded-lg border border-slate-200 p-3 dark:border-slate-800"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        {drug.drug_name}
                      </p>
                      <span
                        className={`rounded px-2 py-0.5 font-mono text-[10px] font-semibold ${
                          unverified
                            ? "bg-slate-200 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                            : "bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300"
                        }`}
                        title={drug.prescriber_category_meaning}
                      >
                        {unverified ? "NOT IN FORMULARY" : `CAT. ${drug.prescriber_category}`}
                      </span>
                      <Cite id={drug.source_id} />
                    </div>
                    <p className="mt-1 text-xs text-slate-500">{drug.indication}</p>
                    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                      {[
                        ["Adult dose", drug.adult_dose],
                        ["Paediatric dose", drug.paediatric_dose],
                        ["Route", drug.route],
                        ["Frequency", drug.frequency],
                        ["Duration", drug.duration],
                      ]
                        .filter(([, value]) => value)
                        .map(([label, value]) => (
                          <div key={label}>
                            <dt className="text-slate-400">{label}</dt>
                            <dd className="font-medium text-slate-800 dark:text-slate-200">
                              {value}
                            </dd>
                          </div>
                        ))}
                    </dl>
                    {drug.cautions && (
                      <p className="mt-2 border-t border-slate-100 pt-2 text-xs text-amber-800 dark:border-slate-800 dark:text-amber-400">
                        {drug.cautions}
                      </p>
                    )}
                    {/* The authoritative dose, quoted verbatim. The figure in
                        the table above is the model's; these lines come from the
                        formulary and the guideline. */}
                    {(drug.dose_source_fukkm ||
                      drug.dose_source_cpg ||
                      drug.dose_verdict) && (
                      <div className="mt-2 rounded border-l-2 border-slate-300 bg-slate-50 px-2.5 py-2 dark:border-slate-600 dark:bg-slate-800/50">
                        {drug.dose_verdict && (
                          <span
                            className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-bold ${
                              drug.dose_verdict === "VERIFIED"
                                ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300"
                                : drug.dose_verdict === "EXCEEDS_MAXIMUM"
                                  ? "bg-rose-600 text-white"
                                  : drug.dose_verdict === "DIFFERS_FROM_SOURCE"
                                    ? "bg-amber-300 text-amber-950"
                                    : "bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-300"
                            }`}
                          >
                            {drug.dose_verdict === "VERIFIED"
                              ? "DOSE VERIFIED"
                              : drug.dose_verdict === "EXCEEDS_MAXIMUM"
                                ? "EXCEEDS STATED MAXIMUM"
                                : drug.dose_verdict === "DIFFERS_FROM_SOURCE"
                                  ? "DIFFERS FROM SOURCE"
                                  : "DOSE NOT VERIFIED"}
                          </span>
                        )}
                        {drug.dose_source_fukkm && (
                          <p className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">
                            <span className="font-semibold text-slate-500">Formulary: </span>
                            {drug.dose_source_fukkm}
                          </p>
                        )}
                        {drug.dose_source_cpg && (
                          <p className="mt-1 text-[11px] text-slate-600 dark:text-slate-300">
                            <span className="font-semibold text-slate-500">Guideline: </span>
                            {drug.dose_source_cpg}
                          </p>
                        )}
                        {drug.dose_verdict !== "VERIFIED" &&
                          drug.dose_verdict_detail && (
                            <p className="mt-1 text-[11px] text-amber-800 dark:text-amber-400">
                              <span className="font-semibold">Check: </span>
                              {drug.dose_verdict_detail}
                            </p>
                          )}
                      </div>
                    )}
                    <p className="mt-1 text-[11px] text-slate-400">
                      {drug.prescriber_category_meaning}
                    </p>
                  </li>
                );
              })}
            </ul>
          )}
        </Section>

        <Section title="Disposition">
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            {DISPOSITION_LABEL[d.disposition] ?? d.disposition}
            {d.referral_required && d.referral_to && (
              <span className="ml-2 font-normal text-slate-600 dark:text-slate-400">
                → {d.referral_to}
              </span>
            )}
          </p>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            {d.disposition_justification}
          </p>
        </Section>

        {d.vitals_interpretation.length > 0 && (
          <Section title="Vital-sign interpretation" count={d.vitals_interpretation.length}>
            <ul className="space-y-1">
              {d.vitals_interpretation.map((vital) => (
                <li key={vital.parameter} className="text-xs">
                  <span className="font-medium text-slate-800 dark:text-slate-200">
                    {vital.parameter}: {vital.value}
                  </span>
                  <span className="text-slate-600 dark:text-slate-400">
                    {" "}
                    — {vital.interpretation}
                  </span>
                  {vital.mts_level_triggered && (
                    <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                      triggers {vital.mts_level_triggered}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {d.evidence_gaps && (
          <Section title="Evidence gaps">
            <p className="text-sm text-slate-600 dark:text-slate-400">
              {d.evidence_gaps}
            </p>
          </Section>
        )}

        <div className="border-t border-slate-100 bg-slate-50 px-5 py-3 dark:border-slate-800 dark:bg-slate-950/50">
          <p className="text-[11px] leading-relaxed text-slate-500">
            {result.disclaimer}
          </p>
        </div>
      </div>

      <SourceList sources={result.sources} />
    </div>
  );
}
