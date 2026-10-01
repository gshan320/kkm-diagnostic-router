import type {
  CorpusStats,
  FlagReason,
  TriagePreview,
  ProgressResponse,
  InquiryRequest,
  InquiryResponse,
  TriageRequest,
  TriageResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

/** FastAPI returns `detail` as a string for HTTPException and as an array for
 *  422 validation errors — surface both as one readable message. */
async function readError(response: Response): Promise<string> {
  let detail: unknown;
  try {
    detail = (await response.json())?.detail;
  } catch {
    detail = undefined;
  }
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const field = Array.isArray(d?.loc) ? d.loc.slice(1).join(".") : "";
        return field ? `${field}: ${d?.msg}` : String(d?.msg ?? d);
      })
      .join("; ");
  }
  return `${response.status} ${response.statusText}`;
}

async function post<TRequest, TResponse>(
  path: string,
  body: TRequest,
  headers: Record<string, string> = {},
): Promise<TResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(
      `Cannot reach the API at ${API_BASE}. Start it with ./backend/run.sh`,
    );
  }
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as TResponse;
}

/** Labels this app's requests in the backend audit store, so evaluation
 *  runs and direct API calls can be told apart from clinician use. */
const CLIENT = { "X-Client": "ui" };

/** `jobId` opts into live progress readable at fetchProgress(jobId). */
export const runTriage = (request: TriageRequest, jobId?: string) =>
  post<TriageRequest, TriageResponse>(
    "/api/v1/triage",
    request,
    jobId ? { ...CLIENT, "X-Job-Id": jobId } : CLIENT,
  );

/** Progress of an in-flight run. Returns null on any failure: a progress poll
 *  must never surface an error over the actual triage request. */
export async function fetchProgress(jobId: string): Promise<ProgressResponse | null> {
  try {
    const r = await fetch(`${API_BASE}/api/v1/progress/${encodeURIComponent(jobId)}`, {
      cache: "no-store",
    });
    if (!r.ok) return null;
    return (await r.json()) as ProgressResponse;
  } catch {
    return null;
  }
}

export const runInquiry = (request: InquiryRequest) =>
  post<InquiryRequest, InquiryResponse>("/api/v1/compare-inquire", request, CLIENT);

export async function fetchStats(): Promise<CorpusStats | null> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as CorpusStats;
  } catch {
    return null;
  }
}

/** The deterministic triage (MTS table + red-flag rules), no model: ~1 s.
 *  Returns null on failure - the full report is still coming. */
export async function runPreview(request: TriageRequest): Promise<TriagePreview | null> {
  try {
    return await post<TriageRequest, TriagePreview>("/api/v1/triage/preview", request);
  } catch {
    return null;
  }
}

/** A clinician's one-click objection to an audited report. */
export const flagReport = (reportId: string, reason: FlagReason, note: string) =>
  post<{ reason: FlagReason; note: string }, { ok: boolean; flags: number }>(
    `/api/v1/reports/${encodeURIComponent(reportId)}/flag`,
    { reason, note },
  );

/** A corpus PDF opened at a page, served by the backend. */
export const sourcePdfUrl = (filename: string, page?: number | string | null) =>
  `${API_BASE}/api/v1/sources/pdf/${encodeURIComponent(filename)}${
    page && String(page) !== "N/A" ? `#page=${page}` : ""
  }`;

export { API_BASE };
