import type {
  CorpusStats,
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

/** `jobId` opts into live progress readable at fetchProgress(jobId). */
export const runTriage = (request: TriageRequest, jobId?: string) =>
  post<TriageRequest, TriageResponse>(
    "/api/v1/triage",
    request,
    jobId ? { "X-Job-Id": jobId } : {},
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
  post<InquiryRequest, InquiryResponse>("/api/v1/compare-inquire", request);

export async function fetchStats(): Promise<CorpusStats | null> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as CorpusStats;
  } catch {
    return null;
  }
}

export { API_BASE };
