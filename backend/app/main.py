"""FastAPI surface for the KKM Diagnostic Router.

  GET  /health                     liveness + index / model readiness
  GET  /api/v1/stats               what is indexed, and its known caveats
  POST /api/v1/triage              MODE A - validated DiagnosticSchema JSON
  POST /api/v1/compare-inquire     MODE B - comparative markdown + citations
  GET  /api/v1/reports             recent audit rows (summaries, no bodies)
  GET  /api/v1/reports/{id}        one audited report, input and output
  GET  /api/v1/audit/verify        walk the audit hash chain
  POST /api/v1/triage/preview      the deterministic triage in ~1 s, no model
  POST /api/v1/reports/{id}/flag   a clinician's one-click objection
  GET  /api/v1/sources/pdf/{file}  a corpus PDF, so a citation can open its page

Every Mode A and Mode B run - success or failure - is appended to the audit
store (audit.py) with the code / corpus / model fingerprints that produced it.

Endpoints are plain `def`, so FastAPI runs the (synchronous) ChromaDB queries
and local MLX generation in its threadpool instead of blocking the event loop.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import Header, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from . import audit, config, gpu_queue, progress
from .rag_engine import RagError, get_engine, triage_preview
from .schemas import (
    CorpusStats,
    HealthResponse,
    InquiryRequest,
    InquiryResponse,
    ProgressResponse,
    ReportFlag,
    ReportProvenance,
    TriagePreview,
    TriageRequest,
    TriageResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="KKM Diagnostic Router & CPG Comparative Intelligence",
    version="0.1.0",
    description=(
        "Local RAG over official KKM CPGs, Quick Reference guides, Paediatric "
        "Protocols, MTS 2022 and the FUKKM formulary. Clinical decision support "
        "for registered clinicians - not a medical device, not for patient use."
    ),
)

@app.on_event("startup")
def _warm_up() -> None:
    """Warm the engine in the background: the server answers /health at once,
    and a report submitted during warm-up queues for the GPU behind it."""
    if not config.WARMUP:
        return
    import threading  # noqa: PLC0415

    threading.Thread(target=get_engine().warm_up, name="warm-up", daemon=True).start()


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _busy(exc: gpu_queue.QueueFull) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def _stamp(result, engine, mode: str, origin: str, job_id: str, request: dict,
           started: float, mts_level=None, primary_diagnosis=None) -> None:
    """Attach provenance to a result and append it to the audit store."""
    prov = engine.provenance
    audit_id = audit.record(
        mode=mode, status="ok", origin=origin, job_id=job_id, request=request,
        response=result.model_dump(mode="json"), error="", provenance=prov,
        queue_wait_ms=result.provenance.queue_wait_ms,
        latency_ms=int((time.perf_counter() - started) * 1000),
        mts_level=mts_level, primary_diagnosis=primary_diagnosis,
    )
    result.provenance = ReportProvenance(
        audit_id=audit_id,
        git_sha=prov["git_sha"],
        git_dirty=prov["git_dirty"],
        code_fingerprint=prov["code_fingerprint"],
        corpus_fingerprint=prov["corpus_fingerprint"],
        queue_wait_ms=result.provenance.queue_wait_ms,
    )


def _record_failure(engine, mode: str, origin: str, job_id: str, request: dict,
                    started: float, exc: Exception) -> None:
    try:
        prov = engine.provenance
    except Exception:
        prov = {}
    audit.record(
        mode=mode, status="error", origin=origin, job_id=job_id, request=request,
        response=None, error=f"{type(exc).__name__}: {exc}", provenance=prov,
        queue_wait_ms=None, latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _fail(exc: RagError) -> HTTPException:
    """503 = the service is not ready to answer; 502 = the model misbehaved."""
    message = str(exc).lower()
    if "index empty" in message or "index is empty" in message:
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = get_engine()
    total = engine.count()
    return HealthResponse(
        status="ok" if total else "index_empty",
        chroma_ready=total > 0,
        total_chunks=total,
        model=engine.model_name,
        model_loaded=engine.model_loaded,
        gpu_busy=gpu_queue.snapshot()["running"],
        gpu_waiting=gpu_queue.snapshot()["waiting"],
        warm=engine.warm_state,
    )


@app.get("/api/v1/stats", response_model=CorpusStats)
def stats() -> CorpusStats:
    engine = get_engine()
    total = engine.count()
    documents: list[dict] = []
    if total:
        got = engine.collection.get(include=["metadatas"], limit=total)
        tally: dict[tuple, int] = {}
        for meta in got["metadatas"]:
            key = (
                str(meta.get("filename", "?")),
                str(meta.get("cpg_title", "?")),
                str(meta.get("doc_type", "?")),
                str(meta.get("edition_year", "")),
                str(meta.get("edition", "")),
            )
            tally[key] = tally.get(key, 0) + 1
        documents = [
            {
                "filename": k[0],
                "cpg_title": k[1],
                "doc_type": k[2],
                "edition_year": k[3],
                "edition": k[4],
                "chunks": n,
            }
            for k, n in sorted(tally.items(), key=lambda kv: -kv[1])
        ]
    return CorpusStats(
        collection=config.COLLECTION_NAME,
        total_chunks=total,
        documents=documents,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_tokens=config.CHUNK_TOKENS,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        warnings=engine.corpus_warnings if total else ["Index is empty."],
    )


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(
    request: TriageRequest,
    x_job_id: str = Header(default="", alias="X-Job-Id"),
    x_client: str = Header(default="api", alias="X-Client"),
) -> TriageResponse:
    """MODE A - vitals + complaint -> MTS 2022 level, red flags, diagnosis,
    immediate actions and KKM drug dosing with prescriber category.

    An optional `X-Job-Id` header opts the caller into live progress, readable
    at GET /api/v1/progress/{job_id} while this request is still running.
    `X-Client` labels the audit row's origin ("ui", "eval"; default "api")."""
    log.info(
        "MODE A  age=%s complaint=%r", request.age, request.complaint[:80]
    )
    engine = get_engine()
    started = time.perf_counter()
    body = request.model_dump(mode="json")
    progress.start(x_job_id)
    try:
        result = engine.triage(request, job_id=x_job_id)
        d = result.diagnostic
        _stamp(result, engine, "A", x_client, x_job_id, body, started,
               mts_level=int(d.mts_triage_level),
               primary_diagnosis=str(d.primary_diagnosis.condition or ""))
        progress.finish(x_job_id)
        return result
    except gpu_queue.QueueFull as exc:
        progress.fail(x_job_id, str(exc))
        raise _busy(exc) from exc
    except RagError as exc:
        log.error("MODE A failed: %s", exc)
        _record_failure(engine, "A", x_client, x_job_id, body, started, exc)
        progress.fail(x_job_id, str(exc))
        raise _fail(exc) from exc
    except Exception as exc:
        _record_failure(engine, "A", x_client, x_job_id, body, started, exc)
        progress.fail(x_job_id, str(exc))
        raise


@app.post("/api/v1/triage/preview", response_model=TriagePreview)
def preview(request: TriageRequest) -> TriagePreview:
    """The MTS level, time to treatment and red flags the code alone decides.
    No model, no GPU queue: shown on screen while the full report generates."""
    return triage_preview(request)


@app.get("/api/v1/progress/{job_id}", response_model=ProgressResponse)
def get_progress(job_id: str) -> ProgressResponse:
    """Live progress for an in-flight Mode A run.

    `percent` is exact through retrieval and prefill. The decode phase is an
    ESTIMATE against a calibrated typical output length, because the model's
    output length is not knowable in advance - `estimated` says which."""
    job = progress.get(job_id)
    if job is None:
        return ProgressResponse(job_id=job_id, stage="unknown", percent=0.0,
                                detail="No such job (it may have finished).",
                                done=False, estimated=False, elapsed_s=0.0)
    return ProgressResponse(
        job_id=job.job_id, stage=job.stage, percent=job.percent, detail=job.detail,
        done=job.done, error=job.error, elapsed_s=job.elapsed_s,
        estimated=job.stage == "decode",
    )


@app.post("/api/v1/compare-inquire", response_model=InquiryResponse)
def compare_inquire(
    request: InquiryRequest,
    x_client: str = Header(default="api", alias="X-Client"),
) -> InquiryResponse:
    """MODE B - freeform CPG / formulary question -> comparative markdown."""
    log.info("MODE B  scope=%s query=%r", request.scope.value, request.query[:80])
    engine = get_engine()
    started = time.perf_counter()
    body = request.model_dump(mode="json")
    try:
        result = engine.inquire(request)
        _stamp(result, engine, "B", x_client, "", body, started)
        return result
    except gpu_queue.QueueFull as exc:
        raise _busy(exc) from exc
    except RagError as exc:
        log.error("MODE B failed: %s", exc)
        _record_failure(engine, "B", x_client, "", body, started, exc)
        raise _fail(exc) from exc
    except Exception as exc:
        _record_failure(engine, "B", x_client, "", body, started, exc)
        raise


# ------------------------------------------------------------------- audit
# Bound to 127.0.0.1 during development. These return patient intakes as typed,
# so they must sit behind authentication before any network deployment.

@app.get("/api/v1/reports")
def list_reports(limit: int = 50, origin: str | None = None) -> list[dict]:
    """Recent audited runs, newest first. Summaries only - no intake or output."""
    return audit.recent(limit=max(1, min(limit, 500)), origin=origin)


@app.get("/api/v1/reports/{report_id}")
def get_report(report_id: str) -> dict:
    """One audited run with its full intake and output."""
    row = audit.get(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No audited report {report_id!r}.")
    for key in ("request_json", "response_json"):
        if row.get(key):
            row[key.removesuffix("_json")] = json.loads(row.pop(key))
    row["flags"] = audit.flags(report_id)
    return row


@app.post("/api/v1/reports/{report_id}/flag")
def flag_report(report_id: str, flag: ReportFlag) -> dict:
    """Record a clinician's objection. Flags are append-only and feed the
    improvement loop: every user becomes a one-click reviewer."""
    if not audit.add_flag(report_id, flag.reason, flag.note, flag.flagged_by):
        raise HTTPException(status_code=404, detail=f"No audited report {report_id!r}.")
    return {"ok": True, "report_id": report_id, "flags": len(audit.flags(report_id))}


@app.get("/api/v1/flags")
def list_flags(limit: int = 200) -> list[dict]:
    return audit.flags(limit=max(1, min(limit, 1000)))


@app.get("/api/v1/sources/pdf/{filename}")
def source_pdf(filename: str) -> FileResponse:
    """Serve one corpus PDF so a citation link can open it at its page
    (`...#page=N`). Only names that exist in raw_pdfs/ are served - no paths."""
    names = {p.name: p for p in config.RAW_PDF_DIR.glob("*.pdf")}
    path = names.get(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Not a corpus document.")
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{path.name}"'})


@app.get("/api/v1/audit/verify")
def verify_audit() -> dict:
    """Walk the hash chain: detects a row altered or removed outside the app."""
    intact, rows, problem = audit.verify()
    return {"intact": intact, "rows_checked": rows, "problem": problem}
