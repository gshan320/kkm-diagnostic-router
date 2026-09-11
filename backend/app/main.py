"""FastAPI surface for the KKM Diagnostic Router.

  GET  /health                     liveness + index / model readiness
  GET  /api/v1/stats               what is indexed, and its known caveats
  POST /api/v1/triage              MODE A - validated DiagnosticSchema JSON
  POST /api/v1/compare-inquire     MODE B - comparative markdown + citations

Endpoints are plain `def`, so FastAPI runs the (synchronous) ChromaDB queries
and local MLX generation in its threadpool instead of blocking the event loop.
"""

from __future__ import annotations

import logging

from fastapi import Header, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from . import progress
from .rag_engine import RagError, get_engine
from .schemas import (
    CorpusStats,
    HealthResponse,
    InquiryRequest,
    InquiryResponse,
    ProgressResponse,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
) -> TriageResponse:
    """MODE A - vitals + complaint -> MTS 2022 level, red flags, diagnosis,
    immediate actions and KKM drug dosing with prescriber category.

    An optional `X-Job-Id` header opts the caller into live progress, readable
    at GET /api/v1/progress/{job_id} while this request is still running."""
    log.info(
        "MODE A  age=%s complaint=%r", request.age, request.complaint[:80]
    )
    progress.start(x_job_id)
    try:
        result = get_engine().triage(request, job_id=x_job_id)
        progress.finish(x_job_id)
        return result
    except RagError as exc:
        log.error("MODE A failed: %s", exc)
        progress.fail(x_job_id, str(exc))
        raise _fail(exc) from exc
    except Exception as exc:
        progress.fail(x_job_id, str(exc))
        raise


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
def compare_inquire(request: InquiryRequest) -> InquiryResponse:
    """MODE B - freeform CPG / formulary question -> comparative markdown."""
    log.info("MODE B  scope=%s query=%r", request.scope.value, request.query[:80])
    try:
        return get_engine().inquire(request)
    except RagError as exc:
        log.error("MODE B failed: %s", exc)
        raise _fail(exc) from exc
