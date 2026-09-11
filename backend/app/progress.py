"""Live progress for a single Mode A run.

Why a registry rather than a spinner
------------------------------------
A case takes 3-4 minutes. A spinner tells the user nothing about whether the
machine is working or wedged, and an animated bar that is not fed by real work
is worse - it invents confidence.

What is measurable, and what is not:

  retrieval  exact.       Discrete steps, each one completes or does not.
  prefill    exact.       mlx-lm reports tokens processed against the total via
                          `prompt_progress_callback`.
  decode     ESTIMATED.   The model stops when it stops; the output length is
                          unknown in advance. Progress follows a curve that is
                          linear to a calibrated typical length then asymptotic,
                          so it keeps advancing for ANY output length and never
                          reaches 100 - completion is claimed only by finish().

The decode phase is labelled as an estimate in the UI for that reason. Ranges
are weighted by measured wall-clock: prefill ~60s and decode ~35-150s of a
~220s case, so decode owns the largest span.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

# Weighted by measured share of wall clock, not by step count.
STAGE_BOUNDS = {
    "queued":    (0.0, 2.0),
    "retrieval": (2.0, 12.0),
    "prefill":   (12.0, 45.0),
    "decode":    (45.0, 97.0),
    "checks":    (97.0, 99.0),
    "done":      (100.0, 100.0),
}

# Median output length measured across benchmark runs (258-1518 observed).
# Only used as the denominator for the decode estimate.
TYPICAL_OUTPUT_TOKENS = 520

_MAX_TRACKED = 32


@dataclass
class Job:
    job_id: str
    stage: str = "queued"
    percent: float = 0.0
    detail: str = "Waiting to start"
    done: bool = False
    error: str = ""
    started: float = field(default_factory=time.monotonic)
    updated: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.started, 1)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def start(job_id: str) -> None:
    if not job_id:
        return
    with _lock:
        # Bound memory: this is a single-user local tool, but a long-lived
        # server must not accumulate abandoned jobs forever.
        if len(_jobs) >= _MAX_TRACKED:
            for stale in sorted(_jobs.values(), key=lambda j: j.updated)[: len(_jobs) // 2]:
                _jobs.pop(stale.job_id, None)
        _jobs[job_id] = Job(job_id=job_id)


def _set(job_id: str, *, stage: str, frac: float, detail: str) -> None:
    """`frac` is 0..1 WITHIN the stage; it is mapped into that stage's band."""
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job.done:
            return
        lo, hi = STAGE_BOUNDS.get(stage, (0.0, 100.0))
        pct = lo + (hi - lo) * max(0.0, min(1.0, frac))
        # Never move backwards: a later stage starting must not appear to undo
        # progress the user has already seen.
        job.percent = round(max(job.percent, pct), 1)
        job.stage = stage
        job.detail = detail
        job.updated = time.monotonic()


def stage(job_id: str, name: str, detail: str, frac: float = 0.0) -> None:
    _set(job_id, stage=name, frac=frac, detail=detail)


def prefill(job_id: str, processed: int, total: int) -> None:
    """Exact: fed by mlx-lm's prompt_progress_callback."""
    if total <= 0:
        return
    _set(job_id, stage="prefill", frac=processed / total,
         detail=f"Reading {total:,} tokens of guideline context ({processed:,} done)")


def decode(job_id: str, tokens: int) -> None:
    """Estimated: the output length is not known until the model stops.

    Linear to 85% of the stage at the calibrated typical length, then asymptotic.
    A hard clamp was tried first and measured badly: a shocked-ACS answer ran to
    1,144 tokens against a 520-token estimate, so the bar sat frozen at 96% for a
    minute - which reads as a hang, the exact anxiety a progress bar exists to
    remove. This curve keeps moving for any output length and never reaches 100%,
    so completion is only ever claimed by finish()."""
    n = max(0, tokens)
    if n <= TYPICAL_OUTPUT_TOKENS:
        frac = 0.85 * n / TYPICAL_OUTPUT_TOKENS
    else:
        over = (n - TYPICAL_OUTPUT_TOKENS) / TYPICAL_OUTPUT_TOKENS
        frac = 0.85 + 0.14 * (1.0 - math.exp(-over))
    _set(job_id, stage="decode", frac=frac,
         detail=f"Writing the assessment — {tokens} tokens (length estimated)")


def finish(job_id: str, detail: str = "Complete") -> None:
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.stage, job.percent, job.detail, job.done = "done", 100.0, detail, True
        job.updated = time.monotonic()


def fail(job_id: str, message: str) -> None:
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.stage, job.done, job.error = "error", True, message
        job.detail = f"Failed: {message}"
        job.updated = time.monotonic()


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)
