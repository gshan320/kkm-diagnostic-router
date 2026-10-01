"""One reasoning-model run at a time, first come first served, with a visible place in line.

Why this exists
---------------
The API endpoints are plain `def`, so FastAPI runs each request on its own
threadpool thread - and until 2026-09-30 nothing stopped two of them entering
MLX generation together. On a 16 GB machine that is the worst of both worlds:
both runs share one GPU, so each takes roughly twice as long, and the second
KV cache pushes the machine into swap (6 GB of swap was in use when this was
measured). A second clinician pressing Submit made the FIRST clinician's report
slower too, and nothing on screen said why.

So a run takes a ticket and waits its turn, and its progress reads "position 2
- 1 report ahead" instead of a ring that does not move.

Two layers, because there are two ways to collide:

  in-process   A ticket queue (FIFO, fair) across the API's threads. This is
               what knows your position.
  cross-process  An exclusive `flock` on data/.gpu.lock, taken after the ticket.
               The evaluation harness can run the engine in its own process,
               and a lock inside one process cannot see another. The OS
               releases a flock when its holder dies, so a crash cannot wedge it.

What is deliberately NOT here: a Redis/RQ broker. One GPU is one worker; a
broker would add a service to fail without adding throughput. See the
2026-09-30 platform review in the README.
"""

from __future__ import annotations

import fcntl
import itertools
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from . import config, progress


class QueueFull(RuntimeError):
    """Raised when more runs are already waiting than GPU_QUEUE_MAX allows."""


@dataclass
class Ticket:
    number: int
    job_id: str
    waited_ms: int = 0


_cond = threading.Condition()
_counter = itertools.count(1)
_waiting: list[int] = []      # ticket numbers, in arrival order
_running: int | None = None   # ticket number currently holding the model

_LOCK_PATH = config.DATA_DIR / ".gpu.lock"


def snapshot() -> dict:
    """For /health: how busy the model is right now."""
    with _cond:
        return {"running": _running is not None, "waiting": len(_waiting)}


def _position(number: int) -> int:
    """1 = next to run. Caller holds _cond."""
    return _waiting.index(number) + 1


def _announce(job_id: str, ahead: int) -> None:
    if ahead <= 0:
        progress.stage(job_id, "queued", "Starting")
        return
    noun = "report" if ahead == 1 else "reports"
    progress.stage(
        job_id, "queued",
        f"Waiting for the reasoning model - position {ahead + 1}, "
        f"{ahead} {noun} ahead of you",
    )


@contextmanager
def _process_lock(job_id: str):
    """Exclusive across processes. Polls rather than blocks so the wait can be
    reported - a blocking flock would leave the progress ring frozen."""
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        announced = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    progress.stage(
                        job_id, "queued",
                        "Waiting for another process using the reasoning model "
                        "(an evaluation run?)",
                    )
                    announced = True
                time.sleep(0.5)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def slot(job_id: str = ""):
    """Hold the reasoning model for the duration of the block.

    Raises QueueFull immediately, before waiting, when the line is already
    GPU_QUEUE_MAX long - telling someone "busy, try again" at once is kinder
    than a request that sits for twenty minutes."""
    global _running
    started = time.monotonic()
    with _cond:
        if len(_waiting) >= config.GPU_QUEUE_MAX:
            raise QueueFull(
                f"The reasoning model is busy: {len(_waiting)} reports are already "
                "waiting. Try again in a few minutes."
            )
        ticket = Ticket(number=next(_counter), job_id=job_id)
        _waiting.append(ticket.number)
        last_ahead = -1
        try:
            while _running is not None or _waiting[0] != ticket.number:
                ahead = _position(ticket.number) - 1 + (1 if _running is not None else 0)
                if ahead != last_ahead:
                    _announce(job_id, ahead)
                    last_ahead = ahead
                _cond.wait(timeout=1.0)
        except BaseException:
            # A waiter that dies must leave the line, or everyone behind it
            # waits forever for a ticket that will never be served.
            _waiting.remove(ticket.number)
            _cond.notify_all()
            raise
        _waiting.pop(0)
        _running = ticket.number
    try:
        with _process_lock(job_id):
            ticket.waited_ms = int((time.monotonic() - started) * 1000)
            yield ticket
    finally:
        with _cond:
            _running = None
            _cond.notify_all()
