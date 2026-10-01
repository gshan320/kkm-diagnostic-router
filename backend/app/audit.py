"""Append-only record of every report: input, output, guardrail decisions, provenance.

Why this exists
---------------
Three needs, one store:

  1. Reproducibility. A report can be looked up by its ID and tied to the exact
     code, corpus and model that produced it (see versioning.py).
  2. Measurement. The evaluation harness and the regression gate read outcomes
     from here instead of from screenshots and saved HTML files.
  3. Accountability. A clinical decision-support output that cannot be traced
     afterwards is not one a clinician can defend relying on.

Append-only is enforced by the database itself: triggers abort any UPDATE or
DELETE on `reports`. Tamper-EVIDENCE comes from a hash chain - every row stores
the previous row's hash and a hash over its own content - so `verify()` detects
a row altered or removed behind SQLite's back (e.g. with another tool).

PRIVACY. The rows contain the intake exactly as typed, which in real use is
patient data under PDPA 2010. The database lives under backend/data/audit/,
which is git-ignored, and never leaves the machine. During the pilot only
fictional demonstration cases should be entered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("audit")

_lock = threading.Lock()
_ready: set[str] = set()   # database paths whose schema is in place

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id           TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    mode                TEXT NOT NULL,
    status              TEXT NOT NULL,
    origin              TEXT NOT NULL,
    job_id              TEXT NOT NULL,
    mts_level           INTEGER,
    primary_diagnosis   TEXT,
    model               TEXT NOT NULL,
    git_sha             TEXT NOT NULL,
    git_dirty           INTEGER NOT NULL,
    code_fingerprint    TEXT NOT NULL,
    corpus_fingerprint  TEXT NOT NULL,
    queue_wait_ms       INTEGER,
    latency_ms          INTEGER,
    request_json        TEXT NOT NULL,
    response_json       TEXT,
    error               TEXT,
    prev_hash           TEXT NOT NULL,
    row_hash            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS flags (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    reason      TEXT NOT NULL,
    note        TEXT,
    flagged_by  TEXT
);
CREATE TRIGGER IF NOT EXISTS flags_no_update BEFORE UPDATE ON flags
BEGIN SELECT RAISE(ABORT, 'flags are append-only'); END;
CREATE TRIGGER IF NOT EXISTS flags_no_delete BEFORE DELETE ON flags
BEGIN SELECT RAISE(ABORT, 'flags are append-only'); END;
CREATE INDEX IF NOT EXISTS reports_created ON reports(created_at);
CREATE INDEX IF NOT EXISTS reports_origin ON reports(origin);
CREATE TRIGGER IF NOT EXISTS reports_no_update BEFORE UPDATE ON reports
BEGIN SELECT RAISE(ABORT, 'audit rows are append-only'); END;
CREATE TRIGGER IF NOT EXISTS reports_no_delete BEFORE DELETE ON reports
BEGIN SELECT RAISE(ABORT, 'audit rows are append-only'); END;
"""

# Every column the hash covers, in a fixed order. seq and row_hash are excluded:
# seq is assigned by SQLite, and row_hash is the output.
_HASHED = (
    "report_id", "created_at", "mode", "status", "origin", "job_id", "mts_level",
    "primary_diagnosis", "model", "git_sha", "git_dirty", "code_fingerprint",
    "corpus_fingerprint", "queue_wait_ms", "latency_ms", "request_json",
    "response_json", "error", "prev_hash",
)

GENESIS = "0" * 64


def _path() -> Path:
    return Path(config.AUDIT_DB)


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure(conn: sqlite3.Connection) -> None:
    key = str(_path())
    if key not in _ready:
        conn.executescript(_SCHEMA)
        _ready.add(key)


def _row_hash(fields: dict) -> str:
    canonical = json.dumps([fields.get(k) for k in _HASHED], ensure_ascii=False,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_report_id(mode: str, when: datetime) -> str:
    """Readable and unique: mode, local-date stamp, random suffix."""
    return f"{mode}-{when.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def record(
    *,
    mode: str,
    status: str,
    origin: str,
    job_id: str,
    request: dict,
    response: dict | None,
    error: str,
    provenance: dict,
    queue_wait_ms: int | None,
    latency_ms: int | None,
    mts_level: int | None = None,
    primary_diagnosis: str | None = None,
    report_id: str | None = None,
) -> str:
    """Append one row and return its report_id. Returns "" if auditing is off
    or the write failed - a failed audit write is logged loudly, but it never
    withholds a report that has already been generated."""
    if not config.AUDIT_ENABLED:
        return ""
    now = datetime.now(timezone.utc)
    fields = {
        "report_id": report_id or new_report_id(mode, now.astimezone()),
        "created_at": now.isoformat(timespec="seconds"),
        "mode": mode,
        "status": status,
        "origin": origin or "api",
        "job_id": job_id or "",
        "mts_level": mts_level,
        "primary_diagnosis": primary_diagnosis,
        "model": provenance.get("model", ""),
        "git_sha": provenance.get("git_sha", ""),
        "git_dirty": int(bool(provenance.get("git_dirty"))),
        "code_fingerprint": provenance.get("code_fingerprint", ""),
        "corpus_fingerprint": provenance.get("corpus_fingerprint", ""),
        "queue_wait_ms": queue_wait_ms,
        "latency_ms": latency_ms,
        "request_json": json.dumps(request, ensure_ascii=False, default=str),
        "response_json": json.dumps(response, ensure_ascii=False, default=str) if response is not None else None,
        "error": error or None,
    }
    try:
        with _lock:
            conn = _connect()
            try:
                _ensure(conn)
                conn.execute("BEGIN IMMEDIATE")
                last = conn.execute(
                    "SELECT row_hash FROM reports ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                fields["prev_hash"] = last["row_hash"] if last else GENESIS
                fields["row_hash"] = _row_hash(fields)
                cols = ", ".join(fields)
                marks = ", ".join("?" for _ in fields)
                conn.execute(f"INSERT INTO reports ({cols}) VALUES ({marks})", list(fields.values()))
                conn.commit()
            finally:
                conn.close()
        return fields["report_id"]
    except Exception as exc:
        log.error("AUDIT WRITE FAILED - report %s was returned but NOT recorded: %s",
                  fields["report_id"], exc)
        return ""


def get(report_id: str) -> dict | None:
    conn = _connect()
    try:
        _ensure(conn)
        row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def recent(limit: int = 50, origin: str | None = None) -> list[dict]:
    """Summary rows, newest first - no request/response bodies."""
    conn = _connect()
    try:
        _ensure(conn)
        cols = ("seq, report_id, created_at, mode, status, origin, mts_level, "
                "primary_diagnosis, model, git_sha, git_dirty, code_fingerprint, "
                "corpus_fingerprint, queue_wait_ms, latency_ms, error")
        if origin:
            rows = conn.execute(
                f"SELECT {cols} FROM reports WHERE origin = ? ORDER BY seq DESC LIMIT ?",
                (origin, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM reports ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def add_flag(report_id: str, reason: str, note: str = "", flagged_by: str = "") -> bool:
    """Record a clinician's objection to a report. False if the report does not
    exist - a flag must always point at something that can be looked up."""
    with _lock:
        conn = _connect()
        try:
            _ensure(conn)
            if not conn.execute("SELECT 1 FROM reports WHERE report_id = ?", (report_id,)).fetchone():
                return False
            conn.execute(
                "INSERT INTO flags (report_id, created_at, reason, note, flagged_by) VALUES (?, ?, ?, ?, ?)",
                (report_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), reason, note, flagged_by),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def flags(report_id: str | None = None, limit: int = 200) -> list[dict]:
    conn = _connect()
    try:
        _ensure(conn)
        if report_id:
            rows = conn.execute("SELECT * FROM flags WHERE report_id = ? ORDER BY seq", (report_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM flags ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def verify() -> tuple[bool, int, str]:
    """Walk the chain. Returns (intact, rows_checked, first problem or "")."""
    conn = _connect()
    try:
        _ensure(conn)
        prev = GENESIS
        n = 0
        for row in conn.execute("SELECT * FROM reports ORDER BY seq"):
            n += 1
            fields = dict(row)
            if fields["prev_hash"] != prev:
                return False, n, f"seq {fields['seq']}: chain broken (a row before it was removed or altered)"
            if _row_hash(fields) != fields["row_hash"]:
                return False, n, f"seq {fields['seq']} ({fields['report_id']}): content does not match its hash"
            prev = fields["row_hash"]
        return True, n, ""
    finally:
        conn.close()
