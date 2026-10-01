"""What exactly produced a report: code, corpus and model, as fingerprints.

Why this exists
---------------
A triage report is only reproducible - and only defensible - if it can be tied
to the exact code, guideline corpus and model that produced it. "Qwen3-8B on the
KKM corpus" is not enough: the corpus changed four times in September 2026 and
a guardrail that is correct today may not have existed when an old report was
printed. So every report carries three fingerprints, and the audit store keeps
them beside the input and output.

  code_fingerprint    sha256 over every app/*.py file's bytes. Exact whether or
                      not the tree is committed - `git_sha` alone would claim a
                      clean commit for a working tree full of edits.
  git_sha / git_dirty the commit, and whether the working tree differed from it.
  corpus_fingerprint  sha256 over what the INDEX holds (filename + chunk count
                      per document) plus the FUKKM JSON's bytes. What is in
                      raw_pdfs/ is not what was retrieved from until re-ingest,
                      so the folder is the wrong thing to hash.

All three are computed once per process: the code cannot change under a running
process without a reload, and the corpus only changes through re-ingest, which
also restarts the server.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from . import config

_APP_DIR = Path(__file__).resolve().parent
_REPO_DIR = _APP_DIR.parent.parent


@dataclass(frozen=True)
class Provenance:
    git_sha: str
    git_dirty: bool
    code_fingerprint: str
    corpus_fingerprint: str
    corpus_documents: int
    corpus_chunks: int
    model: str

    def as_dict(self) -> dict:
        return asdict(self)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_DIR), *args],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(_APP_DIR.glob("*.py")):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def git_state() -> tuple[str, bool]:
    sha = _git("rev-parse", "--short=12", "HEAD")
    # Only the backend app counts: an edited README must not mark a report's
    # clinical logic as uncommitted.
    dirty = bool(_git("status", "--porcelain", "--", "backend/app"))
    return sha or "unknown", dirty


def corpus_fingerprint(collection) -> tuple[str, int, int]:
    """(fingerprint, documents, chunks) for what the index actually holds."""
    h = hashlib.sha256()
    tally: dict[str, int] = {}
    total = 0
    try:
        total = collection.count()
        if total:
            got = collection.get(include=["metadatas"], limit=total)
            for meta in got["metadatas"]:
                name = str(meta.get("filename", "?"))
                tally[name] = tally.get(name, 0) + 1
    except Exception:
        pass
    for name, n in sorted(tally.items()):
        h.update(f"{name}\t{n}\n".encode())
    fukkm = Path(config.FUKKM_JSON)
    if fukkm.exists():
        h.update(b"FUKKM\0")
        h.update(fukkm.read_bytes())
    return h.hexdigest()[:16], len(tally), total
