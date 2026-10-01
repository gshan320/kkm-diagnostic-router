"""Regression tests for the Phase 0 platform pieces added 2026-09-30.

    gpu_queue    one run at a time, first come first served, a visible place in
                 line, a hard cap, and a waiter that dies leaves the line.
    audit        append-only enforced by SQLite itself, and a hash chain that
                 notices a row changed behind the application's back.
    versioning   the fingerprints are stable within a process and change when
                 the code does.

None of these load the model or the index.

Run from the backend directory:

    ../.venv/bin/python -m tests.test_platform
"""
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

from app import audit, config, gpu_queue, progress, versioning

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ===========================================================================
print("1. GPU queue: one at a time, in arrival order.")
# ===========================================================================
order: list[str] = []
active = 0
peak = 0
guard = threading.Lock()


def run(name: str, hold: float) -> None:
    global active, peak
    progress.start(name)
    with gpu_queue.slot(name):
        with guard:
            active += 1
            peak = max(peak, active)
            order.append(name)
        time.sleep(hold)
        with guard:
            active -= 1


threads = []
for i, name in enumerate(["first", "second", "third", "fourth"]):
    t = threading.Thread(target=run, args=(name, 0.3))
    t.start()
    threads.append(t)
    time.sleep(0.05)  # arrive in a known order
time.sleep(0.1)
waiting_detail = progress.get("fourth").detail
for t in threads:
    t.join()

check("never more than one run holds the model", peak == 1, f"peak={peak}")
check("runs are served in arrival order", order == ["first", "second", "third", "fourth"], str(order))
check("a waiting run is told its position",
      "position" in waiting_detail and "ahead of you" in waiting_detail, waiting_detail)
check("queue is empty afterwards", gpu_queue.snapshot() == {"running": False, "waiting": 0},
      str(gpu_queue.snapshot()))

# ===========================================================================
print("2. GPU queue: the cap refuses at once instead of queueing forever.")
# ===========================================================================
saved_max = config.GPU_QUEUE_MAX
config.GPU_QUEUE_MAX = 1
release = threading.Event()


def holder() -> None:
    with gpu_queue.slot("holder"):
        release.wait(5)


def waiter() -> None:
    with gpu_queue.slot("waiter"):
        pass


h = threading.Thread(target=holder)
h.start()
time.sleep(0.1)
w = threading.Thread(target=waiter)
w.start()
time.sleep(0.2)  # the waiter now fills the one waiting place
refused = False
t0 = time.monotonic()
try:
    with gpu_queue.slot("third"):
        pass
except gpu_queue.QueueFull:
    refused = True
check("a run beyond GPU_QUEUE_MAX is refused", refused)
check("the refusal is immediate, not after waiting", time.monotonic() - t0 < 0.5)
release.set()
h.join()
w.join()
config.GPU_QUEUE_MAX = saved_max
check("the held and waiting runs still completed", gpu_queue.snapshot()["waiting"] == 0)

# ===========================================================================
print("3. GPU queue: a waiter that raises leaves the line.")
# ===========================================================================
release2 = threading.Event()


def holder2() -> None:
    with gpu_queue.slot("holder2"):
        release2.wait(5)


class Boom(Exception):
    pass


h2 = threading.Thread(target=holder2)
h2.start()
time.sleep(0.1)
# Simulate a waiter dying mid-wait by making progress.stage raise inside _announce.
real_stage = progress.stage


def exploding(job_id, *a, **k):
    if job_id == "doomed":
        raise Boom()
    return real_stage(job_id, *a, **k)


progress.stage = exploding
died = False
try:
    with gpu_queue.slot("doomed"):
        pass
except Boom:
    died = True
progress.stage = real_stage
check("the dying waiter's exception propagates", died)
check("it no longer occupies a place in line", gpu_queue.snapshot()["waiting"] == 0,
      str(gpu_queue.snapshot()))
release2.set()
h2.join()
ok_after = []


def next_run() -> None:
    with gpu_queue.slot("after"):
        ok_after.append(True)


n = threading.Thread(target=next_run)
n.start()
n.join(3)
check("the next run is not blocked by the dead ticket", ok_after == [True])

# ===========================================================================
print("4. Audit: rows are recorded, append-only, and chained.")
# ===========================================================================
tmp = Path(tempfile.mkdtemp()) / "audit.db"
config.AUDIT_DB = tmp
config.AUDIT_ENABLED = True
prov = {"model": "test-model", "git_sha": "abc", "git_dirty": False,
        "code_fingerprint": "c0de", "corpus_fingerprint": "c0rpus"}
ids = [
    audit.record(mode="A", status="ok", origin="test", job_id=f"j{i}",
                 request={"age": 40 + i, "complaint": "chest pain"},
                 response={"diagnostic": {"mts_triage_level": 3}}, error="",
                 provenance=prov, queue_wait_ms=0, latency_ms=100 + i,
                 mts_level=3, primary_diagnosis="ACS")
    for i in range(3)
]
check("record returns a report id for each row", all(ids) and len(set(ids)) == 3, str(ids))
row = audit.get(ids[1])
check("a row reads back with its provenance",
      row is not None and row["code_fingerprint"] == "c0de" and row["mts_level"] == 3)
check("recent() lists newest first", [r["report_id"] for r in audit.recent(10)] == ids[::-1])
intact, n_rows, problem = audit.verify()
check("an untouched chain verifies", intact and n_rows == 3, problem)

conn = sqlite3.connect(tmp)
blocked_update = blocked_delete = False
try:
    conn.execute("UPDATE reports SET mts_level = 5 WHERE report_id = ?", (ids[0],))
except sqlite3.DatabaseError:
    blocked_update = True
try:
    conn.execute("DELETE FROM reports WHERE report_id = ?", (ids[0],))
except sqlite3.DatabaseError:
    blocked_delete = True
check("UPDATE is refused by the database", blocked_update)
check("DELETE is refused by the database", blocked_delete)

# Tamper the way a determined editor would: drop the trigger, then edit.
conn.execute("DROP TRIGGER reports_no_update")
conn.execute("UPDATE reports SET mts_level = 5 WHERE report_id = ?", (ids[1],))
conn.commit()
conn.close()
intact, _, problem = audit.verify()
check("an edited row is detected by verify()", not intact and ids[1] in problem, problem)

tmp2 = Path(tempfile.mkdtemp()) / "audit.db"
config.AUDIT_DB = tmp2
for i in range(3):
    audit.record(mode="A", status="ok", origin="test", job_id="", request={"i": i},
                 response=None, error="", provenance=prov, queue_wait_ms=0, latency_ms=1)
conn = sqlite3.connect(tmp2)
conn.execute("DROP TRIGGER reports_no_delete")
conn.execute("DELETE FROM reports WHERE seq = 2")
conn.commit()
conn.close()
intact, _, problem = audit.verify()
check("a removed row is detected by verify()", not intact and "chain broken" in problem, problem)

config.AUDIT_ENABLED = False
check("auditing off records nothing and returns an empty id",
      audit.record(mode="A", status="ok", origin="t", job_id="", request={}, response=None,
                   error="", provenance=prov, queue_wait_ms=0, latency_ms=0) == "")
config.AUDIT_ENABLED = True

# ===========================================================================
print("5. Versioning: fingerprints are stable and content-derived.")
# ===========================================================================
a = versioning.code_fingerprint()
b = versioning.code_fingerprint()
check("code fingerprint is stable within a process", a == b and len(a) == 16)
versioning.code_fingerprint.cache_clear()
check("and recomputes identically from the same files", versioning.code_fingerprint() == a)
sha, dirty = versioning.git_state()
check("git state resolves", sha != "" and isinstance(dirty, bool), sha)


class FakeCollection:
    def __init__(self, metas):
        self.metas = metas

    def count(self):
        return len(self.metas)

    def get(self, include, limit):
        return {"metadatas": self.metas[:limit]}


one = versioning.corpus_fingerprint(FakeCollection([{"filename": "a.pdf"}, {"filename": "b.pdf"}]))
two = versioning.corpus_fingerprint(FakeCollection([{"filename": "b.pdf"}, {"filename": "a.pdf"}]))
three = versioning.corpus_fingerprint(FakeCollection([{"filename": "a.pdf"}, {"filename": "a.pdf"}]))
check("corpus fingerprint ignores chunk order", one[0] == two[0])
check("corpus fingerprint changes when the indexed content does", one[0] != three[0])
check("corpus fingerprint counts documents and chunks", one[1:] == (2, 2), str(one))

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("All checks passed.")
