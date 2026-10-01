"""Attention layer A4.1: where the time went, and what the model actually used.

Why this exists
---------------
A1-A3 changed what the model reads on evidence from the literature and from
hand measurements of one case at a time. Nothing recorded, per report, how long
each stage took or which of the passages the model was given it went on to
cite - so a threshold (passages per document, sentences per passage, the token
budget) could only be tuned by guessing. Run 7 (2026-09-30) took 167 s and
nothing in the report said where.

Per report this records:

  stages_ms        hypotheses, retrieval (search, gates, distillation),
                   reading (prefill), writing (decode), second pass, checks
  tokens           prompt, reused from the prefix cache, written
  passages         given vs cited, by position in the prompt (front / middle /
                   back thirds - the positions "Lost in the Middle" is about),
                   and the documents given but never cited
  corrections      citations moved or left unsupported (A4.2), complications
                   added from source (A4.3)

It rides on the response, so the audit log keeps it. Aggregate across reports:

    python -m app.attention_audit            # every report in the audit log
    python -m app.attention_audit --last 10
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from contextlib import contextmanager

from .schemas import AttentionReport

_SID = re.compile(r"\bS(\d+)\b")

STAGE_LABELS = (
    ("hypotheses", "hypotheses"),
    ("retrieval", "retrieval"),
    ("reading", "reading"),
    ("writing", "writing"),
    ("second_pass", "second pass"),
    ("checks", "checks"),
)


class Trace:
    """Collects stage times and generation statistics for one report."""

    def __init__(self) -> None:
        self.stages: dict[str, int] = {}
        self.gen: dict[str, dict] = {}
        self.second_pass = ""

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, int((time.perf_counter() - t0) * 1000))

    def add(self, name: str, ms: int) -> None:
        self.stages[name] = self.stages.get(name, 0) + max(0, int(ms))

    def generation(self, name: str) -> dict:
        """A dict for _generate(stats=...) to fill."""
        return self.gen.setdefault(name, {})


def _position(i: int, n: int) -> str:
    if n <= 0:
        return "middle"
    if i <= n / 3:
        return "front"
    if i > 2 * n / 3:
        return "back"
    return "middle"


def build(trace: Trace, sources: list, first_pass: int, diagnostic) -> AttentionReport:
    """The per-report record. `first_pass` is how many sources the first
    prompt carried; later ones came from the second pass or A4.3."""
    cited: set[int] = set()
    for c in diagnostic.citations:
        cited |= {int(x) for x in _SID.findall(str(c.source_id or ""))}
    given_pos: Counter = Counter()
    cited_pos: Counter = Counter()
    cited_titles = {src.cpg_title for i, src in enumerate(sources, start=1) if i in cited}
    given_titles: list[str] = []
    for i, src in enumerate(sources, start=1):
        pos = _position(i, first_pass) if i <= first_pass else "added later"
        given_pos[pos] += 1
        if i in cited:
            cited_pos[pos] += 1
        if i <= first_pass and src.cpg_title not in given_titles:
            given_titles.append(src.cpg_title)
    triage = trace.gen.get("triage", {})
    stage2 = trace.gen.get("stage2", {})
    hyp = trace.gen.get("hypotheses", {})
    stages = dict(trace.stages)
    if triage:
        stages["reading"] = triage.get("prefill_ms", 0)
        stages["writing"] = triage.get("decode_ms", 0)
    return AttentionReport(
        stages_ms=stages,
        prompt_tokens=triage.get("prompt_tokens", 0),
        prompt_tokens_reused=triage.get("reused_tokens", 0),
        output_tokens=triage.get("output_tokens", 0),
        stage2_output_tokens=stage2.get("output_tokens", 0),
        hypothesis_output_tokens=hyp.get("output_tokens", 0),
        passages_given=len(sources),
        passages_cited=len(cited),
        cited=[f"S{i}" for i in sorted(cited)],
        uncited_documents=[t for t in given_titles if t not in cited_titles],
        cited_by_position=dict(cited_pos),
        given_by_position=dict(given_pos),
        second_pass=trace.second_pass,
    )


def summary(rep: AttentionReport) -> str:
    """One line for "How the sources were chosen"."""
    pos = ", ".join(f"{p} {rep.cited_by_position.get(p, 0)}/{rep.given_by_position[p]}"
                    for p in ("front", "middle", "back", "added later") if rep.given_by_position.get(p))
    parts = [f"Attention: the answer cites {rep.passages_cited} of {rep.passages_given} passages ({pos})"]
    if rep.uncited_documents:
        parts.append("never cited: " + "; ".join(rep.uncited_documents[:4]))
    t = rep.stages_ms
    timing = " · ".join(f"{label} {t[k] / 1000:.0f} s" for k, label in STAGE_LABELS if t.get(k))
    tokens = (f"read {rep.prompt_tokens:,} tokens ({rep.prompt_tokens_reused:,} cached), "
              f"wrote {rep.output_tokens:,}"
              + (f" + {rep.stage2_output_tokens:,} in the second pass" if rep.stage2_output_tokens else ""))
    return "; ".join(parts) + f". Time: {timing}; {tokens}."


# ------------------------------------------------------------- aggregate CLI
def _reports(last: int | None) -> list[AttentionReport]:
    from . import audit  # noqa: PLC0415

    out = []
    for row in audit.recent(last or 10_000):
        full = audit.get(row["report_id"]) or {}
        try:
            att = (json.loads(full.get("response_json") or "{}") or {}).get("attention")
        except ValueError:
            continue
        if att:
            out.append(AttentionReport.model_validate(att))
    return out


def aggregate(reps: list[AttentionReport]) -> str:
    if not reps:
        return "No report in the audit log carries attention data yet (A4.1 records from 2026-09-30)."
    lines = [f"{len(reps)} report(s)"]
    for key, label in STAGE_LABELS:
        vals = [r.stages_ms[key] / 1000 for r in reps if r.stages_ms.get(key)]
        if vals:
            lines.append(f"  {label:12s} median {statistics.median(vals):6.1f} s  (max {max(vals):.1f})")
    lines.append(f"  prompt tokens median {statistics.median(r.prompt_tokens for r in reps):,.0f}; "
                 f"output tokens median {statistics.median(r.output_tokens for r in reps):,.0f}")
    given, cited = Counter(), Counter()
    for r in reps:
        given.update(r.given_by_position)
        cited.update(r.cited_by_position)
    lines.append("  cited share by position: " + ", ".join(
        f"{p} {cited[p] / given[p]:.0%}" for p in ("front", "middle", "back", "added later") if given[p]))
    never = Counter(t for r in reps for t in r.uncited_documents)
    if never:
        lines.append("  given but never cited: " + "; ".join(f"{t} ({n})" for t, n in never.most_common(6)))
    lines.append(f"  citations moved {sum(r.citation_moves for r in reps)}, "
                 f"left unsupported {sum(r.citations_unsupported for r in reps)}, "
                 f"complications added from source {sum(r.complications_added for r in reps)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--last", type=int, default=None, help="only the N most recent reports")
    args = ap.parse_args()
    print(aggregate(_reports(args.last)))


if __name__ == "__main__":
    main()
