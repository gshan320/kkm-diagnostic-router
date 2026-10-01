"""Run the evaluation cases through the live API and write a scorecard.

    # validate every case and check every answer-key quote against its source
    ../.venv/bin/python -m tests.eval.run --verify

    # run all cases (the API must be up: ./run.sh), then score
    ../.venv/bin/python -m tests.eval.run

    # a subset, by id or tier
    ../.venv/bin/python -m tests.eval.run --only rhabdo-exertional-55m,acs-stemi-58m
    ../.venv/bin/python -m tests.eval.run --tier mts_cell

    # re-score saved responses with the current answer keys - no model needed
    ../.venv/bin/python -m tests.eval.run --rescore data/eval_runs/<run-id>

Cases go through HTTP, not an in-process engine, on purpose: that is the path a
clinician's request takes, it goes through the GPU queue (so a run never
collides with someone using the UI), and every response is written to the audit
store with origin "eval".

A run is resumable: responses already saved in the run directory are not
re-requested. Outputs, all under data/eval_runs/<run-id>/ (git-ignored):

    responses/<case>.json   the raw API response, or the error
    scorecard.json          per-case and aggregate results
    scorecard.md            the same, readable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from app.schemas import TriageRequest

from . import harness as H

BACKEND = Path(__file__).resolve().parents[2]
RUNS = BACKEND / "data" / "eval_runs"


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, body: dict, headers: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"detail": str(e)}


def verify(cases: list[dict]) -> int:
    bad = 0
    total = 0
    for case in cases:
        try:
            TriageRequest.model_validate(case["intake"])
        except Exception as exc:
            print(f"  INTAKE INVALID  {case['id']}: {exc}")
            bad += 1
        for r in H.verify_expected(case.get("expected", {})):
            total += 1
            if r["status"] not in ("ok", "ok-spacing"):
                bad += 1
                print(f"  QUOTE {r['status'].upper():11s} {case['id']} / {r['where']}: "
                      f"{r['source']} p{r['page']} - {r['detail']}\n      \"{r['quote']}\"")
    print(f"{len(cases)} cases, {total} quotes checked, {bad} problem(s).")
    return 1 if bad else 0


def run_cases(cases: list[dict], run_dir: Path, base: str, timeout: float) -> None:
    out = run_dir / "responses"
    out.mkdir(parents=True, exist_ok=True)
    health = _get(f"{base}/health")
    print(f"API {base}: {health.get('status')}, model {health.get('model')}, "
          f"{health.get('total_chunks')} chunks")
    for i, case in enumerate(cases, 1):
        path = out / f"{case['id']}.json"
        if path.exists():
            print(f"  [{i}/{len(cases)}] {case['id']}: already saved, skipping")
            continue
        job = f"eval-{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        status, body = _post(f"{base}/api/v1/triage", case["intake"],
                             {"X-Client": "eval", "X-Job-Id": job}, timeout)
        elapsed = round(time.time() - t0, 1)
        record = {"case_id": case["id"], "http_status": status, "elapsed_s": elapsed,
                  "requested_at": datetime.now().isoformat(timespec="seconds"),
                  "response": body if status == 200 else None,
                  "error": None if status == 200 else body.get("detail", body)}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        lvl = body.get("diagnostic", {}).get("mts_triage_level") if status == 200 else "-"
        print(f"  [{i}/{len(cases)}] {case['id']}: HTTP {status} in {elapsed}s, MTS {lvl}")


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def score_run(cases: list[dict], run_dir: Path) -> dict:
    results = []
    provenance = None
    for case in cases:
        path = run_dir / "responses" / f"{case['id']}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("response") is None:
            results.append({"case_id": case["id"], "tier": case.get("tier"), "score": None,
                            "sections": {}, "safety": [], "error": rec.get("error")})
            continue
        provenance = provenance or rec["response"].get("provenance")
        r = H.score_case(case, rec["response"])
        r["elapsed_s"] = rec.get("elapsed_s")
        r["title"] = case.get("title")
        results.append(r)
    summary = H.aggregate(results)
    card = {"run": run_dir.name, "scored_at": datetime.now().isoformat(timespec="seconds"),
            "provenance": provenance, "summary": summary, "cases": results}
    (run_dir / "scorecard.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "scorecard.md").write_text(render_md(card), encoding="utf-8")
    return card


def render_md(card: dict) -> str:
    s = card["summary"]
    p = card.get("provenance") or {}
    lines = [
        f"# Evaluation scorecard - run {card['run']}",
        "",
        f"Scored {card['scored_at']}. Code `{p.get('code_fingerprint', '?')}`"
        f"{' (uncommitted)' if p.get('git_dirty') else ''}, corpus `{p.get('corpus_fingerprint', '?')}`, "
        f"git `{p.get('git_sha', '?')}`.",
        "",
        f"**Overall {_pct(s['overall'])}** over {s['scored']} scored cases "
        f"({s['cases']} run, {s['errors']} errors). Micro precision {_pct(s['micro_precision'])}, "
        f"micro recall {_pct(s['micro_recall'])}.",
        "",
        "## Safety",
        "",
        "| event | count |", "|---|---:|",
    ]
    for k in ("under_triage", "forbidden_drug", "unsupported_drug", "forbidden_source",
              "under_disposition", "over_triage"):
        lines.append(f"| {k.replace('_', ' ')}{' (informational)' if k == 'over_triage' else ''} "
                     f"| {s['safety'].get(k, 0)} |")
    lines += ["", f"Under-triage rate: {_pct(s['under_triage_rate'])}.", "",
              "## By tier", "", "| tier | cases | score |", "|---|---:|---:|"]
    for tier, v in s["by_tier"].items():
        lines.append(f"| {tier} | {v['n']} | {_pct(v['score'])} |")
    lines += ["", "## By section", "", "| section | cases | score | precision | recall |",
              "|---|---:|---:|---:|---:|"]
    for sec, v in s["per_section"].items():
        lines.append(f"| {sec} | {v['n']} | {_pct(v['score'])} | {_pct(v['precision'])} | {_pct(v['recall'])} |")
    lines += ["", "## Cases", ""]
    for r in card["cases"]:
        head = f"### {r['case_id']} - {_pct(r.get('score'))}"
        lines += [head, ""]
        if r.get("error"):
            lines += [f"ERROR: {r['error']}", ""]
            continue
        for ev in r.get("safety", []):
            lines.append(f"- **{ev['type'].replace('_', ' ')}**: {ev['detail']}")
        for sec, v in r["sections"].items():
            bits = [f"{sec}: {_pct(v['score'])}"]
            if v.get("detail"):
                bits.append(v["detail"])
            if v.get("missed_required"):
                bits.append("missed: " + "; ".join(v["missed_required"]))
            if v.get("unmatched_entries"):
                bits.append("unmatched: " + "; ".join(e[:60] for e in v["unmatched_entries"]))
            lines.append("- " + " — ".join(bits))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="validate cases and quotes only")
    ap.add_argument("--rescore", metavar="RUN_DIR", help="score saved responses without calling the API")
    ap.add_argument("--run-id", help="name of the run directory (default: timestamp)")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--tier", help="comma-separated tiers (mts_cell, management, ktas, mimic, published)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per case, including queue wait")
    a = ap.parse_args(argv)

    only = set(a.only.split(",")) if a.only else None
    tiers = set(a.tier.split(",")) if a.tier else None
    cases = H.load_cases(only=only, tiers=tiers)
    if not cases:
        print("No cases selected.")
        return 1
    if a.verify:
        return verify(cases)

    if a.rescore:
        run_dir = Path(a.rescore)
        if not run_dir.is_absolute():
            run_dir = BACKEND / run_dir
    else:
        run_dir = RUNS / (a.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
        run_cases(cases, run_dir, a.base_url.rstrip("/"), a.timeout)
    card = score_run(cases, run_dir)
    s = card["summary"]
    print(f"\nOverall {_pct(s['overall'])} over {s['scored']} cases; safety {s['safety']}")
    print(f"Scorecard: {run_dir / 'scorecard.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
