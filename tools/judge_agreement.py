#!/usr/bin/env python3
"""Judge the same stored answers with two judges and report where they disagree.

All reported scores so far came from one judge, so the only way to tell whether a
candidate judge is a drop-in replacement is to put both in front of identical records
and compare verdict by verdict. An aggregate score agreement of 99% can still hide a
judge that flips `contradicted` to `not_stated` on every second record, and those two
verdicts are worth -1 and 0 -- the difference between a penalty and nothing.

Usage:
    python3 tools/judge_agreement.py <run-dir> [--judge-a M] [--judge-b M]
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchkit.cli import load_env_file          # noqa: E402
from benchkit.llm import LiteLLMClient          # noqa: E402
from benchkit.judge import Judge, verdict_score  # noqa: E402
from benchkit.run import as_prediction          # noqa: E402
from benchkit.sources import attach_sources     # noqa: E402


def load_benchmark_module():
    path = REPO / "benchmarks" / "us-startup-programs" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("usp_agreement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdicts_by_key(record_verdicts, records):
    """Flatten one judge's output into {(record_index, kind, field): verdict}."""
    flat = {}
    for index, outcome in enumerate(record_verdicts):
        for field, (verdict, _weight) in (outcome.get("claims") or {}).items():
            flat[(index, "claim", field)] = verdict["verdict"]
        if "reputation" in outcome:
            flat[(index, "reputation", None)] = outcome["reputation"][0]["verdict"]
        if "is_program" in outcome:
            flat[(index, "is_program", None)] = outcome["is_program"][0]["verdict"]
    return flat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--judge-a", default="openrouter/deepseek/deepseek-v4.1-flash",
                        help="the incumbent judge every reported score came from")
    parser.add_argument("--judge-b", default="openrouter/z-ai/glm-5.3-flash",
                        help="the candidate judge")
    parser.add_argument("--effort-b", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="tasks to compare")
    parser.add_argument("--adjudicate", default=None,
                        help="a third judge to settle the cases where A and B disagree")
    parser.add_argument("--verdicts-out", default=None,
                        help="persist every verdict so a re-analysis needs no new calls")
    args = parser.parse_args()

    load_env_file(REPO / ".env")
    module = load_benchmark_module()
    benchmark = module.StartupPrograms()
    source_text = benchmark._sources(benchmark.directory / ".cache")

    judge_a = Judge(LiteLLMClient(args.judge_a))
    judge_b = Judge(LiteLLMClient(args.judge_b, reasoning_effort=args.effort_b))
    print(f"A (incumbent): {args.judge_a}")
    print(f"B (candidate): {args.judge_b}  effort={args.effort_b}  temp={judge_b.client.effective_temperature}")

    rows = [json.loads(line) for line in open(Path(args.run_dir) / "responses.jsonl")]
    if args.limit:
        rows = rows[: args.limit]

    WEIGHTS = dict(module.CLAIM_WEIGHTS)
    WEIGHTS.setdefault("reputation", 0.5)
    WEIGHTS.setdefault("is_program", 1.0)

    def awarded(flat):
        """The judge-dependent part of the score, using the benchmark's own weights."""
        total = 0.0
        for (_index, _kind, field), verdict in flat.items():
            total += verdict_score({"verdict": verdict}) * WEIGHTS.get(field, 0.0)
        return total

    adjudicator = (Judge(LiteLLMClient(args.adjudicate, reasoning_effort="high"))
                   if args.adjudicate else None)
    if adjudicator:
        print(f"C (adjudicator): {args.adjudicate}  effort=high")
    evidence = {}
    both = {}
    score_a = score_b = 0.0
    per_kind = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        records = benchmark._records(as_prediction(row))
        if not records:
            continue
        attach_sources(records, source_text)
        started = time.time()
        va = verdicts_by_key(benchmark._verify_items(records, judge_a, args.workers), records)
        vb = verdicts_by_key(benchmark._verify_items(records, judge_b, args.workers), records)
        score_a += awarded(va)
        score_b += awarded(vb)
        for key in sorted(set(va) & set(vb), key=str):
            kind = key[1]
            both[(row["task_id"],) + key] = (va[key], vb[key])
            per_kind[kind][(va[key], vb[key])] += 1
            if adjudicator and va[key] != vb[key]:
                index, _kind, field = key
                record = records[index]
                claim = (f"{record.get('name')} {field}: {record.get(field)}" if field
                         else "This page is the program's official page.")
                try:
                    verdict = adjudicator.verify(record, claim,
                                                 source_text=record.get("source_text"),
                                                 source_url=record.get("url"))
                    evidence[(row["task_id"],) + key] = verdict.get("verdict")
                except Exception as exc:  # a failed adjudication is not a vote
                    evidence[(row["task_id"],) + key] = f"error: {str(exc)[:60]}"
        print(f"  {row['task_id'][:34]:36s} {len(va)} verdicts from A, {len(vb)} from B  ({time.time()-started:.0f}s)")

    total = len(both)
    agree = sum(1 for a, b in both.values() if a == b)
    print(f"\n=== agreement: {agree}/{total} = {agree/max(1,total)*100:.1f}%")
    print(f"=== judge-dependent score:  A(deepseek)={score_a:+.1f}   B(glm)={score_b:+.1f}"
          f"   delta={score_b-score_a:+.1f} ({(score_b-score_a)/max(1e-9,abs(score_a))*100:+.0f}%)")
    for kind, matrix in sorted(per_kind.items()):
        n = sum(matrix.values())
        same = sum(count for (a, b), count in matrix.items() if a == b)
        print(f"  {kind:12s} {same}/{n} = {same/max(1,n)*100:5.1f}%")
        for (a, b), count in sorted(matrix.items(), key=lambda kv: -kv[1]):
            if a != b:
                print(f"      A={a:<13} B={b:<13} {count}")

    if adjudicator:
        decided = {k: v for k, v in evidence.items() if v in ("supported", "contradicted", "not_stated")}
        a_right = sum(1 for k, v in decided.items() if v == both[k][0])
        b_right = sum(1 for k, v in decided.items() if v == both[k][1])
        print(f"\n=== adjudicated {len(decided)} contested verdicts with {args.adjudicate}")
        print(f"    A(deepseek) agreed with adjudicator: {a_right}/{len(decided)} = {a_right/max(1,len(decided))*100:.0f}%")
        print(f"    B(glm)      agreed with adjudicator: {b_right}/{len(decided)} = {b_right/max(1,len(decided))*100:.0f}%")
        tally = collections.Counter((both[k][0], both[k][1], v) for k, v in decided.items())
        for (a, b, c), count in sorted(tally.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      A={a:<13} B={b:<13} adjudicator={c:<13} {count}")

    if args.verdicts_out:
        Path(args.verdicts_out).write_text(json.dumps(
            {"both": {str(k): v for k, v in both.items()},
             "adjudicated": {str(k): v for k, v in evidence.items()}}, indent=1))
        print(f"\nverdicts written to {args.verdicts_out}")

    print("\n=== disagreements (first 15)")
    for key, (a, b) in sorted(both.items(), key=str):
        if a != b:
            task, index, kind, field = key
            print(f"  {task[:28]:30s} #{index:<3} {kind:11s} {str(field):12s} A={a:<13} B={b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
