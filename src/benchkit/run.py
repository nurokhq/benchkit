"""Drive a benchmark's tasks through a harness and score what comes back.

The engine is benchmark-agnostic: it moves tasks, collects predictions, asks the
benchmark to score each one, and aggregates. It never inspects identifiers, gold
shapes or answer semantics — that is what lets a new benchmark be added without
touching this file.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from .case import Prediction, RunMetrics
from .harness.claude_docker import WEB_TOOLS
from .normalize import normalize_answer
from .pricing import cost_usd


def read_jsonl(path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from None
    return rows


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_collapsed(metrics: dict) -> int:
    """How many verdicts were lost, when that is enough to invalidate the score.

    A failed judge call marks its claim unverifiable, and unverifiable claims are
    *excluded* from the score rather than zeroed -- the honest treatment for a page
    that could not be read. But when most of a task's verdicts fail, the exclusions
    stop describing the answer and start describing the grading run, and the number
    that comes out is a score with the answer removed from it. One round reported 48.5
    for a condition whose third task had judged zero of 152 claims and scored -6.0;
    read back, that is indistinguishable from a bad answer, and it was reported as
    run-to-run variance for a day.

    Returns the failure count when the run should stop, else 0. A healthy task fails a
    handful out of a hundred -- retries absorb those.
    """
    failed = int(metrics.get("judge_failures") or 0)
    judged = int(metrics.get("claims_judged") or 0)
    if failed < 20:
        return 0
    return failed if failed > judged else 0


def as_prediction(row, task_id: str | None = None) -> Prediction:
    """Rebuild a Prediction from a stored responses row, for regrading without a rerun.

    The schema-validated payload is restored as well as the text: it is the answer a
    benchmark grades, so a re-scoring that lost it would score zero for a reason that
    has nothing to do with the answer.
    """
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, RunMetrics):
        metrics = RunMetrics(
            wall_ms=metrics.get("wall_ms"),
            turns=metrics.get("turns") or metrics.get("num_turns"),
            cost_usd=metrics.get("cost_usd") or metrics.get("total_cost_usd"),
            tokens=metrics.get("tokens") or {
                "input": metrics.get("input_tokens"), "output": metrics.get("output_tokens"),
                "cache_read": metrics.get("cache_read_input_tokens"),
                "cache_creation": metrics.get("cache_creation_input_tokens"),
            },
            tool_use=metrics.get("tool_use") or {},
            returncode=metrics.get("returncode"),
            timed_out=bool(metrics.get("timed_out")),
            extra={k: v for k, v in metrics.items() if k in {"duration_ms", "stop_reason", "workspace_files"}},
        )
    return Prediction(
        task_id=task_id or row.get("task_id") or row.get("case_id"),
        answer_text=row.get("answer_text") or "",
        normalized=row.get("normalized") or {},
        structured=row.get("structured"),
        raw=row.get("raw"),
        metrics=metrics,
        status=row.get("status") or "ok",
    )


def write_manifest(run_dir, benchmark, harness, judge, tasks) -> dict:
    """Record everything that could change a score, beside the scores.

    A score is only interpretable against the versions that produced it. A corpus that
    is not the one believed under test, a reader changed between two runs, a provider
    swapped for another, a different judge — each of these moves a number without
    leaving a trace, and a comparison across one is not a comparison.

    Hashing the prompt and the benchmark source catches the two that are hardest to see
    afterwards: an edited instruction, and a scoring change.
    """
    scenario = getattr(harness, "scenario", None)
    config = getattr(harness, "config", None)
    judge_client = getattr(judge, "client", None)
    source = Path(getattr(benchmark, "directory", Path("."))) / "benchmark.py"
    prompt = getattr(scenario, "system_prompt", "") or ""
    manifest = {
        "benchmark": getattr(benchmark, "key", None),
        "scenario": getattr(scenario, "key", None),
        "model": getattr(config, "model", None),
        "provider": getattr(config, "base_url", None) or "anthropic (default)",
        "image": getattr(config, "image", None),
        "judge_model": getattr(judge_client, "model", None),
        "judge_effort": getattr(judge_client, "reasoning_effort", None),
        "judge_temperature": getattr(judge_client, "effective_temperature", None),
        "system_prompt_sha256": sha256(prompt.encode()).hexdigest()[:16],
        "system_prompt_chars": len(prompt),
        "benchmark_source_sha256": (sha256(source.read_bytes()).hexdigest()[:16]
                                    if source.is_file() else None),
        "parser_version": getattr(benchmark, "parser_version", None),
        "corpus_revision": getattr(benchmark, "corpus_revision", None),
        "rubric": rubric_constants(benchmark),
        "task_ids": [task.get("task_id") for task in tasks],
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (Path(run_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


#: The scoring constants a benchmark exposes, recorded with the run so a stored score
#: can be recomputed under different weights without guessing what produced it.
_RUBRIC_NAMES = ("CLAIM_WEIGHTS", "LEGIT_WEIGHT", "RELEVANCE_WEIGHT", "REPUTATION_WEIGHT",
                 "NOT_PUBLISHED_CREDIT", "NO_CITATION_PENALTY", "DEAD_LINK_PENALTY")


def rubric_constants(benchmark) -> dict:
    module = sys.modules.get(type(benchmark).__module__)
    if module is None:
        return {}
    return {name: getattr(module, name) for name in _RUBRIC_NAMES
            if hasattr(module, name)}


def run_benchmark(benchmark, harness, run_dir, judge=None, normalizer=None,
                  limit=None, only=None, retries=0, question_suffix=None,
                  workers=1, cache_dir=None) -> tuple[list[dict], list[dict]]:
    """Run every task, scoring as it goes. Returns (rows, summary).

    Tasks are independent, so `workers` runs several at once — each in its own
    container with its own scratch directory, so there is no shared state to corrupt.
    Rows are written under a lock, which keeps `responses.jsonl` well formed without
    making the order of completion significant.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = benchmark.load_tasks()
    write_manifest(run_dir, benchmark, harness, judge, tasks)
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if t["task_id"] in wanted]
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    rows: list[dict] = []
    lock = threading.Lock()
    total = len(tasks)

    def write_rows():
        with lock:
            write_jsonl(run_dir / "responses.jsonl", sorted(rows, key=lambda r: r["task_id"]))

    def run_one(index_task):
        index, task = index_task
        prediction = None
        for attempt in range(retries + 1):
            prediction = harness.run(task, question_suffix=question_suffix)
            if prediction.status in ("ok", "auth_error", "fatal_error"):
                break
            print(f"    {task['task_id']} retry {attempt + 1}: status={prediction.status}", flush=True)
        if prediction.status == "auth_error":
            raise SystemExit("harness reported an authentication error; fix credentials and rerun")
        if prediction.status == "fatal_error":
            # Stopping beats continuing: every remaining task would fail the same way,
            # and a run that writes zero scores for them is indistinguishable from a
            # condition that answered and scored nothing.
            raise SystemExit(
                "harness reported an API error that ends the run: "
                f"{(prediction.answer_text or '').strip()[:200]}")

        if normalizer is not None:
            extracted = normalizer.extract(task, prediction)
            if extracted is not None:
                prediction.normalized = normalize_answer(
                    {"structured": extracted, "answer_text": prediction.answer_text})
        if not prediction.normalized:
            prediction.normalized = normalize_answer(prediction)

        result = benchmark.score(task, prediction, judge=judge, cache_dir=cache_dir)
        broken = judge_collapsed(result.metrics)
        if broken:
            raise SystemExit(
                f"grading {task['task_id']} lost {broken} verdicts to judge failures; "
                "a score computed from that is not a result. First errors: "
                f"{result.metrics.get('judge_error_samples')}")
        row = {
            "task_id": task["task_id"],
            "status": prediction.status,
            # A row counts as answered only when the harness finished and produced
            # text; a budget stop or timeout is a spend without a result.
            "error": None if prediction.status == "ok" and prediction.answer_text else prediction.status,
            "score": result.score,
            "metrics": result.metrics,
            "notes": result.notes,
            "answer_text": prediction.answer_text,
            "normalized": prediction.normalized,
            # The answer as validated data. Stored because it is what grading reads:
            # a regrade that had to reconstruct it from the prose would be re-running
            # the very inference the declared schema exists to remove.
            "structured": prediction.structured,
            "run_metrics": prediction.metrics.__dict__,
        }
        with lock:
            rows.append(row)
        write_rows()
        print(f"[{index}/{total}] {task['task_id']} score={result.score:+.4f} "
              f"cost=${prediction.metrics.cost_usd}", flush=True)
        return row

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_one, list(enumerate(tasks, 1))))
    else:
        for item in enumerate(tasks, 1):
            run_one(item)

    summary = summarize(rows, benchmark)
    record_judge_accounting(summary, judge)
    write_json(run_dir / "summary.json", summary)
    return rows, summary


def record_judge_accounting(summary: dict, judge) -> None:
    """Add what the judging cost to a summary, in place.

    Three numbers: how much of the judging was answered from cache rather than bought
    again, the tokens the provider billed for it, and what those tokens cost. Without
    them a run reports only what the agent spent, and the verifier -- which costs about
    as much as the agent, and more than the cheapest condition costs altogether -- does
    not appear in any total. It was added to the regrade path first, so a round's own
    summaries carried no judging cost at all and the only way to see it was to
    re-score answers that had already been scored.
    """
    if judge is None:
        return
    if getattr(judge, "cache_stats", None):
        summary["judge_cache"] = dict(judge.cache_stats)
    client = getattr(judge, "client", None)
    usage = getattr(client, "usage", None)
    if not usage:
        return
    summary["judge_usage"] = dict(usage)
    cost = cost_usd(getattr(client, "model", None),
                    {"input": usage.get("prompt_tokens", 0),
                     "output": usage.get("completion_tokens", 0)})
    if cost is not None:
        summary["judge_cost_usd"] = round(cost, 6)


def regrade_stored(benchmark, run_dir, judge=None, reparse=False, cache_dir=None):
    """Re-score answers already on disk, without calling a harness again.

    Answers are the expensive part, so a scoring change should never require a rerun.
    With `reparse`, a benchmark that can re-extract records from stored answer text is
    asked to do so first, which is what makes a parser or rubric fix regradeable
    without re-querying the agent.
    """
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "responses.jsonl")
    task_by_id = {task["task_id"]: task for task in benchmark.load_tasks()}
    rescored = []
    for row in rows:
        task = task_by_id.get(row["task_id"])
        if task is None:
            continue
        prediction = as_prediction(row)
        if reparse and hasattr(benchmark, "reparse"):
            prediction.normalized = benchmark.reparse(prediction)
        result = benchmark.score(task, prediction, judge=judge, cache_dir=cache_dir)
        rescored.append({
            "task_id": row["task_id"], "status": row.get("status"),
            "score": result.score, "metrics": result.metrics, "notes": result.notes,
            "answer_text": row.get("answer_text"), "normalized": row.get("normalized"),
            "structured": row.get("structured"),
            "run_metrics": row.get("run_metrics") or {},
        })
    summary = summarize(rescored, benchmark)
    # Recorded so a reader can tell a fresh scoring from a cached one. A regrade whose
    # verdicts were all hits replayed the verdict the first scoring reached, which is
    # what makes two scorings of the same answers comparable; one that missed is a new
    # judgement.
    record_judge_accounting(summary, judge)
    write_jsonl(run_dir / "rescored.jsonl", rescored)
    write_json(run_dir / "rescored-summary.json", summary)
    return rescored, summary


def summarize(rows, benchmark=None) -> dict:
    """Aggregate scores plus what they cost.

    Cost, latency and efficiency describe the tasks that actually produced an answer,
    so they ignore rows that failed or reported nothing; a timeout has no meaningful
    per-task cost and must not dilute the ones that ran. Scores still average over
    every row, because an unanswered task is a real zero that must not be dropped.
    """
    def present(values):
        return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]

    answered = [row for row in rows if not row.get("error")]
    scores = [row.get("score") for row in rows]
    costs = present([(row.get("run_metrics") or {}).get("cost_usd") for row in answered])
    walls = present([(row.get("run_metrics") or {}).get("wall_ms") for row in answered])
    spend = present([(row.get("run_metrics") or {}).get("cost_usd") for row in rows])
    total_cost = sum(spend)
    # How much the answer leaned on the live web. This is an outcome, not overhead: a
    # knowledge base earns its place by making these calls unnecessary, so the count
    # belongs next to the score rather than reconstructed later from a transcript.
    lookups = [(row.get("run_metrics") or {}).get("tool_use") or {} for row in answered]
    web_calls = sum(count for tools in lookups for name, count in tools.items() if name in WEB_TOOLS)
    # Verified quality per dollar: only answers that earned something can demonstrate
    # efficiency, so a failed attempt does not drag the ratio to zero.
    earned = sum(present(row.get("score") for row in answered))
    earned_cost = sum(present((row.get("run_metrics") or {}).get("cost_usd") for row in answered))
    # The same run costs twice as much at some hours, because the vendor doubles its
    # price for part of the day. A total is therefore only comparable with another total
    # billed at the same rate, and two rounds of this benchmark were compared at face
    # value when one had run inside the peak window and the other outside it -- reading
    # as a 56% saving that was the multiplier. This is what the run would have cost at
    # the base rate, so rounds can be compared like for like.
    at_base = 0.0
    for row in rows:
        metrics = row.get("run_metrics") or {}
        cost = metrics.get("cost_usd")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            continue
        multiplier = (metrics.get("extra") or {}).get("price_multiplier") or 1.0
        at_base += cost / multiplier
    return {
        "benchmark": getattr(benchmark, "key", None),
        "task_count": len(rows),
        "answered_count": len(answered),
        "score_total": sum(present(scores)),
        "score_mean": (sum(present(scores)) / len(rows)) if rows else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "total_cost_usd_at_base_rate": round(at_base, 6),
        "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
        "mean_wall_seconds": (sum(walls) / len(walls) / 1000) if walls else None,
        "web_calls_total": web_calls,
        "web_calls_mean": (web_calls / len(answered)) if answered else None,
        "score_per_dollar": (earned / earned_cost) if earned_cost else None,
        "verdicts": verdict_tally(rows),
        "signals": getattr(benchmark, "signals", []),
    }


def verdict_tally(rows) -> dict:
    """Every judge verdict in the run, counted under the question it answered.

    The headline score is a weighted sum, and a weighted sum cannot be read back into
    the one thing a reader asks first: how often was this answer actually right. This
    is that, kept beside the score. `claim.supported / (claim.supported +
    claim.contradicted)` is a plain accuracy over the facts the answer chose to state;
    the `not_stated` share says how often the cited page was silent, and
    `claim.unverifiable` says how often the page could not be read at all.
    """
    tally: dict[str, int] = {}
    for row in rows:
        for label, count in ((row.get("metrics") or {}).get("verdicts") or {}).items():
            if isinstance(count, int) and not isinstance(count, bool):
                tally[label] = tally.get(label, 0) + count
    return dict(sorted(tally.items()))
