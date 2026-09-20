"""Command line entry point.

Two jobs: run a benchmark's tasks through a harness, and re-score stored answers
without spending anything. Both are driven by a benchmark key, so adding a benchmark
never means adding a CLI flag.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .harness.claude_docker import SCENARIOS, ClaudeDockerHarness, HarnessConfig
from .judge import Judge, VerdictCache
from .run import run_benchmark
from .registry import load_benchmark


def load_env_file(path) -> dict:
    """Load KEY=VALUE lines into os.environ without overwriting existing values."""
    path = Path(path)
    if not path.is_file():
        return {}
    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def build_parser():
    parser = argparse.ArgumentParser(prog="benchkit", description="Run and grade benchmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("benchmarks", help="list available benchmarks")
    listing.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="run a benchmark through a harness")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--image", default="benchkit-agent")
    run.add_argument("--mcp-url", default=None,
                     help="override the endpoint a corpus condition reaches its "
                          "deployment at; default is the benchmark's own")
    run.add_argument("--corpus-api-url", default=None,
                     help="override the API base handed to a command-line corpus client")
    run.add_argument("--corpus-web-url", default=None,
                     help="override the web base handed to a command-line corpus client")
    run.add_argument("--env-file", type=Path, default=None)
    run.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    run.add_argument("--base-url", default=os.environ.get("BENCHKIT_BASE_URL"),
                     help="model provider base URL (Anthropic-compatible); "
                          "default is the CLI's own, i.e. Anthropic")
    run.add_argument("--model", default=None)
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--max-budget-usd", type=float, default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--task-id", action="append", default=None)
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--workers", type=int, default=3,
                     help="how many tasks to run at once (each in its own container)")
    # Every returned program is asked about legitimacy, relevance and its citation on
    # top of its factual claims -- roughly seven verdicts per program -- so the per-task
    # pool is sized to keep judging off the critical path. The process-wide cap still
    # bounds the provider.
    run.add_argument("--judge-workers", type=int, default=16,
                     help="how many claim verdicts to keep in flight per task")
    run.add_argument("--no-fetch", action="store_true",
                        help="score against the frozen page snapshot in --source-cache; "
                             "a cited page that is not in it counts as unreadable rather "
                             "than being fetched, which is what makes a re-scoring exact")
    run.add_argument("--verdict-cache", type=Path,
                        default=Path(os.environ["BENCHKIT_VERDICT_CACHE"]) if os.environ.get("BENCHKIT_VERDICT_CACHE") else None,
                        help="directory of cached judge verdicts; makes a re-scoring of "
                             "the same answers reproducible instead of re-sampling the judge")
    run.add_argument("--judge-base-url", default=os.environ.get("BENCHKIT_JUDGE_BASE_URL"),
                        help="OpenAI-compatible base URL for the judge, when the same "
                             "model is reachable on more than one route")
    run.add_argument("--judge-api-key-env", default=os.environ.get("BENCHKIT_JUDGE_API_KEY_ENV"),
                        help="environment variable holding the judge's API key")
    run.add_argument("--max-inflight", type=int, default=32,
                     help="process-wide cap on simultaneous model calls (429 protection)")
    run.add_argument("--source-cache", type=Path, default=None,
                     help="where to cache the pages claims are verified against; one "
                          "directory per comparison round judges every condition "
                          "against the same fetched bytes")
    run.add_argument("--keep-transcript", action="store_true",
                     help="store the full per-message transcript in raw/ (large, but the "
                          "only way to audit tool-call sequences)")
    run.add_argument("--question-suffix", default=None)
    run.add_argument("--judge-model", default=os.environ.get("BENCHKIT_JUDGE_MODEL"),
                     help="LiteLLM model for per-claim verification")
    run.add_argument("--judge-effort", default=os.environ.get("BENCHKIT_JUDGE_EFFORT"),
                     help="reasoning effort for the judge model, e.g. low|medium|high")
    run.add_argument("--normalizer-model", default=None, help="LiteLLM model to extract facts from prose")

    regrade = sub.add_parser("regrade", help="re-score stored answers without rerunning")
    regrade.add_argument("--benchmark", required=True)
    regrade.add_argument("--run-dir", type=Path, required=True)
    regrade.add_argument("--env-file", type=Path, default=None)
    regrade.add_argument("--judge-model", default=os.environ.get("BENCHKIT_JUDGE_MODEL"))
    regrade.add_argument("--judge-effort", default=os.environ.get("BENCHKIT_JUDGE_EFFORT"))
    regrade.add_argument("--reparse", action="store_true",
                         help="re-extract records from stored answer text before scoring")
    regrade.add_argument("--no-fetch", action="store_true",
                        help="score against the frozen page snapshot in --source-cache; "
                             "a cited page that is not in it counts as unreadable rather "
                             "than being fetched, which is what makes a re-scoring exact")
    regrade.add_argument("--verdict-cache", type=Path,
                        default=Path(os.environ["BENCHKIT_VERDICT_CACHE"]) if os.environ.get("BENCHKIT_VERDICT_CACHE") else None,
                        help="directory of cached judge verdicts; makes a re-scoring of "
                             "the same answers reproducible instead of re-sampling the judge")
    regrade.add_argument("--judge-base-url", default=os.environ.get("BENCHKIT_JUDGE_BASE_URL"),
                        help="OpenAI-compatible base URL for the judge, when the same "
                             "model is reachable on more than one route")
    regrade.add_argument("--judge-api-key-env", default=os.environ.get("BENCHKIT_JUDGE_API_KEY_ENV"),
                        help="environment variable holding the judge's API key")
    regrade.add_argument("--max-inflight", type=int, default=32)
    regrade.add_argument("--source-cache", type=Path, default=None)
    regrade.add_argument("--judge-workers", type=int, default=16,
                         help="how many claim verdicts to keep in flight per task")
    return parser


def _env_value(name):
    """The key named by a `--*-api-key-env` flag, read after `--env-file` has loaded."""
    return os.environ.get(name) if name else None


def _env_file_arg(argv) -> Path:
    """Which env file to load, before the parser's defaults are evaluated.

    `build_parser` reads several defaults out of the environment, so the env file has
    to be in it *first* — otherwise `BENCHKIT_JUDGE_MODEL=...` in `.env` is invisible
    and the flag silently falls back to nothing. Only `--env-file` is read here; the
    real parse happens afterwards.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(argv):
        if token == "--env-file" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if token.startswith("--env-file="):
            return Path(token.split("=", 1)[1])
    return Path(".env")


def main(argv=None) -> int:
    # The env file must be in os.environ before the parser is built, because several
    # flags take their default from it.
    load_env_file(_env_file_arg(argv))
    args = build_parser().parse_args(argv)
    if args.command == "benchmarks":
        from .registry import available
        keys = available()
        print(json.dumps(keys, indent=2) if args.json else "\n".join(keys))
        return 0
    # The limiter is a module-level singleton shared by every client in the process, so
    # it is configured before the first import rather than mutated afterwards.
    if args.command in ("run", "regrade"):
        os.environ["BENCHKIT_MAX_INFLIGHT"] = str(max(1, args.max_inflight))
    from .llm import LiteLLMClient
    benchmark = load_benchmark(args.benchmark)

    if hasattr(benchmark, "judge_workers"):
        benchmark.judge_workers = max(1, args.judge_workers)
    if getattr(args, "no_fetch", False):
        benchmark.source_fetch = False

    if args.command == "regrade":
        from .run import regrade_stored
        judge = (Judge(LiteLLMClient(args.judge_model, reasoning_effort=args.judge_effort,
                                     api_key=_env_value(args.judge_api_key_env),
                                     api_base=args.judge_base_url),
                       cache=VerdictCache(args.verdict_cache))
                 if args.judge_model else None)
        _, summary = regrade_stored(benchmark, args.run_dir, judge=judge,
                                    reparse=args.reparse, cache_dir=args.source_cache)
        print(json.dumps(summary, indent=2))
        return 0
    # A benchmark specializes the shared scenarios with its own subject wording and
    # knowledge base. `scenario_for` keeps that vocabulary out of the harness, and a
    # benchmark that needs nothing special can simply not implement `scenarios`.
    build_scenario = getattr(benchmark, "scenarios", None)
    scenario = build_scenario()[args.scenario] if callable(build_scenario) else SCENARIOS[args.scenario]
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"set {args.api_key_env} (environment or .env)")
    if not shutil.which("docker"):
        raise SystemExit("docker is required to run a harness")

    # Endpoint resolution order, most specific first: the flag, then the environment,
    # then whatever the benchmark declares. The engine holds no vendor's address, so a
    # benchmark that reaches a hosted corpus says where that corpus lives and this
    # module stays general.
    declared = getattr(benchmark, "endpoints", None)
    declared = declared() if callable(declared) else {}
    mcp_url = (args.mcp_url or os.environ.get("BENCHKIT_MCP_URL")
               or declared.get("mcp_url"))
    config = HarnessConfig(
        image=args.image, api_key=api_key, base_url=args.base_url,
        model=args.model or os.environ.get("BENCHKIT_MODEL") or None,
        timeout=args.timeout, max_budget_usd=args.max_budget_usd, mcp_url=mcp_url,
        corpus_api_url=args.corpus_api_url or declared.get("corpus_api_url"),
        corpus_web_url=args.corpus_web_url or declared.get("corpus_web_url"),
        keep_transcript=args.keep_transcript,
    )
    harness = ClaudeDockerHarness(scenario, config, Path(args.run_dir) / "raw")
    judge = (Judge(LiteLLMClient(args.judge_model, reasoning_effort=args.judge_effort,
                                 api_key=_env_value(args.judge_api_key_env),
                                 api_base=args.judge_base_url),
                   cache=VerdictCache(args.verdict_cache))
             if args.judge_model else None)
    normalizer = LiteLLMClient(args.normalizer_model) if args.normalizer_model else None

    rows, summary = run_benchmark(
        benchmark, harness, args.run_dir, judge=judge, normalizer=normalizer,
        limit=args.limit, only=args.task_id, retries=args.retries,
        question_suffix=args.question_suffix, workers=max(1, args.workers),
        cache_dir=args.source_cache,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
