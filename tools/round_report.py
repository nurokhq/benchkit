"""Aggregate a round's runs by condition, with the spread across repeats.

One run per condition cannot separate a capability difference from a run-to-run one,
and this benchmark has already reported both orderings from single runs. This reads
whatever run directories exist and reports each condition's mean and range, so a
margin can be read against the noise it has to clear.

    python tools/round_report.py results/<round>-*
"""
import json
import pathlib
import sys
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parent.parent


def condition_of(run_dir: pathlib.Path) -> str:
    """`r01-kb-mcp.rep2` -> `r01 / kb-mcp / deepseek-flash`."""
    name = run_dir.name
    scenario = json.load(open(run_dir / "manifest.json")).get("scenario")
    model = json.load(open(run_dir / "manifest.json")).get("model") or "?"
    label = {"web-agent": "web-agent", "kb-mcp": "kb-mcp", "kb-cli": "kb-cli"}.get(
        scenario, scenario or name)
    # The round is part of the identity, not decoration. Two rounds are not comparable
    # unless they were scored the same way, and pooling them into one mean reports an
    # average of two different measurements. Runs from different rounds are therefore
    # never grouped, however similar their condition labels look.
    round_name = name.split("-", 1)[0]
    return f"{round_name} / {label} / {model}"


def base_rate_cost(run_dir: pathlib.Path, summary: dict, rows: list[dict]) -> float:
    """The agent cost with any time-of-day price multiplier divided out.

    A vendor that doubles its price for part of the day makes a raw total comparable
    only with another total billed at the same rate. Two rounds of this benchmark were
    compared at face value when one had run inside DeepSeek's peak window and the other
    outside it, which read as a 56% saving that was the multiplier. Runs made after the
    multiplier started being recorded carry it per task; older ones are derived from the
    manifest's start time, which is the only record of the hour they were billed at.
    """
    recorded = [(r.get("run_metrics") or {}).get("extra") or {} for r in rows]
    multipliers = [e.get("price_multiplier") for e in recorded]
    if not any(m for m in multipliers):
        try:
            sys.path.insert(0, str(REPO / "src"))
            from benchkit import pricing
            manifest = json.load(open(run_dir / "manifest.json"))
            started = datetime.fromisoformat(manifest["started_at"])
            multiplier = 2.0 if pricing.is_peak(manifest.get("model"), started) else 1.0
        except Exception:
            multiplier = 1.0
        multipliers = [multiplier] * len(rows)
    total = 0.0
    for row, m in zip(rows, multipliers):
        cost = (row.get("run_metrics") or {}).get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total += cost / (m or 1.0)
    return total


def main() -> None:
    dirs = [pathlib.Path(p) for p in sys.argv[1:]] or sorted(
        d for d in (REPO / "results").glob("r*") if d.is_dir())
    groups: dict[str, list[dict]] = {}
    for run_dir in dirs:
        summary_path = run_dir / "rescored-summary.json"
        if not summary_path.is_file():
            summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.load(open(summary_path))
        detail = run_dir / "rescored.jsonl"
        if not detail.is_file():
            detail = run_dir / "responses.jsonl"
        rows = [json.loads(line) for line in open(detail)]
        groups.setdefault(condition_of(run_dir), []).append({
            "run": run_dir.name,
            "score": summary["score_total"],
            "cost": summary.get("total_cost_usd") or 0.0,
            "base_cost": base_rate_cost(run_dir, summary, rows),
            "judge_cost": summary.get("judge_cost_usd") or 0.0,
            "wall": summary.get("mean_wall_seconds") or 0.0,
            "web": summary.get("web_calls_total") or 0,
            "schema_failures": sum(r["metrics"].get("schema_failures", 0) for r in rows),
            "programs": sum(r["metrics"].get("programs_returned", 0) for r in rows),
            "answered": summary.get("answered_count"),
        })

    print(f"{'condition':34} {'n':>2} {'score':>16} {'cost':>7} {'@base':>7} "
          f"{'judge':>7} {'wall':>6} {'web':>5} {'progs':>6}")
    for label, runs in sorted(groups.items(), key=lambda kv: -max(r["score"] for r in kv[1])):
        scores = [r["score"] for r in runs]
        mean = sum(scores) / len(scores)
        print(f"{label:34} {len(runs):>2} {mean:>10.1f} "
              f"{'(' + f'{min(scores):.0f}-{max(scores):.0f}' + ')':>16} "
              f"{max(scores) - min(scores):>7.1f} "
              f"{sum(r['cost'] for r in runs) / len(runs):>7.3f} "
              f"{sum(r['base_cost'] for r in runs) / len(runs):>7.3f} "
              f"{sum(r['judge_cost'] for r in runs) / len(runs):>7.3f} "
              f"{sum(r['wall'] for r in runs) / len(runs):>7.0f} "
              f"{sum(r['web'] for r in runs) / len(runs):>5.0f} "
              f"{sum(r['programs'] for r in runs) / len(runs):>6.0f} "
              f"{sum(r['schema_failures'] for r in runs):>6}")
    print()
    for label, runs in sorted(groups.items()):
        if len(runs) > 1:
            print(f"  {label}: " + "  ".join(f"{r['run'].split('.')[-1]}={r['score']:.1f}"
                                             for r in sorted(runs, key=lambda r: r["run"])))


if __name__ == "__main__":
    main()
