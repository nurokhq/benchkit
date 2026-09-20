"""Pull the numbers a plain-language write-up needs out of a round's runs.

    python3 tools/headline_metrics.py results/<run> ...

Reports, per condition, the things a reader who is not going to read `responses.jsonl`
still needs: how much was returned, how much of it survived checking, what a task cost,
how long it took, and how much of the open web the agent had to go and read. Every
figure is a mean over the runs given, with the range kept beside it, because a single
run of this benchmark has never been enough to rank anything.
"""
import json
import pathlib
import sys

LABELS = {"kb-mcp": "corpus MCP", "kb-cli": "corpus CLI", "web-agent": "web agent"}


def load(run_dir: pathlib.Path) -> dict:
    manifest = json.load(open(run_dir / "manifest.json"))
    detail = run_dir / "rescored.jsonl"
    if not detail.is_file():
        detail = run_dir / "responses.jsonl"
    summary_path = run_dir / "rescored-summary.json"
    if not summary_path.is_file():
        summary_path = run_dir / "summary.json"
    rows = [json.loads(line) for line in open(detail)]
    summary = json.load(open(summary_path))
    m = [r["metrics"] for r in rows]
    rm = [r.get("run_metrics") or {} for r in rows]

    def total(key):
        return sum(x.get(key) or 0 for x in m)

    fields = ["field_status", "field_deadline", "field_terms", "field_eligibility"]
    return {
        "run": run_dir.name,
        "condition": f"{LABELS.get(manifest.get('scenario'), manifest.get('scenario'))} / {manifest.get('model')}",
        "tasks": len(rows),
        "score": summary["score_total"],
        "score_per_task": summary["score_total"] / len(rows),
        "agent_cost": summary["total_cost_usd"],
        "judge_cost": summary.get("judge_cost_usd") or 0.0,
        "wall_mean": summary["mean_wall_seconds"],
        "wall_total": sum((r.get("wall_ms") or 0) for r in rm) / 1000,
        "web_calls": summary["web_calls_total"],
        "turns": sum(x.get("turns") or 0 for x in rm),
        "returned": total("programs_returned"),
        "recognised": total("programs_recognised"),
        "legit": total("legit"),
        "reputation": total("reputation"),
        "relevant": total("relevant"),
        "not_relevant": total("not_relevant"),
        "voided": total("items_voided"),
        "claims": total("claims"),
        "claims_judged": total("claims_judged"),
        "unverifiable": total("unverifiable_claims"),
        "unsourced": total("unsourced_records"),
        "penalty": total("penalty"),
        "absences": total("documented_absences"),
        "fields": {f: total(f) for f in fields},
        "sources": {
            k: sum((x.get("sources") or {}).get(k) or 0 for x in m)
            for k in ("cache", "fetched", "missing")
        },
        "schema_failures": total("schema_failures"),
        "judge_failures": total("judge_failures"),
        "tokens": {
            k: sum(((x.get("tokens") or {}).get(k) or 0) for x in rm)
            for k in ("input", "output", "cache_read", "cache_creation")
        },
    }


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def spread(values, fmt="{:.0f}"):
    if not values:
        return "-"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(mean(values))} ({fmt.format(min(values))}-{fmt.format(max(values))})"


def main() -> None:
    dirs = [pathlib.Path(p) for p in sys.argv[1:]]
    groups: dict[str, list[dict]] = {}
    for d in dirs:
        r = load(d)
        groups.setdefault(r["condition"], []).append(r)

    order = sorted(groups, key=lambda k: -mean([r["score"] for r in groups[k]]))
    print(f"{'condition':38} {'n':>2} {'score':>16} {'$/run':>16} {'wall s':>14} "
          f"{'web calls':>14} {'turns':>10}")
    for label in order:
        runs = groups[label]
        print(f"{label:38} {len(runs):>2} "
              f"{spread([r['score'] for r in runs], '{:.1f}'):>16} "
              f"{spread([r['agent_cost'] for r in runs], '{:.3f}'):>16} "
              f"{spread([r['wall_mean'] for r in runs]):>14} "
              f"{spread([r['web_calls'] for r in runs]):>14} "
              f"{spread([r['turns'] for r in runs]):>10}")

    print("\n--- what the answers contained (summed over the round's tasks, mean per run) ---")
    print(f"{'condition':38} {'returned':>10} {'legit':>9} {'own page':>9} {'on topic':>9} "
          f"{'fields/rec':>11} {'no url':>7} {'claims':>8} {'readable':>9}")
    for label in order:
        runs = groups[label]
        returned = mean([r["returned"] for r in runs])
        print(f"{label:38} {returned:>10.0f} "
              f"{mean([r['legit'] for r in runs]):>4.0f} ({mean([r['legit'] / r['returned'] for r in runs]):>4.0%}) "
              f"{mean([r['reputation'] / r['returned'] for r in runs]):>9.0%} "
              f"{mean([(r['relevant'] - r['not_relevant']) / r['returned'] for r in runs]):>9.0%} "
              f"{mean([sum(r['fields'].values()) / r['returned'] for r in runs]):>11.1f} "
              f"{mean([r['unsourced'] for r in runs]):>7.1f} "
              f"{mean([r['claims_judged'] for r in runs]):>8.0f} "
              f"{mean([r['sources']['cache'] + r['sources']['fetched'] for r in runs]):>9.0f}")
    print("  'legit' = judge read the cited page and found an applyable program.")
    print("  'own page' = cited page is published by the program's operator, not a listicle.")
    print("  'on topic' = share of records the relevance gate did not contradict.")
    print("  'fields/rec' = of 4 asked-for facts, how many the answer actually stated.")
    print("  'readable' = cited pages the checker could open and read.")

    print("\n--- per run ---")
    for label in order:
        for r in sorted(groups[label], key=lambda r: r["run"]):
            print(f"  {r['run']:22} score={r['score']:6.1f} ${r['agent_cost']:.3f} "
                  f"wall={r['wall_mean']:5.0f}s web={r['web_calls']:>4} "
                  f"returned={r['returned']:>3} legit={r['legit']:>5.1f} "
                  f"claims={r['claims_judged']:>4} turns={r['turns']:>4}")


if __name__ == "__main__":
    main()
