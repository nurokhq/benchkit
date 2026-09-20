#!/usr/bin/env python3
"""Recompute a run's ranking under different rubric weights.

The weights in a scoring rubric are a statement about what a reader is owed, not a
measurement, and a ranking that only holds at one particular set of them is not a
finding about the agents. Every component of a stored score is written into
`responses.jsonl`, so the ranking can be recomputed for any weights without paying
for the agents again -- which is the only practical way to check that a conclusion
is not an artefact of the arithmetic.

Usage:

    python3 tools/reweight.py results/<run>
    python3 tools/reweight.py results/<run> --legit 0 --claims 3
    python3 tools/reweight.py results/<run> --sweep legit=0,0.5,1,2

A run scored before a component existed reports `None` for it; those runs are listed
as unusable rather than silently scored as zero.
"""

import argparse
import json
from pathlib import Path

#: The metric keys whose values sum to a run's score, exactly. `penalty` is already
#: negative, and `relevance` is deliberately absent: it gates an item rather than
#: scoring, so it has no total to re-weight. A benchmark that adds a component must
#: add it here and keep the sum equal to the score -- a test enforces that.
COMPONENTS = ("legit", "claims", "reputation", "penalty")

DEFAULT_WEIGHTS = {name: 1.0 for name in COMPONENTS}


def load_run(run_dir: Path):
    """(name, per-task metrics) for one run directory, or None when unusable."""
    # A regraded run is the one worth re-weighting: it holds the current parser and
    # rubric applied to the same answers. Prefer it when it is there.
    path = run_dir / "rescored.jsonl"
    if not path.is_file():
        path = run_dir / "responses.jsonl"
    if not path.is_file():
        return None
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics") or {}
        if any(metrics.get(name) is None for name in COMPONENTS):
            return None
        tasks.append(metrics)
    return (run_dir.name, tasks) if tasks else None


def score(tasks, weights) -> float:
    """Total score under `weights`, summed over the run's tasks."""
    return sum(sum(metrics[name] * weights[name] for name in COMPONENTS) for metrics in tasks)


def ranking(runs, weights):
    scored = [(name, score(tasks, weights)) for name, tasks in runs]
    return sorted(scored, key=lambda pair: -pair[1])


def parse_weights(argv) -> dict:
    weights = dict(DEFAULT_WEIGHTS)
    for name in COMPONENTS:
        value = getattr(argv, name)
        if value is not None:
            weights[name] = value
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    for name in COMPONENTS:
        parser.add_argument(f"--{name}", type=float, default=None,
                            help=f"weight for the {name} component (default 1.0)")
    parser.add_argument("--sweep", metavar="COMPONENT=V1,V2,...",
                        help="re-rank across a range of values for one component")
    args = parser.parse_args()

    runs, unusable = [], []
    for run_dir in args.run_dirs:
        loaded = load_run(run_dir)
        (runs if loaded else unusable).append(loaded or run_dir.name)
    for name in unusable:
        print(f"skipped (missing metrics or failed tasks): {name}")
    if not runs:
        return 1

    if args.sweep:
        component, _, values = args.sweep.partition("=")
        if component not in COMPONENTS or not values:
            parser.error(f"--sweep needs COMPONENT= V1,V2,... with COMPONENT in {COMPONENTS}")
        print(f"sweeping {component}\n")
        for value in values.split(","):
            weights = parse_weights(args)
            weights[component] = float(value)
            order = "  ".join(f"{name}={total:.1f}" for name, total in ranking(runs, weights))
            print(f"  {component}={value:<6} {order}")
        return 0

    weights = parse_weights(args)
    print("weights: " + ", ".join(f"{name}={weights[name]:g}" for name in COMPONENTS) + "\n")
    baseline = dict(ranking(runs, DEFAULT_WEIGHTS))
    for name, total in ranking(runs, weights):
        delta = total - baseline[name]
        marker = "" if abs(delta) < 1e-9 else f"   ({delta:+.1f} vs equal weights)"
        print(f"  {name:<20} {total:8.1f}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
