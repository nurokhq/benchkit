"""Generic metric primitives.

These are the reusable pieces of scoring: set overlap, field comparison, table
equality, ranking quality. They know nothing about any benchmark's domain. A
benchmark picks the ones it needs and reports them under its own signal names.

Nothing here averages metrics together. Combining them is a benchmark's decision,
made in its own `score`, because only the benchmark knows what its numbers mean.
"""

import json
import math
import re


def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 over three counts.

    An empty gold set with an empty answer scores 1.0 rather than 0.0: answering
    "there are none" correctly is not a failure. An empty answer against a non-empty
    gold still scores 0.0.
    """
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall, (2 * precision * recall / (precision + recall) if precision + recall else 1.0)


def set_metrics(expected: set, actual: set, unresolved: int = 0) -> dict:
    """Overlap of two identity sets, treating unresolved mentions as false positives.

    `unresolved` counts answers that named something the benchmark could not resolve
    to an identity at all. They are wrong answers, not facts to discard, so they are
    added to `false_positive` and also broken out separately for diagnosis.
    """
    tp = len(expected & actual)
    fp = len(actual - expected) + unresolved
    fn = len(expected - actual)
    precision, recall, score = f1(tp, fp, fn)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "unresolved": unresolved, "precision": precision, "recall": recall, "f1": score,
        "exact_set": float(expected == actual and not unresolved),
    }


def label_key(value) -> str | None:
    """Normalise a label for comparison.

    YC-style sources spell the same label two ways — a displayed `Workflow
    Automation` and its URL slug `workflow-automation` — so separators and case are
    folded. This is the only normalisation applied to strings; it does not make
    different labels equal.
    """
    if not isinstance(value, str):
        return None
    return re.sub(r"[\s_-]+", " ", value).strip().casefold()


def values_match(expected, actual, compare: str | None = None) -> bool:
    """Compare one expected value against the answer's version of it.

    `compare="city"` reduces a location to its first component, because a source may
    display `San Francisco` where the record stores `San Francisco, CA, USA`. Lists
    compare as unordered sets, because label order carries no meaning.
    """
    if compare == "city":
        cut = lambda v: label_key(str(v).split(",")[0]) if isinstance(v, str) else v
        return cut(expected) == cut(actual)
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        key = lambda item: label_key(item) if isinstance(item, str) else json.dumps(item, sort_keys=True)
        return sorted(map(key, expected)) == sorted(map(key, actual))
    if isinstance(expected, str) and isinstance(actual, str):
        return label_key(expected) == label_key(actual)
    return expected == actual


def field_metrics(expected: dict, actual: dict, comparisons: dict | None = None) -> dict:
    """Per-field accuracy over a record-shaped answer."""
    comparisons = comparisons or {}
    correct = sum(values_match(v, actual.get(k), comparisons.get(k)) for k, v in expected.items())
    total = len(expected)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 1.0}


def table_metrics(expected_rows: list, actual_rows: list) -> dict:
    """Whole-row equality of two tables, order-insensitive."""
    key = lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
    expected = sorted(key(row) for row in expected_rows)
    actual = sorted(key(row) for row in actual_rows)
    return {
        "expected_rows": len(expected), "actual_rows": len(actual),
        "exact": float(expected == actual),
        "missing_rows": len(set(expected) - set(actual)), "extra_rows": len(set(actual) - set(expected)),
    }


def ranking_metrics(expected: list, actual: list) -> dict:
    """nDCG plus exact-match flags for an ordered answer.

    The ideal is the best achievable score for the same number of returned slots, so
    nDCG stays bounded by 1. It must be summed over the same slot count as the actual
    gain; over a different count a fully reversed answer scores 1.0.
    """
    expected_set = set(expected)
    top = min(len(expected), len(actual))
    # Gain only counts the first occurrence of an expected item, so a duplicated
    # answer cannot inflate its own score.
    seen: set = set()
    dcg = 0.0
    for index, key in enumerate(actual[:top]):
        if key in expected_set and key not in seen:
            seen.add(key)
            dcg += 1 / math.log2(index + 2)
    ideal = sum(1 / math.log2(index + 2) for index in range(top))
    return {
        "expected_order": expected, "actual_order": actual,
        "ndcg": dcg / ideal if ideal else 1.0,
        "top1_exact": float(bool(actual) and bool(expected) and actual[0] == expected[0]),
        "order_exact": float(actual == expected),
    }


def scalar_metrics(expected: dict, actual: dict) -> dict:
    """Exactness of scalar aggregates such as a maximum or a total."""
    details, scores = {}, []
    for key, want in expected.items():
        got = actual.get(key)
        exact = float(values_match(want, got) if got is not None else False)
        details[key] = {"expected": want, "actual": got, "exact": exact}
        scores.append(exact)
    details["exact"] = sum(scores) / len(scores) if scores else 0.0
    return details


def count_metric(expected: int, actual: int | None) -> dict:
    return {"expected": expected, "actual": actual, "exact": float(actual == expected)}


def mean(values) -> float:
    present = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(present) / len(present) if present else 0.0
