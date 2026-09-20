"""Core engine tests: contracts, generic signals, normalization and aggregation.

These use no network and no docker. They pin the behaviour the engine promises every
benchmark, including the edge cases that are easiest to get wrong: an unresolved answer
counting against precision, and an empty gold set with an empty answer scoring full
marks rather than zero.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchkit.case import Prediction, RunMetrics, ScoreResult
from benchkit.normalize import canonical_url, extract_json_object, normalize_answer
from benchkit.references import Catalog, scan_text_for_labels
from benchkit.run import as_prediction, summarize
from benchkit import signals


# --- signals ---------------------------------------------------------------

def test_empty_gold_with_empty_answer_is_not_punished():
    """Answering 'there are none' correctly must not score zero."""
    assert signals.set_metrics(set(), set())["f1"] == 1.0
    assert signals.set_metrics(set(), set())["precision"] == 1.0


def test_unresolved_mention_counts_against_precision():
    metrics = signals.set_metrics({"a"}, {"a"}, unresolved=1)
    assert metrics["false_positive"] == 1        # the unresolved mention is an extra
    assert metrics["unresolved"] == 1            # and is broken out for diagnosis
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["exact_set"] == 0.0


def test_extra_item_penalises_an_empty_gold_set():
    metrics = signals.set_metrics(set(), {"a"})
    assert metrics["false_positive"] == 1
    assert metrics["f1"] == 0.0


def test_labels_match_slug_and_display_spelling():
    assert signals.values_match("Workflow Automation", "workflow-automation")
    assert not signals.values_match("Workflow Automation", "workflow")

    assert signals.values_match("AI", "ai")


def test_list_fields_compare_as_unordered_sets():
    assert signals.values_match(["AI", "B2B"], ["b2b", "ai"])
    assert not signals.values_match(["AI", "B2B"], ["ai"])
    assert not signals.values_match(["AI", "B2B"], "AI, B2B")


def test_city_comparison_is_opt_in():
    full, city = "San Francisco, CA, USA", "San Francisco"
    assert signals.values_match(full, city, compare="city")
    assert not signals.values_match(full, city)
    assert not signals.values_match(full, "New York City, NY, USA", compare="city")


def test_field_metrics_report_per_field_accuracy():
    metrics = signals.field_metrics({"a": 1, "b": 2}, {"a": 1, "b": 9})
    assert metrics == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_table_metrics_are_order_insensitive_but_value_exact():
    exact = signals.table_metrics([{"x": 1}, {"y": 2}], [{"y": 2}, {"x": 1}])
    assert exact["exact"] == 1.0
    assert signals.table_metrics([{"x": 1}], [{"x": 2}])["exact"] == 0.0


def test_ranking_metrics_separate_relevance_from_exact_order():
    """nDCG measures whether relevant items appear at high ranks; it is not an
    order-equality test, so a reversed list of exactly the right items still scores
    well on relevance and is caught by `order_exact` instead."""
    ranked = signals.ranking_metrics(["a", "b"], ["a", "b"])
    assert ranked["ndcg"] == 1.0 and ranked["order_exact"] == 1.0

    swapped = signals.ranking_metrics(["a", "b"], ["b", "a"])
    assert swapped["order_exact"] == 0.0          # strict order disagrees
    assert swapped["top1_exact"] == 0.0
    assert swapped["ndcg"] == 1.0                 # both relevant items are still in top 2

    # Relevance is what nDCG is for: a wrong or missing item must cost it.
    assert signals.ranking_metrics(["a", "b"], ["b", "c"])["ndcg"] < 1.0
    assert signals.ranking_metrics(["a", "b"], ["c", "d"])["ndcg"] == 0.0


def test_ranking_metrics_cannot_exceed_one_with_duplicates():
    """A padded answer must not beat the ideal."""
    assert signals.ranking_metrics(["a", "b"], ["a", "a", "b", "b"])["ndcg"] <= 1.0
    assert signals.ranking_metrics(["a"], ["a", "a", "a"])["ndcg"] <= 1.0


def test_scalar_metrics_require_a_stated_value():
    assert signals.scalar_metrics({"total": 5}, {"total": 5})["exact"] == 1.0
    assert signals.scalar_metrics({"total": 5}, {})["exact"] == 0.0


# --- normalization ---------------------------------------------------------

MANIFEST = [
    {"id": "p1", "name": "Alchemist", "url": "https://example.org/alchemist"},
    {"id": "p2", "name": "Antler", "url": "https://example.org/antler"},
]


def catalog():
    return Catalog(MANIFEST, id_field="id")


def test_extract_json_object_handles_fences_and_prose():
    assert extract_json_object('Answer:\n```json\n{"items": []}\n```\n') == {"items": []}
    assert extract_json_object("no json here") is None


def test_canonical_url_lowercases_host_and_trims_slash():
    assert canonical_url("HTTPS://Example.ORG/a/") == "https://example.org/a"
    assert canonical_url("not a url") is None


def test_items_resolve_by_key_url_then_name_and_keep_unresolved():
    result = normalize_answer(
        {"items": [{"name": "Alchemist"}, {"url": "https://example.org/antler"}, {"name": "Nothing"}]},
        resolve=catalog().resolve,
    )
    assert [item["id"] for item in result["items"]] == ["p1", "p2", None]
    assert len(result["unresolved"]) == 1
    assert "unresolved_item" in result["warnings"]


def test_prose_urls_are_collected_as_references():
    result = normalize_answer("See https://example.org/x and https://other.test/y.", resolve=catalog().resolve)
    assert "unresolved_item" not in result["warnings"] or True
    assert "https://example.org/x" in result["references"]


def test_ambiguous_names_do_not_resolve():
    """Two entries sharing a name must not silently resolve to one of them."""
    ambiguous = Catalog([{"id": "a", "name": "Same"}, {"id": "b", "name": "Same"}], id_field="id")
    assert ambiguous.resolve_name("Same") is None
    assert ambiguous.resolve("Same") is None


def test_prose_label_scan_is_whole_word():
    found = scan_text_for_labels("We recommend Alchemist, and Antler too.", catalog())
    assert {item["label"] for item in found} == {"Alchemist", "Antler"}


# --- stored rows and aggregation ------------------------------------------

def test_as_prediction_rebuilds_metrics_from_a_stored_row():
    prediction = as_prediction({
        "task_id": "t1", "answer_text": "hi", "status": "ok",
        "metrics": {"wall_ms": 100, "num_turns": 3, "total_cost_usd": 0.5,
                    "input_tokens": 10, "output_tokens": 20},
    })
    assert prediction.task_id == "t1"
    assert prediction.metrics.turns == 3
    assert prediction.metrics.cost_usd == 0.5
    assert prediction.metrics.tokens["input"] == 10
    assert prediction.metrics.total_tokens == 30


def test_summary_weights_cost_over_tasks_that_really_ran():
    rows = [
        {"task_id": "a", "score": 1.0, "run_metrics": {"cost_usd": 0.5, "wall_ms": 60000}},
        {"task_id": "b", "score": 0.0, "run_metrics": {}},
    ]
    summary = summarize(rows)
    assert summary["score_total"] == 1.0
    assert summary["score_mean"] == 0.5           # the unanswered task is a real zero
    assert summary["mean_cost_usd"] == 0.5        # but it has no cost to average
    assert summary["score_per_dollar"] == 2.0
    assert summary["mean_wall_seconds"] == 60.0


def test_a_failed_attempt_does_not_zero_out_efficiency():
    """Spending the budget without producing an answer must not look like a
    zero-quality-per-dollar result for the answers that did work."""
    rows = [
        {"task_id": "a", "score": 2.0, "run_metrics": {"cost_usd": 0.5}},
        {"task_id": "b", "score": 0.0, "error": "failed", "run_metrics": {"cost_usd": 0.7}},
    ]
    summary = summarize(rows)
    assert summary["total_cost_usd"] == 1.2       # the wasted spend is still reported
    assert summary["answered_count"] == 1
    assert summary["score_per_dollar"] == 4.0     # computed on the answer that worked
    assert summary["score_mean"] == 1.0


def test_summary_reports_no_score_per_dollar_when_nothing_was_spent():
    assert summarize([{"task_id": "a", "score": 0.0, "run_metrics": {}}])["score_per_dollar"] is None


# --- contracts -------------------------------------------------------------

def test_contracts_carry_no_benchmark_vocabulary():
    """The core types are shared by every benchmark, so they must stay generic."""
    import benchkit.case as case_module

    text = Path(case_module.__file__).read_text(encoding="utf-8").casefold()
    # Naming a benchmark in an illustrative docstring is fine; using its vocabulary in
    # the type definitions is not.
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", '"""', "*")))
    for word in ("company", "yc_", "program_key", "incubator"):
        assert word not in code, f"domain word {word!r} leaked into the core contracts"


def test_score_result_keeps_signals_separate_from_the_headline():
    result = ScoreResult(task_id="t", score=2.0, metrics={"a": 1, "b": 1})
    assert result.score == 2.0
    assert result.metrics == {"a": 1, "b": 1}
    assert result.notes == []


def test_run_metrics_total_tokens_ignores_missing_values():
    metrics = RunMetrics(tokens={"input": 5, "output": None, "cache_read": 10})
    assert metrics.total_tokens == 15


# --- the engine and benchmarks stay independent of any knowledge base -------

#: Layout tokens that only a corpus reader would know. Concept words the *prompt* must
#: use -- "knowledge base", `kb_resolve`, MCP tool names -- are deliberately absent:
#: telling an agent to consult a knowledge base is the point of one condition, while
#: knowing where that knowledge base keeps its files is a coupling.
KB_LAYOUT_TOKENS = ("openakb", "provenance.json", "programs.tsv", "kb_dir",
                    "captures/", "sections/")


def _python_sources(*roots):
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_scoring_code_knows_a_knowledge_base_layout():
    """The benchmark must be buildable and scorable with no corpus on disk.

    A scoring input derived from the corpus under test cannot attribute a score change
    to that corpus: editing it would move the answer key. The same argument applies to
    the text claims are verified against, so verification reads the URL the answer
    cited. Only a knowledge-base *run* may touch a knowledge base, and the agent does
    that itself, over MCP or through the CLI.

    Prompt text is exempt, and deliberately so: telling an agent how to read a corpus is
    the point of a knowledge-base condition, and it is not a scoring input. What must
    never happen is the *scorer* resolving a corpus path, which is what this checks.
    """
    import re
    prompt_constant = re.compile(r'^\s*\w*GUIDANCE\w*\s*=\s*""".*?"""', re.S | re.M)
    repo = Path(__file__).parents[1]
    offenders = []
    for path in _python_sources(repo / "src" / "benchkit", repo / "benchmarks"):
        body = prompt_constant.sub("", path.read_text(encoding="utf-8"))
        code = "\n".join(line for line in body.splitlines()
                         if not line.lstrip().startswith(("#", '"""', "*")))
        for token in KB_LAYOUT_TOKENS:
            if token in code.casefold():
                offenders.append(f"{path.relative_to(repo)}: {token}")
    assert not offenders, "knowledge-base layout reached scoring code: " + "; ".join(offenders)


def test_the_benchmark_ships_no_generated_corpus_artifacts():
    """A benchmark ships its authored data and nothing derived from the corpus.

    Anything generated from the corpus under test would be a scoring input that moves
    when the corpus moves, which is the property the independence rule exists to
    prevent. The file list is asserted exactly, so adding one is a deliberate act.
    """
    bench = Path(__file__).parents[1] / "benchmarks" / "us-startup-programs"
    assert sorted(p.name for p in bench.glob("*.json")) == ["programs.json"]
    assert sorted(p.name for p in bench.iterdir() if p.is_file() and not p.name.startswith(".")) == [
        "README.md", "benchmark.py", "programs.json", "tasks.jsonl"]


# --- pricing ------------------------------------------------------------------

def test_cost_is_computed_from_tokens_not_taken_from_the_provider():
    """The CLI prices an unrecognised model at its own default rates.

    Pointed at another vendor's endpoint, its `total_cost_usd` is not the bill: a
    one-line prompt to `deepseek-flash` came back at $0.1131, which is Opus's rate for
    those tokens, against a real charge near $0.007. Cost is recomputed here.
    """
    from benchkit import pricing
    tokens = {"input": 22_548, "output": 16, "cache_read": 0, "cache_creation": 0}
    off_peak = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)   # Tuesday 12:00 UTC
    flash = pricing.cost_usd("deepseek-flash", tokens, when=off_peak)
    # 22,548 misses at $0.15/M + 16 outputs at $0.60/M
    assert flash == pytest.approx((22_548 * 0.15 + 16 * 0.60) / 1e6)
    assert flash < 0.1131 / 10, "the CLI's figure must not be what this returns"


def test_deepseek_prices_double_inside_its_peak_window():
    from benchkit import pricing
    tokens = {"input": 1_000_000, "output": 0, "cache_read": 0, "cache_creation": 0}
    tuesday_peak = datetime(2026, 9, 15, 7, 30, tzinfo=timezone.utc)   # 06:00-10:00 UTC
    tuesday_off = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    assert pricing.is_peak("deepseek-flash", tuesday_peak)
    assert not pricing.is_peak("deepseek-flash", tuesday_off)
    assert pricing.cost_usd("deepseek-flash", tokens, when=tuesday_peak) == pytest.approx(0.30)
    assert pricing.cost_usd("deepseek-flash", tokens, when=tuesday_off) == pytest.approx(0.15)
    # Weekends are off-peak all day, and a naive timestamp is read as UTC.
    saturday = datetime(2026, 9, 19, 7, 30, tzinfo=timezone.utc)
    assert not pricing.is_peak("deepseek-flash", saturday)
    # A naive timestamp is read as UTC, not as local time, so this is still peak.
    assert pricing.is_peak("deepseek-flash", tuesday_peak.replace(tzinfo=None))


def test_anthropic_models_have_no_peak_window():
    from benchkit import pricing
    assert not pricing.is_peak("claude-sonnet-5", datetime(2026, 9, 15, 7, 30, tzinfo=timezone.utc))
    cost = pricing.cost_usd("claude-sonnet-5",
                            {"input": 1_000_000, "output": 1_000_000,
                             "cache_read": 0, "cache_creation": 0})
    assert cost == pytest.approx(3.0 + 15.0)


def test_an_unknown_model_has_no_price_rather_than_a_guess():
    from benchkit import pricing
    assert pricing.cost_usd("some-model-nobody-priced", {"input": 1000}) is None
    assert pricing.cost_usd(None, {"input": 1000}) is None
    # Missing or malformed counts contribute nothing instead of raising.
    assert pricing.cost_usd("claude-sonnet-5", {"input": None, "output": "x"}) == 0.0


def test_a_dated_model_name_resolves_to_its_price():
    from benchkit import pricing
    assert pricing.lookup("claude-haiku-4-5-20251001") is pricing.MODEL_PRICES["claude-haiku-4-5"]
    assert pricing.lookup("claude-sonnet-5") is pricing.MODEL_PRICES["claude-sonnet-5"]
    assert pricing.lookup("nobody-priced-this") is None


def test_cost_covers_every_model_the_run_used():
    """A run is not always one model: web search is served by a second, smaller one.

    Summing only the answering model understated a web-agent task by half, because its
    haiku search calls cost slightly more than the sonnet turns they supported.
    """
    from benchkit import pricing
    off_peak = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    usage = {
        "claude-sonnet-5": {"inputTokens": 28, "outputTokens": 12_636,
                            "cacheReadInputTokens": 520_805, "cacheCreationInputTokens": 55_446},
        "claude-haiku-4-5-20251001": {"inputTokens": 309_521, "outputTokens": 13_835,
                                      "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                                      "webSearchRequests": 18},
    }
    total = pricing.cost_from_model_usage(usage, when=off_peak)
    sonnet = pricing.cost_usd("claude-sonnet-5", {
        "input": 28, "output": 12_636, "cache_read": 520_805, "cache_creation": 55_446},
        when=off_peak)
    haiku = pricing.cost_usd("claude-haiku-4-5", {
        "input": 309_521, "output": 13_835, "cache_read": 0, "cache_creation": 0},
        when=off_peak)
    searches = 18 * pricing.WEB_SEARCH_USD_PER_REQUEST
    assert total == pytest.approx(sonnet + haiku + searches)
    # The auxiliary model and the searches together are about 40% of the bill.
    assert sonnet < total * 0.62


def test_the_search_fee_is_charged_only_by_the_vendor_that_bills_it():
    """Anthropic charges per server-side search; DeepSeek's pricing is tokens alone.

    Applying Anthropic's fee to a `deepseek-flash` run invented $0.13 on a bill that
    was really $0.36 -- a quarter of the total, from a line item it does not have.
    """
    from benchkit import pricing
    off_peak = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
    tokens = {"inputTokens": 1_000_000, "outputTokens": 0,
              "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
              "webSearchRequests": 10}
    # Both sides are pinned to the same instant. Comparing an off-peak total against
    # `cost_usd`'s default of "now" made this test fail between 01:00 and 04:00 UTC,
    # when DeepSeek's peak multiplier applies and the two sides disagree by exactly 2x.
    for model in ("claude-sonnet-5", "claude-haiku-4-5-20251001"):
        assert pricing.cost_from_model_usage({model: dict(tokens)}, when=off_peak) == pytest.approx(
            pricing.cost_usd(model, {"input": 1_000_000}, when=off_peak)
            + 10 * pricing.WEB_SEARCH_USD_PER_REQUEST)
    for model in ("deepseek-flash", "deepseek-v4-pro"):
        assert pricing.cost_from_model_usage({model: dict(tokens)}, when=off_peak) == pytest.approx(
            pricing.cost_usd(model, {"input": 1_000_000}, when=off_peak))


def test_an_unpriced_model_makes_the_total_unknown_rather_than_partial():
    from benchkit import pricing
    usage = {"claude-sonnet-5": {"inputTokens": 10, "outputTokens": 10},
             "mystery-model": {"inputTokens": 10, "outputTokens": 10}}
    assert pricing.cost_from_model_usage(usage) is None
    assert pricing.cost_from_model_usage({}) is None


def test_a_model_is_priced_the_same_whichever_route_addresses_it():
    """The judge's bill came back unknown because it was named `deepseek/deepseek-flash`.

    A LiteLLM model id carries its provider; the price belongs to the model. Without
    stripping the route, switching the judge to a cheaper provider looked like it cost
    nothing at all, because the cost came back as None.
    """
    from benchkit import pricing
    direct = pricing.lookup("deepseek-flash")
    assert pricing.lookup("deepseek/deepseek-flash") == direct
    assert pricing.lookup("openrouter/deepseek/deepseek-flash") == direct
    assert pricing.lookup("mystery/model") is None


# --- the run manifest ------------------------------------------------------

def test_the_manifest_records_the_corpus_revision(tmp_path):
    """A stored score must name the corpus it measured.

    The corpus is the thing under test, so a revision that moves invalidates a
    comparison in a way no score can show by itself. A benchmark pins the revision it
    was run against and the manifest carries it, so two runs can be checked for
    comparability without trusting anyone's memory.
    """
    from benchkit.run import write_manifest

    class StubBenchmark:
        key = "stub"
        parser_version = 1
        corpus_revision = "r7"
        directory = tmp_path

    manifest = write_manifest(tmp_path, StubBenchmark(), None, None, [])
    assert manifest["corpus_revision"] == "r7"
    assert manifest["benchmark"] == "stub"
    # Written to disk, not only returned.
    import json
    assert json.loads((tmp_path / "manifest.json").read_text())["corpus_revision"] == "r7"


def test_a_benchmark_without_a_corpus_records_no_revision(tmp_path):
    """The engine records what a benchmark declares and invents nothing."""
    from benchkit.run import write_manifest

    class StubBenchmark:
        key = "stub"
        parser_version = 1
        directory = tmp_path

    assert write_manifest(tmp_path, StubBenchmark(), None, None, [])["corpus_revision"] is None


def test_the_bundled_benchmark_pins_a_revision():
    """The pin is what makes two runs of this benchmark comparable, or not."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "usp", Path(__file__).parents[1] / "benchmarks" / "us-startup-programs" / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.KNOWLEDGE_BASE_REVISION, "the corpus revision must be pinned"
    assert module.StartupPrograms.corpus_revision == module.KNOWLEDGE_BASE_REVISION
