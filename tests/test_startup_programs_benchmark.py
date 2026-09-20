"""Scoring tests for the us-startup-programs benchmark.

The property that matters most here is that an open-ended answer is graded per
returned program, with no completeness term: padding must not pay, and a short
accurate answer must beat a long speculative one. These tests use a stub judge so
they run offline.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchkit.case import Prediction, RunMetrics  # noqa: E402

BENCH_DIR = REPO / "benchmarks" / "us-startup-programs"

#: The directory seeded by the `source_cache` fixture. Module-level because the test
#: helpers that build records are plain functions, and every URL a test cites has to be
#: placed where the resolver will find it before scoring runs.
CACHE: dict = {}


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("usp_benchmark", BENCH_DIR / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["usp_benchmark"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_benchmark_module()


@pytest.fixture(scope="module")
def source_cache(tmp_path_factory):
    """Where the stub tests' "fetched" pages live.

    Scoring resolves the page each record cites, and the tests must not reach the
    network to do it. Every URL a test cites is seeded into this directory, and the
    resolver is built with `allow_fetch=False`, so an unseeded URL is a genuine miss
    rather than a slow timeout.
    """
    path = tmp_path_factory.mktemp("sources")
    CACHE["dir"] = path
    return path


@pytest.fixture(scope="module")
def benchmark(mod, source_cache):
    from benchkit.sources import SourceText
    instance = mod.StartupPrograms()
    instance._source_text = SourceText(source_cache, allow_fetch=False)
    return instance


def cache_source(url, text="Official site of the program. Applications are open now."):
    """Seed the page the resolver will find for `url`, and return the url.

    Keyed on the canonical form, because that is what the parser stores on a record
    and therefore what the resolver looks up: `.../yc/` and `.../yc` are one page,
    and seeding the uncanonical spelling would leave the resolver with a miss.
    """
    import hashlib

    from benchkit.normalize import canonical_url
    CACHE["dir"].mkdir(parents=True, exist_ok=True)
    canonical = canonical_url(url)
    for key in {url, canonical}:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        (CACHE["dir"] / f"{digest}.txt").write_text(text, encoding="utf-8")
    return url


class StubJudge:
    """Answers every verification with a fixed verdict.

    `kinds` maps a claim kind to the verdict for that kind, for tests that need
    different answers to the structural questions and the factual ones. Kinds are
    `is_program`, `relevance`, `reputation`, and the fields in `CLAIM_WEIGHTS`. The
    match is on a distinctive fragment of each claim the benchmark actually sends, so
    a reworded claim surfaces here as a misclassified verdict rather than as a test
    that silently passes for the wrong reason.
    """

    MARKERS = (
        ("is_program", "startup-credit program"),
        ("relevance", "fits what the task question asks for"),
        ("reputation", "published by the organization"),
        ("status", " status:"),
        ("deadline", " deadline:"),
        ("terms", " terms:"),
        ("eligibility", " eligibility:"),
    )

    def __init__(self, verdict="supported", program=True, kinds=None):
        self.verdict = verdict
        self.program = program
        self.kinds = kinds or {}
        self.calls = []
        #: The `samples` each call asked for, so a test can assert which kinds are
        #: worth more than one verdict.
        self.sample_counts = []
        #: The claim names of each batched call, so a test can assert what was grouped
        #: into one request. Batched judging is the only mode the benchmark has.
        self.batches = []

    @classmethod
    def _kind(cls, claim):
        for kind, marker in cls.MARKERS:
            if marker in claim:
                return kind
        raise AssertionError(f"unrecognised claim sent to the judge: {claim!r}")

    def verify(self, item, claim, source_text=None, source_url=None, question=None,
               samples=None):
        kind = self._kind(claim)
        self.calls.append((item.get("name"), claim, kind, question))
        self.sample_counts.append(samples)
        if kind == "is_program" and kind not in self.kinds:
            verdict = "supported" if self.program else "contradicted"
        else:
            verdict = self.kinds.get(kind, self.verdict)
        return {"verdict": verdict, "evidence": "stub", "confidence": "high"}

    def verify_many(self, item, claims, source_text=None, source_url=None, question=None,
                    samples=None):
        """Answer a whole batch, recording each claim as its own call.

        The stub decides one claim at a time because that is what the tests assert on;
        what matters here is that the benchmark groups the right claims into one
        request, which `batches` records.
        """
        self.batches.append(tuple(claims))
        return {name: self.verify(item, claim, source_text=source_text, source_url=source_url,
                                  question=question, samples=samples)
                for name, claim in claims.items()}

    def is_program(self, item, source_text=None):
        self.calls.append((item.get("name"), "is_program", "is_program", None))
        return {"verdict": "supported" if self.program else "contradicted",
                "evidence": "stub", "confidence": "high"}


def task():
    return {"task_id": "t1", "question": "which programs?", "kind": "shortlist"}


def answer(programs, text=""):
    """A schema-valid answer carrying `programs`, as the harness delivers one."""
    return Prediction(task_id="t1", answer_text=text or json_dump(programs),
                      structured={"programs": programs},
                      normalized={"items": [{"label": p.get("name"), "id": p.get("key"),
                                             "raw": p} for p in programs]})


def prose_only(text):
    """An answer with no schema-valid result: the text is all there is."""
    return Prediction(task_id="t1", answer_text=text, structured=None, normalized={})


def json_dump(value):
    import json
    return json.dumps(value)


def known_program(benchmark, index=0):
    """A program from the benchmark's own universe, cited on its own domain."""
    program = benchmark.programs[index]
    return {"name": program["title"], "url": cache_source(f"https://{program['domains'][0]}/"),
            "key": program["key"], "status": "open", "deadline": "2026-10-01"}


def unknown_program(name="Obscure Program", url="https://obscure.example/"):
    """A program the benchmark's universe does not contain, cited on its own site."""
    return {"name": name, "url": cache_source(url), "status": "open", "deadline": "2026-10-01"}


ALL_SUPPORTED = StubJudge("supported")


# --- the core property: no completeness term -------------------------------

def test_completeness_is_not_a_scored_signal(benchmark):
    """The whole point of this benchmark: an unbounded list is never penalised."""
    joined = " ".join(benchmark.signals).casefold()
    for word in ("complete", "recall", "coverage", "missing"):
        assert word not in joined


# --- the fairness properties ----------------------------------------------

def test_catalog_membership_scores_nothing_by_itself(benchmark):
    """A program in the universe and one outside it earn the same, judged the same.

    This is the property that keeps the key from becoming a scoring input: a catalog
    key that awarded its point without consulting a judge would put the conditions that
    can read the key a point per recognised program ahead of the ones that cannot.
    """
    in_universe = benchmark.score(task(), answer([known_program(benchmark, 0)]),
                                  judge=ALL_SUPPORTED)
    out_of_universe = benchmark.score(task(), answer([unknown_program()]),
                                      judge=ALL_SUPPORTED)
    assert out_of_universe.score == in_universe.score
    assert in_universe.metrics["programs_recognised"] == 1
    assert out_of_universe.metrics["programs_recognised"] == 0


def test_every_record_is_asked_the_same_structural_questions(benchmark):
    """`is_program` and `relevance` are asked of catalog members too."""
    judge = StubJudge("supported")
    benchmark.score(task(), answer([known_program(benchmark, 0), unknown_program()]), judge=judge)
    kinds = [kind for _, _, kind, _ in judge.calls]
    assert kinds.count("is_program") == 2
    assert kinds.count("relevance") == 2


def test_the_relevance_question_carries_the_task_question(benchmark):
    judge = StubJudge("supported")
    benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=judge)
    asked = [question for _, _, kind, question in judge.calls if kind == "relevance"]
    assert asked == ["which programs?"]


def test_an_unverifiable_record_is_excluded_rather_than_zeroed(benchmark):
    """A page that could not be fetched says nothing about the answer's quality.

    Scoring a claim `not_stated` = 0 when the citation fails to resolve would cost a
    condition the whole item whenever a page is down, redirected or rate-limited, which
    is a fact about the page rather than about what the answer wrote.
    """
    record = {**known_program(benchmark, 0), "url": "https://unreachable.example/x"}
    result = benchmark.score(task(), answer([record]), judge=ALL_SUPPORTED)
    assert result.metrics["unverifiable_claims"] > 0
    assert result.metrics["claims_judged"] == 0
    assert result.metrics["claims"] == 0.0
    # The unreadable citation is still charged, once, as a citation defect.
    assert result.metrics["penalty"] == -1.0


def test_a_documented_absence_earns_more_than_silence(benchmark):
    """"No deadline is published" is a finding; the source merely omitting it is not."""
    record = known_program(benchmark, 0)
    silence = benchmark.score(task(), answer([record]), judge=StubJudge("not_stated"))
    documented = benchmark.score(task(), answer([record]), judge=StubJudge("not_published"))
    assert documented.score > silence.score
    assert documented.metrics["documented_absences"] > 0


def test_an_irrelevant_program_earns_nothing(benchmark):
    """Otherwise the score is an unnormalised sum and padding with real programs pays.

    Relevance gates rather than scores. A relevance *bonus* would add a constant to
    every item and so pay twice for a long list; gating makes a long list pay only
    when its entries are on topic.
    """
    on_topic = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=ALL_SUPPORTED)
    off_topic = benchmark.score(task(), answer([known_program(benchmark, 0)]),
                                judge=StubJudge("supported", kinds={"relevance": "contradicted"}))
    assert off_topic.metrics["not_relevant"] == 1
    assert off_topic.metrics["items_voided"] == 1
    assert off_topic.score == 0.0
    assert on_topic.score > 0.0


def test_a_contradicted_program_still_costs_after_the_gate(benchmark):
    """Padding a list with things that are not programs must cost, not merely fail to pay."""
    judge = StubJudge(kinds={"is_program": "contradicted"})
    result = benchmark.score(task(), answer([unknown_program()]), judge=judge)
    assert result.metrics["items_voided"] == 1
    assert result.score == -1.0


def test_every_verdict_is_tallied_under_the_question_it_answered(benchmark):
    """The score is a weighted sum; this keeps the raw count beside it.

    A weighted sum cannot be read back into "how often was this answer right", which is
    the first thing a reader who is not going to read the scoring code asks. The tally
    is what makes that answerable from a stored run, so it has to count every verdict
    once and file it under the question rather than the field.
    """
    judge = StubJudge(kinds={"status": "contradicted", "deadline": "not_stated"})
    result = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=judge)
    verdicts = result.metrics["verdicts"]

    assert verdicts["claim.contradicted"] == 1
    assert verdicts["claim.not_stated"] == 1
    assert verdicts.get("claim.supported", 0) == 0
    assert verdicts["is_program.supported"] == 1
    assert verdicts["relevance.supported"] == 1
    assert verdicts["reputation.supported"] == 1
    # One verdict per question asked, so the structural tallies match the record count.
    structural = sum(count for label, count in verdicts.items()
                     if not label.startswith("claim."))
    assert structural == 3


def test_a_duplicate_outside_the_catalog_is_not_double_counted(benchmark):
    """Duplicate detection comes from the answer, not from catalog keys."""
    once = benchmark.score(task(), answer([unknown_program()]), judge=ALL_SUPPORTED)
    twice = benchmark.score(task(), answer([unknown_program(), unknown_program()]),
                            judge=ALL_SUPPORTED)
    assert twice.score == once.score
    assert twice.metrics["duplicates"] == 1


def test_the_parser_version_is_reported(benchmark):
    """Scores must not be compared across a change in how answers are read."""
    result = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=ALL_SUPPORTED)
    assert result.metrics["parser_version"] == benchmark.parser_version


# --- per-item rules --------------------------------------------------------

def test_empty_answer_scores_zero(benchmark):
    result = benchmark.score(task(), prose_only("I don't know."))
    assert result.score == 0.0
    assert result.metrics["programs_returned"] == 0
    assert any("no schema-valid result" in note for note in result.notes)


def test_missing_citation_is_penalised(benchmark):
    record = {**known_program(benchmark, 0), "url": None}
    without = benchmark.score(task(), answer([record]), judge=ALL_SUPPORTED)
    with_url = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=ALL_SUPPORTED)
    assert without.score < with_url.score
    assert without.metrics["penalty"] == -0.5


def test_duplicate_rows_are_not_double_counted(benchmark):
    once = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=ALL_SUPPORTED).score
    twice = benchmark.score(task(), answer([known_program(benchmark, 0), known_program(benchmark, 0)]),
                            judge=ALL_SUPPORTED)
    assert twice.score == once
    assert twice.metrics["duplicates"] == 1


def test_duplicate_rows_are_merged_rather_than_dropped(benchmark):
    """The two rows carry different fields; neither may be lost.

    A program listed in prose and again in a summary table is one program, but the two
    rows state different things -- one may carry the deadline, the other the citation.
    Dropping the second discarded facts the answer did state, which is why the merge
    keeps every field rather than keeping the first row.
    """
    record = known_program(benchmark, 0)
    bare = {k: v for k, v in record.items() if k in ("name", "url", "key")}
    single = benchmark.score(task(), answer([record]), judge=ALL_SUPPORTED)
    merged = benchmark.score(task(), answer([bare, record]), judge=ALL_SUPPORTED)
    assert merged.metrics["duplicates"] == 1
    assert merged.score == single.score
    assert merged.metrics["claims"] == single.metrics["claims"]


def test_an_unknown_program_is_judged_on_its_merits(benchmark):
    """Absence from the catalog is not proof of absence from the world."""
    rejected = benchmark.score(task(), answer([unknown_program()]), judge=StubJudge(program=False))
    accepted = benchmark.score(task(), answer([unknown_program()]), judge=ALL_SUPPORTED)
    assert accepted.score > rejected.score
    assert rejected.metrics["programs_recognised"] == 0


def test_scoring_without_a_judge_applies_only_structural_rules(benchmark):
    """No judge means no verdicts: the score is the structural part and no more.

    It must not score above zero without one: every point comes from a verdict, so an
    unjudged run cannot be mistaken for a good one.
    """
    with_url = benchmark.score(task(), answer([known_program(benchmark, 0)]))
    without_url = benchmark.score(task(), answer([{**known_program(benchmark, 0), "url": None}]))
    assert with_url.metrics["judge_used"] is False
    assert with_url.score == 0.0
    assert without_url.score == -0.5


def test_a_contradicted_claim_costs_more_than_an_omitted_one(benchmark):
    """A confidently wrong row must not be better than staying silent about it."""
    record = known_program(benchmark, 0)
    omitted = benchmark.score(task(), answer([{**record, "deadline": None}]), judge=StubJudge("not_stated"))
    wrong = benchmark.score(task(), answer([record]), judge=StubJudge("contradicted"))
    assert wrong.score < omitted.score


def test_judge_is_consulted_for_claims_and_reputation(benchmark):
    judge = StubJudge("supported")
    result = benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=judge)
    kinds = {kind for _, _, kind, _ in judge.calls}
    assert "status" in kinds
    assert "reputation" in kinds
    assert result.metrics["judge_used"] is True


def test_reputation_is_judged_rather_than_looked_up(benchmark):
    """The old check compared the URL against domains enumerated in the catalog, so a
    program outside the catalog could never earn it, however official its citation."""
    on_site = benchmark.score(task(), answer([unknown_program()]),
                              judge=StubJudge("supported", kinds={"reputation": "supported"}))
    off_site = benchmark.score(task(), answer([unknown_program()]),
                               judge=StubJudge("supported", kinds={"reputation": "contradicted"}))
    assert on_site.metrics["reputation"] > 0
    assert off_site.metrics["reputation"] < 0


# --- prose fallback --------------------------------------------------------



def test_unparseable_prose_returns_no_rows_rather_than_guesses(benchmark):
    result = benchmark.score(task(), Prediction(
        task_id="t1", answer_text="There are many great programs out there, such as YC and Techstars."))
    assert result.metrics["programs_returned"] == 0
    assert result.score == 0.0










def test_a_qualifier_after_a_program_name_still_resolves(benchmark):
    """`PearX (Pear VC)` and `Techstars (Spring 2027 programs)` name real programs.

    Reported only: resolution feeds `programs_recognised`, which is a diagnostic and
    scores nothing. It still has to be right, or the diagnostic misleads.
    """
    title = benchmark.programs[0]["title"]
    qualified = f"{title} (Spring 2027 programs)"
    url = cache_source("https://example.com/")
    plain = benchmark.score(task(), answer([{"name": title, "url": url}]))
    decorated = benchmark.score(task(), answer([{"name": qualified, "url": url}]))
    assert plain.metrics["programs_recognised"] == 1
    assert decorated.metrics["programs_recognised"] == 1


# --- scenario wiring -------------------------------------------------------

def test_scenarios_carry_this_benchmarks_scope_and_corpus(mod):
    scenarios = mod.StartupPrograms().scenarios()
    assert set(scenarios) == {"web-agent", "kb-mcp", "kb-cli"}
    # The CLI condition reaches the same corpus without an MCP server.
    cli = scenarios["kb-cli"]
    assert mod.KNOWLEDGE_BASE in cli.system_prompt
    assert mod.SCOPE in cli.system_prompt
    assert cli.mcp_server is None
    assert not any(tool.startswith("mcp__") for tool in cli.tools)
    mcp = scenarios["kb-mcp"]
    assert mod.KNOWLEDGE_BASE in mcp.system_prompt
    assert mod.SCOPE in mcp.system_prompt
    assert f"mcp__{mcp.mcp_server}" in mcp.tools
    assert not any(tool.startswith("mcp__") for tool in scenarios["web-agent"].tools)


def test_the_prompts_differ_only_in_which_corpus_is_reachable(mod):
    """The control the README claims. If technique leaks into one prompt, it is not one.

    Every condition states the task and which tools it has. None states how to use
    them: that is the tool's own documentation to give, through the MCP handshake or
    `--help`, and it reaches every condition the same way. This test is what keeps the
    claim in the README true rather than aspirational.
    """
    prompts = mod.StartupPrograms().scenarios()
    texts = {key: scenario.system_prompt for key, scenario in prompts.items()}

    # The corpus clause is the only difference, so removing it must leave one text.
    stripped = set()
    for text in texts.values():
        without = text
        for clause in (f" You also have a knowledge base about {mod.SCOPE}, whose address "
                       f"is `{mod.KNOWLEDGE_BASE}`, exposed through the `kb` MCP server.",
                       f" You also have the `nurok` command-line client, which can read "
                       f"a knowledge base about {mod.SCOPE}, whose address is "
                       f"`{mod.KNOWLEDGE_BASE}`."):
            without = without.replace(clause, "")
        stripped.add(without)
    assert len(stripped) == 1, "the conditions' prompts differ by more than the corpus"

    # And nothing in any prompt teaches technique: no measured result, no call recipe,
    # no instruction about which source to prefer.
    blob = " ".join(texts.values()).casefold()
    for leak in ("scored", "cheaper", "% higher", "depth 3", "kb_sections_list",
                 "kb_subtree_read", "rg -n", "openakb", "start from the knowledge",
                 "prefer it for", "only to fill gaps"):
        assert leak not in blob, f"technique leaked into a prompt: {leak!r}"


def test_the_universe_is_well_formed_and_frozen(benchmark):
    """The answer key lives with the benchmark, not with the corpus under test."""
    keys = [p["key"] for p in benchmark.programs]
    assert len(keys) == len(set(keys)), "program keys must be unique"
    for program in benchmark.programs:
        assert program["title"] and program["domains"], program["key"]
        for domain in program["domains"]:
            assert domain == domain.casefold() and "/" not in domain, domain


def test_scoring_reads_only_files_from_this_directory(benchmark, tmp_path):
    """Scoring must work from this benchmark's own two data files and nothing else.

    If it read anything derived from the corpus under test, editing that corpus would
    move the answer key, and a score change could not be attributed to the corpus.
    Run against a directory holding only `programs.json` and `tasks.jsonl`, it has to
    behave identically.
    """
    import shutil
    for name in ("programs.json", "tasks.jsonl"):
        shutil.copy(BENCH_DIR / name, tmp_path / name)
    isolated = type(benchmark)()
    isolated.directory = tmp_path
    isolated.__init__(tmp_path)  # must not raise with nothing else present
    result = isolated.score(task(), answer([{"name": "Y Combinator",
                                            "url": "https://www.ycombinator.com/apply"}]))
    assert result.metrics["programs_recognised"] == 1


def test_a_citation_on_the_operators_own_domain_is_credited(benchmark):
    """`ycombinator.com/deal` is the operator's own page, corpus or no corpus.

    The citation now earns this from a judge reading the page, not from a domain list
    enumerated in the benchmark's own catalog -- which only ever credited programs the
    catalog already contained.
    """
    url = cache_source("https://www.ycombinator.com/deal", "Y Combinator. Apply to YC.")
    result = benchmark.score(
        task(), answer([{"name": "Y Combinator", "url": url, "status": "open"}]),
        judge=StubJudge("supported", kinds={"reputation": "supported"}))
    assert result.metrics["reputation"] > 0, result.notes


def test_a_citation_off_the_operators_domain_is_not_credited(benchmark):
    url = cache_source("https://techcrunch.com/2026/09/14/yc/", "TC covers YC demo day.")
    result = benchmark.score(
        task(), answer([{"name": "Y Combinator", "url": url, "status": "open"}]),
        judge=StubJudge("supported", kinds={"reputation": "contradicted"}))
    assert result.metrics["reputation"] < 0
    assert any("not on the operator's own site" in note for note in result.notes)




def test_every_engine_scenario_is_provided_by_this_benchmark(mod):
    """`run --scenario X` indexes the benchmark's dict, so a missing key is a KeyError.

    The engine decides which conditions exist; a benchmark decides how each is
    specialized. If those two drift, the failure lands on whoever runs the condition
    that was forgotten, so it is asserted here instead.
    """
    import importlib.util
    from pathlib import Path as _Path
    spec = importlib.util.spec_from_file_location(
        "benchkit_harness", _Path(__file__).parents[1] / "src" / "benchkit" / "harness" / "claude_docker.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    provided = mod.StartupPrograms().scenarios()
    assert set(harness.SCENARIOS) <= set(provided), (
        f"engine scenarios {sorted(harness.SCENARIOS)} are not all provided: {sorted(provided)}")
    for key, scenario in provided.items():
        assert scenario.key == key
        assert scenario.system_prompt.strip()






def test_a_trailing_parenthesised_domain_is_not_part_of_the_name(mod):
    """It is a citation, not a qualifier -- `PearX (Pear VC)` still keeps its own."""
    assert mod._clean_name("**NSF SBIR/STTR** (seedfund.nsf.gov)") == "NSF SBIR/STTR"
    assert mod._clean_name("PearX (Pear VC)") == "PearX (Pear VC)"




def test_the_components_sum_to_the_score(benchmark):
    """`tools/reweight.py` re-ranks from these, so an approximation is a wrong answer.

    A gated item's verdicts and its contribution differ -- its claims were judged and
    recorded, but none of them scored -- so the metrics must hold what scored.
    """
    cases = [
        (ALL_SUPPORTED, [known_program(benchmark, 0), unknown_program()]),
        (StubJudge("supported", kinds={"relevance": "contradicted"}), [known_program(benchmark, 1)]),
        (StubJudge(kinds={"is_program": "contradicted"}), [unknown_program("Pad")]),
        (StubJudge("not_published"), [known_program(benchmark, 2), known_program(benchmark, 3)]),
    ]
    for judge, programs in cases:
        result = benchmark.score(task(), answer(programs), judge=judge)
        components = sum(result.metrics[name]
                         for name in ("legit", "claims", "reputation", "penalty"))
        assert abs(components - result.score) < 1e-9, (components, result.score, result.notes)


def test_a_record_is_asked_in_batches_that_share_a_page(benchmark):
    """The grouping is the point: one request per page, not one per question.

    A record stating four facts, with two structural questions asked three times each,
    put the same cited page through the model ten times. Grouped, it goes three: the
    stated fields together, the structural pair together, and `relevance` on its own
    because it is the only question about the task rather than the page.
    """
    judge = StubJudge("supported")
    benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=judge)
    assert set(judge.batches[0]) == {"status", "deadline"}
    assert judge.batches[1] == ("is_program", "reputation")
    assert judge.batches[2] == ("relevance",)


def test_the_structural_questions_are_asked_more_than_once(benchmark):
    """They flip one time in five and carry most of the score; a claim flips 1 in 40."""
    judge = StubJudge("supported")
    benchmark.score(task(), answer([known_program(benchmark, 0)]), judge=judge)
    asked = dict(zip((k for _, _, k, _ in judge.calls), judge.sample_counts))
    assert asked["is_program"] == 3
    assert asked["reputation"] == 3
    assert asked["status"] == 1


def test_field_coverage_is_reported(benchmark):
    """The signal that says why a score is what it is, not just what it is.

    The same condition stated `status` on 44 of 46 items in one round and 8 of 49 in
    the next, and the score followed the format. Without this, that reads as a
    capability difference.
    """
    record = known_program(benchmark, 0)
    result = benchmark.score(task(), answer([record, unknown_program()]), judge=ALL_SUPPORTED)
    assert result.metrics["field_status"] == 2
    assert result.metrics["field_deadline"] == 2
    # two items, two fields each
    assert result.metrics["fields_per_item"] == pytest.approx(2.0)


# --- the declared answer schema --------------------------------------------
#
# Grading reads the schema-validated result and nothing else. Reconstructing records
# from prose instead means guessing which heading starts an entry, which word means
# "deadline" and which column means "open", and the guessing fails unevenly -- so the
# error tracks how an answer is written rather than what it says. The tests below pin
# the properties that make reading the answer unnecessary.


def test_the_schema_is_declared_to_every_condition(mod):
    """All three conditions answer in one shape, so a score gap is a finding gap."""
    scenarios = mod.StartupPrograms().scenarios()
    schemas = {key: scenario.output_schema for key, scenario in scenarios.items()}
    assert all(schemas.values()), schemas
    assert len({json_dump(s) for s in schemas.values()}) == 1


def test_the_schema_requires_only_what_every_answer_must_have(mod):
    """`name` and `url`; the graded fields stay optional on purpose.

    A required field with nothing behind it invites a guess, and a program that
    publishes no deadline is a real answer the schema has to keep expressible.
    """
    item = mod.OUTPUT_SCHEMA["properties"]["programs"]["items"]
    assert set(item["required"]) == {"name", "url"}
    assert "programs" in mod.OUTPUT_SCHEMA["required"]


def test_the_schema_declares_every_field_that_is_graded(mod):
    """A field the scorer credits but the schema never mentions is unreachable.

    This is the invariant that the old reader could not state and repeatedly broke:
    the scorer credited `terms`, and no answer could be sure which word would make it
    readable.
    """
    item = mod.OUTPUT_SCHEMA["properties"]["programs"]["items"]
    for field in mod.CLAIM_WEIGHTS:
        assert field in item["properties"], field


def test_an_answer_without_a_schema_result_scores_zero_and_is_counted(benchmark):
    """Prose is not read. A condition that cannot produce the declared shape fails.

    Counted apart from an empty list, because the two are different facts: one is a
    task contract not met, the other is a search that found nothing.
    """
    result = benchmark.score(task(), prose_only(
        "**1. Y Combinator** — https://www.ycombinator.com/apply\n- **Status:** Open\n"),
        judge=ALL_SUPPORTED)
    assert result.score == 0.0
    assert result.metrics["programs_returned"] == 0
    assert result.metrics["schema_failures"] == 1
    assert any("no schema-valid result" in note for note in result.notes)


def test_an_empty_program_list_is_not_a_schema_failure(benchmark):
    result = benchmark.score(task(), answer([]), judge=ALL_SUPPORTED)
    assert result.score == 0.0
    assert result.metrics["schema_failures"] == 0
    assert any("returned no programs" in note for note in result.notes)


def test_an_entry_is_graded_from_its_own_fields(benchmark):
    """What the entry states is what is checked -- no vocabulary in between.

    The reader this replaced read `- **Offers:**` as no terms and a column headed
    `Open on 2026-09-14?` as no status, on every row, while the answer stated both.
    """
    url = cache_source("https://example.com/")
    result = benchmark.score(task(), answer([{
        "name": "Example Program", "url": url,
        "status": "open", "deadline": "2026-10-01",
    }]), judge=StubJudge("supported"))
    assert result.metrics["field_status"] == 1
    assert result.metrics["field_deadline"] == 1
    assert result.metrics["field_terms"] == 0
    assert result.metrics["field_eligibility"] == 0


def test_an_extra_property_does_not_break_grading(benchmark):
    """Agents add fields the schema did not name; that must cost nothing."""
    url = cache_source("https://example.com/")
    result = benchmark.score(task(), answer([{
        "name": "Example Program", "url": url, "status": "open",
        "notes": "extra", "confidence": 0.9,
    }]), judge=StubJudge("supported"))
    assert result.metrics["programs_returned"] == 1
    assert result.metrics["schema_failures"] == 0














def test_a_name_that_begins_with_a_domain_is_still_a_name(mod):
    """A program can be called after the site that lists it.

    `SBIR.gov Participating Federal Agencies (America's Seed Fund directory)` matches the
    citation pattern at position zero, so cutting there left an empty prefix, cleaning it
    returned None, and scoring crashed on `.strip`. The cut is only right when something
    is left of it.
    """
    name = "SBIR.gov Participating Federal Agencies (America's Seed Fund directory)"
    assert mod._program_name(name) == name
    # A name that is nothing but a citation keeps its text too. It is what the answer
    # stated, and the judge should rule on it rather than the scorer inventing a
    # `None` -- which is what the crash amounted to. Nothing merges on an empty name,
    # so the only thing losing it would buy is silence.
    assert mod._program_name("https://example.com/apply") == "https://example.com/apply"
    assert mod._program_name("") is None
