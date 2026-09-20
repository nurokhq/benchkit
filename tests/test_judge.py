"""Tests for the claim verifier.

The judge is the one component whose behaviour a reader cannot inspect from a score,
so the properties that keep it honest are pinned here: it may not invent a verdict
outside the schema, it may not abstain on a claim by claiming the source is
unreadable (only the caller knows that), and the task question reaches it for the
claim kinds that need it and no others.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchkit.judge import (REPUTATION_CLAIM, UNVERIFIABLE, Judge, VERDICTS,  # noqa: E402
                            VerdictCache, is_program_claim, relevance_claim,
                            batch_prompt, verdict_score)
from benchkit.llm import LLMError  # noqa: E402


class StubClient:
    """Returns a canned payload and records what it was asked.

    The payload is a batched envelope, because there is only one judging path now:
    a single claim travels as a one-entry batch.
    """

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def judge(self, system, payload, max_tokens=None):
        self.calls.append((system, payload))
        return self.payload


def test_every_verdict_in_the_schema_maps_to_a_score():
    for verdict in VERDICTS:
        assert verdict_score({"verdict": verdict}) is not None


def test_not_published_is_a_verdict_the_judge_may_return():
    """An explicit "the operator publishes no such page" is a finding, not silence."""
    client = StubClient({"verdicts": {"claim": {"verdict": "not_published",
                                                "evidence": "not published",
                                                "confidence": "high"}}})
    result = Judge(client).verify({"name": "X"}, "X deadline: none")
    assert result["verdict"] == "not_published"
    assert verdict_score(result, not_published=0.5) == 0.5
    assert verdict_score(result) == 0.0


def test_an_unknown_verdict_is_an_error_not_a_zero():
    """A malformed verdict must surface, not be silently scored as one."""
    client = StubClient({"verdicts": {"claim": {"verdict": "maybe", "evidence": "",
                                                "confidence": "high"}}})
    with pytest.raises(Exception):
        Judge(client).verify({"name": "X"}, "X status: open")


def test_unverifiable_is_not_a_verdict_the_judge_may_return():
    """Abstaining for an unreadable source is the caller's call, not the judge's.

    A judge allowed to answer `unverifiable` has a way to abstain on a claim it merely
    finds difficult, and every such abstention would be excluded from the score.
    """
    assert "unverifiable" not in VERDICTS
    assert UNVERIFIABLE["verdict"] == "unverifiable"
    client = StubClient({"verdicts": {"claim": {"verdict": "unverifiable", "evidence": "",
                                                "confidence": "low"}}})
    with pytest.raises(Exception):
        Judge(client).verify({"name": "X"}, "X status: open")


def test_the_task_question_reaches_the_judge_only_when_asked_for():
    question = "Which AI pre-seed programs should I apply to?"
    with_question = batch_prompt({"name": "X"}, {"claim": relevance_claim(question)}, "text", "https://x/",
                                   question=question)
    without = batch_prompt({"name": "X"}, {"claim": "X status: open"}, "text", "https://x/")
    assert with_question["task_question"] == question
    assert question in with_question["claims_to_verify"]["claim"]
    assert "task_question" not in without


def test_the_structural_claims_are_the_same_text_for_every_condition():
    """One definition each, so no condition is asked a softer question than another."""
    assert "startup-credit program" in is_program_claim("Anything")
    assert "published by the organization" in REPUTATION_CLAIM
    assert relevance_claim(None, "X") and relevance_claim("q", "X").startswith(
        relevance_claim(None, "X"))


def test_the_structural_claims_name_the_item():
    """A claim about the page is answered yes by any page listing programs.

    The old wording -- "the source describes an applyable program" -- was a question
    about the page, so a row the parser misread out of a section heading collected the
    same legitimacy point as a real program, as long as its citation pointed at a page
    full of programs.
    """
    assert '"Scope and method"' in is_program_claim("Scope and method")
    assert '"Scope and method"' in relevance_claim("which programs?", "Scope and method")
    assert "unnamed item" in is_program_claim(None)


def test_an_item_specific_claim_reaches_the_client():
    client = StubClient({"verdicts": {"claim": {"verdict": "contradicted", "evidence": "",
                                                "confidence": "high"}}})
    Judge(client).is_program({"name": "Scope and method", "url": "https://sbir.gov/"})
    _, payload = client.calls[0]
    assert "Scope and method" in payload["claims_to_verify"]["claim"]


class CountingClient:
    """A client that returns a different verdict per call, to expose re-sampling."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0
        self.model = "stub/model"
        self.reasoning_effort = None
        self.effective_temperature = 0.0

    def judge(self, system, payload, max_tokens=None):
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return {"verdicts": {"claim": {"verdict": verdict, "evidence": "stub",
                                       "confidence": "high"}}}


class BatchedClient:
    """Answers every claim in a batch, so one call yields a whole record's verdicts."""

    def __init__(self, per_call):
        #: A list of {name: verdict} dicts, one per call.
        self.per_call = list(per_call)
        self.calls = 0
        self.payloads = []
        self.max_tokens = []
        self.model = "stub/model"
        #: Mirrors the real client, so the batched path can ask for more room than the
        #: per-verdict ceiling -- which is what truncation made necessary.
        self.BATCH_MAX_TOKENS = 4096
        self.reasoning_effort = None
        self.effective_temperature = 0.0

    def judge(self, system, payload, max_tokens=None):
        self.payloads.append(payload)
        self.max_tokens.append(max_tokens)
        given = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return {"verdicts": {name: {"verdict": verdict, "evidence": "stub",
                                    "confidence": "high"}
                             for name, verdict in given.items()}}


def test_a_batch_answers_every_claim_in_one_call(tmp_path):
    """The page is sent once for the whole record instead of once per field."""
    client = BatchedClient([{"status": "supported", "deadline": "not_stated"}])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    got = judge.verify_many({"name": "X"}, {"status": "X status: open",
                                            "deadline": "X deadline: none"},
                            source_text="the page", source_url="https://x/")
    assert client.calls == 1
    assert got["status"]["verdict"] == "supported"
    assert got["deadline"]["verdict"] == "not_stated"
    # One payload, carrying both claims and the page exactly once.
    assert len(client.payloads) == 1
    assert set(client.payloads[0]["claims_to_verify"]) == {"status", "deadline"}
    assert client.payloads[0]["source_text"] == "the page"
    assert client.max_tokens == [4096], "a batch must ask for room for every verdict"


def test_a_batch_keeps_each_claim_separate(tmp_path):
    """No field may soften another: each verdict is decided on its own."""
    client = BatchedClient([{"status": "supported", "eligibility": "contradicted"}])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    got = judge.verify_many({"name": "X"}, {"status": "s", "eligibility": "e"},
                            source_text="p", source_url="https://x/")
    assert got["status"]["verdict"] == "supported"
    assert got["eligibility"]["verdict"] == "contradicted"


def test_a_batch_stops_sampling_once_the_whole_set_agrees(tmp_path):
    """Two identical samples settle every claim in the batch, so a third is wasted."""
    client = BatchedClient([{"status": "supported", "deadline": "supported"}])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    judge.verify_many({"name": "X"}, {"status": "s", "deadline": "d"},
                      source_text="p", source_url="https://x/", samples=3)
    assert client.calls == 2


def test_a_batch_resamples_when_any_claim_disagrees(tmp_path):
    """One unsettled claim in the set is enough to need the tie-breaker."""
    client = BatchedClient([
        {"status": "supported", "deadline": "supported"},
        {"status": "supported", "deadline": "contradicted"},
        {"status": "supported", "deadline": "supported"},
    ])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    got = judge.verify_many({"name": "X"}, {"status": "s", "deadline": "d"},
                            source_text="p", source_url="https://x/", samples=3)
    assert client.calls == 3
    assert got["deadline"]["verdict"] == "supported"


def test_an_omitted_claim_in_a_batch_is_a_failure_not_silence(tmp_path):
    """The judge was asked and did not answer; that is not `not_stated`.

    Recording an omission as a verdict would credit the answer for a question nobody
    decided, and `not_stated` is exactly the verdict an omission looks like.
    """
    client = BatchedClient([{"status": "supported"}])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    with pytest.raises(LLMError):
        judge.verify_many({"name": "X"}, {"status": "s", "deadline": "d"},
                          source_text="p", source_url="https://x/")


def test_a_cached_verdict_is_reused_instead_of_re_sampled(tmp_path):
    """The whole point: the same question must not get two different answers.

    Scoring the same six answers four times moved the total by 10% because the judge
    samples even at temperature 0. That noise is larger than every difference this
    benchmark has been asked to resolve, so the decision is pinned to the question.
    """
    client = CountingClient(["supported", "contradicted"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    first = judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    second = judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    assert first == second
    assert first["verdict"] == "supported"
    assert client.calls == 1, "the second identical question must not reach the model"
    assert judge.cache_stats == {"hit": 1, "miss": 1}


def test_a_different_question_is_not_served_from_the_cache(tmp_path):
    """The key covers the claim, the source text and the model, so none can be confused."""
    client = CountingClient(["supported", "contradicted", "not_stated"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    judge.verify({"name": "X"}, "X status: closed", source_text="open", source_url="https://x/")
    changed_page = judge.verify({"name": "X"}, "X status: open",
                                source_text="applications are closed", source_url="https://x/")
    assert client.calls == 3
    assert changed_page["verdict"] == "not_stated"


def test_a_reworded_page_is_judged_afresh(tmp_path):
    """A cache keyed on the URL alone would pin a verdict to a page that changed."""
    client = CountingClient(["supported", "contradicted"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    judge.verify({"name": "X"}, "X status: open", source_text="v1 text", source_url="https://x/")
    after = judge.verify({"name": "X"}, "X status: open", source_text="v2 text", source_url="https://x/")
    assert after["verdict"] == "contradicted"
    assert client.calls == 2


def test_without_a_cache_every_call_reaches_the_model(tmp_path):
    client = CountingClient(["supported", "contradicted"])
    judge = Judge(client)
    judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    assert client.calls == 2


def test_a_corrupt_cache_entry_is_a_miss_not_a_verdict(tmp_path):
    """A half-written entry must not become a score."""
    cache = VerdictCache(tmp_path)
    client = CountingClient(["supported"])
    judge = Judge(client, cache=cache)
    judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    (tmp_path / "00").mkdir(exist_ok=True)
    for entry in tmp_path.rglob("*.json"):
        entry.write_text('{"verdict": "supp', encoding="utf-8")
    again = judge.verify({"name": "X"}, "X status: open", source_text="open", source_url="https://x/")
    assert again["verdict"] == "supported"
    assert client.calls == 2


def test_the_judge_does_not_see_grading_internals():
    """`key` is the benchmark's own answer key and `source_text` is sent separately.

    `source_provenance` rewrites itself between a scoring and a replay ("fetched",
    then "cache"), which also made an identical question hash to a different cache
    key and miss its own verdict.
    """
    from benchkit.judge import judge_item_fields
    record = {"name": "X", "url": "https://x/", "status": "open",
              "source_text": "the page", "source_provenance": "cache",
              "url_recovered_from": "X", "key": "x-key"}
    assert judge_item_fields(record) == {"name": "X", "url": "https://x/", "status": "open"}


def test_the_page_is_not_repeated_in_the_prompt():
    """It was in `item_fields` and again as `source_text`, doubling every prompt."""
    payload = batch_prompt({"name": "X", "source_text": "the page"}, {"claim": "X status: open"},
                           "the page", "https://x/")
    assert "source_text" not in payload["item_fields"]
    assert payload["source_text"] == "the page"


def test_majority_of_three_settles_a_flipping_question(tmp_path):
    """A question answered inconsistently should land on the majority, not on luck."""
    client = CountingClient(["supported", "contradicted", "supported"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    verdict = judge.verify({"name": "X"}, "is X a program", source_text="p", samples=3)
    assert verdict["verdict"] == "supported"
    assert verdict["agreement"] == "2/3"
    assert client.calls == 3


def test_a_tied_majority_abstains(tmp_path):
    """An unsettled question must not become a point either way."""
    client = CountingClient(["supported", "contradicted"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    verdict = judge.verify({"name": "X"}, "is X a program", source_text="p", samples=2)
    assert verdict["verdict"] == "not_stated"
    assert verdict_score(verdict) == 0.0


def test_an_agreed_pair_does_not_buy_a_third_verdict(tmp_path):
    """Once two samples agree they are a majority of three, so the third is wasted.

    The structural questions are asked of every returned program with three samples,
    which made them 55% of a round's judging. This is not a cheaper approximation of
    the majority-of-three answer: when the first two agree, that agreement *is* the
    majority of three, whatever a third sample would have said.
    """
    client = CountingClient(["supported", "supported", "contradicted"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    verdict = judge.verify({"name": "X"}, "is X a program", source_text="p", samples=3)
    assert client.calls == 2, "the third verdict could not change the outcome"
    assert verdict["verdict"] == "supported"


def test_a_split_pair_still_buys_the_third_verdict(tmp_path):
    """Two samples that disagree settle nothing, so the tie-breaker is still asked."""
    client = CountingClient(["supported", "contradicted", "supported"])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    verdict = judge.verify({"name": "X"}, "is X a program", source_text="p", samples=3)
    assert client.calls == 3
    assert verdict["verdict"] == "supported"


def test_each_sample_is_cached_separately(tmp_path):
    """A cache that returned one verdict for all k would make self-consistency a no-op."""
    cache = VerdictCache(tmp_path)
    client = CountingClient(["supported", "contradicted", "supported"])
    first = Judge(client, cache=cache).verify({"name": "X"}, "is X a program",
                                              source_text="p", samples=3)
    assert client.calls == 3
    second = Judge(client, cache=cache).verify({"name": "X"}, "is X a program",
                                               source_text="p", samples=3)
    assert client.calls == 3, "all three samples must come from cache on a replay"
    assert second["verdict"] == first["verdict"] == "supported"


# --- the output ceiling -----------------------------------------------------

class _FakeResponse:
    def __init__(self, content, finish_reason):
        self.choices = [type("C", (), {
            "message": type("M", (), {"content": content})(),
            "finish_reason": finish_reason})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()


def test_a_truncated_response_buys_more_room_instead_of_failing(monkeypatch):
    """A reasoning model spends the budget before it writes anything.

    A batched call over a full page hit the per-verdict ceiling, returned empty content
    and failed to parse -- and the retry asked for the same ceiling again, five times,
    for 5,000 input and 2,048 output tokens an attempt and no verdict at all. Judging
    ran at six calls a minute. Truncation is self-correcting now: the budget doubles.
    """
    import benchkit.llm as llm
    seen = []

    def fake_completion(**kwargs):
        seen.append(kwargs["max_tokens"])
        if len(seen) == 1:
            return _FakeResponse("", "length")
        return _FakeResponse('{"verdict": "supported"}', "stop")

    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setitem(sys.modules, "litellm",
                        type("L", (), {"completion": staticmethod(fake_completion)})())
    client = llm.LiteLLMClient("stub/model", api_key="k")
    got = client.complete_json("sys", "user", max_tokens=client.BATCH_MAX_TOKENS)
    assert got == {"verdict": "supported"}
    assert seen == [4096, 8192], "the retry must ask for more, not the same again"


def test_truncation_at_the_ceiling_is_a_named_failure(monkeypatch):
    """Once doubling cannot help, say so -- do not let it read as a parse error."""
    import benchkit.llm as llm

    def always_truncated(**_kwargs):
        return _FakeResponse("", "length")

    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setitem(sys.modules, "litellm",
                        type("L", (), {"completion": staticmethod(always_truncated)})())
    client = llm.LiteLLMClient("stub/model", api_key="k")
    with pytest.raises(llm.LLMError, match="truncated"):
        client.complete_json("sys", "user", max_tokens=client.TOKENS_CEILING)


def test_the_output_ceiling_is_part_of_the_cache_key():
    """A verdict decided under one ceiling must not answer for another.

    The ceiling changes the answer, not just the room: a reasoning model expands to
    fill the budget, and the same call used 2,319 output tokens at 4,096 and 3,378 at
    8,192. Keyed without it, a cache filled before the batch ceiling existed would
    answer for calls that now reason further.
    """
    base = dict(system="s", payload={"a": 1}, model="m", effort=None, temperature=0.0)
    assert VerdictCache.key(**base, max_tokens=2048) != VerdictCache.key(**base, max_tokens=4096)
    assert VerdictCache.key(**base, max_tokens=2048) == VerdictCache.key(**base, max_tokens=2048)


def test_a_one_claim_batch_accepts_the_key_the_model_chose(tmp_path):
    """Given one claim and one verdict there is nothing to disambiguate.

    The model sometimes re-keys a one-entry batch -- it is asked for `{"claim": ...}` and
    answers under a key of its own choosing. Requiring the exact key failed a call that
    had in fact been answered. Several claims stay strict, where a missing key is a real
    omission.
    """
    client = BatchedClient([{"the_claim": "supported"}])
    judge = Judge(client, cache=VerdictCache(tmp_path))
    got = judge.verify_many({"name": "X"}, {"claim": "X is a program"}, source_text="p",
                            source_url="https://x/", samples=1)
    assert got["claim"]["verdict"] == "supported"
