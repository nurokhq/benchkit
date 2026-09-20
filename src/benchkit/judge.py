"""Judging, for benchmarks whose answers cannot be checked against a fixed key.

A judge here is deliberately narrow: it verifies **specific claims against a specific
source**, one item at a time. That is a far tighter task than "rate this answer", and
it is what makes an open-ended benchmark gradeable without a gold list.

Two rules keep it honest:

* The judge never sees which condition produced an answer, and never sees a gold
  answer list. It sees one item and the source that item cites.
* The judge must answer about the source, not about its own opinion of the program.
  When the source does not state something, the correct answer is "not stated", not
  a guess.

An item the judge cannot ground in a given source scores nothing for that claim
rather than defaulting to credit.

Verdicts are kept distinct because conflating them biases the score:

``supported`` / ``contradicted`` / ``not_stated``
    What the source says about the claim.
``not_published``
    The source explicitly records that the information is not published. This is a
    finding, not a gap: an answer reporting a documented absence has told the reader
    something true and checkable, and scoring it as silence penalises exactly the
    sources that document their absences.
``unverifiable``
    No verdict is available because the source could not be read at all. That is a
    fact about the grading run, not about the answer. The judge never returns it;
    the caller assigns it when a cited page cannot be fetched, and it is excluded
    from the score rather than counted as a failed claim.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .llm import LLMError

#: Verdicts the judge may return. `unverifiable` is deliberately absent: only the
#: caller knows whether a source could be read, and a judge that could return it
#: would have a way to abstain on a claim it merely found difficult.
VERDICTS = ("supported", "contradicted", "not_stated", "not_published")

UNVERIFIABLE = {"verdict": "unverifiable", "evidence": "", "confidence": "low"}

VERDICT_SYSTEM = (
    "You verify one claim about one program against one source page. You are a fact "
    "checker, not a reviewer: judge only whether the SOURCE states the claim. Never use "
    "outside knowledge, never guess, and never reward plausibility. If the source does "
    "not state the claim, answer \"not_stated\" even when you believe the claim is true. "
    "Answer \"not_published\" only when the source itself says the information is not "
    "published, not available, or not disclosed -- an explicit statement of absence, "
    "not merely a page that omits it. Return JSON only."
)

VERDICT_SCHEMA = {
    "verdict": "one of: supported, contradicted, not_stated, not_published",
    "evidence": "the exact sentence from the source that decides it, or empty when not_stated",
    "confidence": "one of: high, medium, low",
}

#: The question asked of every returned program, whatever condition produced it.
#: Catalog membership does not answer it: awarding the point for being in the key would
#: give a condition that can read the key a point every other condition must earn from
#: a judge.
#:
#: It names the item, and that matters. A fixed sentence -- "the source describes an
#: applyable program" -- is a question about the *page*, and a page listing forty
#: programs answers it yes however wrong the item is. A row the parser misread out of a
#: section heading then collected the same point as a real program. Asking about the
#: named item makes the verdict about the row the reader is being shown.
IS_PROGRAM_TEMPLATE = (
    "The source describes \"{name}\", and \"{name}\" is an incubator, accelerator, "
    "residency, fellowship, grant or startup-credit program that a startup can apply to."
)

#: Whether a citation sits on the operator's own site. Phrased so the page can actually
#: answer it -- the old wording ("this page is the program's official page") asked the
#: page to assert something about itself, so the judge answered `not_stated` for most
#: records and the signal collapsed.
REPUTATION_CLAIM = (
    "This page is published by the organization that runs the program -- the program's "
    "own website or an official subdomain of it -- rather than by a third party such as "
    "a news site, listicle, directory, aggregator, or someone else's blog."
)


def is_program_claim(name: str | None) -> str:
    """The legitimacy claim, stated about one named item."""
    label = (name or "").strip() or "the unnamed item"
    return IS_PROGRAM_TEMPLATE.format(name=label)


def relevance_claim(question: str | None, name: str | None = None) -> str:
    """The question-fit claim, stated about one named item and the task."""
    label = (name or "").strip() or "the unnamed item"
    asked = (question or "").strip()
    base = (f"The source describes \"{label}\", and \"{label}\" is a program that fits "
            f"what the task question asks for, including its sector, stage and program type.")
    return f"{base} Task question: {asked}" if asked else base


#: Record members that describe how the grading run is going rather than what the
#: answer said. They are stripped before the judge sees an item, for three reasons:
#: `source_text` is already sent as the source, so including it doubled the page in
#: every prompt; `source_provenance` changes between a first scoring and a replay
#: ("fetched" then "cache"), which made an identical question hash differently and
#: miss its own cached verdict; and `key` is the benchmark's own answer key, which a
#: claim checker has no business seeing.
_GRADING_INTERNALS = ("source_text", "source_provenance", "url_recovered_from", "key")


def judge_item_fields(item: dict) -> dict:
    """The part of a record a claim checker should see."""
    if not isinstance(item, dict):
        return {}
    return {name: value for name, value in item.items() if name not in _GRADING_INTERNALS}


#: How much of the cited page reaches the judge. This must not be smaller than the
#: resolver's own cap, or the judge is asked about a claim the page may support beyond
#: the window it was shown -- and answers `not_stated`. Stripping the record down to
#: its own fields once silently halved the evidence per verdict (the page was being
#: sent twice, once capped here and once whole inside `item_fields`) and cost the MCP
#: condition 14 points, every one of them a fact the page did state.
SOURCE_TEXT_MAX = 20000


def batch_prompt(item: dict, claims: dict, source_text: str | None, source_url: str | None,
                 question: str | None = None) -> dict:
    """Several claims about one record, against one page, in one call.

    The page is the expensive part of a verdict and it was being sent once per field.
    A record stating four facts, with two structural questions asked three times each,
    put the same page through the model ten times; the reading of it is shared here
    instead. The claims stay separate -- each gets its own verdict -- so nothing is
    averaged and no field can soften another.
    """
    payload = {
        "claims_to_verify": dict(claims),
        "item_fields": judge_item_fields(item),
        "source_url": source_url,
        "source_text": (source_text or "")[:SOURCE_TEXT_MAX],
        "instructions": "Judge every claim in `claims_to_verify` separately and return "
                        '{"verdicts": {"<claim name>": {...}}} where each value follows '
                        "this schema: " + json.dumps(VERDICT_SCHEMA)
                        + " Use the claim's own name as the key, exactly as given.",
    }
    if question:
        payload["task_question"] = question
    return payload


class VerdictCache:
    """A content-addressed store of judge verdicts, so a regrade is reproducible.

    The judge runs at temperature 0, but a temperature-0 LLM still samples: scoring
    the same six answers four times gave 123.5, 129.5, 132.0 and 136.5 -- a 10% spread
    with the fetched pages held constant. That noise is larger than every difference
    this benchmark has been asked to resolve. Caching each verdict against the exact
    question that produced it removes it from the measurement entirely: a re-scoring of
    the same answers becomes bit-for-bit identical, and a difference in the result can
    only come from a difference in the answers or the pages they cite.

    The key covers everything that can change a verdict -- the system prompt, the full
    payload including the source text, the model, its reasoning effort, and the
    temperature actually sent. A page whose text changes therefore misses the cache and
    is judged afresh, which is the point: the cache pins the *decision*, not the claim.

    Entries are one small JSON object per verdict, named by the digest, so a cache is
    inspectable with `grep` and safe to share between runs and processes.
    """

    def __init__(self, directory, enabled=True):
        self.directory = Path(directory) if directory else None
        self.enabled = bool(enabled and directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.stats = {"hit": 0, "miss": 0, "write": 0}

    @staticmethod
    def key(system: str, payload: dict, model, effort, temperature, sample: int = 0,
            max_tokens: int | None = None) -> str:
        """Digest of everything that can change a verdict, including which sample.

        `sample` is part of the key on purpose. Asking the same question k times to
        take a majority only helps if the k answers are independent, and a cache that
        returned one verdict for all k would silently turn self-consistency into a
        no-op.

        `max_tokens` is part of it because it changes the answer, not just the room:
        a reasoning model expands to fill the budget it is given, and the same call
        used 2,319 output tokens at a 4,096 ceiling and 3,378 at 8,192. A verdict
        decided under one ceiling must not answer for another.
        """
        body = json.dumps(
            {"system": system, "payload": payload, "model": model,
             "effort": effort, "temperature": temperature, "sample": sample,
             "max_tokens": max_tokens},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def get(self, key):
        if not self.enabled:
            return None
        path = self.directory / key[:2] / f"{key}.json"
        if not path.is_file():
            self.stats["miss"] += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated entry is a miss, not a verdict. Re-judging costs one call;
            # trusting a half-written file would put junk into the score.
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return entry

    def put(self, key, entry):
        if not self.enabled:
            return
        path = self.directory / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temporary name and renamed, so a crash or a concurrent reader
        # never sees a partial entry.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        self.stats["write"] += 1


class Judge:
    """Wraps an LLM client with the verification prompts.

    `client` only needs `complete_json(system, user) -> dict`, so tests can pass a
    stub. Failures raise `LLMError` and are surfaced by the caller rather than being
    silently scored as zero, which would report a grading failure as a bad answer.

    `cache` is optional. With one, a verdict already decided for an identical question
    is reused rather than re-sampled, which is what makes repeated scoring of the same
    answers agree.
    """

    def __init__(self, client, cache=None, samples=1):
        self.client = client
        self.cache = cache
        #: How many independent verdicts to take per question, keeping the majority.
        #: See `verify` for why this is worth the calls on some questions and not
        #: others.
        self.samples = max(1, int(samples))
        self.cache_stats = {"hit": 0, "miss": 0}

    def _decide_many(self, payload: dict, sample: int, names: tuple[str, ...]) -> dict:
        """One batched verdict set, from the cache when it is there and the model when not."""
        key = None
        if self.cache is not None:
            key = VerdictCache.key(VERDICT_SYSTEM, payload, getattr(self.client, "model", None),
                                   getattr(self.client, "reasoning_effort", None),
                                   getattr(self.client, "effective_temperature", None),
                                   sample, getattr(self.client, "BATCH_MAX_TOKENS", None))
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_stats["hit"] += 1
                return cached
            self.cache_stats["miss"] += 1
        # A batched call answers every field of a record at once, so it needs room for
        # all of them; the per-verdict ceiling truncates it before it writes any JSON.
        returned = self.client.judge(VERDICT_SYSTEM, payload,
                                     max_tokens=getattr(self.client, "BATCH_MAX_TOKENS", None))
        given = returned.get("verdicts") if isinstance(returned, dict) else None
        if not isinstance(given, dict):
            raise LLMError(f"judge returned no `verdicts` object: {str(returned)[:120]!r}")
        if len(names) == 1 and names[0] not in given and len(given) == 1:
            # A one-entry batch is answered under whatever key the model prefers --
            # given one claim and one verdict there is nothing to disambiguate, and
            # requiring the exact key failed a call that had in fact been answered.
            # A batch of several stays strict, where a missing key is a real omission.
            given = {names[0]: next(iter(given.values()))}
        decided = {}
        for name in names:
            entry = given.get(name)
            if not isinstance(entry, dict):
                # A missing key is a failed answer, not a `not_stated`: the judge was
                # asked about this claim and did not answer it. Recording silence as a
                # verdict would credit an omission the answer never made.
                raise LLMError(f"judge omitted {name!r} from a batched answer")
            result = str(entry.get("verdict") or "").strip().casefold()
            if result not in VERDICTS:
                raise LLMError(f"judge returned an unusable verdict for {name!r}: "
                               f"{entry.get('verdict')!r}")
            confidence = str(entry.get("confidence") or "").strip().casefold()
            decided[name] = {
                "verdict": result,
                "evidence": str(entry.get("evidence") or "")[:600],
                "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
            }
        if self.cache is not None:
            self.cache.put(key, decided)
        return decided

    def verify_many(self, item: dict, claims: dict, source_text: str | None = None,
                    source_url: str | None = None, question: str | None = None,
                    samples: int | None = None) -> dict:
        """Decide several claims about one record in one call per sample.

        Same decision rule as `verify`, applied to the whole set at once: samples are
        taken until two agree, because two agreeing samples are already a majority of
        three and a third cannot change the outcome. Here that saves a call per *record*
        rather than per claim.
        """
        names = tuple(claims)
        if not names:
            return {}
        wanted = max(1, self.samples if samples is None else samples)
        payload = batch_prompt(item, claims, source_text, source_url, question)
        taken: list[dict] = []
        for index in range(wanted):
            taken.append(self._decide_many(payload, index, names))
            if len(taken) == 2 and wanted == 3:
                if all(taken[0][name]["verdict"] == taken[1][name]["verdict"]
                       for name in names):
                    break
        return {
            name: self._majority([sample[name] for sample in taken]) if len(taken) > 1
            else taken[0][name]
            for name in names
        }

    @staticmethod
    def _majority(verdicts: list[dict]) -> dict:
        """The verdict a strict majority agrees on, or the conservative one.

        A tie abstains rather than picking a side: `not_stated` awards nothing, so an
        unsettled question cannot become a point either way. That is the same reason a
        contradicted claim is negative and an unverifiable one is zero -- the score
        should reward what the sources settle, not what the judge happened to say.
        """
        if not verdicts:
            raise LLMError("no verdicts to decide on")
        if len(verdicts) == 1:
            return verdicts[0]
        tally = Counter(entry["verdict"] for entry in verdicts)
        winner, count = tally.most_common(1)[0]
        if count * 2 <= len(verdicts):
            winner = "not_stated"
        for entry in verdicts:
            if entry["verdict"] == winner:
                return {**entry, "agreement": f"{count}/{len(verdicts)}"}
        return {**verdicts[0], "verdict": winner, "evidence": "", "confidence": "low",
                "agreement": f"{count}/{len(verdicts)}"}

    def verify(self, item: dict, claim: str, source_text: str | None = None,
               source_url: str | None = None, question: str | None = None,
               samples: int | None = None) -> dict:
        """One claim, decided by the same path as a batch of them.

        Kept for callers that rule on a single contested claim
        (`tools/judge_agreement.py`) and for tests. It goes through `verify_many` rather
        than keeping its own prompt and its own sampling loop: two paths deciding the
        same question by different means is how the measurement the tools rely on drifts
        away from the measurement the scores are built from.
        """
        return self.verify_many(item, {"claim": claim}, source_text=source_text,
                                source_url=source_url, question=question,
                                samples=samples)["claim"]

    def is_program(self, item: dict, source_text: str | None = None) -> dict:
        """Is this actually an accelerator/incubator/funding program, not a company?"""
        return self.verify(item, is_program_claim(item.get("name")), source_text, item.get("url"))


def verdict_score(verdict: dict, supported: float = 1.0, contradicted: float = -1.0,
                  not_stated: float = 0.0, not_published: float = 0.0) -> float:
    """Map a verdict to a score contribution.

    A contradicted claim is negative on purpose: a confidently wrong answer should be
    worse than an omitted one, so padding a list with unverifiable claims cannot pay.
    `unverifiable` is absent from the mapping and so scores 0 through the default --
    but callers that build a denominator must exclude it explicitly, because a claim
    nobody could check is not a claim the answer got wrong.
    """
    return {"supported": supported, "contradicted": contradicted,
            "not_stated": not_stated, "not_published": not_published}.get(
        verdict.get("verdict"), 0.0)
