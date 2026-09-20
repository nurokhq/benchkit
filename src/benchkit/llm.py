"""Lazy LiteLLM bridge, used only when a model is actually configured."""

import json
import os
import random
import threading
import time


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """A process-wide cap on in-flight model calls.

    Tasks run concurrently and each judges its own answer, so without a shared ceiling
    the number of simultaneous requests multiplies by the worker count and providers
    start returning 429s. A semaphore is enough: throughput is bounded by the network
    anyway, and waiting is cheaper than being throttled.
    """

    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self.peak = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def __enter__(self):
        self._semaphore.acquire()
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._in_flight -= 1
        self._semaphore.release()
        return False


def _content(response) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMError("LiteLLM returned no assistant message") from exc
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        raise LLMError("LiteLLM assistant content was not text")
    return content


#: Shared by every client in the process; set once from the CLI.
#:
#: This is the one ceiling that binds. `--workers` and `--judge-workers` only decide how
#: many callers *want* to run, and every one of them queues here: three tasks asking for
#: eight judge workers each is twenty-four callers behind a semaphore of eight, so two
#: thirds of the requested concurrency did nothing. Sized for the callers rather than for
#: a single task now. Raising it risks provider 429s, which is what this exists to
#: prevent, so it stays tunable per run with `--max-inflight`.
LIMITER = RateLimiter(int(os.environ.get("BENCHKIT_MAX_INFLIGHT", "32")))


class LiteLLMClient:
    #: How many times a request is retried before its failure is reported. A judge call
    #: that fails leaves its claim unverifiable, which is excluded from the score -- so a
    #: transient 429 does not merely slow a run down, it silently removes a fact from the
    #: comparison. Five attempts, from a two-second base with jitter, is what a round of
    #: judging needs: one round lost 152 consecutive verdicts on a single task, every one
    #: of them scored as though the answer had said nothing.
    ATTEMPTS = 5
    BACKOFF_SECONDS = 2.0

    #: Ceiling on one response. Every call here answers with a small JSON object, and
    #: asking a provider for its default maximum instead reserves the model's whole
    #: context against the account: OpenRouter refused every verdict with "you requested
    #: up to 131072 tokens, but can only afford 32410" once the balance ran down, which
    #: silently turned every claim in a regrade into an excluded one.
    #:
    #: This is the ceiling for *one* verdict, which is what it was sized for. A call that
    #: answers several at once needs its own: a reasoning model spends the budget
    #: thinking before it writes anything, so a batched call over a full page hit this
    #: cap, returned empty content, failed to parse, and was retried five times -- 5,000
    #: input and 2,048 output tokens per attempt, thirty seconds of backoff, and no
    #: verdict at all. Judging ran at six calls a minute.
    MAX_TOKENS = 2048
    #: Room for a record's worth of verdicts in one call. Measured on the worst page in
    #: the corpus -- 20,000 characters, four claims -- which needs 2,319 output tokens;
    #: 8,192 also works but reasoning expands to fill the budget and costs 45% more.
    BATCH_MAX_TOKENS = 4096
    #: How far truncation may raise the ceiling before it is a failure. Truncation is
    #: self-correcting: the budget doubles and the call is retried, so a page longer than
    #: any yet seen costs one retry rather than a lost verdict.
    TOKENS_CEILING = 16384

    #: Prompt/completion tokens this client has spent, summed over every call. Judging
    #: is a real cost that grows with the number of claims an answer states, and it was
    #: invisible: a round reported the agent's bill and not the verifier's. Switching
    #: the judge to a cheaper route is only checkable if the route's usage is recorded.
    USAGE_KEYS = ("prompt_tokens", "completion_tokens")

    def __init__(self, model, api_key=None, api_base=None, temperature=0.0,
                 reasoning_effort=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_base = api_base or os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.temperature = temperature
        #: How hard a reasoning model should think before answering. Deciding whether a
        #: page supports a claim is exactly the call that benefits; providers without
        #: the parameter ignore it rather than failing, so this is safe to set broadly.
        #: `None` leaves the provider's own default alone.
        self.reasoning_effort = reasoning_effort
        if not self.api_key:
            raise LLMError("Set OPENROUTER_API_KEY or pass api_key before using LiteLLM.")

    def _record_usage(self, usage) -> None:
        if usage is None:
            return
        if not hasattr(self, "usage"):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for key in self.USAGE_KEYS:
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value
        self.usage["calls"] = self.usage.get("calls", 0) + 1

    @property
    def effective_temperature(self) -> float:
        """The temperature actually sent.

        A reasoning model pins sampling, and not all providers say so politely: asking
        `gpt-5.6-terra` for `temperature=0` while reasoning is active is rejected
        outright -- only 1 is accepted unless the effort resolves to `none`. So an
        active effort forces 1 rather than sending a value the provider will refuse.

        That is worth knowing when reading a score: it means the judge samples, and two
        regrades of the same answers can differ. Measured spread on an unchanged run was
        about 3%.
        """
        if self.reasoning_effort and self.reasoning_effort != "none":
            return 1.0
        return self.temperature

    def complete_json(self, system, user, max_tokens: int | None = None) -> dict:
        """One JSON object from the model, retried as a whole.

        The retry covers the *answer*, not just the transport. A reasoning model
        occasionally returns a response that arrives but carries no text, and that is
        the same class of failure as a dropped request: it should cost one call, not the
        run. Both sit inside the loop for that reason.
        """
        options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        room = max_tokens or self.MAX_TOKENS
        last: Exception | None = None
        for attempt in range(self.ATTEMPTS):
            try:
                from litellm import completion
                with LIMITER:
                    response = completion(
                        model=self.model, api_key=self.api_key, api_base=self.api_base,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=self.effective_temperature,
                        response_format={"type": "json_object"},
                        max_tokens=room,
                        **options,
                    )
                # A response cut off at the ceiling carries no JSON to parse. Named here
                # rather than left to surface as "Expecting value: line 1 column 1",
                # because the two need different answers: this one wants more room, and
                # a parse error does not.
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    if room < self.TOKENS_CEILING:
                        room = min(room * 2, self.TOKENS_CEILING)
                        raise LLMError(f"response truncated at {room // 2} tokens; "
                                       f"retrying with {room}")
                    raise LLMError(f"response truncated at the {room}-token ceiling")
                value = json.loads(_content(response))
                if not isinstance(value, dict):
                    raise LLMError("LiteLLM JSON result must be an object")
                self._record_usage(getattr(response, "usage", None))
                return value
            except Exception as exc:  # litellm, json and the shape check all retry
                last = exc
                if attempt + 1 == self.ATTEMPTS:
                    break
                # Jittered exponential backoff. A fixed 2s/4s ladder synchronises every
                # worker onto the same retry instants, so a burst of rate limiting is
                # answered by a burst of retries. Judging a round runs dozens of calls
                # at once and this is the path that recovers them.
                delay = self.BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(delay * (0.5 + random.random()))
        raise LLMError(f"LiteLLM request failed after {self.ATTEMPTS} attempts: {last}") from last

    def normalize(self, question, answer_text, expected_fields=None) -> dict:
        """Extract facts from prose, reusing the caller's field names.

        Record-style answers are scored per field, so a fact filed under an invented
        name counts as missing even when the answer is correct. When the caller
        declares field names, the model must use them and keep JSON types intact.
        """
        instructions = (
            "Extract facts from the answer only. Never browse, infer, correct, or add missing facts. "
            "Copy names and URLs exactly as written, and report a thing under `items` only when the "
            "answer presents it as an answer rather than mentioning it in passing. Put single "
            "aggregate numbers in `scalars`. Return JSON."
        )
        payload = {"question": question, "answer_text": answer_text, "schema": {
            "items": [{"name": "string", "url": "string"}], "fields": {}, "tables": [],
            "scalars": {}, "count": "integer or null", "citations": [],
        }}
        if expected_fields:
            instructions += (
                " Use exactly these `fields` keys and no others: " + ", ".join(expected_fields) + "."
                " Preserve the original JSON type of every value: a boolean as true or false, a number"
                " as a number, and anything the answer does not state as null."
            )
            payload["fields_to_fill"] = list(expected_fields)
        return self.complete_json(instructions, json.dumps(payload, ensure_ascii=False))

    def judge(self, system, payload, max_tokens=None) -> dict:
        return self.complete_json(system, json.dumps(payload, ensure_ascii=False),
                                  max_tokens=max_tokens)
