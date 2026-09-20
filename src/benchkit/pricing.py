"""Cost from tokens, priced per model.

The harness records the cost the CLI reports, and the CLI prices anything it does not
recognise at its own default rates. Pointed at a third-party Anthropic-compatible
endpoint that number is not the bill: a one-line prompt to `deepseek-flash` came back
at $0.1131, which is Opus's rate applied to the same tokens -- roughly 17x what
DeepSeek charges for them. A comparison that mixed the two would rank providers by
whichever rate the CLI happened to guess.

So cost is recomputed here from the token counts against a published price table, and
the figure the CLI reported is kept beside it rather than discarded, so the two can be
compared. `None` means no price is known for that model: callers should then fall back
to the reported number rather than inventing one.

Prices are per 1,000,000 tokens, in USD, and are the vendor's list price at the time
of writing. They change; the table is data, not logic, so updating it is a one-line
edit and the date is recorded next to it.

**The CLI's budget guard uses its own rates too.** `--max-budget-usd` is enforced
against the same unrecognised-model estimate, so on a third-party endpoint it trips at
roughly 1/33rd of the money it names: a $4 cap stops a `deepseek-flash` run after
about twelve cents of real spend. Any cap set for such a run has to be scaled by that
ratio to mean what it says. Nothing here adjusts it, because silently rewriting a
caller's budget is worse than the surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

#: Prices last checked 2026-09-15 against the vendors' own pages:
#: Anthropic (console rates) and DeepSeek https://api-docs.deepseek.com/quick_start/pricing
PRICES_CHECKED = "2026-09-15"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens.

    `cache_read` is the discounted rate for a context-cache hit. `cache_write` is what
    an explicit cache write costs; where a provider caches automatically and bills a
    miss at the ordinary input rate, it equals `input`.
    """

    input: float
    output: float
    cache_read: float
    cache_write: float


#: Vendors that double their price during a daily peak window, and that window.
#: DeepSeek: "Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through
#: Friday (all other hours are off-peak)", off-peak being half of peak.
DEEPSEEK_PEAK_WINDOWS_UTC = ((1, 4), (6, 10))

#: Keyed by the model name passed on the request, which is the name both the vendor and
#: the harness see. Prices are the off-peak figures where a peak multiplier applies.
MODEL_PRICES: dict[str, Price] = {
    # Anthropic
    "claude-opus-5": Price(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25),
    "claude-sonnet-5": Price(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    "claude-haiku-4-5": Price(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25),
    # DeepSeek, served over its Anthropic-compatible endpoint. The shim ignores
    # `cache_control`, so there is no explicit cache write to bill: a miss costs the
    # input rate. Cache hits do occur, from DeepSeek's own automatic prefix caching.
    "deepseek-flash": Price(input=0.15, output=0.60, cache_read=0.003, cache_write=0.15),
    "deepseek-v4-pro": Price(input=0.66, output=1.98, cache_read=0.022, cache_write=0.66),
}

#: Models whose vendor charges double during the peak window above.
DEEPSEEK_MODELS = ("deepseek-flash", "deepseek-v4-pro")

#: Anthropic bills each server-side web search separately from tokens. Recovered from a
#: measured run: one task's haiku entry came to $0.5587, of which $0.3787 was tokens at
#: haiku rates, and the remaining $0.18 was exactly its 18 searches. Leaving this out
#: understates the open-web condition, which is the only one that searches.
WEB_SEARCH_USD_PER_REQUEST = 0.01

#: Which models' vendors bill that fee. Only Anthropic publishes one: DeepSeek's pricing
#: is tokens alone, so charging its web searches at Anthropic's rate invents money. The
#: distinction is not academic -- on one measured `deepseek-flash` run the fee would have
#: been $0.13, a quarter of a bill that was really $0.36.
WEB_SEARCH_BILLED_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")


def lookup(model: str | None) -> Price | None:
    """The price for a model name, tolerating a dated or versioned suffix.

    Vendors append a build identifier the price does not depend on:
    `claude-haiku-4-5-20251001` is `claude-haiku-4-5`, and `deepseek-flash-0731` is
    `deepseek-flash`. Longest known prefix wins, so a specific entry always beats a
    general one.
    """
    if not model:
        return None
    # A LiteLLM model id carries its route: `deepseek/deepseek-flash`,
    # `openrouter/deepseek/deepseek-v4.1-flash`. The price belongs to the model, not the
    # route, so the last path segment is tried as well -- without it the judge's own
    # bill came back unknown purely because it was addressed by a different provider.
    for name in (model, model.rsplit("/", 1)[-1]):
        if name in MODEL_PRICES:
            return MODEL_PRICES[name]
    parts = model.rsplit("/", 1)[-1].split("-")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:cut])
        if candidate in MODEL_PRICES:
            return MODEL_PRICES[candidate]
    return None


def _base_model(model: str | None) -> str | None:
    """The table key a model name resolves to, or the name itself when unknown."""
    if not model:
        return None
    if model in MODEL_PRICES:
        return model
    parts = model.split("-")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = "-".join(parts[:cut])
        if candidate in MODEL_PRICES:
            return candidate
    return model


def is_peak(model: str, when: datetime) -> bool:
    """Is `when` inside the vendor's peak-pricing window?

    Only DeepSeek has one. A naive datetime is read as UTC, because the window is
    defined in UTC and a local-time guess would silently pick the wrong rate.
    """
    if _base_model(model) not in DEEPSEEK_MODELS:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    if when.weekday() >= 5:  # Saturday, Sunday
        return False
    return any(start <= when.hour < end for start, end in DEEPSEEK_PEAK_WINDOWS_UTC)


def cost_usd(model: str | None, tokens: dict, when: datetime | None = None) -> float | None:
    """What these tokens cost, or `None` when the model has no known price.

    `tokens` uses the harness's names: `input`, `output`, `cache_read`,
    `cache_creation`. A cache write is billed as a miss where the provider has no
    separate write price, which is what `Price.cache_write` carries.
    """
    price = lookup(model)
    if price is None:
        return None
    multiplier = 2.0 if is_peak(model, when or datetime.now(timezone.utc)) else 1.0

    def count(name) -> float:
        value = (tokens or {}).get(name)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    billed = (
        count("input") * price.input
        + count("cache_creation") * price.cache_write
        + count("cache_read") * price.cache_read
        + count("output") * price.output
    )
    return billed / 1_000_000 * multiplier


def cost_from_model_usage(model_usage: dict, when: datetime | None = None) -> float | None:
    """Total cost across every model a run used, or `None` if any of them is unpriced.

    A run is not always one model. The CLI answers web searches with a small auxiliary
    model and that is part of the bill -- one `web-agent` task spent slightly more on
    its haiku search calls than on the sonnet turns that followed. Summing is not
    optional, and returning `None` rather than a partial total is what stops a caller
    reporting an incomplete figure as a complete one.
    """
    if not model_usage:
        return None
    total = 0.0
    for name, entry in model_usage.items():
        if lookup(name) is None:
            return None
        part = cost_usd(name, {
            "input": entry.get("inputTokens"),
            "output": entry.get("outputTokens"),
            "cache_read": entry.get("cacheReadInputTokens"),
            "cache_creation": entry.get("cacheCreationInputTokens"),
        }, when=when)
        total += part or 0.0
        searches = entry.get("webSearchRequests")
        if (isinstance(searches, (int, float)) and not isinstance(searches, bool)
                and _base_model(name) in WEB_SEARCH_BILLED_MODELS):
            total += searches * WEB_SEARCH_USD_PER_REQUEST
    return total
