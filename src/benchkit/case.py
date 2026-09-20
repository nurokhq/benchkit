"""Core data contracts shared by every benchmark.

These types carry no domain vocabulary. A benchmark supplies the meaning of its own
identifiers and its own scoring; the engine only moves tasks through a harness and
hands the answers back for scoring.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RunMetrics:
    """What it cost to produce one answer."""

    wall_ms: int | None = None
    turns: int | None = None
    cost_usd: float | None = None
    #: {"input", "output", "cache_read", "cache_creation"}
    tokens: dict[str, int | None] = field(default_factory=dict)
    #: tool name -> call count
    tool_use: dict[str, int] = field(default_factory=dict)
    returncode: int | None = None
    timed_out: bool = False
    #: Everything else a harness wants to record (duration_ms, stop_reason, ...).
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(v for v in self.tokens.values() if isinstance(v, int))


@dataclass
class Prediction:
    """One answer to one task, as produced by a harness.

    `normalized` is the harness's or normalizer's extraction of the answer into the
    generic shape described in `benchkit.normalize`. It stays a plain dict because
    benchmarks define their own item fields.
    """

    task_id: str
    answer_text: str = ""
    normalized: dict[str, Any] = field(default_factory=dict)
    #: The answer as validated data, when the harness asked the agent for a declared
    #: schema instead of prose. This is what a benchmark grades; `answer_text` stays
    #: for the human reading and for the audit trail.
    structured: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)
    status: str = "ok"  # "ok" | "failed" | "auth_error" | "fatal_error"


@dataclass
class ScoreResult:
    """What a benchmark makes of one answer.

    `score` is the single headline number for the task; `metrics` holds the named
    signals behind it, kept separate so nothing is hidden inside an average. `notes`
    is for human-readable audit detail such as which items failed verification.
    """

    task_id: str
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Benchmark(Protocol):
    """The whole abstraction: everything domain-specific lives behind this.

    Adding a benchmark means adding one module that satisfies this protocol. The
    engine never inspects a benchmark's identifiers or gold shapes, which is what
    keeps this file free of any single benchmark's vocabulary.
    """

    #: Stable identifier, e.g. "us-startup-programs".
    key: str
    #: Answer shapes this benchmark produces; benchmarks interpret their own values.
    answer_kinds: set[str]
    #: Names of the signals `score` may report.
    signals: list[str]

    def tasks_path(self):
        """Directory holding `tasks.jsonl`."""

    def load_tasks(self) -> list[dict]:
        """Return the task dicts, each with at least `task_id` and `question`."""

    def score(self, task: dict, prediction: Prediction, judge: Any = None) -> ScoreResult:
        """Grade one answer. `judge` is an optional `benchkit.judge.Judge`."""
