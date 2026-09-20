"""Drive Claude Code inside a Docker container, for any benchmark.

A *scenario* is a prompt plus an explicit tool list; a *harness* runs one task in
one container and reports what the CLI envelope said. Nothing here knows what the
tasks are about: a benchmark hands in `{"task_id", "question"}` and gets back a
`Prediction`, so the same code serves any closed-world question set.

Isolation comes from the container, not from the CLI: the process sees the image's
filesystem, the task's scratch directory and nothing else. Grading inputs are never
mounted, so no shell command can reach them. Authentication is an `ANTHROPIC_API_KEY`
passed to the container's environment, named but never written into argv; nothing is
copied from the user's own `~/.claude`.

Two details are deliberate and easy to break:

* The question goes in on **stdin**, never argv. Anything in argv is readable by any
  process in the container through `ps`, and the question is the one thing an agent
  should not be able to read out of the process table. The API key is handled the same
  way, for the same reason.
* `--permission-mode bypassPermissions` is never passed. The CLI refuses that flag
  under root, so a container that runs as root would fail every task. The container
  is already the boundary, and `--allowedTools` is what makes print mode usable.

**What a scenario may and may not put in a prompt.** The prompt states the task and
which tools the condition has. It does not explain how to use them. A tool's own
documentation is the right channel for that and reaches every condition the same way:
an MCP server ships its instructions in the `initialize` response, and a CLI ships
`--help`. Injecting corpus-specific technique here instead would make the comparison
measure the prompt rather than the tooling, so this harness offers no way to do it.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchkit import pricing
from benchkit.case import Prediction, RunMetrics

#: Tools every scenario gets: web search/fetch, a shell and file tools. The tool
#: list is one of exactly two things that differ between the conditions -- the other
#: is the prompt's statement of which corpus, if any, is reachable. Nothing else does.
BASE_TOOLS = ["WebSearch", "WebFetch", "Read", "Glob", "Grep", "Bash", "BashOutput",
              "KillShell", "Write", "Edit"]

#: The tools that reach the live web, as opposed to the model's own context or the
#: filesystem. Named here because the harness owns the tool vocabulary, and counted in
#: the summary because how often an answer needed them is a result, not bookkeeping.
WEB_TOOLS = ("WebSearch", "WebFetch")

#: Where a benchmark's prompt lands when it does not name its own subject area.
DEFAULT_SCOPE = "the subject area this benchmark covers"

#: Read by the Claude Code image as the model to use, mirroring the CLI's own env.
MODEL_ENV = "BENCHKIT_MODEL"

#: The MCP server name a corpus condition registers under. The CLI prefixes its tools
#: with it (`mcp__kb__kb_search`). Deliberately generic: the benchmark supplies the
#: endpoint, not the label.
DEFAULT_MCP_SERVER = "kb"


@dataclass
class Scenario:
    """One experimental condition: a prompt plus the tools it may reach."""

    key: str
    description: str
    tools: list[str]
    system_prompt: str
    mcp_server: str | None = None
    #: The shape the answer must arrive in, when the condition declares one. The CLI
    #: validates the agent's final output against it and returns the parsed object, so
    #: grading reads data instead of re-deriving records from prose. A benchmark that
    #: knows its own fields supplies this; without it the answer stays free text.
    output_schema: dict | None = None
    #: Environment the condition needs inside the container. A `None` value means
    #: "resolve from `HarnessConfig`", so an endpoint follows the run's configuration
    #: instead of being frozen into whichever module built the prompt.
    env: dict[str, str | None] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Everything about *how* a scenario runs, identical for every task in a run."""

    image: str = "benchkit-agent"
    api_key: str = ""
    #: Where the model provider lives. `None` uses the CLI's own default, which is
    #: Anthropic. Point this at another Anthropic-compatible endpoint to run the same
    #: conditions on a different vendor; `model` then names that vendor's model.
    base_url: str | None = None
    model: str | None = None
    timeout: float = 900.0
    max_budget_usd: float | None = None
    #: The HTTP endpoint a corpus condition reaches its deployment at. `None` means the
    #: benchmark supplies it (`Benchmark.endpoints()`) and, failing that, the run fails
    #: with a readable message rather than silently pointing at nothing.
    mcp_url: str | None = None
    #: The same deployment's API and web bases, for a condition that reaches the corpus
    #: through a command-line client instead of MCP. Passed to the client as the
    #: variables it documents for this purpose; `None` leaves the client on its own
    #: configured defaults, which is what an ordinary authenticated install uses.
    corpus_api_url: str | None = None
    corpus_web_url: str | None = None
    #: Ask the CLI for the full message stream (`--verbose`). Without it the CLI
    #: emits only the result object, so tool-use counts come back empty.
    verbose: bool = True
    #: Store every message in the raw envelope (large, only for debugging).
    keep_transcript: bool = False


def _scenario_prompts(scope, knowledge_base, mcp_server, declares_schema=False):
    """Build the prompt templates around benchmark-supplied wording.

    Every condition gets the same brief with one clause changed: which corpus, if any,
    it can reach. That is the experiment. A condition that also received technique --
    how to enumerate a corpus, which call is cheaper, what an earlier run measured --
    would be measured on its prompt instead of its tooling, and the more detailed the
    prompt the larger the effect, so the prompts are deliberately the same length and
    the same kind. Whatever a tool needs its user to know, the tool says itself: an MCP
    server ships instructions in its handshake, a CLI ships `--help`.
    """
    common = (
        "Cite the source URL for every factual claim. When a question asks for a full "
        "list or an exact count, say explicitly that the list is complete, or state "
        "what you could not verify."
    )
    if declares_schema:
        # Said to every condition in the same words, because it is part of the task
        # rather than part of any one condition. The CLI enforces the schema on the
        # final message; this only stops an agent spending turns discovering that.
        common = (
            "Do the work however you like, but your final answer must be the structured "
            "result the output schema defines, with one entry per program. Give each "
            "field the answer you would have written in prose; leave a field out when "
            "you found nothing to put in it rather than filling it with a guess. "
            f"{common}"
        )
    opening = (
        f"You are answering closed-world questions about {scope}. You have web search, "
        "web fetch, a shell and file tools, and a scratch directory to work in."
    )
    named = knowledge_base or "a curated knowledge base"
    # The three arms differ in this clause and nothing else: same opening, same closing,
    # same length to within a few words.
    web_agent = f"{opening} {common}"
    kb_mcp = (f"{opening} You also have a knowledge base about {scope}, whose address "
              f"is `{named}`, exposed through the `{mcp_server}` MCP server. {common}")
    kb_cli = (f"{opening} You also have the `nurok` command-line client, which can read "
              f"a knowledge base about {scope}, whose address is `{named}`. {common}")
    return {"web-agent": web_agent, "kb-mcp": kb_mcp, "kb-cli": kb_cli}


def scenario_for(key: str, *, scope: str | None = None, knowledge_base: str | None = None,
                 mcp_server: str = DEFAULT_MCP_SERVER,
                 output_schema: dict | None = None) -> Scenario:
    """Return one of the standard scenarios, optionally specialized.

    A benchmark passes `scope` (what its questions are about) and, for the corpus
    conditions, `knowledge_base` (the corpus to start from) plus `output_schema` (the
    shape its answers must arrive in). None is required, and none is remembered here:
    two benchmarks can run the same scenario against different corpora without this
    module knowing about either.

    There is deliberately no parameter for prompt guidance. See the module docstring.
    """
    if key not in ("web-agent", "kb-mcp", "kb-cli"):
        raise KeyError(f"unknown scenario: {key!r}")
    prompts = _scenario_prompts(scope or DEFAULT_SCOPE, knowledge_base, mcp_server,
                                declares_schema=bool(output_schema))
    if key == "web-agent":
        return Scenario(
            key="web-agent",
            description="Web search, web fetch, shell and file tools.",
            tools=list(BASE_TOOLS),
            system_prompt=prompts["web-agent"],
            output_schema=output_schema,
        )
    if key == "kb-cli":
        # No MCP server: this condition reaches the same deployment through a
        # command-line client, which the agent drives from the shell like any other
        # tool. The client reads its own credentials from its own home directory; the
        # benchmark supplies no account, so a condition that is not logged in fails
        # loudly at the first corpus call rather than quietly answering from the web.
        return Scenario(
            key="kb-cli",
            description="The same tools plus a command-line client for the same corpus.",
            tools=list(BASE_TOOLS),
            system_prompt=prompts["kb-cli"],
            output_schema=output_schema,
            env={
                "NUROK_API_URL": None,
                "NUROK_WEB_URL": None,
                # The install lives in the image; if it rewrites itself, two runs of the
                # same condition stop being the same condition.
                "NUROK_NO_AUTO_UPDATE": "1",
            },
        )
    return Scenario(
        key="kb-mcp",
        description="The same tools plus a corpus over MCP.",
        tools=list(BASE_TOOLS) + [f"mcp__{mcp_server}"],
        system_prompt=prompts["kb-mcp"],
        mcp_server=mcp_server,
        output_schema=output_schema,
    )


#: The conditions, with no benchmark vocabulary baked in. Specialize with
#: `scenario_for` or `dataclasses.replace` rather than editing these defaults.
SCENARIOS = {key: scenario_for(key) for key in ("web-agent", "kb-mcp", "kb-cli")}


def scenario_env(scenario: Scenario, config: HarnessConfig) -> dict[str, str]:
    """The container environment a condition needs, endpoints resolved from `config`.

    A scenario names the variables it wants; `None` means "take the run's value". That
    keeps a deployment endpoint in one place instead of frozen into prompt-building.
    An unset endpoint is left unset rather than defaulted, so a client falls back to
    its own configuration instead of being pointed at a guess.
    """
    run_values = {
        "NUROK_API_URL": config.corpus_api_url,
        "NUROK_WEB_URL": config.corpus_web_url,
    }
    resolved = {}
    for key, value in scenario.env.items():
        if value is None:
            value = run_values.get(key)
        if value:
            resolved[key] = value
    return resolved


def mcp_config(scenario: Scenario, mcp_url: str) -> str | None:
    """Inline MCP server config for a corpus condition.

    A corpus that is readable without a credential needs nothing embedded here, and
    this never embeds one: an endpoint that requires authentication is the caller's
    problem to solve outside the benchmark, not a secret to thread through a prompt.
    `--strict-mcp-config` keeps it the only server in play.
    """
    if not scenario.mcp_server:
        return None
    return json.dumps({"mcpServers": {scenario.mcp_server: {"type": "http", "url": mcp_url}}})


def container_command(config: HarnessConfig, workspace: str | Path,
                      scenario: Scenario | None = None) -> list[str]:
    """Wrap the CLI invocation in `docker run`.

    Only the scratch directory is mounted, so the benchmark is absent from the
    container's filesystem. `--network host` lets the agent reach a corpus service
    bound to the host's loopback, which is how a corpus condition talks to a
    locally-run deployment. The Claude config sits on a tmpfs, so every task starts
    clean and nothing it writes outlives the container.

    The key is named with `-e ANTHROPIC_API_KEY` and **not** with its value. The
    container gets it from the environment of the `docker` client, which the harness
    sets on the subprocess; writing `-e ANTHROPIC_API_KEY=<key>` would put the
    credential in this process's argv, where any other local user can read it out of
    `ps`. That is the same reason the question goes in on stdin.
    """
    command = [
        "docker", "run", "--rm", "-i",
        "--network", "host",
        "-v", f"{workspace}:/work",
        "-w", "/work",
        "--tmpfs", "/home/agent/.claude:rw,exec",
        "-e", "ANTHROPIC_API_KEY",
    ]
    if config.base_url:
        command += ["-e", f"ANTHROPIC_BASE_URL={config.base_url}"]
    if config.model:
        command += ["-e", f"{MODEL_ENV}={config.model}"]
    if scenario is not None:
        for key, value in scenario_env(scenario, config).items():
            command += ["-e", f"{key}={value}"]
    return command + [config.image, "claude", "--print", "--output-format", "json"]


def claude_flags(config: HarnessConfig, scenario: Scenario) -> list[str]:
    """Flags the scenario controls, identical for every task in a run."""
    flags = [
        # Exactly these tools, listed as allowed because print mode cannot prompt for
        # permission. Nothing outside the list is reachable. `bypassPermissions` is not
        # used: the CLI refuses it under root, and the container is already the boundary.
        "--tools", *scenario.tools,
        "--allowedTools", *scenario.tools,
        # Skills, custom commands, agents and resumable state stay out of the picture.
        "--disable-slash-commands",
        "--no-session-persistence",
        "--append-system-prompt", scenario.system_prompt,
    ]
    if scenario.output_schema:
        # The CLI validates the agent's final message against this and returns the
        # parsed object beside the text, so the answer arrives as data. `--json-schema`
        # is implemented as a terminal tool call, which is why it works against a
        # non-Anthropic Anthropic-compatible endpoint as long as that endpoint supports
        # tool use -- verified against DeepSeek's, which is what three of the four
        # conditions run on.
        flags += ["--json-schema", json.dumps(scenario.output_schema, ensure_ascii=False)]
    if config.model:
        flags += ["--model", config.model]
    if config.max_budget_usd:
        flags += ["--max-budget-usd", str(config.max_budget_usd)]
    if config.verbose:
        flags.append("--verbose")
    servers = mcp_config(scenario, config.mcp_url) if config.mcp_url else None
    if servers:
        flags += ["--mcp-config", servers, "--strict-mcp-config"]
    elif scenario.mcp_server:
        raise ValueError(
            f"scenario {scenario.key!r} needs a corpus over MCP but no endpoint was "
            "configured; pass --mcp-url, set BENCHKIT_MCP_URL, or give the benchmark an "
            "endpoints() default")
    return flags


def parse_envelope(stdout: str | None) -> tuple[dict | None, list[dict]]:
    """Read the CLI output: one result object, or an array of messages with --verbose."""
    stdout = (stdout or "").strip()
    if not stdout:
        return None, []
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("type") == "result":
                return candidate, []
        return None, []
    if isinstance(decoded, dict):
        return (decoded, []) if decoded.get("type") == "result" else (None, [])
    if isinstance(decoded, list):
        messages = [item for item in decoded if isinstance(item, dict)]
        results = [item for item in messages if item.get("type") == "result"]
        return (results[-1] if results else None), messages
    return None, []


def count_tool_use(messages: list[dict]) -> dict[str, int]:
    """Count tool calls from a verbose transcript, for cost and behaviour evidence."""
    counts: dict[str, int] = {}
    for message in messages:
        body = message.get("message")
        if not isinstance(body, dict):
            continue
        for block in body.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name") or "unknown"
                counts[name] = counts.get(name, 0) + 1
    return counts


def is_auth_error(result: dict | None, answer_text: str) -> bool:
    """Detect an unauthenticated CLI result, which retrying cannot fix."""
    text = f"{(result or {}).get('result') or ''} {answer_text or ''}".casefold()
    return "not logged in" in text or "please run /login" in text or "invalid api key" in text


#: What a provider says when the run cannot proceed for a reason no retry fixes.
#: A wave of three conditions hit `402 Insufficient Balance` and every affected run
#: wrote three zero-score rows, which read like a result -- a condition that scored
#: nothing -- rather than like a run that never happened. The statuses are checked as
#: well as the text, because the CLI reports both and the wording changes.
_FATAL_MARKERS = ("insufficient balance", "insufficient_quota", "quota exceeded",
                  "credit balance is too low", "exceeded your current quota")
_FATAL_STATUSES = (402, 403)


def is_fatal_api_error(result: dict | None, answer_text: str) -> bool:
    """Detect an API failure that ends the run rather than the one answer.

    Distinct from a rate limit, which is worth retrying, and from a dead source page,
    which is a fact about one citation.
    """
    if (result or {}).get("api_error_status") in _FATAL_STATUSES:
        return True
    text = f"{(result or {}).get('result') or ''} {answer_text or ''}".casefold()
    return any(marker in text for marker in _FATAL_MARKERS)


def _as_text(value: Any) -> str:
    """TimeoutExpired hands back bytes even when `text=True` was requested."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ClaudeDockerHarness:
    """Run tasks through one scenario, one container per task."""

    def __init__(self, scenario: Scenario, config: HarnessConfig, raw_dir: Path):
        self.scenario = scenario
        self.config = config
        self.raw_dir = Path(raw_dir)

    def run(self, task: dict, *, question_suffix: str | None = None) -> Prediction:
        """Answer one task in its own container and report the metrics.

        `task` needs `task_id` and `question`. The question is written to the
        container's stdin; `question_suffix` is appended to every question when a
        run wants one extra instruction (it is never part of the task itself).
        """
        task_id = str(task["task_id"])
        question = str(task["question"]).strip()
        if question_suffix:
            question = f"{question}\n\n{question_suffix.strip()}"

        workspace = Path(tempfile.mkdtemp(prefix=f"claude-docker-{task_id[:24]}-"))
        try:
            command = (container_command(self.config, workspace, self.scenario)
                       + claude_flags(self.config, self.scenario))
            started = time.monotonic()
            timed_out = False
            try:
                completed = subprocess.run(command, input=question + "\n", capture_output=True,
                                           text=True, timeout=self.config.timeout,
                                           env={**os.environ,
                                                "ANTHROPIC_API_KEY": self.config.api_key})
                stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _as_text(exc.stdout)
                stderr = _as_text(exc.stderr)
                returncode = None
            elapsed = time.monotonic() - started

            raw, messages = parse_envelope(stdout)
            answer_text = (raw or {}).get("result") or ""
            # The schema-validated answer, when the condition declared one. Absent means
            # the agent did not produce a conforming result, which is a fact about the
            # answer and is recorded as such rather than reconstructed from the text.
            structured = (raw or {}).get("structured_output")
            usage = (raw or {}).get("usage") or {}
            server_tool_use = usage.get("server_tool_use") or {}
            tokens = {
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "cache_read": usage.get("cache_read_input_tokens"),
                "cache_creation": usage.get("cache_creation_input_tokens"),
            }
            # The CLI prices any model it does not recognise at its own default rates,
            # which is not the bill when the request went to another vendor. Cost is
            # recomputed from the tokens against published prices, and the CLI's figure
            # is kept beside it; where no price is known, the reported one stands.
            reported_cost = (raw or {}).get("total_cost_usd")
            # Price every model the run used, not just the one answering: web search is
            # served by a second, smaller model and that spend is part of the run.
            model_usage = (raw or {}).get("modelUsage") or {}
            computed_cost = pricing.cost_from_model_usage(model_usage)
            if computed_cost is None:
                computed_cost = pricing.cost_usd(self.config.model, tokens)
            # Vendors that double their price for part of the day make a cost figure
            # meaningless without the hour it was billed at. Two rounds of this benchmark
            # were compared at face value when one had run inside DeepSeek's peak window
            # and the other outside it, which read as a 56% saving that was a 2x
            # multiplier. The multiplier is recorded so the comparison can be normalised.
            multiplier = 2.0 if pricing.is_peak(
                self.config.model, datetime.now(timezone.utc)) else 1.0
            metrics = RunMetrics(
                wall_ms=round(elapsed * 1000),
                turns=(raw or {}).get("num_turns"),
                cost_usd=computed_cost if computed_cost is not None else reported_cost,
                tokens=tokens,
                tool_use=count_tool_use(messages),
                returncode=returncode,
                timed_out=timed_out,
                extra={
                    "reported_cost_usd": reported_cost,
                    "cost_source": "tokens" if computed_cost is not None else "cli",
                    "price_multiplier": multiplier,
                    "duration_ms": (raw or {}).get("duration_ms"),
                    "duration_api_ms": (raw or {}).get("duration_api_ms"),
                    "num_turns": (raw or {}).get("num_turns"),
                    "stop_reason": (raw or {}).get("stop_reason"),
                    "subtype": (raw or {}).get("subtype"),
                    "is_error": (raw or {}).get("is_error"),
                    # Server-side web tool counts live under usage, not at the top level.
                    "web_search_requests": server_tool_use.get("web_search_requests"),
                    "web_fetch_requests": server_tool_use.get("web_fetch_requests"),
                    "models": (raw or {}).get("modelUsage"),
                    "permission_denials": (raw or {}).get("permission_denials"),
                    # Recorded because a condition that cannot produce the declared
                    # shape is a different result from one that produced it and was
                    # wrong, and the two must not be averaged into one number silently.
                    "schema_declared": bool(self.scenario.output_schema),
                    "schema_returned": isinstance(structured, dict),
                    "attempt": task.get("attempt", 1),
                },
            )
            files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file())
            if files:
                metrics.extra["workspace_files"] = files[:200]
            # Keep the envelope so an audit of tool use, or a re-grade with a different
            # normalizer, never has to re-query the model.
            self._write_raw(task_id, question, metrics, raw, messages, stderr, stdout)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        # A structured answer is an answer even when the text beside it is empty: the
        # CLI puts the object in `structured_output` and may leave `result` bare, and
        # requiring text would file a perfectly good answer as a failed run.
        produced = bool(answer_text) or isinstance(structured, dict)
        status = "ok" if returncode == 0 and not timed_out and produced else "failed"
        if status == "failed" and is_auth_error(raw, answer_text):
            status = "auth_error"
        elif status == "failed" and is_fatal_api_error(raw, answer_text):
            status = "fatal_error"
        return Prediction(task_id=task_id, answer_text=answer_text, structured=structured,
                          raw=raw, metrics=metrics, status=status)

    def _write_raw(self, task_id: str, question: str, metrics: RunMetrics, raw: dict | None,
                   messages: list[dict], stderr: str | None, stdout: str | None) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.raw_dir / f"{task_id}.json", {
            "task_id": task_id,
            "question": question,
            "metrics": asdict(metrics),
            "result": raw,
            "messages": messages if self.config.keep_transcript else [],
            "stderr": (stderr or "")[-4000:],
            "stdout_raw": None if raw is not None else (stdout or "")[-4000:],
        })
