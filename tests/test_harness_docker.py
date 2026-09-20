"""Unit tests for the Claude Code + Docker harness.

Nothing here needs docker or the network: command construction is inspected as data,
and a single case is run against a fake `subprocess.run` that returns a canned CLI
envelope.
"""

import json
import subprocess
from pathlib import Path

import pytest

from benchkit.case import Prediction, RunMetrics
from benchkit.harness import claude_docker as harness

RESULT_ENVELOPE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "The answer is 42.",
    "stop_reason": "end_turn",
    "num_turns": 7,
    "duration_ms": 12345,
    "duration_api_ms": 9000,
    "total_cost_usd": 0.25,
    "permission_denials": [],
    "modelUsage": {"claude-sonnet-4-5": {"input_tokens": 100, "cost_usd": 0.25}},
    "usage": {
        "input_tokens": 22,
        "output_tokens": 4792,
        "cache_read_input_tokens": 91976,
        "cache_creation_input_tokens": 12172,
        "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 2},
    },
}

VERBOSE_TRANSCRIPT = [
    {"type": "system", "subtype": "init"},
    {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "searching"},
        {"type": "tool_use", "name": "WebSearch"},
    ]}},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "WebSearch"},
        {"type": "tool_use", "name": "WebFetch"},
    ]}},
    RESULT_ENVELOPE,
]


def make_config(**overrides):
    # A corpus endpoint is part of any real run configuration, so it is part of the
    # default here too; a test that wants the unconfigured case removes it explicitly.
    values = {"image": "test-image", "api_key": "sk-test", "model": None,
              "mcp_url": "http://corpus.test:9000/mcp"}
    values.update(overrides)
    return harness.HarnessConfig(**values)


@pytest.fixture
def capture_run(monkeypatch):
    """Install a fake `subprocess.run` and hand back the call it recorded."""
    calls = {}

    def install(stdout="", stderr="", returncode=0, timeout_exc=None, on_call=None):
        def fake(command, **kwargs):
            calls["command"] = command
            calls["kwargs"] = kwargs
            if on_call is not None:
                on_call(command)
            if timeout_exc is not None:
                raise timeout_exc
            return subprocess.CompletedProcess(command, returncode, stdout, stderr)

        monkeypatch.setattr(harness.subprocess, "run", fake)
        return calls

    return install


def workspace_of(command):
    """Recover the host scratch directory from the single `-v ...:/work` mount."""
    mount = next(part for part in command if str(part).endswith(":/work"))
    return Path(str(mount)[: -len(":/work")])


# --- command construction -----------------------------------------------------

# Derived from the registry, not a literal list: a condition added to SCENARIOS must
# be exercised here automatically, which is exactly what a hand-written list missed
# when the corpus-over-CLI condition was added.
@pytest.mark.parametrize("key", sorted(harness.SCENARIOS))
def test_container_command_isolates_the_case(key, tmp_path):
    workspace = tmp_path / "scratch"
    command = harness.container_command(make_config(api_key="sk-secret"), workspace)

    assert command[:3] == ["docker", "run", "--rm"]
    assert "-i" in command  # stdin must stay open: the question arrives there
    assert command[command.index("--network") + 1] == "host"
    assert f"{workspace}:/work" in command
    assert command[command.index("-w") + 1] == "/work"
    assert command[command.index("--tmpfs") + 1] == "/home/agent/.claude:rw,exec"
    # The key is named, never written: `-e VAR=value` would put the credential in the
    # docker client's argv, where another local user can read it out of `ps`.
    assert command[command.index("-e") + 1] == "ANTHROPIC_API_KEY"
    assert not any("sk-secret" in str(part) for part in command)
    # Only the scratch directory is mounted; the benchmark itself is unreachable.
    assert command.count("-v") == 1
    assert command[-5:] == ["test-image", "claude", "--print", "--output-format", "json"]


def test_model_is_passed_to_container_and_cli(tmp_path):
    with_model = harness.container_command(make_config(model="claude-x"), tmp_path)
    assert "BENCHKIT_MODEL=claude-x" in with_model
    assert not any(str(part).startswith("BENCHKIT_MODEL") for part in
                   harness.container_command(make_config(), tmp_path))

    flags = harness.claude_flags(make_config(model="claude-x"), harness.SCENARIOS["web-agent"])
    assert flags[flags.index("--model") + 1] == "claude-x"
    assert "--model" not in harness.claude_flags(make_config(), harness.SCENARIOS["web-agent"])


@pytest.mark.parametrize("key", sorted(harness.SCENARIOS))
def test_claude_flags_allowlist_exactly_the_scenario_tools(key):
    scenario = harness.SCENARIOS[key]
    flags = harness.claude_flags(make_config(), scenario)

    tools = flags[flags.index("--tools") + 1: flags.index("--allowedTools")]
    allowed = flags[flags.index("--allowedTools") + 1: flags.index("--disable-slash-commands")]
    assert tools == scenario.tools
    assert allowed == scenario.tools
    assert "--disable-slash-commands" in flags
    assert "--no-session-persistence" in flags
    assert flags[flags.index("--append-system-prompt") + 1] == scenario.system_prompt


def test_bypass_permissions_is_never_passed():
    """The CLI refuses `bypassPermissions` under root; the container is the boundary."""
    for scenario in harness.SCENARIOS.values():
        flags = harness.claude_flags(make_config(), scenario)
        rendered = " ".join(flags)
        assert "bypassPermissions" not in rendered
        assert "--permission-mode" not in rendered
        assert "--dangerously-skip-permissions" not in rendered


def test_verbose_and_budget_flags():
    flags = harness.claude_flags(make_config(), harness.SCENARIOS["web-agent"])
    assert "--verbose" in flags
    assert "--max-budget-usd" not in flags

    quiet = harness.claude_flags(make_config(verbose=False), harness.SCENARIOS["web-agent"])
    assert "--verbose" not in quiet

    budgeted = harness.claude_flags(make_config(max_budget_usd=2.5), harness.SCENARIOS["web-agent"])
    assert budgeted[budgeted.index("--max-budget-usd") + 1] == "2.5"


def test_mcp_flags_only_for_the_mcp_scenario():
    web = harness.claude_flags(make_config(), harness.SCENARIOS["web-agent"])
    assert "--mcp-config" not in web
    assert "--strict-mcp-config" not in web
    assert harness.mcp_config(harness.SCENARIOS["web-agent"], "http://x/mcp") is None

    flags = harness.claude_flags(make_config(mcp_url="http://example.test:1234/mcp"),
                                 harness.SCENARIOS["kb-mcp"])
    assert flags[flags.index("--mcp-config") + 1] == json.dumps(
        {"mcpServers": {"kb": {"type": "http", "url": "http://example.test:1234/mcp"}}})
    assert "--strict-mcp-config" in flags


def test_a_corpus_scenario_without_an_endpoint_fails_loudly():
    """An unconfigured corpus must stop the run, not quietly answer from the web.

    A run that reaches no corpus still produces an answer, and that answer looks like a
    result. Raising here means the mistake costs a second instead of a whole round.
    """
    with pytest.raises(ValueError, match="no endpoint was configured"):
        harness.claude_flags(make_config(mcp_url=None), harness.SCENARIOS["kb-mcp"])
    # The open-web baseline needs no endpoint and is unaffected.
    assert harness.claude_flags(make_config(mcp_url=None), harness.SCENARIOS["web-agent"])


def test_scenarios_are_benchmark_agnostic():
    assert sorted(harness.SCENARIOS) == ["kb-cli", "kb-mcp", "web-agent"]
    assert harness.SCENARIOS["web-agent"].tools == harness.BASE_TOOLS
    assert harness.SCENARIOS["web-agent"].mcp_server is None
    assert harness.SCENARIOS["kb-mcp"].tools == harness.BASE_TOOLS + ["mcp__kb"]
    assert harness.SCENARIOS["kb-mcp"].mcp_server == harness.DEFAULT_MCP_SERVER
    # The CLI condition reads the same corpus without an MCP server, so the two corpus
    # conditions differ only in how they reach it.
    cli = harness.SCENARIOS["kb-cli"]
    assert cli.tools == harness.BASE_TOOLS
    assert cli.mcp_server is None
    assert not any(tool.startswith("mcp__") for tool in cli.tools)
    # The defaults carry no benchmark's subject, corpus address or owner namespace: a
    # benchmark supplies those, and the engine has to work for the next one too.
    blob = " ".join(f"{s.description} {s.system_prompt}" for s in harness.SCENARIOS.values())
    for forbidden in ("startup", "accelerator", "/", "http"):
        assert forbidden not in blob, f"{forbidden!r} is baked into the defaults"


def test_scenario_for_specializes_without_touching_the_defaults():
    specialized = harness.scenario_for("kb-mcp", scope="the Acme 2026 batch", knowledge_base="acme/kb")
    assert "the Acme 2026 batch" in specialized.system_prompt
    assert "`acme/kb`" in specialized.system_prompt
    assert "Acme" not in harness.SCENARIOS["kb-mcp"].system_prompt
    assert specialized.tools == harness.SCENARIOS["kb-mcp"].tools
    with pytest.raises(KeyError):
        harness.scenario_for("no-such-scenario")


def test_harness_accepts_a_benchmark_defined_scenario(tmp_path):
    scenario = harness.Scenario(key="custom", description="d", tools=["Read"],
                                system_prompt="Answer briefly.")
    flags = harness.claude_flags(make_config(), scenario)
    assert flags[flags.index("--tools") + 1: flags.index("--allowedTools")] == ["Read"]
    assert flags[flags.index("--append-system-prompt") + 1] == "Answer briefly."


# --- running one case (fake subprocess: no docker, no network) -----------------

def test_question_goes_on_stdin_not_argv(tmp_path, capture_run):
    calls = capture_run(stdout=json.dumps(RESULT_ENVELOPE))
    config = make_config(timeout=42.0)
    scenario = harness.SCENARIOS["web-agent"]
    harness.ClaudeDockerHarness(scenario, config, tmp_path / "raw").run(
        {"task_id": "t-1", "question": "What is the launch date?"})

    assert calls["kwargs"]["input"] == "What is the launch date?\n"
    assert calls["kwargs"]["text"] is True
    assert calls["kwargs"]["capture_output"] is True
    assert calls["kwargs"]["timeout"] == 42.0
    # `ps` inside the container must not reveal the question.
    assert not any("launch date" in str(part) for part in calls["command"])
    assert calls["command"][-len(harness.claude_flags(config, scenario)):] == \
        harness.claude_flags(config, scenario)


def test_run_maps_the_envelope_into_metrics_and_writes_raw(tmp_path, capture_run):
    capture_run(stdout=json.dumps(RESULT_ENVELOPE), stderr="progress noise")
    raw_dir = tmp_path / "raw"
    prediction = harness.ClaudeDockerHarness(
        harness.SCENARIOS["kb-mcp"], make_config(), raw_dir).run({"task_id": "t-1", "question": "Q?"})

    assert isinstance(prediction, Prediction)
    assert isinstance(prediction.metrics, RunMetrics)
    assert prediction.status == "ok"
    assert prediction.task_id == "t-1"
    assert prediction.answer_text == "The answer is 42."
    assert prediction.raw == RESULT_ENVELOPE

    metrics = prediction.metrics
    assert metrics.returncode == 0
    assert metrics.timed_out is False
    assert metrics.turns == 7
    assert metrics.cost_usd == 0.25
    assert metrics.wall_ms >= 0
    assert metrics.tokens == {"input": 22, "output": 4792, "cache_read": 91976, "cache_creation": 12172}
    assert metrics.tool_use == {}
    assert metrics.extra["duration_ms"] == 12345
    assert metrics.extra["duration_api_ms"] == 9000
    assert metrics.extra["num_turns"] == 7
    assert metrics.extra["stop_reason"] == "end_turn"
    assert metrics.extra["web_search_requests"] == 3
    assert metrics.extra["web_fetch_requests"] == 2
    assert metrics.extra["models"] == RESULT_ENVELOPE["modelUsage"]
    assert metrics.extra["attempt"] == 1

    payload = json.loads((raw_dir / "t-1.json").read_text(encoding="utf-8"))
    assert set(payload) == {"task_id", "question", "metrics", "result", "messages", "stderr", "stdout_raw"}
    assert payload["task_id"] == "t-1"
    assert payload["question"] == "Q?"
    assert payload["result"] == RESULT_ENVELOPE
    assert payload["metrics"]["tokens"]["cache_read"] == 91976
    assert payload["messages"] == []
    assert payload["stderr"] == "progress noise"
    assert payload["stdout_raw"] is None


def test_verbose_transcript_is_counted_and_stored_only_on_request(tmp_path, capture_run):
    capture_run(stdout=json.dumps(VERBOSE_TRANSCRIPT))
    raw_dir = tmp_path / "raw"
    task = {"task_id": "t-2", "question": "Q?"}

    plain = harness.ClaudeDockerHarness(harness.SCENARIOS["web-agent"], make_config(), raw_dir).run(task)
    assert plain.metrics.tool_use == {"WebSearch": 2, "WebFetch": 1}
    assert json.loads((raw_dir / "t-2.json").read_text(encoding="utf-8"))["messages"] == []

    keeper = harness.ClaudeDockerHarness(
        harness.SCENARIOS["web-agent"], make_config(keep_transcript=True), raw_dir)
    keeper.run(task)
    stored = json.loads((raw_dir / "t-2.json").read_text(encoding="utf-8"))["messages"]
    assert stored == VERBOSE_TRANSCRIPT


def test_run_reads_the_question_suffix_and_attempt(tmp_path, capture_run):
    calls = capture_run(stdout=json.dumps(RESULT_ENVELOPE))
    prediction = harness.ClaudeDockerHarness(harness.SCENARIOS["web-agent"], make_config(),
                                             tmp_path / "raw").run(
        {"task_id": "t-3", "question": "Q?", "attempt": 2}, question_suffix="  Answer in one line.  ")
    assert calls["kwargs"]["input"] == "Q?\n\nAnswer in one line.\n"
    assert prediction.metrics.extra["attempt"] == 2


def test_workspace_files_are_recorded_and_the_scratch_dir_removed(tmp_path, capture_run):
    def write_output(command):
        workspace = workspace_of(command)
        (workspace / "notes.md").write_text("scratch", encoding="utf-8")
        (workspace / "sub").mkdir(exist_ok=True)
        (workspace / "sub" / "data.txt").write_text("more", encoding="utf-8")

    calls = capture_run(stdout=json.dumps(RESULT_ENVELOPE), on_call=write_output)
    prediction = harness.ClaudeDockerHarness(harness.SCENARIOS["web-agent"], make_config(),
                                             tmp_path / "raw").run({"task_id": "t-4", "question": "Q?"})
    assert prediction.metrics.extra["workspace_files"] == ["notes.md", "sub/data.txt"]
    assert not workspace_of(calls["command"]).exists()


@pytest.mark.parametrize("result,returncode,expected", [
    ("Invalid API key · Please run /login", 1, "auth_error"),
    ("not logged in", 1, "auth_error"),
    ("the model hit an error", 1, "failed"),
    ("", 0, "failed"),
    ("an answer", 0, "ok"),
])
def test_run_status_covers_failure_modes(tmp_path, capture_run, result, returncode, expected):
    envelope = dict(RESULT_ENVELOPE, result=result)
    capture_run(stdout=json.dumps(envelope), returncode=returncode)
    prediction = harness.ClaudeDockerHarness(harness.SCENARIOS["web-agent"], make_config(),
                                             tmp_path / "raw").run({"task_id": "t-5", "question": "Q?"})
    assert prediction.status == expected


def test_timeout_keeps_the_partial_output(tmp_path, capture_run):
    exc = subprocess.TimeoutExpired(cmd="docker", timeout=1.0, output=b"docker: daemon not running\n")
    capture_run(timeout_exc=exc)
    raw_dir = tmp_path / "raw"
    prediction = harness.ClaudeDockerHarness(harness.SCENARIOS["web-agent"], make_config(timeout=1.0),
                                             raw_dir).run({"task_id": "t-6", "question": "Q?"})

    assert prediction.metrics.timed_out is True
    assert prediction.metrics.returncode is None
    assert prediction.status == "failed"
    assert prediction.raw is None
    payload = json.loads((raw_dir / "t-6.json").read_text(encoding="utf-8"))
    assert payload["stdout_raw"] == "docker: daemon not running\n"


def test_timeout_that_still_yielded_a_result_object(tmp_path, capture_run):
    exc = subprocess.TimeoutExpired(cmd="docker", timeout=1.0,
                                    output=json.dumps(RESULT_ENVELOPE).encode())
    capture_run(timeout_exc=exc)
    prediction = harness.ClaudeDockerHarness(harness.SCENARIOS["kb-mcp"], make_config(),
                                             tmp_path / "raw").run({"task_id": "t-7", "question": "Q?"})
    assert prediction.metrics.timed_out is True
    assert prediction.answer_text == "The answer is 42."
    assert prediction.status == "failed"  # no return code, so the case did not finish


# --- pure helpers -------------------------------------------------------------

def test_parse_envelope_accepts_a_single_result_object():
    raw, messages = harness.parse_envelope(json.dumps(RESULT_ENVELOPE))
    assert raw == RESULT_ENVELOPE
    assert messages == []


def test_parse_envelope_accepts_a_verbose_message_array():
    raw, messages = harness.parse_envelope(json.dumps(VERBOSE_TRANSCRIPT))
    assert raw == RESULT_ENVELOPE
    assert messages == VERBOSE_TRANSCRIPT


def test_parse_envelope_takes_the_last_result_of_an_array():
    first = dict(RESULT_ENVELOPE, result="first")
    second = dict(RESULT_ENVELOPE, result="second")
    raw, messages = harness.parse_envelope(json.dumps([first, second]))
    assert raw["result"] == "second"
    assert messages == [first, second]


def test_parse_envelope_scans_lines_when_the_stream_is_not_one_document():
    stream = "some log line\n" + json.dumps(RESULT_ENVELOPE) + "\ntrailing note"
    raw, messages = harness.parse_envelope(stream)
    assert raw == RESULT_ENVELOPE
    assert messages == []


@pytest.mark.parametrize("stdout", [
    None,
    "",
    "   ",
    "not json at all",
    json.dumps({"type": "assistant", "message": {}}),          # a message, not a result
    "log line\n" + json.dumps({"type": "assistant"}),           # no result line to find
])
def test_parse_envelope_rejects_anything_that_is_not_a_result(stdout):
    assert harness.parse_envelope(stdout) == (None, [])


def test_parse_envelope_returns_messages_without_a_result_object():
    """A truncated --verbose transcript still yields its messages, but no result."""
    messages = [{"type": "assistant", "message": {"content": []}}, "not a dict"]
    raw, decoded = harness.parse_envelope(json.dumps(messages))
    assert raw is None
    assert decoded == [{"type": "assistant", "message": {"content": []}}]


def test_count_tool_use_counts_named_calls_and_skips_noise():
    messages = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "WebSearch"},
            {"type": "tool_use", "name": "WebSearch"},
            {"type": "tool_use"},                       # unnamed call
            {"type": "text", "text": "hello"},
            "not a block",
        ]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "system"},                              # no message body
        {"type": "assistant", "message": "not a dict"},
        {"type": "assistant", "message": None},
    ]
    assert harness.count_tool_use(messages) == {"WebSearch": 2, "unknown": 1, "Bash": 1}
    assert harness.count_tool_use([]) == {}


@pytest.mark.parametrize("text", [
    "Invalid API key",
    "invalid api key · please run /login",
    "Not logged in. Please run /login to authenticate.",
])
def test_is_auth_error_detects_unauthenticated_results(text):
    assert harness.is_auth_error({"result": text}, "") is True
    assert harness.is_auth_error(None, text) is True


@pytest.mark.parametrize("result,answer", [
    ({"result": "the answer is 42"}, "the answer is 42"),
    (None, ""),
    ({"result": "the model ran out of turns"}, ""),
    ({}, "some ordinary answer"),
])
def test_is_auth_error_ignores_ordinary_failures(result, answer):
    assert harness.is_auth_error(result, answer) is False


@pytest.mark.parametrize("result,answer", [
    ({"api_error_status": 402, "result": "API Error: 402 Insufficient Balance"},
     "API Error: 402 Insufficient Balance"),
    ({"result": "insufficient_quota"}, ""),
    ({}, "You exceeded your current quota, please check your plan"),
])
def test_is_fatal_api_error_ends_the_run(result, answer):
    """A balance or quota failure is not one bad answer, it is every remaining one.

    A whole wave -- three conditions, three tasks each -- hit `402 Insufficient
    Balance` and each run wrote nine zero-score rows. Read back later they look like a
    condition that answered and scored nothing, which is why the run now stops.
    """
    assert harness.is_fatal_api_error(result, answer) is True


@pytest.mark.parametrize("result,answer", [
    ({"result": "the answer is 42"}, "the answer is 42"),
    (None, ""),
    ({"api_error_status": 429, "result": "rate limit reached"}, "rate limit reached"),
    ({"result": "the cited page could not be read"}, ""),
])
def test_is_fatal_api_error_leaves_retryable_failures_alone(result, answer):
    """429 is worth another attempt, and a dead page is a fact about one citation."""
    assert harness.is_fatal_api_error(result, answer) is False


# --- the CLI condition reaches the same deployment ----------------------------

def test_cli_scenario_passes_endpoints_from_the_run_config(tmp_path):
    """One place owns the endpoint, so retargeting a run moves every condition."""
    scenario = harness.scenario_for("kb-cli", scope="x", knowledge_base="o/k")
    config = make_config(corpus_api_url="http://kb.test:9000",
                         corpus_web_url="http://web.test:9001")
    command = harness.container_command(config, tmp_path, scenario)
    assert "NUROK_API_URL=http://kb.test:9000" in command
    assert "NUROK_WEB_URL=http://web.test:9001" in command
    # A scenario's literal values are passed through unchanged.
    assert "NUROK_NO_AUTO_UPDATE=1" in command


def test_mcp_scenario_passes_no_nurok_environment(tmp_path):
    scenario = harness.scenario_for("kb-mcp", scope="x", knowledge_base="o/k")
    command = harness.container_command(make_config(), tmp_path, scenario)
    assert not any(str(part).startswith("NUROK_") for part in command)


def test_every_registered_scenario_builds_a_distinct_condition():
    """Each condition must be uniquely identified and independently runnable."""
    keys = sorted(harness.SCENARIOS)
    assert keys == ["kb-cli", "kb-mcp", "web-agent"], "update the expectations with the registry"
    scenarios = [harness.SCENARIOS[k] for k in keys]
    assert len({s.key for s in scenarios}) == len(scenarios)
    assert len({s.description for s in scenarios}) == len(scenarios)
    for scenario in scenarios:
        assert scenario.system_prompt.strip(), scenario.key
        assert scenario.tools, scenario.key
        # Exactly the knowledge-base conditions reach a deployment; the web condition
        # must not, or it would no longer be the open-web baseline.
        reaches = scenario.mcp_server is not None or bool(scenario.env)
        assert reaches == (scenario.key != "web-agent"), scenario.key


def test_the_two_knowledge_base_conditions_share_one_endpoint():
    """Retargeting the run must move both, or they would read different deployments."""
    config = make_config(corpus_api_url="http://kb.test:1", corpus_web_url="http://web.test:2")
    mcp = harness.SCENARIOS["kb-mcp"]
    cli = harness.SCENARIOS["kb-cli"]
    assert harness.mcp_config(mcp, config.mcp_url) is not None
    assert harness.mcp_config(cli, config.mcp_url) is None
    assert harness.scenario_env(cli, config)["NUROK_API_URL"] == "http://kb.test:1"
    assert harness.scenario_env(mcp, config) == {}


# --- provider override --------------------------------------------------------

def test_base_url_is_passed_only_when_configured(tmp_path):
    """Pointing a run at another Anthropic-compatible endpoint is one variable."""
    default = harness.container_command(make_config(), tmp_path)
    assert not any(str(p).startswith("ANTHROPIC_BASE_URL") for p in default)

    elsewhere = harness.container_command(
        make_config(base_url="https://api.deepseek.com/anthropic"), tmp_path)
    assert "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic" in elsewhere
    # The key variable is unchanged, and still carries no value of its own: the caller
    # chooses which key to supply, the harness passes it through the subprocess env.
    assert "ANTHROPIC_API_KEY" in harness.container_command(
        make_config(api_key="sk-secret", base_url="https://x/anthropic"), tmp_path)
