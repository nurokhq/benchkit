"""Harnesses that produce `benchkit.case.Prediction` objects from tasks.

A harness owns execution and nothing else: it turns a task dict into an answer plus
metrics, and never reads an eval set, scores anything, or knows a benchmark's name.
"""

from benchkit.harness.claude_docker import (
    BASE_TOOLS,
    DEFAULT_SCOPE,
    MODEL_ENV,
    SCENARIOS,
    ClaudeDockerHarness,
    HarnessConfig,
    Scenario,
    claude_flags,
    container_command,
    count_tool_use,
    is_auth_error,
    mcp_config,
    parse_envelope,
    scenario_for,
)

__all__ = [
    "BASE_TOOLS",
    "DEFAULT_SCOPE",
    "MODEL_ENV",
    "SCENARIOS",
    "ClaudeDockerHarness",
    "HarnessConfig",
    "Scenario",
    "claude_flags",
    "container_command",
    "count_tool_use",
    "is_auth_error",
    "mcp_config",
    "parse_envelope",
    "scenario_for",
]
