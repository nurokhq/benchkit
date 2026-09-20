# Contributing

## Running the checks

```bash
python -m venv .venv && .venv/bin/pip install -e . pytest
.venv/bin/python -m pytest
```

The suite uses no network and no docker, so it runs anywhere and needs no credentials.
Anything that needs a provider key or a container belongs in a benchmark run, not in a
test.

## Adding a benchmark

Create `benchmarks/<key>/` with a `benchmark.py` exposing `build()`, and register the
key in `src/benchkit/registry.py`. Nothing in `src/benchkit/` should change; if it has
to, the engine was carrying an assumption it should not have.

A benchmark directory holds its own data — questions, answer key, provenance — and
nothing derived from a corpus it is used to evaluate. `tests/test_core.py` asserts that
file list exactly, and asserts that no scoring code knows a corpus's layout. Both
checks exist because a scoring input derived from the system under test cannot measure
that system.

## Rules the engine depends on

These are not style preferences. Each one, broken, has produced a wrong number.

1. **A benchmark declares its signals; the engine never infers them.** Deciding what to
   measure from the shape of a gold answer is how generic code acquires one benchmark's
   assumptions.
2. **Completeness is opt-in.** A closed-world benchmark can score set overlap. An
   open-ended one cannot, because there is no universe to overlap with, and pretending
   otherwise turns a missing entry into a measured fact.
3. **Nothing a condition's prompt says may teach it technique.** The prompt states the
   task and which tools exist. Tool documentation is the tool's own job, through an MCP
   handshake or `--help`. A prompt that explains how to work one source makes the
   comparison a measurement of prompts.
4. **A claim is checked against the page the answer cited**, never against the
   benchmark's key and never against a corpus that only some conditions can read.
5. **Two runs are only comparable if they were scored the same way.** Record what
   changed — the parser version, the judge, the corpus revision — wherever a score is
   recorded, and do not compare across a change.

## Reporting a result

Record, alongside any score you publish:

- the benchmark and the scenario names,
- the model, and the provider it was reached through,
- the judge model, if claim verification was on,
- the corpus revision, if a corpus condition ran,
- how many repeats, because one run of this benchmark has never ranked anything.

Costs computed by the harness are token counts priced at the vendor's published list
rate. Some vendors charge more at some hours; if you compare two runs, compare them at
the same rate or normalise both.

## Scope

The engine is meant to stay benchmark-agnostic. A change that adds one benchmark's
vocabulary to `src/benchkit/` belongs in that benchmark instead. A change that makes
one condition's prompt more informative than another's is a bug, not a feature.
