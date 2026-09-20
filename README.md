# benchkit

[![test](https://github.com/nurokhq/benchkit/actions/workflows/test.yml/badge.svg)](https://github.com/nurokhq/benchkit/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An engine for evaluating an agent against a curated corpus: run a benchmark's questions
through an agent in a container, then score the answers.

The engine knows **nothing** about any particular benchmark: domain vocabulary — what an
item is, how it is identified, what a good answer looks like — lives in the benchmark,
behind one protocol. Two benchmarks can share it without sharing a corpus or a vendor.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .
docker build -t benchkit-agent -f docker/Dockerfile .
cp .env.example .env                      # then put your provider key in it

.venv/bin/benchkit run \
  --benchmark us-startup-programs \
  --scenario web-agent \
  --run-dir results/web-agent
```

- **`.env` is read automatically.** A real environment variable always wins over it.
- **`benchkit benchmarks`** lists what is registered.
- **`python -m pytest`** runs the suite — no network, no docker, no credentials.

## The three conditions

| scenario | reaches a corpus | tools |
| --- | --- | --- |
| `web-agent` | no | WebSearch, WebFetch, shell and file tools — the open-web baseline |
| `kb-mcp` | yes | the same tools, plus a corpus over **MCP** |
| `kb-cli` | yes | the same tools, plus a **command-line client** for that corpus |

**The conditions differ in exactly two things: which corpus is reachable, and which
tools are on the list.** Nothing else.

- **Same-length, same-kind prompts.** Each states the task and which tools exist; none explains how to use them.
- **Tool documentation is the tool's own job.** An MCP handshake or `--help` delivers it, and it reaches every condition the same way.
- **No injected technique.** The harness offers no way to put corpus-specific advice in one prompt — measuring the prompt instead of the tooling is the failure this prevents.

### Endpoints are the benchmark's business

The engine holds no vendor's address. A benchmark declares its own:

```python
class MyBenchmark:
    def endpoints(self) -> dict:
        return {"mcp_url": "https://…/mcp",
                "corpus_api_url": "https://api.…",
                "corpus_web_url": "https://…"}
```

- **Override per run:** `--mcp-url`, `--corpus-api-url`, `--corpus-web-url`.
- **No endpoint, no silent fallback:** a `kb-mcp` run that is not pointed at a corpus stops immediately rather than quietly answering from the open web.

## How an answer is scored

- **Answers arrive as data.** The benchmark declares a JSON schema, and the agent's final message is validated against it before grading sees anything.
- **No fallback to prose.** An answer with no schema-valid result scores zero, and is counted apart from an answer that searched and found nothing — failing the contract and finding nothing are different results.
- **Facts are checked against the page the answer itself cited** — never against the benchmark's key, and never against a corpus one condition could read and another could not.
- **Silence is not error.** A claim the cited page does not settle earns nothing; one it contradicts costs. A page that cannot be fetched is excluded rather than charged to the answer, though the citation is still charged once as a defect.
- **Every component is written into the run**, so a ranking can be recomputed under different weights without paying for the agents again:

```bash
python3 tools/reweight.py results/<run> --sweep legit=0,0.5,1,2
```

A ranking that holds at only one set of weights is not a finding, and this is how that
is checked rather than asserted.

## Adding a benchmark

Create `benchmarks/<key>/` with a `benchmark.py` exposing `build()`, and register the key
in `src/benchkit/registry.py`. **Nothing in `src/benchkit/` changes.**

```python
class MyBenchmark:
    key = "my-benchmark"
    answer_kinds = {"entity_list"}
    signals = ["entity_f1", "citation_coverage"]

    def load_tasks(self) -> list[dict]: ...          # each needs task_id and question
    def scenarios(self):                              # optional: specialize the prompts
        return {"web-agent": scenario_for("web-agent", scope="..."),
                "kb-mcp": scenario_for("kb-mcp", scope="...", knowledge_base="owner/slug"),
                "kb-cli": scenario_for("kb-cli", scope="...", knowledge_base="owner/slug")}
    def score(self, task, prediction, judge=None) -> ScoreResult: ...
```

Four rules keep the engine general:

1. **A benchmark declares its signals; the engine never infers them.** Deciding what to measure from the shape of a gold answer is how generic code acquires one benchmark's assumptions.
2. **References are the benchmark's business.** A benchmark-specific URL field must not appear in the core.
3. **Completeness is opt-in.** A closed-world benchmark can score set overlap; an open-ended one must not pretend to, because there is no universe to overlap with.
4. **Cost and tokens stay in the engine.** They are properties of the run, not of the answer, which is what lets efficiency fall out for free.

## Working with a run

- **Narrow it:** `--limit`, `--task-id`, `--workers`.
- **Bound it:** `--max-budget-usd` per task, `--timeout` per task.
- **Choose a provider:** `--model`, `--base-url`, `--api-key-env`.
- **Verify claims:** `--judge-model` turns on per-claim checking against the cited page.
- **Re-score without re-running:** answers are the expensive part, so a scoring change never needs the agent again:

```bash
.venv/bin/benchkit regrade --benchmark us-startup-programs --run-dir results/web-agent
```

**A run directory holds:**

- **`responses.jsonl`** — the answers and their metrics, rewritten after every task, so an interrupted run is still usable.
- **`raw/<task_id>.json`** — the CLI envelope and what it reported.
- **`summary.json`** — the aggregate, including the manifest of what produced it.

## Layout

```
src/benchkit/              the engine: no benchmark vocabulary, no vendor's address
  case.py                  shared contracts: Prediction, RunMetrics, ScoreResult
  signals.py               generic metric primitives (set overlap, fields, tables, ranking)
  references.py            identity resolution from a benchmark-supplied catalog
  normalize.py             answer → generic structure
  judge.py                 per-claim verification (for answers with no gold key)
  run.py                   orchestration, aggregation, offline regrade
  registry.py              benchmark discovery by key
  llm.py                   LiteLLM client
  pricing.py               tokens → cost, per model
  sources.py               fetch and cache the text a claim is checked against
  cli.py                   `benchkit run` / `regrade` / `benchmarks`
  harness/claude_docker.py run one task in a container (Claude Code)
benchmarks/<key>/          one self-contained benchmark each
docker/Dockerfile          the container image the harness runs
tools/                     reporting and analysis scripts that read a finished run
```

## The bundled benchmark

**`us-startup-programs`** — open-ended *"give me a list of incubator and accelerator
programs I could apply to, with requirements, deadlines and citations."*

- **No gold list exists.** There is no registry of such programs, so no defensible answer key can be written and **completeness is not scored**.
- **Each returned program is graded on its own terms:** is it really an applyable program, does it fit the question, are its claims supported by the official page it cites.
- **Score is the sum of per-item credit:** `item_quality = legit + claims + reputation − penalties`. A dead link, a missing citation, a duplicate row or a contradicted claim is negative, so padding cannot pay.

Corpus: **[nurok.ai/frankfang/us-startup-programs](https://nurok.ai/frankfang/us-startup-programs)**.
Its questions, answer key and limitations live in
[`benchmarks/us-startup-programs/README.md`](benchmarks/us-startup-programs/README.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — including the rules a result must record to be
comparable with another.
