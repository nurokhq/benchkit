# us-startup-programs

An open-ended benchmark about US startup funding, accelerator, residency and credit
programs. A founder-style question — *"which programs can I apply to on 2026-09-14,
what do they offer, who is eligible, where is the official page?"* — is answered by an
agent that either has the open web, or the open web plus a curated knowledge base.

## The independence rule

**This benchmark is built entirely from files in this directory.** It is never
generated from, and never reads, the knowledge base it is used to evaluate.

That is what makes the comparison mean anything. A scoring input derived from the
corpus under test cannot attribute a score change to that corpus: editing the corpus
would move the answer key, and a corpus improvement could raise the score without
improving a single answer. The same argument covers the text that claims are verified
against, so verification reads the URL each answer itself cited. The only place a
knowledge base appears at all is the `kb-mcp` and `kb-cli` conditions' *run*, where the
agent reaches it like any other tool.

Two tests enforce this: no file under `src/benchkit/` or `benchmarks/` may contain
knowledge-base layout tokens, and this directory may ship no generated corpus
artifacts.

## Files

| file | what it is |
| --- | --- |
| `programs.json` | **The answer key.** The benchmark's own frozen universe: 45 programs with their title, accepted aliases, and the internet domains each operator publishes on. Authored here, never regenerated. Editing it is a rubric change. |
| `tasks.jsonl` | Three open-ended shortlist tasks. The first is the primary task; the other two probe the same capability by sector/stage and by open-vs-rolling timing. |
| `benchmark.py` | Scoring: read the returned programs from the declared schema, resolve each against the universe, verify claims, and score. |
| `README.md` | This file. |

## The corpus

The corpus conditions read a public knowledge base:

| | |
| --- | --- |
| address | [`frankfang/us-startup-programs`](https://nurok.ai/frankfang/us-startup-programs) |
| MCP endpoint | `https://mcp.nurok.ai/mcp` — readable without an account |
| CLI | `nurok`, from `https://static.nurok.ai/cli/install.sh` |
| revision | `r4`, pinned in `benchmark.py` and written into each run's manifest |

**Every condition runs without an account.**

| condition | needs |
| --- | --- |
| `web-agent` | nothing — no corpus, no account |
| `kb-mcp` | nothing beyond the endpoint above; anonymous reads are enough |
| `kb-cli` | nothing beyond the client in the image; public corpora pull anonymously |

`kb-cli` reaches the corpus by pulling it into the scratch directory and working it from
disk. The pull is anonymous — verified against `r4` with an empty `NUROK_HOME` and no
credential of any kind. No `NUROK_API_KEY` and no `nurok login` are involved.

**The client version matters.** Releases before `0.4.3` send no `User-Agent`, and the
edge in front of `nurok.ai` refuses those requests with `403` whatever the credentials.
The Dockerfile pins `0.4.3` for that reason, and because Docker caches a `RUN` layer by
its command string — with the version unpinned, a rebuild silently keeps the CLI it
downloaded the first time, across releases.

That combination is worth knowing about, because the failure is quiet: the client is
refused, the agent gives up on the corpus, and it answers from the open web instead. The
run still scores, and nothing in the result says the corpus was never read. If a
`kb-cli` run's web-call count looks like `web-agent`'s, that is what happened.

A score is only comparable within one corpus revision, for the same reason the answer
key is authored rather than generated: the corpus is the thing under test, and a corpus
edit must not be able to move a past score unnoticed. Record the revision you ran
against (`kb_resolve` reports the live one) wherever you record the score.

## How an answer is scored

Each returned program is scored on its own and the results are summed. **Completeness
is deliberately not a signal** — an unbounded list is never penalised, so padding
cannot pay and a short accurate answer can beat a long speculative one.

Every program is asked the same questions, whether the answer key lists it or not.
`programs.json` is an answer key for diagnosis, **not** a source of credit: awarding a
point per recognised program would put the conditions that can read the key ahead of
the ones that cannot before grading began, and would cap every other condition at
whatever the key happens to contain. A program the key does not list is judged on its
merits like any other.

Per program:

| component | weight | how it is decided |
| --- | --- | --- |
| `legit` | ±1.0 | A judge reads the cited page: does it describe an applyable incubator, accelerator, residency, fellowship, grant or credit program? Asked of every record, key member or not. |
| `relevance` | gate | A judge reads the cited page against the task question: does this program fit what was asked, including sector, stage and program type? A **gate, not a bonus** — a contradicted program scores zero for the item, because a bonus would add a constant to every item and so pay twice for a long list instead of making a long list pay only when its entries are on topic. `not_stated` does not gate: a silent page is not a wrong answer. |
| `claims` | status 1.0, deadline/terms/eligibility 0.5 | A judge verifies each stated field against the text of the URL the answer cited. `supported` scores, `contradicted` is negative, `not_stated` is zero, and `not_published` — the source explicitly records that the operator publishes no such page — scores half. A documented absence is a finding; silence is not. |
| `reputation` | ±0.5 | A judge reads the cited page: is it published by the program's operator rather than a listicle, news site, directory or aggregator? Judged rather than looked up, so a program outside the key can earn it. |
| `penalty` | −0.5 / −1.0 | No citable URL; or a citation that was given but could not be read. |

An item the gate voids for `is_program` contradicted scores `−1.0` and nothing else: a
row for something that is not a program must cost, and none of the facts it asserts
about itself may earn, or padding with plausible-looking entries would still pay.

`overlap_with_universe`, `programs_recognised`, `mean_item_score`, `relevant`,
`not_relevant`, `items_voided`, `claims_judged`, `unverifiable_claims`,
`documented_absences` and `parser_version` are reported for diagnosis but do not feed
the headline score.

Two rules keep the score a statement about the answer rather than about the grading run:

* **An unverifiable claim is excluded, not zeroed.** When a cited page cannot be
  fetched, no verdict is possible, and scoring the record's claims as `not_stated`
  would charge the answer for a page that was down or rate-limited. The citation itself
  is still penalised once, as a citation defect. Fetch failures are not written to the
  cache, so a transient failure does not become permanent.
* **Weights are auditable.** Every component is written into `responses.jsonl`, so a
  ranking can be recomputed under different weights from a stored run:
  `python3 tools/reweight.py results/<run> --sweep legit=0,0.5,1,2`. A ranking that
  only holds at one set of weights is not a finding, and this is how that is checked
  rather than asserted.

## What this benchmark does not measure

Worth knowing before quoting a number from it.

* **It measures what an answer delivers, not how precise each sentence is.** The score
  sums per-program credit, so a longer list of correct, well-sourced programs outscores
  a shorter one. It cannot tell a curated list from an exhaustive one.
* **A claim is checked against the page the answer cited.** A condition that
  synthesises a fact from one document and cites a different page — the program's home
  page, say — is marked `not_stated` on that fact, correctly, because the cited page
  does not say it. The effect is not symmetric between conditions.
* **Three tasks, one domain.** Everything here is US startup funding. It is a real and
  common shape of work — building a sourced shortlist from many official sources — but
  it is one shape, and three tasks is a small sample.
* **The corpus is public, so every condition can reach it.** That is deliberate — it is
  what lets a stranger reproduce the comparison — but it means "the open-web baseline
  cannot see the corpus" is not enforced. The image carries the client on `PATH` for
  every condition, `Bash` is available to all of them, and the corpus is a public web
  page regardless. What the run reports is `web_calls` for every condition and, for
  `kb-mcp`, its `mcp__*` tool uses; **corpus use by `kb-cli` or by the baseline is not
  separately counted**, so a `web-agent` run that quietly pulled the corpus would look
  like a good open-web run. Nothing of the sort has been observed, and the prompts never
  mention the corpus, but the benchmark cannot currently show that it did not happen.

## Running

```bash
# open web
.venv/bin/benchkit run --benchmark us-startup-programs --scenario web-agent \
    --run-dir results/web-agent --workers 3

# the same corpus over MCP
.venv/bin/benchkit run --benchmark us-startup-programs --scenario kb-mcp \
    --run-dir results/kb-mcp --workers 3

# the same corpus through the CLI (needs `nurok login` first)
.venv/bin/benchkit run --benchmark us-startup-programs --scenario kb-cli \
    --run-dir results/kb-cli --workers 3
```

Add `--judge-model <model>` to verify factual claims; without it the benchmark still
scores what it can check offline and says so in its signals.

Answers are the expensive part, so a scoring change never requires re-running the agent:

```bash
.venv/bin/benchkit regrade --benchmark us-startup-programs --run-dir results/kb-mcp
```
