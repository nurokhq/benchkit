"""The `us-startup-programs` benchmark: an open-ended shortlist task.

What makes this benchmark different from a captured-corpus lookup is that there is
**no universe to match against**. "Which accelerator programs could I apply to today"
has no authoritative registry, so a set-overlap score would need a gold list that only
this knowledge base's curation could supply — which would grade the curation, not the
answer. Completeness is therefore not scored at all.

What is scored instead is each program the answer actually returns. Every claim is
local: one program, verified against the official page that program cites. That needs
no universe, stays fair between conditions, and rewards an answer that is accurate
and well-sourced rather than merely long.

    score = sum over returned programs of item_quality
    item_quality = legit + claims + reputation - penalties

A confidently wrong row is negative, so padding a list with plausible-sounding
programs cannot pay. A row whose link is dead or missing scores nothing.

Judging is optional: without a `Judge` the benchmark still scores what it can check
offline (citation presence, duplicate rows, dead links) and says so in its signals, so
`--judge-model` is an upgrade rather than a prerequisite. Catalog membership is
reported as a diagnostic and earns nothing either way.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from benchkit.case import Prediction, ScoreResult
from benchkit.harness.claude_docker import scenario_for
from benchkit.judge import (REPUTATION_CLAIM, UNVERIFIABLE, is_program_claim,
                            relevance_claim, verdict_score)
from benchkit.references import Catalog, label_key
from benchkit.run import read_jsonl
from benchkit.sources import SourceText, attach_sources

HERE = Path(__file__).resolve().parent

#: The subject area the shared scenario prompts are specialized with.
SCOPE = "US startup funding, accelerator, residency and credit programs"

#: The corpus the corpus conditions read. A public knowledge base, readable in a
#: browser at <https://nurok.ai/frankfang/us-startup-programs> and, without an account,
#: over the MCP endpoint below. Stated here rather than in the engine, because which
#: corpus a benchmark measures is the benchmark's business.
#:
#: The `kb-cli` condition reaches the same corpus by pulling it into its scratch
#: directory, which needs no account. Its client must be 0.4.3 or later -- earlier
#: releases are refused by the edge in front of the API -- which is why the image pins
#: that version.
KNOWLEDGE_BASE = "frankfang/us-startup-programs"

#: The corpus revision this benchmark measures against. Recorded in every run manifest,
#: because scores are only comparable within one revision -- for the same reason the
#: answer key is authored and frozen: the corpus is the thing under test, so a corpus
#: edit must not be able to move a past score unnoticed. `kb_resolve` reports the live
#: revision; when it moves past this pin, runs either side are not comparable and the
#: pin is what says so.
KNOWLEDGE_BASE_REVISION = "r4"

#: Where that corpus is served from: the MCP endpoint the `kb-mcp` condition talks to,
#: and the API and web bases the `kb-cli` condition's client reads it through.
MCP_URL = "https://mcp.nurok.ai/mcp"
CORPUS_API_URL = "https://api.nurok.ai"
CORPUS_WEB_URL = "https://nurok.ai"



#: The shape an answer must arrive in. Declared to every condition in the same words
#: and enforced by the harness, which asks the CLI to validate the agent's final message
#: against it and return the parsed object.
#:
#: Declaring the shape is what removes a whole class of measurement error. Reconstructing
#: records from prose means guessing a vocabulary -- which heading starts an entry, which
#: word means "deadline", which column means "open" -- and every condition writes a list
#: its own way, so the guessing fails unevenly and the failure tracks the answer's style
#: rather than its content. A declared shape has no vocabulary to guess: a program is an
#: object and a field is a key.
#:
#: Only `name` and `url` are required. A required field with nothing behind it invites a
#: guess, and "the operator publishes no deadline" is a real answer that must stay
#: expressible -- forcing a value would manufacture the fact the benchmark is trying to
#: check for.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "programs": {
            "type": "array",
            "description": "One entry per program the question asks about.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The program's name."},
                    "url": {
                        "type": "string",
                        "description": "The official page the facts in this entry were "
                                       "read from. Every claim below is checked against it.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Whether applications are open on the question's "
                                       "date, and what the operator says about it. If the "
                                       "operator publishes no status page, say so and say "
                                       "when you checked.",
                    },
                    "deadline": {
                        "type": "string",
                        "description": "The deadline or next cutoff, with its date. If "
                                       "there is none, or none is published, say that.",
                    },
                    "terms": {
                        "type": "string",
                        "description": "What it provides: cash, equity or repayment terms, "
                                       "credits.",
                    },
                    "eligibility": {
                        "type": "string",
                        "description": "Who may apply.",
                    },
                    "citation": {
                        "type": "string",
                        "description": "The specific page the fields above were read from, "
                                       "when it is not the same as `url`.",
                    },
                },
                "required": ["name", "url"],
            },
        },
        "notes": {
            "type": "string",
            "description": "Anything the reader needs that does not fit an entry: what "
                           "could not be verified, and any caveat on the list.",
        },
    },
    "required": ["programs"],
}

#: Bumped whenever how an answer is read changes, so two scores are never compared
#: across such a change. Reported in the metrics and written into the run manifest:
#: a change to how answers are read can move a condition's score by several points with
#: no answer changing, and a comparison that does not record it is not a comparison.
#:
#: 1: records come from the declared schema; nothing is inferred from the answer text.
PARSER_VERSION = 12

#: Claims worth verifying per returned program, and how much each is worth. The
#: citation is checked structurally (it must be present) rather than by the judge.
#:
#: Weights are a statement about what a reader is owed, not a measurement. They are
#: deliberately few, and every component is reported separately in the metrics, so a
#: ranking can be recomputed under different weights from a stored run without
#: re-running anything (see `tools/reweight.py`).
CLAIM_WEIGHTS = {"status": 1.0, "deadline": 0.5, "terms": 0.5, "eligibility": 0.5}

#: Asked of every returned program, catalog member or not, so that membership in the
#: key decides nothing. Awarding it for membership would give a condition that can read
#: the key a point per recognised program before grading began, and would cap every
#: other condition at whatever the key happens to contain.
LEGIT_WEIGHT = 1.0

#: Whether the program fits what the task actually asked for. Without this, the score is
#: an unnormalised sum and padding a list with real-but-irrelevant programs pays.
RELEVANCE_WEIGHT = 1.0

#: Whether the citation is on the operator's own site. Judged, not looked up: the old
#: check compared the URL against domains enumerated in the catalog, so it was only
#: reachable for programs the catalog already knew, and a program outside the catalog
#: could not earn it however official its citation was.
REPUTATION_WEIGHT = 0.5

#: How many independent verdicts to take for each question kind, keeping the majority.
#:
#: Measured with `tools/judge_agreement_probe.py` on 120 claims judged three times each:
#: a claim checked against a cited page flips 2.6% of the time, `is_program` flips 20%
#: and `reputation` 26%. The two structural questions are also the ones carrying most of
#: the score, so they get three verdicts and the factual claims get one. That is roughly
#: a hundred extra calls per condition for the part of the score that was least stable.
SELF_CONSISTENCY = {"is_program": 3, "reputation": 3}

#: Ways an answer says "the operator publishes none of this". Counted, never scored --
#: see `absence_assertions` in the metrics.
_ABSENCE_PHRASES = ("not published", "no published", "not stated", "none published",
                    "no public", "not disclosed", "no dedicated", "no standalone",
                    "none stated", "not available")

#: An explicit "the operator publishes no such page", which the judge confirms from the
#: source. Worth half a supported claim: it is a true, checkable finding, but it carries
#: less than the fact it reports the absence of.
NOT_PUBLISHED_CREDIT = 0.5

NO_CITATION_PENALTY = 0.5
DEAD_LINK_PENALTY = 1.0


def _norm_url(value):
    from benchkit.normalize import canonical_url
    return canonical_url(value)


#: A host written without a scheme, which is how prose shortlists cite: "**Antler** —
#: antler.co". The suffix list is closed so ordinary dotted prose is not read as a URL.
_BARE_DOMAIN = (r"(?:[a-z0-9][a-z0-9-]*\.)+"
                r"(?:com|org|net|io|co|vc|ai|gov|edu|dev|app|xyz|so|fund|capital|ventures)\b")

#: Either citation form as prose writes it: an absolute URL, or a bare host such as
#: `**Antler** — antler.co`. One definition, used both to find a citation in a line and
#: to recognise the start of a program entry, because a rule that accepts `https://`
#: where the text has a bare host silently loses the whole entry.
_CITATION = (r"(?<![\w@.-])(?:https?://[^\s)\]<>\"'`,]+|" + _BARE_DOMAIN
             + r"(?:/[^\s)\]<>\"'`,]*)?)")

#: The lookbehind stops a match inside a longer token, and the `@` exclusion stops an
#: email address being read as a citation.
_URL_IN_TEXT = re.compile(_CITATION, re.I)


def _clean_name(value):
    """Reduce a table cell to a program name.

    A shortlist row is often `[Name](url)`, so the link is unwrapped and stray markup
    dropped; otherwise the URL becomes the item's name and every note reads as a URL.
    The name cell also frequently repeats the link beside the name —
    `**Y Combinator** — ycombinator.com/apply` — and left in place that name matches
    no catalog entry, so a whole table can come back unrecognised.
    """
    if not isinstance(value, str):
        return None
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    # A trailing parenthesised citation is a citation, not part of the name. Left in
    # place, `NSF SBIR/STTR (seedfund.nsf.gov)` matches no catalog entry and reads as a
    # URL in every note. Only a domain or URL is stripped, so genuine qualifiers such as
    # `PearX (Pear VC)` survive.
    text = re.sub(r"\s*[\(\[]\s*(?:https?://\S+|" + _BARE_DOMAIN + r"(?:/\S*)?)\s*[\)\]]\s*$",
                  "", text, flags=re.I)
    text = re.sub(
        r"\s*[\u2014\u2013\-|\u00b7:,]+\s*(?:https?://\S+|" + _BARE_DOMAIN + r"(?:/\S*)?)\s*$",
        "", text, flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" -|:")
    return text or None


def _program_name(value):
    """Reduce a heading or cell to a program name, dropping anything from the citation on.

    A name is never a citation. Entries are written `**a16z Speedrun** · https://… · Open
    year-round`, and reading the whole line as the name produced records called
    `a16z Speedrun · https://speedrun.a16z.com/apply · Open year-round; priority window
    Oct 12`. That name resolves against no catalog entry and carries no label, so the
    record was neither recognised nor verified -- a 38-program answer scored as though
    it had stated nothing. Everything from the first citation onward is not the name.
    """
    if not isinstance(value, str):
        return None
    # Unwrap markdown links before looking for the citation: in
    # `Name ([host](https://host/))` the first thing that looks like a citation is the
    # link *text*, so cutting there would leave `Name ([`.
    unwrapped = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    match = _URL_IN_TEXT.search(unwrapped)
    # A name may *begin* with something that reads as a citation. `SBIR.gov Participating
    # Federal Agencies (…)` is a program called after the site that lists it, and the
    # citation pattern matches `SBIR.gov` at position zero -- so the prefix is empty,
    # cleaning it yields None, and the caller crashed on `.strip`. Cutting at a citation
    # is only right when something is left; otherwise the whole name stands.
    text = (_clean_name(unwrapped[:match.start()]) if match else None) or _clean_name(unwrapped) or ""
    return text.strip(" -|:\u00b7(\u2014\u2013") or None



class StartupPrograms:
    """Open-ended shortlist grading, per returned program."""

    key = "us-startup-programs"
    answer_kinds = {"shortlist"}
    #: Written into each run's manifest, so a stored score names the corpus it measured.
    corpus_revision = KNOWLEDGE_BASE_REVISION

    def endpoints(self) -> dict:
        """The deployment this benchmark measures against.

        The engine reads this so it can stay free of any vendor's address, and a run
        can override any of it on the command line (`--mcp-url`, `--corpus-api-url`,
        `--corpus-web-url`).
        """
        return {"mcp_url": MCP_URL,
                "corpus_api_url": CORPUS_API_URL,
                "corpus_web_url": CORPUS_WEB_URL}

    #: Reported with every score, so two runs are never compared across a change in
    #: how answers are read. See `PARSER_VERSION` above.
    parser_version = PARSER_VERSION
    #: Reported by `score`; completeness is deliberately absent.
    signals = ["programs_returned", "programs_recognised", "legit", "claims",
               "reputation", "penalty", "mean_item_score", "fields_per_item",
               "field_status", "field_deadline", "field_terms", "field_eligibility",
               "relevant", "not_relevant", "items_voided", "overlap_with_universe",
               "claims_judged", "unverifiable_claims", "judge_failures",
               "schema_failures", "documented_absences", "absence_assertions", "judge_used"]

    def __init__(self, directory=None, judge_workers=8):
        #: How many claim verdicts to have in flight at once. Judging is network bound
        #: and unchanged by ordering, so this trades latency for concurrency only.
        self.judge_workers = judge_workers
        self.directory = Path(directory or HERE)
        #: The benchmark's own frozen universe. NOT read from the knowledge base:
        #: a scoring input derived from the system under test cannot attribute a score
        #: change to that system, because editing the corpus would move the answer key.
        #: See programs.json for the provenance rules.
        self.universe_doc = json.loads((self.directory / "programs.json").read_text(encoding="utf-8"))
        self.programs = self.universe_doc["programs"]
        #: One catalog entry per accepted name, so an alias resolves like a title while
        #: remaining ambiguous (and therefore unresolved) if two programs share it. A
        #: name repeated within one program must be collapsed first: two identical
        #: entries read as an ambiguous name and resolve to nothing at all.
        expanded = []
        for program in self.programs:
            accepted = set()
            for name in [program["title"], *(program.get("aliases") or [])]:
                folded = label_key(name)
                if not folded or folded in accepted:
                    continue
                accepted.add(folded)
                expanded.append({**program, "title": name})
        self.catalog = Catalog(expanded, id_field="key", url_fields=(), name_fields=("title",))
        #: Kept for diagnostics only. The catalog decides no part of the score:
        #: membership is reported as `programs_recognised`, and whether a program is
        #: real, relevant, and cited on its operator's own site is asked of the judge for
        #: every returned record, including the ones the catalog does not contain.
        self._source_text = None
        #: Whether a page missing from the cache may be fetched. Set false to score
        #: against a frozen snapshot of the cited pages: a fetch that fails is retried
        #: on the next scoring pass and sometimes succeeds, so a page nobody could read
        #: during one pass can become judgeable in the next and move the score without
        #: any answer changing. Freezing makes a re-scoring reproduce exactly.
        self.source_fetch = True

    # --- data ---------------------------------------------------------------

    def tasks_path(self) -> Path:
        return self.directory / "tasks.jsonl"

    def load_tasks(self) -> list[dict]:
        tasks = read_jsonl(self.tasks_path())
        if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
            raise ValueError("tasks.jsonl is empty or has duplicate task_ids")
        return tasks


    def scenarios(self):
        """The conditions, specialized with this benchmark's subject and corpus.

        The output schema is passed to every condition, so all three answer in the same
        shape. The prompts differ in one clause -- which corpus, if any, is reachable --
        and in nothing else; the benchmark injects no technique into any of them, so a
        score difference is a difference in what the conditions could find rather than
        in how much they were told.
        """
        return {
            "web-agent": scenario_for("web-agent", scope=SCOPE, output_schema=OUTPUT_SCHEMA),
            "kb-mcp": scenario_for("kb-mcp", scope=SCOPE, knowledge_base=KNOWLEDGE_BASE,
                                   output_schema=OUTPUT_SCHEMA),
            "kb-cli": scenario_for("kb-cli", scope=SCOPE, knowledge_base=KNOWLEDGE_BASE,
                                   output_schema=OUTPUT_SCHEMA),
        }

    # --- extraction ---------------------------------------------------------

    def _records(self, prediction: Prediction) -> list[dict]:
        """The programs the answer returned, read from the declared schema.

        Nothing is inferred from the text. The answer arrives as validated data --
        `--json-schema` makes the agent's final message conform and the CLI returns the
        parsed object -- so a record is a record and a field is a field, and grading
        never depends on which of the many ways of writing a list an answer happened to
        choose.

        Reading records out of prose instead means guessing all of it, and the guessing
        fails unevenly -- an entry written as a bullet is missed while the same entry as
        a heading is found, `Offers` is not `offer`, a column headed `Open on
        2026-09-14?` means nothing to a reader that knows only the word `status`. The
        error then tracks how an answer is written rather than what it says, which is
        the one thing a comparison must not do. Re-scoring answers under a repaired
        reader moved the
        two corpus conditions by +30.0 and +11.4 and the two web conditions by +2.5 and
        +2.0 -- measurement error, aimed at whoever wrote the least conventional prose.
        """
        records = []
        for item in (prediction.normalized or {}).get("items") or []:
            raw = item.get("raw") or {}
            merged = dict(raw)
            merged.setdefault("name", item.get("label"))
            merged.setdefault("key", item.get("id"))
            records.append(merged)
        return records

    # --- scoring ------------------------------------------------------------

    @staticmethod
    def _claim_requests(record: dict) -> list[tuple[str, str, float]]:
        """The (field, claim, weight) triples worth verifying for one record."""
        return [(field, f"{record.get('name')} {field}: {record.get(field)}", weight)
                for field, weight in CLAIM_WEIGHTS.items() if record.get(field)]

    def _verify_items(self, records: list[dict], judge, workers: int,
                      task: dict | None = None) -> list[dict]:
        """Verify every claim of every record, concurrently.

        Judging is network-bound and a single answer can carry well over a hundred
        claims, so sequential verdicts dominate wall time. Each call is independent
        and its result is written back to a fixed index, so concurrency changes the
        latency and not the score.

        Every record is asked the same questions. The catalog is not consulted here:
        it decides nothing that the judge is not asked about every other program too.
        """
        question = (task or {}).get("question")
        jobs = []           # (record_index, {name: claim}, {name: (kind, field, weight)}, samples, question)
        stale = []          # records whose cited page could not be read at all
        for index, record in enumerate(records):
            if not record.get("source_text"):
                # Nothing to verify against. The judge would be guessing, so no call is
                # made and the record is marked unverifiable instead: this is a fact
                # about the fetch, and scoring it as a failed claim would charge the
                # answer for a page that was down, a redirect, or a rate limit.
                stale.append(index)
                continue
            name = _program_name(record.get("name"))
            # One job per group of questions that share a page and a sample count. The
            # page is the expensive part of a call, and asking each field separately put
            # the same page through the model once per field: a record stating four
            # facts, with two structural questions asked three times each, sent it ten
            # times. Grouped, the reading is shared and calls per record fall from about
            # eleven to about three. Batched judging is the only mode there is.
            stated = {field: f"{name} {field}: {record.get(field)}"
                      for field in CLAIM_WEIGHTS if record.get(field)}
            if stated:
                jobs.append((index, stated,
                             {field: ("claim", field, CLAIM_WEIGHTS[field]) for field in stated},
                             1, None))
            structural = {"is_program": is_program_claim(name)}
            slots = {"is_program": ("is_program", None, LEGIT_WEIGHT)}
            if record.get("url"):
                structural["reputation"] = REPUTATION_CLAIM
                slots["reputation"] = ("reputation", None, REPUTATION_WEIGHT)
            jobs.append((index, structural, slots,
                         max(SELF_CONSISTENCY.get(kind, 1) for kind, _f, _w in slots.values()),
                         None))
            # `relevance` stays on its own: it is the only question that is about the
            # task rather than the page, and the only one whose verdict gates the item.
            jobs.append((index, {"relevance": relevance_claim(question, name)},
                         {"relevance": ("relevance", None, RELEVANCE_WEIGHT)}, 1, question))

        results = [{} for _ in records]
        for index in stale:
            record = records[index]
            slots = results[index].setdefault("claims", {})
            for field, _claim, weight in self._claim_requests(record):
                slots[field] = (UNVERIFIABLE, weight)
            slots["is_program"] = (UNVERIFIABLE, LEGIT_WEIGHT)
            slots["relevance"] = (UNVERIFIABLE, RELEVANCE_WEIGHT)
            if record.get("url"):
                slots["reputation"] = (UNVERIFIABLE, REPUTATION_WEIGHT)
        if not jobs:
            return results

        errors: list[str] = []

        def run(job):
            index, claims, slots, samples, ask = job
            record = records[index]
            try:
                verdicts = judge.verify_many(record, claims, source_text=record.get("source_text"),
                                             source_url=record.get("url"), question=ask,
                                             samples=samples)
            except Exception as exc:
                # A judge failure must not abort the run, but it must not be silent
                # either: a failed call and an unverifiable claim are different facts.
                # Tagged so the two reasons a claim has no verdict stay apart: a source
                # that could not be read is a fact about the citation, a judge call that
                # failed is a fact about the grading run, and only the first says
                # anything about the answer.
                errors.append(f"{record.get('name')}:{'+'.join(claims)}: {exc}")
                return [(index, slots[name],
                         {**UNVERIFIABLE, "error": str(exc)[:160]}) for name in claims]
            return [(index, slots[name], verdicts[name]) for name in claims]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for batch in pool.map(run, jobs):
                for index, (kind, field, weight), verdict in batch:
                    results[index].setdefault("claims", {})[field or kind] = (verdict, weight)
        self.last_judge_errors = errors
        return results

    def _sources(self, cache_dir) -> SourceText:
        """One resolver per run, so repeated citations are fetched once.

        Text comes from the URL each answer cites, never from a local knowledge base.
        A condition allowed to read a corpus and a condition not allowed to read it
        must be judged against the same independent text, or the comparison is rigged
        in whichever direction the corpus happens to agree with.
        """
        if self._source_text is None:
            self._source_text = SourceText(cache_dir, allow_fetch=self.source_fetch)
        return self._source_text

    @staticmethod
    def _strip_qualifier(name):
        """Drop a trailing qualifier an answer adds to a program's name.

        Prose writes "PearX (Pear VC)", "Antler US (San Francisco, Austin)", "Techstars
        (Spring 2027 programs)". The catalog keys on the program name alone, so an
        exact lookup misses them and a correctly-named program scores as unrecognised.
        """
        if not isinstance(name, str):
            return None
        text = re.sub(r"\s*[\(\[]([^)\]]*)[\)\]]\s*$", "", name).strip()
        text = re.split(r"\s+[\u2014\u2013]\s+", text)[0].strip()
        return text or None

    def _resolve(self, record: dict):
        """Resolve a program, tolerating the suffixes a prose answer adds.

        Diagnostic only. Resolution feeds `programs_recognised`, which is reported and
        never scored -- see the note in `__init__`.
        """
        name = record.get("name")
        for candidate in (name, self._strip_qualifier(name)):
            if candidate:
                key = self.catalog.resolve(candidate, record)
                if key:
                    return key
        return None

    @staticmethod
    def _merge_by(records: list[dict], identity) -> tuple[list[dict], int]:
        """Collapse records sharing an identity, keeping every field from each.

        Merging is a union of the fields, not a choice between rows: two mentions of one
        program carry different facts -- one may state the deadline, the other the
        citation -- and keeping only the first discarded facts the answer did state.
        """
        merged: list[dict] = []
        index: dict[str, dict] = {}
        duplicates = 0
        for record in records:
            key = identity(record)
            existing = index.get(key) if key else None
            if existing is None:
                merged.append(record)
                if key:
                    index[key] = record
                continue
            duplicates += 1
            for field, value in record.items():
                if value and not existing.get(field):
                    existing[field] = value
        return merged, duplicates

    @classmethod
    def _merge_duplicates(cls, records: list[dict]) -> tuple[list[dict], int]:
        """Collapse the rows describing one program, on either identity it is given.

        An answer that lists the same programs in several tables and again in prose
        states one program per section, not one per mention, and it names the same
        program with a different page each time -- the FAQ for one mention, the apply
        page for another. Merging on the name alone leaves those apart; merging on the
        citation alone does the same in reverse. Both passes are run.

        This is why duplicates are found from the answer rather than from catalog keys:
        a program the catalog does not contain is repeated just as easily, and two rows
        for one program must cost what one row for it costs.
        """
        def by_name(record):
            return re.sub(r"[^a-z0-9]+", "", (_program_name(record.get("name")) or "").casefold())

        def by_url(record):
            return _norm_url(record.get("url"))

        merged, first = cls._merge_by(records, by_name)
        merged, second = cls._merge_by(merged, by_url)
        return merged, first + second

    def score(self, task: dict, prediction: Prediction, judge=None,
              cache_dir=None) -> ScoreResult:
        returned = self._records(prediction)
        records, duplicates = self._merge_duplicates(returned)
        #: An answer that produced no schema-valid result is a different fact from one
        #: that produced a valid result with nothing in it, and the two must not both
        #: read as a zero. The first is the condition failing the task's stated contract;
        #: the second is a condition that answered and found nothing.
        schema_failure = prediction.structured is None
        metrics = {"parser_version": PARSER_VERSION,
                   "programs_returned": len(returned), "programs_recognised": 0,
                   "legit": 0.0, "relevance": 0.0, "claims": 0.0, "reputation": 0.0,
                   "penalty": 0.0, "judge_used": judge is not None, "duplicates": duplicates,
                   "judge_errors": 0, "unsourced_records": 0, "unverifiable_claims": 0,
                   "claims_judged": 0, "documented_absences": 0, "not_relevant": 0,
                   "relevant": 0, "items_voided": 0, "judge_failures": 0,
                   "schema_failures": int(schema_failure), "absence_assertions": 0}
        #: How many returned items stated each claim field. This is the signal that says
        #: *why* a score is what it is: two answers covering the same programs can differ
        #: by 20 points because one states an application status per program and the
        #: other does not, and that is invisible in a total built from verified claims.
        #: It also separates a condition that knows less from one that says less.
        for _field in CLAIM_WEIGHTS:
            metrics[f"field_{_field}"] = 0
        notes: list[str] = []
        if not records:
            notes.append("the answer carried no schema-valid result"
                         if schema_failure else "the answer returned no programs")
            return ScoreResult(task["task_id"], 0.0, metrics, notes)

        if judge is not None:
            # The judge can only verify a claim against text it can see. Sources come
            # from the URL the answer cited -- never from a local corpus, which one
            # condition was allowed to read and another was not -- and are resolved in
            # parallel because one answer can cite dozens of pages.
            source_text = self._sources(cache_dir or getattr(self, "cache_dir", None)
                                        or self.directory / ".cache")
            attach_sources(records, source_text)
            verdicts = self._verify_items(records, judge, self.judge_workers, task)
            metrics["judge_errors"] = len(getattr(self, "last_judge_errors", []))
            # The reasons, not just the count. A round lost 152 consecutive verdicts on
            # one task and the only record was the number 152 -- the exception text was
            # built, counted and dropped, so the cause could not be diagnosed after the
            # fact and the task was indistinguishable from an answer that said nothing.
            metrics["judge_error_samples"] = list(getattr(self, "last_judge_errors", []))[:5]
            metrics["unsourced_records"] = sum(1 for r in records if not r.get("source_text"))
            metrics["sources"] = dict(source_text.stats)
        else:
            verdicts = [{} for _ in records]

        total = 0.0
        scored = 0
        for index, record in enumerate(records):
            name = _program_name(record.get("name")) or "?"
            url = _norm_url(record.get("url"))
            # Reported, never scored. Resolution is attempted for every record so the
            # diagnostic is comparable across conditions, including the ones that
            # produce text rather than structured items.
            if record.get("key") or self._resolve(record):
                metrics["programs_recognised"] += 1

            for _field in CLAIM_WEIGHTS:
                if record.get(_field):
                    metrics[f"field_{_field}"] += 1

            outcome = verdicts[index].get("claims", {})
            item = 0.0

            def verdict_of(kind):
                return (outcome.get(kind) or ({}, 0.0))[0].get("verdict")

            # The gate. A source that contradicts the program's existence, or its fit
            # with the question, voids the rest of the item: there is nothing a reader
            # is owed about a program that is not a program, or not what was asked for.
            # Silence does not gate -- `not_stated` means the page did not say, which is
            # not the same as the answer being wrong.
            #
            # This is a gate rather than a bonus on purpose. A relevance *bonus* adds a
            # constant to every item, which makes the score pay twice for a long list
            # instead of making a long list pay only when its entries are on topic.
            voided = any(verdict_of(kind) == "contradicted" for kind in ("is_program", "relevance"))
            earned = 0.0
            #: What each component actually contributed to this item's score, as opposed
            #: to what its verdicts said. The two differ for a voided item, and the
            #: metrics must hold the former: they are the record `tools/reweight.py`
            #: re-ranks from, and a run of them that does not sum to the score makes
            #: every re-weighting approximate.
            credited = {"legit": 0.0, "claims": 0.0, "reputation": 0.0}

            # Every verdict the judge returned, counted under the question it answered.
            # The score is a weighted sum and cannot be read back into "how often was
            # this answer right", which is the first thing anyone who is not going to
            # read the code asks. Keeping the raw tally means the answer to that
            # survives into the stored run instead of having to be re-derived, and
            # `supported / (supported + contradicted)` is a plain accuracy over the
            # facts the answer chose to state.
            metrics.setdefault("verdicts", {})

            def tally(kind, verdict):
                label = f"{kind}.{verdict.get('verdict') or 'unknown'}"
                metrics["verdicts"][label] = metrics["verdicts"].get(label, 0) + 1

            # Was this actually an applyable program? The same judged question for every
            # record: catalog membership is reported as a diagnostic and decides nothing.
            # Absence from the catalog is not proof of absence from the world, and
            # presence in it is not proof of anything the reader cannot check.
            if "is_program" in outcome:
                verdict, weight = outcome["is_program"]
                tally("is_program", verdict)
                credit = verdict_score(verdict, supported=weight, contradicted=-weight)
                earned += credit
                credited["legit"] += credit
                if verdict["verdict"] == "contradicted":
                    notes.append(f"{name}: judged not an applyable program")
            if "relevance" in outcome:
                verdict, weight = outcome["relevance"]
                tally("relevance", verdict)
                if verdict["verdict"] == "contradicted":
                    metrics["not_relevant"] += 1
                    notes.append(f"{name}: judged not to fit the question asked")
                elif verdict["verdict"] == "supported":
                    metrics["relevant"] += 1

            for _field in CLAIM_WEIGHTS:
                _value = str(record.get(_field) or "").casefold()
                if _value and any(_phrase in _value for _phrase in _ABSENCE_PHRASES):
                    metrics["absence_assertions"] += 1

            # Claim verdicts, already collected concurrently.
            claim_total = 0.0
            claim_count = 0
            for field, weight in CLAIM_WEIGHTS.items():
                if field not in outcome:
                    continue
                verdict, _ = outcome[field]
                tally("claim", verdict)
                if verdict["verdict"] == "unverifiable":
                    # Excluded, not zeroed: the answer is not wrong about a page that
                    # could not be fetched, and the citation penalties below already
                    # charge for the citation itself. A claim the judge failed on is
                    # counted apart, because that one is our fault, not the answer's.
                    if verdict.get("error"):
                        metrics["judge_failures"] += 1
                    else:
                        metrics["unverifiable_claims"] += 1
                    continue
                claim_count += 1
                metrics["claims_judged"] += 1
                if verdict["verdict"] == "not_published":
                    metrics["documented_absences"] += 1
                score = verdict_score(verdict, not_published=NOT_PUBLISHED_CREDIT) * weight
                claim_total += score
                if verdict["verdict"] == "contradicted":
                    notes.append(f"{name}: {field} contradicted")
            earned += claim_total
            credited["claims"] += claim_total
            if claim_count and claim_total == 0.0:
                notes.append(f"{name}: claims unverifiable from the cited source")

            if "reputation" in outcome:
                verdict, weight = outcome["reputation"]
                tally("reputation", verdict)
                credit = verdict_score(verdict, supported=1.0, contradicted=-0.5)
                earned += credit * weight
                credited["reputation"] += credit * weight
                if verdict["verdict"] == "contradicted":
                    notes.append(f"{name}: citation is not on the operator's own site")

            if voided:
                metrics["items_voided"] += 1
                # A contradicted `is_program` keeps its penalty and nothing else: a row
                # for something that is not a program must cost, and none of the facts
                # it asserts about itself may earn, or padding with plausible-looking
                # entries would still pay. An item voided only on relevance scores zero.
                # The component totals above are still reported, so the verdicts stay
                # auditable even though the item's score does not use them.
                if verdict_of("is_program") == "contradicted":
                    item = -LEGIT_WEIGHT
                    credited = {"legit": -LEGIT_WEIGHT, "claims": 0.0, "reputation": 0.0}
                else:
                    item = 0.0
                    credited = {name: 0.0 for name in credited}
            else:
                item = earned
            for name, amount in credited.items():
                metrics[name] += amount

            if not url:
                item -= NO_CITATION_PENALTY
                metrics["penalty"] -= NO_CITATION_PENALTY
                notes.append(f"{name}: no citable official URL")
            elif record.get("source_provenance") == "missing":
                # A citation that was given but could not be read. Keyed on the fetch
                # result rather than on `source_text` being empty, so a run scored
                # without a judge -- where nothing is ever fetched -- is not charged
                # for pages it never asked for.
                item -= DEAD_LINK_PENALTY
                metrics["penalty"] -= DEAD_LINK_PENALTY
                notes.append(f"{name}: cited page could not be read")

            notes.append(f"{name}: item={item:+.2f}")
            total += item
            scored += 1

        metrics["items_scored"] = scored
        metrics["mean_item_score"] = (total / scored) if scored else 0.0
        metrics["fields_per_item"] = (
            sum(metrics[f"field_{_field}"] for _field in CLAIM_WEIGHTS) / scored
            if scored else 0.0)
        overlap = metrics["programs_recognised"] / len(self.programs) if self.programs else 0.0
        metrics["overlap_with_universe"] = overlap
        return ScoreResult(task["task_id"], total, metrics, notes)


def build():
    return StartupPrograms()
