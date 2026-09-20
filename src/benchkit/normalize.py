"""Turn a harness answer into a benchmark-agnostic structure.

The normalizer has one domain-dependent step: deciding what identity a named thing
refers to. That decision belongs to the benchmark, so it is injected as a resolver
rather than imported. Everything else here is generic.

The output shape, which benchmarks read in their `score`:

    {
      "items":      [{"label": ..., "id": <canonical or None>, "raw": {...}}, ...],
      "unresolved": [{"label": ..., "id": None, ...}, ...],
      "fields":     {...},
      "tables":     [...],
      "scalars":    {...},
      "references": ["<canonical reference>", ...],
      "count":      <int or None>,
      "answer_text": "<the prose>",
      "warnings":   [...],
      "source":     "structured" | "text" | "structured+text",
    }

`items` keeps unresolved mentions with `id: None` so a scorer can count them as
false positives instead of silently discarding part of the answer.
"""

import json
import re
from urllib.parse import urlsplit


def canonical_url(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/') or '/'}"


def extract_json_object(text) -> dict | None:
    """Parse a JSON object from a response, fenced or bare."""
    if not isinstance(text, str):
        return None
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _urls_in(text: str) -> list[str]:
    return [canonical_url(match) for match in re.findall(r"https?://[^\s\)\]\>,;\"']+", text or "")]


def normalize_answer(response, resolve=None, item_fields=("name", "label", "title", "program", "id")):
    """Extract an answer into the generic shape.

    `response` may be a harness `Prediction` (whose `structured` payload is used when
    present), a mapping carrying `structured` and/or `answer_text`, or a plain string.
    `resolve(label, entry) -> id | None` is the benchmark's identity resolution;
    without it nothing resolves and every item is reported as unresolved, which is the
    honest result rather than a guess.
    """
    declared = getattr(response, "structured", None)
    if isinstance(response, dict):
        structured = response.get("normalized") or response.get("structured") or response
        if not isinstance(structured, dict):
            structured = {}
        raw_text = response.get("answer_text") or response.get("answer") or ""
        if not raw_text and response.get("raw"):
            raw_text = (response["raw"] or {}).get("result") or ""
        if not raw_text and structured is response:
            raw_text = json.dumps(response, ensure_ascii=False)
    elif isinstance(declared, dict):
        # A schema-validated payload is already the answer. Reading it does not depend
        # on the prose beside it, and the prose is kept only as the human-readable record.
        structured = declared
        raw_text = getattr(response, "answer_text", "") or json.dumps(declared, ensure_ascii=False)
    else:
        raw_text = str(response or "")
        structured = extract_json_object(raw_text) or {}

    text = str(raw_text or "")
    warnings: list[str] = []
    items: list[dict] = []
    seen: set = set()

    def add(label, entry=None, identifier=None):
        entry = entry if isinstance(entry, dict) else {}
        label = label if isinstance(label, str) else (entry.get("name") or entry.get("title") or None)
        key = identifier
        if key is None and resolve is not None:
            # Resolve on whatever identifies the entry: its label, else its URL.
            for candidate in (label, entry.get("url"), entry.get("official_url")):
                if candidate:
                    key = resolve(candidate, entry)
                    if key:
                        break
        if not label:
            # Fall back to a URL or the id itself so a label-less entry is neither
            # dropped nor allowed to collide with every other label-less entry.
            label = entry.get("url") or entry.get("official_url") or key
        # Dedupe on identity when known, otherwise on the text we actually recorded.
        dedupe = key if key else ("label", (label or json.dumps(entry, sort_keys=True)).casefold())
        if dedupe in seen:
            return
        seen.add(dedupe)
        items.append({"label": label, "id": key, "raw": entry})

    def walk(collection):
        if isinstance(collection, dict):
            collection = [collection]
        if not isinstance(collection, list):
            return
        for entry in collection:
            if isinstance(entry, str):
                add(entry)
            elif isinstance(entry, dict):
                add(None, entry, entry.get("id") or entry.get("key"))

    # Structured shapes a benchmark may use.
    for field_name in ("items", "entities", "programs", "records", "results"):
        if field_name in structured:
            walk(structured[field_name])
    for field_name in item_fields:
        if field_name in structured and not items:
            add(structured.get(field_name))

    source = "structured" if items else "text"

    # Prose fallback: URLs first (unambiguous), then labels the resolver accepts.
    if not items and text:
        for url in _urls_in(text):
            add(url, {"url": url})
        source = "text"

    unresolved = [item for item in items if not item["id"]]
    if unresolved:
        warnings.append("unresolved_item")

    values = structured.get("values") or {}
    fields = structured.get("fields") or {}
    scalars = dict(values) if isinstance(values, dict) else {}
    if isinstance(structured.get("scalars"), dict):
        scalars.update(structured["scalars"])

    references = []
    for key in ("citations", "references", "sources", "urls"):
        value = structured.get(key)
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            references += [canonical_url(v) or v for v in value if isinstance(v, str)]
    references += [url for url in _urls_in(text)]
    references = sorted({r for r in references if r})

    tables = structured.get("tables") or []
    if isinstance(tables, dict):
        tables = [tables]

    return {
        "items": items,
        "unresolved": unresolved,
        "fields": fields if isinstance(fields, dict) else {},
        "tables": tables if isinstance(tables, list) else [],
        "scalars": scalars,
        "references": references,
        "count": structured.get("count"),
        "answer_text": text,
        "warnings": sorted(set(warnings)),
        "source": source,
    }
