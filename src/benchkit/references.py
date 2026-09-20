"""Identity resolution, driven entirely by a benchmark-supplied catalog.

Deciding that a name refers to a known thing is domain knowledge, so the catalog
lives with the benchmark. This module only provides the generic machinery: build
lookup indexes from a catalog, then resolve a label by exact key, then canonical
URL, then an unambiguous exact name. Nothing here is fuzzy — a near miss must not
silently become a hit, because a wrong identity inflates a correctness score.
"""

import re

from .normalize import canonical_url


def label_key(value) -> str:
    """Normalise a label for exact comparison (case and separator folding only)."""
    return re.sub(r"[\s_-]+", " ", str(value or "")).strip().casefold()


class Catalog:
    """Lookup indexes over benchmark entries.

    Each entry needs a canonical id under `id_field`. `url_fields` and `name_fields`
    say where to find the entry's canonical URL and its display names; entries whose
    names collide are dropped from the name index, so an ambiguous name never
    resolves to an arbitrary pick.
    """

    def __init__(self, entries, id_field="id", url_fields=("url",), name_fields=("name", "title")):
        self.entries = list(entries or [])
        self.id_field = id_field
        self.by_id: dict = {}
        self.by_url: dict = {}
        self.by_name: dict[str, list] = {}
        for entry in self.entries:
            key = entry.get(id_field)
            if key is None:
                continue
            self.by_id[key] = entry
            for field in url_fields:
                url = canonical_url(entry.get(field))
                if url:
                    self.by_url.setdefault(url, entry)
            for field in name_fields:
                name = label_key(entry.get(field))
                if name:
                    self.by_name.setdefault(name, []).append(entry)

    def resolve_url(self, value):
        url = canonical_url(value)
        return self.by_url.get(url) if url else None

    def resolve_name(self, value):
        matches = self.by_name.get(label_key(value), [])
        return matches[0] if len(matches) == 1 else None

    def resolve(self, label, entry=None):
        """Resolve to a canonical id by key, then URL, then unambiguous exact name."""
        entry = entry if isinstance(entry, dict) else {}
        for candidate in (entry.get(self.id_field), entry.get("id"), entry.get("key")):
            if candidate in self.by_id:
                return candidate
        for field in ("url", "yc_url", "official_url", "source_url", "link"):
            found = self.resolve_url(entry.get(field))
            if found:
                return found.get(self.id_field)
        for candidate in (label, entry.get("name"), entry.get("title")):
            found = self.resolve_name(candidate)
            if found:
                return found.get(self.id_field)
        return None

    def labels(self) -> list[str]:
        """Every resolvable display name, longest first, for prose scanning."""
        return sorted(self.by_name, key=len, reverse=True)


def scan_text_for_labels(text: str, catalog: Catalog, limit: int = 200) -> list:
    """Find catalog labels mentioned in prose.

    Used only as a fallback when an answer carries no structured items. Matches are
    whole-word and exact after folding; substring hits are deliberately excluded
    because `AI` inside `said` is not a mention.
    """
    lowered = (text or "").casefold()
    found = []
    for name in catalog.labels():
        if len(name) < 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lowered):
            entry = catalog.by_name[name][0]
            found.append({"label": entry.get("name") or entry.get("title"), "entry": entry})
            if len(found) >= limit:
                break
    return found
