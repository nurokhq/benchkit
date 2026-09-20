"""Fetch the text a claim is verified against.

Verification is only meaningful if the judge can actually see the source, so the text
comes from the one place that is independent of every condition under test: the URL
the answer itself cited. Reading a local copy of a knowledge base here would make the
engine's idea of the truth depend on a corpus that one condition was allowed to read
and another was not, which is exactly the asymmetry scoring must not have.

Everything is cached on disk, because a single task can cite dozens of pages and a
comparison run scores two conditions over the same corpus.

Text is truncated before it reaches the judge. The claim being checked is always a
short field, so a bounded window is enough, and an unbounded page would blow the
context budget for no accuracy gain.
"""

import hashlib
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120 Safari/537.36")
MAX_CHARS = 20000


def _strip_markup(text: str) -> str:
    """Reduce a page to readable prose so the judge sees content, not boilerplate."""
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;?", "&", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


class SourceText:
    """Resolve and cache the text behind a URL."""

    def __init__(self, cache_dir, allow_fetch=True, timeout=20, attempts=2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_fetch = allow_fetch
        self.timeout = timeout
        #: A transient network failure must not become a permanent property of a
        #: citation. Failures were cached as an empty file, so one blip during grading
        #: zeroed every claim on that record -- and did it again on every regrade,
        #: because the empty file counted as a cache hit. Retry instead, and write
        #: nothing when it still fails, so a later run can recover.
        self.attempts = max(1, attempts)
        # Judging and task execution run concurrently, so the counters are shared state.
        self._lock = threading.Lock()
        self.stats = {"cache": 0, "fetched": 0, "missing": 0}

    def _bump(self, key):
        with self._lock:
            self.stats[key] += 1

    def get(self, url):
        """Return (text, provenance) for a URL, or (None, "missing")."""
        if not url:
            self._bump("missing")
            return None, "missing"

        cache_file = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".txt")
        if cache_file.is_file():
            cached = cache_file.read_text(encoding="utf-8", errors="replace")[:MAX_CHARS]
            if cached.strip():
                self._bump("cache")
                return cached, "cache"
            # An empty entry is a leftover from a failed fetch, not a page that says
            # nothing. Drop it and go to the network rather than treating it as evidence.
            cache_file.unlink(missing_ok=True)
        if not self.allow_fetch:
            self._bump("missing")
            return None, "missing"

        for _attempt in range(self.attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, OSError, ValueError):
                continue
            text = _strip_markup(raw)[:MAX_CHARS]
            if text.strip():
                cache_file.write_text(text, encoding="utf-8")
                self._bump("fetched")
                return text, "fetched"
        self._bump("missing")
        return None, "missing"

    def prime(self, urls):
        """Resolve several URLs up front, in parallel.

        Called before judging so the page fetches overlap instead of being serialised
        behind the claim checks.
        """
        from concurrent.futures import ThreadPoolExecutor

        unique = [u for u in dict.fromkeys(urls) if u]
        if not unique:
            return {}
        with ThreadPoolExecutor(max_workers=min(16, len(unique))) as pool:
            texts = list(pool.map(lambda u: self.get(u)[0], unique))
        return dict(zip(unique, texts))


def attach_sources(records, source_text: SourceText):
    """Give each record the text of the page it cites, in place.

    Resolved in parallel first. Each distinct citation is a live HTTP request once the
    disk cache misses, and doing that inside the loop serialised every page fetch
    behind the one before it -- forty records meant forty sequential round trips before
    a single claim could be judged. The parallel pass populates the cache, so the loop
    below is then a dictionary lookup.
    """
    source_text.prime([record.get("url") for record in records])
    for record in records:
        text, provenance = source_text.get(record.get("url"))
        record["source_text"] = text
        record["source_provenance"] = provenance
    return records
