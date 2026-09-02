"""Persistent clue cache.

An LLM call for a clue is expensive and, for a fixed (model, clue, length,
pattern, mode), deterministic enough to reuse: the whole evaluation loop is
"run the same 30 puzzles again with one knob changed", so without a cache every
iteration pays full price for answers that have not changed.

Two design notes that matter downstream:

* Hits and misses are counted separately and reported by :meth:`ClueCache.stats`.
  The harness quotes "cost on a cold cache", and a cache hit that silently
  incremented the call counter would make that number a lie.
* Storage is SQLite in WAL mode with one connection per thread. The candidate
  source issues batches from a thread pool, and SQLite connections are not
  shareable across threads; WAL is what keeps concurrent readers from blocking
  on the writer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from xword.config import DEFAULT_CACHE_PATH
from xword.core.types import Candidate

#: Bumped when the stored payload shape changes, so old rows simply miss rather
#: than deserialising into something the reader does not expect.
SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS clue_cache (
    key      TEXT PRIMARY KEY,
    version  INTEGER NOT NULL,
    payload  TEXT NOT NULL,
    created  REAL NOT NULL
)
"""


def context_digest(
    puzzle_meta: Mapping[str, str] | None = None,
    crossing_clues: Sequence[str] = (),
) -> str:
    """Short digest of the prompt context that is not the clue itself.

    The batch prompt carries the puzzle's title, day of week and difficulty, and
    the hard-clue prompt additionally lists the crossing clues. Both change what
    the model is asked, so both have to reach the cache key -- otherwise one
    puzzle's answers get served for another puzzle's identically-worded clue,
    which is wrong quietly rather than loudly.

    Only keys the prompts actually render are included, so an unrelated metadata
    field cannot needlessly cost a cache miss.
    """
    meta = puzzle_meta or {}
    parts = [f"{k}={meta[k]}" for k in ("title", "dow", "difficulty") if meta.get(k)]
    parts.extend(f"x={c}" for c in crossing_clues if c)
    if not parts:
        return ""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_key(
    model: str,
    clue: str,
    length: int,
    pattern: str | None,
    mode: str,
    context: str = "",
) -> str:
    """Stable content hash for one clue lookup.

    Every field that can change the answer is in the key. ``pattern`` is part of
    it because a request made with known crossing letters is a genuinely
    different question from the same clue asked cold, and ``mode`` because the
    hard-clue prompt asks for more analysis than the batch prompt.

    Note what is *not* in the key: ``k``. Asking for more candidates than a
    cached entry happens to hold returns the smaller list rather than paying for
    a new call, which is the right trade for an eval harness that re-runs the
    same puzzles.

    ``context`` carries whatever else the prompt renders -- see
    :func:`context_digest`. It defaults to empty so a caller that genuinely has
    no puzzle context keeps the old key.
    """
    clue_norm = " ".join(clue.split()).casefold()
    pattern_norm = (pattern or "").upper()
    blob = "\x1f".join(
        (
            str(SCHEMA_VERSION),
            model,
            mode,
            str(length),
            pattern_norm,
            clue_norm,
            context,
        )
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ClueCache:
    """Thread-safe SQLite cache of candidate lists, created on demand."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._writes = 0
        # Open once eagerly so a bad path fails at construction rather than in
        # the middle of a solve, on a worker thread.
        self._conn()

    # -- connections ------------------------------------------------------- #

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, opened and migrated on first use."""
        if self._closed:
            raise RuntimeError("ClueCache is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            # WAL is unavailable on some network filesystems; the default
            # journal is slower but correct, and a cache is not worth crashing
            # a solve over.
            pass
        conn.execute(_CREATE_SQL)
        conn.commit()
        self._local.conn = conn
        with self._lock:
            self._conns.append(conn)
        return conn

    # -- api --------------------------------------------------------------- #

    def get(self, key: str) -> list[Candidate] | None:
        """Cached candidates for ``key``, or ``None`` on a miss."""
        conn = self._conn()
        row = conn.execute(
            "SELECT payload FROM clue_cache WHERE key = ? AND version = ?",
            (key, SCHEMA_VERSION),
        ).fetchone()
        if row is None:
            with self._lock:
                self._misses += 1
            return None
        try:
            raw = json.loads(row[0])
            candidates = [
                Candidate(
                    answer=str(item["answer"]),
                    score=float(item["score"]),
                    source=str(item.get("source", "llm")),
                    rationale=str(item.get("rationale", "")),
                )
                for item in raw
            ]
        except (ValueError, TypeError, KeyError):
            # A corrupt row is treated as a miss and dropped, so a bad write can
            # never wedge the cache permanently.
            conn.execute("DELETE FROM clue_cache WHERE key = ?", (key,))
            conn.commit()
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            self._hits += 1
        return candidates

    def put(self, key: str, candidates: Sequence[Candidate]) -> None:
        """Store (or replace) the candidate list for ``key``."""
        payload = json.dumps(
            [
                {
                    "answer": c.answer,
                    "score": c.score,
                    "source": c.source,
                    "rationale": c.rationale,
                }
                for c in candidates
            ],
            separators=(",", ":"),
        )
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO clue_cache (key, version, payload, created) "
            "VALUES (?, ?, ?, ?)",
            (key, SCHEMA_VERSION, payload, time.time()),
        )
        conn.commit()
        with self._lock:
            self._writes += 1

    def stats(self) -> dict[str, int]:
        """Hit/miss/write counters for this instance, plus the row count."""
        rows = 0
        if not self._closed:
            rows = int(
                self._conn().execute("SELECT COUNT(*) FROM clue_cache").fetchone()[0]
            )
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
                "rows": rows,
            }

    def close(self) -> None:
        """Close every thread's connection. Idempotent."""
        with self._lock:
            self._closed = True
            conns, self._conns = self._conns, []
        for conn in conns:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._local = threading.local()

    # -- niceties ---------------------------------------------------------- #

    def __enter__(self) -> ClueCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "closed" if self._closed else "open"
        return f"ClueCache(path={str(self.path)!r}, {state})"


__all__ = ["ClueCache", "SCHEMA_VERSION", "cache_key"]
