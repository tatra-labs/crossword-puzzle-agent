"""Fetching an evaluation corpus from public archives.

Nothing in this module ships data. It fetches it, onto the machine of whoever
runs the tool, into a gitignored directory.

The distinction matters legally. Two very different sources are handled here:

``NYT_BASE`` (the ``doshea/nyt_crosswords`` archive)
    New York Times puzzles, 1976-2018. **This content is New York Times
    copyright.** It is fetched by the user, for evaluating a solver on their own
    machine, and it MUST NOT be committed to this repository or redistributed.
    :func:`fetch` calls :func:`ensure_gitignored` on the destination before it
    writes a single byte, so an accidental ``git add -A`` cannot sweep it in.

``WORDLIST_URL`` (the ``dwyl/english-words`` list)
    Released under the Unlicense (public domain). This one *may* be committed,
    and the bundled lexicon is built from it.

Everything is network I/O, so nothing here runs at import time and no other
module should import this one for anything but the fetch commands.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from xword.config import FETCHED_PUZZLE_DIR, LEXICON_DIR

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

#: Raw-file root of the public NYT archive. Content is NYT copyright: fetch it,
#: evaluate against it, never commit it.
NYT_BASE = "https://raw.githubusercontent.com/doshea/nyt_crosswords/master"

#: Years the archive covers. Anything outside this range is a guaranteed 404.
NYT_YEARS = range(1976, 2019)

#: The bundled lexicon's source list. Unlicense (public domain) -- safe to
#: commit, unlike the puzzles above.
WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

#: Weekday names indexed by ``date.weekday()``. Hard-coded rather than taken
#: from ``calendar.day_name`` because that is locale-dependent, and these
#: strings are report keys that must be stable across machines. They also match
#: the ``dow`` field inside the NYT JSON.
DOW_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

#: A real, identifying User-Agent. GitHub asks for one, and an anonymous
#: scraper-looking client is the first thing a public host rate-limits.
USER_AGENT = (
    "crossword-puzzle-agent/0.1 "
    "(+https://github.com/tatra-labs/crossword-puzzle-agent) "
    f"python-httpx/{httpx.__version__}"
)

#: HTTP statuses worth another attempt: transient server and throttling errors.
#: A 404 is *not* retried -- the archive simply has no puzzle for many dates.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 0.6

#: What :func:`ensure_gitignored` writes. The bare ``*`` also ignores the
#: ``.gitignore`` itself, which is what we want: the whole directory is invisible
#: to git, permanently.
_GITIGNORE_BODY = """\
# New York Times puzzle JSON, fetched for local evaluation only.
# This content is NYT copyright and must never be committed or redistributed.
# The bare '*' hides this file too, so the directory is invisible to git.
*
"""


# --------------------------------------------------------------------------- #
# Plans and reports
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FetchPlan:
    """A concrete list of puzzle dates and where they will land.

    Built by :func:`plan_fetch` and handed to :func:`fetch`. Keeping the plan a
    separate value means a run can be printed, diffed, or replayed exactly --
    reproducibility of the eval set is the whole point.
    """

    dates: tuple[date, ...]
    dest: Path

    def urls(self) -> tuple[str, ...]:
        """Upstream URL for each planned date, in plan order."""
        return tuple(f"{NYT_BASE}/{d:%Y/%m/%d}.json" for d in self.dates)

    def paths(self) -> tuple[Path, ...]:
        """Local destination for each planned date, in plan order."""
        return tuple(puzzle_path(self.dest, d) for d in self.dates)

    def by_dow(self) -> dict[str, int]:
        """Planned count per weekday -- the balance check before spending bandwidth."""
        counts = dict.fromkeys(DOW_NAMES, 0)
        for d in self.dates:
            counts[DOW_NAMES[d.weekday()]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.dates)


@dataclass(slots=True)
class FetchReport:
    """What a :func:`fetch` run actually did.

    ``by_dow`` counts the puzzles *present on disk* afterwards (freshly
    downloaded plus already there), not the ones requested: a plan can be
    perfectly balanced and still yield a lopsided corpus if a weekday's dates
    happen to be missing upstream, and that is exactly the thing that would
    quietly bias a solve rate.

    ``failures`` holds ``(iso_date, reason)`` pairs. Missing dates are normal --
    the archive is not complete -- so failures are recorded, never raised.
    """

    requested: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    by_dow: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DOW_NAMES, 0))
    failures: list[tuple[str, str]] = field(default_factory=list)
    #: Downloaded but not checked against the parser, because ``xword.io.nyt_json``
    #: could not be imported. These files are kept; treat the count as a warning.
    unvalidated: int = 0

    @property
    def available(self) -> int:
        """Puzzles on disk for this plan after the run."""
        return self.downloaded + self.skipped_existing

    def summary(self) -> str:
        """One-line digest for a CLI to print. Library code never prints."""
        balance = " ".join(f"{name[:3]}={self.by_dow[name]}" for name in DOW_NAMES)
        extra = f", {self.unvalidated} unvalidated" if self.unvalidated else ""
        return (
            f"{self.available}/{self.requested} puzzles on disk "
            f"({self.downloaded} new, {self.skipped_existing} existing, "
            f"{self.failed} failed{extra}) [{balance}]"
        )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def puzzle_path(root: Path, day: date) -> Path:
    """Local path for one puzzle: ``<root>/YYYY/MM/DD.json``.

    Mirrors the upstream layout so the date is recoverable from the path alone
    and a fetched tree can be diffed against the archive.
    """
    return Path(root) / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.json"


def date_from_path(path: Path) -> date | None:
    """Recover the puzzle date from a ``YYYY/MM/DD.json`` path, or ``None``."""
    try:
        day = int(path.stem)
        month = int(path.parent.name)
        year = int(path.parent.parent.name)
        return date(year, month, day)
    except (ValueError, OSError):
        return None


def ensure_gitignored(root: Path) -> None:
    """Make ``root`` permanently invisible to git.

    Writes a ``.gitignore`` containing ``*`` inside the fetched-corpus directory.
    The corpus is NYT copyright, so "someone forgot and ran ``git add -A``" has
    to be structurally impossible rather than merely discouraged. Idempotent:
    an existing file that already ignores everything is left alone, and one that
    does not is appended to rather than overwritten.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / ".gitignore"

    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        if any(line.strip() == "*" for line in existing.splitlines()):
            return
        separator = "" if existing.endswith("\n") or not existing else "\n"
        target.write_text(existing + separator + _GITIGNORE_BODY, encoding="utf-8", newline="\n")
        return

    target.write_text(_GITIGNORE_BODY, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _dates_in_year(year: int) -> list[date]:
    first = date(year, 1, 1)
    last = date(year, 12, 31)
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _sample_year(year: int, per_year: int, balanced_dow: bool, rng: random.Random) -> list[date]:
    days = _dates_in_year(year)
    if per_year >= len(days):
        return days

    if not balanced_dow:
        return sorted(rng.sample(days, per_year))

    buckets: list[list[date]] = [[] for _ in range(7)]
    for day in days:
        buckets[day.weekday()].append(day)

    base, remainder = divmod(per_year, 7)
    # Spread the remainder over a seeded choice of weekdays rather than always
    # the first few, so repeated years do not all over-sample Monday.
    bonus = set(rng.sample(range(7), remainder)) if remainder else set()

    picked: list[date] = []
    for weekday in range(7):
        want = base + (1 if weekday in bonus else 0)
        pool = buckets[weekday]
        picked.extend(rng.sample(pool, min(want, len(pool))))
    return sorted(picked)


def plan_fetch(
    *,
    years: Sequence[int] | None = None,
    per_year: int = 14,
    balanced_dow: bool = True,
    seed: int = 0,
    dest: Path | None = None,
) -> FetchPlan:
    """Choose which puzzles to fetch, reproducibly.

    The sample is balanced across the seven days of the week by default because
    NYT difficulty rises monotonically from Monday to Saturday (Sunday is
    Thursday-hard but much larger). A sample that drifted toward Mondays would
    make any reported solve rate meaningless and incomparable between runs, so
    balance is the default and turning it off is an explicit choice.

    Draws come from a ``random.Random(seed)``, so ``(years, per_year,
    balanced_dow, seed)`` fully determines the date list; the eval set can be
    described in a paper by those four values instead of a file of dates.

    Some planned dates will not exist upstream -- the archive is not complete --
    and :func:`fetch` records those as failures rather than raising.
    """
    if per_year < 0:
        raise ValueError(f"per_year must be non-negative, got {per_year}")

    chosen = sorted({int(y) for y in (years if years is not None else NYT_YEARS)})
    out_of_range = [y for y in chosen if y not in NYT_YEARS]
    if out_of_range:
        raise ValueError(
            f"years outside the archive {NYT_YEARS.start}-{NYT_YEARS.stop - 1}: {out_of_range}"
        )

    rng = random.Random(seed)
    dates: list[date] = []
    for year in chosen:
        dates.extend(_sample_year(year, per_year, balanced_dow, rng))

    return FetchPlan(dates=tuple(sorted(dates)), dest=Path(dest or FETCHED_PUZZLE_DIR))


# --------------------------------------------------------------------------- #
# Validation hook
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Validator:
    """Adapter around ``xword.io.nyt_json``, which may not exist yet.

    That module is written independently, so this resolves it lazily and copes
    with the parser taking a path, raw text, or a decoded dict. If it cannot be
    imported at all, ``available`` is ``False`` and downloads are kept but
    counted as unvalidated -- a missing reader is not a reason to throw away a
    corpus the user just spent bandwidth on.
    """

    func: Callable[[Any], Any] | None
    style: str = "unknown"

    @property
    def available(self) -> bool:
        return self.func is not None

    def check(self, path: Path, text: str) -> tuple[bool, str]:
        """``(ok, reason)``. ``ok`` is ``True`` when unavailable -- see above."""
        if self.func is None:
            return True, "validator unavailable"

        # ``None`` is a sentinel for "the decoded JSON object"; the rest are the
        # literal argument. Ordering puts the shape the signature suggests first
        # and keeps the others as fallbacks.
        arguments: list[Any]
        if self.style == "path":
            arguments = [path, None, text]
        elif self.style == "text":
            arguments = [text, None, path]
        elif self.style == "data":
            arguments = [None, path, text]
        else:
            arguments = [path, None, text]

        first_error: Exception | None = None
        for argument in arguments:
            payload = self._decode(text) if argument is None else argument
            if payload is None:
                continue
            try:
                result = self.func(payload)
            except TypeError as exc:  # possibly just the wrong argument shape
                first_error = first_error or exc
                continue
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"
            if result is None:
                first_error = first_error or ValueError("parser returned None")
                continue
            slots = getattr(result, "slots", None)
            if slots is not None and len(slots) == 0:
                return False, "parsed puzzle has no entries"
            return True, "ok"

        reason = f"{type(first_error).__name__}: {first_error}" if first_error else "unparseable"
        return False, reason

    @staticmethod
    def _decode(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def _load_validator() -> _Validator:
    """Resolve ``parse_nyt_json`` if the reader module is importable."""
    try:
        from xword.io import nyt_json  # noqa: PLC0415 - deliberately lazy
    except Exception:
        return _Validator(func=None)

    func = None
    for name in ("parse_nyt_json", "read_nyt_json", "load_nyt_json"):
        candidate = getattr(nyt_json, name, None)
        if callable(candidate):
            func = candidate
            break
    if func is None:
        return _Validator(func=None)

    return _Validator(func=func, style=_argument_style(func))


def _argument_style(func: Callable[..., Any]) -> str:
    """Guess whether the parser wants a path, raw text, or a decoded dict."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return "unknown"
    parameters = list(signature.parameters.values())
    if not parameters:
        return "unknown"

    first = parameters[0]
    annotation = str(first.annotation).lower()
    name = first.name.lower()
    if "path" in annotation or name in {"path", "file", "filename", "fp", "source"}:
        return "path"
    if "mapping" in annotation or "dict" in annotation or name in {"data", "payload", "obj", "raw"}:
        return "data"
    if "str" in annotation or name in {"text", "content", "blob"}:
        return "text"
    return "unknown"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _backoff_delay(url: str, attempt: int) -> float:
    """Exponential backoff with jitter that is deterministic per URL.

    Jitter keeps eight workers from retrying in lockstep; seeding it on the URL
    keeps a re-run byte-identical in timing as well as in output, which the
    repo's determinism rule asks for.
    """
    spread = random.Random(f"{url}#{attempt}").random()
    return _BACKOFF_BASE * (2**attempt) * (0.5 + spread)


def _get(
    client: httpx.Client,
    url: str,
    *,
    attempts: int = _MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str | None, str]:
    """Fetch ``url`` with retries. Returns ``(text, reason)``; text is ``None`` on failure."""
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                if not response.content:
                    return None, "empty response body"
                return response.text, "ok"
            last = f"HTTP {response.status_code}"
            if response.status_code not in RETRY_STATUS:
                return None, last
        if attempt + 1 < attempts:
            sleep(_backoff_delay(url, attempt))
    return None, f"{last} (after {attempts} attempts)"


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and ``os.replace`` so a killed run leaves no half-file.

    A truncated JSON file that still parses as *something* is the worst outcome
    for an eval corpus, so partial writes never get the real name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


@dataclass(slots=True)
class _Outcome:
    day: date
    status: str  # "downloaded" | "skipped" | "failed"
    reason: str = "ok"
    validated: bool = True


def _fetch_one(
    client: httpx.Client,
    day: date,
    dest: Path,
    *,
    overwrite: bool,
    validator: _Validator,
) -> _Outcome:
    path = puzzle_path(dest, day)
    if path.exists() and not overwrite:
        return _Outcome(day, "skipped")

    url = f"{NYT_BASE}/{day:%Y/%m/%d}.json"
    text, reason = _get(client, url)
    if text is None:
        return _Outcome(day, "failed", reason)

    _write_atomic(path, text)

    ok, detail = validator.check(path, text)
    if not ok:
        # A corpus with silently-broken files corrupts every downstream number,
        # so a file that will not parse is deleted rather than left to be found
        # months later by a confusing eval result.
        path.unlink(missing_ok=True)
        return _Outcome(day, "failed", f"unparseable, deleted: {detail}")

    return _Outcome(day, "downloaded", validated=validator.available)


def fetch(
    plan: FetchPlan,
    *,
    workers: int = 8,
    timeout: float = 20.0,
    overwrite: bool = False,
    progress: bool = True,
) -> FetchReport:
    """Download every puzzle in ``plan``, skipping what is already on disk.

    The destination is gitignored first (see :func:`ensure_gitignored`) because
    the NYT content this pulls down must never reach the repository.

    Individual dates fail routinely -- the archive has gaps, and a date with no
    published puzzle is not an error -- so per-date problems land in
    ``report.failures`` and the run continues. Each downloaded file is checked
    against ``xword.io.nyt_json.parse_nyt_json`` and deleted if it does not
    parse; if that reader is not importable the file is kept and counted in
    ``report.unvalidated``.

    Progress goes to stderr, never stdout, so piping a CLI's JSON output stays
    clean.
    """
    dest = Path(plan.dest)
    ensure_gitignored(dest)

    report = FetchReport(requested=len(plan.dates))
    if not plan.dates:
        return report

    validator = _load_validator()
    workers = max(1, min(workers, len(plan.dates)))

    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    bar = _make_bar(len(plan.dates), progress)
    try:
        with httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            limits=limits,
        ) as client, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fetch_one,
                    client,
                    day,
                    dest,
                    overwrite=overwrite,
                    validator=validator,
                ): day
                for day in plan.dates
            }
            for future in as_completed(futures):
                day = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # a worker bug must not lose the rest
                    outcome = _Outcome(day, "failed", f"{type(exc).__name__}: {exc}")
                _record(report, outcome)
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    report.failures.sort()
    return report


def _record(report: FetchReport, outcome: _Outcome) -> None:
    if outcome.status == "downloaded":
        report.downloaded += 1
        if not outcome.validated:
            report.unvalidated += 1
    elif outcome.status == "skipped":
        report.skipped_existing += 1
    else:
        report.failed += 1
        report.failures.append((outcome.day.isoformat(), outcome.reason))
        return
    report.by_dow[DOW_NAMES[outcome.day.weekday()]] += 1


def _make_bar(total: int, progress: bool) -> Any:
    if not progress:
        return None
    try:
        from tqdm import tqdm  # noqa: PLC0415 - optional at runtime
    except Exception:
        return None
    return tqdm(total=total, unit="puzzle", desc="fetch", file=sys.stderr, leave=False)


# --------------------------------------------------------------------------- #
# Inspecting what is on disk
# --------------------------------------------------------------------------- #


def corpus_stats(root: Path | None = None) -> dict[str, object]:
    """Summarise the fetched corpus: counts by year and weekday, plus totals.

    Used by ``xword doctor`` to answer "do I have an eval set, and is it
    balanced?". Weekdays come from the ``YYYY/MM/DD`` path rather than the
    ``dow`` field inside each file, so this stays a directory walk instead of
    parsing thousands of JSON documents.
    """
    root = Path(root or FETCHED_PUZZLE_DIR)
    by_year: dict[int, int] = {}
    by_dow = dict.fromkeys(DOW_NAMES, 0)
    total = 0
    total_bytes = 0
    unrecognised = 0
    earliest: date | None = None
    latest: date | None = None

    if root.exists():
        for path in sorted(root.rglob("*.json")):
            if not path.is_file():
                continue
            day = date_from_path(path)
            if day is None:
                unrecognised += 1
                continue
            total += 1
            total_bytes += path.stat().st_size
            by_year[day.year] = by_year.get(day.year, 0) + 1
            by_dow[DOW_NAMES[day.weekday()]] += 1
            earliest = day if earliest is None or day < earliest else earliest
            latest = day if latest is None or day > latest else latest

    return {
        "root": str(root),
        "exists": root.exists(),
        "total": total,
        "bytes": total_bytes,
        "by_year": {year: by_year[year] for year in sorted(by_year)},
        "by_dow": by_dow,
        "years": len(by_year),
        "earliest": earliest.isoformat() if earliest else None,
        "latest": latest.isoformat() if latest else None,
        "unrecognised": unrecognised,
        "gitignored": (root / ".gitignore").exists(),
    }


# --------------------------------------------------------------------------- #
# Word list
# --------------------------------------------------------------------------- #

_WORDLIST_LOCK = threading.Lock()


def fetch_wordlist(
    dest: Path | None = None, *, timeout: float = 60.0, overwrite: bool = False
) -> Path:
    """Download the source word list for the bundled lexicon.

    Unlike the puzzles, this file is public domain -- ``dwyl/english-words`` is
    released under the Unlicense -- so it MAY be committed to the repository and
    shipped with the package. It is the raw input to the lexicon builder and is
    written verbatim apart from newline normalisation.

    Returns the path, whether it was just downloaded or already present. Raises
    on failure: unlike an individual puzzle date, this is a single required
    resource and a silent miss would leave the lexicon empty.
    """
    target = Path(dest or (LEXICON_DIR / "words_alpha.txt"))
    if target.exists() and not overwrite and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with _WORDLIST_LOCK:
        with httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            text, reason = _get(client, WORDLIST_URL)
        if text is None:
            raise RuntimeError(f"could not fetch word list from {WORDLIST_URL}: {reason}")

        words = [line.strip() for line in text.splitlines()]
        words = [w for w in words if w]
        if len(words) < 10_000:
            # The real list has ~370k entries; anything tiny means we were served
            # an error page or a truncated body, and writing it would quietly
            # cripple the lexicon.
            raise RuntimeError(
                f"word list from {WORDLIST_URL} looks wrong: only {len(words)} lines"
            )

        _write_atomic(target, "\n".join(words) + "\n")
    return target


__all__ = [
    "DOW_NAMES",
    "NYT_BASE",
    "NYT_YEARS",
    "USER_AGENT",
    "WORDLIST_URL",
    "FetchPlan",
    "FetchReport",
    "corpus_stats",
    "date_from_path",
    "ensure_gitignored",
    "fetch",
    "fetch_wordlist",
    "plan_fetch",
    "puzzle_path",
]
