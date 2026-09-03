"""Accepting what you already have, so the tool can be adopted at all.

A checker that fails a build on day one over problems that predate it does not
get fixed -- it gets deleted. Two ways to stop that, and they are for different
situations:

  ignore    A rule you have decided is not a problem here, permanently.
            "This key is deliberately non-unique; a composite is built
            downstream." Written down, with a reason, in the repository.

  baseline  Everything wrong today, accepted as debt, so the build fails only
            on something NEW. This is how a checker gets into a codebase that
            already has problems, which is every codebase.

Both work on fingerprints -- rule code plus column, never counts -- so a
finding stays suppressed when the number of affected rows drifts.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

from .finding import Finding


def matches(fingerprint: str, pattern: str) -> bool:
    """Glob match against a fingerprint.

    Patterns are shell globs over "code:column", so all of these work:

        parse_carnage/mixed_formats:Queue Date   one exact finding
        parse_carnage/*                          every date-format finding
        *:Queue ID                               anything about that column
        duplicate_rows/*                         a whole rule family

    A pattern with no colon matches the code alone, so "parse_carnage" and
    "parse_carnage/*" both do the obvious thing.
    """
    if ":" not in pattern:
        pattern = pattern.rstrip("*").rstrip("/") + "*:*"
    return fnmatch(fingerprint, pattern)


def apply(findings: Iterable[Finding], ignore: Iterable[str] | None = None,
          baseline: Iterable[str] | None = None) -> tuple[list[Finding], int]:
    """Return the findings that survive suppression, and how many did not."""
    all_ = list(findings)
    pats = list(ignore or [])
    known = set(baseline or [])
    kept = [f for f in all_
            if f.fingerprint not in known
            and not any(matches(f.fingerprint, p) for p in pats)]
    return kept, len(all_) - len(kept)


def read_baseline(path: str | Path) -> set[str]:
    """Load accepted fingerprints. A missing file is not an error -- it means
    nothing has been accepted yet."""
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    return set(data.get("accepted", []))


def write_baseline(path: str | Path, findings: Iterable[Finding],
                   note: str = "") -> int:
    """Freeze the current findings as accepted debt.

    Stores fingerprints and, alongside them, a human-readable line per entry.
    The messages are for the reader; only the fingerprints are matched on, so
    editing the prose cannot change what is suppressed.
    """
    fs = list(findings)
    payload = {
        "note": note or ("Findings accepted as pre-existing. Delete a line to "
                         "start failing on it again."),
        "accepted": sorted({f.fingerprint for f in fs}),
        "detail": sorted({f"{f.fingerprint}  ({f.severity}) {f.message[:100]}"
                          for f in fs}),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return len(payload["accepted"])
