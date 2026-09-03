"""The public surface: audit(), guard(), and baseline helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .checks import ALL_CHECKS
from .finding import Report
from .profile import drift as _drift
from .profile import load_profile, profile, save_profile
from .stream import DEFAULT_CHUNK, DEFAULT_SAMPLE, exact_findings, scan
from .suppress import apply, read_baseline, write_baseline


def audit(df: pd.DataFrame, target: str | None = None,
          key: Iterable[str] | None = None,
          ignore: Iterable[str] | None = None,
          baseline: str | Path | Iterable[str] | None = None,
          profile_path: str | Path | dict | None = None,
          checks=ALL_CHECKS) -> Report:
    """Run every check against a table and return what it found.

    Arguments:
        df:       the table, after whatever cleaning you already do.
        target:   the outcome column, if there is one. Unlocks the missingness
                  check -- the one that catches selection on the answer.
        key:      columns you believe uniquely identify a row. Checked, not
                  trusted; this is how "Queue ID" turned out not to be a key in
                  three sources out of four.
        ignore:   glob patterns over "code:column" for rules you have decided
                  are not problems here. See suppress.matches.
        baseline: a path to a baseline file, or fingerprints directly. Findings
                  already accepted there are suppressed, so only new ones fail.
        profile_path: a profile written from an earlier version of this table.
                  Adds drift findings -- a column that vanished, a dtype that
                  flipped, a field that stopped being populated, a unit that
                  changed. This is the question a running pipeline actually has.

    Nothing here mutates the frame.
    """
    found, ran = [], []
    for check in checks:
        ran.append(check.__name__)
        found.extend(check(df, target=target, key=key))

    if profile_path is not None:
        ran.append("drift")
        found.extend(_drift(df, profile_path))

    known = baseline
    if isinstance(baseline, (str, Path)):
        known = read_baseline(baseline)
    kept, hidden = apply(found, ignore=ignore, baseline=known)

    return Report(findings=kept, checks_run=ran, rows=len(df),
                  columns=df.shape[1], suppressed=hidden)


def guard(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """audit(), but raise on anything critical. Returns the frame for chaining.

        df = pipelie.guard(df, target="built", key=["id"])

    The point is to fail the run. A pipeline that reports green while the
    numbers are wrong is the failure mode this whole package exists for.
    """
    audit(df, **kw).raise_for_critical()
    return df


def accept(df: pd.DataFrame, path: str | Path = "pipelie-baseline.json",
           **kw) -> int:
    """Freeze today's findings as accepted debt, and return how many.

    Run once when adopting the tool on a table that already has problems. From
    then on `audit(df, baseline=path)` reports only what is new.
    """
    kw.pop("baseline", None)
    return write_baseline(path, audit(df, **kw).findings)


def snapshot(df: pd.DataFrame, path: str | Path = "pipelie-profile.json") -> dict:
    """Record what this table looks like now, for comparison later.

    Stores a small JSON summary -- dtypes, null rates, quantiles, top category
    shares -- and never the data itself. Pass the same path to
    `audit(..., profile_path=path)` on a later run to be told what changed.
    """
    prof = profile(df)
    save_profile(prof, path)
    return prof


def audit_file(path: str | Path, target: str | None = None,
               key: Iterable[str] | None = None,
               ignore: Iterable[str] | None = None,
               baseline: str | Path | Iterable[str] | None = None,
               profile_path: str | Path | dict | None = None,
               chunksize: int = DEFAULT_CHUNK,
               sample: int = DEFAULT_SAMPLE,
               exact_duplicates: bool = True) -> Report:
    """Audit a file too large to hold in memory.

    Streams it once. Row counts and duplicates are computed over every row --
    duplicates cannot be sampled, because two copies of a row will almost never
    both be drawn. Everything else runs on a uniform reservoir sample, which is
    ample for the proportions those checks test.

    Below `sample` rows this is exactly `audit()` on the whole file, and says so
    by leaving `sampled_rows` at zero.
    """
    df, exact = scan(path, key=key, chunksize=chunksize, sample=sample,
                     exact_duplicates=exact_duplicates)

    # duplicate_rows is computed exactly below; running it on the sample too
    # would report a smaller count for the same defect.
    checks = [c for c in ALL_CHECKS if c.__name__ != "duplicate_rows"]
    found, ran = [], []
    for check in checks:
        ran.append(check.__name__)
        found.extend(check(df, target=target, key=None))
    ran.append("duplicate_rows")
    found.extend(exact_findings(exact, key))

    if profile_path is not None:
        ran.append("drift")
        found.extend(_drift(df, profile_path))

    known = baseline
    if isinstance(baseline, (str, Path)):
        known = read_baseline(baseline)
    kept, hidden = apply(found, ignore=ignore, baseline=known)

    return Report(findings=kept, checks_run=ran, rows=exact["rows"],
                  columns=len(exact["columns"]), suppressed=hidden,
                  sampled_rows=exact["sample_rows"] if exact["sampled"] else 0)
