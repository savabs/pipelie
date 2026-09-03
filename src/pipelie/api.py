"""The public surface: audit(), and a guard for use inside a pipeline."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .checks import ALL_CHECKS
from .finding import Report


def audit(df: pd.DataFrame, target: str | None = None,
          key: Iterable[str] | None = None,
          checks=ALL_CHECKS) -> Report:
    """Run every check against a table and return what it found.

    Arguments:
        df:     the table, after whatever cleaning you already do.
        target: the outcome column, if there is one. Unlocks the missingness
                check -- the one that catches selection on the answer.
        key:    columns you believe uniquely identify a row. Checked, not
                trusted; this is how "Queue ID" turned out not to be a key in
                three sources out of four.

    Nothing here mutates the frame.
    """
    findings, ran = [], []
    for check in checks:
        ran.append(check.__name__)
        findings.extend(check(df, target=target, key=key))
    return Report(findings=findings, checks_run=ran,
                  rows=len(df), columns=df.shape[1])


def guard(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """audit(), but raise on anything critical. Returns the frame for chaining.

        df = pipelie.guard(df, target="built", key=["id"])

    The point is to fail the run. A pipeline that reports green while the
    numbers are wrong is the failure mode this whole package exists for.
    """
    audit(df, **kw).raise_for_critical()
    return df
