"""What a table looked like once, and what has changed since.

The checks in checks.py answer "is this table wrong on its own terms". They
cannot answer the question a running pipeline actually has, which is "is this
table different from the one my code was written against".

That difference is where most production breakage lives. Nobody ships a
pipeline against a table that is already broken; they ship against a table that
works, and then something upstream changes. A field stops being populated. An
integer column starts arriving as text because one row had a comma in it. Somebody
switches megawatts to kilowatts. A category that every downstream branch expects
disappears. Every one of those keeps the schema valid and the row count healthy.

A profile is a small JSON summary of a table -- never the data itself -- and
drift() compares a later table against it.

Thresholds favour silence, for the same reason as everywhere else here: a
monitor that cries wolf on ordinary variation gets muted, and then it is not a
monitor. Structural changes (a column vanished, a dtype flipped, a field went
empty) are reported as critical because they break code. Distributional moves
are reported as warnings because they are usually the business, not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .finding import CRITICAL, INFO, WARNING, Finding

FORMAT = 1
TOP_K = 20

# A null rate moving this many percentage points is a field that stopped being
# written, not noise.
NULL_JUMP = 0.10
# Row counts wander. An order of magnitude is a partial load or a double load.
ROW_FACTOR = 2.0
# Medians moving by this factor almost always mean a unit changed.
UNIT_FACTOR = 50.0
# Standardised mean shift. 1.0 is a whole standard deviation -- deliberately
# blunt, because ordinary business drift lives well below it.
SHIFT_D = 1.0
# A category has to be worth noticing before its arrival or departure is news.
CATEGORY_SHARE = 0.01


def _num_summary(s: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {}
    q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
            "min": float(v.min()), "max": float(v.max()),
            "p01": float(q.loc[0.01]), "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.5]), "p75": float(q.loc[0.75]),
            "p99": float(q.loc[0.99])}


def _cat_summary(s: pd.Series) -> dict[str, Any]:
    v = s.dropna().astype(str)
    if v.empty:
        return {}
    share = v.value_counts(normalize=True)
    return {"distinct": int(v.nunique()),
            "top": {str(k): round(float(x), 6) for k, x in share.head(TOP_K).items()}}


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise a table small enough to store and compare. Holds no data."""
    cols: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        entry: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_rate": float(s.isna().mean()) if len(s) else 0.0,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            entry["kind"] = "numeric"
            entry.update(_num_summary(s))
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = s.dropna()
            entry["kind"] = "datetime"
            if not v.empty:
                entry["min"] = str(v.min())
                entry["max"] = str(v.max())
        else:
            entry["kind"] = "categorical"
            entry.update(_cat_summary(s))
        cols[str(c)] = entry
    return {"format": FORMAT, "rows": int(len(df)), "columns": cols}


def save_profile(prof: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(prof, indent=2) + "\n")


def load_profile(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no profile at {p}. Write one first with --profile-write, or "
            "pipelie.save_profile(pipelie.profile(df), path).")
    prof = json.loads(p.read_text())
    if prof.get("format") != FORMAT:
        raise ValueError(f"profile format {prof.get('format')} is not {FORMAT}; "
                         "rewrite it against the current version")
    return prof


def _pct(x: float) -> str:
    return f"{x:.1%}"


def drift(df: pd.DataFrame, prof: dict[str, Any] | str | Path) -> list[Finding]:
    """Compare a table against a stored profile. Returns findings."""
    if isinstance(prof, (str, Path)):
        prof = load_profile(prof)
    old_cols: dict[str, Any] = prof["columns"]
    now = profile(df)
    new_cols: dict[str, Any] = now["columns"]
    out: list[Finding] = []

    # ---- table shape
    was, is_ = prof["rows"], now["rows"]
    # >= not >: a table loaded twice is exactly ROW_FACTOR, and that is the
    # single most common way a row count goes wrong.
    if was and is_ and (is_ / was >= ROW_FACTOR or was / is_ >= ROW_FACTOR):
        out.append(Finding(
            "drift/row_count", WARNING, None,
            f"row count went from {was:,} to {is_:,} "
            f"({'x%.1f' % (is_ / was) if is_ > was else '/%.1f' % (was / is_)}).",
            "A partial load and a double load both look like this. Check the "
            "source before trusting anything aggregated from it.",
            {"before": f"{was:,}", "after": f"{is_:,}"}))

    for name in old_cols:
        if name not in new_cols:
            out.append(Finding(
                "drift/column_missing", CRITICAL, name,
                "column was in the profile and is not in this table.",
                "Renamed upstream, or dropped. Anything reading it by name is "
                "now getting nothing, usually without complaining.",
                {"dtype_before": old_cols[name].get("dtype")}))
    for name in new_cols:
        if name not in old_cols:
            out.append(Finding(
                "drift/column_added", INFO, name,
                "column is new since the profile was written.",
                "Harmless in itself. Worth knowing the schema moved.",
                {"dtype": new_cols[name].get("dtype")}))

    # ---- per column
    for name, old in old_cols.items():
        new = new_cols.get(name)
        if new is None:
            continue

        if old.get("dtype") != new.get("dtype"):
            out.append(Finding(
                "drift/dtype_changed", CRITICAL, name,
                f"dtype changed from {old['dtype']} to {new['dtype']}.",
                "A numeric column arriving as text usually means one bad value "
                "poisoned the whole column on read.",
                {"before": old["dtype"], "after": new["dtype"]}))

        ob, nb = old.get("null_rate", 0.0), new.get("null_rate", 0.0)
        if nb - ob > NULL_JUMP:
            crit = nb > 0.99 and ob <= 0.99
            out.append(Finding(
                "drift/nulls_rose", CRITICAL if crit else WARNING, name,
                f"missing values went from {_pct(ob)} to {_pct(nb)}"
                + (" -- the column is now entirely empty." if crit else "."),
                "A field that stopped being populated upstream looks exactly "
                "like this, and every rate computed from it silently changes.",
                {"before": _pct(ob), "after": _pct(nb)}))

        if old.get("kind") == "numeric" and new.get("kind") == "numeric":
            out.extend(_numeric_drift(name, old, new))
        elif old.get("kind") == "categorical" and new.get("kind") == "categorical":
            out.extend(_categorical_drift(name, old, new))

    return out


def _numeric_drift(name: str, old: dict, new: dict) -> list[Finding]:
    out = []
    o50, n50 = old.get("p50"), new.get("p50")
    # A ratio of medians only means anything for a column that stays on one side
    # of zero. A column straddling zero has a median that wanders across it, and
    # a move from 0.001 to -0.002 is a ratio of -2, which reads as a hundredfold
    # shrink and fires on ordinary resampling -- 10 false alarms in 30 resamples
    # before this guard.
    #
    # The test is the quartiles, not the standard deviation. Skewed positive data
    # -- capacity, price, revenue -- routinely has a standard deviation larger
    # than its median, and that is exactly the data where unit errors happen.
    q1, q3 = old.get("p25"), old.get("p75")
    straddles_zero = q1 is None or q3 is None or q1 * q3 <= 0
    same_sign = (o50 or 0) * (n50 or 0) > 0
    if o50 and n50 and same_sign and not straddles_zero:
        ratio = n50 / o50
        if ratio > UNIT_FACTOR or ratio < 1 / UNIT_FACTOR:
            out.append(Finding(
                "drift/scale_shift", CRITICAL, name,
                f"typical value moved from {o50:,.4g} to {n50:,.4g}, a factor "
                f"of {ratio if ratio > 1 else 1 / ratio:,.0f}.",
                "A jump this size is a unit change -- megawatts to kilowatts, "
                "dollars to cents -- far more often than it is real movement.",
                {"median_before": f"{o50:,.4g}", "median_after": f"{n50:,.4g}"}))
            return out   # a unit change makes every other numeric test noise

    sd = old.get("std") or 0.0
    om, nm = old.get("mean"), new.get("mean")
    if sd > 0 and om is not None and nm is not None:
        d = abs(nm - om) / sd
        if d > SHIFT_D:
            out.append(Finding(
                "drift/distribution", WARNING, name,
                f"mean moved from {om:,.4g} to {nm:,.4g}, {d:.1f} standard "
                "deviations of the original spread.",
                "Could be the business, could be a broken source. Worth one "
                "look before anything downstream is trusted.",
                {"mean_before": f"{om:,.4g}", "mean_after": f"{nm:,.4g}",
                 "shift_in_sd": f"{d:.1f}"}))

    if (old.get("min"), old.get("max")) != (None, None) and new.get("min") is not None:
        if old.get("min") is not None and new["min"] < old["min"] - abs(old["min"] or 1) * 0.5 \
                and old.get("min", 0) >= 0 > new["min"]:
            out.append(Finding(
                "drift/sign_change", WARNING, name,
                f"column was never negative in the profile (min {old['min']:,.4g}) "
                f"and now reaches {new['min']:,.4g}.",
                "Negative values in a quantity that cannot be negative are "
                "usually a sentinel or a sign error.",
                {"min_before": f"{old['min']:,.4g}", "min_after": f"{new['min']:,.4g}"}))
    return out


def _categorical_drift(name: str, old: dict, new: dict) -> list[Finding]:
    out = []
    o, n = old.get("top", {}), new.get("top", {})
    if not o or not n:
        return out

    od, nd = old.get("distinct"), new.get("distinct")
    collapsed = bool(od and nd and nd == 1 and od > 1)

    gone = [k for k, v in o.items() if v >= CATEGORY_SHARE and k not in n]
    if gone and not collapsed:   # became_constant below already says this
        out.append(Finding(
            "drift/categories_lost", WARNING, name,
            f"{len(gone)} category value(s) present in the profile no longer "
            f"appear: {gone[:5]}",
            "Downstream branches keyed on these are now dead code, and any "
            "group-by has silently lost a row.",
            {"lost": gone[:10]}))

    fresh = [k for k, v in n.items() if v >= CATEGORY_SHARE and k not in o]
    if fresh:
        out.append(Finding(
            "drift/categories_new", WARNING, name,
            f"{len(fresh)} category value(s) not in the profile now appear: "
            f"{fresh[:5]}",
            "New levels reach a model as unknowns and a group-by as an extra "
            "row. Check they are not respellings of something you already have.",
            {"new": fresh[:10]}))

    if collapsed:
        out.append(Finding(
            "drift/became_constant", CRITICAL, name,
            f"column had {od:,} distinct values in the profile and now has one "
            f"({list(n)[0]!r}).",
            "The source stopped varying. A default is being written where a "
            "computed value used to be.",
            {"distinct_before": od, "distinct_after": nd}))
    return out
