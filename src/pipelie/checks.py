"""The detectors.

Every check here exists because the author shipped the bug it looks for, and
every one shares a property: the pipeline reported success while it happened.
Schema tests, null counts and row counts all passed. That is the gap this
fills -- not "is the data present" but "does the data mean what it is about
to be used to mean".

Each check is deliberately conservative. A tool that cries wolf gets muted,
and a muted tool is worse than no tool, so thresholds favour silence over
noise and anything uncertain is reported as a warning rather than a failure.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .finding import CRITICAL, INFO, WARNING, Finding

# Column names whose whole purpose is to carry a magnitude. When one of these
# turns out to be a clock, a constant, or a placeholder, it is not a curiosity
# -- something downstream is ranking or thresholding on it.
MEANINGFUL = re.compile(
    r"score|rank|similarit|distance|prob|conf|weight|anomal|signal|corr|pred|risk",
    re.I,
)

# Epoch seconds for 2001-09-09 to 2033-05-18, and the same window in
# milliseconds. A "score" living in this window is almost always a timestamp.
EPOCH_S = (1_000_000_000, 2_000_000_000)
EPOCH_MS = (1_000_000_000_000, 2_000_000_000_000)

PLACEHOLDER_STRINGS = {
    "todo", "tbd", "n/a", "na", "null", "none", "nan", "missing", "unknown",
    "placeholder", "fixme", "xxx", "test", "dummy", "sample", "foo", "bar",
    "lorem ipsum", "changeme", "example",
}
PLACEHOLDER_NUMBERS = {-1, -999, -9999, 999, 9999, 99999, -99999}
PLACEHOLDER_DATES = {"1970-01-01", "1900-01-01", "1899-12-30", "2000-01-01"}


def _numeric(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _datetimes(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation without a scipy dependency."""
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return float("nan")
    ra, rb = a[m].rank(), b[m].rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ checks

def clock_in_disguise(df: pd.DataFrame, **_) -> Iterator[Finding]:
    """A number that is really a timestamp, or really the row order.

    The original: an anomaly ranking that sorted by Unix timestamp. Every
    score was distinct, the distribution looked plausible, and the ordering
    was chronological rather than anomalous. Nothing in the schema was wrong.
    """
    dates = _datetimes(df)
    for col in _numeric(df):
        s = df[col].dropna()
        if len(s) < 20 or s.nunique() < 5:
            continue
        named = bool(MEANINGFUL.search(str(col)))
        # One column, one finding. Three separate reports of the same clock is
        # the noise that gets a tool muted.
        reported = False

        lo, hi = float(s.min()), float(s.max())
        for unit, (a, b) in (("seconds", EPOCH_S), ("milliseconds", EPOCH_MS)):
            if a <= lo and hi <= b:
                yield Finding(
                    "clock_in_disguise", CRITICAL if named else WARNING, str(col),
                    f"every value falls inside the Unix epoch range for {unit}. "
                    "This column is a timestamp wearing a number's name.",
                    "Convert it to a datetime and find whatever was supposed to "
                    "populate this column instead.",
                    {"min": f"{lo:,.0f}", "max": f"{hi:,.0f}",
                     "as_dates": f"{pd.to_datetime(lo, unit='s' if unit=='seconds' else 'ms')} "
                                 f"to {pd.to_datetime(hi, unit='s' if unit=='seconds' else 'ms')}"},
                )
                reported = True
                break

        for dcol in dates:
            if reported:
                break
            rho = _spearman(df[col], df[dcol].astype("int64", errors="ignore"))
            if not np.isnan(rho) and abs(rho) > 0.95:
                yield Finding(
                    "clock_in_disguise", CRITICAL if named else WARNING, str(col),
                    f"moves almost perfectly with '{dcol}' (rank correlation "
                    f"{rho:+.3f}). Sorting by it sorts by time.",
                    "Check whether the value was ever computed, or whether a "
                    "date leaked into the field.",
                    {"vs": dcol, "spearman": f"{rho:+.3f}"},
                )
                reported = True

        if named and not reported:
            rho_idx = _spearman(s.reset_index(drop=True),
                                pd.Series(np.arange(len(s)), dtype=float))
            if not np.isnan(rho_idx) and abs(rho_idx) > 0.99:
                yield Finding(
                    "clock_in_disguise", WARNING, str(col),
                    f"is monotone in row order (rank correlation {rho_idx:+.3f}). "
                    "It may be an index, an insertion order, or a counter.",
                    "Confirm the value is computed from the data and not from "
                    "the position of the row.",
                    {"spearman_vs_row_order": f"{rho_idx:+.3f}"},
                )


def degenerate(df: pd.DataFrame, **_) -> Iterator[Finding]:
    """A column that cannot vary, in a slot where variation is the point.

    The original: a cosine similarity that returned 0.9998583 for all 42 rows
    because every vector had been built from the same field. It was not
    constant enough to look broken, and it was ranked on anyway.
    """
    n = len(df)
    if n == 0:
        return
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            yield Finding(
                "degenerate", CRITICAL, str(col),
                "is entirely null. Nothing downstream can be using this, or "
                "something downstream is using nulls.",
                "Find the write that was supposed to fill it. A field renamed "
                "upstream fails exactly like this.",
                {"nulls": f"{n:,}/{n:,}"},
            )
            continue

        nun = s.nunique(dropna=True)
        named = bool(MEANINGFUL.search(str(col)))
        if nun == 1:
            yield Finding(
                "degenerate", CRITICAL if named else WARNING, str(col),
                f"has exactly one distinct value ({s.iloc[0]!r}) across "
                f"{len(s):,} rows.",
                "A constant cannot rank, threshold or discriminate. Check the "
                "computation actually receives varying inputs.",
                {"value": repr(s.iloc[0]), "rows": f"{len(s):,}"},
            )
            continue

        top_share = s.value_counts(normalize=True).iloc[0]
        if top_share > 0.99 and len(s) >= 50:
            yield Finding(
                "degenerate", WARNING, str(col),
                f"is {top_share:.1%} a single value "
                f"({s.value_counts().index[0]!r}).",
                "Near-constant columns usually mean a default is being written "
                "where a computed value was intended.",
                {"dominant_share": f"{top_share:.2%}", "distinct": nun},
            )
            continue

        if pd.api.types.is_numeric_dtype(s) and len(s) >= 20:
            spread = float(s.max()) - float(s.min())
            scale = max(abs(float(s.mean())), 1e-12)
            # Restricted to columns living in a bounded range -- similarities,
            # correlations, probabilities. Those are the ones where a spread
            # this small means the measure cannot separate anything. A column
            # with a large offset (a timestamp, a price) can have a tiny
            # relative spread and still be perfectly informative.
            #
            # Applied whatever the column is called. The original bug was in a
            # column named "similarity", but the defect lives in the values,
            # and a column called "sim" or "s2" is exactly as broken.
            bounded = float(s.min()) >= -1.5 and float(s.max()) <= 1.5
            if bounded and spread / scale < 1e-4:
                yield Finding(
                    "degenerate", CRITICAL, str(col),
                    f"varies by {spread:.3g} across {len(s):,} rows, on values "
                    f"around {float(s.mean()):.6g}. It is numerically constant.",
                    "A similarity or score that cannot separate anything is not "
                    "measuring what its name claims.",
                    {"min": f"{float(s.min()):.9g}", "max": f"{float(s.max()):.9g}",
                     "relative_spread": f"{spread / scale:.2e}"},
                )


def _shape(v: str) -> str:
    """Collapse a value to its format skeleton: digits to D, keep separators."""
    return re.sub(r"\d+", "D", v.strip())


def parse_carnage(df: pd.DataFrame, **_) -> Iterator[Finding]:
    """One date column, more than one date format.

    The original: four sources wrote dates four ways in a single column --
    ISO dates, ISO timestamps with a timezone, space-separated timestamps, and
    US M/D/Y. On pandas 2.x that inferred one format from the first value and
    coerced the rest to NaT: 7,400 of 9,640 values destroyed, 84%, looking
    exactly like ordinary missingness. On pandas 3.x the same column raises
    instead.

    So this does not test how the installed pandas happens to behave. It reads
    the formats present and reports heterogeneity directly, which is the actual
    hazard whichever library sees it next.
    """
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        present = s.dropna().astype(str).str.strip()
        present = present[present != ""]
        if len(present) < 30:
            continue

        datey = present.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True)
        if datey.mean() < 0.5:
            continue

        shapes = present[datey].map(_shape).value_counts(normalize=True)
        real = shapes[shapes >= 0.01]
        if len(real) > 1:
            ex = {}
            for shp in real.index[:4]:
                sample = present[datey][present[datey].map(_shape) == shp].iloc[0]
                ex[sample] = f"{real[shp]:.0%}"
            yield Finding(
                "parse_carnage", CRITICAL, str(col),
                f"holds {len(real)} different date formats in one column. "
                "Parsed together, one format wins and the rest become NaT -- "
                "which is indistinguishable from data that was simply absent.",
                'Parse per source with format="mixed", then assert the survival '
                "rate instead of trusting it.",
                {"formats": len(real), "examples": ex, "values": f"{int(datey.sum()):,}"},
            )
            continue

        # Single format, but check the library still keeps the column.
        try:
            kept = int(pd.to_datetime(present[datey], errors="coerce",
                                      format="mixed", utc=True).notna().sum())
        except Exception:
            kept = 0
        total = int(datey.sum())
        if total and kept / total < 0.98:
            yield Finding(
                "parse_carnage", CRITICAL, str(col),
                f"looks like dates but only {kept:,} of {total:,} values parse "
                f"({kept / total:.1%}).",
                "Find out what the unparsed values actually are before dropping "
                "them; they are rarely a random sample.",
                {"parsed": f"{kept:,}", "of": f"{total:,}"},
            )


def informative_missingness(df: pd.DataFrame, target: str | None = None,
                            **_) -> Iterator[Finding]:
    """Whether a value is missing predicts the answer.

    The original: this is the check that caught the parse bug above. If rows
    with a missing date fail at a different rate to rows with one, then
    dropping them is not cleaning -- it is selecting on the outcome, and every
    rate computed afterwards is biased.
    """
    if target is None or target not in df.columns:
        return
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) != 2:
        return
    if not pd.api.types.is_numeric_dtype(y):
        y = (y == y.dropna().unique()[0]).astype(float)
    y = pd.to_numeric(y, errors="coerce")

    candidates = [c for c in df.columns if c != target]
    tested = [c for c in candidates
              if 0.01 < df[c].isna().mean() < 0.99 and df[c].isna().sum() >= 20]
    if not tested:
        return
    # Bonferroni across the columns actually tested, so a wide table does not
    # manufacture a finding.
    alpha = 0.01 / len(tested)
    z_crit = float(np.sqrt(2) * _erfinv(1 - alpha))

    for col in tested:
        miss = df[col].isna()
        a, b = y[miss].dropna(), y[~miss].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        pa, pb = a.mean(), b.mean()
        na, nb = len(a), len(b)
        pooled = (a.sum() + b.sum()) / (na + nb)
        se = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb)) if 0 < pooled < 1 else np.nan
        if not se or np.isnan(se) or se == 0:
            continue
        z = (pa - pb) / se
        if abs(z) > z_crit:
            yield Finding(
                "informative_missingness", CRITICAL, str(col),
                f"whether this is missing predicts '{target}'. Rows where it is "
                f"missing average {pa:.3f}; rows where it is present average "
                f"{pb:.3f}.",
                "Do not drop these rows and quote a rate. Either model the "
                "missingness or restrict to a population where it is absent.",
                {"missing_rate": f"{miss.mean():.1%}", f"{target}|missing": f"{pa:.3f}",
                 f"{target}|present": f"{pb:.3f}", "z": f"{z:+.1f}"},
            )


def _erfinv(x: float) -> float:
    """Inverse error function, Giles' rational approximation. Avoids scipy."""
    w = -np.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                  -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
            p = p * w + c
    else:
        w = np.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
                  -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
            p = p * w + c
    return float(p * x)


def vocabulary_collisions(df: pd.DataFrame, max_categories: int = 200,
                          **_) -> Iterator[Finding]:
    """Two spellings of one thing, counted as two things.

    The original: "Solar" and "Photovoltaic" were separate categories, as were
    "Wind" and "Wnd". The apparent gap between technologies was measuring the
    vocabulary of the source, not the physics.
    """
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if vals.empty:
            continue
        uniq = vals.unique()
        if not 2 <= len(uniq) <= max_categories:
            continue

        groups: dict[str, list[str]] = {}
        for v in uniq:
            key = re.sub(r"[^a-z0-9]", "", v.casefold())
            if key:
                groups.setdefault(key, []).append(v)
        collisions = {k: v for k, v in groups.items() if len(v) > 1}
        if collisions:
            shown = list(collisions.values())[:5]
            yield Finding(
                "vocabulary_collisions", WARNING, str(col),
                f"{len(collisions)} label(s) differ only by case, spacing or "
                f"punctuation: {shown}",
                "Normalise before grouping, or every rate you compute by this "
                "column is split across spellings.",
                {"colliding_groups": len(collisions), "distinct_labels": len(uniq)},
            )

        # Consonant skeletons catch abbreviations the casefold pass cannot:
        # "Wnd" and "Wind" both reduce to "wnd".
        skel: dict[str, list[str]] = {}
        for v in uniq:
            k = re.sub(r"[aeiou\W_]", "", v.casefold())
            if len(k) >= 3:
                # Keyed on the first letter too. Without it "MADERA" and
                # "AMADOR" both reduce to "mdr" and get reported as the same
                # county, which is the kind of false positive that teaches a
                # user to stop reading the output.
                skel.setdefault(f"{v.casefold()[:1]}:{k}", []).append(v)
        # Anything the punctuation pass already reported is not reported again.
        seen = {x for grp in collisions.values() for x in grp}
        abbrev = {k: v for k, v in skel.items()
                  if len(v) > 1 and len({x.casefold() for x in v}) > 1
                  and not set(v) <= seen}
        if abbrev:
            shown = list(abbrev.values())[:5]
            yield Finding(
                "vocabulary_collisions", WARNING, str(col),
                f"{len(abbrev)} group(s) look like abbreviations of each other: "
                f"{shown}",
                "Check whether these are the same category written two ways "
                "before treating them as separate.",
                {"suspect_groups": len(abbrev)},
            )


def placeholders(df: pd.DataFrame, **_) -> Iterator[Finding]:
    """Defaults that outlived the promise to replace them."""
    n = len(df)
    if n == 0:
        return
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        named = bool(MEANINGFUL.search(str(col)))

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            low = s.astype(str).str.strip().str.casefold()
            hits = low.isin(PLACEHOLDER_STRINGS)
            if hits.any() and hits.mean() > 0.01:
                top = low[hits].value_counts().head(3).to_dict()
                yield Finding(
                    "placeholders", WARNING, str(col),
                    f"{hits.mean():.1%} of values are placeholder text.",
                    "These are not categories. Decide whether they are missing "
                    "data before they end up as a level in a model.",
                    {"share": f"{hits.mean():.1%}", "examples": top},
                )
        elif pd.api.types.is_numeric_dtype(s):
            for sentinel in PLACEHOLDER_NUMBERS:
                share = float((s == sentinel).mean())
                if share > 0.01:
                    yield Finding(
                        "placeholders", CRITICAL if named else WARNING, str(col),
                        f"{share:.1%} of values are exactly {sentinel}, a "
                        "conventional 'no data' marker.",
                        "Convert to null. Averaged or ranked as a number, this "
                        "silently drags every statistic it touches.",
                        {"sentinel": sentinel, "share": f"{share:.1%}",
                         "rows": f"{int((s == sentinel).sum()):,}"},
                    )
        elif pd.api.types.is_datetime64_any_dtype(s):
            iso = s.dt.strftime("%Y-%m-%d")
            hits = iso.isin(PLACEHOLDER_DATES)
            if hits.any() and hits.mean() > 0.005:
                yield Finding(
                    "placeholders", WARNING, str(col),
                    f"{hits.mean():.1%} of dates sit on an epoch or sentinel "
                    f"value ({sorted(set(iso[hits]))[:3]}).",
                    "A zero timestamp is missing data that sorts as very old.",
                    {"share": f"{hits.mean():.1%}"},
                )


def duplicate_rows(df: pd.DataFrame, key: Iterable[str] | None = None,
                   **_) -> Iterator[Finding]:
    """Re-ingested rows, and keys that are not keys.

    Two originals. Duplicate ingestion manufactured a "structural break this
    week" that was the same week counted twice. And Queue ID was not unique in
    three of four sources -- one operator reused "0031" for a project that
    completed and one that withdrew.
    """
    n = len(df)
    if n == 0:
        return

    dup = int(df.duplicated().sum())
    if dup:
        yield Finding(
            "duplicate_rows", CRITICAL if dup / n > 0.01 else WARNING, None,
            f"{dup:,} of {n:,} rows ({dup / n:.1%}) are exact duplicates.",
            "Deduplicate before aggregating. Repeated ingestion of one period "
            "shows up downstream as a real-looking jump in volume.",
            {"duplicate_rows": f"{dup:,}", "share": f"{dup / n:.1%}"},
        )

    if key:
        cols = [c for c in key if c in df.columns]
        missing = [c for c in key if c not in df.columns]
        if missing:
            yield Finding(
                "duplicate_rows", CRITICAL, None,
                f"declared key column(s) not present: {missing}",
                "A key that does not exist silently becomes no key at all.",
                {"missing": missing},
            )
        if cols:
            d = int(df.duplicated(subset=cols).sum())
            if d:
                ex = (df[df.duplicated(subset=cols, keep=False)]
                      .groupby(cols, dropna=False).size().sort_values(ascending=False)
                      .head(3).to_dict())
                yield Finding(
                    "duplicate_rows", CRITICAL, ", ".join(cols),
                    f"declared key is not unique: {d:,} duplicate row(s) across "
                    f"{cols}.",
                    "Widen the key until it is unique, and report the rows that "
                    "still are not rather than dropping them.",
                    {"duplicates": f"{d:,}", "worst": {str(k): v for k, v in ex.items()}},
                )


ALL_CHECKS = (
    clock_in_disguise,
    degenerate,
    parse_carnage,
    informative_missingness,
    vocabulary_collisions,
    placeholders,
    duplicate_rows,
)
