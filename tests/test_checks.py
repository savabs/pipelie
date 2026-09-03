"""Each test plants one real bug, or plants none and demands silence.

The silence tests matter most. A checker that fires on clean data gets muted,
and a muted checker is worse than none.
"""
import numpy as np
import pandas as pd
import pytest

import pipelie
from pipelie.checks import (clock_in_disguise, degenerate, duplicate_rows,
                            informative_missingness, parse_carnage,
                            placeholders, vocabulary_collisions)


def names(findings):
    return {(f.check, f.column) for f in findings}


@pytest.fixture
def clean():
    """A frame with nothing wrong. Every check must stay silent on it."""
    rng = np.random.default_rng(7)
    n = 800
    return pd.DataFrame({
        "id": [f"P{i:05d}" for i in range(n)],
        "score": rng.beta(2, 5, n),
        "capacity_mw": rng.lognormal(4, 1, n).round(1),
        "price": rng.normal(1_000_000, 40_000, n),      # big offset, small spread
        "fuel": rng.choice(["solar", "wind", "gas", "storage"], n),
        "county": rng.choice([f"C{i}" for i in range(30)], n),
        "queue_date": pd.to_datetime("2015-01-01") + pd.to_timedelta(
            rng.integers(0, 3000, n), unit="D"),
        "built": rng.integers(0, 2, n),
    })


def test_clean_frame_is_silent(clean):
    r = pipelie.audit(clean, target="built", key=["id"])
    assert r.findings == [], "\n".join(str(f) for f in r.findings)
    assert r.ok


def test_clean_frame_silent_without_target_or_key(clean):
    assert pipelie.audit(clean).findings == []


# ---------------------------------------------------------------- each lie

def test_epoch_seconds_masquerading_as_a_score(clean):
    df = clean.assign(anomaly_score=np.arange(1_700_000_000, 1_700_000_000 + len(clean)))
    f = list(clock_in_disguise(df))
    assert ("clock_in_disguise", "anomaly_score") in names(f)
    assert f[0].severity == pipelie.CRITICAL


def test_score_that_tracks_a_date_column(clean):
    df = clean.copy()
    df["risk_score"] = df["queue_date"].rank() + np.random.default_rng(0).normal(0, .01, len(df))
    assert ("clock_in_disguise", "risk_score") in names(list(clock_in_disguise(df)))


def test_a_clock_is_reported_once_not_three_times(clean):
    df = clean.assign(anomaly_score=np.arange(1_700_000_000, 1_700_000_000 + len(clean)))
    hits = [f for f in clock_in_disguise(df) if f.column == "anomaly_score"]
    assert len(hits) == 1, [f.message for f in hits]


def test_similarity_that_cannot_vary(clean):
    df = clean.assign(similarity=0.9998583 + np.random.default_rng(0).normal(0, 1e-9, len(clean)))
    hit = [f for f in degenerate(df) if f.column == "similarity"]
    assert hit and hit[0].severity == pipelie.CRITICAL


def test_degenerate_bounded_column_caught_whatever_it_is_called(clean):
    """The defect is in the values. A column named "sim" is as broken as one
    named "similarity", and naming must not be what saves it."""
    df = clean.assign(sim=0.9998 + np.random.default_rng(0).normal(0, 1e-9, len(clean)))
    assert [f for f in degenerate(df) if f.column == "sim"]


def test_large_magnitude_column_is_not_called_constant(clean):
    """A price around 1e6 has a tiny relative spread and is perfectly fine."""
    assert not [f for f in degenerate(clean) if f.column == "price"]


def test_all_null_column(clean):
    df = clean.assign(promised_field=np.nan)
    hit = [f for f in degenerate(df) if f.column == "promised_field"]
    assert hit and hit[0].severity == pipelie.CRITICAL


def test_mixed_date_formats_destroy_the_column(clean):
    """The bug that cost 7,400 of 9,640 dates: two sources, two formats."""
    half = len(clean) // 2
    raw = ["2021-05-06"] * half + ["5/6/2021"] * (len(clean) - half)
    df = clean.assign(filed=raw)
    hit = [f for f in parse_carnage(df) if f.column == "filed"]
    assert hit, "mixed formats not detected"
    assert hit[0].severity == pipelie.CRITICAL
    assert hit[0].evidence["formats"] == 2


def test_four_formats_as_seen_in_production(clean):
    """The exact shape of the real column: ISO, ISO+tz, spaced, and US M/D/Y."""
    n = len(clean)
    q = n // 4
    raw = (["2008-01-30"] * q + ["2003-11-18 08:00:00"] * q
           + ["2025-10-08T00:37:52+00:00"] * q + ["1/14/2025"] * (n - 3 * q))
    hit = [f for f in parse_carnage(clean.assign(filed=raw)) if f.column == "filed"]
    assert hit and hit[0].evidence["formats"] == 4


def test_two_digit_and_four_digit_days_are_one_format(clean):
    """'1/4/2025' and '12/14/2025' are the same format, not two."""
    n = len(clean)
    raw = ["1/4/2025"] * (n // 2) + ["12/14/2025"] * (n - n // 2)
    assert not [f for f in parse_carnage(clean.assign(filed=raw)) if f.column == "filed"]


def test_single_date_format_is_silent(clean):
    df = clean.assign(filed=["2021-05-06"] * len(clean))
    assert not [f for f in parse_carnage(df) if f.column == "filed"]


def test_missingness_that_predicts_the_outcome(clean):
    df = clean.copy()
    df["completion_date"] = np.where(df["built"] == 1, "2022-01-01", None)
    hit = [f for f in informative_missingness(df, target="built")]
    assert ("informative_missingness", "completion_date") in names(hit)


def test_missingness_unrelated_to_outcome_is_silent(clean):
    rng = np.random.default_rng(3)
    df = clean.copy()
    df.loc[rng.random(len(df)) < 0.2, "capacity_mw"] = np.nan
    assert not list(informative_missingness(df, target="built"))


def test_vocabulary_split_across_spellings(clean):
    df = clean.copy()
    df.loc[:100, "fuel"] = "Solar"
    df.loc[101:200, "fuel"] = "solar "
    hit = [f for f in vocabulary_collisions(df) if f.column == "fuel"]
    assert hit


def test_abbreviation_collision(clean):
    df = clean.copy()
    df.loc[:100, "fuel"] = "Wnd"
    hit = [f for f in vocabulary_collisions(df) if f.column == "fuel"]
    assert any("abbrev" in f.message for f in hit)


def test_numeric_sentinel(clean):
    df = clean.copy()
    df.loc[:100, "capacity_mw"] = -999
    hit = [f for f in placeholders(df) if f.column == "capacity_mw"]
    assert hit and "-999" in str(hit[0].evidence)


def test_placeholder_text(clean):
    df = clean.copy()
    df.loc[:200, "county"] = "TODO"
    assert [f for f in placeholders(df) if f.column == "county"]


def test_declared_key_is_not_unique(clean):
    df = pd.concat([clean, clean.head(20)], ignore_index=True)
    hit = [f for f in duplicate_rows(df, key=["id"])]
    assert any(f.severity == pipelie.CRITICAL for f in hit)


def test_missing_key_column_is_reported(clean):
    hit = list(duplicate_rows(clean, key=["nonexistent"]))
    assert hit and hit[0].severity == pipelie.CRITICAL


# ---------------------------------------------------------------- surface

def test_guard_raises_on_critical(clean):
    df = clean.assign(anomaly_score=np.arange(1_700_000_000, 1_700_000_000 + len(clean)))
    with pytest.raises(pipelie.PipelineLied):
        pipelie.guard(df)


def test_guard_returns_frame_when_clean(clean):
    assert pipelie.guard(clean, target="built", key=["id"]) is clean


def test_audit_does_not_mutate(clean):
    before = clean.copy()
    pipelie.audit(clean, target="built", key=["id"])
    pd.testing.assert_frame_equal(clean, before)


def test_empty_frame_does_not_crash():
    r = pipelie.audit(pd.DataFrame())
    assert r.rows == 0
