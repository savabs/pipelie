"""Drift: has this table changed since the one my code was written against?

The silence tests carry most of the weight. A monitor that fires on ordinary
resampling gets muted within a week, and a muted monitor is not a monitor.
"""
import json

import numpy as np
import pandas as pd
import pytest

import pipelie
from pipelie.profile import drift, load_profile, profile, save_profile


def table(seed=0, n=4000, mult=1.0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": range(n),
        "capacity_mw": r.lognormal(4, 1, n) * mult,     # skewed, strictly positive
        "score": r.beta(2, 5, n),
        "centred": np.where(r.random(n) < 0.12, np.nan, r.normal(size=n)),
        "region": r.choice(["north", "south", "east", "west"], n, p=[.4, .3, .2, .1]),
    })


@pytest.fixture
def base():
    return profile(table(0))


# ------------------------------------------------------------------ silence

def test_identical_table_is_silent(base):
    assert drift(table(0), base) == []


def test_resampling_does_not_fire(base):
    """50 independent draws from the same process, no findings.

    This exists because an earlier version compared medians as a ratio, which
    is meaningless for a column straddling zero: 0.001 -> -0.002 is a ratio of
    -2 and read as a hundredfold shrink. It fired on 10 of 30 resamples.
    """
    noisy = [s for s in range(1, 51) if drift(table(s), base)]
    assert not noisy, f"false positives on seeds {noisy}"


def test_modest_growth_is_silent(base):
    assert drift(table(9, n=4800), base) == []


# ----------------------------------------------------------- structural drift

def test_column_removed(base):
    f = drift(table(1).drop(columns=["region"]), base)
    assert [x.code for x in f] == ["drift/column_missing"]
    assert f[0].severity == pipelie.CRITICAL


def test_column_added_is_only_information(base):
    f = [x for x in drift(table(1).assign(extra=1), base)
         if x.code == "drift/column_added"]
    assert f and f[0].severity == pipelie.INFO


def test_dtype_flip(base):
    f = [x for x in drift(table(1).assign(id=lambda d: d["id"].astype(str)), base)
         if x.code == "drift/dtype_changed"]
    assert f and f[0].severity == pipelie.CRITICAL


def test_field_stops_being_populated(base):
    t = table(1)
    t.loc[t.sample(frac=0.5, random_state=0).index, "score"] = np.nan
    f = [x for x in drift(t, base) if x.code == "drift/nulls_rose"]
    assert f


def test_column_emptied_entirely_is_critical(base):
    f = [x for x in drift(table(1).assign(score=np.nan), base)
         if x.code == "drift/nulls_rose"]
    assert f and f[0].severity == pipelie.CRITICAL


def test_doubled_load(base):
    t = table(1)
    f = [x for x in drift(pd.concat([t, t], ignore_index=True), base)
         if x.code == "drift/row_count"]
    assert f, "a table loaded twice is exactly 2x and must still be caught"


# ------------------------------------------------------------- value drift

@pytest.mark.parametrize("mult", [1000.0, 0.001])
def test_unit_change_both_directions(base, mult):
    f = [x for x in drift(table(1, mult=mult), base) if x.code == "drift/scale_shift"]
    assert f and f[0].severity == pipelie.CRITICAL


def test_skewed_positive_column_is_not_exempt(base):
    """capacity_mw has a standard deviation larger than its median, like most
    money and capacity data. An earlier guard used |median| < std to detect
    zero-centred columns and so skipped exactly the data unit errors happen in."""
    assert [x for x in drift(table(1, mult=100), base)
            if x.code == "drift/scale_shift"]


def test_new_and_lost_categories(base):
    t = table(1)
    t["region"] = t["region"].replace("west", "pacific")
    codes = {x.code for x in drift(t, base)}
    assert "drift/categories_new" in codes
    assert "drift/categories_lost" in codes


def test_collapse_reported_once(base):
    """A column that goes constant should say so once, not also report every
    category it lost."""
    codes = [x.code for x in drift(table(1).assign(region="north"), base)]
    assert "drift/became_constant" in codes
    assert "drift/categories_lost" not in codes


# --------------------------------------------------------------- the profile

def test_profile_round_trips(tmp_path, base):
    p = tmp_path / "prof.json"
    save_profile(base, p)
    assert load_profile(p) == base


def test_profile_holds_no_rows(base):
    """It is a summary, not a copy. Nothing row-level may be in it."""
    text = json.dumps(base)
    assert len(text) < 8000, "profile should stay small regardless of table size"
    assert "id" in base["columns"]
    assert "top" not in base["columns"]["id"] or len(base["columns"]["id"].get("top", {})) <= 20


def test_missing_profile_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no profile at"):
        load_profile(tmp_path / "absent.json")


def test_snapshot_and_audit_together(tmp_path):
    p = tmp_path / "prof.json"
    pipelie.snapshot(table(0), p)
    r = pipelie.audit(table(1).drop(columns=["region"]), profile_path=p)
    assert "drift" in r.checks_run
    assert not r.ok


def test_drift_findings_respect_baseline(tmp_path):
    p = tmp_path / "prof.json"
    pipelie.snapshot(table(0), p)
    changed = table(1).drop(columns=["region"])
    fps = pipelie.audit(changed, profile_path=p).fingerprints
    assert pipelie.audit(changed, profile_path=p, baseline=fps).findings == []
