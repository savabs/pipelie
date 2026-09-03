"""Streaming a file too large to hold: what stays exact, and what is sampled."""
import numpy as np
import pandas as pd
import pytest

import pipelie
from pipelie.stream import Reservoir, scan


def write(tmp_path, df, name="t.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def frame():
    r = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame({
        "id": [f"A{i % 19_000}" for i in range(n)],       # some ids repeat
        "risk_score": np.arange(1_700_000_000, 1_700_000_000 + n),
        "region": r.choice(["north", "NORTH", "south"], n),
        "cap": r.lognormal(4, 1, n),
    })


# ------------------------------------------------------------------ parity

def test_streamed_matches_in_memory_on_a_small_file(tmp_path, frame):
    """Below the sample size the two paths must agree exactly, or the CLI
    quietly gives different answers to the library."""
    p = write(tmp_path, frame)
    a = sorted(f.fingerprint for f in pipelie.audit(frame, key=["id"]).findings)
    b = sorted(f.fingerprint for f in pipelie.audit_file(p, key=["id"]).findings)
    assert a == b


def test_small_file_is_not_marked_sampled(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), key=["id"])
    assert r.sampled_rows == 0
    assert r.rows == len(frame)


# ------------------------------------------------------------------- exact

def test_duplicates_are_counted_over_every_row(tmp_path, frame):
    """The point of streaming rather than sampling. Two copies of a row will
    almost never both land in a sample, so a sampled duplicate check reports
    'clean' on a table that is half duplicates."""
    doubled = pd.concat([frame, frame], ignore_index=True)
    p = write(tmp_path, doubled)
    r = pipelie.audit_file(p, sample=500)          # sample far smaller than the file
    hit = [f for f in r.findings if f.code == "duplicate_rows/exact"]
    assert hit, "duplicates missed when sampling"
    assert "20,000" in hit[0].evidence["duplicate_rows"]
    assert hit[0].evidence["counted_over"] == "every row"


def test_key_duplicates_exact_under_sampling(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), key=["id"], sample=500)
    hit = [f for f in r.findings if f.code == "duplicate_rows/key_not_unique"]
    assert hit and hit[0].evidence["counted_over"] == "every row"


def test_row_count_is_exact_under_sampling(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), sample=500)
    assert r.rows == len(frame)
    assert r.sampled_rows == 500


def test_missing_key_column_reported(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), key=["nope"])
    assert [f for f in r.findings if f.code == "duplicate_rows/key_missing"]


def test_exact_duplicates_can_be_switched_off(tmp_path, frame):
    p = write(tmp_path, pd.concat([frame, frame], ignore_index=True))
    r = pipelie.audit_file(p, exact_duplicates=False)
    assert not [f for f in r.findings if f.code.startswith("duplicate_rows/")]


# --------------------------------------------------------------- sampling

def test_sampled_checks_still_fire(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), sample=2000)
    codes = {f.code for f in r.findings}
    assert "clock_in_disguise/epoch" in codes
    assert "vocabulary_collisions/punctuation" in codes


def test_report_says_it_sampled(tmp_path, frame):
    r = pipelie.audit_file(write(tmp_path, frame), sample=1000)
    assert r.sampled_rows == 1000
    assert "uniform sample" in str(r)
    assert r.to_dict()["sampled_rows"] == 1000


# -------------------------------------------------------------- reservoir

def test_reservoir_is_uniform_not_the_first_n():
    """Taking the head of a file would make every share an estimate of the
    first chunk. Rows must be drawn from throughout."""
    res = Reservoir(1000, seed=1)
    for start in range(0, 100_000, 10_000):
        res.add(pd.DataFrame({"i": range(start, start + 10_000)}))
    got = res.frame()["i"].to_numpy()
    assert len(got) == 1000
    # a uniform sample of 0..99,999 has mean near 50,000; the head would be ~500
    assert 40_000 < got.mean() < 60_000, got.mean()
    assert got.max() > 90_000, "nothing from the tail of the stream"


def test_reservoir_holds_bounded_memory():
    """Without compaction the parts list grows as size*ln(seen/size)."""
    res = Reservoir(1000, seed=2)
    for _ in range(200):
        res.add(pd.DataFrame({"i": range(10_000)}))
    assert res._held <= 2 * res.size
    assert len(res.frame()) == 1000


def test_reservoir_smaller_than_target():
    res = Reservoir(1000)
    res.add(pd.DataFrame({"i": range(10)}))
    assert len(res.frame()) == 10
