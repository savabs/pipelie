"""Adoption behaviour: accept what exists, fail on what is new."""
import json

import numpy as np
import pandas as pd
import pytest

import pipelie
from pipelie.suppress import matches, read_baseline, write_baseline


@pytest.fixture
def day1():
    """A table with one known problem: the id column is not unique."""
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({"id": [f"A{i % 550}" for i in range(n)],
                         "v": rng.normal(size=n)})


@pytest.fixture
def day2(day1):
    """The same table, plus a timestamp shipped as a risk score."""
    return day1.assign(risk_score=np.arange(1_700_000_000, 1_700_000_000 + len(day1)))


# ------------------------------------------------------------ fingerprints

def test_fingerprint_ignores_counts(day1):
    """A finding must keep its identity when the number of bad rows drifts,
    or every baseline breaks the moment the data changes slightly."""
    def key_finding(df):
        hits = [f for f in pipelie.audit(df, key=["id"]).findings
                if f.code == "duplicate_rows/key_not_unique"]
        assert hits, "expected the key finding"
        return hits[0]

    a = key_finding(day1)
    # More rows sharing an id: the same defect, a bigger number.
    b = key_finding(pd.concat([day1, day1.head(50)], ignore_index=True))
    assert a.evidence != b.evidence          # the counts did change
    assert a.fingerprint == b.fingerprint    # the identity did not


def test_code_carries_the_family(day2):
    f = [x for x in pipelie.audit(day2).findings if x.column == "risk_score"][0]
    assert f.code == "clock_in_disguise/epoch"
    assert f.check == "clock_in_disguise"


# ---------------------------------------------------------------- patterns

@pytest.mark.parametrize("pattern,expected", [
    ("duplicate_rows/key_not_unique:id", True),
    ("duplicate_rows/*", True),
    ("duplicate_rows", True),
    ("*:id", True),
    ("parse_carnage/*", False),
    ("*:other_column", False),
])
def test_ignore_patterns(pattern, expected):
    assert matches("duplicate_rows/key_not_unique:id", pattern) is expected


def test_ignore_suppresses(day1):
    before = pipelie.audit(day1, key=["id"])
    after = pipelie.audit(day1, key=["id"], ignore=["duplicate_rows/*"])
    assert before.findings and not after.findings
    assert after.suppressed == len(before.findings)


# ---------------------------------------------------------------- baseline

def test_accept_then_clean(day1, tmp_path):
    p = tmp_path / "base.json"
    n = pipelie.accept(day1, p, key=["id"])
    assert n == 1
    assert pipelie.audit(day1, key=["id"], baseline=p).findings == []


def test_new_problem_still_fails_after_baseline(day1, day2, tmp_path):
    """The whole point. Accepting old debt must not hide tomorrow's bug."""
    p = tmp_path / "base.json"
    pipelie.accept(day1, p, key=["id"])
    r = pipelie.audit(day2, key=["id"], baseline=p)
    assert not r.ok, "a new critical finding was wrongly suppressed"
    assert [f.code for f in r.findings] == ["clock_in_disguise/epoch"]
    assert r.suppressed == 1


def test_missing_baseline_file_is_not_an_error(day1, tmp_path):
    r = pipelie.audit(day1, key=["id"], baseline=tmp_path / "nope.json")
    assert r.findings


def test_baseline_file_is_readable_and_editable(day1, tmp_path):
    p = tmp_path / "base.json"
    pipelie.accept(day1, p, key=["id"])
    data = json.loads(p.read_text())
    assert set(data) == {"note", "accepted", "detail"}
    assert data["accepted"] == ["duplicate_rows/key_not_unique:id"]
    # deleting a line starts failing on it again
    p.write_text(json.dumps({"accepted": []}))
    assert pipelie.audit(day1, key=["id"], baseline=p).findings


def test_baseline_accepts_fingerprints_directly(day1):
    fps = pipelie.audit(day1, key=["id"]).fingerprints
    assert pipelie.audit(day1, key=["id"], baseline=fps).findings == []


# -------------------------------------------------------------------- json

def test_report_serialises(day2):
    d = pipelie.audit(day2, key=["id"]).to_dict()
    assert json.loads(json.dumps(d, default=str))
    assert {"rows", "columns", "ok", "suppressed", "findings"} <= set(d)
    assert all({"code", "fingerprint", "severity"} <= set(f) for f in d["findings"])
