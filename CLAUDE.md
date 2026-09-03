# pipelie — working notes

Finds the bugs that leave a data pipeline reporting green while the numbers are
wrong. Published on PyPI as `pipelie`, source at github.com/savabs/pipelie.

## Setup

```bash
./venv/bin/python -m pytest tests/ -q        # 71 tests, ~2s
./venv/bin/python -m pipelie.cli FILE --key id
```

The venv is Python 3.12 (`/opt/homebrew/bin/python3.12`). Run everything through
`./venv/bin/…`; the system python is 3.9 and will not import the package.

## What it is

Seven checks plus drift, ~1,500 lines. `audit(df)` returns a report,
`guard(df)` raises, `audit_file(path)` streams, the CLI exits 1 on critical.

| module | holds |
|---|---|
| `checks.py` | the seven detectors. Each takes a frame, yields Findings |
| `finding.py` | Finding / Report, rule codes, fingerprints |
| `suppress.py` | baselines and ignore patterns |
| `profile.py` | drift: profile a table, compare a later one |
| `stream.py` | reservoir sampling + exact duplicate counting |
| `api.py` | audit / guard / accept / snapshot / audit_file |

## Design rules — do not relax these without measuring

**Silence is the default.** A checker that fires on clean data gets muted, and a
muted checker is worse than none. Every threshold favours silence. There are
explicit "must stay quiet" tests on clean frames and on 50 resamples of the same
process; keep them passing.

**Every finding carries its evidence** — the counts and correlations behind the
claim, not a prose summary. A finding a user cannot verify is one they learn to
ignore.

**Never quietly check less.** When `audit_file` samples, the report says so.
Counting checks (duplicates, row counts) stay exact over every row because they
cannot be sampled; proportions are sampled because they can.

**Fingerprints exclude numbers.** `code:column`, never counts — a column with
1,351 duplicates today and 1,352 tomorrow is the same finding, or every baseline
breaks on contact with real data.

## Bugs already made here, do not repeat

- `Finding` is constructed **positionally** in `checks.py`. Inserting a field
  before `fix` silently shifts every argument.
- Comparing medians as a ratio to detect unit changes fires constantly on any
  column straddling zero (0.001 → -0.002 reads as a hundredfold shrink): 10
  false alarms in 30 resamples. Guard on whether the **quartiles** straddle
  zero. Guarding on `|median| < std` is worse than the bug — skewed positive
  data like capacity and revenue routinely has std above median, which is
  exactly where unit errors live.
- A doubled load is exactly 2.0x. Row-count thresholds need `>=`, not `>`.
- `np.unique` allocates a second copy. Concatenate once, release the chunks,
  sort in place — 1,165MB → 449MB on 12M rows.
- The relative-spread degeneracy test must apply to **all** numeric columns in a
  bounded range, not only ones whose *name* looks score-like. A column called
  `sim` is as broken as one called `similarity`.

## Releasing

```bash
# bump version in pyproject.toml AND src/pipelie/__init__.py
rm -rf dist && ./venv/bin/python -m build
./venv/bin/python -m twine check dist/*
# install the sdist into a throwaway venv and run it before uploading
./venv/bin/python -m twine upload --repository pypi dist/*
git tag -a vX.Y.Z -m "..." && git push --tags
```

`~/.pypirc` holds both `[testpypi]` and `[pypi]` tokens. A version can never be
re-uploaded to PyPI, so the throwaway-install step is not optional.

## State

0.4.0 published. The three adoption blockers (suppression, drift, streaming) are
done. **It has no users** — that is the live problem, and it is not a code
problem. Next code work, in value order: a hosted/scheduled version that keeps
profile history (the part that is a business), then more checks.
