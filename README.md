# pipelie

**Finds the bugs that leave your pipeline reporting green while the numbers are wrong.**

```bash
pip install pipelie
```

```python
import pipelie
print(pipelie.audit(df, target="built", key=["id"]))
```

Or in CI, where it belongs:

```python
df = pipelie.guard(df, target="built", key=["id"])   # raises on anything critical
```

---

## Why this exists and not dbt tests

Schema tests, null counts and row counts answer **"is the data there?"**

They do not answer **"does the data mean what it is about to be used to mean?"**

Every check in this package corresponds to a bug that shipped to production while
the tests passed. Not hypothetical failure modes — things that actually happened,
were written up, and cost real work:

| the bug | what every existing test said |
|---|---|
| An anomaly ranking sorted by Unix timestamp | schema valid, no nulls, values distinct |
| A cosine similarity that returned 0.9998583 for all 42 rows | numeric, in range, populated |
| One date column holding four different date formats, 84% of values silently coerced to `NaT` | looked like ordinary missing data |
| `Queue ID` used as a primary key when it was not unique in 3 of 4 sources | column present, correct dtype |
| "Solar" and "Photovoltaic" counted as different technologies | valid categories, no nulls |
| Whether a field was missing predicted the outcome being measured | 20% null, within tolerance |

Each of those produced a plausible-looking number that was wrong. That is the
failure mode this package is for.

## What it checks

| check | catches |
|---|---|
| `clock_in_disguise` | A number that is really a timestamp, a row index, or a counter. Epoch-range detection plus rank correlation against every date column. |
| `degenerate` | All-null columns, constants, and near-constants — a "similarity" that cannot separate anything. |
| `parse_carnage` | One column holding more than one date format, with the share of each. This is the bug that destroys most of a column and looks like missingness. |
| `informative_missingness` | Whether a value is missing predicts your target. If so, dropping those rows is selecting on the answer, and every rate you compute afterwards is biased. |
| `vocabulary_collisions` | The same category written two ways — case, punctuation, or abbreviation — splitting one rate into two. |
| `placeholders` | `-999`, `TODO`, `1970-01-01` and friends, still present in the data long after the promise to replace them. |
| `duplicate_rows` | Re-ingested rows that manufacture a "structural break", and declared keys that are not unique. |

`informative_missingness` is Bonferroni-corrected across the columns it tests, so a
wide table does not manufacture a finding.

## A real example

Four US grid operators publish their interconnection queues. Merged into one frame
and audited with no configuration:

```
CRITICAL parse_carnage [Queue Date]
    holds 4 different date formats in one column. Parsed together, one format
    wins and the rest become NaT -- which is indistinguishable from data that
    was simply absent.
    evidence: formats=4  examples={'2025-10-08T00:37:52+00:00': '40%',
              '2003-11-18 08:00:00': '24%', '2008-01-30': '19%',
              '1/14/2025': '18%'}  values=9,640

CRITICAL duplicate_rows [Queue ID]
    declared key is not unique: 1,351 duplicate row(s) across ['Queue ID'].
```

Both are real. The first cost 7,400 of 9,640 dates on an earlier pandas, and the
loss was invisible because it looked like ordinary missing data. The second would
have silently corrupted every change record built on that key.

Finding them by hand took hours. `pipelie` finds them in under a second, and it
found two more date columns with the same defect that had not been checked at all.

## Design

**Silence is the default.** A checker that fires on clean data gets muted, and a
muted checker is worse than none. Thresholds favour silence over noise, anything
uncertain is a `warning` rather than a `critical`, and the test suite contains
explicit "must stay quiet" cases on clean frames.

**Every finding carries its evidence.** Not a prose summary — the counts,
shares and correlations behind the claim, so you can check it rather than trust it.

**Nothing is mutated.** `audit()` reads. What to do about a finding is your call.

## What it does not do

- It does not check orchestration. A DAG that reports success when the write threw,
  a timeout that cannot stop the work, or a test that passes by never running the
  code are all real failures, and none of them are visible in the resulting table.
- It does not know your domain. It cannot tell you a capacity figure is in the
  wrong unit, only that it is oddly constant.
- **Silence is not proof of correctness.** It means these particular lies are absent.

## Background

The failure modes come from [Thirteen Ways a Pipeline Lies](https://savabs.github.io/2026/08/29/thirteen-ways-a-pipeline-lies.html),
a write-up of thirteen documented bugs in one system, each of which reported green.

Apache-2.0.
