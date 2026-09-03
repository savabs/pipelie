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

## What it looks like

`examples/a_wrong_number.py` builds a customer table with five ordinary bugs in
it, none of which any schema test can see. Run it and the first thing printed is
the reassuring part: no nulls, correct dtypes, plausible ranges. Everything
passes. That is the moment the number gets shipped.

Then:

```
CRITICAL clock_in_disguise [churn_risk]
    every value falls inside the Unix epoch range for seconds.
CRITICAL informative_missingness [support_note]
    whether this is missing predicts 'churned'.
    evidence: churned|missing=0.000  churned|present=1.000  z=-65.6
CRITICAL duplicate_rows
    300 of 4,300 rows (7.0%) are exact duplicates.
CRITICAL parse_carnage [signup_date]
    holds 2 different date formats in one column.
WARNING  placeholders [credit_limit]
    12.1% of values are exactly -999, a conventional 'no data' marker.
```

What each one costs:

- The "highest-risk customers, call these first" list is the five most recent
  signups. The risk score was never computed.
- Drop the rows with a missing support note -- a standard cleaning step -- and
  the churn rate goes from 50% to 100%.
- The reporting job ran twice on Tuesday, so every total is 7% too high.
- Average credit limit was reported as 4,275. Every real limit is 5,000.
- Half the signup dates would silently become missing.

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

## Files bigger than memory

```bash
pipelie huge.csv --key id          # streams; no flag needed
```

```python
pipelie.audit_file("huge.csv", key=["id"])
```

12 million rows, a 352MB CSV: **4 seconds, 449MB peak.** Memory is bounded by
the sample, not the file.

The split matters and the report states it:

> `streamed: duplicates and row counts exact over all 12,000,000 rows;`
> `every other check on a uniform sample of 200,000`

**Proportions are sampled.** Null rates, category shares, date-format shares. A
200,000-row sample pins a 10% rate to within about a tenth of a percentage
point, and no threshold here is remotely that tight.

**Counting is not.** Duplicates cannot be sampled: draw 200,000 rows from 50
million and two copies of the same row will almost never both appear. A sampled
duplicate check would report "clean" on a table that is half duplicates -- the
exact reporting-green-while-wrong failure this package exists to stop. So every
row is hashed and duplicates are counted over all of them.

That is the only part costing memory per row: 8 bytes, or 16 with a key. 100
million rows is about 1.6GB. Pass `--no-exact-duplicates` to trade those two
checks for constant memory.

The sample is a proper reservoir, not the first N rows -- otherwise every share
would describe the head of the file rather than the file. Below the sample size
nothing is sampled at all, and `audit_file` returns exactly what `audit` does on
the same data. There is a test asserting that.

## Catching a table that CHANGED

The checks above ask whether a table is wrong on its own terms. A running
pipeline has a different question: **is this table different from the one my
code was written against?**

That is where most production breakage actually lives. Nobody ships against a
broken table -- they ship against a working one, and then something upstream
moves. A field stops being populated. An integer column arrives as text because
one row had a comma in it. Somebody switches megawatts to kilowatts. A category
every downstream branch expects disappears. All of it keeps the schema valid and
the row count healthy.

Record what the table looks like today:

```bash
pipelie data.csv --snapshot          # writes pipelie-profile.json
```

Then on any later run:

```bash
pipelie data.csv --profile           # what changed?
```

```
CRITICAL drift/column_missing [region]
    column was in the profile and is not in this table.
CRITICAL drift/scale_shift [capacity_mw]
    typical value moved from 52.8 to 5.28e+04, a factor of 1,000.
    fix: A jump this size is a unit change -- megawatts to kilowatts, dollars
         to cents -- far more often than it is real movement.
```

| it catches | severity |
|---|---|
| a column vanished or was renamed | critical |
| a dtype flipped (numbers arriving as text) | critical |
| a field stopped being populated | critical / warning |
| a unit changed | critical |
| a column stopped varying | critical |
| the row count doubled or halved | warning |
| category values appeared or disappeared | warning |
| the distribution moved by more than a standard deviation | warning |

From Python:

```python
pipelie.snapshot(df, "profile.json")             # once
pipelie.audit(df, profile_path="profile.json")   # later
```

The profile is a small JSON summary -- dtypes, null rates, quantiles, the top
category shares -- and never the data itself. It stays a few kilobytes whether
the table has a thousand rows or a billion.

**It is tuned to stay quiet.** Fifty independent resamples of the same process
produce zero findings, and that is a test in the suite rather than a claim.
Structural changes are critical because they break code; distributional moves
are warnings, because usually they are the business rather than a bug.

## Adopting it on a table that already has problems

Every table has problems. A checker that fails your build on day one over
things that predate it gets deleted, so there are two ways to say "not this
one".

**Accept today's findings as debt.** Run once:

```bash
pipelie data.csv --key id --accept
```

That writes `pipelie-baseline.json`. From then on:

```bash
pipelie data.csv --key id --baseline
```

reports only what is **new**. Old problems stay suppressed, tomorrow's bug
still fails the build. Delete a line from the baseline file to start failing on
it again.

**Or ignore a rule permanently**, when you have decided it is not a problem
here:

```bash
pipelie data.csv --ignore 'duplicate_rows/*' --ignore '*:legacy_id'
```

Patterns are globs over `code:column`, so `parse_carnage/*`, `*:Queue ID` and
`duplicate_rows/key_not_unique:id` all work. The same arguments exist on
`audit()`:

```python
pipelie.audit(df, key=["id"], ignore=["duplicate_rows/*"], baseline="pipelie-baseline.json")
pipelie.accept(df, key=["id"])        # write the baseline from Python
```

Findings are identified by **rule code and column, never by counts** -- a
column with 1,351 duplicates today and 1,352 tomorrow is the same finding, so a
baseline does not fall apart the moment the data moves.

## Machine-readable output

```bash
pipelie data.csv --json
```

Emits the full report -- rows, columns, checks run, and every finding with its
`code`, `fingerprint`, `severity`, `message`, `fix` and `evidence`. Exit code is
1 when anything critical survives suppression, 0 otherwise, so it drops into CI
unchanged.

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
