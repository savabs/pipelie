"""Auditing a file that does not fit in memory.

Two kinds of check live in this package, and they scale differently.

Proportions -- null rates, category shares, date-format shares, sentinel
shares -- are estimated perfectly well from a uniform sample. A 200,000-row
sample pins a 10% rate to within about a tenth of a percentage point, and no
threshold here is remotely that tight.

Counting is different. Duplicates cannot be sampled: take 200,000 rows out of
50 million and two copies of the same row will almost never both be drawn. A
sampled duplicate check would report "no duplicates" on a table that is half
duplicates, which is precisely the reporting-green-while-wrong failure this
package exists to stop.

So the file is streamed once. Row counts and duplicates are computed exactly
over every row via 64-bit hashes -- 8 bytes a row, so 50 million rows costs
400MB of numpy array rather than an unbounded Python set. Everything else runs
on a reservoir sample, which is a genuine uniform sample of the whole file
rather than the first N rows.

The report says which it did. A tool that quietly checks less on big inputs is
worse than one that refuses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .finding import CRITICAL, WARNING, Finding

DEFAULT_CHUNK = 250_000
DEFAULT_SAMPLE = 200_000


def _reader(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        # Parquet is columnar; read row groups rather than pretending to stream.
        import pyarrow.parquet as pq  # noqa: PLC0415  (optional dependency)
        f = pq.ParquetFile(path)
        for batch in f.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    elif suf == ".ndjson":
        yield from pd.read_json(path, lines=True, chunksize=chunksize)
    elif suf == ".json":
        yield pd.read_json(path)
    else:
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)


def _hash_rows(df: pd.DataFrame, cols: Iterable[str] | None = None) -> np.ndarray:
    sub = df[list(cols)] if cols else df
    return pd.util.hash_pandas_object(sub, index=False).to_numpy(dtype="uint64")


class Reservoir:
    """Uniform sample of a stream, without holding the stream.

    Vitter's algorithm R. Every row that has gone past has an equal chance of
    being in the sample, which is what makes a share measured on the sample an
    estimate of the share in the file rather than of its first chunk.
    """

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        self.seen = 0
        self._parts: list[pd.DataFrame] = []
        self._held = 0
        self._rng = np.random.default_rng(seed)

    def add(self, chunk: pd.DataFrame) -> None:
        n = len(chunk)
        if self.seen < self.size:
            take = min(self.size - self.seen, n)
            self._parts.append(chunk.iloc[:take])
            self._held += take
            self.seen += take
            chunk, n = chunk.iloc[take:], n - take
            if n == 0:
                return
        # Beyond the first `size` rows, keep each with probability size/seen.
        idx = np.arange(n) + self.seen
        keep = self._rng.random(n) < (self.size / (idx + 1))
        if keep.any():
            self._parts.append(chunk.iloc[keep])
            self._held += int(keep.sum())
        self.seen += n
        # Without this the parts list grows as size*ln(seen/size) -- about 3x
        # the target sample on a 4M-row file, and worse as the file grows.
        # Compacting keeps the held rows bounded at 2x regardless of length.
        if self._held > 2 * self.size:
            self._compact()

    def _compact(self) -> None:
        out = pd.concat(self._parts, ignore_index=True)
        out = out.sample(self.size, random_state=int(self._rng.integers(1 << 31)))
        self._parts = [out.reset_index(drop=True)]
        self._held = len(out)

    def frame(self) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame()
        out = pd.concat(self._parts, ignore_index=True)
        if len(out) > self.size:
            out = out.sample(self.size, random_state=0).reset_index(drop=True)
        return out


def scan(path: str | Path, key: Iterable[str] | None = None,
         chunksize: int = DEFAULT_CHUNK,
         sample: int = DEFAULT_SAMPLE,
         exact_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Stream a file once. Return a uniform sample and the exact counts.

    Exact duplicate detection is the entire memory cost of this function: one
    64-bit hash per row, plus another per row if a key is given. That is 8
    bytes a row, so 100 million rows costs about 1.6GB with a key. Everything
    else here is bounded by `sample`. Pass exact_duplicates=False to trade the
    duplicate checks for constant memory.
    """
    p = Path(path)
    res = Reservoir(sample)
    row_hashes: list[np.ndarray] = []
    key_hashes: list[np.ndarray] = []
    rows = 0
    columns: list[str] = []
    missing_key: list[str] = []

    for chunk in _reader(p, chunksize):
        if not columns:
            columns = [str(c) for c in chunk.columns]
            if key:
                missing_key = [c for c in key if c not in chunk.columns]
        rows += len(chunk)
        res.add(chunk)
        if exact_duplicates:
            row_hashes.append(_hash_rows(chunk))
            if key and not missing_key:
                key_hashes.append(_hash_rows(chunk, key))

    def dupes(parts: list[np.ndarray]) -> int:
        """Count repeats without np.unique, which allocates a second copy.

        Concatenate once, sort in place, count adjacent equals. Peak cost is
        one 8-byte slot per row rather than three.
        """
        if not parts:
            return 0
        allh = np.concatenate(parts)
        parts.clear()               # release the per-chunk arrays immediately
        allh.sort()
        return int(np.count_nonzero(allh[1:] == allh[:-1]))

    exact = {"rows": rows, "columns": columns,
             "duplicates_checked": exact_duplicates,
             "duplicate_rows": dupes(row_hashes),
             "duplicate_keys": dupes(key_hashes) if key_hashes else 0,
             "missing_key": missing_key,
             "sampled": rows > sample, "sample_rows": min(rows, sample)}
    return res.frame(), exact


def exact_findings(exact: dict, key: Iterable[str] | None) -> list[Finding]:
    """The counting checks, computed over every row rather than the sample."""
    out: list[Finding] = []
    n = exact["rows"]
    d = exact["duplicate_rows"]
    if d and n:
        out.append(Finding(
            "duplicate_rows/exact", CRITICAL if d / n > 0.01 else WARNING, None,
            f"{d:,} of {n:,} rows ({d / n:.1%}) are exact duplicates.",
            "Deduplicate before aggregating. Repeated ingestion of one period "
            "shows up downstream as a real-looking jump in volume.",
            {"duplicate_rows": f"{d:,}", "share": f"{d / n:.1%}",
             "counted_over": "every row"}))
    if exact["missing_key"]:
        out.append(Finding(
            "duplicate_rows/key_missing", CRITICAL, None,
            f"declared key column(s) not present: {exact['missing_key']}",
            "A key that does not exist silently becomes no key at all.",
            {"missing": exact["missing_key"]}))
    elif key and exact["duplicate_keys"]:
        k = exact["duplicate_keys"]
        out.append(Finding(
            "duplicate_rows/key_not_unique", CRITICAL, ", ".join(key),
            f"declared key is not unique: {k:,} duplicate row(s) across {list(key)}.",
            "Widen the key until it is unique, and report the rows that still "
            "are not rather than dropping them.",
            {"duplicates": f"{k:,}", "counted_over": "every row"}))
    return out
