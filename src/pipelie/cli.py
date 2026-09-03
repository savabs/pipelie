"""pipelie <file> [--target col] [--key a,b] -- audit a CSV or Parquet file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .api import audit


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".ndjson"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".ndjson")
    return pd.read_csv(path, low_memory=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pipelie",
        description="Find the bugs that leave a pipeline reporting green.")
    ap.add_argument("path", type=Path, help="CSV, Parquet or JSON file")
    ap.add_argument("--target", help="outcome column; unlocks the missingness check")
    ap.add_argument("--key", help="comma-separated columns you believe are unique")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any finding, not just critical ones")
    a = ap.parse_args(argv)

    if not a.path.exists():
        print(f"no such file: {a.path}", file=sys.stderr)
        return 2

    df = _load(a.path)
    key = [c.strip() for c in a.key.split(",")] if a.key else None
    report = audit(df, target=a.target, key=key)
    print(report)

    if report.critical:
        return 1
    return 1 if (a.strict and report.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
