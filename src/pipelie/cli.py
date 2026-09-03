"""pipelie <file> -- audit a CSV, Parquet or JSON file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .api import audit
from .suppress import write_baseline

BASELINE = "pipelie-baseline.json"


def _load(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf in {".json", ".ndjson"}:
        return pd.read_json(path, lines=suf == ".ndjson")
    return pd.read_csv(path, low_memory=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pipelie",
        description="Find the bugs that leave a pipeline reporting green.",
        epilog="Adopting on a table that already has problems: run once with "
               "--accept to record today's findings as debt, then use "
               "--baseline so only new problems fail the build.")
    ap.add_argument("path", type=Path, help="CSV, Parquet or JSON file")
    ap.add_argument("--target", help="outcome column; unlocks the missingness check")
    ap.add_argument("--key", help="comma-separated columns you believe are unique")
    ap.add_argument("--ignore", action="append", metavar="PATTERN", default=[],
                    help="suppress findings matching a glob over 'code:column', "
                         "e.g. 'parse_carnage/*' or '*:Queue ID'. Repeatable.")
    ap.add_argument("--baseline", nargs="?", const=BASELINE, metavar="FILE",
                    help=f"suppress findings recorded in FILE (default {BASELINE})")
    ap.add_argument("--accept", nargs="?", const=BASELINE, metavar="FILE",
                    help="write today's findings to FILE as accepted debt and exit 0")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any finding, not just critical ones")
    a = ap.parse_args(argv)

    if not a.path.exists():
        print(f"no such file: {a.path}", file=sys.stderr)
        return 2

    df = _load(a.path)
    key = [c.strip() for c in a.key.split(",")] if a.key else None

    if a.accept:
        fs = audit(df, target=a.target, key=key).findings
        n = write_baseline(a.accept, fs)
        print(f"accepted {n} finding(s) as baseline -> {a.accept}")
        print("future runs with --baseline will report only what is new.")
        return 0

    report = audit(df, target=a.target, key=key,
                   ignore=a.ignore, baseline=a.baseline)

    if a.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report)

    if report.critical:
        return 1
    return 1 if (a.strict and report.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
