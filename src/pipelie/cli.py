"""pipelie <file> -- audit a CSV, Parquet or JSON file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .api import audit_file, snapshot
from .stream import DEFAULT_CHUNK, DEFAULT_SAMPLE
from .suppress import write_baseline

BASELINE = "pipelie-baseline.json"
PROFILE = "pipelie-profile.json"


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
               "--baseline so only new problems fail the build. To catch a "
               "table CHANGING rather than being wrong, --snapshot it once and "
               "pass --profile on later runs.")
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
    ap.add_argument("--profile", nargs="?", const=PROFILE, metavar="FILE",
                    help=f"compare against a profile written earlier and report "
                         f"what changed (default {PROFILE})")
    ap.add_argument("--snapshot", nargs="?", const=PROFILE, metavar="FILE",
                    help="record what this table looks like now, for later "
                         "comparison, and exit 0")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, metavar="N",
                    help=f"rows to sample for the distribution checks when the "
                         f"file is larger (default {DEFAULT_SAMPLE:,}). Row "
                         f"counts and duplicates stay exact regardless.")
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK, metavar="N",
                    help=f"rows read at a time (default {DEFAULT_CHUNK:,})")
    ap.add_argument("--no-exact-duplicates", action="store_true",
                    help="skip duplicate detection, which is the only part "
                         "that costs memory per row (8 bytes)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any finding, not just critical ones")
    a = ap.parse_args(argv)

    if not a.path.exists():
        print(f"no such file: {a.path}", file=sys.stderr)
        return 2

    key = [c.strip() for c in a.key.split(",")] if a.key else None

    if a.snapshot or a.accept:
        df = _load(a.path)

    if a.snapshot:
        prof = snapshot(df, a.snapshot)
        print(f"profiled {prof['rows']:,} rows x {len(prof['columns'])} columns "
              f"-> {a.snapshot}")
        print("run again later with --profile to be told what changed.")
        return 0

    if a.accept:
        fs = audit_file(a.path, target=a.target, key=key).findings
        n = write_baseline(a.accept, fs)
        print(f"accepted {n} finding(s) as baseline -> {a.accept}")
        print("future runs with --baseline will report only what is new.")
        return 0

    report = audit_file(a.path, target=a.target, key=key, ignore=a.ignore,
                        baseline=a.baseline, profile_path=a.profile,
                        chunksize=a.chunksize, sample=a.sample,
                        exact_duplicates=not a.no_exact_duplicates)

    if a.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report)

    if report.critical:
        return 1
    return 1 if (a.strict and report.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
