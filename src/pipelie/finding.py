"""What a check reports, how findings are identified, and how a run is summarised."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass
class Finding:
    """One thing that is wrong, with the evidence to check it.

    `code` is a stable rule identifier -- "parse_carnage/mixed_formats" -- and
    it deliberately excludes every number in the message. A column that had
    1,351 duplicates last week and 1,352 today is the same finding, and a
    fingerprint that changed with the count would make baselines useless.

    `evidence` carries the numbers behind the claim rather than a prose
    summary, because a finding a user cannot verify is one they will learn to
    ignore. `fix` says what to do, since every check here exists because the
    author shipped the bug it detects.
    """

    check: str
    severity: str
    column: str | None
    message: str
    fix: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    # Last, and keyword-only in practice: the checks build Findings positionally
    # and inserting a field ahead of `fix` would silently shift every argument.
    code: str = ""

    def __post_init__(self) -> None:
        # Checks pass a slashed code as the first argument -- "degenerate/all_null".
        # The family before the slash is the check name; the whole string is the
        # rule identity a baseline pins on.
        if not self.code:
            self.code = self.check
        self.check = self.code.split("/", 1)[0]

    @property
    def fingerprint(self) -> str:
        """Stable identity: rule and column, never the counts."""
        return f"{self.code}:{self.column or '*'}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "check": self.check, "severity": self.severity,
                "column": self.column, "message": self.message, "fix": self.fix,
                "evidence": self.evidence, "fingerprint": self.fingerprint}

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        out = f"{self.severity.upper():<8} {self.code}{where}\n    {self.message}"
        if self.fix:
            out += f"\n    fix: {self.fix}"
        if self.evidence:
            ev = "  ".join(f"{k}={v}" for k, v in self.evidence.items())
            out += f"\n    evidence: {ev}"
        return out


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    rows: int = 0
    columns: int = 0
    suppressed: int = 0
    # Set when the table was streamed and checks ran on a uniform sample.
    # Counting checks stay exact; see stream.py for which and why.
    sampled_rows: int = 0

    def __post_init__(self) -> None:
        self.findings.sort(key=lambda f: (_ORDER.get(f.severity, 9), f.code, f.column or ""))

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def ok(self) -> bool:
        """True when nothing critical was found. Warnings do not fail a run."""
        return not self.critical

    @property
    def fingerprints(self) -> list[str]:
        return sorted({f.fingerprint for f in self.findings})

    def raise_for_critical(self) -> None:
        """Fail loudly in a pipeline. The whole point is not shipping green."""
        if self.critical:
            joined = "\n".join(str(f) for f in self.critical)
            raise PipelineLied(f"{len(self.critical)} critical finding(s):\n{joined}")

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "columns": self.columns,
                "checks_run": self.checks_run, "suppressed": self.suppressed,
                "sampled_rows": self.sampled_rows, "ok": self.ok,
                "findings": [f.to_dict() for f in self.findings]}

    def __str__(self) -> str:
        head = (f"pipelie: {self.rows:,} rows x {self.columns} columns, "
                f"{len(self.checks_run)} checks")
        if self.sampled_rows:
            head += (f"\n  streamed: duplicates and row counts exact over all "
                     f"{self.rows:,} rows; every other check on a uniform "
                     f"sample of {self.sampled_rows:,}")
        if self.suppressed:
            head += f", {self.suppressed} suppressed"
        if not self.findings:
            tail = ("Nothing found. That is not proof of correctness -- "
                    "it means these particular lies are absent.")
            return f"{head}\n\n{tail}"
        counts = {s: sum(1 for f in self.findings if f.severity == s)
                  for s in (CRITICAL, WARNING, INFO)}
        tally = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        body = "\n\n".join(str(f) for f in self.findings)
        return f"{head}\n{tally}\n\n{body}"


class PipelineLied(AssertionError):
    """Raised by Report.raise_for_critical."""
