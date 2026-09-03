"""What a check reports, and how a run of them is summarised."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}


@dataclass
class Finding:
    """One thing that is wrong, with the evidence to check it.

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

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        out = f"{self.severity.upper():<8} {self.check}{where}\n    {self.message}"
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

    def __post_init__(self) -> None:
        self.findings.sort(key=lambda f: (_ORDER.get(f.severity, 9), f.check))

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CRITICAL]

    @property
    def ok(self) -> bool:
        """True when nothing critical was found. Warnings do not fail a run."""
        return not self.critical

    def raise_for_critical(self) -> None:
        """Fail loudly in a pipeline. The whole point is not shipping green."""
        if self.critical:
            joined = "\n".join(str(f) for f in self.critical)
            raise PipelineLied(f"{len(self.critical)} critical finding(s):\n{joined}")

    def __str__(self) -> str:
        head = (f"pipelie: {self.rows:,} rows x {self.columns} columns, "
                f"{len(self.checks_run)} checks")
        if not self.findings:
            return head + "\n\nNothing found. That is not proof of correctness -- "\
                          "it means these particular lies are absent."
        counts = {s: sum(1 for f in self.findings if f.severity == s)
                  for s in (CRITICAL, WARNING, INFO)}
        tally = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        body = "\n\n".join(str(f) for f in self.findings)
        return f"{head}\n{tally}\n\n{body}"


class PipelineLied(AssertionError):
    """Raised by Report.raise_for_critical."""
