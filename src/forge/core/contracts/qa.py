"""Structured QA output. QA validates against the requirement contract."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from forge.core.contracts.base import Contract


class Severity(StrEnum):
    P0 = "P0"  # critical: security hole, data loss, total failure
    P1 = "P1"  # requirement violated / major bug
    P2 = "P2"  # real defect with limited impact
    P3 = "P3"  # minor / quality


BLOCKING_SEVERITIES = frozenset({Severity.P0, Severity.P1})


class FindingType(StrEnum):
    REQUIREMENT_VIOLATION = "REQUIREMENT_VIOLATION"
    FUNCTIONAL_BUG = "FUNCTIONAL_BUG"
    REGRESSION = "REGRESSION"
    INTEGRATION = "INTEGRATION"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    QUALITY = "QUALITY"
    TEST_GAP = "TEST_GAP"


class QAStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class QAScope(StrEnum):
    FULL = "full"
    TARGETED = "targeted"


class QAFinding(Contract):
    id: str
    severity: Severity
    type: FindingType
    description: str
    evidence: str = ""
    affected_files: list[str] = Field(default_factory=list)
    criterion_ids: list[str] = Field(default_factory=list)
    required_rework: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.severity in BLOCKING_SEVERITIES


class CriterionVerdict(Contract):
    criterion_id: str
    satisfied: bool | None = Field(
        description="True = proven, False = disproven, None = could not be determined."
    )
    evidence: str = ""


class QAReport(Contract):
    status: QAStatus
    scope: QAScope = QAScope.FULL
    summary: str = ""
    criteria: list[CriterionVerdict] = Field(default_factory=list)
    findings: list[QAFinding] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[QAFinding]:
        return [f for f in self.findings if f.blocking]

    def unproven_criteria(self) -> list[str]:
        return [c.criterion_id for c in self.criteria if c.satisfied is not True]
