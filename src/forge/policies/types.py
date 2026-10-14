from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from forge.core.contracts.base import Contract


class RiskLevel(StrEnum):
    """How much damage an action can do. Ordered: LOW < MEDIUM < HIGH < CRITICAL."""

    LOW = "low"  # read-only, or trivially reversible
    MEDIUM = "medium"  # modifies the workspace; recoverable via checkpoints
    HIGH = "high"  # destructive, hard to reverse, or security-sensitive
    CRITICAL = "critical"  # production, secrets, irreversible external effects

    @property
    def rank(self) -> int:
        return _RISK_RANK[self]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank >= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank > other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank <= other.rank

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.rank < other.rank


_RISK_RANK = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


class Permission(StrEnum):
    READ = "read"  # read files / repository metadata
    WRITE = "write"  # create, modify, delete workspace files
    EXECUTE = "execute"  # run processes
    VCS = "vcs"  # mutate version-control state (commit, checkout, reset)
    NETWORK = "network"  # reach external hosts


class Verdict(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"

    @property
    def strictness(self) -> int:
        return {"allow": 0, "require_approval": 1, "deny": 2}[self.value]


class ProposedAction(Contract):
    """Something an agent (or the orchestrator) wants to do, as seen by the gate."""

    kind: str = Field(description="e.g. 'tool', 'workflow'.")
    name: str = Field(description="Tool or action name.")
    actor: str = Field(default="unknown", description="Role proposing the action.")
    risk: RiskLevel = RiskLevel.LOW
    permissions: list[Permission] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    command: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class PolicyDecision(Contract):
    verdict: Verdict
    reason: str
    policy: str
