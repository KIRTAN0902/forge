"""Deterministic policy evaluation.

"The LLM has intelligence. The system has authority." Policies are plain Python, are
evaluated without any model involvement, and the most restrictive verdict wins.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from forge.policies.types import (
    Permission,
    PolicyDecision,
    ProposedAction,
    RiskLevel,
    Verdict,
)


class Policy(Protocol):
    name: str

    def evaluate(self, action: ProposedAction) -> PolicyDecision | None:
        """Return a decision, or ``None`` when the policy has no opinion."""
        ...


class RiskThresholdPolicy:
    """Low risk runs automatically; at/above the threshold a human must approve."""

    name = "risk_threshold"

    def __init__(self, approval_threshold: RiskLevel = RiskLevel.HIGH) -> None:
        self.approval_threshold = approval_threshold

    def evaluate(self, action: ProposedAction) -> PolicyDecision:
        if action.risk >= self.approval_threshold:
            return PolicyDecision(
                verdict=Verdict.REQUIRE_APPROVAL,
                reason=f"risk '{action.risk}' requires approval "
                f"(threshold '{self.approval_threshold}')",
                policy=self.name,
            )
        return PolicyDecision(
            verdict=Verdict.ALLOW, reason=f"risk '{action.risk}' is auto-approved", policy=self.name
        )


class PermissionPolicy:
    """Rejects actions that need a permission the project has not granted."""

    name = "permissions"

    def __init__(self, granted: Iterable[Permission]) -> None:
        self.granted = frozenset(granted)

    def evaluate(self, action: ProposedAction) -> PolicyDecision | None:
        missing = sorted(p.value for p in action.permissions if p not in self.granted)
        if missing:
            return PolicyDecision(
                verdict=Verdict.DENY,
                reason=f"permission not granted: {', '.join(missing)}",
                policy=self.name,
            )
        return None


class PolicyEngine:
    def __init__(self, policies: Sequence[Policy]) -> None:
        if not policies:
            raise ValueError("PolicyEngine needs at least one policy")
        self.policies = list(policies)

    def evaluate(self, action: ProposedAction) -> PolicyDecision:
        decisions = [d for p in self.policies if (d := p.evaluate(action)) is not None]
        if not decisions:
            # No policy claimed the action: fail closed.
            return PolicyDecision(
                verdict=Verdict.REQUIRE_APPROVAL,
                reason="no policy matched this action",
                policy="default",
            )
        # Most restrictive verdict wins; among equals, the first registered policy's reason.
        return max(decisions, key=lambda d: d.verdict.strictness)
