from __future__ import annotations

import pytest

from forge.policies import (
    Permission,
    PermissionPolicy,
    PolicyDecision,
    PolicyEngine,
    ProposedAction,
    RiskLevel,
    RiskThresholdPolicy,
    StaticApprover,
    Verdict,
)


def action(risk: RiskLevel, *perms: Permission) -> ProposedAction:
    return ProposedAction(kind="tool", name="t", risk=risk, permissions=list(perms))


def test_risk_ordering_is_semantic_not_alphabetical() -> None:
    assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL
    assert RiskLevel.CRITICAL >= RiskLevel.HIGH
    assert max(RiskLevel.LOW, RiskLevel.HIGH) == RiskLevel.HIGH


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (RiskLevel.LOW, Verdict.ALLOW),
        (RiskLevel.MEDIUM, Verdict.ALLOW),
        (RiskLevel.HIGH, Verdict.REQUIRE_APPROVAL),
        (RiskLevel.CRITICAL, Verdict.REQUIRE_APPROVAL),
    ],
)
def test_risk_threshold(risk: RiskLevel, expected: Verdict) -> None:
    assert RiskThresholdPolicy(RiskLevel.HIGH).evaluate(action(risk)).verdict == expected


def test_most_restrictive_verdict_wins() -> None:
    engine = PolicyEngine(
        [RiskThresholdPolicy(RiskLevel.HIGH), PermissionPolicy([Permission.READ])]
    )
    # Low risk, but needs a permission that is not granted -> DENY beats ALLOW.
    decision = engine.evaluate(action(RiskLevel.LOW, Permission.NETWORK))
    assert decision.verdict == Verdict.DENY
    assert decision.policy == "permissions"
    assert "network" in decision.reason

    assert engine.evaluate(action(RiskLevel.LOW, Permission.READ)).verdict == Verdict.ALLOW


def test_engine_fails_closed_without_opinion() -> None:
    class Silent:
        name = "silent"

        def evaluate(self, action: ProposedAction) -> PolicyDecision | None:
            return None

    decision = PolicyEngine([Silent()]).evaluate(action(RiskLevel.LOW))
    assert decision.verdict == Verdict.REQUIRE_APPROVAL


async def test_static_approver_records_requests() -> None:
    approver = StaticApprover(approve=False)
    a = action(RiskLevel.HIGH)
    decision = RiskThresholdPolicy().evaluate(a)
    assert await approver.request_approval(a, decision) is False
    assert approver.requests == [a]
