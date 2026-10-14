"""Human approval channel.

The engine asks an :class:`Approver` when a policy returns ``REQUIRE_APPROVAL``. The CLI
provides an interactive approver; a future web UI can provide a remote one.
"""

from __future__ import annotations

from typing import Protocol

from forge.policies.types import PolicyDecision, ProposedAction


class Approver(Protocol):
    async def request_approval(self, action: ProposedAction, decision: PolicyDecision) -> bool:
        """Return True to approve the action, False to reject it."""
        ...


class StaticApprover:
    """Always gives the same answer. Used for tests and non-interactive runs."""

    def __init__(self, approve: bool) -> None:
        self.approve = approve
        self.requests: list[ProposedAction] = []

    async def request_approval(self, action: ProposedAction, decision: PolicyDecision) -> bool:
        self.requests.append(action)
        return self.approve
