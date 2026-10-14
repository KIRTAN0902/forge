"""Convergence and agent thrashing detection (spec §16, §22).

Tracks failure signatures across iterations to detect:
1. Repeated identical failures (same failure signature >= 3 times).
2. Oscillating fixes (alternating failure signatures).
3. Stagnation (high iterations with zero progress).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from forge.core.contracts.plan import TaskStatus
from forge.core.contracts.qa import QAFinding, QAStatus
from forge.core.contracts.state import WorkflowState


def compute_failure_signature(
    *,
    error: str | None = None,
    qa_findings: Sequence[QAFinding] = (),
    failed_test_tail: str | None = None,
) -> str:
    """Compute a normalized fingerprint string representing the current failure."""
    parts: list[str] = []

    if qa_findings:
        for f in sorted(qa_findings, key=lambda x: x.id):
            files = ",".join(sorted(f.affected_files))
            parts.append(f"QA:{f.severity.value}:{f.type.value}:{f.description.strip()}:{files}")

    if failed_test_tail:
        # Strip timestamps and whitespace to normalize test failure output
        normalized_tail = "".join(failed_test_tail.split()[-30:])
        parts.append(f"TEST:{normalized_tail}")

    if error:
        parts.append(f"ERR:{error.strip()}")

    if not parts:
        return "NO_FAILURE"

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ThrashingVerdict:
    """Outcome of thrashing evaluation."""

    thrashing: bool
    reason: str = ""
    signature: str | None = None


class ConvergenceTracker:
    """Tracks workflow trajectory and flags loops or agent thrashing."""

    def __init__(
        self,
        *,
        max_same_failures: int = 3,
        stagnation_threshold: int = 6,
    ) -> None:
        self.max_same_failures = max_same_failures
        self.stagnation_threshold = stagnation_threshold
        self.history: list[str] = []

    def record_signature(self, signature: str) -> None:
        """Record the failure signature for the current turn."""
        if signature != "NO_FAILURE":
            self.history.append(signature)

    def check_thrashing(self, state: WorkflowState) -> ThrashingVerdict:
        """Analyze failure history and state to identify thrashing patterns."""
        # 1. Compute current failure signature
        current_sig = self._signature_from_state(state)
        if current_sig != "NO_FAILURE":
            self.record_signature(current_sig)

        if not self.history:
            return ThrashingVerdict(thrashing=False)

        # 2. Check for repeated identical failure signatures
        if len(self.history) >= self.max_same_failures:
            recent = self.history[-self.max_same_failures :]
            if len(set(recent)) == 1:
                return ThrashingVerdict(
                    thrashing=True,
                    reason=(
                        f"Identical failure signature '{recent[0]}' repeated "
                        f"{self.max_same_failures} consecutive times."
                    ),
                    signature=recent[0],
                )

        # 3. Check for oscillating failures (A -> B -> A -> B)
        if len(self.history) >= 4:
            s1, s2, s3, s4 = self.history[-4:]
            if s1 == s3 and s2 == s4 and s1 != s2:
                return ThrashingVerdict(
                    thrashing=True,
                    reason=(
                        f"Oscillating failure pattern detected between '{s1}' and '{s2}'. "
                        "Fixes are alternating without resolving underlying issues."
                    ),
                    signature=s4,
                )

        # 4. Check for stagnation (many iterations without completing tasks or resolving findings)
        if state.iteration >= self.stagnation_threshold and state.plan:
            completed_tasks = sum(
                1 for t in state.plan.tasks if t.status in {TaskStatus.DONE, TaskStatus.SKIPPED}
            )
            if completed_tasks == 0 and state.latest_qa and state.latest_qa.status != QAStatus.PASS:
                return ThrashingVerdict(
                    thrashing=True,
                    reason=(
                        f"Stagnation detected: {state.iteration} iterations elapsed with zero "
                        "completed tasks and ongoing QA failures."
                    ),
                )

        return ThrashingVerdict(thrashing=False)

    def _signature_from_state(self, state: WorkflowState) -> str:
        findings = state.latest_qa.findings if state.latest_qa else []
        failed_tests = [t.output_tail for t in state.test_results if not t.succeeded]
        test_tail = failed_tests[-1] if failed_tests else None
        err = state.errors[-1].message if state.errors else None
        return compute_failure_signature(
            error=err,
            qa_findings=findings,
            failed_test_tail=test_tail,
        )
