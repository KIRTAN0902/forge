from __future__ import annotations

from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import FindingType, QAFinding, QAReport, QAStatus, Severity
from forge.core.contracts.state import ErrorRecord, WorkflowState
from forge.workflow.convergence import (
    ConvergenceTracker,
    compute_failure_signature,
)
from forge.workflow.policy import WorkflowPolicyValidator


class TestConvergence:
    def test_compute_failure_signature_empty(self) -> None:
        sig = compute_failure_signature()
        assert sig == "NO_FAILURE"

    def test_compute_failure_signature_deterministic(self) -> None:
        f1 = QAFinding(
            id="QA-1",
            severity=Severity.P1,
            type=FindingType.FUNCTIONAL_BUG,
            description="Null pointer exception in parser",
            affected_files=["src/parser.py"],
        )
        f2 = QAFinding(
            id="QA-2",
            severity=Severity.P2,
            type=FindingType.QUALITY,
            description="Unused import",
            affected_files=["src/utils.py"],
        )

        sig1 = compute_failure_signature(
            error="SyntaxError: invalid syntax",
            qa_findings=[f1, f2],
            failed_test_tail="FAILED tests/test_parser.py - SyntaxError",
        )
        # Order of findings swapped should produce identical signature due to internal sorting
        sig2 = compute_failure_signature(
            error="SyntaxError: invalid syntax",
            qa_findings=[f2, f1],
            failed_test_tail="FAILED tests/test_parser.py - SyntaxError",
        )
        assert len(sig1) == 16
        assert sig1 == sig2

    def test_compute_failure_signature_different_inputs(self) -> None:
        sig1 = compute_failure_signature(error="Error A")
        sig2 = compute_failure_signature(error="Error B")
        assert sig1 != sig2

    def test_tracker_repeated_identical_failures(self) -> None:
        tracker = ConvergenceTracker(max_same_failures=3)
        state = WorkflowState(user_request="Test")

        # Turn 1: failure A
        state.errors.append(ErrorRecord(source="test", message="AssertionError: 1 != 2"))
        verdict = tracker.check_thrashing(state)
        assert not verdict.thrashing

        # Turn 2: same failure A
        verdict = tracker.check_thrashing(state)
        assert not verdict.thrashing

        # Turn 3: same failure A -> triggers thrashing
        verdict = tracker.check_thrashing(state)
        assert verdict.thrashing
        assert "repeated 3 consecutive times" in verdict.reason

    def test_tracker_oscillating_failures(self) -> None:
        tracker = ConvergenceTracker()
        # Feed alternating signatures A, B, A, B
        tracker.record_signature("sig_A")
        tracker.record_signature("sig_B")
        tracker.record_signature("sig_A")
        tracker.record_signature("sig_B")

        state = WorkflowState(user_request="Test")
        verdict = tracker.check_thrashing(state)
        assert verdict.thrashing
        assert "Oscillating failure pattern detected" in verdict.reason

    def test_tracker_stagnation_detection(self) -> None:
        tracker = ConvergenceTracker(stagnation_threshold=5)
        plan = Plan(
            goal="Goal",
            tasks=[Task(id="T-1", description="Task 1", status=TaskStatus.IN_PROGRESS)],
        )
        state = WorkflowState(
            user_request="Test",
            plan=plan,
            iteration=6,
        )
        state.qa_reports.append(
            QAReport(
                status=QAStatus.FAIL,
                summary="Still failing",
                findings=[
                    QAFinding(
                        id="Q-1",
                        severity=Severity.P1,
                        type=FindingType.FUNCTIONAL_BUG,
                        description="Bug",
                    )
                ],
            )
        )

        verdict = tracker.check_thrashing(state)
        assert verdict.thrashing
        assert "Stagnation detected" in verdict.reason

    def test_policy_validator_requests_human_review_on_thrashing(self) -> None:
        tracker = ConvergenceTracker(max_same_failures=3)
        validator = WorkflowPolicyValidator(tracker=tracker)

        state = WorkflowState(user_request="Fix issue")
        state.errors.append(ErrorRecord(source="tool", message="File locked error"))

        # Trigger 3 times
        tracker.check_thrashing(state)
        tracker.check_thrashing(state)

        # 3rd time through policy validator
        proposed = Decision(action=OrchestratorAction.EXECUTE, rationale="Try again")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.REQUEST_HUMAN_REVIEW
        assert "Thrashing detected" in validated.rationale
