from __future__ import annotations

from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import FindingType, QAFinding, QAReport, QAStatus, Severity
from forge.core.contracts.state import WorkflowState
from forge.workflow.policy import WorkflowPolicyValidator


def _sample_state_with_plan() -> WorkflowState:
    plan = Plan(
        goal="Test goal",
        tasks=[
            Task(id="T-1", description="First task", status=TaskStatus.DONE),
            Task(
                id="T-2",
                description="Second task",
                dependencies=["T-1"],
                status=TaskStatus.PENDING,
            ),
        ],
    )
    return WorkflowState(
        user_request="Test request",
        plan=plan,
        iteration=1,
    )


class TestWorkflowPolicyValidator:
    def test_enforces_plan_when_no_plan_exists(self) -> None:
        state = WorkflowState(user_request="Do something")
        validator = WorkflowPolicyValidator()

        # LLM erroneously proposes EXECUTE before planning
        proposed = Decision(action=OrchestratorAction.EXECUTE, rationale="Jump to execute")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.PLAN
        assert "No plan exists" in validated.rationale

    def test_iteration_limit_triggers_block(self) -> None:
        state = _sample_state_with_plan()
        state.iteration = 20
        validator = WorkflowPolicyValidator(max_iterations=20)

        proposed = Decision(action=OrchestratorAction.EXECUTE, rationale="Continue")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.BLOCK
        assert "Exceeded max iterations" in validated.rationale

    def test_execute_selects_ready_task_if_unspecified(self) -> None:
        state = _sample_state_with_plan()
        validator = WorkflowPolicyValidator()

        proposed = Decision(action=OrchestratorAction.EXECUTE, rationale="Execute")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.EXECUTE
        assert validated.task_ids == ["T-2"]

    def test_cannot_complete_without_qa(self) -> None:
        state = _sample_state_with_plan()
        # Mark all tasks done
        for t in state.plan.tasks:  # type: ignore[union-attr]
            t.status = TaskStatus.DONE

        validator = WorkflowPolicyValidator()
        proposed = Decision(action=OrchestratorAction.COMPLETE, rationale="Done")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.FULL_QA
        assert "without running QA" in validated.rationale

    def test_cannot_complete_with_blocking_qa_findings(self) -> None:
        state = _sample_state_with_plan()
        for t in state.plan.tasks:  # type: ignore[union-attr]
            t.status = TaskStatus.DONE

        state.qa_reports.append(
            QAReport(
                status=QAStatus.FAIL,
                summary="Fails due to bug",
                findings=[
                    QAFinding(
                        id="QA-1",
                        severity=Severity.P1,
                        type=FindingType.FUNCTIONAL_BUG,
                        description="Bug",
                    )
                ],
            )
        )

        validator = WorkflowPolicyValidator()
        proposed = Decision(action=OrchestratorAction.COMPLETE, rationale="Try to complete")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.REEXECUTE
        assert validated.finding_ids == ["QA-1"]

    def test_cannot_complete_with_incomplete_tasks(self) -> None:
        state = _sample_state_with_plan()
        # T-2 is still PENDING
        validator = WorkflowPolicyValidator()
        proposed = Decision(action=OrchestratorAction.COMPLETE, rationale="Done")
        validated = validator.validate(proposed, state)

        assert validated.action == OrchestratorAction.EXECUTE
        assert validated.task_ids == ["T-2"]
