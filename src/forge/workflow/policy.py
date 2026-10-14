"""Deterministic policy validation and guardrails for Orchestrator decisions (spec §8, §16).

"The LLM has intelligence. The system has authority."
Ensures decisions adhere to state constraints, iteration limits, thrashing detection,
and completion requirements.
"""

from __future__ import annotations

from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import TaskStatus
from forge.core.contracts.qa import QAStatus
from forge.core.contracts.state import WorkflowState
from forge.workflow.convergence import ConvergenceTracker


class WorkflowPolicyValidator:
    """Deterministic policy gate enforcing workflow invariants."""

    def __init__(
        self,
        *,
        max_iterations: int = 20,
        tracker: ConvergenceTracker | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.tracker = tracker or ConvergenceTracker()

    def validate(self, decision: Decision, state: WorkflowState) -> Decision:
        """Validate and potentially correct or constrain a proposed Orchestrator decision."""
        # Rule 1: Iteration limits
        if state.iteration >= self.max_iterations and decision.action not in {
            OrchestratorAction.BLOCK,
            OrchestratorAction.COMPLETE,
        }:
            return Decision(
                action=OrchestratorAction.BLOCK,
                rationale=f"Exceeded max iterations limit ({self.max_iterations}).",
                instructions="Workflow reached safety iteration limit.",
            )

        # Rule 2: Thrashing and convergence checks
        thrashing_verdict = self.tracker.check_thrashing(state)
        if thrashing_verdict.thrashing:
            return Decision(
                action=OrchestratorAction.REQUEST_HUMAN_REVIEW,
                rationale=f"Thrashing detected: {thrashing_verdict.reason}",
                instructions="Requesting human intervention due to repeated failure loop.",
            )

        # Rule 3: Rollback validity
        if decision.action == OrchestratorAction.ROLLBACK and not state.checkpoints:
            return Decision(
                action=OrchestratorAction.REPLAN,
                rationale="No checkpoints exist to roll back to; replanning instead.",
                instructions="Formulate a new plan to work around issues.",
            )

        # Rule 4: Plan existence prerequisite
        if state.plan is None:
            if decision.action not in {
                OrchestratorAction.PLAN,
                OrchestratorAction.REQUEST_RESEARCH,
            }:
                return Decision(
                    action=OrchestratorAction.PLAN,
                    rationale="No plan exists yet; must plan requirements before executing.",
                    instructions=decision.instructions,
                )
            return decision

        # Rule 5: Execution requires valid task targeting
        if decision.action == OrchestratorAction.EXECUTE:
            ready = state.plan.ready_tasks()
            if not ready:
                all_done = all(
                    t.status in {TaskStatus.DONE, TaskStatus.SKIPPED} for t in state.plan.tasks
                )
                if all_done:
                    if not state.qa_reports:
                        return Decision(
                            action=OrchestratorAction.FULL_QA,
                            rationale="All tasks complete; performing full verification QA.",
                        )
                    return Decision(
                        action=OrchestratorAction.COMPLETE,
                        rationale="All tasks complete and QA conducted.",
                    )
                return Decision(
                    action=OrchestratorAction.REPLAN,
                    rationale="Pending tasks have unresolved dependencies; replanning.",
                )

            target_id = decision.task_ids[0] if decision.task_ids else ready[0].id
            known_ids = {t.id for t in state.plan.tasks}
            if target_id not in known_ids or target_id not in {t.id for t in ready}:
                target_id = ready[0].id

            return Decision(
                action=OrchestratorAction.EXECUTE,
                rationale=decision.rationale or f"Executing ready task {target_id}",
                task_ids=[target_id],
                finding_ids=decision.finding_ids,
                instructions=decision.instructions,
            )

        # Rule 6: Completion guards
        if decision.action == OrchestratorAction.COMPLETE:
            incomplete = [
                t.id
                for t in state.plan.tasks
                if t.status not in {TaskStatus.DONE, TaskStatus.SKIPPED}
            ]
            if incomplete:
                ready = state.plan.ready_tasks()
                target_id = ready[0].id if ready else incomplete[0]
                return Decision(
                    action=OrchestratorAction.EXECUTE,
                    rationale=f"Cannot complete: tasks {incomplete} remain unfinished.",
                    task_ids=[target_id],
                    instructions="Implement remaining tasks.",
                )

            if not state.qa_reports:
                return Decision(
                    action=OrchestratorAction.FULL_QA,
                    rationale="Cannot complete without running QA verification first.",
                )

            latest_qa = state.qa_reports[-1]
            if latest_qa.status != QAStatus.PASS or latest_qa.blocking_findings:
                finding_ids = [f.id for f in latest_qa.blocking_findings]
                rework_task_id: str | None = state.current_task_id or (
                    state.plan.tasks[-1].id if state.plan and state.plan.tasks else None
                )
                return Decision(
                    action=OrchestratorAction.REEXECUTE,
                    rationale=(
                        f"Cannot complete: QA status {latest_qa.status.value} with "
                        f"blocking findings: {finding_ids}"
                    ),
                    task_ids=[rework_task_id] if rework_task_id else [],
                    finding_ids=finding_ids,
                    instructions="Address blocking QA findings before completing.",
                )

        return decision
