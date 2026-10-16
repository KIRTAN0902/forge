"""The LLM Orchestrator: decision engine and control plane (spec §7-8, §22)."""

from __future__ import annotations

from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message
from forge.storage.sqlite import SQLiteStore
from forge.tools.git import RestoreResult
from forge.workflow.policy import WorkflowPolicyValidator
from forge.workflow.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from forge.workflow.render import render_workflow_state
from forge.workflow.router import AgentRouter


class Orchestrator:
    """LLM-driven orchestrator with deterministic policy validation and persistence."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        router: AgentRouter,
        validator: WorkflowPolicyValidator | None = None,
        store: SQLiteStore | None = None,
        events: EventBus | None = None,
        workflow_id: str | None = None,
        max_iterations: int = 20,
    ) -> None:
        self.adapter = adapter
        self.router = router
        self.validator = validator or WorkflowPolicyValidator(max_iterations=max_iterations)
        self.store = store
        self.events = events or adapter.events
        self.workflow_id = workflow_id or adapter.workflow_id
        self.max_iterations = max_iterations

    async def decide(self, state: WorkflowState) -> Decision:
        """Formulate and deterministically validate the next workflow decision."""
        state.iteration += 1

        messages: list[Message] = [
            Message.system(ORCHESTRATOR_SYSTEM_PROMPT),
            Message.user(
                f"Review the current workflow state and decide the single next action:\n\n"
                f"{render_workflow_state(state)}"
            ),
        ]

        proposed = await self.adapter.structured(messages, Decision)
        validated = self.validator.validate(proposed, state)

        state.decisions.append(validated)

        if self.events is not None:
            await self.events.publish(
                Event(
                    type=EventType.DECISION_MADE,
                    workflow_id=state.workflow_id,
                    source="orchestrator",
                    payload={
                        "action": validated.action.value,
                        "rationale": validated.rationale,
                        "task_ids": validated.task_ids,
                        "iteration": state.iteration,
                    },
                )
            )

        return validated

    async def step(self, state: WorkflowState) -> tuple[Decision, WorkflowState]:
        """Perform a single decision and execution step, persisting state."""
        if state.is_terminal:
            default_action = (
                OrchestratorAction.COMPLETE
                if state.status == WorkflowStatus.COMPLETED
                else OrchestratorAction.BLOCK
            )
            last = (
                state.decisions[-1]
                if state.decisions
                else Decision(
                    action=default_action,
                    rationale="Workflow already in terminal state.",
                )
            )
            return last, state

        if state.status == WorkflowStatus.AWAITING_APPROVAL:
            last = (
                state.decisions[-1]
                if state.decisions
                else Decision(
                    action=OrchestratorAction.REQUEST_HUMAN_REVIEW,
                    rationale="Workflow is awaiting human approval.",
                )
            )
            return last, state

        decision = await self.decide(state)
        state = await self.router.route(decision, state)

        if self.store is not None:
            await self.store.save_workflow(state)

        return decision, state

    async def run(
        self,
        state: WorkflowState,
        *,
        max_steps: int | None = None,
        auto_mode: bool = False,
    ) -> WorkflowState:
        """Run the workflow loop until completion, block, or iteration ceiling."""
        is_resume = state.iteration > 0
        start_event = EventType.WORKFLOW_RESUMED if is_resume else EventType.WORKFLOW_STARTED

        if self.events is not None:
            await self.events.publish(
                Event(
                    type=start_event,
                    workflow_id=state.workflow_id,
                    source="orchestrator",
                    payload={
                        "user_request": state.user_request,
                        "iteration": state.iteration,
                    },
                )
            )

        if state.status == WorkflowStatus.PENDING:
            state.status = WorkflowStatus.RUNNING

        limit = max_steps or self.max_iterations
        steps_executed = 0

        while not state.is_terminal and steps_executed < limit:
            steps_executed += 1
            _, state = await self.step(state)
            if state.status == WorkflowStatus.AWAITING_APPROVAL:
                if auto_mode:
                    state = await self.approve(
                        state,
                        feedback="Auto-approved in AUTO mode: proceeding with execution.",
                    )
                else:
                    break

        if (
            not state.is_terminal
            and state.status != WorkflowStatus.AWAITING_APPROVAL
            and steps_executed >= limit
        ):
            state.status = WorkflowStatus.BLOCKED

        if self.store is not None:
            await self.store.save_workflow(state)

        if self.events is not None and (
            state.is_terminal or state.status == WorkflowStatus.BLOCKED
        ):
            end_event = (
                EventType.WORKFLOW_COMPLETED
                if state.status == WorkflowStatus.COMPLETED
                else (
                    EventType.WORKFLOW_BLOCKED
                    if state.status == WorkflowStatus.BLOCKED
                    else EventType.WORKFLOW_FAILED
                )
            )
            await self.events.publish(
                Event(
                    type=end_event,
                    workflow_id=state.workflow_id,
                    source="orchestrator",
                    payload={
                        "status": state.status.value,
                        "iterations": state.iteration,
                    },
                )
            )

        # Synchronize .forge/forge.md and .forge/forge.json
        try:
            from pathlib import Path

            from forge.workflow.reporter import sync_forge_artifacts

            env = getattr(self.router.runtime, "env", None)
            workspace = getattr(env, "workspace", None) if env else None
            ws_root = getattr(workspace, "root", None) if workspace else None
            root_path = Path(ws_root) if ws_root is not None else Path.cwd()
            sync_forge_artifacts(root_path, state)
        except Exception:
            pass

        return state

    async def approve(
        self,
        state: WorkflowState,
        *,
        feedback: str | None = None,
        decision: Decision | None = None,
    ) -> WorkflowState:
        """Approve and resolve a pending human review gate."""
        if state.status == WorkflowStatus.AWAITING_APPROVAL:
            state.status = WorkflowStatus.RUNNING
            if feedback:
                state.user_request = f"{state.user_request}\n\n[Human Guidance]: {feedback}"
            if decision is not None:
                state.decisions.append(decision)

            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.APPROVAL_RESOLVED,
                        workflow_id=state.workflow_id,
                        source="orchestrator",
                        payload={
                            "feedback": feedback,
                            "decision": decision.model_dump(mode="json") if decision else None,
                        },
                    )
                )

            if self.store is not None:
                await self.store.save_workflow(state)

        return state

    async def resume(
        self,
        state: WorkflowState,
        *,
        feedback: str | None = None,
        max_steps: int | None = None,
        auto_mode: bool = False,
    ) -> WorkflowState:
        """Approve (if awaiting) and resume running the workflow."""
        if state.status == WorkflowStatus.AWAITING_APPROVAL:
            await self.approve(state, feedback=feedback)
        return await self.run(state, max_steps=max_steps, auto_mode=auto_mode)

    async def rollback(
        self,
        state: WorkflowState,
        checkpoint_id: str | None = None,
    ) -> RestoreResult | None:
        """Manually trigger rollback to a checkpoint."""
        if checkpoint_id:
            res = await self.router.checkpoint_mgr.rollback(self.router.runtime.env, checkpoint_id)
        else:
            res = await self.router.checkpoint_mgr.rollback_to_latest(
                self.router.runtime.env, state.checkpoints
            )

        if res and self.events is not None:
            await self.events.publish(
                Event(
                    type=EventType.ROLLBACK_PERFORMED,
                    workflow_id=state.workflow_id,
                    source="orchestrator",
                    payload={
                        "checkpoint_id": res.checkpoint_id,
                        "changed_files": res.changed_files,
                        "removed_files": res.removed_files,
                    },
                )
            )

        if self.store is not None:
            await self.store.save_workflow(state)

        return res
