"""Agent Router: dispatches validated decisions to specialist agents (spec §7, §22)."""

from __future__ import annotations

from forge.agents.executor import ExecutorAgent
from forge.agents.orchestrator import OrchestratorAgent
from forge.agents.qa import QAAgent
from forge.context.analyzer import RepositoryAnalyzer
from forge.context.builder import ContextBuilder
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.execution import ExecutionStatus
from forge.core.contracts.plan import Task, TaskStatus
from forge.core.contracts.qa import QAScope, QAStatus
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.harness import AdversarialQAGatekeeper, EpisodicRepoMemory
from forge.skills import (
    AutonomousSkillRouter,
    ModelTier,
    OperationType,
    RepoSignals,
    SkillEvaluator,
    SkillSignal,
    format_skills_for_prompt,
)
from forge.tools.runtime import ToolRuntime
from forge.workflow.recovery import CheckpointManager


class AgentRouter:
    """Dispatches decisions to Orchestrator (Planning), Executor, or QA and updates WorkflowState."""

    def __init__(
        self,
        *,
        orchestrator: OrchestratorAgent | None = None,
        planner: OrchestratorAgent | None = None,
        executor: ExecutorAgent,
        qa: QAAgent,
        runtime: ToolRuntime,
        checkpoint_mgr: CheckpointManager | None = None,
        events: EventBus | None = None,
    ) -> None:
        resolved_orch = orchestrator or planner
        if resolved_orch is None:
            raise ValueError("An orchestrator agent must be provided to AgentRouter")
        self.orchestrator: OrchestratorAgent = resolved_orch
        self.planner: OrchestratorAgent = resolved_orch
        self.executor = executor
        self.qa = qa
        self.runtime = runtime
        self.checkpoint_mgr = checkpoint_mgr or CheckpointManager()
        self.events = events
        self.skills = SkillEvaluator()
        self.autonomous_router = AutonomousSkillRouter()
        workspace = getattr(runtime.env, "workspace", None)
        ws_root = getattr(workspace, "root", None) if workspace is not None else None
        self.memory = EpisodicRepoMemory(ws_root)

    async def route(self, decision: Decision, state: WorkflowState) -> WorkflowState:
        """Execute the action specified by decision and record results in state."""
        self.orchestrator.workflow_id = state.workflow_id
        self.executor.workflow_id = state.workflow_id
        self.qa.workflow_id = state.workflow_id
        action = decision.action

        planning_actions = {
            OrchestratorAction.PLAN,
            OrchestratorAction.REPLAN,
            OrchestratorAction.REQUEST_RESEARCH,
        }
        execution_actions = {
            OrchestratorAction.EXECUTE,
            OrchestratorAction.REEXECUTE,
            OrchestratorAction.RETRY,
        }

        if action in planning_actions:
            await self._handle_planning(decision, state)
        elif action in execution_actions:
            await self._handle_execution(decision, state)
        elif action in {OrchestratorAction.TARGETED_QA, OrchestratorAction.FULL_QA}:
            await self._handle_qa(decision, state)
        elif action == OrchestratorAction.ROLLBACK:
            await self._handle_rollback(decision, state)
        elif action == OrchestratorAction.REQUEST_HUMAN_REVIEW:
            state.status = WorkflowStatus.AWAITING_APPROVAL
            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.APPROVAL_REQUESTED,
                        workflow_id=state.workflow_id,
                        source="router",
                        payload={
                            "rationale": decision.rationale,
                            "instructions": decision.instructions,
                        },
                    )
                )
        elif action == OrchestratorAction.BLOCK:
            state.status = WorkflowStatus.BLOCKED
        elif action == OrchestratorAction.COMPLETE:
            state.status = WorkflowStatus.COMPLETED

        state.touch()
        return state

    async def _build_context_builder(self) -> ContextBuilder:
        env = self.runtime.env
        analyzer = RepositoryAnalyzer(
            env.workspace, symbols=env.symbols, test_command=env.test_command
        )
        git_repo = await env.git()
        analysis = await analyzer.analyze(git_repo)
        return ContextBuilder(analysis)

    async def _handle_planning(self, decision: Decision, state: WorkflowState) -> None:
        builder = await self._build_context_builder()
        context = builder.planner(
            request=state.user_request,
            capabilities=self.planner.adapter.capabilities,
            optimization=self.planner.adapter.optimization,
            requirements=state.requirements,
        )

        is_replan = decision.action == OrchestratorAction.REPLAN
        is_research = decision.action == OrchestratorAction.REQUEST_RESEARCH

        consec_fails = sum(1 for d in state.decisions[-3:] if d.action == OrchestratorAction.REPLAN)
        plan_signal = SkillSignal(
            user_request=state.user_request,
            consecutive_failures=consec_fails,
        )
        plan_skills = self.skills.evaluate(plan_signal, agent_role="orchestrator")
        for act in plan_skills:
            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.SKILL_ACTIVATED,
                        workflow_id=state.workflow_id,
                        source="router",
                        payload={
                            "skill_id": act.skill.id,
                            "name": act.skill.name,
                            "agent": "orchestrator",
                            "rationale": act.rationale,
                        },
                    )
                )

        effective_request = state.user_request
        if plan_skills:
            effective_request += f"\n\n{format_skills_for_prompt(plan_skills)}"

        result = await self.planner.plan(
            request=effective_request,
            context=context,
            requirements=state.requirements if is_replan else None,
            existing_plan=state.plan if is_replan else None,
            investigate=is_research,
            max_investigation_steps=5 if is_research else 0,
        )

        state.requirements = result.requirements
        state.plan = result.plan
        ready = state.plan.ready_tasks()
        if ready:
            state.current_task_id = ready[0].id
        state.status = WorkflowStatus.RUNNING

    async def _handle_execution(self, decision: Decision, state: WorkflowState) -> None:
        if state.plan is None:
            return

        target_task: Task | None = None
        if decision.task_ids:
            target_task = state.plan.task(decision.task_ids[0])
        elif state.current_task_id:
            target_task = state.plan.task(state.current_task_id)
        else:
            ready = state.plan.ready_tasks()
            if ready:
                target_task = ready[0]

        if target_task is None:
            if state.plan and state.plan.tasks:
                target_task = state.plan.tasks[-1]
            else:
                return

        # Snapshot working tree before mutating operations
        ckpt = await self.checkpoint_mgr.create_checkpoint(
            self.runtime.env,
            label=f"task_{target_task.id}_iter_{state.iteration}",
            iteration=state.iteration,
        )
        if ckpt:
            state.checkpoints.append(ckpt)
            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.CHECKPOINT_CREATED,
                        workflow_id=state.workflow_id,
                        source="router",
                        payload={"checkpoint_id": ckpt.id, "label": ckpt.label},
                    )
                )

        target_task.status = TaskStatus.IN_PROGRESS
        state.current_task_id = target_task.id

        # Rework guidance: extract specific QA findings to address
        findings = []
        if decision.finding_ids and state.latest_qa:
            findings = [f for f in state.latest_qa.findings if f.id in decision.finding_ids]
        elif state.latest_qa:
            findings = state.latest_qa.blocking_findings

        builder = await self._build_context_builder()

        # Autonomous Skill Routing for Executor
        git_repo = await self.runtime.env.git()
        diff_str = await git_repo.diff() if git_repo is not None else ""
        last_test_exit = state.test_results[-1].exit_code if state.test_results else None
        last_test_out = state.test_results[-1].output_tail if state.test_results else ""

        exec_signal = SkillSignal(
            user_request=target_task.description,
            touched_files=list(target_task.files_hint),
            diff_text=diff_str,
            test_exit_code=last_test_exit,
            test_output=last_test_out,
            consecutive_failures=target_task.attempts,
        )
        exec_skills = self.skills.evaluate(exec_signal, agent_role="executor")
        for act in exec_skills:
            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.SKILL_ACTIVATED,
                        workflow_id=state.workflow_id,
                        source="router",
                        payload={
                            "skill_id": act.skill.id,
                            "name": act.skill.name,
                            "agent": "executor",
                            "rationale": act.rationale,
                        },
                    )
                )

        effective_task = target_task
        if exec_skills:
            skill_prompt = format_skills_for_prompt(exec_skills)
            effective_task = target_task.model_copy(
                update={"description": f"{target_task.description}\n\n{skill_prompt}"}
            )

        context = builder.executor(
            task=effective_task,
            capabilities=self.executor.adapter.capabilities,
            optimization=self.executor.adapter.optimization,
            plan=state.plan,
            requirements=state.requirements,
            findings=findings,
        )

        result = await self.executor.execute(
            task=effective_task,
            context=context,
            plan=state.plan,
            requirements=state.requirements,
            findings=findings,
            max_steps=15,
        )

        state.execution_results.append(result)
        if result.tests:
            state.test_results.extend(result.tests)

        if result.status == ExecutionStatus.COMPLETED:
            target_task.status = TaskStatus.DONE
            next_ready = state.plan.ready_tasks()
            state.current_task_id = next_ready[0].id if next_ready else None
        else:
            target_task.status = TaskStatus.FAILED
            target_task.attempts += 1

    async def _handle_qa(self, decision: Decision, state: WorkflowState) -> None:
        scope = (
            QAScope.TARGETED if decision.action == OrchestratorAction.TARGETED_QA else QAScope.FULL
        )

        changed_files: list[str] = []
        for r in state.execution_results:
            for c in r.changes:
                if c.path not in changed_files:
                    changed_files.append(c.path)

        # Autonomous Skill Routing for QA
        git_repo = await self.runtime.env.git()
        diff_str = await git_repo.diff() if git_repo is not None else ""
        qa_signal = SkillSignal(
            user_request=state.user_request,
            touched_files=changed_files,
            diff_text=diff_str,
        )
        qa_skills = self.skills.evaluate(qa_signal, agent_role="qa")
        for act in qa_skills:
            if self.events is not None:
                await self.events.publish(
                    Event(
                        type=EventType.SKILL_ACTIVATED,
                        workflow_id=state.workflow_id,
                        source="router",
                        payload={
                            "skill_id": act.skill.id,
                            "name": act.skill.name,
                            "agent": "qa",
                            "rationale": act.rationale,
                        },
                    )
                )

        effective_reqs = state.requirements
        if qa_skills and effective_reqs:
            qa_skill_prompt = format_skills_for_prompt(qa_skills)
            effective_reqs = effective_reqs.model_copy(
                update={"summary": f"{effective_reqs.summary}\n\n{qa_skill_prompt}"}
            )

        builder = await self._build_context_builder()
        context = builder.qa(
            capabilities=self.qa.adapter.capabilities,
            optimization=self.qa.adapter.optimization,
            requirements=effective_reqs,
            changed_files=changed_files,
            plan=state.plan,
            test_results=state.test_results,
        )

        report = await self.qa.review(
            context=context,
            requirements=effective_reqs,
            plan=state.plan,
            changed_files=changed_files,
            execution_results=state.execution_results,
            scope=scope,
            max_steps=10,
        )

        state.qa_reports.append(report)
        if report.status == QAStatus.PASS:
            self.memory.record_sft_trace(
                timestamp=str(state.iteration),
                prompt=state.user_request,
                initial_code="",
                sliced_context=f"Files: {', '.join(changed_files)}",
                patch_diff=diff_str,
                verification_evidence=f"QA passed ({len(state.test_results)} test executions)",
            )
        elif self.events is not None:
            await self.events.publish(
                Event(
                    type=EventType.REWORK_REQUESTED,
                    workflow_id=state.workflow_id,
                    source="router",
                    payload={
                        "status": report.status.value,
                        "findings": [f.id for f in report.blocking_findings],
                        "summary": report.summary,
                    },
                )
            )

    async def _handle_rollback(self, decision: Decision, state: WorkflowState) -> None:
        restore = await self.checkpoint_mgr.rollback_to_latest(self.runtime.env, state.checkpoints)
        if restore and self.events is not None:
            await self.events.publish(
                Event(
                    type=EventType.ROLLBACK_PERFORMED,
                    workflow_id=state.workflow_id,
                    source="router",
                    payload={
                        "checkpoint_id": restore.checkpoint_id,
                        "changed_files": restore.changed_files,
                        "removed_files": restore.removed_files,
                    },
                )
            )
        if state.current_task_id and state.plan:
            state.plan.task(state.current_task_id).status = TaskStatus.PENDING
