"""Planner specialist agent (spec §4.1).

Responsible for requirements elicitation and dependency-ordered task graph planning.
The planner investigates the codebase using read-only tools if needed, but never modifies code.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.agents.base import BaseAgent
from forge.agents.prompts import PLANNER_SYSTEM_PROMPT
from forge.context.builder import BuiltContext
from forge.context.render import render_plan, render_requirements
from forge.core.contracts.plan import Plan
from forge.core.contracts.requirements import RequirementContract
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message
from forge.tools.defaults import ROLE_TOOLSETS
from forge.tools.runtime import ToolRuntime


@dataclass
class PlannerResult:
    """The complete structured output from the planning phase."""

    requirements: RequirementContract
    plan: Plan


class PlannerAgent(BaseAgent):
    """Senior software architect and problem solver."""

    def __init__(
        self,
        adapter: ModelAdapter,
        runtime: ToolRuntime | None = None,
        *,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        super().__init__(
            role="planner",
            adapter=adapter,
            runtime=runtime,
            events=events,
            workflow_id=workflow_id,
        )

    async def plan(
        self,
        request: str,
        *,
        context: BuiltContext | None = None,
        requirements: RequirementContract | None = None,
        existing_plan: Plan | None = None,
        investigate: bool = False,
        max_investigation_steps: int = 5,
    ) -> PlannerResult:
        """Create or revise requirements and task graph for the given request."""
        investigation_notes = ""
        if investigate and self.runtime is not None and max_investigation_steps > 0:
            investigation_notes = await self._investigate(
                request, context=context, max_steps=max_investigation_steps
            )

        effective_requirements = requirements
        if effective_requirements is None:
            effective_requirements = await self._generate_requirements(
                request, context=context, notes=investigation_notes
            )
            await self._emit(
                EventType.REQUIREMENTS_DEFINED,
                summary=effective_requirements.summary,
                criteria_count=len(effective_requirements.acceptance_criteria),
                requirements=effective_requirements.model_dump(),
            )

        plan = await self._generate_plan(
            request,
            requirements=effective_requirements,
            context=context,
            existing_plan=existing_plan,
            notes=investigation_notes,
        )

        event_type = EventType.PLAN_REVISED if existing_plan else EventType.PLAN_CREATED
        await self._emit(
            event_type,
            goal=plan.goal,
            tasks_count=len(plan.tasks),
            plan=plan.model_dump(),
        )

        return PlannerResult(requirements=effective_requirements, plan=plan)

    async def _investigate(self, request: str, context: BuiltContext | None, max_steps: int) -> str:
        """Run read-only tools to explore relevant files before planning."""
        messages: list[Message] = [
            Message.system(PLANNER_SYSTEM_PROMPT),
        ]
        if context is not None:
            messages.append(context.to_message())
        messages.append(
            Message.user(
                f"Goal to plan:\n{request}\n\n"
                "Investigate the codebase using read-only tools to understand current interfaces, "
                "dependencies, and files that need to be changed or created. Summarize when done."
            )
        )
        outcome = await self._run_tool_loop(
            messages,
            allowed_tools=ROLE_TOOLSETS["planner"],
            max_steps=max_steps,
        )
        return outcome.final_text

    async def _generate_requirements(
        self, request: str, context: BuiltContext | None, notes: str = ""
    ) -> RequirementContract:
        messages: list[Message] = [
            Message.system(PLANNER_SYSTEM_PROMPT),
        ]
        if context is not None:
            messages.append(context.to_message())
        if notes:
            messages.append(Message.user(f"Investigation findings:\n{notes}"))

        prompt = (
            f"Define a formal RequirementContract for the following user request:\n\n"
            f"{request}\n\n"
            "Requirements:\n"
            "- Functional requirements must use IDs with format 'FR-001', 'FR-002', etc.\n"
            "- Non-functional requirements (if any) must use IDs with format 'NFR-001', etc.\n"
            "- Every acceptance criterion must use format 'AC-001', etc., and specify a "
            "verification method (automated_test, command, inspection, manual) and "
            "link back to requirement_ids.\n"
            "- Include assumptions and out-of-scope items if appropriate."
        )
        messages.append(Message.user(prompt))

        return await self.adapter.structured(messages, RequirementContract)

    async def _generate_plan(
        self,
        request: str,
        requirements: RequirementContract,
        context: BuiltContext | None,
        existing_plan: Plan | None,
        notes: str = "",
    ) -> Plan:
        messages: list[Message] = [
            Message.system(PLANNER_SYSTEM_PROMPT),
        ]
        if context is not None:
            messages.append(context.to_message())
        if notes:
            messages.append(Message.user(f"Investigation findings:\n{notes}"))

        rendered_reqs = render_requirements(requirements)
        replan_context = ""
        if existing_plan is not None:
            replan_context = (
                f"\n\nAn existing plan exists and requires revision:\n"
                f"{render_plan(existing_plan)}\n"
                "Please replan to address defects or update tasks accordingly."
            )

        prompt = (
            f"User goal:\n{request}\n\n"
            f"Approved requirements:\n{rendered_reqs}{replan_context}\n\n"
            "Construct a dependency-ordered task graph (Plan) covering these requirements:\n"
            "1. Each Task must have a unique id (e.g. 'TASK-1', 'TASK-2'), a clear description, "
            "and list prerequisite task IDs in 'dependencies'.\n"
            "2. Tasks must form a Directed Acyclic Graph (no cycles, no self-dependencies).\n"
            "3. Link relevant acceptance criterion IDs (e.g. ['AC-001']) to each task in "
            "'acceptance_criteria'.\n"
            "4. Specify expected touched files in 'files_hint'.\n"
            "5. The overall plan must list covered 'acceptance_criteria' and identified 'risks'."
        )
        messages.append(Message.user(prompt))

        return await self.adapter.structured(messages, Plan)
