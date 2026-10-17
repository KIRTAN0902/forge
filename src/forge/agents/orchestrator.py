"""Orchestrator specialist agent (spec §§4-8).

Responsible for:
1. Conversational brainstorming, technical leadership, and developer pairing.
2. Codebase inspection using read-only tools to answer developer questions.
3. Synthesizing discussed requirements and handing them off to the Planning Agent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from forge.agents.base import AgentChatTurn, BaseAgent, ToolResultHandler
from forge.agents.prompts import ORCHESTRATOR_CHAT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from forge.context.builder import BuiltContext
from forge.context.render import render_plan, render_requirements
from forge.core.contracts.plan import Plan
from forge.core.contracts.requirements import RequirementContract
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import ContentPart, Message, Usage
from forge.tools.defaults import READ_ONLY_TOOLS, ROLE_TOOLSETS
from forge.tools.runtime import ToolRuntime

HANDOFF_PATTERN = re.compile(
    r"```(?:forge-handoff|handoff)\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE
)

EXECUTION_CONFIRM_TRIGGERS = (
    "build it",
    "let's build it",
    "lets build it",
    "implement it",
    "let's implement",
    "lets implement",
    "go ahead",
    "proceed",
    "start implementation",
    "execute the plan",
    "do it",
    "yes build it",
    "yes implement it",
    "begin",
    "orchestrator begin",
    "start",
    "let's begin",
    "lets begin",
    "let's start",
    "lets start",
    "begin implementation",
    "start now",
    "execute",
)


@dataclass
class OrchestratorChatTurn(AgentChatTurn):
    """The result of an interactive conversational turn with the Orchestrator."""

    response_text: str
    handoff_requirements: str | None = None
    messages: list[Message] = field(default_factory=list)


@dataclass
class PlannerResult:
    """The complete structured output from the planning phase."""

    requirements: RequirementContract
    plan: Plan


class OrchestratorAgent(BaseAgent):
    """Senior Technical Lead, Architecture Planner, and Conversational Orchestrator."""

    def __init__(
        self,
        adapter: ModelAdapter,
        runtime: ToolRuntime | None = None,
        *,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        super().__init__(
            role="orchestrator",
            adapter=adapter,
            runtime=runtime,
            events=events,
            workflow_id=workflow_id,
        )

    async def chat(
        self,
        user_message: str | list[ContentPart],
        history: Sequence[Message] | None = None,
        *,
        project_context: str = "",
        max_tool_steps: int = 5,
        allowed_tools: Sequence[str] | None = None,
        on_tool_result: ToolResultHandler | None = None,
    ) -> OrchestratorChatTurn:
        """Chat with the developer to brainstorm or synthesize requirements for planning."""
        conversation: list[Message] = []

        # 1. System prompt with workspace context
        system_content = ORCHESTRATOR_CHAT_SYSTEM_PROMPT
        if project_context:
            system_content += f"\n\n[Active Workspace Context]:\n{project_context}"

        conversation.append(Message.system(system_content))

        # 2. Append prior conversation history (excluding previous system messages)
        if history:
            for m in history:
                if m.role != "system":
                    conversation.append(m)

        # 3. Append current user message
        conversation.append(Message.user(user_message))

        usage = Usage()
        tool_names: list[str] = []

        # 4. Run interactive tool loop (read-only tools available for inspecting repo)
        if self.runtime is not None:
            requested = allowed_tools if allowed_tools is not None else READ_ONLY_TOOLS
            allowed = [t for t in requested if t in self.runtime.tools]
            outcome = await self._run_tool_loop(
                conversation,
                allowed_tools=allowed,
                max_steps=max_tool_steps,
                on_tool_result=on_tool_result,
            )
            raw_text = outcome.final_text
            updated_messages = outcome.messages
            usage = outcome.usage
            tool_names = outcome.tool_names
        else:
            resp = await self.adapter.generate(conversation)
            raw_text = resp.text
            updated_messages = [*list(conversation), resp.message]
            usage = resp.usage

        # 5. Extract handoff block if present
        handoff_match = HANDOFF_PATTERN.search(raw_text)
        handoff_requirements: str | None = None
        clean_text = raw_text

        if handoff_match:
            handoff_requirements = handoff_match.group(1).strip()
            # Remove the handoff block from user-facing text
            clean_text = HANDOFF_PATTERN.sub("", raw_text).strip()
        else:
            # Check if user message was an explicit confirmation to build what was brainstormed
            from forge.models.types import TextPart

            text_msg = (
                user_message
                if isinstance(user_message, str)
                else "".join(p.text for p in user_message if isinstance(p, TextPart))
            )
            norm_input = text_msg.strip().lower()
            if any(
                norm_input == trig or norm_input.startswith(f"{trig} ")
                for trig in EXECUTION_CONFIRM_TRIGGERS
            ):
                # Synthesize requirements from recent discussion
                handoff_requirements = self._synthesize_from_history(history, text_msg)

        if handoff_requirements:
            await self._emit(
                EventType.DECISION_MADE,
                action="handoff_to_planner",
                rationale="User confirmed implementation or direct build instruction given.",
                task_ids=[],
            )

        return OrchestratorChatTurn(
            response_text=clean_text,
            handoff_requirements=handoff_requirements,
            messages=updated_messages,
            usage=usage,
            tool_names=tool_names,
        )

    def _synthesize_from_history(self, history: Sequence[Message] | None, user_trigger: str) -> str:
        """Fallback synthesis when user confirms implementation without model tag."""
        recent_turns: list[str] = []
        if history:
            for m in history[-6:]:
                if m.role in {"user", "assistant"} and m.text:
                    prefix = "User" if m.role == "user" else "Orchestrator"
                    recent_turns.append(f"{prefix}: {m.text[:400]}")

        context_str = "\n".join(recent_turns)
        return (
            f"User confirmed implementation: {user_trigger}\n\n"
            f"Context from brainstorming session with Orchestrator:\n{context_str}"
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
        allowed = ROLE_TOOLSETS.get("orchestrator", READ_ONLY_TOOLS)
        outcome = await self._run_tool_loop(
            messages,
            allowed_tools=allowed,
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
