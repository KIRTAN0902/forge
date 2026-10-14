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
from typing import Any

from forge.agents.base import AgentChatTurn, BaseAgent
from forge.agents.prompts import ORCHESTRATOR_CHAT_SYSTEM_PROMPT
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message
from forge.tools.defaults import READ_ONLY_TOOLS
from forge.tools.runtime import ToolRuntime

HANDOFF_PATTERN = re.compile(r"```(?:forge-handoff|handoff)\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)

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
)


@dataclass
class OrchestratorChatTurn(AgentChatTurn):
    """The result of an interactive conversational turn with the Orchestrator."""

    response_text: str
    handoff_requirements: str | None = None
    messages: list[Message] = field(default_factory=list)


class OrchestratorAgent(BaseAgent):
    """Senior Technical Lead and Conversational Orchestrator."""

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
        user_message: str,
        history: Sequence[Message] | None = None,
        *,
        project_context: str = "",
        max_tool_steps: int = 5,
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

        # 4. Run interactive tool loop (read-only tools available for inspecting repo)
        if self.runtime is not None:
            allowed = [t for t in READ_ONLY_TOOLS if t in self.runtime.tools]
            outcome = await self._run_tool_loop(
                conversation,
                allowed_tools=allowed,
                max_steps=max_tool_steps,
            )
            raw_text = outcome.final_text
            updated_messages = outcome.messages
        else:
            resp = await self.adapter.generate(conversation)
            raw_text = resp.text
            updated_messages = list(conversation) + [resp.message]

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
            norm_input = user_message.strip().lower()
            if any(norm_input == trig or norm_input.startswith(f"{trig} ") for trig in EXECUTION_CONFIRM_TRIGGERS):
                # Synthesize requirements from recent discussion
                handoff_requirements = self._synthesize_from_history(history, user_message)

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
        )

    def _synthesize_from_history(
        self, history: Sequence[Message] | None, user_trigger: str
    ) -> str:
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
