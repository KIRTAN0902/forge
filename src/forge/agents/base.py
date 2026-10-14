"""Base agent abstractions and loop execution (spec §§4-6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from forge.context.history import compact_history
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message
from forge.tools.base import ToolResult
from forge.tools.runtime import ToolRuntime


@dataclass
class AgentLoopOutcome:
    """Outcome of an interactive tool execution loop."""

    final_text: str
    steps: int
    results: list[ToolResult] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


class BaseAgent:
    """Base class for specialist agents."""

    def __init__(
        self,
        *,
        role: str,
        adapter: ModelAdapter,
        runtime: ToolRuntime | None = None,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self.role = role
        self.adapter = adapter
        self.runtime = runtime
        self.events = events or adapter.events
        self.workflow_id = workflow_id or adapter.workflow_id

    async def _emit(self, event_type: EventType, **payload: Any) -> None:
        if self.events is not None:
            await self.events.publish(
                Event(
                    type=event_type,
                    workflow_id=self.workflow_id,
                    source=f"agent.{self.role}",
                    payload=payload,
                )
            )

    async def _run_tool_loop(
        self,
        messages: Sequence[Message],
        *,
        allowed_tools: Sequence[str],
        max_steps: int = 15,
        compact_budget: int | None = None,
    ) -> AgentLoopOutcome:
        """Execute turns using the model and tool runtime until conclusion or max_steps."""
        if self.runtime is None:
            raise RuntimeError(f"Agent {self.role!r} requires a ToolRuntime to execute tools.")

        conversation = list(messages)
        definitions = self.runtime.definitions(allowed_tools)
        allowed = [d.name for d in definitions]
        outcome = AgentLoopOutcome(final_text="", steps=0, messages=conversation)

        budget = compact_budget or int(self.adapter.capabilities.context_window * 0.8)

        for step in range(1, max_steps + 1):
            outcome.steps = step
            if len(conversation) > 4:
                conversation = compact_history(conversation, budget)

            response = await self.adapter.generate(conversation, tools=definitions)
            conversation.append(response.message)

            if not response.tool_calls:
                outcome.final_text = response.text
                outcome.messages = conversation
                break

            results = await self.runtime.execute_many(
                response.tool_calls, actor=self.role, allowed=allowed
            )
            outcome.results.extend(results)
            conversation.extend(ToolRuntime.to_messages(response.tool_calls, results))
            outcome.messages = conversation
        else:
            outcome.final_text = conversation[-1].text or "Reached maximum step limit."

        return outcome
