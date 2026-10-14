"""A minimal tool-using agent loop for exercising the model + tool runtimes together.

Phase 4 builds the real agents; this is just enough to drive a model against a
repository through :class:`ToolRuntime`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from forge.models import Message, ModelAdapter
from forge.tools import ToolResult, ToolRuntime


@dataclass
class LoopOutcome:
    final_text: str
    steps: int
    results: list[ToolResult] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)


async def run_tool_loop(
    adapter: ModelAdapter,
    runtime: ToolRuntime,
    messages: Sequence[Message],
    *,
    tool_names: Sequence[str] | None = None,
    max_steps: int = 15,
) -> LoopOutcome:
    conversation = list(messages)
    definitions = runtime.definitions(tool_names)
    allowed = [d.name for d in definitions]
    outcome = LoopOutcome(final_text="", steps=0, transcript=conversation)
    for step in range(1, max_steps + 1):
        outcome.steps = step
        response = await adapter.generate(conversation, tools=definitions)
        conversation.append(response.message)
        if not response.tool_calls:
            outcome.final_text = response.text
            break
        results = await runtime.execute_many(response.tool_calls, actor="executor", allowed=allowed)
        outcome.results.extend(results)
        conversation.extend(ToolRuntime.to_messages(response.tool_calls, results))
    return outcome
