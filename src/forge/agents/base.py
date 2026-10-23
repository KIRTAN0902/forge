"""Base agent abstractions and loop execution (spec §§4-6)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from forge.context.history import compact_history
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import (
    ContentPart,
    Message,
    ModelResponse,
    ReasoningDelta,
    StreamDone,
    TextDelta,
    ToolCall,
    ToolDefinition,
    Usage,
)
from forge.providers.client import ThinkingStreamParser
from forge.tools.base import ToolResult
from forge.tools.runtime import ToolRuntime

ToolResultHandler = Callable[[ToolResult], None]
ToolCallHandler = Callable[[ToolCall], None]
TokenHandler = Callable[[str], None]
ThinkingHandler = Callable[[str], None]


@dataclass
class AgentLoopOutcome:
    """Outcome of an interactive tool execution loop."""

    final_text: str
    steps: int
    results: list[ToolResult] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    tool_names: list[str] = field(default_factory=list)
    limit_reached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


@dataclass
class AgentChatTurn:
    """The result of an interactive conversational turn with an agent."""

    response_text: str
    handoff_requirements: str | None = None
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    tool_names: list[str] = field(default_factory=list)
    limit_reached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


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

    async def _generate(
        self,
        conversation: Sequence[Message],
        definitions: Sequence[ToolDefinition],
        on_token: TokenHandler | None,
        on_thinking: ThinkingHandler | None = None,
    ) -> ModelResponse:
        """One model turn, streaming real token deltas through ``on_token`` when possible.

        Streaming is only used when the model advertises streaming and speaks the native tool
        protocol; otherwise the (prompt-protocol) reply must be parsed in full and we fall back
        to a single non-streamed generation. Native reasoning events and inline reasoning tags
        are routed exclusively to ``on_thinking``.
        """
        if (
            (on_token is None and on_thinking is None)
            or not self.adapter.capabilities.streaming
            or self.adapter.tool_strategy() != "native"
        ):
            response = await self.adapter.generate(conversation, tools=list(definitions))
            _replay_response(response, on_token, on_thinking)
            return response

        parser = ThinkingStreamParser(on_token=on_token, on_thinking=on_thinking)
        done: ModelResponse | None = None
        async for event in self.adapter.stream(conversation, tools=list(definitions)):
            if isinstance(event, TextDelta):
                parser.feed_text(event.text)
            elif isinstance(event, ReasoningDelta):
                parser.feed_reasoning(event.text)
            elif isinstance(event, StreamDone):
                done = event.response
        parser.close()
        if done is None:  # pragma: no cover - provider contract guarantees StreamDone
            response = await self.adapter.generate(conversation, tools=list(definitions))
            _replay_response(response, on_token, on_thinking)
            return response
        return done

    async def _run_tool_loop(
        self,
        messages: Sequence[Message],
        *,
        allowed_tools: Sequence[str],
        max_steps: int = 15,
        compact_budget: int | None = None,
        on_tool_call: ToolCallHandler | None = None,
        on_tool_result: ToolResultHandler | None = None,
        on_token: TokenHandler | None = None,
        on_thinking: ThinkingHandler | None = None,
    ) -> AgentLoopOutcome:
        """Execute turns using the model and tool runtime until conclusion or max_steps.

        Token usage is accumulated from the real provider ``usage`` metadata returned by
        every model turn; ``on_tool_result`` is invoked as each real tool finishes so a UI
        can stream live tool output into its viewport, and ``on_token`` receives streamed
        text deltas so a UI can render the reply as it is generated.
        """
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

            response = await self._generate(
                conversation, definitions, on_token, on_thinking
            )
            outcome.usage = outcome.usage + response.usage
            conversation.append(response.message)

            if not response.tool_calls:
                outcome.final_text = response.text
                outcome.messages = conversation
                break

            outcome.tool_names.extend(call.name for call in response.tool_calls)
            if on_tool_call is not None:
                for call in response.tool_calls:
                    on_tool_call(call)
            results = await self.runtime.execute_many(
                response.tool_calls, actor=self.role, allowed=allowed
            )
            outcome.results.extend(results)
            if on_tool_result is not None:
                for result in results:
                    on_tool_result(result)
            conversation.extend(ToolRuntime.to_messages(response.tool_calls, results))
            outcome.messages = conversation
        else:
            outcome.limit_reached = True
            outcome.final_text = (
                f"Stopped at the {max_steps}-step tool limit before the agent concluded. "
                "Review the tool evidence above and continue with a narrower instruction."
            )

        return outcome

    async def chat(
        self,
        user_message: str | list[ContentPart],
        history: Sequence[Message] | None = None,
        *,
        project_context: str = "",
        max_tool_steps: int = 5,
        allowed_tools: Sequence[str] | None = None,
        on_tool_call: ToolCallHandler | None = None,
        on_tool_result: ToolResultHandler | None = None,
        on_token: TokenHandler | None = None,
        on_thinking: ThinkingHandler | None = None,
    ) -> AgentChatTurn:
        """Chat with the developer in this agent's specialization."""
        from forge.agents.prompts import (
            EXECUTOR_CHAT_SYSTEM_PROMPT,
            ORCHESTRATOR_CHAT_SYSTEM_PROMPT,
            PLANNER_CHAT_SYSTEM_PROMPT,
            QA_CHAT_SYSTEM_PROMPT,
        )
        from forge.tools.defaults import ROLE_TOOLSETS

        prompt_map = {
            "orchestrator": ORCHESTRATOR_CHAT_SYSTEM_PROMPT,
            "planner": PLANNER_CHAT_SYSTEM_PROMPT,
            "executor": EXECUTOR_CHAT_SYSTEM_PROMPT,
            "qa": QA_CHAT_SYSTEM_PROMPT,
        }

        system_content = prompt_map.get(self.role, ORCHESTRATOR_CHAT_SYSTEM_PROMPT)
        if project_context:
            system_content += f"\n\n[Active Workspace Context]:\n{project_context}"

        conversation: list[Message] = [Message.system(system_content)]
        if history:
            for m in history:
                if m.role != "system":
                    conversation.append(m)
        conversation.append(Message.user(user_message))

        allowed = allowed_tools if allowed_tools is not None else ROLE_TOOLSETS.get(self.role, ())
        limit_reached = False
        if self.runtime is not None and allowed:
            active_allowed = [t for t in allowed if t in self.runtime.tools]
            outcome = await self._run_tool_loop(
                conversation,
                allowed_tools=active_allowed,
                max_steps=max_tool_steps,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                on_token=on_token,
                on_thinking=on_thinking,
            )
            raw_text = outcome.final_text
            updated_messages = outcome.messages
            usage = outcome.usage
            tool_names = outcome.tool_names
            limit_reached = outcome.limit_reached
        else:
            resp = await self._generate(conversation, (), on_token, on_thinking)
            raw_text = resp.text
            updated_messages = [*list(conversation), resp.message]
            usage = resp.usage
            tool_names = []

        return AgentChatTurn(
            response_text=raw_text,
            messages=updated_messages,
            usage=usage,
            tool_names=tool_names,
            limit_reached=limit_reached,
        )


def _replay_response(
    response: ModelResponse,
    on_token: TokenHandler | None,
    on_thinking: ThinkingHandler | None,
) -> None:
    """Replay a non-streamed response through the same reasoning-safe callbacks."""
    if on_token is None and on_thinking is None:
        return
    parser = ThinkingStreamParser(on_token=on_token, on_thinking=on_thinking)
    parser.feed_reasoning(response.message.reasoning or "")
    parser.feed_text(response.message.text or "")
    parser.close()
