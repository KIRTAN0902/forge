"""The agent-facing model API.

Agents only ever hold a :class:`ModelAdapter`. It merges profile/role/call parameters,
uses the model's native capabilities when it has them, transparently falls back to
prompt-level protocols when it does not, and records usage. Swapping the model behind
a role therefore never requires changing agent code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from forge.core.errors import ModelCapabilityError, ModelOutputError
from forge.core.logging import get_logger
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.capabilities import ModelCapabilities, OptimizationConfig
from forge.models.profile import ModelProfile
from forge.models.protocols import (
    extract_json,
    parse_text_tool_calls,
    render_repair_message,
    render_schema_instructions,
    render_tool_instructions,
    split_reasoning,
    to_prompt_protocol,
    with_system_suffix,
)
from forge.models.provider import ModelProvider, ProviderError
from forge.models.types import (
    FinishReason,
    GenerationParams,
    Message,
    ModelRequest,
    ModelResponse,
    ResponseFormat,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolDefinition,
)
from forge.models.usage import UsageTracker

T = TypeVar("T", bound=BaseModel)

StructuredStrategy = Literal["native", "json_mode", "prompt"]
ToolStrategy = Literal["native", "prompt"]

_DOWNGRADE: dict[StructuredStrategy, StructuredStrategy | None] = {
    "native": "json_mode",
    "json_mode": "prompt",
    "prompt": None,
}

log = get_logger("models.adapter")


class ModelAdapter:
    def __init__(
        self,
        *,
        profile_name: str,
        profile: ModelProfile,
        provider: ModelProvider,
        capabilities: ModelCapabilities,
        role: str = "default",
        params: GenerationParams | None = None,
        optimization: OptimizationConfig | None = None,
        tracker: UsageTracker | None = None,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.provider = provider
        self.capabilities = capabilities
        self.role = role
        self.params = params or profile.params
        self.optimization = optimization or profile.optimization
        self.tracker = tracker or UsageTracker()
        self.events = events
        self.workflow_id = workflow_id
        self._structured_floor: StructuredStrategy | None = None

    @property
    def model(self) -> str:
        return self.profile.model

    def describe(self) -> str:
        return f"{self.profile_name} ({self.profile.provider}:{self.profile.model}) as {self.role}"

    # -- strategy selection -------------------------------------------------------------

    def tool_strategy(self) -> ToolStrategy:
        configured = self.optimization.tool_call_strategy
        if configured == "auto":
            return "native" if self.capabilities.tool_calling else "prompt"
        return configured

    def structured_strategy(self) -> StructuredStrategy:
        configured = self.optimization.structured_output_strategy
        if configured == "auto":
            if self.capabilities.structured_output:
                strategy: StructuredStrategy = "native"
            elif self.capabilities.json_mode:
                strategy = "json_mode"
            else:
                strategy = "prompt"
        else:
            strategy = configured
        # A server that rejected a stronger strategy earlier stays downgraded.
        order: list[StructuredStrategy] = ["native", "json_mode", "prompt"]
        if self._structured_floor and order.index(self._structured_floor) > order.index(strategy):
            return self._structured_floor
        return strategy

    # -- generation ---------------------------------------------------------------------

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: str | None = None,
        params: GenerationParams | None = None,
    ) -> ModelResponse:
        """One model turn. Tool calls (native or prompt-protocol) come back as ToolCalls."""
        self._check_inputs(messages)
        tool_list = list(tools)
        merged = self.params.merged(params)

        if tool_list and tool_choice != "none" and self.tool_strategy() == "prompt":
            prompt_messages = with_system_suffix(
                to_prompt_protocol(list(messages)),
                render_tool_instructions(tool_list, tool_choice),
            )
            request = ModelRequest(messages=prompt_messages, params=merged)
            response = await self._call(request, strategy="prompt_tools")
            remaining, parsed = parse_text_tool_calls(response.text, tool_list)
            # Keep any native calls the server returned anyway; never drop a call.
            calls = [*response.tool_calls, *parsed]
            response.message = response.message.model_copy(
                update={"content": remaining or None, "tool_calls": calls}
            )
            if calls:
                response.finish_reason = FinishReason.TOOL_CALLS
            return response

        if tool_choice == "none":
            tool_list = []
        request = ModelRequest(
            messages=self._history(messages),
            tools=tool_list,
            tool_choice=tool_choice if tool_list else None,
            parallel_tool_calls=self._parallel_hint() if tool_list else None,
            params=merged,
        )
        response = await self._call(request, strategy="native_tools" if tool_list else "text")
        return self._recover_text_tool_calls(response, tool_list)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: str | None = None,
        params: GenerationParams | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn. Falls back to a single chunk when streaming is unavailable
        or when prompt-protocol tool calls must be parsed from the complete reply."""
        self._check_inputs(messages)
        tool_list = list(tools)
        prompt_tools = bool(tool_list) and self.tool_strategy() == "prompt"
        if not self.capabilities.streaming or prompt_tools:
            response = await self.generate(
                messages, tools=tool_list, tool_choice=tool_choice, params=params
            )
            if response.text:
                yield TextDelta(text=response.text)
            yield StreamDone(response=response)
            return

        request = ModelRequest(
            messages=self._history(messages),
            tools=tool_list,
            tool_choice=tool_choice if tool_list else None,
            parallel_tool_calls=self._parallel_hint() if tool_list else None,
            params=self.params.merged(params),
        )
        async for event in self.provider.stream(self.profile.model, request):
            if isinstance(event, StreamDone):
                response = self._recover_text_tool_calls(
                    self._normalize(event.response), tool_list
                )
                await self._record(response, strategy="stream")
                yield StreamDone(response=response)
            else:
                yield event

    async def structured(
        self,
        messages: Sequence[Message],
        output_type: type[T],
        *,
        params: GenerationParams | None = None,
    ) -> T:
        """Return a validated ``output_type`` instance, using the strongest strategy the
        model supports and repairing invalid output with feedback round-trips."""
        self._check_inputs(messages)
        schema = output_type.model_json_schema()
        merged = self.params.merged(params)
        conversation = self._history(messages)
        repairs_left = self.optimization.max_repair_attempts
        last_error = "no attempt made"

        while True:
            strategy = self.structured_strategy()
            request = ModelRequest(
                messages=with_system_suffix(
                    conversation, render_schema_instructions(schema, native=strategy == "native")
                ),
                response_format=(
                    ResponseFormat(name=_schema_name(output_type), schema_=schema)
                    if strategy == "native"
                    else None
                ),
                json_mode=strategy == "json_mode",
                params=merged,
            )
            try:
                response = await self._call(request, strategy=f"structured_{strategy}")
            except ProviderError as exc:
                fallback = _DOWNGRADE[strategy]
                if exc.is_request_rejected and fallback is not None:
                    log.warning(
                        "%s rejected %s structured output (%s); falling back to %s",
                        self.describe(),
                        strategy,
                        exc,
                        fallback,
                    )
                    self._structured_floor = fallback
                    continue
                raise

            try:
                return output_type.model_validate(extract_json(response.text))
            except (ValueError, ValidationError) as exc:
                last_error = _describe_error(exc)
                if response.finish_reason == FinishReason.LENGTH:
                    last_error += " (the reply was cut off by the output token limit)"
            if repairs_left <= 0:
                raise ModelOutputError(
                    f"{self.describe()} did not produce a valid {output_type.__name__}: "
                    f"{last_error}"
                )
            repairs_left -= 1
            conversation = [
                *conversation,
                Message.assistant(response.text or "(empty reply)"),
                Message.user(render_repair_message(last_error)),
            ]

    # -- internals ----------------------------------------------------------------------

    def _recover_text_tool_calls(
        self, response: ModelResponse, tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        """Recover text-protocol calls from a reply that advertised native tools.

        Some OpenAI-compatible servers accept the ``tools`` field but stream the model's
        training-format ``<tool_call>`` block instead of native ``delta.tool_calls``. Without
        this recovery the agent only sees prose and never executes a write.
        """
        if not tools or response.tool_calls:
            return response
        text = response.text
        if "tool_call>" not in text and '"arguments"' not in text:
            return response
        remaining, calls = parse_text_tool_calls(text, list(tools))
        if not calls:
            return response
        response.message = response.message.model_copy(
            update={"content": remaining or None, "tool_calls": calls}
        )
        response.finish_reason = FinishReason.TOOL_CALLS
        return response

    def _history(self, messages: Sequence[Message]) -> list[Message]:
        """Conversation as this model can accept it.

        A model without native tool support must never receive ``tool``-role messages
        or assistant ``tool_calls``, even on turns that offer no tools (e.g. a final
        structured answer after a tool loop), so its history uses the text protocol.
        """
        if self.tool_strategy() == "prompt":
            return to_prompt_protocol(list(messages))
        return list(messages)

    def _parallel_hint(self) -> bool | None:
        batching = self.optimization.tool_batching
        if not self.capabilities.parallel_tool_calls:
            return None  # don't send a parameter the model may not accept
        return batching != "sequential"

    def _check_inputs(self, messages: Sequence[Message]) -> None:
        if not messages:
            raise ValueError("at least one message is required")
        if not self.capabilities.vision and any(m.has_images for m in messages):
            model_name = (self.profile.model or "").lower()
            vision_names = (
                "gpt-4o",
                "gpt-4-turbo",
                "claude-3",
                "claude-3-5",
                "claude-3.7",
                "gemini",
                "vision",
                "vl",
                "pixtral",
                "llava",
                "qwen-vl",
            )
            if any(vn in model_name for vn in vision_names):
                self.capabilities = self.capabilities.model_copy(update={"vision": True})
            else:
                raise ModelCapabilityError(f"{self.describe()} does not accept images")

    async def _call(self, request: ModelRequest, *, strategy: str) -> ModelResponse:
        response = await self.provider.complete(self.profile.model, request)
        response = self._normalize(response)
        await self._record(response, strategy=strategy)
        return response

    def _normalize(self, response: ModelResponse) -> ModelResponse:
        text = response.message.text
        answer, inline_reasoning = split_reasoning(text)
        if inline_reasoning or answer != text:
            reasoning = "\n\n".join(r for r in (response.message.reasoning, inline_reasoning) if r)
            response.message = response.message.model_copy(
                update={"content": answer or None, "reasoning": reasoning or None}
            )
        if response.usage.cost_usd is None and self.profile.pricing is not None:
            response.usage = response.usage.model_copy(
                update={
                    "cost_usd": self.profile.pricing.cost(
                        response.usage.input_tokens, response.usage.output_tokens
                    )
                }
            )
        return response

    async def _record(self, response: ModelResponse, *, strategy: str) -> None:
        self.tracker.record(
            profile=self.profile_name,
            role=self.role,
            usage=response.usage,
            latency_s=response.latency_s,
        )
        if self.events is not None:
            await self.events.emit(
                EventType.MODEL_CALLED,
                workflow_id=self.workflow_id,
                source=self.role,
                profile=self.profile_name,
                model=self.profile.model,
                strategy=strategy,
                finish_reason=str(response.finish_reason),
                tool_calls=len(response.tool_calls),
                usage=response.usage.model_dump(),
                latency_s=response.latency_s,
            )


def _schema_name(output_type: type[BaseModel]) -> str:
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in output_type.__name__)
    return name[:64] or "output"


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()[:10]
        ]
        return "schema validation failed: " + "; ".join(problems)
    return str(exc)
