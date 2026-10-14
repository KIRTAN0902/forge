"""Provider-neutral request/response types.

These mirror what modern chat APIs have in common (messages, tool calls, structured
output, streaming, usage) without encoding any one provider's wire format. Anything a
provider supports beyond this travels in ``GenerationParams.extra`` (native passthrough)
and comes back in ``ModelResponse.raw``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextPart(_Model):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    type: Literal["image"] = "image"
    url: str = Field(description="http(s) URL or data: URI")


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ToolCall(_Model):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = Field(
        default=None, description="Set when the model emitted arguments that are not valid JSON."
    )


class Message(_Model):
    role: Role
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = Field(
        default=None, description="Reasoning text, when the provider exposes it."
    )

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=text)

    @classmethod
    def user(cls, content: str | list[ContentPart]) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, text: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=text, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=call.id, name=call.name)

    @property
    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def has_images(self) -> bool:
        return isinstance(self.content, list) and any(
            isinstance(p, ImagePart) for p in self.content
        )


class ToolDefinition(_Model):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the arguments object.",
    )


ToolChoice = Literal["auto", "none", "required"] | str
"""``auto`` / ``none`` / ``required``, or the name of a specific tool to force."""


class ResponseFormat(_Model):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GenerationParams(_Model):
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] | None = None
    seed: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-native request fields, passed through unchanged.",
    )

    def merged(self, override: GenerationParams | None) -> GenerationParams:
        """Fields set on ``override`` win; ``extra`` dicts are merged."""
        if override is None:
            return self.model_copy(deep=True)
        data = self.model_dump(exclude={"extra"})
        data.update(override.model_dump(exclude={"extra"}, exclude_unset=True))
        return GenerationParams(**data, extra={**self.extra, **override.extra})


class ModelRequest(_Model):
    messages: list[Message] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None
    json_mode: bool = False
    params: GenerationParams = Field(default_factory=GenerationParams)


class Usage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cost_usd=cost,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class ModelResponse(_Model):
    message: Message
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_s: float | None = None
    raw: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls


# -- streaming ---------------------------------------------------------------------------


class TextDelta(_Model):
    type: Literal["text"] = "text"
    text: str


class ReasoningDelta(_Model):
    type: Literal["reasoning"] = "reasoning"
    text: str


class ToolCallDelta(_Model):
    type: Literal["tool_call"] = "tool_call"
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class StreamDone(_Model):
    """Always the last stream event; carries the fully assembled response."""

    type: Literal["done"] = "done"
    response: ModelResponse


StreamEvent = Annotated[
    TextDelta | ReasoningDelta | ToolCallDelta | StreamDone, Field(discriminator="type")
]


class ModelInfo(_Model):
    """What a provider reports about a model (used for capability detection)."""

    id: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    tool_calling: bool | None = None
    structured_output: bool | None = None
    json_mode: bool | None = None
    reasoning: bool | None = None
    vision: bool | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None
