"""A deterministic in-process provider for tests, demos and offline development.

Responses are taken from a script in order. Each entry may be a ``ModelResponse``, a
plain string (an assistant text reply), or a callable that receives the request and
returns either. ``reject`` can simulate a server refusing a request shape.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from forge.models.provider import ModelProvider, ProviderError
from forge.models.types import (
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)

Responder = ModelResponse | str | Callable[[ModelRequest], "ModelResponse | str"]


class ScriptedProvider(ModelProvider):
    def __init__(
        self,
        script: Iterable[Responder] = (),
        *,
        name: str = "scripted",
        info: ModelInfo | None = None,
        reject: Callable[[ModelRequest], str | None] | None = None,
    ) -> None:
        self.name = name
        self._script: deque[Responder] = deque(script)
        self._info = info
        self._reject = reject
        self.requests: list[ModelRequest] = []
        self.closed = False

    def add(self, *responses: Responder) -> None:
        self._script.extend(responses)

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._reject is not None:
            reason = self._reject(request)
            if reason:
                raise ProviderError(f"{self.name}: HTTP 400: {reason}", status=400, body=reason)
        if not self._script:
            raise ProviderError(f"{self.name}: script exhausted")
        entry = self._script.popleft()
        result = entry(request) if callable(entry) else entry
        if isinstance(result, str):
            result = text_response(result)
        return result.model_copy(deep=True, update={"model": result.model or model})

    async def stream(self, model: str, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        response = await self.complete(model, request)
        words = response.text.split(" ") if response.text else []
        for index, word in enumerate(words):
            yield TextDelta(text=word if index == len(words) - 1 else word + " ")
        for index, call in enumerate(response.tool_calls):
            yield ToolCallDelta(
                index=index, id=call.id, name=call.name, arguments_delta=call.raw_arguments
            )
        yield StreamDone(response=response)

    async def describe_model(self, model: str) -> ModelInfo | None:
        return self._info

    async def aclose(self) -> None:
        self.closed = True


def text_response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(text),
        usage=usage or Usage(input_tokens=10, output_tokens=max(1, len(text) // 4)),
    )


def tool_call_response(
    name: str, arguments: dict[str, Any], *, call_id: str = "call_1"
) -> ModelResponse:
    import json

    return ModelResponse(
        message=Message.assistant(
            None,
            [
                ToolCall(
                    id=call_id, name=name, arguments=arguments, raw_arguments=json.dumps(arguments)
                )
            ],
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
    )
