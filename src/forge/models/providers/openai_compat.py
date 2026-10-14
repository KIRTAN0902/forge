"""Provider for any OpenAI-compatible Chat Completions endpoint.

Covers OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp server, and the
OpenAI-compatible endpoints of GLM, Qwen, DeepSeek and others.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from forge.core.logging import get_logger
from forge.models.profile import ProviderConfig
from forge.models.provider import ModelProvider, ProviderError
from forge.models.types import (
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    Role,
    StreamDone,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)

log = get_logger("models.openai_compat")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        name: str = "openai_compatible",
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.name = name
        self.config = config
        headers = {"Content-Type": "application/json", **config.headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 15.0)),
            transport=transport,
        )
        self._sleep = sleep
        self._models_cache: list[dict[str, Any]] | None = None

    # -- public API ----------------------------------------------------------------------

    async def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        payload = self.build_payload(model, request, stream=False)
        started = time.perf_counter()
        response = await self._post_with_retries("chat/completions", payload)
        data = _json_or_error(response)
        result = parse_completion(data)
        result.latency_s = time.perf_counter() - started
        return result

    async def stream(self, model: str, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(model, request, stream=True)
        started = time.perf_counter()
        assembler = _StreamAssembler()
        async with self._open_stream("chat/completions", payload) as response:
            async for line in response.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is None:
                    continue
                if chunk == "[DONE]":
                    break
                for event in assembler.feed(chunk):
                    yield event
        final = assembler.build()
        final.latency_s = time.perf_counter() - started
        yield StreamDone(response=final)

    async def list_models(self) -> list[ModelInfo]:
        return [_model_info(entry) for entry in await self._model_entries()]

    async def describe_model(self, model: str) -> ModelInfo | None:
        info: ModelInfo | None = None
        try:
            for entry in await self._model_entries():
                if entry.get("id") == model:
                    info = _model_info(entry)
                    break
        except ProviderError as exc:
            log.debug("model listing unavailable: %s", exc)
        if info is None or info.context_window is None:
            ollama = await self._describe_ollama(model)
            if ollama is not None:
                return ollama
        return info

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- request building ----------------------------------------------------------------

    def build_payload(self, model: str, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_encode_message(m) for m in request.messages],
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
            if request.tool_choice is not None:
                payload["tool_choice"] = _encode_tool_choice(request.tool_choice)
            if request.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = request.parallel_tool_calls

        if request.response_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_format.name,
                    "schema": request.response_format.schema_,
                    "strict": request.response_format.strict,
                },
            }
        elif request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        params = request.params
        for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
            value = getattr(params, key)
            if value is not None:
                payload[key] = value
        if params.reasoning_effort is not None:
            style = self.config.reasoning_style
            if style == "openai":
                payload["reasoning_effort"] = params.reasoning_effort
            elif style == "openrouter":
                payload["reasoning"] = (
                    {"enabled": False}
                    if params.reasoning_effort == "none"
                    else {"effort": params.reasoning_effort}
                )
        if stream:
            payload["stream"] = True
            if self.config.stream_usage:
                payload["stream_options"] = {"include_usage": True}
        payload.update(params.extra)  # native passthrough wins
        return payload

    # -- transport -----------------------------------------------------------------------

    async def _post_with_retries(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise ProviderError(f"{self.name}: request failed: {exc!r}") from exc
                await self._backoff(attempt, None)
                attempt += 1
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self.config.max_retries:
                await self._backoff(attempt, response.headers.get("retry-after"))
                attempt += 1
                continue
            if response.status_code >= 400:
                raise _status_error(self.name, response.status_code, response.text)
            return response

    def _open_stream(self, path: str, payload: dict[str, Any]) -> _StreamContext:
        return _StreamContext(self, path, payload)

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(0.5 * 2**attempt, 20.0) + random.uniform(0, 0.25)
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, min(float(retry_after), 60.0))
        log.warning("%s: retrying in %.1fs (attempt %d)", self.name, delay, attempt + 1)
        await self._sleep(delay)

    async def _model_entries(self) -> list[dict[str, Any]]:
        if self._models_cache is None:
            response = await self._post_or_get("models")
            data = _json_or_error(response)
            entries = data.get("data", []) if isinstance(data, dict) else []
            self._models_cache = [e for e in entries if isinstance(e, dict)]
        return self._models_cache

    async def _post_or_get(self, path: str) -> httpx.Response:
        try:
            response = await self._client.get(path)
        except httpx.TransportError as exc:
            raise ProviderError(f"{self.name}: request failed: {exc!r}") from exc
        if response.status_code >= 400:
            raise _status_error(self.name, response.status_code, response.text)
        return response

    async def _describe_ollama(self, model: str) -> ModelInfo | None:
        """Ollama's native /api/show reports capabilities the /v1 listing omits."""
        root = str(self._client.base_url).rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        try:
            response = await self._client.post(f"{root}/api/show", json={"model": model})
        except httpx.TransportError:
            return None
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        if not isinstance(data, dict) or "capabilities" not in data:
            return None
        caps = set(data.get("capabilities") or [])
        trained_ctx = next(
            (
                v
                for k, v in (data.get("model_info") or {}).items()
                if k.endswith(".context_length") and isinstance(v, int)
            ),
            None,
        )
        # Ollama's OpenAI endpoint serves with the *server's* num_ctx (often 4096), not the
        # model's trained context, and silently truncates beyond it. So we do not report
        # the trained window as usable; users set context_window once they raise
        # OLLAMA_CONTEXT_LENGTH.
        log.debug("ollama model %s trained context %s", model, trained_ctx)
        return ModelInfo(
            id=model,
            tool_calling="tools" in caps,
            vision="vision" in caps,
            reasoning="thinking" in caps,
            json_mode=True,
            structured_output=True,
        )


class _StreamContext:
    """Opens a streaming POST with the same retry policy as non-streaming calls.

    Retries only happen before any data is received, so no partial output is replayed.
    """

    def __init__(self, provider: OpenAICompatibleProvider, path: str, payload: dict[str, Any]):
        self._provider = provider
        self._path = path
        self._payload = payload
        self._response: httpx.Response | None = None

    async def __aenter__(self) -> httpx.Response:
        provider = self._provider
        attempt = 0
        while True:
            request = provider._client.build_request("POST", self._path, json=self._payload)
            try:
                response = await provider._client.send(request, stream=True)
            except httpx.TransportError as exc:
                if attempt >= provider.config.max_retries:
                    raise ProviderError(f"{provider.name}: stream failed: {exc!r}") from exc
                await provider._backoff(attempt, None)
                attempt += 1
                continue
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                await response.aclose()
                if (
                    response.status_code in RETRYABLE_STATUS
                    and attempt < provider.config.max_retries
                ):
                    await provider._backoff(attempt, response.headers.get("retry-after"))
                    attempt += 1
                    continue
                raise _status_error(provider.name, response.status_code, body)
            self._response = response
            return response

    async def __aexit__(self, *exc: object) -> None:
        if self._response is not None:
            await self._response.aclose()


# -- encoding ----------------------------------------------------------------------------


def _encode_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role.value}
    if message.role == Role.TOOL:
        out["tool_call_id"] = message.tool_call_id
        out["content"] = message.text
        return out
    if isinstance(message.content, list):
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                parts.append({"type": "image_url", "image_url": {"url": part.url}})
        out["content"] = parts
    else:
        out["content"] = message.content if message.content is not None else ""
    if message.role == Role.ASSISTANT and message.tool_calls:
        if not message.text:
            out["content"] = None
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    if message.name and message.role == Role.USER:
        out["name"] = message.name
    return out


def _encode_tool_choice(choice: str) -> Any:
    if choice in {"auto", "none", "required"}:
        return choice
    return {"type": "function", "function": {"name": choice}}


# -- decoding ----------------------------------------------------------------------------


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str, str | None]:
    """Return (arguments, raw_text, parse_error)."""
    if isinstance(raw, dict):
        return raw, json.dumps(raw), None
    text = raw if isinstance(raw, str) else ""
    if not text.strip():
        return {}, text, None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, text, f"invalid JSON arguments: {exc.msg}"
    if not isinstance(value, dict):
        return {}, text, "arguments must be a JSON object"
    return value, text, None


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or []):
        function = raw.get("function") or {}
        arguments, raw_text, error = parse_tool_arguments(function.get("arguments"))
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{index}_{int(time.time() * 1000)}",
                name=function.get("name") or "",
                arguments=arguments,
                raw_arguments=raw_text,
                parse_error=error,
            )
        )
    return calls


def parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    completion_details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    cost = raw.get("cost")
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
    )


def parse_completion(data: dict[str, Any]) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"response has no choices: {json.dumps(data)[:500]}")
    choice = choices[0]
    raw_message = choice.get("message") or {}
    tool_calls = _parse_tool_calls(raw_message.get("tool_calls"))
    message = Message(
        role=Role.ASSISTANT,
        content=raw_message.get("content"),
        tool_calls=tool_calls,
        reasoning=raw_message.get("reasoning") or raw_message.get("reasoning_content"),
    )
    finish = _FINISH.get(choice.get("finish_reason") or "", FinishReason.OTHER)
    if tool_calls:
        finish = FinishReason.TOOL_CALLS  # some servers report "stop" alongside tool calls
    return ModelResponse(
        message=message,
        finish_reason=finish,
        usage=parse_usage(data.get("usage")),
        model=data.get("model") or "",
        raw=data,
    )


def _json_or_error(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"non-JSON response: {response.text[:300]}", status=response.status_code
        ) from exc
    if isinstance(data, dict) and "error" in data and not data.get("choices"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        code = error.get("code") if isinstance(error, dict) else None
        raise ProviderError(
            f"provider error: {message}",
            status=code if isinstance(code, int) else response.status_code,
            body=json.dumps(data)[:2000],
        )
    if not isinstance(data, dict):
        raise ProviderError("unexpected response shape", status=response.status_code)
    return data


def _status_error(name: str, status: int, body: str) -> ProviderError:
    detail = body[:500]
    try:
        parsed = json.loads(body)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict) and error.get("message"):
            detail = str(error["message"])
    except ValueError:
        pass
    return ProviderError(f"{name}: HTTP {status}: {detail}", status=status, body=body[:2000])


def _parse_sse_line(line: str) -> dict[str, Any] | str | None:
    line = line.strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return "[DONE]"
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        log.debug("skipping malformed SSE line: %s", data[:200])
        return None
    return parsed if isinstance(parsed, dict) else None


class _StreamAssembler:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage = Usage()
        self.model = ""

    def feed(self, chunk: dict[str, Any] | str) -> list[StreamEvent]:
        if not isinstance(chunk, dict):
            return []
        if "error" in chunk and not chunk.get("choices"):
            error = chunk["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ProviderError(f"stream error: {message}")
        self.model = chunk.get("model") or self.model
        if chunk.get("usage"):
            self.usage = parse_usage(chunk["usage"])
        events: list[StreamEvent] = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text.append(delta["content"])
                events.append(TextDelta(text=delta["content"]))
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                self.reasoning.append(reasoning)
                events.append(ReasoningDelta(text=reasoning))
            for raw in delta.get("tool_calls") or []:
                index = int(raw.get("index", len(self.calls)))
                slot = self.calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                function = raw.get("function") or {}
                if raw.get("id"):
                    slot["id"] = raw["id"]
                if function.get("name"):
                    slot["name"] += function["name"]
                args = function.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args)
                if args:
                    slot["arguments"] += args
                events.append(
                    ToolCallDelta(
                        index=index,
                        id=raw.get("id"),
                        name=function.get("name"),
                        arguments_delta=args or "",
                    )
                )
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def build(self) -> ModelResponse:
        calls = _parse_tool_calls(
            [
                {
                    "id": slot["id"],
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for _, slot in sorted(self.calls.items())
            ]
        )
        default = FinishReason.OTHER if self.finish_reason else FinishReason.STOP
        finish = _FINISH.get(self.finish_reason or "", default)
        if calls:
            finish = FinishReason.TOOL_CALLS
        return ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                content="".join(self.text) or None,
                tool_calls=calls,
                reasoning="".join(self.reasoning) or None,
            ),
            finish_reason=finish,
            usage=self.usage,
            model=self.model,
        )


def _model_info(entry: dict[str, Any]) -> ModelInfo:
    supported = entry.get("supported_parameters")
    supported_set = set(supported) if isinstance(supported, list) else None
    top = entry.get("top_provider") or {}
    modalities = (entry.get("architecture") or {}).get("input_modalities")
    pricing = entry.get("pricing") or {}

    def per_mtok(value: Any) -> float | None:
        try:
            return float(value) * 1e6 if value is not None else None
        except (TypeError, ValueError):
            return None

    return ModelInfo(
        id=str(entry.get("id")),
        context_window=entry.get("context_length") or None,
        max_output_tokens=top.get("max_completion_tokens") or None,
        tool_calling=("tools" in supported_set) if supported_set is not None else None,
        structured_output=("structured_outputs" in supported_set)
        if supported_set is not None
        else None,
        json_mode=("response_format" in supported_set) if supported_set is not None else None,
        reasoning=("reasoning" in supported_set) if supported_set is not None else None,
        vision=("image" in modalities) if isinstance(modalities, list) else None,
        input_cost_per_mtok=per_mtok(pricing.get("prompt")),
        output_cost_per_mtok=per_mtok(pricing.get("completion")),
    )
