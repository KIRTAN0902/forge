"""Unified, provider-native streaming LLM client for Forge.

One async entry point (:meth:`UnifiedLLMClient.complete_or_stream`) speaks the native REST/SSE
protocol of every provider Forge supports, with no third-party SDKs beyond ``httpx``:

* **Google Gemini** native API (``:generateContent`` / ``:streamGenerateContent?alt=sse``)
* **OpenAI-compatible** API (OpenAI, OpenRouter, NVIDIA NIM, DeepSeek, Groq — ``/chat/completions``)
* **Anthropic Messages** API (``/v1/messages``)
* **Local Ollama** API (``/api/chat`` and ``/api/tags``)

Nothing here fabricates a reply: every call issues a real HTTP request and reports the real
token usage back to the caller (falling back to a conservative estimate only when a provider
omits usage metadata).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from forge.config.settings import DEFAULT_PROVIDERS, ProviderName, StageModelBinding

TokenHandler = Callable[[str], None]
MessageLike = Mapping[str, str]

_OPENAI_COMPATIBLE: frozenset[ProviderName] = frozenset(
    {
        ProviderName.OPENAI,
        ProviderName.OPENROUTER,
        ProviderName.NVIDIA,
        ProviderName.DEEPSEEK,
        ProviderName.GROQ,
        ProviderName.CUSTOM,
    }
)

_ANTHROPIC_VERSION = "2023-06-01"


class LLMResponse(BaseModel):
    """A fully-assembled model reply plus real token accounting."""

    model_config = ConfigDict(extra="ignore")

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""

    def finish(self, model: str, provider: ProviderName) -> LLMResponse:
        """Backfill totals (and any missing halves) after streaming completes."""
        if not self.prompt_tokens:
            self.prompt_tokens = _estimate_tokens(self.content)
        if not self.completion_tokens:
            self.completion_tokens = _estimate_tokens(self.content)
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        if not self.model:
            self.model = model
        if not self.provider:
            self.provider = provider.value
        return self


def _estimate_tokens(text: str) -> int:
    """Conservative ~4-chars-per-token estimate used only when usage is omitted."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class UnifiedLLMClient:
    """Executes real async HTTP streaming calls across every supported provider."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 120.0,
        api_key_resolver: Callable[[ProviderName], str | None] | None = None,
        base_url_resolver: Callable[[ProviderName], str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._api_key_resolver = api_key_resolver
        self._base_url_resolver = base_url_resolver
        self._env = dict(os.environ if env is None else env)

    # -- resolution helpers --------------------------------------------------------------

    def api_key(self, provider: ProviderName) -> str | None:
        key: str | None = None
        if self._api_key_resolver is not None:
            key = self._api_key_resolver(provider)
        if not key:
            env_var = DEFAULT_PROVIDERS[provider].api_key_env
            if env_var:
                key = os.environ.get(env_var) or self._env.get(env_var)
        if not key:
            try:
                from forge.config.settings import ForgeConfigManager

                key = ForgeConfigManager().api_key_for(provider)
            except Exception:
                key = None
        if key and provider is ProviderName.NVIDIA:
            clean = key.strip()
            if clean.startswith("vapi-"):
                clean = "n" + clean
            return clean
        return key.strip() if key else None

    def base_url(self, provider: ProviderName) -> str:
        if self._base_url_resolver is not None:
            return self._base_url_resolver(provider).rstrip("/")
        return DEFAULT_PROVIDERS[provider].base_url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=self._timeout)

    # -- public API ----------------------------------------------------------------------

    async def complete_or_stream(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None = None,
        on_token: TokenHandler | None = None,
        stage: str = "anvil",
    ) -> LLMResponse:
        """Run one real completion, streaming token deltas to ``on_token`` when provided."""
        start = time.perf_counter()
        provider = binding.provider
        if provider is ProviderName.GEMINI:
            resp = await self._gemini(binding, messages, system_prompt, on_token)
        elif provider is ProviderName.ANTHROPIC:
            resp = await self._anthropic(binding, messages, system_prompt, on_token)
        elif provider is ProviderName.OLLAMA:
            resp = await self._ollama(binding, messages, system_prompt, on_token)
        else:
            resp = await self._openai_compatible(binding, messages, system_prompt, on_token)

        latency_ms = (time.perf_counter() - start) * 1000.0
        try:
            from forge.config.usage_tracker import get_usage_tracker

            get_usage_tracker().record_call(
                provider=binding.provider.value,
                model=binding.model,
                stage=stage,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception:
            pass
        return resp

    # Alias used by the TUI / spec.
    async def stream_chat(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None = None,
        on_token: TokenHandler | None = None,
        stage: str = "anvil",
    ) -> LLMResponse:
        return await self.complete_or_stream(
            binding, messages, system_prompt, on_token, stage=stage
        )

    async def list_live_models(self, provider: ProviderName) -> list[str]:
        """Query the provider's live model catalog (Ollama ``/api/tags``, else ``/models``)."""
        key = self.api_key(provider)
        headers = self._auth_headers(provider, key)
        try:
            async with self._client() as client:
                if provider is ProviderName.OLLAMA:
                    resp = await client.get(self.base_url(provider) + "/api/tags", headers=headers)
                    resp.raise_for_status()
                    return [
                        m["name"]
                        for m in resp.json().get("models", [])
                        if isinstance(m, dict) and m.get("name")
                    ]
                if provider is ProviderName.GEMINI:
                    resp = await client.get(self.base_url(provider) + "/models", headers=headers)
                    resp.raise_for_status()
                    return [
                        str(m["name"]).removeprefix("models/")
                        for m in resp.json().get("models", [])
                        if isinstance(m, dict) and m.get("name")
                    ]
                resp = await client.get(self.base_url(provider) + "/models", headers=headers)
                resp.raise_for_status()
                return [
                    str(m["id"])
                    for m in resp.json().get("data", [])
                    if isinstance(m, dict) and m.get("id")
                ]
        except Exception:
            return []

    async def test_connection(self, binding: StageModelBinding) -> tuple[bool, str]:
        """Send a minimal ping to verify the API key + model name work end to end."""
        start = time.perf_counter()
        try:
            response = await self.complete_or_stream(
                binding,
                [{"role": "user", "content": "ping"}],
                system_prompt="Reply with the single word: pong.",
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = _http_error_detail(exc)
            return False, f"HTTP {status}: {detail}"
        except Exception as exc:  # pragma: no cover - network dependent
            return False, f"Connection failed: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not response.content.strip():
            return False, "Connected, but the model returned no content."
        return True, f"Connected (latency: {latency_ms:.0f}ms)"

    # -- request plumbing ----------------------------------------------------------------

    def _auth_headers(self, provider: ProviderName, key: str | None) -> dict[str, str]:
        if provider is ProviderName.GEMINI:
            return {"x-goog-api-key": key} if key else {}
        if provider is ProviderName.ANTHROPIC:
            headers = {"anthropic-version": _ANTHROPIC_VERSION}
            if key:
                headers["x-api-key"] = key
            return headers
        if provider is ProviderName.OLLAMA:
            return {}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if provider is ProviderName.OPENROUTER:
            headers["X-Title"] = "Forge"
        return headers

    async def _openai_compatible(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        on_token: TokenHandler | None,
    ) -> LLMResponse:
        provider = binding.provider
        url = self.base_url(provider) + "/chat/completions"
        model_name = binding.model
        if provider is ProviderName.NVIDIA and "/" not in model_name:
            model_name = f"nvidia/{model_name}"
        wire = _openai_messages(messages, system_prompt)
        body: dict[str, Any] = {
            "model": model_name,
            "messages": wire,
            "temperature": binding.temperature,
            "max_tokens": binding.max_tokens,
        }
        result = LLMResponse(model=model_name, provider=provider.value)
        async with self._client() as client:
            if on_token is None:
                resp = await client.post(
                    url, headers=self._auth_headers(provider, self.api_key(provider)), json=body
                )
                resp.raise_for_status()
                return _parse_openai_response(resp.json(), result)
            body["stream"] = True
            if provider in {ProviderName.OPENAI, ProviderName.OPENROUTER}:
                body["stream_options"] = {"include_usage": True}
            async with client.stream(
                "POST", url, headers=self._auth_headers(provider, self.api_key(provider)), json=body
            ) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_openai_chunk(payload, result, on_token)
        return result.finish(model_name, provider)

    async def _gemini(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        on_token: TokenHandler | None,
    ) -> LLMResponse:
        provider = binding.provider
        base = self.base_url(provider)
        model_name = binding.model.removeprefix("models/")
        result = LLMResponse(model=model_name, provider=provider.value)
        body: dict[str, Any] = {
            "contents": _gemini_contents(messages),
            "generationConfig": {
                "temperature": binding.temperature,
                "maxOutputTokens": binding.max_tokens,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        key = self.api_key(provider)
        headers = self._auth_headers(provider, key)
        params = {"key": key} if key else {}
        async with self._client() as client:
            if on_token is None:
                url = f"{base}/models/{model_name}:generateContent"
                resp = await client.post(url, headers=headers, params=params, json=body)
                resp.raise_for_status()
                return _parse_gemini_response(resp.json(), result)
            url = f"{base}/models/{model_name}:streamGenerateContent?alt=sse"
            async with client.stream(
                "POST", url, headers=headers, params=params, json=body
            ) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_gemini_chunk(payload, result, on_token)
        return result.finish(model_name, provider)

    async def _anthropic(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        on_token: TokenHandler | None,
    ) -> LLMResponse:
        provider = binding.provider
        url = self.base_url(provider) + "/messages"
        result = LLMResponse(model=binding.model, provider=provider.value)
        body: dict[str, Any] = {
            "model": binding.model,
            "messages": _anthropic_messages(messages),
            "max_tokens": binding.max_tokens,
            "temperature": binding.temperature,
        }
        if system_prompt:
            body["system"] = system_prompt
        headers = self._auth_headers(provider, self.api_key(provider))
        async with self._client() as client:
            if on_token is None:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                return _parse_anthropic_response(resp.json(), result)
            body["stream"] = True
            async with client.stream("POST", url, headers=headers, json=body) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_anthropic_chunk(payload, result, on_token)
        return result.finish(binding.model, provider)

    async def _ollama(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        on_token: TokenHandler | None,
    ) -> LLMResponse:
        provider = binding.provider
        url = self.base_url(provider) + "/api/chat"
        result = LLMResponse(model=binding.model, provider=provider.value)
        body: dict[str, Any] = {
            "model": binding.model,
            "messages": _openai_messages(messages, system_prompt),
            "options": {
                "temperature": binding.temperature,
                "num_predict": binding.max_tokens,
            },
        }
        async with self._client() as client:
            if on_token is None:
                body["stream"] = False
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                return _parse_ollama_response(resp.json(), result)
            body["stream"] = True
            async with client.stream("POST", url, json=body) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _consume_ollama_chunk(chunk, result, on_token)
        return result.finish(binding.model, provider)


# -- wire-format conversion --------------------------------------------------------------


def _sanitize_turns(
    messages: list[MessageLike] | list[dict[str, str]],
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for m in messages:
        role = (
            "assistant"
            if str(m.get("role", "")).lower() in {"assistant", "model", "forge"}
            else "user"
        )
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if not cleaned and role == "assistant":
            # Models like Gemini and Claude reject conversations starting with an assistant turn
            continue
        if cleaned and cleaned[-1]["role"] == role:
            # Merge consecutive turns of the same role
            cleaned[-1]["content"] += "\n\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    if not cleaned:
        cleaned.append({"role": "user", "content": "Hello"})
    return cleaned


def _openai_messages(
    messages: list[MessageLike] | list[dict[str, str]], system_prompt: str | None
) -> list[dict[str, str]]:
    wire: list[dict[str, str]] = []
    if system_prompt:
        wire.append({"role": "system", "content": system_prompt})
    wire.extend(_sanitize_turns(messages))
    return wire


def _gemini_contents(
    messages: list[MessageLike] | list[dict[str, str]],
) -> list[dict[str, Any]]:
    sanitized = _sanitize_turns(messages)
    contents: list[dict[str, Any]] = []
    for message in sanitized:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def _anthropic_messages(
    messages: list[MessageLike] | list[dict[str, str]],
) -> list[dict[str, str]]:
    # Anthropic rejects system turns in ``messages``; they travel in the top-level ``system``.
    sanitized = _sanitize_turns(messages)
    return [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in sanitized
        if m["role"] != "system"
    ]


async def _iter_sse(stream: httpx.Response) -> Any:
    """Yield decoded JSON payloads from a Server-Sent-Events stream."""
    async for line in stream.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


# -- response parsing --------------------------------------------------------------------


def _parse_openai_response(payload: dict[str, Any], result: LLMResponse) -> LLMResponse:
    choices = payload.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    result.content = str(message.get("content") or "")
    result.reasoning_content = str(message.get("reasoning_content") or "")
    result.tool_calls = list(message.get("tool_calls") or [])
    usage = payload.get("usage") or {}
    result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    result.completion_tokens = int(usage.get("completion_tokens") or 0)
    result.total_tokens = int(usage.get("total_tokens") or 0)
    if result.model == "":
        result.model = str(payload.get("model") or "")
    provider = ProviderName(result.provider)
    return result.finish(result.model, provider)


def _parse_gemini_response(payload: dict[str, Any], result: LLMResponse) -> LLMResponse:
    result.content = _gemini_text(payload)
    usage = payload.get("usageMetadata") or {}
    result.prompt_tokens = int(usage.get("promptTokenCount") or 0)
    result.completion_tokens = int(usage.get("candidatesTokenCount") or 0)
    result.total_tokens = int(usage.get("totalTokenCount") or 0)
    return result.finish(result.model, ProviderName.GEMINI)


def _parse_anthropic_response(payload: dict[str, Any], result: LLMResponse) -> LLMResponse:
    blocks = payload.get("content") or []
    result.content = "".join(
        str(block.get("text") or "") for block in blocks if isinstance(block, dict)
    )
    usage = payload.get("usage") or {}
    result.prompt_tokens = int(usage.get("input_tokens") or 0)
    result.completion_tokens = int(usage.get("output_tokens") or 0)
    return result.finish(result.model, ProviderName.ANTHROPIC)


def _parse_ollama_response(payload: dict[str, Any], result: LLMResponse) -> LLMResponse:
    message = payload.get("message") or {}
    result.content = str(message.get("content") or "")
    result.prompt_tokens = int(payload.get("prompt_eval_count") or 0)
    result.completion_tokens = int(payload.get("eval_count") or 0)
    return result.finish(result.model, ProviderName.OLLAMA)


def _gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))


# -- streaming chunk consumption ---------------------------------------------------------


def _consume_openai_chunk(
    payload: dict[str, Any], result: LLMResponse, on_token: TokenHandler
) -> None:
    choices = payload.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            result.content += content
            on_token(content)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            result.reasoning_content += reasoning
    usage = payload.get("usage")
    if usage:
        result.prompt_tokens = int(usage.get("prompt_tokens") or result.prompt_tokens)
        result.completion_tokens = int(usage.get("completion_tokens") or result.completion_tokens)
        result.total_tokens = int(usage.get("total_tokens") or result.total_tokens)


def _consume_gemini_chunk(
    payload: dict[str, Any], result: LLMResponse, on_token: TokenHandler
) -> None:
    text = _gemini_text(payload)
    if text:
        result.content += text
        on_token(text)
    usage = payload.get("usageMetadata")
    if usage:
        result.prompt_tokens = int(usage.get("promptTokenCount") or result.prompt_tokens)
        result.completion_tokens = int(
            usage.get("candidatesTokenCount") or result.completion_tokens
        )
        result.total_tokens = int(usage.get("totalTokenCount") or result.total_tokens)


def _consume_anthropic_chunk(
    payload: dict[str, Any], result: LLMResponse, on_token: TokenHandler
) -> None:
    event_type = payload.get("type")
    if event_type == "content_block_delta":
        delta = payload.get("delta") or {}
        text = delta.get("text")
        if text:
            result.content += text
            on_token(text)
    elif event_type == "message_start":
        usage = (payload.get("message") or {}).get("usage") or {}
        result.prompt_tokens = int(usage.get("input_tokens") or result.prompt_tokens)
    elif event_type == "message_delta":
        usage = payload.get("usage") or {}
        result.completion_tokens = int(usage.get("output_tokens") or result.completion_tokens)


def _consume_ollama_chunk(
    payload: dict[str, Any], result: LLMResponse, on_token: TokenHandler
) -> None:
    message = payload.get("message") or {}
    content = message.get("content")
    if content:
        result.content += content
        on_token(content)
    if payload.get("done"):
        result.prompt_tokens = int(payload.get("prompt_eval_count") or result.prompt_tokens)
        result.completion_tokens = int(payload.get("eval_count") or result.completion_tokens)


def _http_error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
    except Exception:
        text = exc.response.text.strip()
        return text[:200] if text else "no details"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if payload.get("message"):
            return str(payload["message"])
    return str(payload)[:200]


__all__ = ["LLMResponse", "TokenHandler", "UnifiedLLMClient"]
