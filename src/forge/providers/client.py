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
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from forge.config.settings import DEFAULT_PROVIDERS, ProviderName, StageModelBinding

TokenHandler = Callable[[str], None]
ThinkingHandler = Callable[[str], None]
MessageLike = Mapping[str, str]

_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_THINK_TAGS = ("<think>", "<thinking>", "</think>", "</thinking>")
_MAX_THINK_TAG_LEN = max(len(tag) for tag in _THINK_TAGS)


def _incomplete_tag_suffix(text: str) -> str:
    """Return a trailing partial ``<think`` / ``</think`` tag that must be buffered."""
    for size in range(min(len(text), _MAX_THINK_TAG_LEN - 1), 0, -1):
        suffix = text[-size:]
        if suffix.startswith("<") and any(
            tag.startswith(suffix.lower()) for tag in _THINK_TAGS
        ):
            return suffix
    return ""


class ThinkingStreamParser:
    """Incremental reasoning/answer separator shared by every provider wire format.

    Native reasoning fields (OpenAI ``reasoning_content``, Gemini ``thought`` parts,
    Anthropic ``thinking`` blocks, Ollama ``thinking``) are fed through
    :meth:`feed_reasoning`. Inline ``<think>...</think>`` / ``<thinking>`` tags inside
    ordinary text are handled by :meth:`feed_text`, including tags split across chunks.

    Only visible answer text is emitted through ``on_token``; hidden reasoning is emitted
    through ``on_thinking`` as the latest thought line.
    """

    def __init__(
        self,
        on_token: TokenHandler | None = None,
        on_thinking: ThinkingHandler | None = None,
    ) -> None:
        self.on_token = on_token
        self.on_thinking = on_thinking
        self.in_think_block = False
        self.think_buffer = ""
        self.content_buffer = ""
        self.reasoning_incomplete = False
        self._pending = ""

    @property
    def streaming_requested(self) -> bool:
        return self.on_token is not None or self.on_thinking is not None

    @property
    def latest_thought_line(self) -> str:
        """Most recent non-empty line held in the hidden reasoning buffer."""
        lines = [line.strip() for line in self.think_buffer.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return self.think_buffer.strip()

    def feed(self, text: str) -> str:
        """Alias for :meth:`feed_text`."""
        return self.feed_text(text)

    def feed_text(self, text: str) -> str:
        """Consume visible-model text, returning only the newly emitted answer text."""
        if not text:
            return ""
        data = self._pending + text
        self._pending = ""
        emitted: list[str] = []

        while data:
            if self.in_think_block:
                close_match = _THINK_CLOSE_RE.search(data)
                if close_match is not None:
                    self._append_thought(data[: close_match.start()])
                    data = data[close_match.end() :]
                    self.in_think_block = False
                    continue
                held = _incomplete_tag_suffix(data)
                if held:
                    self._append_thought(data[: -len(held)])
                    self._pending = held
                else:
                    self._append_thought(data)
                break

            open_match = _THINK_OPEN_RE.search(data)
            close_match = _THINK_CLOSE_RE.search(data)
            if close_match is not None and (
                open_match is None or close_match.start() < open_match.start()
            ):
                # Chat templates often place the opening tag in the prompt, leaving only
                # a closing tag in the model output. Everything before it is reasoning.
                self._append_thought(data[: close_match.start()])
                data = data[close_match.end() :]
                continue

            if open_match is not None:
                self._emit_content(data[: open_match.start()], emitted)
                data = data[open_match.end() :]
                self.in_think_block = True
                continue

            held = _incomplete_tag_suffix(data)
            if held:
                self._emit_content(data[: -len(held)], emitted)
                self._pending = held
            else:
                self._emit_content(data, emitted)
            break

        return "".join(emitted)

    def feed_reasoning(self, text: str) -> None:
        """Consume a native reasoning delta; it is never emitted as visible content."""
        if text:
            self._append_thought(text)

    def close(self) -> tuple[str, str]:
        """Flush pending text and return ``(visible_content, hidden_reasoning)``."""
        if self._pending:
            pending = self._pending
            self._pending = ""
            if self.in_think_block:
                self._append_thought(pending)
            else:
                self._emit_content(pending, [])
        if self.in_think_block:
            # The output budget was exhausted inside a thought block. The hidden
            # reasoning is retained for diagnostics, but it is never promoted to content.
            self.in_think_block = False
            self.reasoning_incomplete = True
        return self.content_buffer, self.think_buffer

    def _emit_content(self, text: str, sink: list[str]) -> None:
        if not text:
            return
        self.content_buffer += text
        sink.append(text)
        if self.on_token is not None:
            self.on_token(text)

    def _append_thought(self, text: str) -> None:
        if not text:
            return
        self.think_buffer += text
        if self.on_thinking is not None:
            latest = self.latest_thought_line
            if latest:
                self.on_thinking(latest)


def strip_thinking_tags(raw_text: str) -> tuple[str, str]:
    """Split arbitrary model text into ``(clean_answer, hidden_reasoning)``.

    Handles closed and unclosed ``<think>`` / ``<thinking>`` blocks, plus orphan closing
    tags produced when the opening tag lived in the prompt template.
    """
    parser = ThinkingStreamParser()
    parser.feed_text(raw_text or "")
    return parser.close()

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
    reasoning_incomplete: bool = False
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""

    def finish(self, model: str, provider: ProviderName) -> LLMResponse:
        """Backfill totals after streaming completes.

        Reasoning is deliberately **not** copied into ``content``. Hidden chain-of-thought
        must never be displayed or parsed as the model's answer; callers surface a
        diagnostic when the model produced reasoning but no final content.
        """
        if not self.prompt_tokens:
            self.prompt_tokens = _estimate_tokens(self.content)
        if not self.completion_tokens:
            self.completion_tokens = _estimate_tokens(
                f"{self.content}\n{self.reasoning_content}".strip()
            )
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
        on_thinking: ThinkingHandler | None = None,
        stage: str = "anvil",
    ) -> LLMResponse:
        """Run one real completion, streaming visible deltas to ``on_token``.

        Provider-native reasoning and inline ``<think>`` blocks are routed exclusively to
        ``on_thinking`` and never leak into ``on_token`` or the final answer.
        """
        start = time.perf_counter()
        provider = binding.provider
        parser = ThinkingStreamParser(on_token=on_token, on_thinking=on_thinking)
        if provider is ProviderName.GEMINI:
            resp = await self._gemini(binding, messages, system_prompt, parser)
        elif provider is ProviderName.ANTHROPIC:
            resp = await self._anthropic(binding, messages, system_prompt, parser)
        elif provider is ProviderName.OLLAMA:
            resp = await self._ollama(binding, messages, system_prompt, parser)
        else:
            resp = await self._openai_compatible(binding, messages, system_prompt, parser)

        resp.content, resp.reasoning_content = parser.close()
        resp.reasoning_incomplete = parser.reasoning_incomplete
        resp.finish(resp.model or binding.model, provider)

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
        on_thinking: ThinkingHandler | None = None,
        stage: str = "anvil",
    ) -> LLMResponse:
        return await self.complete_or_stream(
            binding,
            messages,
            system_prompt,
            on_token=on_token,
            on_thinking=on_thinking,
            stage=stage,
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
        parser: ThinkingStreamParser,
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
            if not parser.streaming_requested:
                resp = await client.post(
                    url, headers=self._auth_headers(provider, self.api_key(provider)), json=body
                )
                resp.raise_for_status()
                _parse_openai_response(resp.json(), result, parser)
                return result
            body["stream"] = True
            if provider in {ProviderName.OPENAI, ProviderName.OPENROUTER}:
                body["stream_options"] = {"include_usage": True}
            async with client.stream(
                "POST", url, headers=self._auth_headers(provider, self.api_key(provider)), json=body
            ) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_openai_chunk(payload, result, parser)
        return result

    async def _gemini(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        parser: ThinkingStreamParser,
    ) -> LLMResponse:
        provider = binding.provider
        base = self.base_url(provider)
        model_name = binding.model.removeprefix("models/")
        result = LLMResponse(model=model_name, provider=provider.value)
        generation_config: dict[str, Any] = {
            "temperature": binding.temperature,
            "maxOutputTokens": binding.max_tokens,
        }
        if "2.5" in model_name.lower() or "thinking" in model_name.lower():
            # Gemini only emits separate thought parts when explicitly requested.
            generation_config["thinkingConfig"] = {"includeThoughts": True}
        body: dict[str, Any] = {
            "contents": _gemini_contents(messages),
            "generationConfig": generation_config,
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        key = self.api_key(provider)
        headers = self._auth_headers(provider, key)
        params = {"key": key} if key else {}
        async with self._client() as client:
            if not parser.streaming_requested:
                url = f"{base}/models/{model_name}:generateContent"
                resp = await client.post(url, headers=headers, params=params, json=body)
                resp.raise_for_status()
                _parse_gemini_response(resp.json(), result, parser)
                return result
            url = f"{base}/models/{model_name}:streamGenerateContent?alt=sse"
            async with client.stream(
                "POST", url, headers=headers, params=params, json=body
            ) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_gemini_chunk(payload, result, parser)
        return result

    async def _anthropic(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        parser: ThinkingStreamParser,
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
            if not parser.streaming_requested:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                _parse_anthropic_response(resp.json(), result, parser)
                return result
            body["stream"] = True
            async with client.stream("POST", url, headers=headers, json=body) as stream:
                stream.raise_for_status()
                async for payload in _iter_sse(stream):
                    _consume_anthropic_chunk(payload, result, parser)
        return result

    async def _ollama(
        self,
        binding: StageModelBinding,
        messages: list[MessageLike] | list[dict[str, str]],
        system_prompt: str | None,
        parser: ThinkingStreamParser,
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
            if not parser.streaming_requested:
                body["stream"] = False
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                _parse_ollama_response(resp.json(), result, parser)
                return result
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
                    _consume_ollama_chunk(chunk, result, parser)
        return result


# Public compatibility name used by the specification and external integrations.
ForgeLLMClient = UnifiedLLMClient


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


def _parse_openai_response(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    choices = payload.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    reasoning = (
        message.get("reasoning_content") or message.get("reasoning") or message.get("thought") or ""
    )
    parser.feed_reasoning(str(reasoning))
    parser.feed_text(str(message.get("content") or ""))
    result.tool_calls = list(message.get("tool_calls") or [])
    usage = payload.get("usage") or {}
    result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    result.completion_tokens = int(usage.get("completion_tokens") or 0)
    result.total_tokens = int(usage.get("total_tokens") or 0)
    if result.model == "":
        result.model = str(payload.get("model") or "")


def _parse_gemini_response(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    _feed_gemini_parts(payload, parser)
    usage = payload.get("usageMetadata") or {}
    result.prompt_tokens = int(usage.get("promptTokenCount") or 0)
    result.completion_tokens = int(usage.get("candidatesTokenCount") or 0)
    result.total_tokens = int(usage.get("totalTokenCount") or 0)


def _parse_anthropic_response(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            parser.feed_reasoning(str(block.get("thinking") or block.get("text") or ""))
        elif block_type == "text" or block_type is None:
            parser.feed_text(str(block.get("text") or ""))
    usage = payload.get("usage") or {}
    result.prompt_tokens = int(usage.get("input_tokens") or 0)
    result.completion_tokens = int(usage.get("output_tokens") or 0)


def _parse_ollama_response(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    message = payload.get("message") or {}
    parser.feed_reasoning(str(message.get("thinking") or message.get("reasoning") or ""))
    parser.feed_text(str(message.get("content") or ""))
    result.prompt_tokens = int(payload.get("prompt_eval_count") or 0)
    result.completion_tokens = int(payload.get("eval_count") or 0)


def _feed_gemini_parts(payload: dict[str, Any], parser: ThinkingStreamParser) -> None:
    candidates = payload.get("candidates") or []
    if not candidates:
        return
    parts = (candidates[0].get("content") or {}).get("parts") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = str(part.get("text") or "")
        if part.get("thought") is True:
            parser.feed_reasoning(text)
        else:
            parser.feed_text(text)


# -- streaming chunk consumption ---------------------------------------------------------


def _consume_openai_chunk(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    choices = payload.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        reasoning = (
            delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thought")
        )
        if reasoning:
            parser.feed_reasoning(str(reasoning))
        content = delta.get("content")
        if content:
            parser.feed_text(str(content))
    usage = payload.get("usage")
    if usage:
        result.prompt_tokens = int(usage.get("prompt_tokens") or result.prompt_tokens)
        result.completion_tokens = int(usage.get("completion_tokens") or result.completion_tokens)
        result.total_tokens = int(usage.get("total_tokens") or result.total_tokens)


def _consume_gemini_chunk(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    _feed_gemini_parts(payload, parser)
    usage = payload.get("usageMetadata")
    if usage:
        result.prompt_tokens = int(usage.get("promptTokenCount") or result.prompt_tokens)
        result.completion_tokens = int(
            usage.get("candidatesTokenCount") or result.completion_tokens
        )
        result.total_tokens = int(usage.get("totalTokenCount") or result.total_tokens)


def _consume_anthropic_chunk(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    event_type = payload.get("type")
    if event_type == "content_block_start":
        block = payload.get("content_block") or {}
        if block.get("type") == "thinking":
            parser.feed_reasoning(str(block.get("thinking") or block.get("text") or ""))
        elif block.get("type") == "text":
            parser.feed_text(str(block.get("text") or ""))
    elif event_type == "content_block_delta":
        delta = payload.get("delta") or {}
        if delta.get("type") == "thinking_delta":
            parser.feed_reasoning(str(delta.get("thinking") or delta.get("text") or ""))
        else:
            text = delta.get("text")
            if text:
                parser.feed_text(str(text))
    elif event_type == "message_start":
        usage = (payload.get("message") or {}).get("usage") or {}
        result.prompt_tokens = int(usage.get("input_tokens") or result.prompt_tokens)
    elif event_type == "message_delta":
        usage = payload.get("usage") or {}
        result.completion_tokens = int(usage.get("output_tokens") or result.completion_tokens)


def _consume_ollama_chunk(
    payload: dict[str, Any], result: LLMResponse, parser: ThinkingStreamParser
) -> None:
    message = payload.get("message") or {}
    reasoning = message.get("thinking") or message.get("reasoning")
    if reasoning:
        parser.feed_reasoning(str(reasoning))
    content = message.get("content")
    if content:
        parser.feed_text(str(content))
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


__all__ = [
    "LLMResponse",
    "ThinkingHandler",
    "ThinkingStreamParser",
    "TokenHandler",
    "UnifiedLLMClient",
    "strip_thinking_tags",
]
