"""Unified model provider bridging UnifiedLLMClient into the ModelProvider contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from forge.config.settings import (
    ProviderName,
    StageModelBinding,
    classify_model_tier,
    parse_provider_model,
)
from forge.core.logging import get_logger
from forge.models.capabilities import ModelCapabilities
from forge.models.profile import ProviderConfig
from forge.models.provider import ModelProvider, ProviderError
from forge.models.types import (
    FinishReason,
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)
from forge.providers.client import LLMResponse, UnifiedLLMClient

log = get_logger("models.providers.unified")


class UnifiedModelProvider(ModelProvider):
    """Bridges Forge's UnifiedLLMClient across Gemini, Anthropic, Ollama, OpenAI, etc."""

    def __init__(
        self,
        name: str,
        *,
        client: UnifiedLLMClient | None = None,
        config: ProviderConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self._api_key = api_key
        self.client = client or UnifiedLLMClient()

    def _resolve_binding(self, model: str) -> StageModelBinding:
        """Resolve a StageModelBinding from provider name and model."""
        clean_model = model
        try:
            prov_enum, mod = parse_provider_model(f"{self.name}/{model}")
            clean_model = mod
        except Exception:
            try:
                prov_enum = ProviderName(self.name.lower())
            except Exception:
                prov_enum = ProviderName.CUSTOM

        tier = classify_model_tier(prov_enum, clean_model)
        return StageModelBinding(
            provider=prov_enum,
            model=clean_model,
            model_tier=tier,
            spec_string=f"{prov_enum.value}/{clean_model}",
        )

    def _format_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for m in messages:
            role = m.role
            if role == "tool":
                role = "user"
                content = f"[Tool Output]: {m.text or ''}"
            else:
                content = m.text or ""
            formatted.append({"role": role, "content": content})
        return formatted

    async def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        binding = self._resolve_binding(model)
        messages = self._format_messages(list(request.messages))
        try:
            resp: LLMResponse = await self.client.complete_or_stream(
                binding,
                messages,
                system_prompt=None,
                stage="executor",
            )
        except Exception as exc:
            raise ProviderError(f"UnifiedModelProvider call failed ({binding.spec_string}): {exc}") from exc

        return ModelResponse(
            message=Message.assistant(
                resp.content,
                reasoning=resp.reasoning_content or None,
            ),
            usage=Usage(
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                total_tokens=resp.total_tokens,
            ),
            finish_reason=FinishReason.STOP,
            latency_s=resp.latency_ms / 1000.0 if resp.latency_ms else 0.0,
        )

    async def stream(self, model: str, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        binding = self._resolve_binding(model)
        messages = self._format_messages(list(request.messages))
        queue: asyncio.Queue[StreamEvent | Exception] = asyncio.Queue()

        def on_token(delta: str) -> None:
            if delta:
                queue.put_nowait(TextDelta(text=delta))

        def on_thinking(delta: str) -> None:
            if delta:
                queue.put_nowait(ReasoningDelta(text=delta))

        async def _worker() -> None:
            try:
                resp = await self.client.complete_or_stream(
                    binding,
                    messages,
                    system_prompt=None,
                    on_token=on_token,
                    on_thinking=on_thinking,
                    stage="executor",
                )
                done = ModelResponse(
                    message=Message.assistant(
                        resp.content,
                        reasoning=resp.reasoning_content or None,
                    ),
                    usage=Usage(
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        total_tokens=resp.total_tokens,
                    ),
                    finish_reason=FinishReason.STOP,
                    latency_s=resp.latency_ms / 1000.0 if resp.latency_ms else 0.0,
                )
                queue.put_nowait(StreamDone(response=done))
            except Exception as exc:
                queue.put_nowait(exc)

        task = asyncio.create_task(_worker())
        try:
            while True:
                item = await queue.get()
                if isinstance(item, Exception):
                    raise ProviderError(
                        f"UnifiedModelProvider stream failed ({binding.spec_string}): {item}"
                    ) from item
                yield item
                if isinstance(item, StreamDone):
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def describe_model(self, model: str) -> ModelInfo | None:
        binding = self._resolve_binding(model)
        return ModelInfo(
            name=model,
            capabilities=ModelCapabilities(
                streaming=True,
                tool_calling=False,
                structured_output=False,
                context_window=128_000,
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        try:
            prov_enum = ProviderName(self.name.lower())
            models = await self.client.list_live_models(prov_enum)
            return [
                ModelInfo(
                    name=m,
                    capabilities=ModelCapabilities(
                        streaming=True,
                        tool_calling=False,
                        structured_output=False,
                        context_window=128_000,
                    ),
                )
                for m in models
            ]
        except Exception:
            return []
