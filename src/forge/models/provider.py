"""The transport layer: a provider turns a ``ModelRequest`` into a ``ModelResponse``.

Providers do no prompt engineering and make no capability decisions; they faithfully
translate the neutral request to the wire format and back. Fallback strategies live in
:class:`forge.models.adapter.ModelAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from forge.core.errors import ModelError
from forge.models.types import ModelInfo, ModelRequest, ModelResponse, StreamEvent


class ProviderError(ModelError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def is_request_rejected(self) -> bool:
        """The server refused the request shape (bad param), as opposed to failing."""
        return self.status in {400, 404, 422}


class ModelProvider(ABC):
    name: str = "provider"

    @abstractmethod
    async def complete(self, model: str, request: ModelRequest) -> ModelResponse: ...

    @abstractmethod
    def stream(self, model: str, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Yield deltas; the final event is always ``StreamDone``."""
        ...

    async def describe_model(self, model: str) -> ModelInfo | None:
        """Report what the provider knows about ``model``; ``None`` when unknown."""
        return None

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def aclose(self) -> None:
        return None
