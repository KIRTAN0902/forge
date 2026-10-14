"""In-process async event bus.

Observability must never break a workflow: a failing subscriber is logged and the
remaining subscribers still receive the event.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from forge.core.logging import get_logger
from forge.events.types import Event, EventType

Handler = Callable[[Event], Awaitable[None] | None]

log = get_logger("events")


@dataclass(frozen=True)
class _Subscription:
    handler: Handler
    types: frozenset[EventType] | None


class EventBus:
    def __init__(self) -> None:
        self._subscriptions: list[_Subscription] = []

    def subscribe(
        self, handler: Handler, *, types: Iterable[EventType] | None = None
    ) -> Callable[[], None]:
        """Register ``handler``; returns a function that unsubscribes it."""
        sub = _Subscription(handler, frozenset(types) if types is not None else None)
        self._subscriptions.append(sub)

        def unsubscribe() -> None:
            if sub in self._subscriptions:
                self._subscriptions.remove(sub)

        return unsubscribe

    async def publish(self, event: Event) -> Event:
        for sub in list(self._subscriptions):
            if sub.types is not None and event.type not in sub.types:
                continue
            try:
                result = sub.handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("event subscriber failed", extra={"event_type": str(event.type)})
        return event

    async def emit(
        self,
        type: EventType,
        *,
        workflow_id: str | None = None,
        source: str = "engine",
        **payload: Any,
    ) -> Event:
        return await self.publish(
            Event(type=type, workflow_id=workflow_id, source=source, payload=payload)
        )
