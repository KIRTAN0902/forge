from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware current UTC time. All persisted timestamps use this."""
    return datetime.now(UTC)
