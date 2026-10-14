"""Token and cost accounting across all model calls in a session/workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.models.types import Usage


@dataclass
class UsageRecord:
    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


class UsageTracker:
    def __init__(self) -> None:
        self.total = UsageRecord()
        self.by_profile: dict[str, UsageRecord] = {}
        self.by_role: dict[str, UsageRecord] = {}

    def record(self, *, profile: str, role: str, usage: Usage, latency_s: float | None) -> None:
        for record in (
            self.total,
            self.by_profile.setdefault(profile, UsageRecord()),
            self.by_role.setdefault(role, UsageRecord()),
        ):
            record.calls += 1
            record.usage = record.usage + usage
            record.latency_s += latency_s or 0.0
