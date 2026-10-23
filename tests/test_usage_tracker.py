"""Usage ledger filtering and persistence regression tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from forge.config import usage_tracker as usage_module
from forge.config.settings import ProviderName, StageModelBinding
from forge.config.usage_tracker import UsageTracker
from forge.providers.client import UnifiedLLMClient


def record(tracker: UsageTracker, session_id: str, timestamp: str) -> None:
    tracker.record_call(
        provider="gemini",
        model="gemini-2.5-flash",
        stage="anvil",
        prompt_tokens=10,
        completion_tokens=5,
        session_id=session_id,
        timestamp=timestamp,
    )


def test_session_filter_applies_to_all_window(tmp_path: Path) -> None:
    tracker = UsageTracker(ledger_path=tmp_path / "usage.json")
    now = datetime.now(UTC).isoformat()
    record(tracker, "session-a", "2025-01-01T00:00:00+00:00")
    record(tracker, "session-b", now)

    selected = tracker.get_entries(session_id="session-a", time_window="all")

    assert [entry.session_id for entry in selected] == ["session-a"]
    assert {entry.session_id for entry in tracker.get_entries(time_window="all")} == {
        "session-a",
        "session-b",
    }
    assert [entry.session_id for entry in tracker.get_entries("session-b", "today")] == [
        "session-b"
    ]
    assert tracker.get_entries("session-a", "today") == []


def test_usage_ledger_round_trips_and_rejects_unknown_window(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    tracker = UsageTracker(ledger_path=path)
    record(tracker, "session-a", "2025-01-01T00:00:00+00:00")

    reloaded = UsageTracker(ledger_path=path)

    assert len(reloaded.get_entries()) == 1
    assert reloaded.get_entries()[0].session_id == "session-a"
    with pytest.raises(ValueError, match="unknown usage time window"):
        reloaded.get_entries(time_window="week")


@pytest.mark.asyncio
async def test_llm_client_records_usage_against_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = UsageTracker(ledger_path=tmp_path / "usage.json")
    monkeypatch.setattr(usage_module, "_GLOBAL_TRACKER", tracker)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            },
        )

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler),
        env={"OPENAI_API_KEY": "sk-test"},
    )
    binding = StageModelBinding(provider=ProviderName.OPENAI, model="gpt-4o-mini")

    await client.stream_chat(
        binding,
        [{"role": "user", "content": "hi"}],
        stage="hammer",
        session_id="foreground-session",
    )

    entries = tracker.get_entries(session_id="foreground-session", time_window="all")
    assert len(entries) == 1
    assert entries[0].stage == "hammer"
    assert entries[0].total_tokens == 11
    assert json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))[0][
        "session_id"
    ] == "foreground-session"

