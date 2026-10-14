from __future__ import annotations

import logging
from pathlib import Path

import pytest

from forge.core.contracts import WorkflowState, WorkflowStatus
from forge.core.errors import NotFoundError, StorageError
from forge.events import Event, EventBus, EventType
from forge.storage import SQLiteStore, Store


class TestEventBus:
    async def test_sync_and_async_handlers_receive_events(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def sync_handler(event: Event) -> None:
            seen.append(f"sync:{event.type}")

        async def async_handler(event: Event) -> None:
            seen.append(f"async:{event.type}")

        bus.subscribe(sync_handler)
        bus.subscribe(async_handler)
        await bus.emit(EventType.WORKFLOW_STARTED, workflow_id="wf_1", request="x")
        assert seen == ["sync:WorkflowStarted", "async:WorkflowStarted"]

    async def test_type_filter_and_unsubscribe(self) -> None:
        bus = EventBus()
        seen: list[EventType] = []
        unsubscribe = bus.subscribe(lambda e: seen.append(e.type), types=[EventType.TOOL_CALLED])
        await bus.emit(EventType.WORKFLOW_STARTED)
        await bus.emit(EventType.TOOL_CALLED)
        unsubscribe()
        await bus.emit(EventType.TOOL_CALLED)
        assert seen == [EventType.TOOL_CALLED]

    async def test_failing_subscriber_does_not_break_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = EventBus()
        seen: list[EventType] = []

        def boom(event: Event) -> None:
            raise RuntimeError("subscriber bug")

        bus.subscribe(boom)
        bus.subscribe(lambda e: seen.append(e.type))
        logging.getLogger("forge").propagate = True
        try:
            await bus.emit(EventType.PLAN_CREATED)
        finally:
            logging.getLogger("forge").propagate = False
        assert seen == [EventType.PLAN_CREATED]
        assert "event subscriber failed" in caplog.text

    def test_payload_is_made_json_safe(self) -> None:
        event = Event(type=EventType.FILE_MODIFIED, payload={"path": Path("a/b.py")})
        assert event.payload == {"path": str(Path("a/b.py"))}


class TestSQLiteStore:
    async def test_satisfies_protocol(self, store: SQLiteStore) -> None:
        assert isinstance(store, Store)

    async def test_workflow_round_trip_and_update(self, store: SQLiteStore) -> None:
        state = WorkflowState(user_request="Add OAuth")
        await store.save_workflow(state)
        state.status = WorkflowStatus.RUNNING
        state.iteration = 2
        await store.save_workflow(state)

        loaded = await store.load_workflow(state.workflow_id)
        assert loaded.status == WorkflowStatus.RUNNING
        assert loaded.iteration == 2

        summaries = await store.list_workflows()
        assert len(summaries) == 1
        assert summaries[0].workflow_id == state.workflow_id
        assert await store.list_workflows(status=WorkflowStatus.COMPLETED) == []

    async def test_missing_workflow(self, store: SQLiteStore) -> None:
        with pytest.raises(NotFoundError):
            await store.load_workflow("wf_missing")

    async def test_events_are_ordered_and_filterable(self, store: SQLiteStore) -> None:
        bus = EventBus()
        bus.subscribe(store.append_event)
        await bus.emit(EventType.WORKFLOW_STARTED, workflow_id="wf_a")
        await bus.emit(EventType.TOOL_CALLED, workflow_id="wf_a", tool="read_file")
        await bus.emit(EventType.WORKFLOW_STARTED, workflow_id="wf_b")
        await bus.emit(EventType.WORKFLOW_COMPLETED, workflow_id="wf_a")

        events = await store.list_events("wf_a")
        assert [e.type for e in events] == [
            EventType.WORKFLOW_STARTED,
            EventType.TOOL_CALLED,
            EventType.WORKFLOW_COMPLETED,
        ]
        assert events[1].payload == {"tool": "read_file"}
        tool_events = await store.list_events(types=[EventType.TOOL_CALLED])
        assert len(tool_events) == 1
        assert len(await store.list_events(limit=2)) == 2

    async def test_persists_across_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "forge.db"
        store = await SQLiteStore.open(db)
        state = WorkflowState(user_request="persist me")
        await store.save_workflow(state)
        await store.close()

        reopened = await SQLiteStore.open(db)
        try:
            assert (await reopened.load_workflow(state.workflow_id)).user_request == "persist me"
        finally:
            await reopened.close()

    async def test_rejects_newer_schema(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "future.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 999")
        conn.close()
        with pytest.raises(StorageError, match="newer"):
            await SQLiteStore.open(db)
