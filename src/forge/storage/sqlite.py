"""SQLite-backed store.

Uses the standard-library driver on a worker thread so the event loop never blocks on
disk I/O. One connection is shared and serialized with an asyncio lock: Forge is a
local single-user engine, so write concurrency is not a concern yet.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.core.errors import NotFoundError, StorageError
from forge.events.types import Event, EventType
from forge.storage.base import WorkflowSummary

T = TypeVar("T")

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE workflows (
        id           TEXT PRIMARY KEY,
        user_request TEXT NOT NULL,
        status       TEXT NOT NULL,
        iteration    INTEGER NOT NULL,
        state_json   TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );
    CREATE INDEX idx_workflows_updated ON workflows(updated_at);
    CREATE TABLE events (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        id           TEXT NOT NULL UNIQUE,
        workflow_id  TEXT,
        type         TEXT NOT NULL,
        source       TEXT NOT NULL,
        at           TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );
    CREATE INDEX idx_events_workflow ON events(workflow_id, seq);
    """,
}


class SQLiteStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path | str) -> SQLiteStore:
        """Open (creating and migrating if needed) a store. ``":memory:"`` is allowed."""

        def connect() -> sqlite3.Connection:
            if str(path) != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _migrate(conn)
            return conn

        try:
            conn = await asyncio.to_thread(connect)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open store at {path}: {exc}") from exc
        return cls(conn)

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        async with self._lock:
            try:
                return await asyncio.to_thread(fn, self._conn)
            except sqlite3.Error as exc:
                raise StorageError(str(exc)) from exc

    # -- workflows -------------------------------------------------------------------

    async def save_workflow(self, state: WorkflowState) -> None:
        state.touch()
        row = (
            state.workflow_id,
            state.user_request,
            str(state.status),
            state.iteration,
            state.model_dump_json(),
            state.created_at.isoformat(),
            state.updated_at.isoformat(),
        )

        def write(conn: sqlite3.Connection) -> None:
            with conn:
                conn.execute(
                    """
                    INSERT INTO workflows
                        (id, user_request, status, iteration, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status=excluded.status,
                        iteration=excluded.iteration,
                        state_json=excluded.state_json,
                        updated_at=excluded.updated_at
                    """,
                    row,
                )

        await self._run(write)

    async def load_workflow(self, workflow_id: str) -> WorkflowState:
        def read(conn: sqlite3.Connection) -> Any:
            return conn.execute(
                "SELECT state_json FROM workflows WHERE id = ?", (workflow_id,)
            ).fetchone()

        row = await self._run(read)
        if row is None:
            raise NotFoundError(f"workflow {workflow_id} not found")
        return WorkflowState.model_validate_json(row["state_json"])

    async def list_workflows(
        self, *, limit: int = 20, status: WorkflowStatus | None = None
    ) -> list[WorkflowSummary]:
        def read(conn: sqlite3.Connection) -> list[Any]:
            sql = (
                "SELECT id, user_request, status, iteration, created_at, updated_at FROM workflows"
            )
            params: list[Any] = []
            if status is not None:
                sql += " WHERE status = ?"
                params.append(str(status))
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            return conn.execute(sql, params).fetchall()

        rows = await self._run(read)
        return [
            WorkflowSummary(
                workflow_id=r["id"],
                user_request=r["user_request"],
                status=WorkflowStatus(r["status"]),
                iteration=r["iteration"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # -- events ----------------------------------------------------------------------

    async def append_event(self, event: Event) -> None:
        row = (
            event.id,
            event.workflow_id,
            str(event.type),
            event.source,
            event.at.isoformat(),
            json.dumps(event.payload),
        )

        def write(conn: sqlite3.Connection) -> None:
            with conn:
                conn.execute(
                    "INSERT INTO events (id, workflow_id, type, source, at, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )

        await self._run(write)

    async def list_events(
        self,
        workflow_id: str | None = None,
        *,
        types: Iterable[EventType] | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        type_list = [str(t) for t in types] if types is not None else None

        def read(conn: sqlite3.Connection) -> list[Any]:
            clauses: list[str] = []
            params: list[Any] = []
            if workflow_id is not None:
                clauses.append("workflow_id = ?")
                params.append(workflow_id)
            if type_list is not None:
                clauses.append(f"type IN ({','.join('?' * len(type_list))})")
                params.extend(type_list)
            sql = "SELECT id, workflow_id, type, source, at, payload_json FROM events"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY seq ASC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            return conn.execute(sql, params).fetchall()

        rows = await self._run(read)
        return [
            Event(
                id=r["id"],
                workflow_id=r["workflow_id"],
                type=EventType(r["type"]),
                source=r["source"],
                at=r["at"],
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.close)


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise StorageError(f"store schema v{current} is newer than this Forge (v{SCHEMA_VERSION})")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        with conn:
            conn.executescript(_MIGRATIONS[version])
            conn.execute(f"PRAGMA user_version = {version}")
