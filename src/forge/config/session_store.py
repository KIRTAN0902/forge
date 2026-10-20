"""Global Cross-Directory Conversation History & Session Store for Forge.

Persists every conversation turn, stage buffers (Anvil, Hammer, Temper), cycle state,
and token telemetry to ``~/.forge/sessions/{session_id}.json`` with an index at
``~/.forge/sessions/index.json`` so any session can be resumed from any directory.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge.config.settings import default_config_dir

if TYPE_CHECKING:
    from forge.ui.state import ForgeTUIState


def default_sessions_dir() -> Path:
    """Return persistent global sessions directory (~/.forge/sessions)."""
    return default_config_dir() / "sessions"


@dataclass
class SessionSummary:
    """Brief metadata indexed for fast lookup and display in the session picker."""

    session_id: str
    title: str
    created_at: str
    updated_at: str
    cwd: str
    total_tokens: int
    cycle_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSummary:
        return cls(
            session_id=str(data.get("session_id", "")),
            title=str(data.get("title", "(untitled)")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            cwd=str(data.get("cwd", "")),
            total_tokens=int(data.get("total_tokens", 0)),
            cycle_count=int(data.get("cycle_count", 1)),
        )


class SessionStore:
    """Saves and restores full conversation sessions globally."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = (
            Path(sessions_dir) if sessions_dir is not None else default_sessions_dir()
        )
        self.index_path = self.sessions_dir / "index.json"
        self._index: dict[str, SessionSummary] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.is_file():
            self._index = {}
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._index = {
                    item["session_id"]: SessionSummary.from_dict(item)
                    for item in data
                    if isinstance(item, dict) and "session_id" in item
                }
            elif isinstance(data, dict):
                self._index = {
                    k: SessionSummary.from_dict(v)
                    for k, v in data.items()
                    if isinstance(v, dict)
                }
        except Exception:
            self._index = {}

    def _save_index(self) -> None:
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            summaries = sorted(
                self._index.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )
            payload = json.dumps([s.to_dict() for s in summaries], indent=2)
            self.index_path.write_text(payload, encoding="utf-8")
        except Exception:
            pass

    def save_session(
        self,
        state: ForgeTUIState,
        session_id: str,
        cwd: Path | str,
        title: str | None = None,
    ) -> Path:
        """Persist entire state to ~/.forge/sessions/{session_id}.json."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        session_file = self.sessions_dir / f"{session_id}.json"
        now = datetime.now(UTC).isoformat()

        # Derive title if missing
        if not title:
            existing = self._index.get(session_id)
            if existing and existing.title and existing.title != "(untitled)":
                title = existing.title
            else:
                from forge.ui.state import ForgeStage

                # Find first user message from any stage buffer
                for stg in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
                    buf = state.stage_buffers.get(stg)
                    if buf and buf.chat_history:
                        for speaker, msg in buf.chat_history:
                            if speaker.lower() in {"you", "user"}:
                                title = msg.strip().splitlines()[0][:60]
                                break
                    if title:
                        break
        if not title:
            title = "(untitled session)"

        # Determine created_at
        created_at = now
        if session_id in self._index:
            created_at = self._index[session_id].created_at

        # Extract stage buffers
        stage_buffers_data: dict[str, Any] = {}
        for stg, buf in state.stage_buffers.items():
            stage_buffers_data[stg.value] = {
                "stage": stg.value,
                "thoughts": list(buf.thoughts),
                "tool_events": list(buf.tool_events),
                "chat_history": [list(item) for item in buf.chat_history],
                "final_output": buf.final_output,
                "stage_tokens": buf.stage_tokens,
                "stage_input_tokens": buf.stage_input_tokens,
                "stage_output_tokens": buf.stage_output_tokens,
                "status": buf.status,
            }

        pending_plan_data: dict[str, Any] | None = None
        if state.pending_plan is not None:
            pending_plan_data = {
                "request": state.pending_plan.request,
                "plan_text": state.pending_plan.plan_text,
                "revision": state.pending_plan.revision,
                "requirements": [
                    {
                        "req_id": req.req_id,
                        "text": req.text,
                        "met": req.met,
                        "evidence": req.evidence,
                    }
                    for req in state.pending_plan.requirements
                ],
            }

        # Full payload
        data = {
            "session_id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": now,
            "cwd": str(cwd),
            "cycle_count": state.cycle_count,
            "cycle_mode": state.cycle_mode.value,
            "active_stage_tokens": state.active_stage_tokens,
            "total_session_tokens": state.total_session_tokens,
            "total_session_input_tokens": state.total_session_input_tokens,
            "total_session_output_tokens": state.total_session_output_tokens,
            "cycle_usage": state.cycle_usage,
            "all_requirements_met": state.all_requirements_met,
            "pending_plan": pending_plan_data,
            "selected_view_stage": state.selected_view_stage.value,
            "stage_buffers": stage_buffers_data,
        }

        with contextlib.suppress(Exception):
            session_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Update index
        summary = SessionSummary(
            session_id=session_id,
            title=title,
            created_at=created_at,
            updated_at=now,
            cwd=str(cwd),
            total_tokens=state.total_session_tokens,
            cycle_count=state.cycle_count,
        )
        self._index[session_id] = summary
        self._save_index()
        return session_file

    def list_sessions(self) -> list[SessionSummary]:
        """List all indexed sessions ordered from newest to oldest."""
        return sorted(
            self._index.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session data from disk."""
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.is_file():
            # Check prefix match
            for p in self.sessions_dir.glob("*.json"):
                if p.stem == "index":
                    continue
                if p.stem.startswith(session_id):
                    session_file = p
                    break
        if not session_file.is_file():
            return None
        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def restore_session(self, session_id: str, state: ForgeTUIState) -> bool:
        """Restore saved session state into an active ForgeTUIState instance."""
        from forge.ui.state import CycleMode, ForgeStage

        data = self.load_session(session_id)
        if not data or not isinstance(data, dict):
            return False

        # Restore cycle count & mode
        state.cycle_count = int(data.get("cycle_count", 1))
        mode_val = data.get("cycle_mode", CycleMode.UNTIL_FULFILLED.value)
        with contextlib.suppress(Exception):
            state.cycle_mode = CycleMode(mode_val)

        state.active_stage_tokens = int(data.get("active_stage_tokens", 0))
        state.total_session_tokens = int(data.get("total_session_tokens", 0))
        state.total_session_input_tokens = int(data.get("total_session_input_tokens", 0))
        state.total_session_output_tokens = int(data.get("total_session_output_tokens", 0))
        state.all_requirements_met = bool(data.get("all_requirements_met", False))
        raw_cycle_usage = data.get("cycle_usage")
        if isinstance(raw_cycle_usage, dict):
            state.cycle_usage = {}
            for cycle_key, stages in raw_cycle_usage.items():
                if not isinstance(stages, dict):
                    continue
                try:
                    cycle_number = int(cycle_key)
                except (TypeError, ValueError):
                    continue
                state.cycle_usage[cycle_number] = {
                    str(stage_name): {
                        "input": int(usage.get("input", 0)),
                        "output": int(usage.get("output", 0)),
                        "total": int(usage.get("total", 0)),
                    }
                    for stage_name, usage in stages.items()
                    if isinstance(usage, dict)
                }

        from forge.ui.state import PendingPlan, PlanRequirement

        raw_plan = data.get("pending_plan")
        if isinstance(raw_plan, dict):
            requirements = [
                PlanRequirement(
                    req_id=str(item.get("req_id", f"REQ-{index}")),
                    text=str(item.get("text", "")),
                    met=bool(item.get("met", False)),
                    evidence=str(item.get("evidence", "")),
                )
                for index, item in enumerate(raw_plan.get("requirements", []), start=1)
                if isinstance(item, dict)
            ]
            state.pending_plan = PendingPlan(
                request=str(raw_plan.get("request", "")),
                plan_text=str(raw_plan.get("plan_text", "")),
                requirements=requirements,
                revision=int(raw_plan.get("revision", 1)),
            )
        else:
            state.pending_plan = None

        # Restore stage buffers
        stage_buffers_data = data.get("stage_buffers") or {}
        for stg in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            s_data = stage_buffers_data.get(stg.value)
            if not isinstance(s_data, dict):
                continue
            buf = state.stage_buffers[stg]
            buf.thoughts = list(s_data.get("thoughts", []))
            buf.tool_events = list(s_data.get("tool_events", []))
            buf.chat_history = [
                (str(item[0]), str(item[1]))
                for item in s_data.get("chat_history", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ]
            buf.final_output = str(s_data.get("final_output", ""))
            buf.stage_tokens = int(s_data.get("stage_tokens", 0))
            buf.stage_input_tokens = int(s_data.get("stage_input_tokens", 0))
            buf.stage_output_tokens = int(s_data.get("stage_output_tokens", 0))
            buf.status = s_data.get("status", "idle")

        view_stg = data.get("selected_view_stage")
        if view_stg:
            for s in ForgeStage:
                if s.value == view_stg:
                    state.selected_view_stage = s
                    break

        state._notify_redraw()
        return True


# Global singleton
_GLOBAL_STORE: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return the global session store singleton."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = SessionStore()
    return _GLOBAL_STORE
