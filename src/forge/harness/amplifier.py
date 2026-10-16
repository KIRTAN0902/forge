"""Forge Amplifier Mode: Advanced capabilities for Frontier models (Gemini, Claude, GPT-4o, DeepSeek-R1)."""

from __future__ import annotations

import collections
import inspect
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# 1. Live Runtime Variable Tracer ([HRN-04])
# ==============================================================================


class TraceRecord(BaseModel):
    """Single execution event recorded in the tracer ring buffer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float
    filename: str
    lineno: int
    func_name: str
    event: str
    local_vars: dict[str, str] = Field(default_factory=dict)
    exception_repr: str | None = None


class LiveRuntimeTracer:
    """Live runtime execution and variable tracer using sys.monitoring or sys.settrace ring buffer."""

    def __init__(self, max_records: int = 60) -> None:
        self._max_records = max_records
        self._buffer: collections.deque[TraceRecord] = collections.deque(maxlen=max_records)
        self._active = False
        self._prev_trace: Any = None
        self._monitoring_tool_id: int | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def _should_filter_file(self, filename: str) -> bool:
        """Filter out stdlib, site-packages, and test framework internal modules."""
        lower = filename.lower()
        return (
            "site-packages" in lower
            or "\\lib\\" in lower
            or "/lib/" in lower
            or "pytest" in lower
            or "pluggy" in lower
            or "<" in lower
        )

    def _capture_frame(self, frame: FrameType, event: str, arg: Any = None) -> None:
        """Capture local variables and line metadata into ring buffer."""
        if not self._active:
            return

        code = frame.f_code
        filename = code.co_filename
        if self._should_filter_file(filename):
            return

        lineno = frame.f_lineno
        func_name = code.co_name

        captured_locals: dict[str, str] = {}
        for k, v in list(frame.f_locals.items()):
            if k.startswith("__"):
                continue
            try:
                val_repr = repr(v)
                if len(val_repr) > 80:
                    val_repr = val_repr[:77] + "..."
                captured_locals[k] = val_repr
            except Exception:
                captured_locals[k] = "<unprintable>"

        exc_str: str | None = None
        if event == "exception" and arg is not None:
            exc_str = f"{arg[0].__name__ if hasattr(arg[0], '__name__') else str(arg[0])}: {arg[1]}"

        self._buffer.append(
            TraceRecord(
                timestamp=time.time(),
                filename=Path(filename).name,
                lineno=lineno,
                func_name=func_name,
                event=event,
                local_vars=captured_locals,
                exception_repr=exc_str,
            )
        )

    def _trace_dispatch(self, frame: FrameType, event: str, arg: Any) -> Any:
        """sys.settrace dispatch function."""
        if not self._active:
            return None
        if event in {"line", "call", "exception", "return"}:
            self._capture_frame(frame, event, arg)
        return self._trace_dispatch

    def start(self) -> None:
        """Activate the runtime tracer."""
        if self._active:
            return

        self._buffer.clear()
        self._active = True

        # Use sys.monitoring if on Python >= 3.12 and available, otherwise sys.settrace
        monitoring_mod: Any = getattr(sys, "monitoring", None)
        if monitoring_mod is not None:
            try:
                tool_id = monitoring_mod.DEBUGGER_ID
                monitoring_mod.use_tool_id(tool_id, "forge_tracer")
                self._monitoring_tool_id = tool_id

                # Set global events
                monitoring_mod.set_events(
                    tool_id,
                    monitoring_mod.events.LINE | monitoring_mod.events.EXCEPTION,
                )

                def on_line(code: Any, instruction_offset: int) -> Any:
                    if not self._active:
                        return monitoring_mod.DISABLE
                    frame = inspect.currentframe()
                    if frame and frame.f_back:
                        self._capture_frame(frame.f_back, "line")
                    return monitoring_mod.DISABLE

                def on_exception(code: Any, instruction_offset: int, exc: Any) -> Any:
                    if not self._active:
                        return monitoring_mod.DISABLE
                    frame = inspect.currentframe()
                    if frame and frame.f_back:
                        self._capture_frame(frame.f_back, "exception", (type(exc), exc, None))
                    return monitoring_mod.DISABLE

                monitoring_mod.register_callback(tool_id, monitoring_mod.events.LINE, on_line)
                monitoring_mod.register_callback(
                    tool_id, monitoring_mod.events.EXCEPTION, on_exception
                )
                return
            except (ValueError, AttributeError):
                # Fall back to sys.settrace
                pass

        self._prev_trace = sys.gettrace()
        sys.settrace(self._trace_dispatch)

    def stop(self) -> None:
        """Stop tracing and unhook handlers."""
        if not self._active:
            return

        self._active = False

        monitoring_mod: Any = getattr(sys, "monitoring", None)
        if self._monitoring_tool_id is not None and monitoring_mod is not None:
            try:
                monitoring_mod.set_events(self._monitoring_tool_id, 0)
                monitoring_mod.register_callback(
                    self._monitoring_tool_id, monitoring_mod.events.LINE, None
                )
                monitoring_mod.register_callback(
                    self._monitoring_tool_id, monitoring_mod.events.EXCEPTION, None
                )
                monitoring_mod.free_tool_id(self._monitoring_tool_id)
            except Exception:
                pass
            self._monitoring_tool_id = None
        else:
            sys.settrace(self._prev_trace)

    def __enter__(self) -> LiveRuntimeTracer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def get_records(self) -> list[TraceRecord]:
        return list(self._buffer)

    def get_trace_summary(self) -> str:
        """Format captured execution ring buffer for prompt injection ([DBG-01])."""
        records = list(self._buffer)
        if not records:
            return "No runtime trace events captured."

        lines = [f"[LIVE RUNTIME EXECUTION TRACE (Last {len(records)} Events)]"]
        for record in records:
            var_str = ", ".join(f"{k}={v}" for k, v in record.local_vars.items())
            if not var_str:
                var_str = "<none>"
            entry = f"  -> {record.filename}:{record.lineno} ({record.func_name}) [{record.event.upper()}] | vars: {var_str}"
            if record.exception_repr:
                entry += f" | EXCEPTION: {record.exception_repr}"
            lines.append(entry)

        return "\n".join(lines)


# ==============================================================================
# 2. Pre-Flight Falsifier ([HRN-05]) & Chesterton's Fence ([REF-01])
# ==============================================================================


class FalsificationCondition(BaseModel):
    """Observable condition formulated before high-impact changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assumption: str
    observable_falsifier: str
    verified: bool = False
    rationale: str = ""


class PreFlightFalsifier:
    """Pre-flight check forcing models to falsify assumptions before modifying code."""

    @staticmethod
    def formulate_condition(assumption: str, observable_falsifier: str) -> FalsificationCondition:
        """Create structured falsification contract."""
        return FalsificationCondition(
            assumption=assumption.strip(),
            observable_falsifier=observable_falsifier.strip(),
            verified=False,
            rationale="Formulated pre-flight condition prior to code mutation.",
        )

    @staticmethod
    def verify_condition_against_stderr(
        condition: FalsificationCondition, stderr_or_output: str
    ) -> FalsificationCondition:
        """Evaluate if observable condition was satisfied by runtime output."""
        falsifier = condition.observable_falsifier.lower()
        matched = falsifier in stderr_or_output.lower()
        return FalsificationCondition(
            assumption=condition.assumption,
            observable_falsifier=condition.observable_falsifier,
            verified=matched,
            rationale=(
                f"Observable falsifier '{condition.observable_falsifier}' "
                f"{'was successfully observed' if matched else 'was NOT observed'} in execution output."
            ),
        )


def check_chestertons_fence(
    file_path: str | Path,
    line_range: tuple[int, int],
    cwd: Path | None = None,
) -> str:
    """Retrieve historical git commit intent for lines before refactoring ([REF-01])."""
    p = Path(file_path)
    start_line, end_line = line_range
    repo_cwd = cwd or Path.cwd()

    try:
        cmd = [
            "git",
            "log",
            f"-L{start_line},{end_line}:{p.as_posix()}",
            "-n",
            "1",
            "--pretty=format:Commit %h by %an (%ar): %s%n%b",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo_cwd),
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return (
                f"[CHESTERTON'S FENCE: Historical Intent for {p.name}:{start_line}-{end_line}]\n"
                f"{proc.stdout.strip()}"
            )
        return f"No git history found for {p.name}:{start_line}-{end_line}."
    except Exception as e:
        return f"Could not inspect Chesterton's Fence: {e}"
