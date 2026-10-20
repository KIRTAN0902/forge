"""End-to-end proof that Forge writes real files and verifies them with real pytest.

The scripted models only choose tool calls; every write, diff, command and test run is
executed by the real ``ToolRuntime`` against a temporary workspace on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from forge.agents import ExecutorAgent, OrchestratorAgent, QAAgent
from forge.models import Message, ModelAdapter, ModelCapabilities, ModelProfile
from forge.models.providers.scripted import ScriptedProvider, text_response, tool_call_response
from forge.models.types import StreamDone
from forge.ui.cycle_controller import ForgeCycleController
from forge.ui.state import ForgeStage, ForgeTUIState


def _adapter(provider: ScriptedProvider, role: str) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model=f"{role}-test", detect_capabilities=False)
    return ModelAdapter(
        profile_name=role,
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
    )


def _python_command() -> str:
    return f'"{Path(sys.executable).as_posix()}" -m pytest -q'


def _plan_text() -> str:
    return (
        "Create a tiny calculator package and its regression test.\n\n"
        "1. Add `add(a, b)` in `src/calc/ops.py`.\n"
        "2. Export it from `src/calc/__init__.py`.\n"
        "3. Cover it with `tests/test_ops.py` and run pytest.\n\n"
        "REQ-1: src/calc/ops.py defines add(a, b) returning the sum.\n"
        "REQ-2: tests/test_ops.py verifies add and the real pytest command exits 0.\n"
        "TARGET_FILES: src/calc/ops.py, src/calc/__init__.py, tests/test_ops.py"
    )


def _approval() -> str:
    return json.dumps(
        {
            "intent": "APPROVE_AS_IS",
            "reasoning_summary": "The developer authorized immediate implementation.",
        }
    )


def _write(path: str, content: str, call_id: str) -> Any:
    return tool_call_response("write_file", {"path": path, "content": content}, call_id=call_id)


def _test_file() -> str:
    return (
        "import pathlib\n"
        "import sys\n\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from calc.ops import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n"
    )


@pytest.mark.asyncio
async def test_full_cycle_writes_real_files_and_runs_real_pytest(
    tmp_path: Path, make_runtime: Any
) -> None:
    runtime = make_runtime(tmp_path)
    state = ForgeTUIState()

    anvil_provider = ScriptedProvider([text_response(_plan_text()), text_response(_approval())])
    hammer_provider = ScriptedProvider(
        [
            _write(
                "src/calc/__init__.py",
                '"""Calculator package."""\n\nfrom calc.ops import add\n\n__all__ = ["add"]\n',
                "write_init",
            ),
            _write(
                "src/calc/ops.py",
                '"""Arithmetic operations."""\n\n\n'
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n",
                "write_ops",
            ),
            _write("tests/test_ops.py", _test_file(), "write_test"),
            tool_call_response(
                "run_tests",
                {"command": _python_command(), "paths": ["tests"]},
                call_id="run_tests",
            ),
            text_response("Implemented the calculator package and confirmed pytest passes."),
        ]
    )
    temper_provider = ScriptedProvider(
        [
            tool_call_response(
                "run_tests",
                {"command": _python_command(), "paths": ["tests"]},
                call_id="verify_tests",
            ),
            text_response(
                "REQ-1: PASS - src/calc/ops.py defines add and returns the sum.\n"
                "REQ-2: PASS - pytest ran against tests/test_ops.py with exit 0.\n"
                "ALL_REQUIREMENTS_MET: TRUE"
            ),
        ]
    )

    agents = {
        ForgeStage.ANVIL: OrchestratorAgent(
            _adapter(anvil_provider, "orchestrator"), runtime=runtime
        ),
        ForgeStage.HAMMER: ExecutorAgent(
            _adapter(hammer_provider, "executor"), runtime=runtime
        ),
        ForgeStage.TEMPER: QAAgent(_adapter(temper_provider, "qa"), runtime=runtime),
    }
    controller = ForgeCycleController(
        state=state, agents=agents, runtime=runtime, root=tmp_path
    )

    plan_result = await controller.run_cycle_loop(
        "Implement a calculator module with addition and regression tests"
    )
    assert "Final Plan" in plan_result
    assert state.pending_plan is not None
    assert not (tmp_path / "src/calc/ops.py").exists()

    result = await controller.run_cycle_loop("yes, proceed")

    # Real files were physically created by Hammer's tool loop.
    ops = tmp_path / "src/calc/ops.py"
    init = tmp_path / "src/calc/__init__.py"
    tests = tmp_path / "tests/test_ops.py"
    assert ops.is_file()
    assert init.is_file()
    assert tests.is_file()
    assert "return a + b" in ops.read_text(encoding="utf-8")
    assert "from calc.ops import add" in init.read_text(encoding="utf-8")

    assert state.all_requirements_met is True
    assert state.pending_plan is None
    assert state.cycle_count == 1
    assert "ALL_REQUIREMENTS_MET: TRUE" in result

    hammer_events = state.stage_buffers[ForgeStage.HAMMER].tool_events
    temper_events = state.stage_buffers[ForgeStage.TEMPER].tool_events
    assert any("[FILE] Created src/calc/ops.py" in event for event in hammer_events)
    assert any("+def add" in event for event in hammer_events)
    assert any("[CMD] $" in event and "[exit 0" in event for event in hammer_events)
    assert any("[CMD] $" in event and "[exit 0" in event for event in temper_events)
    assert any("passed=1" in event for event in temper_events)
    assert all(not buffer.is_working for buffer in state.stage_buffers.values())


@pytest.mark.asyncio
async def test_streamed_text_tool_call_is_recovered_and_written(
    tmp_path: Path, make_runtime: Any
) -> None:
    """A server that streams the text protocol instead of delta.tool_calls must still write."""
    runtime = make_runtime(tmp_path)
    content = "def answer() -> int:\n    return 42\n"
    call = json.dumps(
        {"name": "write_file", "arguments": {"path": "src/answer.py", "content": content}}
    )
    provider = ScriptedProvider([text_response(f"<tool_call>\n{call}\n</tool_call>")])
    adapter = _adapter(provider, "executor")

    done = None
    async for event in adapter.stream(
        [Message.user("Create src/answer.py")],
        tools=runtime.definitions(["write_file"]),
    ):
        if isinstance(event, StreamDone):
            done = event.response

    assert done is not None
    assert done.tool_calls and done.tool_calls[0].name == "write_file"
    result = await runtime.execute(done.tool_calls[0], allowed=["write_file"])
    assert result.ok
    assert (tmp_path / "src/answer.py").read_text(encoding="utf-8") == content


def test_viewport_renders_real_file_diff_and_command_output() -> None:
    """The stage viewport keeps coloured diff and terminal evidence, not just a status line."""
    state = ForgeTUIState()
    state.append_tool_event(
        ForgeStage.HAMMER,
        "[FILE] Modified src/calc/ops.py (+2 -1)\n"
        "--- a/src/calc/ops.py\n"
        "+++ b/src/calc/ops.py\n"
        "@@ -1,2 +1,3 @@\n"
        "-return a - b\n"
        "+return a + b\n"
        "+\n",
    )
    state.append_tool_event(
        ForgeStage.HAMMER,
        "[CMD] $\n[exit 0, 0.4s]\n2 passed in 0.31s",
    )
    rendered = state.stage_buffers[ForgeStage.HAMMER].render_to_ansi(width=100)
    assert "[FILE] Modified src/calc/ops.py (+2 -1)" in rendered
    assert "+return a + b" in rendered
    assert "-return a - b" in rendered
    assert "[CMD] $" in rendered
    assert "2 passed in 0.31s" in rendered
