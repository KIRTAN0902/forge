from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from forge.events import Event, EventType
from forge.models import ToolCall
from forge.policies import RiskLevel, StaticApprover
from forge.tools import Tool, ToolInput, ToolOutput, ToolRuntime
from forge.tools.environment import ToolEnvironment

from ...conftest import RuntimeFactory

_counter = 0


def call(name: str, /, **arguments: Any) -> ToolCall:
    global _counter
    _counter += 1
    return ToolCall(id=f"c{_counter}", name=name, arguments=arguments)


class TestGate:
    async def test_unknown_tool_lists_available(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(call("rm_everything"))
        assert not result.ok
        assert "unknown tool" in result.content
        assert "read_file" in result.content

    async def test_role_restriction(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(
            call("write_file", path="x.txt", content="x"), allowed=["read_file"]
        )
        assert not result.ok
        assert not (sample_repo / "x.txt").exists()

    async def test_bad_arguments_are_explained(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(call("read_file", file="README.md"))
        assert not result.ok
        assert "path" in result.content
        assert "file" in result.content  # unknown argument reported
        unparsable = ToolCall(id="x", name="read_file", parse_error="invalid JSON")
        assert "could not parse" in (await runtime.execute(unparsable)).content

    async def test_escape_is_an_error_not_a_crash(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(call("read_file", path="../../etc/passwd"))
        assert not result.ok
        assert "outside the workspace" in result.content

    async def test_secret_read_requires_approval(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        events: list[Event] = []
        rejecting = make_runtime(sample_repo, approver=StaticApprover(False), events=events)
        result = await rejecting.execute(call("read_file", path=".env"))
        assert not result.ok
        assert "not approved" in result.content
        assert "super-secret" not in result.content
        types = [e.type for e in events]
        assert EventType.APPROVAL_REQUESTED in types
        assert EventType.TOOL_CALLED not in types  # never ran

        approver = StaticApprover(True)
        approving = make_runtime(sample_repo, approver=approver)
        assert (await approving.execute(call("read_file", path=".env"))).ok
        assert approver.requests[0].paths == [".env"]

    async def test_no_approver_means_rejected(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)  # non-interactive: nobody can approve
        result = await runtime.execute(call("delete_file", path="src", recursive=True))
        assert not result.ok
        assert (sample_repo / "src").exists()

    async def test_events_and_audit(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        events: list[Event] = []
        runtime = make_runtime(sample_repo, events=events)
        result = await runtime.execute(call("write_file", path="notes/a.md", content="# hi\n"))
        assert result.ok
        assert result.changed_paths == ["notes/a.md"]
        types = [e.type for e in events]
        assert types == [
            EventType.POLICY_EVALUATED,
            EventType.TOOL_CALLED,
            EventType.TOOL_COMPLETED,
            EventType.FILE_MODIFIED,
        ]
        assert events[0].payload["verdict"] == "allow"
        assert events[2].payload["ok"] is True
        assert runtime.audit_log[-1] is result

    async def test_large_content_is_summarized_in_audit(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        events: list[Event] = []
        runtime = make_runtime(sample_repo, events=events)
        await runtime.execute(call("write_file", path="big.txt", content="y" * 5000))
        called = next(e for e in events if e.type == EventType.TOOL_CALLED)
        assert len(called.payload["arguments"]["content"]) < 500

    async def test_output_truncation(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        (sample_repo / "long.txt").write_text("\n".join(f"line {i}" * 20 for i in range(3000)))
        runtime = make_runtime(sample_repo, tools={"max_output_chars": 2000})
        result = await runtime.execute(call("read_file", path="long.txt"))
        assert result.ok
        assert result.truncated
        assert len(result.content) < 2300
        assert "characters omitted" in result.content


class _SlowInput(ToolInput):
    pass


class _SlowOutput(ToolOutput):
    pass


class _Slow(Tool[_SlowInput, _SlowOutput]):
    name = "slow"
    description = "sleeps"
    input_model = _SlowInput
    output_model = _SlowOutput
    timeout_s = 0.05

    async def run(self, args: _SlowInput, env: ToolEnvironment) -> _SlowOutput:
        await asyncio.sleep(5)
        return _SlowOutput()


class _Broken(_Slow):
    name = "broken"
    timeout_s = 5

    async def run(self, args: _SlowInput, env: ToolEnvironment) -> _SlowOutput:
        raise KeyError("bug in tool")


async def test_timeouts_and_crashes_become_results(
    sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    base = make_runtime(sample_repo)
    runtime = ToolRuntime([_Slow(), _Broken()], base.env, base.policy)
    slow = await runtime.execute(call("slow"))
    assert not slow.ok
    assert "timed out" in slow.content
    broken = await runtime.execute(call("broken"))
    assert not broken.ok
    assert "KeyError" in broken.content


async def test_execute_many_preserves_order(
    sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    runtime = make_runtime(sample_repo)
    calls = [
        call("read_file", path="README.md"),
        call("search", pattern="def add"),
        call("write_file", path="a.txt", content="1"),
        call("read_file", path="a.txt"),
    ]
    results = await runtime.execute_many(calls)
    assert [r.call_id for r in results] == [c.id for c in calls]
    assert all(r.ok for r in results)
    assert "1" in results[3].content  # read after the write, not before
    messages = ToolRuntime.to_messages(calls, results)
    assert messages[0].tool_call_id == calls[0].id


class TestFilesystemTools:
    async def test_read_paging(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        (sample_repo / "n.txt").write_text("\n".join(str(i) for i in range(1, 101)) + "\n")
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(call("read_file", path="n.txt", offset=10, limit=5))
        assert result.content.splitlines()[0] == "n.txt (lines 10-14 of 100)"
        assert "10\t10" in result.content
        assert "offset=15" in result.content

    async def test_edit_unique_ambiguous_missing(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        fixed = await runtime.execute(
            call(
                "edit_file",
                path="src/calc/ops.py",
                old_string="return a - b  # BUG: should add",
                new_string="return a + b",
            )
        )
        assert fixed.ok, fixed.content
        assert "+    return a + b" in fixed.content
        assert "return a + b" in (sample_repo / "src/calc/ops.py").read_text()

        (sample_repo / "dup.py").write_text("x = 1\nx = 1\n")
        ambiguous = await runtime.execute(
            call("edit_file", path="dup.py", old_string="x = 1", new_string="x = 2")
        )
        assert not ambiguous.ok
        assert "2 times" in ambiguous.content
        everywhere = await runtime.execute(
            call(
                "edit_file", path="dup.py", old_string="x = 1", new_string="x = 2", replace_all=True
            )
        )
        assert everywhere.ok
        assert (sample_repo / "dup.py").read_text() == "x = 2\nx = 2\n"

        missing = await runtime.execute(
            call("edit_file", path="src/calc/ops.py", old_string="def add(a, b):", new_string="x")
        )
        assert not missing.ok
        assert "not found" in missing.content

    async def test_edit_preserves_crlf(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        (sample_repo / "win.txt").write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(
            call("edit_file", path="win.txt", old_string="alpha\nbeta", new_string="ALPHA\nBETA")
        )
        assert result.ok, result.content
        assert (sample_repo / "win.txt").read_bytes() == b"ALPHA\r\nBETA\r\ngamma\r\n"

    async def test_write_and_delete(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(sample_repo, approver=StaticApprover(True))
        created = await runtime.execute(
            call("write_file", path="pkg/new/mod.py", content="x = 1\n")
        )
        assert created.ok
        assert "Created" in created.content
        assert (await runtime.execute(call("delete_file", path="pkg/new/mod.py"))).ok
        assert not (sample_repo / "pkg/new/mod.py").exists()
        directory = await runtime.execute(call("delete_file", path="pkg"))
        assert not directory.ok
        assert "recursive" in directory.content
        assert (await runtime.execute(call("delete_file", path="pkg", recursive=True))).ok
        root = await runtime.execute(call("delete_file", path=".", recursive=True))
        assert not root.ok

    async def test_list_search_find(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(sample_repo)
        listing = await runtime.execute(call("list_directory", depth=2))
        assert "src/ (" in listing.content
        assert "calc/" in listing.content
        assert ".git/" not in listing.content
        assert ".gitignore" in listing.content

        search = await runtime.execute(call("search", pattern=r"^def (add|divide)", glob="*.py"))
        assert search.content.splitlines() == [
            "src/calc/ops.py:4:def add(a: int, b: int) -> int:",
            "src/calc/ops.py:8:def divide(a: float, b: float) -> float:",
        ]

        secret = await runtime.execute(call("search", pattern="super-secret"))
        assert "No matches" in secret.content
        assert "1 secret file(s) not searched" in secret.content

        literal = await runtime.execute(call("search", pattern="a - b", literal=True))
        assert "ops.py:5" in literal.content
        bad_regex = await runtime.execute(call("search", pattern="(unclosed"))
        assert not bad_regex.ok
        assert "literal=true" in bad_regex.content

        found = await runtime.execute(call("find_files", pattern="test_*.py"))
        assert found.content.strip() == "tests/test_ops.py"

    async def test_tool_definitions_are_clean(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        definitions = {d.name: d for d in runtime.definitions()}
        assert len(definitions) == 24
        edit = definitions["edit_file"].parameters
        assert "title" not in edit
        assert set(edit["required"]) == {"path", "old_string", "new_string"}
        assert "PowerShell" in definitions["run_command"].description or "bash" in (
            definitions["run_command"].description
        )


@pytest.mark.parametrize("risk", [RiskLevel.LOW, RiskLevel.MEDIUM])
async def test_low_and_medium_need_no_approval(
    risk: RiskLevel, sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    approver = StaticApprover(False)
    runtime = make_runtime(sample_repo, approver=approver)
    name = "read_file" if risk == RiskLevel.LOW else "write_file"
    args = {"path": "README.md"} if name == "read_file" else {"path": "o.txt", "content": "x"}
    assert (await runtime.execute(call(name, **args))).ok
    assert approver.requests == []
