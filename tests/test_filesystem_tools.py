"""Real filesystem tool tests through validation, policy, and ToolRuntime dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge.models import ToolCall
from forge.policies import StaticApprover
from forge.tools.workspace import Workspace, WorkspaceError


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"{name}-call", name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_write_file_creates_nested_unicode_paths(tmp_path: Path, make_runtime: Any) -> None:
    runtime = make_runtime(tmp_path)
    content = "café\nsecond line\n"

    result = await runtime.execute(
        call("write_file", path="pkg/données/résumé.txt", content=content)
    )

    assert result.ok, result.content
    target = tmp_path / "pkg/données/résumé.txt"
    assert target.read_text(encoding="utf-8") == content
    assert result.data is not None
    assert result.data["created"] is True
    assert result.changed_paths == ["pkg/données/résumé.txt"]
    assert "+café" in result.data["diff"]


def test_write_file_is_atomic_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path)
    target = tmp_path / "safe.txt"
    target.write_text("original\n", encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("forge.tools.workspace.os.replace", fail_replace)

    with pytest.raises(WorkspaceError, match=r"could not write safe\.txt"):
        workspace.write_text("safe.txt", "replacement\n")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".safe.txt.*.tmp")) == []


@pytest.mark.asyncio
async def test_create_directory_is_real_and_idempotent(
    tmp_path: Path, make_runtime: Any
) -> None:
    runtime = make_runtime(tmp_path)
    args = {"path": "build/generated assets"}

    first = await runtime.execute(call("create_directory", **args))
    second = await runtime.execute(call("create_directory", **args))

    assert first.ok, first.content
    assert second.ok, second.content
    assert (tmp_path / "build/generated assets").is_dir()
    assert first.data is not None and first.data["created"] is True
    assert second.data is not None and second.data["created"] is False
    assert first.changed_paths == ["build/generated assets"]
    assert second.changed_paths == []


@pytest.mark.asyncio
async def test_filesystem_rejects_traversal_and_type_conflicts(
    tmp_path: Path, make_runtime: Any
) -> None:
    runtime = make_runtime(tmp_path)
    (tmp_path / "occupied").write_text("not a directory", encoding="utf-8")

    escaped = await runtime.execute(
        call("write_file", path="../outside.txt", content="escape")
    )
    conflict = await runtime.execute(call("create_directory", path="occupied"))

    assert escaped.ok is False
    assert "outside the workspace" in (escaped.error or "")
    assert not (tmp_path.parent / "outside.txt").exists()
    assert conflict.ok is False
    assert "not a directory" in (conflict.error or "")


@pytest.mark.asyncio
async def test_delete_file_requires_recursive_for_directories(
    tmp_path: Path, make_runtime: Any
) -> None:
    runtime = make_runtime(tmp_path, approver=StaticApprover(True))
    folder = tmp_path / "generated"
    folder.mkdir()
    (folder / "one.txt").write_text("1", encoding="utf-8")
    (folder / "two.txt").write_text("2", encoding="utf-8")

    refused = await runtime.execute(call("delete_file", path="generated"))
    deleted = await runtime.execute(call("delete_file", path="generated", recursive=True))

    assert refused.ok is False
    assert "set recursive=true" in (refused.error or "")
    assert deleted.ok is True
    assert deleted.data is not None
    assert deleted.data["deleted_files"] == 2
    assert not folder.exists()
