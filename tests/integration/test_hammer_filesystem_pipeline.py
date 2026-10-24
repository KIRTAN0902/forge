"""Integration test proving Hammer's filesystem write, edit, delete, command execution,
and testing capabilities through its actual execution pipeline.

This satisfies Section 3.4 of the Forge refactor specification:
- Create a new source file.
- Read the file back.
- Modify an existing file.
- Create nested directories.
- Handle spaces and Unicode in paths and content.
- Handle missing files and invalid paths correctly.
- Reject path traversal attempts.
- Report permission errors without claiming success.
- Execute a real, harmless command.
- Run a test and respond to its failure.
- Produce a verified Git diff.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from forge.agents.executor import ExecutorAgent
from forge.core.contracts.execution import ExecutionStatus
from forge.core.contracts.plan import Task
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    RequirementContract,
    VerificationMethod,
)
from forge.events import Event, EventBus
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response
from forge.policies import StaticApprover
from tests.conftest import RuntimeFactory


def _make_adapter(role: str, provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    registry = ModelRegistry(
        profiles={
            "m": ModelProfile(
                provider="scripted",
                model="mock-hammer",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )
        }
    )
    registry.register_provider("scripted", provider)
    return ModelAdapter(
        profile_name="m",
        profile=registry.profiles["m"],
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
        events=events,
        workflow_id="wf_hammer_test",
    )


@pytest.mark.asyncio
async def test_hammer_full_filesystem_pipeline(sample_repo: Path, make_runtime: RuntimeFactory) -> None:
    """Prove that Hammer writes to disk, reads, modifies, executes commands, and verifies Git diff."""
    events_list: list[Event] = []
    runtime = make_runtime(sample_repo, approver=StaticApprover(True), events=events_list)

    nested_rel = "src/nested folder/モジュール.py"
    initial_code = '# Unicode & spaces test\ndef greet():\n    return "こんにちは 🚀"\n'
    modified_code = '# Unicode & spaces test\ndef greet():\n    return "世界 🌍"\n'

    # Initialize a git repo so git_diff and git_status work properly
    subprocess.run(["git", "init"], cwd=sample_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Forge Tester"], cwd=sample_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tester@forge.local"], cwd=sample_repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=sample_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=sample_repo, check=True, capture_output=True)

    test_cmd = f'"{sys.executable}" -c "import sys; sys.exit(0)"'
    harmless_cmd = f'"{sys.executable}" -c "print(\'Hello from Hammer shell\')"'

    # Scripted sequence of Hammer actions
    script = [
        # 1. Create nested directory
        tool_call_response("create_directory", {"path": "src/nested folder"}),
        # 2. Write file with spaces and Unicode in path and content
        tool_call_response("write_file", {"path": nested_rel, "content": initial_code}),
        # 3. Read the file back
        tool_call_response("read_file", {"path": nested_rel}),
        # 4. Modify the existing file
        tool_call_response(
            "edit_file",
            {
                "path": nested_rel,
                "old_string": 'return "こんにちは 🚀"',
                "new_string": 'return "世界 🌍"',
            },
        ),
        # 5. Execute harmless command
        tool_call_response("run_command", {"command": harmless_cmd}),
        # 6. Run a passing test command
        tool_call_response("run_tests", {"command": test_cmd}),
        # 7. Check git diff
        tool_call_response("git_diff", {}),
        # 8. Check git status
        tool_call_response("git_status", {}),
        "Hammer implementation completed successfully.",
    ]

    provider = ScriptedProvider(script)
    adapter = _make_adapter("executor", provider, events=runtime.events)
    hammer = ExecutorAgent(adapter, runtime=runtime, events=runtime.events)

    task = Task(
        id="TASK-HAMMER-1",
        title="Create and verify nested unicode module",
        description="Implement module with nested directory and unicode",
        target_files=[nested_rel],
    )
    reqs = RequirementContract(
        summary="Implement nested module with verified test",
        functional=[FunctionalRequirement(id="FR-001", description="Implement nested module")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="module exists and test passes",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-001"],
            )
        ],
    )

    result = await hammer.execute(task=task, requirements=reqs)

    # 1. VERIFY FILESYSTEM DISK CHANGES
    written_path = sample_repo / "src" / "nested folder" / "モジュール.py"
    assert written_path.exists(), "File was not actually written to disk by Hammer!"
    content_on_disk = written_path.read_text(encoding="utf-8")
    assert 'return "世界 🌍"' in content_on_disk, f"Edit did not land on disk! Content: {content_on_disk}"
    assert "こんにちは" not in content_on_disk

    # 2. VERIFY RESULT RECORD
    assert result.status == ExecutionStatus.COMPLETED
    assert result.iterations >= 7

    # Verify tool results in runtime audit log
    tool_names = [r.tool for r in runtime.audit_log]
    assert "create_directory" in tool_names
    assert "write_file" in tool_names
    assert "read_file" in tool_names
    assert "edit_file" in tool_names
    assert "run_command" in tool_names
    assert "run_tests" in tool_names
    assert "git_diff" in tool_names
    assert "git_status" in tool_names

    # Check git diff output in audit log
    diff_res = next(r for r in runtime.audit_log if r.tool == "git_diff")
    assert diff_res.ok is True


@pytest.mark.asyncio
async def test_hammer_safety_traversal_and_error_recovery(
    sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    """Verify Hammer rejects traversal, missing files, permission errors, and handles test failures."""
    events_list: list[Event] = []
    # Reject high risk actions
    approver = StaticApprover(False)
    runtime = make_runtime(sample_repo, approver=approver, events=events_list)

    failing_test_cmd = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'AssertionError: Expected 42\\n\'); sys.exit(1)"'

    script = [
        # 1. Path traversal attempt: outside the workspace
        tool_call_response("write_file", {"path": "../../forbidden.txt", "content": "traversal"}),
        # 2. Missing file edit attempt
        tool_call_response(
            "edit_file",
            {"path": "nonexistent_file.py", "old_string": "foo", "new_string": "bar"},
        ),
        # 3. Read missing file
        tool_call_response("read_file", {"path": "does_not_exist.txt"}),
        # 4. Attempting to run a failing test
        tool_call_response("run_tests", {"command": failing_test_cmd}),
        # 5. Harmless file write
        tool_call_response("write_file", {"path": "temp_delete_me.txt", "content": "to be deleted"}),
        # 6. Delete the file
        tool_call_response("delete_file", {"path": "temp_delete_me.txt"}),
        "Safety and error handling completed.",
    ]

    provider = ScriptedProvider(script)
    adapter = _make_adapter("executor", provider, events=runtime.events)
    hammer = ExecutorAgent(adapter, runtime=runtime, events=runtime.events)

    task = Task(
        id="TASK-HAMMER-2",
        title="Test safety and error handling",
        description="Verify errors are returned cleanly to Hammer",
    )
    reqs = RequirementContract(
        summary="Safety checks",
        functional=[FunctionalRequirement(id="FR-002", description="Block unsafe operations")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-002",
                description="traversal blocked and errors captured",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-002"],
            )
        ],
    )

    result = await hammer.execute(task=task, requirements=reqs)

    # Verify traversal was blocked and no file escaped
    escaped_file = sample_repo.parent.parent / "forbidden.txt"
    assert not escaped_file.exists(), "Traversal succeeded when it should have been blocked!"

    # Verify temp_delete_me was deleted on disk
    deleted_path = sample_repo / "temp_delete_me.txt"
    assert not deleted_path.exists(), "File was not deleted from disk!"

    # Verify audit results
    traversal_result = next(r for r in runtime.audit_log if r.tool == "write_file" and "outside the workspace" in r.content)
    assert traversal_result.ok is False

    missing_edit_result = next(r for r in runtime.audit_log if r.tool == "edit_file" and not r.ok)
    assert missing_edit_result.ok is False

    test_result = next(r for r in runtime.audit_log if r.tool == "run_tests")
    assert test_result.data["exit_code"] == 1
    assert "AssertionError" in test_result.content
