"""The standard toolset, role toolsets, and runtime construction from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.config.schema import ForgeConfig
from forge.events.bus import EventBus
from forge.policies.approval import Approver
from forge.tools.analysis_tools import (
    DependencyAnalysis,
    FileOutline,
    RepositoryStructure,
    SymbolSearch,
)
from forge.tools.base import Tool
from forge.tools.environment import ToolEnvironment
from forge.tools.filesystem import (
    DeleteFile,
    EditFile,
    FindFiles,
    ListDirectory,
    ReadFile,
    Search,
    WriteFile,
)
from forge.tools.git_tools import (
    GitBranch,
    GitCheckpoint,
    GitDiff,
    GitListCheckpoints,
    GitLog,
    GitRollback,
    GitStatus,
)
from forge.tools.runtime import ToolRuntime
from forge.tools.shell import ShellRunner, resolve_shell
from forge.tools.terminal import (
    ListProcesses,
    ProcessOutput,
    RunCommand,
    RunTests,
    StartProcess,
    StopProcess,
)
from forge.tools.workspace import Workspace


def default_tools() -> list[Tool[Any, Any]]:
    return [
        # filesystem
        ReadFile(),
        WriteFile(),
        EditFile(),
        DeleteFile(),
        ListDirectory(),
        Search(),
        FindFiles(),
        # terminal
        RunCommand(),
        RunTests(),
        StartProcess(),
        ProcessOutput(),
        StopProcess(),
        ListProcesses(),
        # git
        GitStatus(),
        GitDiff(),
        GitLog(),
        GitBranch(),
        GitCheckpoint(),
        GitListCheckpoints(),
        GitRollback(),
        # analysis
        RepositoryStructure(),
        SymbolSearch(),
        FileOutline(),
        DependencyAnalysis(),
    ]


READ_ONLY_TOOLS: tuple[str, ...] = (
    "read_file",
    "list_directory",
    "search",
    "find_files",
    "repository_structure",
    "symbol_search",
    "file_outline",
    "dependency_analysis",
    "git_status",
    "git_diff",
    "git_log",
    "git_list_checkpoints",
)

ROLE_TOOLSETS: dict[str, tuple[str, ...]] = {
    # The planner investigates but never changes anything.
    "planner": READ_ONLY_TOOLS,
    # The executor implements, so it gets everything.
    "executor": tuple(t.name for t in default_tools()),
    # QA verifies by running things, but does not edit the implementation it is judging.
    "qa": (
        *READ_ONLY_TOOLS,
        "run_tests",
        "run_command",
        "start_process",
        "process_output",
        "stop_process",
        "list_processes",
    ),
}


def create_environment(config: ForgeConfig, root: Path | None = None) -> ToolEnvironment:
    workspace = Workspace(
        root or config.root,
        ignore=config.tools.ignore,
        secret_paths=config.policy.secret_paths,
    )
    shell = ShellRunner(
        spec=resolve_shell(config.tools.shell),
        passthrough_env=tuple(config.tools.env_passthrough),
        default_timeout_s=config.tools.command_timeout_s,
        max_output_chars=config.tools.max_output_chars,
    )
    return ToolEnvironment(workspace=workspace, shell=shell, test_command=config.tools.test_command)


def create_tool_runtime(
    config: ForgeConfig,
    *,
    root: Path | None = None,
    approver: Approver | None = None,
    events: EventBus | None = None,
    workflow_id: str | None = None,
) -> ToolRuntime:
    return ToolRuntime(
        default_tools(),
        create_environment(config, root),
        config.policy.engine(),
        approver=approver,
        events=events,
        workflow_id=workflow_id,
        max_output_chars=config.tools.max_output_chars,
    )
