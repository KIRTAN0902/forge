"""Everything a tool needs to run, shared across one workspace session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from forge.tools.analysis.symbols import SymbolIndex
from forge.tools.git import GitRepo
from forge.tools.processes import ProcessManager
from forge.tools.shell import ShellRunner
from forge.tools.workspace import Workspace


@dataclass
class ToolEnvironment:
    workspace: Workspace
    shell: ShellRunner
    processes: ProcessManager = field(default_factory=ProcessManager)
    symbols: SymbolIndex = field(default_factory=SymbolIndex)
    test_command: str | None = None
    _git: GitRepo | None = field(default=None, init=False, repr=False)
    _git_checked: bool = field(default=False, init=False, repr=False)

    async def git(self) -> GitRepo | None:
        """The git repository containing the workspace, if any (discovered once)."""
        if not self._git_checked:
            self._git = await GitRepo.discover(self.workspace.root)
            self._git_checked = True
        return self._git

    async def refresh_symbols(self) -> None:
        def refresh() -> None:
            self.symbols.refresh(self.workspace, self.workspace.files())

        await asyncio.to_thread(refresh)

    async def aclose(self) -> None:
        await self.processes.stop_all()
