"""Long-running background processes (dev servers, watchers).

Each process keeps a rolling tail of its merged output. Every process is killed when the
environment closes, so an agent can never leave orphans behind.
"""

from __future__ import annotations

import asyncio
import codecs
import time
from dataclasses import dataclass, field
from pathlib import Path

from forge.core.errors import ToolError
from forge.tools.shell import ShellRunner, kill_tree, spawn


@dataclass
class ManagedProcess:
    id: str
    command: str
    cwd: Path
    process: asyncio.subprocess.Process
    started_at: float
    output: list[str] = field(default_factory=list)
    output_chars: int = 0
    total_chars: int = 0
    read_offset: int = 0
    reader: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.process.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self.process.returncode


class ProcessManager:
    def __init__(self, *, max_processes: int = 8, max_buffer_chars: int = 200_000) -> None:
        self.max_processes = max_processes
        self.max_buffer_chars = max_buffer_chars
        self._processes: dict[str, ManagedProcess] = {}
        self._counter = 0

    async def start(self, runner: ShellRunner, command: str, cwd: Path) -> ManagedProcess:
        running = [p for p in self._processes.values() if p.running]
        if len(running) >= self.max_processes:
            raise ToolError(f"{len(running)} background processes already running; stop one first")
        self._counter += 1
        proc_id = f"proc_{self._counter}"
        process = await spawn(runner.spec, command, cwd=cwd, env=runner.env, merge_stderr=True)
        managed = ManagedProcess(
            id=proc_id, command=command, cwd=cwd, process=process, started_at=time.time()
        )
        managed.reader = asyncio.create_task(self._pump(managed, runner))
        self._processes[proc_id] = managed
        return managed

    async def _pump(self, managed: ManagedProcess, runner: ShellRunner) -> None:
        stream = managed.process.stdout
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            decoded = decoder.decode(chunk)
            if not decoded:
                continue
            text = runner.redact(decoded.replace("\r\n", "\n"))
            managed.output.append(text)
            managed.output_chars += len(text)
            managed.total_chars += len(text)
            while managed.output_chars > self.max_buffer_chars and len(managed.output) > 1:
                dropped = managed.output.pop(0)
                managed.output_chars -= len(dropped)

    def get(self, proc_id: str) -> ManagedProcess:
        try:
            return self._processes[proc_id]
        except KeyError:
            known = ", ".join(self._processes) or "none"
            raise ToolError(f"unknown process {proc_id!r} (known: {known})") from None

    def read_new(self, proc_id: str, max_chars: int) -> tuple[str, int]:
        """Output produced since the last read (bounded). Returns (text, chars skipped)."""
        managed = self.get(proc_id)
        buffered_start = managed.total_chars - managed.output_chars
        start = max(managed.read_offset, buffered_start)
        skipped = start - managed.read_offset
        text = "".join(managed.output)[start - buffered_start :]
        if len(text) > max_chars:
            skipped += len(text) - max_chars
            text = text[-max_chars:]
        managed.read_offset = managed.total_chars
        return text, skipped

    def list(self) -> list[ManagedProcess]:
        return list(self._processes.values())

    async def stop(self, proc_id: str) -> ManagedProcess:
        managed = self.get(proc_id)
        await kill_tree(managed.process)
        if managed.reader is not None:
            try:
                await asyncio.wait_for(managed.reader, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                managed.reader.cancel()
        return managed

    async def stop_all(self) -> None:
        for proc_id, managed in list(self._processes.items()):
            if managed.running:
                await self.stop(proc_id)
