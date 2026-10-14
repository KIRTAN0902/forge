"""Running shell commands safely.

* The shell is chosen once and described to the model, so it writes commands for the
  shell that will actually run them (Git Bash is preferred on Windows over cmd).
* Output is captured into bounded buffers (head and tail kept) so a runaway command
  cannot exhaust memory or the model's context.
* On timeout the whole process tree is killed, not just the shell.
* Secret-looking environment variables are withheld from commands by default, and any
  secret value that still appears in output is redacted.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ShellKind = Literal["bash", "sh", "powershell", "cmd"]

_SECRET_ENV = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|AUTH|SESSION|COOKIE|DSN)",
    re.IGNORECASE,
)
_SAFE_ENV = {"SSH_AUTH_SOCK"}  # needed by tooling, not itself a secret
_IS_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class ShellSpec:
    kind: ShellKind
    argv: tuple[str, ...]  # prefix; the command string is appended

    @property
    def display(self) -> str:
        return {"bash": "bash", "sh": "POSIX sh", "powershell": "PowerShell", "cmd": "cmd.exe"}[
            self.kind
        ]


def resolve_shell(preference: str = "auto") -> ShellSpec:
    pref = preference.strip()
    lowered = pref.lower()
    if lowered == "auto":
        if _IS_WINDOWS:
            git_bash = _find_git_bash()
            if git_bash:
                return ShellSpec("bash", (git_bash, "-c"))
            return _powershell()
        bash = shutil.which("bash")
        return ShellSpec("bash", (bash, "-c")) if bash else ShellSpec("sh", ("/bin/sh", "-c"))
    if lowered in {"bash", "sh"}:
        found = (_find_git_bash() if _IS_WINDOWS and lowered == "bash" else None) or shutil.which(
            lowered
        )
        if not found:
            raise ValueError(f"shell {pref!r} not found on PATH")
        return ShellSpec(lowered, (found, "-c"))  # type: ignore[arg-type]
    if lowered in {"powershell", "pwsh"}:
        return _powershell(lowered)
    if lowered == "cmd":
        return ShellSpec("cmd", ("cmd.exe", "/d", "/s", "/c"))
    path = Path(pref)
    name = path.stem.lower()
    if name in {"powershell", "pwsh"}:
        return ShellSpec("powershell", (pref, "-NoProfile", "-NonInteractive", "-Command"))
    if name == "cmd":
        return ShellSpec("cmd", (pref, "/d", "/s", "/c"))
    return ShellSpec("bash" if "bash" in name else "sh", (pref, "-c"))


def _find_git_bash() -> str | None:
    candidates = []
    found = shutil.which("bash")
    # System32\bash.exe and WindowsApps\bash.exe are WSL launchers (runs in another OS).
    if found and "system32" not in found.lower() and "windowsapps" not in found.lower():
        candidates.append(found)
    for base in (
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMW6432"),
        r"C:\Program Files",
    ):
        if base:
            candidates.append(str(Path(base) / "Git" / "bin" / "bash.exe"))
    git = shutil.which("git")
    if git:
        candidates.append(str(Path(git).resolve().parent.parent / "bin" / "bash.exe"))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def _powershell(name: str = "powershell") -> ShellSpec:
    exe = shutil.which("pwsh") if name == "pwsh" else None
    exe = exe or shutil.which("powershell") or shutil.which("pwsh") or "powershell.exe"
    return ShellSpec("powershell", (exe, "-NoProfile", "-NonInteractive", "-Command"))


# -- environment and redaction -----------------------------------------------------------


def scrubbed_environment(
    passthrough: Iterable[str] = (), base: Mapping[str, str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (env for commands, withheld secrets)."""
    source = dict(os.environ if base is None else base)
    allowed = {name.upper() for name in passthrough} | _SAFE_ENV
    env: dict[str, str] = {}
    withheld: dict[str, str] = {}
    for name, value in source.items():
        if _SECRET_ENV.search(name) and name.upper() not in allowed:
            withheld[name] = value
        else:
            env[name] = value
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "CI": env.get("CI", "1"),
        }
    )
    return env, withheld


class SecretRedactor:
    """Replaces known secret values in text."""

    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        self._secrets = sorted(
            ((name, value) for name, value in (secrets or {}).items() if len(value) >= 8),
            key=lambda item: -len(item[1]),
        )

    def __call__(self, text: str) -> str:
        for name, value in self._secrets:
            if value in text:
                text = text.replace(value, f"[REDACTED:{name}]")
        return text


# -- bounded capture -----------------------------------------------------------------------


class BoundedBuffer:
    """Keeps the first and last ``limit // 2`` characters of a stream."""

    def __init__(self, limit: int) -> None:
        self.half = max(limit // 2, 1)
        self.head = ""
        self.tail = ""
        self.dropped = 0

    def write(self, chunk: str) -> None:
        if len(self.head) < self.half:
            take = self.half - len(self.head)
            self.head += chunk[:take]
            chunk = chunk[take:]
        if not chunk:
            return
        self.tail += chunk
        if len(self.tail) > self.half:
            self.dropped += len(self.tail) - self.half
            self.tail = self.tail[-self.half :]

    @property
    def truncated(self) -> bool:
        return self.dropped > 0

    def text(self) -> str:
        if not self.dropped:
            return self.head + self.tail
        return f"{self.head}\n... [{self.dropped} characters omitted] ...\n{self.tail}"


@dataclass
class CommandOutcome:
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False
    shell: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class ShellRunner:
    spec: ShellSpec = field(default_factory=resolve_shell)
    passthrough_env: tuple[str, ...] = ()
    default_timeout_s: float = 300.0
    max_output_chars: int = 30_000

    def __post_init__(self) -> None:
        self.env, withheld = scrubbed_environment(self.passthrough_env)
        self.redact = SecretRedactor(withheld)

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> CommandOutcome:
        timeout = timeout_s or self.default_timeout_s
        env = {**self.env, **(extra_env or {})}
        started = time.perf_counter()
        process = await spawn(self.spec, command, cwd=cwd, env=env, merge_stderr=False)
        out_buf = BoundedBuffer(self.max_output_chars)
        err_buf = BoundedBuffer(self.max_output_chars // 2)
        readers = [
            asyncio.create_task(pump(process.stdout, out_buf)),
            asyncio.create_task(pump(process.stderr, err_buf)),
        ]
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            await kill_tree(process)
        except asyncio.CancelledError:
            await kill_tree(process)
            raise
        finally:
            _, pending = await asyncio.wait(readers, timeout=5)
            for task in pending:
                task.cancel()
        return CommandOutcome(
            command=command,
            exit_code=process.returncode,
            stdout=self.redact(out_buf.text()),
            stderr=self.redact(err_buf.text()),
            duration_s=time.perf_counter() - started,
            timed_out=timed_out,
            truncated=out_buf.truncated or err_buf.truncated,
            shell=self.spec.display,
        )


async def spawn(
    spec: ShellSpec,
    command: str,
    *,
    cwd: Path,
    env: Mapping[str, str],
    merge_stderr: bool,
) -> asyncio.subprocess.Process:
    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return await asyncio.create_subprocess_exec(
        *spec.argv,
        command,
        cwd=str(cwd),
        env=dict(env),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE,
        **kwargs,  # type: ignore[arg-type]
    )


async def pump(stream: asyncio.StreamReader | None, buffer: BoundedBuffer) -> None:
    if stream is None:
        return
    # Incremental decoding: a multi-byte character split across reads stays intact.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await stream.read(65536)
        text = decoder.decode(chunk, final=not chunk)
        if text:
            buffer.write(text.replace("\r\n", "\n"))
        if not chunk:
            return


async def kill_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:  # includes ProcessLookupError: already gone
        pass
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=5)
