"""Terminal tools: commands, tests and background processes."""

from __future__ import annotations

import platform

from pydantic import Field

from forge.core.errors import ToolError
from forge.policies.commands import assess_command
from forge.policies.types import Permission, RiskLevel
from forge.tools.base import Tool, ToolAssessment, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment
from forge.tools.testing import detect_test_command, parse_test_output

EXECUTE = frozenset({Permission.EXECUTE})


def _command_assessment(command: str, cwd: str, env: ToolEnvironment) -> ToolAssessment:
    assessment = assess_command(command)
    return ToolAssessment(
        risk=assessment.risk,
        permissions=frozenset(assessment.permissions),
        paths=[env.workspace.relative(cwd)],
        command=command,
        description="; ".join(assessment.reasons),
    )


# -- run_command --------------------------------------------------------------------------


class RunCommandInput(ToolInput):
    command: str = Field(min_length=1, description="Shell command to run (non-interactive).")
    cwd: str = Field(default=".", description="Working directory relative to the workspace.")
    timeout_s: float | None = Field(default=None, gt=0, le=3600)


class RunCommandOutput(ToolOutput):
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False

    def render(self) -> str:
        status = "timed out" if self.timed_out else f"exit {self.exit_code}"
        parts = [f"$ {self.command}", f"[{status}, {self.duration_s:.1f}s]"]
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(no output)")
        return "\n".join(parts)


class RunCommand(Tool[RunCommandInput, RunCommandOutput]):
    name = "run_command"
    input_model = RunCommandInput
    output_model = RunCommandOutput
    risk = RiskLevel.MEDIUM
    permissions = EXECUTE
    description = "Run a shell command in the workspace and return its output."

    def describe(self, env: ToolEnvironment | None) -> str:
        if env is None:
            return self.description
        return (
            f"Run a non-interactive command with {env.shell.spec.display} on "
            f"{platform.system()} in the workspace. stdin is closed; avoid commands that "
            "prompt or never exit (use start_process for servers). Output is truncated if "
            "very long. Secret environment variables are not available."
        )

    def assess(self, args: RunCommandInput, env: ToolEnvironment) -> ToolAssessment:
        return _command_assessment(args.command, args.cwd, env)

    def timeout_for(self, args: RunCommandInput, env: ToolEnvironment) -> float:
        return (args.timeout_s or env.shell.default_timeout_s) + 15

    async def run(self, args: RunCommandInput, env: ToolEnvironment) -> RunCommandOutput:
        cwd = env.workspace.resolve(args.cwd)
        if not cwd.is_dir():
            raise ToolError(f"cwd {args.cwd!r} is not a directory")
        outcome = await env.shell.run(args.command, cwd=cwd, timeout_s=args.timeout_s)
        return RunCommandOutput(
            command=args.command,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            duration_s=round(outcome.duration_s, 3),
            timed_out=outcome.timed_out,
            truncated=outcome.truncated,
        )


# -- run_tests ----------------------------------------------------------------------------


class RunTestsInput(ToolInput):
    command: str | None = Field(
        default=None, description="Test command; omit to use the project's detected command."
    )
    paths: list[str] = Field(
        default_factory=list, description="Specific test files/dirs (pytest only)."
    )
    timeout_s: float | None = Field(default=None, gt=0, le=7200)


class RunTestsOutput(ToolOutput):
    command: str
    framework: str
    exit_code: int | None
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    duration_s: float
    timed_out: bool = False
    output: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self) -> str:
        counts = ", ".join(
            f"{value} {name}"
            for name, value in (
                ("passed", self.passed),
                ("failed", self.failed),
                ("errors", self.errors),
                ("skipped", self.skipped),
            )
            if value
        )
        verdict = "TIMED OUT" if self.timed_out else ("PASSED" if self.succeeded else "FAILED")
        header = f"$ {self.command}\n[{verdict}: exit {self.exit_code}, {self.duration_s:.1f}s"
        header += f"; {counts}]" if counts else "]"
        return f"{header}\n{self.output.rstrip()}"


class RunTests(Tool[RunTestsInput, RunTestsOutput]):
    name = "run_tests"
    description = (
        "Run the project's test suite (auto-detected: pytest, npm/pnpm/yarn, go, cargo, "
        "maven, gradle) or a given test command, and report pass/fail counts."
    )
    input_model = RunTestsInput
    output_model = RunTestsOutput
    risk = RiskLevel.MEDIUM
    permissions = EXECUTE

    def _resolve(self, args: RunTestsInput, env: ToolEnvironment) -> tuple[str, str]:
        if args.command:
            framework = "pytest" if "pytest" in args.command else "custom"
            command = args.command
        elif env.test_command:
            command = env.test_command
            framework = "pytest" if "pytest" in command else "custom"
        else:
            detected = detect_test_command(env.workspace.root)
            if detected is None:
                raise ToolError("could not detect how to run tests here; pass `command` explicitly")
            command, framework = detected.command, detected.framework
        if args.paths:
            if framework != "pytest":
                raise ToolError("`paths` is only supported for pytest; include them in `command`")
            for path in args.paths:
                env.workspace.resolve(path)  # reject paths outside the workspace
            command = f"{command} " + " ".join(_quote(p) for p in args.paths)
        return command, framework

    def assess(self, args: RunTestsInput, env: ToolEnvironment) -> ToolAssessment:
        command, _ = self._resolve(args, env)
        return _command_assessment(command, ".", env)

    def timeout_for(self, args: RunTestsInput, env: ToolEnvironment) -> float:
        return (args.timeout_s or env.shell.default_timeout_s) + 15

    async def run(self, args: RunTestsInput, env: ToolEnvironment) -> RunTestsOutput:
        command, framework = self._resolve(args, env)
        outcome = await env.shell.run(command, cwd=env.workspace.root, timeout_s=args.timeout_s)
        combined = outcome.stdout + ("\n" + outcome.stderr if outcome.stderr.strip() else "")
        counts = parse_test_output(framework, combined)
        return RunTestsOutput(
            command=command,
            framework=framework,
            exit_code=outcome.exit_code,
            passed=counts.passed,
            failed=counts.failed,
            errors=counts.errors,
            skipped=counts.skipped,
            duration_s=round(outcome.duration_s, 3),
            timed_out=outcome.timed_out,
            output=_tail(combined, 12_000),
        )


def _quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"... [{len(text) - limit} earlier characters omitted]\n{text[-limit:]}"


# -- background processes -----------------------------------------------------------------


class StartProcessInput(ToolInput):
    command: str = Field(min_length=1, description="Long-running command, e.g. a dev server.")
    cwd: str = Field(default=".")


class ProcessInfoOutput(ToolOutput):
    id: str
    command: str
    running: bool
    exit_code: int | None = None
    output: str = ""

    def render(self) -> str:
        state = "running" if self.running else f"exited ({self.exit_code})"
        text = f"{self.id}: {self.command} [{state}]"
        return f"{text}\n{self.output.rstrip()}" if self.output.strip() else text


class StartProcess(Tool[StartProcessInput, ProcessInfoOutput]):
    name = "start_process"
    description = (
        "Start a long-running background process (server, watcher). Returns an id for "
        "process_output and stop_process. All processes are stopped when the session ends."
    )
    input_model = StartProcessInput
    output_model = ProcessInfoOutput
    risk = RiskLevel.MEDIUM
    permissions = EXECUTE

    def assess(self, args: StartProcessInput, env: ToolEnvironment) -> ToolAssessment:
        return _command_assessment(args.command, args.cwd, env)

    async def run(self, args: StartProcessInput, env: ToolEnvironment) -> ProcessInfoOutput:
        managed = await env.processes.start(
            env.shell, args.command, env.workspace.resolve(args.cwd)
        )
        return ProcessInfoOutput(id=managed.id, command=managed.command, running=managed.running)


class ProcessIdInput(ToolInput):
    id: str = Field(description="Process id returned by start_process.")


class ProcessOutput(Tool[ProcessIdInput, ProcessInfoOutput]):
    name = "process_output"
    description = "Read new output from a background process since the last read."
    input_model = ProcessIdInput
    output_model = ProcessInfoOutput
    permissions = frozenset({Permission.READ})

    async def run(self, args: ProcessIdInput, env: ToolEnvironment) -> ProcessInfoOutput:
        managed = env.processes.get(args.id)
        text, skipped = env.processes.read_new(args.id, 20_000)
        if skipped:
            text = f"... [{skipped} characters skipped]\n{text}"
        return ProcessInfoOutput(
            id=managed.id,
            command=managed.command,
            running=managed.running,
            exit_code=managed.exit_code,
            output=text or "(no new output)",
        )


class StopProcess(Tool[ProcessIdInput, ProcessInfoOutput]):
    name = "stop_process"
    description = "Stop a background process (and its children)."
    input_model = ProcessIdInput
    output_model = ProcessInfoOutput
    risk = RiskLevel.LOW
    permissions = EXECUTE

    async def run(self, args: ProcessIdInput, env: ToolEnvironment) -> ProcessInfoOutput:
        managed = await env.processes.stop(args.id)
        return ProcessInfoOutput(
            id=managed.id, command=managed.command, running=managed.running,
            exit_code=managed.exit_code,
        )  # fmt: skip


class NoInput(ToolInput):
    pass


class ProcessListOutput(ToolOutput):
    processes: list[ProcessInfoOutput]

    def render(self) -> str:
        if not self.processes:
            return "No background processes."
        return "\n".join(p.render() for p in self.processes)


class ListProcesses(Tool[NoInput, ProcessListOutput]):
    name = "list_processes"
    description = "List background processes started in this session."
    input_model = NoInput
    output_model = ProcessListOutput

    async def run(self, args: NoInput, env: ToolEnvironment) -> ProcessListOutput:
        return ProcessListOutput(
            processes=[
                ProcessInfoOutput(
                    id=p.id, command=p.command, running=p.running, exit_code=p.exit_code
                )
                for p in env.processes.list()
            ]
        )
