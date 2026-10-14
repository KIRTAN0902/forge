from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from forge.models import ToolCall
from forge.policies import StaticApprover
from forge.tools import GitRepo
from forge.tools.shell import BoundedBuffer, SecretRedactor, ShellRunner, scrubbed_environment
from forge.tools.testing import detect_test_command, parse_test_output

from ...conftest import PYTHON, RuntimeFactory, git

_n = 0


def call(name: str, /, **arguments: Any) -> ToolCall:
    global _n
    _n += 1
    return ToolCall(id=f"t{_n}", name=name, arguments=arguments)


class TestShellPrimitives:
    def test_bounded_buffer_keeps_head_and_tail(self) -> None:
        buffer = BoundedBuffer(20)
        for chunk in ["0123456789", "abcdefghij", "KLMNOPQRST", "uvwxyz"]:
            buffer.write(chunk)
        text = buffer.text()
        assert text.startswith("0123456789")
        assert text.endswith("QRSTuvwxyz")
        assert buffer.truncated
        assert "omitted" in text

    def test_env_scrubbing_and_redaction(self) -> None:
        env, withheld = scrubbed_environment(
            ["DATABASE_URL"],
            base={
                "PATH": "/bin",
                "OPENAI_API_KEY": "sk-abcdefgh123",
                "GITHUB_TOKEN": "ghp_x",
                "DATABASE_URL": "postgres://u:p@h/db",
                "HOME": "/home/u",
            },
        )
        assert "OPENAI_API_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert env["DATABASE_URL"].startswith("postgres")  # explicitly passed through
        assert env["PATH"] == "/bin"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        redact = SecretRedactor(withheld)
        assert redact("key=sk-abcdefgh123") == "key=[REDACTED:OPENAI_API_KEY]"
        assert redact("ghp_x") == "ghp_x"  # too short to redact safely


class TestRunCommand:
    async def test_output_exit_code_and_stderr(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        script = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
        result = await runtime.execute(call("run_command", command=f'"{PYTHON}" -c "{script}"'))
        assert result.ok  # the tool ran; the command failing is data, not a tool error
        assert result.data is not None
        assert result.data["exit_code"] == 3
        assert "out" in result.content
        assert "[stderr]\nerr" in result.content

    async def test_timeout_kills_process_tree(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        started = time.perf_counter()
        result = await runtime.execute(
            call(
                "run_command",
                command=f'"{PYTHON}" -c "import time; time.sleep(30)"',
                timeout_s=1,
            )
        )
        assert time.perf_counter() - started < 15
        assert result.data is not None
        assert result.data["timed_out"] is True
        assert "timed out" in result.content

    async def test_secret_env_hidden_from_commands(
        self, sample_repo: Path, make_runtime: RuntimeFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_TEST_SECRET_TOKEN", "tok-1234567890")
        runtime = make_runtime(sample_repo)
        script = "import os; print(os.environ.get('FORGE_TEST_SECRET_TOKEN', 'absent'))"
        result = await runtime.execute(call("run_command", command=f'"{PYTHON}" -c "{script}"'))
        assert "absent" in result.content
        assert "tok-1234567890" not in result.content

    async def test_forbidden_and_network_commands_blocked(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo, approver=StaticApprover(True))
        forbidden = await runtime.execute(call("run_command", command="rm -rf /"))
        assert not forbidden.ok
        assert "forbidden" in forbidden.content
        network = await runtime.execute(call("run_command", command="pip install leftpad"))
        assert not network.ok
        assert "network" in network.content

    async def test_high_risk_command_needs_approval(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        approver = StaticApprover(False)
        runtime = make_runtime(sample_repo, approver=approver)
        result = await runtime.execute(call("run_command", command="git reset --hard HEAD"))
        assert not result.ok
        assert approver.requests[0].command == "git reset --hard HEAD"


class TestRunTests:
    def test_detection_and_parsing(self, sample_repo: Path) -> None:
        detected = detect_test_command(sample_repo)
        assert detected is not None
        assert detected.framework == "pytest"
        counts = parse_test_output("pytest", "==== 1 failed, 3 passed, 2 skipped in 0.12s ====")
        assert (counts.passed, counts.failed, counts.skipped) == (3, 1, 2)
        jest = parse_test_output("jest", "Tests:       1 failed, 4 passed, 5 total")
        assert (jest.passed, jest.failed) == (4, 1)
        cargo = parse_test_output("cargo", "test result: ok. 7 passed; 0 failed; 0 ignored")
        assert (cargo.passed, cargo.failed) == (7, 0)

    async def test_runs_and_reports_failures(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(
            call("run_tests", command=f'"{PYTHON}" -m pytest -q -p no:cacheprovider')
        )
        assert result.ok, result.content
        assert result.data is not None
        assert result.data["failed"] == 1
        assert result.data["passed"] == 1
        assert "FAILED" in result.content
        assert "test_add" in result.content

    async def test_paths_only_for_pytest(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        result = await runtime.execute(
            call("run_tests", command="npm test", paths=["tests/test_ops.py"])
        )
        assert not result.ok
        assert "only supported for pytest" in result.content


class TestProcesses:
    async def test_start_read_stop(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(sample_repo)
        script = (
            "import time\\nfor i in range(100):\\n"
            "    print('tick', i, flush=True); time.sleep(0.05)"
        )
        started = await runtime.execute(
            call("start_process", command=f'"{PYTHON}" -c "exec(\'{script}\')"')
        )
        assert started.ok, started.content
        proc_id = started.data["id"] if started.data else ""
        await asyncio.sleep(1.0)
        output = await runtime.execute(call("process_output", id=proc_id))
        assert "tick" in output.content
        listing = await runtime.execute(call("list_processes"))
        assert proc_id in listing.content
        stopped = await runtime.execute(call("stop_process", id=proc_id))
        assert stopped.data is not None
        assert stopped.data["running"] is False
        unknown = await runtime.execute(call("process_output", id="proc_999"))
        assert not unknown.ok


class TestGit:
    async def test_status_diff_log(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        (sample_repo / "src/calc/ops.py").write_text(
            (sample_repo / "src/calc/ops.py").read_text().replace("a - b", "a + b")
        )
        (sample_repo / "notes.txt").write_text("new\n")
        runtime = make_runtime(sample_repo)
        status = await runtime.execute(call("git_status"))
        assert "On branch main" in status.content
        assert "src/calc/ops.py" in status.content
        assert "notes.txt" in status.content
        diff = await runtime.execute(call("git_diff"))
        assert "+    return a + b" in diff.content
        log = await runtime.execute(call("git_log", limit=5))
        assert "initial" in log.content
        bad = await runtime.execute(call("git_diff", base="--output=/tmp/x"))
        assert not bad.ok

    async def test_checkpoint_and_rollback_full_worktree(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        (sample_repo / "draft.txt").write_text("untracked before checkpoint\n")
        index_before = git(sample_repo, "diff", "--cached", "--name-only")
        head_before = git(sample_repo, "rev-parse", "HEAD")

        checkpoint = await runtime.execute(call("git_checkpoint", label="before change"))
        assert checkpoint.ok, checkpoint.content
        ckpt_id = checkpoint.data["id"] if checkpoint.data else ""

        # Make a mess: modify, delete tracked, delete untracked, add new files.
        ops = sample_repo / "src/calc/ops.py"
        original_ops = ops.read_text()
        ops.write_text("broken\n")
        (sample_repo / "README.md").unlink()
        (sample_repo / "draft.txt").unlink()
        (sample_repo / "new_dir").mkdir()
        (sample_repo / "new_dir" / "junk.py").write_text("junk\n")

        rollback = await runtime.execute(call("git_rollback", checkpoint_id=ckpt_id))
        assert rollback.ok, rollback.content
        assert ops.read_text() == original_ops
        assert (sample_repo / "README.md").exists()
        assert (sample_repo / "draft.txt").read_text() == "untracked before checkpoint\n"
        assert not (sample_repo / "new_dir").exists()
        # The user's git state is untouched.
        assert git(sample_repo, "diff", "--cached", "--name-only") == index_before
        assert git(sample_repo, "rev-parse", "HEAD") == head_before
        assert git(sample_repo, "stash", "list") == ""
        # Secrets that git ignores... .env is untracked but not ignored here, so it is
        # included in checkpoints; ignored files would be left alone.

        # The rollback itself can be undone.
        safety_id = rollback.data["safety_checkpoint_id"] if rollback.data else ""
        undo = await runtime.execute(call("git_rollback", checkpoint_id=safety_id))
        assert undo.ok
        assert ops.read_text() == "broken\n"
        assert (sample_repo / "new_dir" / "junk.py").exists()

        listing = await runtime.execute(call("git_list_checkpoints"))
        assert ckpt_id in listing.content
        assert "before change" in listing.content

    async def test_ignored_files_survive_rollback(self, sample_repo: Path) -> None:
        repo = await GitRepo.discover(sample_repo)
        assert repo is not None
        info = await repo.checkpoints.create("x")
        cache = sample_repo / "src" / "calc" / "__pycache__"
        cache.mkdir()
        (cache / "ops.pyc").write_bytes(b"\0cache")
        await repo.checkpoints.restore(info.id)
        assert (cache / "ops.pyc").exists()

    async def test_repo_without_commits(self, tmp_path: Path) -> None:
        git(tmp_path, "init", "-q")
        (tmp_path / "a.txt").write_text("a\n")
        repo = await GitRepo.discover(tmp_path)
        assert repo is not None
        info = await repo.checkpoints.create("first")
        (tmp_path / "a.txt").write_text("changed\n")
        (tmp_path / "b.txt").write_text("b\n")
        diff = await repo.checkpoints.diff(info.id)
        assert "+changed" in diff
        assert "b.txt" in diff
        assert set(await repo.changed_files(info.commit)) == {"a.txt", "b.txt"}
        await repo.checkpoints.restore(info.id)
        assert (tmp_path / "a.txt").read_text() == "a\n"
        assert not (tmp_path / "b.txt").exists()

    async def test_branch_tool(self, sample_repo: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(sample_repo)
        assert (await runtime.execute(call("git_branch", action="create", name="feature"))).ok
        listing = await runtime.execute(call("git_branch"))
        assert "* main" in listing.content
        assert "feature" in listing.content
        assert (await runtime.execute(call("git_branch", action="switch", name="feature"))).ok
        assert "feature" in git(sample_repo, "branch", "--show-current")
        injected = await runtime.execute(call("git_branch", action="create", name="--force"))
        assert not injected.ok

    async def test_not_a_repo(self, tmp_path: Path, make_runtime: RuntimeFactory) -> None:
        runtime = make_runtime(tmp_path)
        result = await runtime.execute(call("git_status"))
        assert not result.ok
        assert "not inside a git repository" in result.content


def test_shell_runner_reports_shell() -> None:
    runner = ShellRunner()
    assert runner.spec.display in {"bash", "POSIX sh", "PowerShell", "cmd.exe"}
