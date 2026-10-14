"""Git access and checkpoint/rollback.

Checkpoints snapshot the *entire working tree* (tracked changes and untracked,
non-ignored files) without touching the user's branch, index, stash or HEAD:

1. a temporary index is seeded from HEAD and ``git add -A`` stages the worktree into it;
2. ``write-tree`` + ``commit-tree`` create a dangling commit;
3. the commit is pinned under ``refs/forge/checkpoints/<id>`` so ``git gc`` keeps it.

Restoring writes the checkpoint's files back through another temporary index and removes
files that did not exist at checkpoint time. A "pre-restore" checkpoint is taken first,
so a rollback is itself reversible.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from forge.core.errors import ToolError
from forge.core.ids import new_id
from forge.core.time import utcnow

CHECKPOINT_REF_PREFIX = "refs/forge/checkpoints/"
_IDENTITY = {
    "GIT_AUTHOR_NAME": "Forge",
    "GIT_AUTHOR_EMAIL": "forge@localhost",
    "GIT_COMMITTER_NAME": "Forge",
    "GIT_COMMITTER_EMAIL": "forge@localhost",
}


class GitError(ToolError):
    pass


@dataclass
class GitOutcome:
    code: int
    stdout: str
    stderr: str


@dataclass
class GitStatus:
    branch: str | None
    head: str | None
    upstream: str | None
    ahead: int
    behind: int
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    conflicted: list[str]

    @property
    def dirty(self) -> bool:
        return bool(self.staged or self.unstaged or self.untracked or self.conflicted)


@dataclass
class CheckpointInfo:
    id: str
    commit: str
    label: str
    created_at: datetime


class GitRepo:
    def __init__(self, root: Path, git_dir: Path, cwd: Path | None = None) -> None:
        self.root = root  # repository top level
        self.git_dir = git_dir
        self.cwd = cwd or root  # where path arguments are resolved (the workspace)
        self.checkpoints = CheckpointManager(self)

    @classmethod
    async def discover(cls, start: Path) -> GitRepo | None:
        if shutil.which("git") is None:
            return None
        outcome = await _git(start, "rev-parse", "--show-toplevel", "--absolute-git-dir")
        if outcome.code != 0:
            return None
        lines = outcome.stdout.strip().splitlines()
        if len(lines) < 2:
            return None
        root, git_dir = Path(lines[0]), Path(lines[1])
        return cls(root.resolve(), git_dir.resolve(), start.resolve())  # noqa: ASYNC240

    async def run(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        check: bool = True,
        at_root: bool = False,
    ) -> GitOutcome:
        outcome = await _git(self.root if at_root else self.cwd, *args, env=env)
        if check and outcome.code != 0:
            raise GitError(f"git {' '.join(args[:3])} failed: {outcome.stderr.strip()[:1000]}")
        return outcome

    async def head(self) -> str | None:
        outcome = await self.run("rev-parse", "--verify", "-q", "HEAD", check=False)
        return outcome.stdout.strip() or None

    async def status(self) -> GitStatus:
        out = (
            await self.run("status", "--porcelain=v1", "-b", "-z", "--untracked-files=all")
        ).stdout
        entries = out.split("\0")
        branch = upstream = None
        ahead = behind = 0
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        conflicted: list[str] = []
        skip_next = False
        for entry in entries:
            if skip_next:
                skip_next = False
                continue
            if not entry:
                continue
            if entry.startswith("## "):
                branch, upstream, ahead, behind = _parse_branch_line(entry[3:])
                continue
            code, path = entry[:2], entry[3:]
            if code[0] in "RC":
                skip_next = True  # the next NUL field is the rename source
            if code == "??":
                untracked.append(path)
            elif "U" in code or code in {"AA", "DD"}:
                conflicted.append(path)
            else:
                if code[0] not in " ?":
                    staged.append(path)
                if code[1] not in " ?":
                    unstaged.append(path)
        return GitStatus(
            branch=branch,
            head=await self.head(),
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            conflicted=conflicted,
        )

    async def diff(
        self,
        *,
        paths: list[str] | None = None,
        staged: bool = False,
        base: str | None = None,
        stat: bool = False,
        context: int = 3,
    ) -> str:
        args = ["diff", "--no-color", "--no-ext-diff", f"-U{context}"]
        if stat:
            args.append("--stat")
        if staged:
            args.append("--cached")
        if base:
            args.append(base)
        args.append("--")
        args.extend(paths or [])
        return (await self.run(*args)).stdout

    async def worktree_diff(
        self, base: str, *, paths: list[str] | None = None, stat: bool = False
    ) -> str:
        """Diff from ``base`` to the current worktree *including untracked files*."""
        tree = await self._snapshot_tree()
        args = ["diff", "--no-color", "--no-ext-diff"]
        if stat:
            args.append("--stat")
        args += [base, tree, "--", *(paths or [])]
        return (await self.run(*args)).stdout

    async def changed_files(self, base: str) -> list[str]:
        tree = await self._snapshot_tree()
        out = (await self.run("diff", "--name-only", "-z", base, tree)).stdout
        return [p for p in out.split("\0") if p]

    async def log(self, *, limit: int = 10, path: str | None = None) -> str:
        if await self.head() is None:
            return ""
        args = ["log", f"-n{limit}", "--date=short", "--pretty=format:%h %ad %an  %s"]
        if path:
            args += ["--", path]
        return (await self.run(*args)).stdout

    async def branches(self) -> list[str]:
        out = (await self.run("branch", "--format=%(refname:short)")).stdout
        return [b for b in out.splitlines() if b]

    async def _snapshot_tree(self) -> str:
        """Tree object of the current worktree, built in a throwaway index."""
        index = self.git_dir / f"forge-index-{os.urandom(6).hex()}"
        env = {"GIT_INDEX_FILE": str(index)}
        try:
            if await self.head():
                await self.run("read-tree", "HEAD", env=env)
            else:
                await self.run("read-tree", "--empty", env=env)
            await self.run("add", "-A", "--", ".", env=env, at_root=True)
            return (await self.run("write-tree", env=env)).stdout.strip()
        finally:
            index.unlink(missing_ok=True)


class CheckpointManager:
    def __init__(self, repo: GitRepo) -> None:
        self.repo = repo

    async def create(self, label: str = "") -> CheckpointInfo:
        tree = await self.repo._snapshot_tree()
        head = await self.repo.head()
        checkpoint_id = new_id("ckpt")
        created = utcnow()
        message = f"forge checkpoint {checkpoint_id}\n\n{label}\n\ncreated: {created.isoformat()}"
        args = ["commit-tree", tree, "-m", message]
        if head:
            args[2:2] = ["-p", head]
        commit = (await self.repo.run(*args, env=dict(_IDENTITY))).stdout.strip()
        await self.repo.run("update-ref", CHECKPOINT_REF_PREFIX + checkpoint_id, commit)
        return CheckpointInfo(id=checkpoint_id, commit=commit, label=label, created_at=created)

    async def list(self) -> list[CheckpointInfo]:
        out = (
            await self.repo.run(
                "for-each-ref",
                "--sort=creatordate",
                "--format=%(refname)%00%(objectname)%00%(creatordate:iso-strict)%00%(contents:body)%00%00",
                CHECKPOINT_REF_PREFIX,
            )
        ).stdout
        checkpoints: list[CheckpointInfo] = []
        for record in out.split("\0\0"):
            fields = record.strip("\n").split("\0")
            if len(fields) < 3 or not fields[0]:
                continue
            body = fields[3] if len(fields) > 3 else ""
            label = "\n".join(
                line for line in body.splitlines() if not line.startswith("created:")
            ).strip()
            checkpoints.append(
                CheckpointInfo(
                    id=fields[0].removeprefix(CHECKPOINT_REF_PREFIX),
                    commit=fields[1],
                    label=label,
                    created_at=datetime.fromisoformat(fields[2]),
                )
            )
        return checkpoints

    async def resolve(self, checkpoint_id: str) -> str:
        outcome = await self.repo.run(
            "rev-parse", "--verify", "-q", CHECKPOINT_REF_PREFIX + checkpoint_id, check=False
        )
        commit = outcome.stdout.strip()
        if not commit:
            raise GitError(f"unknown checkpoint {checkpoint_id!r}")
        return commit

    async def files(self, commit: str) -> set[str]:
        out = (await self.repo.run("ls-tree", "-r", "-z", "--name-only", commit)).stdout
        return {p for p in out.split("\0") if p}

    async def restore(self, checkpoint_id: str) -> RestoreResult:
        commit = await self.resolve(checkpoint_id)
        safety = await self.create(f"pre-restore of {checkpoint_id}")
        target_files = await self.files(commit)
        current_tree = await self.repo._snapshot_tree()
        current_files = {
            p for p in (await self.repo.run("ls-tree", "-r", "-z", "--name-only", current_tree))
            .stdout.split("\0") if p
        }  # fmt: skip
        changed = [
            p
            for p in (
                await self.repo.run("diff", "--name-only", "-z", current_tree, commit)
            ).stdout.split("\0")
            if p
        ]
        removed: list[str] = []
        for rel in sorted(current_files - target_files):
            path = self.repo.root / rel
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(rel)
                _prune_empty_dirs(path.parent, self.repo.root)
        index = self.repo.git_dir / f"forge-index-{os.urandom(6).hex()}"
        env = {"GIT_INDEX_FILE": str(index)}
        try:
            await self.repo.run("read-tree", commit, env=env, at_root=True)
            await self.repo.run("checkout-index", "-a", "-f", env=env, at_root=True)
        finally:
            index.unlink(missing_ok=True)
        return RestoreResult(
            checkpoint_id=checkpoint_id,
            safety_checkpoint_id=safety.id,
            changed_files=sorted(changed),
            removed_files=removed,
        )

    async def diff(self, checkpoint_id: str, *, stat: bool = False) -> str:
        return await self.repo.worktree_diff(await self.resolve(checkpoint_id), stat=stat)


@dataclass
class RestoreResult:
    checkpoint_id: str
    safety_checkpoint_id: str
    changed_files: list[str]
    removed_files: list[str]


def _prune_empty_dirs(directory: Path, root: Path) -> None:
    while directory != root and root in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent


def _parse_branch_line(line: str) -> tuple[str | None, str | None, int, int]:
    # "main...origin/main [ahead 1, behind 2]" | "No commits yet on main" | "HEAD (no branch)"
    ahead = behind = 0
    if line.startswith("No commits yet on "):
        return line.removeprefix("No commits yet on ").strip(), None, 0, 0
    if line.startswith("HEAD (no branch)"):
        return None, None, 0, 0
    info = ""
    if " [" in line:
        line, info = line.split(" [", 1)
    branch, _, upstream = line.partition("...")
    for part in info.rstrip("]").split(","):
        part = part.strip()
        if part.startswith("ahead "):
            ahead = int(part[6:])
        elif part.startswith("behind "):
            behind = int(part[7:])
    return branch or None, upstream or None, ahead, behind


async def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> GitOutcome:
    full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", **(env or {})}
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.quotepath=off",
            *args,
            cwd=str(cwd),
            env=full_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    stdout, stderr = await process.communicate()
    return GitOutcome(
        code=process.returncode if process.returncode is not None else -1,
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
    )
