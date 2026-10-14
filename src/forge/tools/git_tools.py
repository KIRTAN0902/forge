"""Git tools exposed to agents: status, diff, log, branch, checkpoint, rollback."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from forge.core.errors import ToolError
from forge.policies.types import Permission, RiskLevel
from forge.tools.base import Tool, ToolAssessment, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment
from forge.tools.git import GitRepo
from forge.tools.terminal import NoInput

READ = frozenset({Permission.READ})
VCS = frozenset({Permission.VCS})


async def _repo(env: ToolEnvironment) -> GitRepo:
    repo = await env.git()
    if repo is None:
        raise ToolError("the workspace is not inside a git repository")
    return repo


class TextOutput(ToolOutput):
    text: str

    def render(self) -> str:
        return self.text


class GitStatusOutput(ToolOutput):
    branch: str | None
    head: str | None
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    conflicted: list[str]

    def render(self) -> str:
        head = (self.head or "no commits")[:10]
        lines = [f"On branch {self.branch or '(detached)'} at {head}"]
        if self.upstream:
            lines.append(f"Tracking {self.upstream} (ahead {self.ahead}, behind {self.behind})")
        for title, items in (
            ("Conflicted", self.conflicted),
            ("Staged", self.staged),
            ("Modified (unstaged)", self.unstaged),
            ("Untracked", self.untracked),
        ):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  {item}" for item in items[:200])
                if len(items) > 200:
                    lines.append(f"  ... {len(items) - 200} more")
        if len(lines) == 1:
            lines.append("Working tree clean.")
        return "\n".join(lines)


class GitStatus(Tool[NoInput, GitStatusOutput]):
    name = "git_status"
    description = "Show the current branch and staged, modified, untracked and conflicted files."
    input_model = NoInput
    output_model = GitStatusOutput

    async def run(self, args: NoInput, env: ToolEnvironment) -> GitStatusOutput:
        status = await (await _repo(env)).status()
        return GitStatusOutput(
            branch=status.branch,
            head=status.head,
            upstream=status.upstream,
            ahead=status.ahead,
            behind=status.behind,
            staged=status.staged,
            unstaged=status.unstaged,
            untracked=status.untracked,
            conflicted=status.conflicted,
        )


class GitDiffInput(ToolInput):
    paths: list[str] = Field(default_factory=list, description="Limit to these paths.")
    staged: bool = Field(default=False, description="Show staged changes instead of unstaged.")
    base: str | None = Field(
        default=None, description="Compare against this commit/branch/checkpoint ref."
    )
    stat: bool = Field(default=False, description="Summary of changed files only.")


class GitDiff(Tool[GitDiffInput, TextOutput]):
    name = "git_diff"
    description = (
        "Show changes as a unified diff: working tree vs index (default), staged changes, or "
        "everything since `base`."
    )
    input_model = GitDiffInput
    output_model = TextOutput

    async def run(self, args: GitDiffInput, env: ToolEnvironment) -> TextOutput:
        repo = await _repo(env)
        for path in args.paths:
            env.workspace.resolve(path)
        if args.base and args.base.startswith(("-",)):
            raise ToolError("base must be a revision, not an option")
        text = await repo.diff(paths=args.paths, staged=args.staged, base=args.base, stat=args.stat)
        return TextOutput(text=text or "(no differences)")


class GitLogInput(ToolInput):
    limit: int = Field(default=10, ge=1, le=200)
    path: str | None = Field(default=None, description="Only commits touching this path.")


class GitLog(Tool[GitLogInput, TextOutput]):
    name = "git_log"
    description = "Show recent commits (hash, date, author, subject)."
    input_model = GitLogInput
    output_model = TextOutput

    async def run(self, args: GitLogInput, env: ToolEnvironment) -> TextOutput:
        if args.path:
            env.workspace.resolve(args.path)
        text = await (await _repo(env)).log(limit=args.limit, path=args.path)
        return TextOutput(text=text or "(no commits yet)")


class GitBranchInput(ToolInput):
    action: Literal["list", "create", "switch"] = "list"
    name: str | None = Field(default=None, description="Branch name for create/switch.")


class GitBranch(Tool[GitBranchInput, TextOutput]):
    name = "git_branch"
    description = "List branches, create a branch (without switching), or switch branches."
    input_model = GitBranchInput
    output_model = TextOutput

    def assess(self, args: GitBranchInput, env: ToolEnvironment) -> ToolAssessment:
        if args.action == "list":
            return ToolAssessment(RiskLevel.LOW, READ)
        return ToolAssessment(RiskLevel.MEDIUM, VCS, description=f"git branch {args.action}")

    async def run(self, args: GitBranchInput, env: ToolEnvironment) -> TextOutput:
        repo = await _repo(env)
        if args.action == "list":
            status = await repo.status()
            names = await repo.branches()
            lines = [("* " if n == status.branch else "  ") + n for n in names]
            return TextOutput(text="\n".join(lines) or "(no branches yet)")
        if not args.name or args.name.startswith("-"):
            raise ToolError("a valid branch `name` is required")
        if args.action == "create":
            await repo.run("branch", args.name)
            return TextOutput(text=f"Created branch {args.name}.")
        await repo.run("switch", args.name)
        return TextOutput(text=f"Switched to branch {args.name}.")


class CheckpointInput(ToolInput):
    label: str = Field(default="", max_length=200, description="What this checkpoint captures.")


class CheckpointOutput(ToolOutput):
    id: str
    commit: str
    label: str

    def render(self) -> str:
        return f"Checkpoint {self.id} created ({self.commit[:10]}). {self.label}".rstrip()


class GitCheckpoint(Tool[CheckpointInput, CheckpointOutput]):
    name = "git_checkpoint"
    description = (
        "Snapshot the whole working tree (including untracked files) so it can be restored "
        "with git_rollback. Does not commit, stage, or change branches."
    )
    input_model = CheckpointInput
    output_model = CheckpointOutput
    risk = RiskLevel.LOW
    permissions = VCS

    async def run(self, args: CheckpointInput, env: ToolEnvironment) -> CheckpointOutput:
        info = await (await _repo(env)).checkpoints.create(args.label)
        return CheckpointOutput(id=info.id, commit=info.commit, label=info.label)


class CheckpointListOutput(ToolOutput):
    checkpoints: list[CheckpointOutput]

    def render(self) -> str:
        if not self.checkpoints:
            return "No checkpoints."
        return "\n".join(f"{c.id}  {c.commit[:10]}  {c.label}" for c in self.checkpoints)


class GitListCheckpoints(Tool[NoInput, CheckpointListOutput]):
    name = "git_list_checkpoints"
    description = "List Forge checkpoints, oldest first."
    input_model = NoInput
    output_model = CheckpointListOutput

    async def run(self, args: NoInput, env: ToolEnvironment) -> CheckpointListOutput:
        checkpoints = await (await _repo(env)).checkpoints.list()
        return CheckpointListOutput(
            checkpoints=[
                CheckpointOutput(id=c.id, commit=c.commit, label=c.label) for c in checkpoints
            ]
        )


class RollbackInput(ToolInput):
    checkpoint_id: str = Field(description="Checkpoint id from git_checkpoint.")


class RollbackOutput(ToolOutput):
    checkpoint_id: str
    safety_checkpoint_id: str
    changed_files: list[str]
    removed_files: list[str]

    def render(self) -> str:
        return (
            f"Restored the working tree to {self.checkpoint_id}: "
            f"{len(self.changed_files)} file(s) restored, {len(self.removed_files)} removed. "
            f"The previous state was saved as {self.safety_checkpoint_id}."
        )

    def changed_paths(self) -> list[str]:
        return sorted(set(self.changed_files) | set(self.removed_files))


class GitRollback(Tool[RollbackInput, RollbackOutput]):
    name = "git_rollback"
    description = (
        "Restore the working tree to a checkpoint. The current state is checkpointed first, "
        "so a rollback can itself be undone."
    )
    input_model = RollbackInput
    output_model = RollbackOutput
    risk = RiskLevel.MEDIUM
    permissions = frozenset({Permission.VCS, Permission.WRITE})

    async def run(self, args: RollbackInput, env: ToolEnvironment) -> RollbackOutput:
        result = await (await _repo(env)).checkpoints.restore(args.checkpoint_id)
        return RollbackOutput(
            checkpoint_id=result.checkpoint_id,
            safety_checkpoint_id=result.safety_checkpoint_id,
            changed_files=result.changed_files,
            removed_files=result.removed_files,
        )
