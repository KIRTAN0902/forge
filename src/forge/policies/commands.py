"""Deterministic shell-command risk classification.

The model proposes a command; this module, not the model, decides how dangerous it is.
A command is split into segments (``&&``, ``||``, ``;``, ``|``, newlines) and each
segment is classified; the command's risk is the highest segment risk and its
permissions are the union. Unknown programs default to MEDIUM: they run arbitrary code
in the workspace, which checkpoints can recover.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field

from forge.policies.types import Permission, RiskLevel

DEFAULT_FORBIDDEN_COMMANDS: tuple[str, ...] = (
    # Recursive deletion of root, home, or everything.
    r"\brm\s+(?:-\S+\s+)*-\S*[rR]\S*\s+(?:-\S+\s+)*(?:/|/\*|~|~/|\$HOME|\*)(?:\s|$)",
    r"(?i)\b(?:rd|rmdir)\s+/s\s+(?:/q\s+)?[a-z]:\\?(?:\s|$)",
    r"(?i)\bformat\s+[a-z]:",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdd\s+.*\bof=/dev/(?!null)",
    r">\s*/dev/sd[a-z]",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # fork bomb
    r"\b(?:shutdown|reboot|halt|poweroff)\b",
    r"(?i)\bdiskpart\b",
    # Piping remote content straight into a shell.
    r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
    r"(?i)\b(?:iwr|invoke-webrequest)\b[^|]*\|\s*(?:iex|invoke-expression)\b",
)


@dataclass
class CommandAssessment:
    risk: RiskLevel = RiskLevel.LOW
    permissions: set[Permission] = field(default_factory=lambda: {Permission.EXECUTE})
    reasons: list[str] = field(default_factory=list)

    def raise_to(self, risk: RiskLevel, reason: str, *perms: Permission) -> None:
        if risk > self.risk:
            self.risk = risk
        self.permissions.update(perms)
        if reason not in self.reasons:
            self.reasons.append(reason)


def forbidden_match(command: str, patterns: Iterable[str]) -> str | None:
    """Return the first forbidden pattern ``command`` matches, if any."""
    for pattern in patterns:
        if re.search(pattern, command):
            return pattern
    return None


_SEGMENT_SPLIT = re.compile(r"\|\||&&|;|\||\n|(?<![<>&\d])&(?![>&])")


def split_segments(command: str) -> list[str]:
    """Split on shell operators that are outside quotes."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(command):
        char = command[i]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and i + 1 < len(command):
                current.append(command[i + 1])
                i += 1
        elif char in "'\"":
            quote = char
            current.append(char)
        else:
            match = _SEGMENT_SPLIT.match(command, i)
            if match:
                segments.append("".join(current))
                current = []
                i = match.end()
                continue
            current.append(char)
        i += 1
    segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


_READ_ONLY = {
    "ls", "dir", "cat", "type", "head", "tail", "wc", "pwd", "echo", "printf", "grep",
    "egrep", "fgrep", "rg", "ag", "tree", "which", "where", "whereis", "stat", "file",
    "diff", "cmp", "sort", "uniq", "cut", "tr", "less", "more", "basename", "dirname",
    "realpath", "readlink", "du", "df", "date", "whoami", "hostname", "uname", "env",
    "printenv", "true", "false", "test", "jq", "get-content", "get-childitem",
    "select-string", "get-location", "nl", "column", "md5sum", "sha256sum", "cd",
}  # fmt: skip
_TEST_RUNNERS = {
    "pytest", "py.test", "tox", "nox", "jest", "vitest", "mocha", "ava", "karma",
    "phpunit", "rspec", "ctest",
}  # fmt: skip
_NETWORK_TOOLS = {
    "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "telnet", "ftp",
    "invoke-webrequest", "iwr", "invoke-restmethod", "irm",
}  # fmt: skip
_SYSTEM_PACKAGE_MANAGERS = {
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "brew",
    "choco",
    "winget",
    "pacman",
    "snap",
}
_DEPLOY_TOOLS = {"kubectl", "helm", "terraform", "pulumi", "flyctl", "fly", "vercel", "netlify",
                 "serverless", "sls", "aws", "gcloud", "az", "heroku", "eb", "ansible-playbook"}  # fmt: skip
_MIGRATION_HINTS = (
    ("alembic", ("upgrade", "downgrade", "stamp")),
    ("flyway", ("migrate", "clean", "repair")),
    ("liquibase", ("update", "rollback", "drop-all")),
)
_GIT_READ = {"status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "ls-tree",
             "grep", "describe", "shortlog", "reflog", "cat-file", "for-each-ref", "config"}  # fmt: skip
_GIT_NETWORK = {"fetch", "pull", "clone", "ls-remote", "submodule"}


def assess_command(command: str) -> CommandAssessment:
    assessment = CommandAssessment()
    segments = split_segments(command)
    if not segments:
        assessment.raise_to(RiskLevel.LOW, "empty command")
        return assessment
    for segment in segments:
        _assess_segment(segment, assessment, depth=0)
    return assessment


def _assess_segment(segment: str, out: CommandAssessment, *, depth: int) -> None:
    tokens = _tokens(segment)
    # Drop leading environment assignments (FOO=bar cmd).
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return
    redirects = [t for t in tokens if t in {">", ">>", "1>", "2>", "&>"} or t.startswith(">")]
    harmless = {"2>/dev/null", ">/dev/null", "2>&1", ">&2", "1>&2"}
    if redirects and not all(t in harmless for t in redirects):
        out.raise_to(RiskLevel.MEDIUM, "writes files via redirection", Permission.WRITE)

    program = re.sub(
        r"\.(exe|cmd|bat|ps1)$", "", tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    )
    args = tokens[1:]
    lowered = [a.lower() for a in args]

    if program in {"sudo", "doas", "runas"}:
        out.raise_to(RiskLevel.HIGH, f"elevated privileges via {program}")
        if args:
            _assess_segment(shlex.join(args), out, depth=depth)
        return
    if program in {"bash", "sh", "zsh", "dash", "pwsh", "powershell", "cmd"} and depth < 3:
        inner_flags = {"-c", "-command", "/c", "/k"}
        for index, arg in enumerate(lowered):
            if arg in inner_flags and index + 1 < len(args):
                for inner in split_segments(" ".join(args[index + 1 :])):
                    _assess_segment(inner, out, depth=depth + 1)
                return
        out.raise_to(RiskLevel.MEDIUM, f"runs a {program} script")
        return

    if program == "git":
        _assess_git(lowered, out)
        return
    if program in _READ_ONLY:
        out.raise_to(RiskLevel.LOW, f"read-only command {program}", Permission.READ)
        return
    if program == "find":
        if any(a in {"-delete", "-exec", "-execdir", "-ok"} for a in lowered):
            out.raise_to(RiskLevel.HIGH, "find with -delete/-exec", Permission.WRITE)
        else:
            out.raise_to(RiskLevel.LOW, "read-only find", Permission.READ)
        return
    if program in {"sed", "perl"} and any(a.startswith("-i") for a in lowered):
        out.raise_to(RiskLevel.MEDIUM, f"in-place edit with {program}", Permission.WRITE)
        return
    if program in {"rm", "del", "erase", "rmdir", "rd", "remove-item", "unlink", "shred"}:
        recursive = any(
            a.startswith("-") and ("r" in a.lstrip("-").lower() or a == "--recursive")
            for a in lowered
        ) or any(a in {"/s", "-recurse"} for a in lowered)
        wildcard = any("*" in a for a in args)
        if recursive or wildcard or program in {"rmdir", "rd", "shred"}:
            out.raise_to(RiskLevel.HIGH, "recursive or wildcard deletion", Permission.WRITE)
        else:
            out.raise_to(RiskLevel.MEDIUM, "deletes files", Permission.WRITE)
        return
    if program in {"chmod", "chown", "chgrp", "icacls", "takeown", "attrib"}:
        recursive = any(a in {"-r", "--recursive", "/t", "/s"} for a in lowered)
        level = RiskLevel.HIGH if recursive else RiskLevel.MEDIUM
        out.raise_to(level, "permission change", Permission.WRITE)
        return
    if program in {"mv", "cp", "move", "copy", "xcopy", "robocopy", "mkdir", "md", "touch",
                   "ln", "move-item", "copy-item", "new-item", "set-content", "out-file", "tee"}:  # fmt: skip
        out.raise_to(RiskLevel.MEDIUM, f"modifies files ({program})", Permission.WRITE)
        return
    if program in _NETWORK_TOOLS:
        out.raise_to(RiskLevel.MEDIUM, f"network access via {program}", Permission.NETWORK)
        return
    if program in _SYSTEM_PACKAGE_MANAGERS:
        out.raise_to(RiskLevel.HIGH, "system package installation", Permission.NETWORK)
        return
    if program in _DEPLOY_TOOLS:
        read_only = bool(lowered) and lowered[0] in {"get", "describe", "plan", "version",
                                                     "show", "list", "status", "logs", "help"}  # fmt: skip
        if read_only:
            out.raise_to(RiskLevel.MEDIUM, f"{program} read-only query", Permission.NETWORK)
        else:
            out.raise_to(
                RiskLevel.CRITICAL, f"deployment/infrastructure change via {program}",
                Permission.NETWORK,
            )  # fmt: skip
        return
    if program == "docker" or program == "podman":
        sub = lowered[0] if lowered else ""
        if sub in {"push", "login"}:
            out.raise_to(RiskLevel.CRITICAL, f"{program} {sub} publishes or authenticates",
                         Permission.NETWORK)  # fmt: skip
        elif sub in {"ps", "images", "logs", "inspect", "version", "info"}:
            out.raise_to(RiskLevel.LOW, f"{program} read-only query")
        else:
            out.raise_to(RiskLevel.HIGH, f"container operation ({program} {sub})")
        return
    for tool, verbs in _MIGRATION_HINTS:
        if program == tool and any(v in lowered for v in verbs):
            out.raise_to(RiskLevel.HIGH, "database migration", Permission.WRITE)
            return
    joined = " ".join(lowered)
    if re.search(r"\bmanage\.py\s+(migrate|flush|sqlflush)\b", joined) or re.search(
        r"\b(prisma\s+(migrate|db\s+push)|db:migrate|db:drop|db:reset|knex\s+migrate|"
        r"sequelize\s+db:migrate|typeorm\s+migration:run)\b",
        f"{program} {joined}",
    ):
        out.raise_to(RiskLevel.HIGH, "database migration", Permission.WRITE)
        return
    if _is_package_install(program, lowered):
        out.raise_to(
            RiskLevel.MEDIUM, "installs dependencies", Permission.NETWORK, Permission.WRITE
        )
        return
    if program in {"npm", "pnpm", "yarn"} and lowered[:1] == ["publish"]:
        out.raise_to(RiskLevel.CRITICAL, "publishes a package", Permission.NETWORK)
        return
    if program in {"twine", "gem"} and lowered[:1] in (["upload"], ["push"]):
        out.raise_to(RiskLevel.CRITICAL, "publishes a package", Permission.NETWORK)
        return
    if program in {"cargo", "uv", "poetry"} and lowered[:1] == ["publish"]:
        out.raise_to(RiskLevel.CRITICAL, "publishes a package", Permission.NETWORK)
        return
    if program in _TEST_RUNNERS or _is_test_invocation(program, lowered):
        out.raise_to(RiskLevel.MEDIUM, "runs tests", Permission.EXECUTE)
        return
    out.raise_to(RiskLevel.MEDIUM, f"runs {program}", Permission.EXECUTE)


def _assess_git(args: list[str], out: CommandAssessment) -> None:
    # Skip global options such as -C <dir> / -c key=value.
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] in {"-c", "-C", "--git-dir", "--work-tree"} else 1
    sub = args[index] if index < len(args) else ""
    rest = args[index + 1 :]
    if sub in _GIT_READ or (sub == "branch" and not any(
        a in {"-d", "-D", "-m", "-M", "--delete"} for a in rest
    )) or (sub == "stash" and rest[:1] in (["list"], ["show"])) or (sub == "tag" and not rest):  # fmt: skip
        out.raise_to(RiskLevel.LOW, f"read-only git {sub}", Permission.READ)
        return
    if sub == "push":
        force = any(a in {"-f", "--force", "--force-with-lease", "--mirror", "--delete"} or
                    a.startswith("+") or a.startswith("--force") for a in rest)  # fmt: skip
        level = RiskLevel.CRITICAL if force else RiskLevel.HIGH
        out.raise_to(level, "git push" + (" (force)" if force else ""), Permission.VCS,
                     Permission.NETWORK)  # fmt: skip
        return
    if sub in _GIT_NETWORK:
        out.raise_to(RiskLevel.MEDIUM, f"git {sub} uses the network", Permission.VCS,
                     Permission.NETWORK)  # fmt: skip
        return
    destructive = (
        (sub == "reset" and "--hard" in rest)
        or (sub == "clean" and any(a.startswith("-") and "f" in a for a in rest))
        or (sub in {"checkout", "restore"} and "." in rest)
        or (sub == "branch" and any(a in {"-d", "-D", "--delete"} for a in rest))
        or (sub == "stash" and rest[:1] in (["drop"], ["clear"]))
        or (sub in {"rebase", "filter-branch", "filter-repo", "update-ref", "gc", "prune"})
    )
    if destructive:
        out.raise_to(RiskLevel.HIGH, f"destructive git {sub}", Permission.VCS, Permission.WRITE)
        return
    out.raise_to(RiskLevel.MEDIUM, f"git {sub or 'command'} changes repository state",
                 Permission.VCS)  # fmt: skip


def _is_package_install(program: str, args: list[str]) -> bool:
    first = args[0] if args else ""
    if program in {"pip", "pip3"} and first in {"install", "download"}:
        return True
    if program in {"python", "python3", "py"} and args[:3] in (["-m", "pip", "install"],):
        return True
    if program == "uv" and (first in {"add", "sync", "lock"} or args[:2] == ["pip", "install"]):
        return True
    if program in {"npm", "pnpm", "yarn", "bun"} and first in {"install", "i", "add", "ci"}:
        return True
    if program == "yarn" and not args:
        return True
    if program in {"cargo"} and first in {"add", "install", "fetch"}:
        return True
    if (program == "go" and first in {"get", "install"}) or (
        program == "go" and args[:2] == ["mod", "download"]
    ):
        return True
    if program in {"poetry", "pipenv", "bundle", "composer", "gem"} and first in {
        "install",
        "add",
        "update",
    }:
        return True
    return program == "dotnet" and first in {"add", "restore"}


def _is_test_invocation(program: str, args: list[str]) -> bool:
    first = args[0] if args else ""
    if program in {"python", "python3", "py"} and args[:2] in (
        ["-m", "pytest"],
        ["-m", "unittest"],
    ):
        return True
    if program in {"npm", "pnpm", "yarn", "bun"} and (
        first == "test" or args[:2] == ["run", "test"]
    ):
        return True
    if (
        program in {"go", "cargo", "dotnet", "mvn", "gradle", "gradlew", "mix", "swift"}
        and first == "test"
    ):
        return True
    return program == "uv" and args[:2] in (["run", "pytest"], ["run", "python"])
