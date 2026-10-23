"""Filesystem tools: read, write, edit, delete, list, search, find."""

from __future__ import annotations

import asyncio
import difflib
import re
import shutil
from typing import ClassVar

from pydantic import Field

from forge.core.errors import ToolError
from forge.core.globs import PathMatcher
from forge.policies.types import Permission, RiskLevel
from forge.tools.base import Tool, ToolAssessment, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment
from forge.tools.workspace import WorkspaceError, decode_text, detect_newline, is_binary

MAX_LINE_CHARS = 2000
READ = frozenset({Permission.READ})
WRITE = frozenset({Permission.WRITE})


# -- read_file ----------------------------------------------------------------------------


class ReadFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="First line to return (1-based).")
    limit: int = Field(default=2000, ge=1, le=5000, description="Maximum lines to return.")


class ReadFileOutput(ToolOutput):
    path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str

    def render(self) -> str:
        if self.total_lines == 0:
            return f"{self.path} is empty."
        header = f"{self.path} (lines {self.start_line}-{self.end_line} of {self.total_lines})"
        more = ""
        if self.end_line < self.total_lines:
            more = (
                f"\n[{self.total_lines - self.end_line} more lines; use offset={self.end_line + 1}]"
            )
        return f"{header}\n{self.content}{more}"


class ReadFile(Tool[ReadFileInput, ReadFileOutput]):
    name = "read_file"
    description = (
        "Read a text file. Returns numbered lines (number<TAB>text). Use offset/limit to page "
        "through large files."
    )
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def assess(self, args: ReadFileInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.LOW, READ, paths=[env.workspace.relative(args.path)])

    async def run(self, args: ReadFileInput, env: ToolEnvironment) -> ReadFileOutput:
        text = await asyncio.to_thread(env.workspace.read_text, args.path)
        lines = text.splitlines()
        start = min(args.offset, max(len(lines), 1))
        selected = lines[start - 1 : start - 1 + args.limit]
        width = len(str(start + len(selected)))
        numbered = "\n".join(
            f"{start + i:>{width}}\t{_clip(line)}" for i, line in enumerate(selected)
        )
        return ReadFileOutput(
            path=env.workspace.relative(args.path),
            start_line=start if lines else 0,
            end_line=start + len(selected) - 1 if lines else 0,
            total_lines=len(lines),
            content=numbered,
        )


def _clip(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    return f"{line[:MAX_LINE_CHARS]}... [{len(line) - MAX_LINE_CHARS} chars clipped]"


# -- create_directory ---------------------------------------------------------------------


class CreateDirectoryInput(ToolInput):
    path: str = Field(description="Directory path relative to the workspace root.")
    parents: bool = Field(default=True, description="Create missing parent directories.")
    exist_ok: bool = Field(default=True, description="Succeed when the directory already exists.")


class CreateDirectoryOutput(ToolOutput):
    path: str
    created: bool

    def render(self) -> str:
        verb = "Created" if self.created else "Already exists"
        return f"{verb} directory {self.path}."

    def changed_paths(self) -> list[str]:
        return [self.path] if self.created else []


class CreateDirectory(Tool[CreateDirectoryInput, CreateDirectoryOutput]):
    name = "create_directory"
    description = "Create a directory, including missing parents by default."
    input_model = CreateDirectoryInput
    output_model = CreateDirectoryOutput
    risk = RiskLevel.MEDIUM
    permissions = WRITE

    def assess(self, args: CreateDirectoryInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.MEDIUM, WRITE, paths=[env.workspace.relative(args.path)])

    async def run(self, args: CreateDirectoryInput, env: ToolEnvironment) -> CreateDirectoryOutput:
        target = env.workspace.resolve(args.path)
        rel = env.workspace.relative(target)
        if target.exists() and not target.is_dir():
            raise ToolError(f"{rel} already exists and is not a directory")
        if target.is_dir() and not args.exist_ok:
            raise ToolError(f"{rel} already exists; set exist_ok=true to accept it")
        created = not target.exists()
        try:
            await asyncio.to_thread(target.mkdir, parents=args.parents, exist_ok=args.exist_ok)
        except FileNotFoundError:
            raise ToolError(
                f"parent directory does not exist for {rel}; set parents=true"
            ) from None
        except OSError as exc:
            raise ToolError(f"could not create {rel}: {exc}") from exc
        return CreateDirectoryOutput(path=rel, created=created)


# -- write_file ---------------------------------------------------------------------------


class WriteFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="The complete new file content.")


class WriteFileOutput(ToolOutput):
    path: str
    created: bool
    bytes_written: int
    lines: int
    diff: str = ""

    def render(self) -> str:
        verb = "Created" if self.created else "Overwrote"
        return f"{verb} {self.path} ({self.lines} lines, {self.bytes_written} bytes)."

    def changed_paths(self) -> list[str]:
        return [self.path]


class WriteFile(Tool[WriteFileInput, WriteFileOutput]):
    name = "write_file"
    description = (
        "Create a file or replace its entire content. Parent directories are created. "
        "Prefer edit_file for changing part of an existing file."
    )
    input_model = WriteFileInput
    output_model = WriteFileOutput
    risk = RiskLevel.MEDIUM
    permissions = WRITE

    def assess(self, args: WriteFileInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.MEDIUM, WRITE, paths=[env.workspace.relative(args.path)])

    async def run(self, args: WriteFileInput, env: ToolEnvironment) -> WriteFileOutput:
        content = args.content
        rel = env.workspace.relative(args.path)
        target = env.workspace.resolve(args.path)
        before = ""
        if target.exists() and target.is_file():
            try:
                before = await asyncio.to_thread(env.workspace.read_text, args.path)
            except ToolError:
                before = ""
            # Keep the file's existing line-ending convention.
            existing = decode_text(target.read_bytes()[:65536])
            if detect_newline(existing) == "\r\n" and "\r\n" not in content:
                content = content.replace("\n", "\r\n")
        created = await asyncio.to_thread(env.workspace.write_text, args.path, content)
        return WriteFileOutput(
            path=rel,
            created=created,
            bytes_written=len(content.encode("utf-8")),
            lines=content.count("\n") + (0 if content.endswith("\n") or not content else 1),
            diff=unified_diff(before, content, rel),
        )


# -- edit_file ----------------------------------------------------------------------------


class EditFileInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root.")
    old_string: str = Field(
        description="Exact text to replace, including indentation. Must be unique in the "
        "file unless replace_all is true; include surrounding lines to disambiguate."
    )
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence.")


class EditFileOutput(ToolOutput):
    path: str
    replacements: int
    diff: str

    def render(self) -> str:
        return f"Edited {self.path} ({self.replacements} replacement(s)).\n{self.diff}"

    def changed_paths(self) -> list[str]:
        return [self.path]


class EditFile(Tool[EditFileInput, EditFileOutput]):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. Read the file first; old_string must match "
        "exactly (whitespace included) and be unique unless replace_all is set."
    )
    input_model = EditFileInput
    output_model = EditFileOutput
    risk = RiskLevel.MEDIUM
    permissions = WRITE

    def assess(self, args: EditFileInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.MEDIUM, WRITE, paths=[env.workspace.relative(args.path)])

    async def run(self, args: EditFileInput, env: ToolEnvironment) -> EditFileOutput:
        rel = env.workspace.relative(args.path)
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical; nothing to change")
        if not args.old_string:
            raise ToolError("old_string is empty; use write_file to create or replace a file")
        original = await asyncio.to_thread(env.workspace.read_text, args.path)
        newline = detect_newline(original)
        old, new = args.old_string, args.new_string
        if newline == "\r\n":
            old = old.replace("\r\n", "\n").replace("\n", "\r\n")
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        count = original.count(old)
        if count == 0:
            hint = _closest_hint(original, args.old_string)
            raise ToolError(f"old_string not found in {rel}.{hint} Re-read the file and retry.")
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_string occurs {count} times in {rel}; include more surrounding context "
                "to make it unique, or set replace_all=true"
            )
        updated = original.replace(old, new) if args.replace_all else original.replace(old, new, 1)
        await asyncio.to_thread(env.workspace.write_text, args.path, updated, create_dirs=False)
        return EditFileOutput(
            path=rel,
            replacements=count if args.replace_all else 1,
            diff=unified_diff(original, updated, rel),
        )


def _closest_hint(text: str, needle: str) -> str:
    first_line = next((line.strip() for line in needle.splitlines() if line.strip()), "")
    if not first_line:
        return ""
    for number, line in enumerate(text.splitlines(), start=1):
        if first_line in line:
            return (
                f" Its first line appears at line {number}, so whitespace or the following "
                "lines differ."
            )
    return ""


def unified_diff(before: str, after: str, path: str, context: int = 3, limit: int = 6000) -> str:
    diff = "".join(
        difflib.unified_diff(
            before.replace("\r\n", "\n").splitlines(keepends=True),
            after.replace("\r\n", "\n").splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=context,
        )
    )
    if len(diff) > limit:
        diff = diff[:limit] + f"\n... [diff truncated, {len(diff) - limit} more chars]"
    return diff


# -- delete_file --------------------------------------------------------------------------


class DeleteFileInput(ToolInput):
    path: str = Field(description="File or directory path relative to the workspace root.")
    recursive: bool = Field(default=False, description="Required to delete a directory.")


class DeleteFileOutput(ToolOutput):
    path: str
    deleted_files: int

    def render(self) -> str:
        return f"Deleted {self.path} ({self.deleted_files} file(s))."

    def changed_paths(self) -> list[str]:
        return [self.path]


class DeleteFile(Tool[DeleteFileInput, DeleteFileOutput]):
    name = "delete_file"
    description = "Delete a file, or a directory with recursive=true."
    input_model = DeleteFileInput
    output_model = DeleteFileOutput
    risk = RiskLevel.MEDIUM
    permissions = WRITE

    def assess(self, args: DeleteFileInput, env: ToolEnvironment) -> ToolAssessment:
        rel = env.workspace.relative(args.path)
        if rel == ".":
            raise WorkspaceError("refusing to delete the workspace root")
        risk = RiskLevel.HIGH if args.recursive else RiskLevel.MEDIUM
        return ToolAssessment(
            risk,
            WRITE,
            paths=[rel],
            description="recursive delete" if args.recursive else "delete file",
        )

    async def run(self, args: DeleteFileInput, env: ToolEnvironment) -> DeleteFileOutput:
        target = env.workspace.resolve(args.path)
        rel = env.workspace.relative(target)
        if not target.exists():
            raise ToolError(f"{rel} does not exist")
        if target.is_dir():
            if not args.recursive:
                raise ToolError(f"{rel} is a directory; set recursive=true to delete it")
            count = sum(1 for p in target.rglob("*") if p.is_file())
            await asyncio.to_thread(shutil.rmtree, target)
        else:
            count = 1
            await asyncio.to_thread(target.unlink)
        return DeleteFileOutput(path=rel, deleted_files=count)


# -- list_directory -----------------------------------------------------------------------


class ListDirectoryInput(ToolInput):
    path: str = Field(default=".", description="Directory relative to the workspace root.")
    depth: int = Field(default=2, ge=1, le=6, description="How many levels to show.")


class ListDirectoryOutput(ToolOutput):
    path: str
    entries: list[str]
    total_files: int
    truncated: bool = False

    def render(self) -> str:
        lines = [f"{self.path}/ ({self.total_files} files)", *self.entries]
        if self.truncated:
            lines.append("... [listing truncated; list a subdirectory for more]")
        return "\n".join(lines)


class ListDirectory(Tool[ListDirectoryInput, ListDirectoryOutput]):
    name = "list_directory"
    description = (
        "List files and directories as an indented tree (ignored files such as .git, "
        "node_modules and .gitignore'd paths are hidden). Directories show file counts."
    )
    input_model = ListDirectoryInput
    output_model = ListDirectoryOutput
    max_entries: ClassVar[int] = 400

    def assess(self, args: ListDirectoryInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.LOW, READ, paths=[env.workspace.relative(args.path)])

    async def run(self, args: ListDirectoryInput, env: ToolEnvironment) -> ListDirectoryOutput:
        base = env.workspace.resolve(args.path)
        if not base.is_dir():
            raise ToolError(f"{env.workspace.relative(base)} is not a directory")
        rel_base = env.workspace.relative(base)
        files = await asyncio.to_thread(env.workspace.files, rel_base)
        prefix = "" if rel_base == "." else rel_base + "/"
        entries, truncated = render_tree(
            [f[len(prefix) :] for f in files], args.depth, self.max_entries
        )
        return ListDirectoryOutput(
            path=rel_base, entries=entries, total_files=len(files), truncated=truncated
        )


def render_tree(paths: list[str], depth: int, max_entries: int) -> tuple[list[str], bool]:
    """Indented tree of ``paths`` (relative, POSIX) down to ``depth`` levels."""
    tree: dict[str, object] = {}
    counts: dict[str, int] = {}
    for path in paths:
        parts = path.split("/")
        for i in range(1, len(parts)):
            key = "/".join(parts[:i])
            counts[key] = counts.get(key, 0) + 1
        node = tree
        for part in parts[:depth]:
            node = node.setdefault(part, {})  # type: ignore[assignment]

    lines: list[str] = []

    def walk(node: dict[str, object], prefix: str, level: int) -> bool:
        dirs = sorted(k for k, v in node.items() if v or f"{prefix}{k}" in counts)
        files = sorted(k for k in node if k not in dirs)
        for name in [*dirs, *files]:
            if len(lines) >= max_entries:
                return True
            full = f"{prefix}{name}"
            indent = "  " * level
            if name in dirs:
                lines.append(f"{indent}{name}/ ({counts.get(full, 0)} files)")
                child = node[name]
                if isinstance(child, dict) and child and walk(child, full + "/", level + 1):
                    return True
            else:
                lines.append(f"{indent}{name}")
        return False

    truncated = walk(tree, "", 0)
    return lines, truncated


# -- search -------------------------------------------------------------------------------


class SearchInput(ToolInput):
    pattern: str = Field(min_length=1, description="Regular expression (or literal text).")
    path: str = Field(default=".", description="Directory or file to search.")
    glob: str | None = Field(default=None, description="Only files matching, e.g. '*.py'.")
    literal: bool = Field(default=False, description="Treat pattern as plain text.")
    case_sensitive: bool | None = Field(
        default=None, description="Default: case-insensitive unless the pattern has capitals."
    )
    context_lines: int = Field(default=0, ge=0, le=10)
    max_results: int = Field(default=100, ge=1, le=1000)


class SearchMatch(ToolOutput):
    path: str
    line: int
    text: str
    before: list[str] = []
    after: list[str] = []


class SearchOutput(ToolOutput):
    pattern: str
    matches: list[SearchMatch]
    files_searched: int
    truncated: bool = False
    skipped_secret_files: int = 0

    def render(self) -> str:
        if not self.matches:
            out = f"No matches for {self.pattern!r} in {self.files_searched} files."
        else:
            lines: list[str] = []
            for m in self.matches:
                for offset, text in enumerate(m.before):
                    lines.append(f"{m.path}-{m.line - len(m.before) + offset}-{text}")
                lines.append(f"{m.path}:{m.line}:{m.text}")
                for offset, text in enumerate(m.after):
                    lines.append(f"{m.path}-{m.line + 1 + offset}-{text}")
            out = "\n".join(lines)
            if self.truncated:
                out += "\n... [more matches; narrow the pattern, path or glob]"
        if self.skipped_secret_files:
            out += f"\n({self.skipped_secret_files} secret file(s) not searched)"
        return out


class Search(Tool[SearchInput, SearchOutput]):
    name = "search"
    description = (
        "Search file contents with a regular expression (like grep -rn). Respects .gitignore. "
        "Output lines are path:line:text."
    )
    input_model = SearchInput
    output_model = SearchOutput
    timeout_s = 120.0

    def assess(self, args: SearchInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.LOW, READ)  # secret files are skipped, not read

    async def run(self, args: SearchInput, env: ToolEnvironment) -> SearchOutput:
        return await asyncio.to_thread(self._search, args, env)

    def _search(self, args: SearchInput, env: ToolEnvironment) -> SearchOutput:
        ws = env.workspace
        target = ws.resolve(args.path)
        sensitive = args.case_sensitive
        if sensitive is None:
            sensitive = any(c.isupper() for c in args.pattern)
        source = re.escape(args.pattern) if args.literal else args.pattern
        try:
            regex = re.compile(source, 0 if sensitive else re.IGNORECASE)
        except re.error as exc:
            raise ToolError(
                f"invalid regular expression: {exc}. Set literal=true for plain text"
            ) from None
        candidates = [ws.relative(target)] if target.is_file() else ws.files(ws.relative(target))
        name_filter = PathMatcher([args.glob]) if args.glob else None
        matches: list[SearchMatch] = []
        searched = 0
        skipped = 0
        for rel in candidates:
            if name_filter is not None and not name_filter.matches(rel):
                continue
            if ws.is_secret(rel):
                skipped += 1
                continue
            try:
                data = ws.read_bytes(rel)
            except WorkspaceError:
                continue
            if is_binary(data):
                continue
            searched += 1
            lines = decode_text(data).splitlines()
            for index, line in enumerate(lines):
                if regex.search(line):
                    if len(matches) >= args.max_results:
                        return SearchOutput(
                            pattern=args.pattern,
                            matches=matches,
                            files_searched=searched,
                            truncated=True,
                            skipped_secret_files=skipped,
                        )
                    c = args.context_lines
                    matches.append(
                        SearchMatch(
                            path=rel,
                            line=index + 1,
                            text=_clip(line),
                            before=[_clip(x) for x in lines[max(0, index - c) : index]],
                            after=[_clip(x) for x in lines[index + 1 : index + 1 + c]],
                        )
                    )
        return SearchOutput(
            pattern=args.pattern,
            matches=matches,
            files_searched=searched,
            skipped_secret_files=skipped,
        )


# -- find_files ---------------------------------------------------------------------------


class FindFilesInput(ToolInput):
    pattern: str = Field(description="Glob such as '**/*.py', 'test_*.py' or 'src/**/models.*'.")
    path: str = Field(default=".", description="Directory to search under.")
    max_results: int = Field(default=200, ge=1, le=2000)


class FindFilesOutput(ToolOutput):
    pattern: str
    files: list[str]
    truncated: bool = False

    def render(self) -> str:
        if not self.files:
            return f"No files match {self.pattern!r}."
        more = "\n... [more files; narrow the pattern]" if self.truncated else ""
        return "\n".join(self.files) + more


class FindFiles(Tool[FindFilesInput, FindFilesOutput]):
    name = "find_files"
    description = "Find files by name/path glob. Respects .gitignore."
    input_model = FindFilesInput
    output_model = FindFilesOutput

    def assess(self, args: FindFilesInput, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(RiskLevel.LOW, READ)

    async def run(self, args: FindFilesInput, env: ToolEnvironment) -> FindFilesOutput:
        base = env.workspace.relative(args.path)
        files = await asyncio.to_thread(env.workspace.files, base)
        matcher = PathMatcher([args.pattern])
        prefix = "" if base == "." else base + "/"
        found = [f for f in files if matcher.matches(f[len(prefix) :] if prefix else f)]
        return FindFilesOutput(
            pattern=args.pattern,
            files=found[: args.max_results],
            truncated=len(found) > args.max_results,
        )
