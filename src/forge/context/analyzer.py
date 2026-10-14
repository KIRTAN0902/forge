"""Repository analysis: one structured picture of the codebase that every role draws on."""

from __future__ import annotations

import asyncio
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from forge.core.contracts import RepositorySnapshot
from forge.tools.analysis import (
    ImportGraph,
    Manifest,
    SymbolIndex,
    build_import_graph,
    is_code,
    is_generated,
    is_test_path,
    language_of,
    parse_manifests,
)
from forge.tools.analysis_tools import key_files
from forge.tools.filesystem import render_tree
from forge.tools.git import GitRepo
from forge.tools.testing import detect_test_command
from forge.tools.workspace import Workspace, WorkspaceError, decode_text, is_binary

MAX_TEXT_BYTES = 400_000

_ENTRY_NAMES = {
    "main.py",
    "__main__.py",
    "app.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "cli.py",
    "server.py",
    "index.js",
    "index.ts",
    "main.js",
    "main.ts",
    "server.js",
    "server.ts",
    "app.js",
    "app.ts",
    "main.go",
    "main.rs",
    "lib.rs",
    "Program.cs",
    "Main.java",
}


@dataclass(frozen=True)
class FileRecord:
    path: str
    language: str | None
    size: int
    is_test: bool
    is_code: bool
    is_generated: bool


@dataclass
class TestSetup:
    __test__ = False

    framework: str | None
    command: str | None
    test_files: list[str]
    test_dirs: list[str]


@dataclass
class GitInfo:
    branch: str | None
    head: str | None
    dirty: bool
    changed_files: list[str]
    recent_commits: list[str]


@dataclass
class RepositoryAnalysis:
    workspace: Workspace
    files: list[FileRecord]
    languages: dict[str, int]
    key_files: list[str]
    entry_points: list[str]
    manifests: list[Manifest]
    symbols: SymbolIndex
    imports: ImportGraph
    tests: TestSetup
    git: GitInfo | None
    readme: str | None
    _text_cache: dict[str, tuple[tuple[int, int], str | None]] = field(
        default_factory=dict, repr=False
    )

    @property
    def root(self) -> Path:
        return self.workspace.root

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def record(self, path: str) -> FileRecord | None:
        return next((f for f in self.files if f.path == path), None)

    @property
    def primary_language(self) -> str | None:
        code = Counter({k: v for k, v in self.languages.items()})
        for language, _ in code.most_common():
            if language not in {"markdown", "json", "yaml", "toml", "text", "ini", "xml"}:
                return language
        return None

    def read(self, path: str) -> str | None:
        """Current text of ``path`` (re-read if it changed); None for binary/secret/huge."""
        if self.workspace.is_secret(path):
            return None
        try:
            stat = self.workspace.resolve(path).stat()
        except (OSError, WorkspaceError):
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._text_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        text: str | None = None
        if stat.st_size <= MAX_TEXT_BYTES:
            try:
                data = self.workspace.read_bytes(path)
                text = None if is_binary(data) else decode_text(data)
            except WorkspaceError:
                text = None
        self._text_cache[path] = (key, text)
        return text

    def tree(self, depth: int = 3, max_entries: int = 250) -> str:
        lines, truncated = render_tree(self.paths, depth, max_entries)
        text = "\n".join(lines)
        return text + ("\n... (tree truncated)" if truncated else "")

    def tests_for(self, path: str) -> list[str]:
        """Test files that look like they exercise ``path`` (by name or by import)."""
        stem = PurePosixPath(path).stem.lower()
        if not stem or stem == "__init__":
            return []
        names = {
            f"test_{stem}",
            f"{stem}_test",
            f"{stem}.test",
            f"{stem}.spec",
            f"{stem}test",
            f"{stem}tests",
            f"{stem}spec",
        }
        found = [
            t
            for t in self.tests.test_files
            if PurePosixPath(t).stem.lower() in names
            or PurePosixPath(t).name.lower().split(".")[0] in names
        ]
        found += [i for i in self.imports.importers.get(path, set()) if is_test_path(i)]
        return sorted(set(found))

    def summary(self) -> str:
        top: Counter[str] = Counter()
        for f in self.files:
            top[f.path.split("/", 1)[0] if "/" in f.path else "(root)"] += 1
        langs = ", ".join(f"{k} ({v})" for k, v in list(self.languages.items())[:6]) or "unknown"
        lines = [
            f"{len(self.files)} files; languages: {langs}",
            "Top-level layout: "
            + ", ".join(
                f"{d}/ ({n})" if d != "(root)" else f"root ({n})" for d, n in top.most_common(12)
            ),
        ]
        if self.entry_points:
            lines.append(f"Entry points: {', '.join(self.entry_points[:10])}")
        if self.manifests:
            lines.append(
                "Manifests: " + ", ".join(f"{m.path} ({m.ecosystem})" for m in self.manifests)
            )
        hubs = sorted(self.imports.importers.items(), key=lambda kv: -len(kv[1]))[:6]
        if hubs:
            lines.append(
                "Most-imported modules: " + ", ".join(f"{p} ({len(u)})" for p, u in hubs if u)
            )
        if self.tests.command:
            lines.append(
                f"Tests: {len(self.tests.test_files)} files"
                f" in {', '.join(self.tests.test_dirs) or 'various places'};"
                f" run with `{self.tests.command}`"
            )
        return "\n".join(lines)

    def to_snapshot(self) -> RepositorySnapshot:
        return RepositorySnapshot(
            root=str(self.root),
            vcs="git" if self.git else None,
            branch=self.git.branch if self.git else None,
            head=self.git.head if self.git else None,
            dirty=self.git.dirty if self.git else False,
            summary=self.summary(),
        )


class RepositoryAnalyzer:
    def __init__(
        self,
        workspace: Workspace,
        *,
        symbols: SymbolIndex | None = None,
        test_command: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.symbols = symbols or SymbolIndex()
        self.test_command = test_command

    async def analyze(self, git: GitRepo | None = None) -> RepositoryAnalysis:
        analysis = await asyncio.to_thread(self._analyze_files)
        if git is not None:
            analysis.git = await _git_info(git)
        return analysis

    def _analyze_files(self) -> RepositoryAnalysis:
        ws = self.workspace
        paths = ws.files()
        records = []
        for rel in paths:
            try:
                size = ws.resolve(rel).stat().st_size
            except OSError:
                continue
            records.append(
                FileRecord(
                    path=rel,
                    language=language_of(rel),
                    size=size,
                    is_test=is_test_path(rel),
                    is_code=is_code(rel),
                    is_generated=is_generated(rel),
                )
            )
        languages = Counter(r.language for r in records if r.language and not r.is_generated)
        self.symbols.refresh(ws, paths)
        detected = detect_test_command(ws.root)
        test_files = [r.path for r in records if r.is_test and r.is_code]
        test_dirs = sorted({p.split("/", 1)[0] for p in test_files if "/" in p})
        tests = TestSetup(
            framework=detected.framework if detected else None,
            command=self.test_command or (detected.command if detected else None),
            test_files=test_files,
            test_dirs=test_dirs,
        )
        readme = next(
            (p for p in paths if p.lower() in {"readme.md", "readme.rst", "readme.txt", "readme"}),
            None,
        )
        readme_text = None
        if readme:
            try:
                readme_text = "\n".join(ws.read_text(readme).splitlines()[:80])
            except WorkspaceError:
                readme_text = None
        manifests = parse_manifests(ws, paths)
        return RepositoryAnalysis(
            workspace=ws,
            files=records,
            languages=dict(languages.most_common()),
            key_files=key_files(paths),
            entry_points=_entry_points(paths, manifests, ws),
            manifests=manifests,
            symbols=self.symbols,
            imports=build_import_graph(ws, paths),
            tests=tests,
            git=None,
            readme=readme_text,
        )


def _entry_points(paths: list[str], manifests: list[Manifest], ws: Workspace) -> list[str]:
    found = [p for p in paths if PurePosixPath(p).name in _ENTRY_NAMES and not is_test_path(p)]
    for manifest in manifests:
        if manifest.path.endswith("pyproject.toml"):
            try:
                data = tomllib.loads(ws.read_text(manifest.path))
            except (WorkspaceError, tomllib.TOMLDecodeError):
                continue
            for name, target in (data.get("project", {}).get("scripts", {}) or {}).items():
                found.append(f"{name} -> {target}")
    found.sort(key=lambda p: (p.count("/"), p))
    return found[:15]


async def _git_info(repo: GitRepo) -> GitInfo:
    status = await repo.status()
    log = await repo.log(limit=8)
    return GitInfo(
        branch=status.branch,
        head=status.head,
        dirty=status.dirty,
        changed_files=sorted({*status.staged, *status.unstaged, *status.untracked}),
        recent_commits=[line for line in log.splitlines() if line.strip()],
    )
