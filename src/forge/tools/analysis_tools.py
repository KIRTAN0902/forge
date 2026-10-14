"""Repository analysis tools: structure, symbol search, file outline, dependencies."""

from __future__ import annotations

import asyncio
from collections import Counter

from pydantic import Field

from forge.core.errors import ToolError
from forge.tools.analysis import (
    build_import_graph,
    is_manifest,
    is_test_path,
    language_of,
    parse_manifests,
)
from forge.tools.base import Tool, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment
from forge.tools.filesystem import render_tree
from forge.tools.testing import detect_test_command

KEY_FILE_NAMES = {
    "readme.md", "readme.rst", "readme.txt", "readme", "contributing.md", "architecture.md",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yaml", "makefile",
    "justfile", ".env.example", "tsconfig.json", "vite.config.ts", "next.config.js",
    "manage.py", "alembic.ini", "pytest.ini", "tox.ini", "noxfile.py", ".pre-commit-config.yaml",
}  # fmt: skip


def key_files(files: list[str]) -> list[str]:
    found = []
    for rel in files:
        name = rel.rsplit("/", 1)[-1].lower()
        if (
            (rel.count("/") <= 1 and (name in KEY_FILE_NAMES or is_manifest(rel)))
            or rel.startswith(".github/workflows/")
            or rel == ".gitlab-ci.yml"
        ):
            found.append(rel)
    return found


# -- repository_structure ------------------------------------------------------------------


class StructureInput(ToolInput):
    path: str = Field(default=".", description="Directory to describe.")
    depth: int = Field(default=3, ge=1, le=6)


class StructureOutput(ToolOutput):
    path: str
    total_files: int
    languages: dict[str, int]
    key_files: list[str]
    test_files: int
    test_command: str | None
    tree: list[str]
    truncated: bool = False

    def render(self) -> str:
        langs = ", ".join(f"{name} {count}" for name, count in self.languages.items()) or "none"
        lines = [
            f"Repository: {self.path} ({self.total_files} files)",
            f"Languages (files): {langs}",
            f"Key files: {', '.join(self.key_files) or 'none'}",
            f"Tests: {self.test_files} test files; command: {self.test_command or 'not detected'}",
            "",
            *self.tree,
        ]
        if self.truncated:
            lines.append("... [tree truncated; use list_directory on a subdirectory]")
        return "\n".join(lines)


class RepositoryStructure(Tool[StructureInput, StructureOutput]):
    name = "repository_structure"
    description = (
        "Overview of the repository: directory tree with file counts, languages, key files "
        "(README, manifests, CI, Docker), and how tests are run. Start here in a new repo."
    )
    input_model = StructureInput
    output_model = StructureOutput

    async def run(self, args: StructureInput, env: ToolEnvironment) -> StructureOutput:
        rel = env.workspace.relative(args.path)
        files = await asyncio.to_thread(env.workspace.files, rel)
        prefix = "" if rel == "." else rel + "/"
        tree, truncated = render_tree([f[len(prefix) :] for f in files], args.depth, 300)
        languages = Counter(lang for f in files if (lang := language_of(f)))
        detected = detect_test_command(env.workspace.resolve(rel))
        return StructureOutput(
            path=rel,
            total_files=len(files),
            languages=dict(languages.most_common(12)),
            key_files=key_files(files),
            test_files=sum(1 for f in files if is_test_path(f)),
            test_command=env.test_command or (detected.command if detected else None),
            tree=tree,
            truncated=truncated,
        )


# -- symbol_search / file_outline -----------------------------------------------------------


class SymbolSearchInput(ToolInput):
    query: str = Field(min_length=1, description="Symbol name or part of it.")
    kind: str | None = Field(
        default=None, description="class, function, method, interface, struct, ..."
    )
    path_prefix: str | None = Field(default=None, description="Only under this directory.")
    limit: int = Field(default=30, ge=1, le=200)


class SymbolsOutput(ToolOutput):
    query: str
    symbols: list[str]

    def render(self) -> str:
        if not self.symbols:
            return f"No symbols match {self.query!r}."
        return "\n".join(self.symbols)


class SymbolSearch(Tool[SymbolSearchInput, SymbolsOutput]):
    name = "symbol_search"
    description = (
        "Find where classes, functions and methods are defined (path:line and signature). "
        "Faster and more precise than text search for definitions."
    )
    input_model = SymbolSearchInput
    output_model = SymbolsOutput
    timeout_s = 120.0

    async def run(self, args: SymbolSearchInput, env: ToolEnvironment) -> SymbolsOutput:
        await env.refresh_symbols()
        found = env.symbols.search(
            args.query, kind=args.kind, path_prefix=args.path_prefix, limit=args.limit
        )
        return SymbolsOutput(query=args.query, symbols=[s.render() for s in found])


class OutlineInput(ToolInput):
    path: str = Field(description="File to outline.")


class FileOutline(Tool[OutlineInput, SymbolsOutput]):
    name = "file_outline"
    description = "List the classes, functions and methods defined in a file, with line ranges."
    input_model = OutlineInput
    output_model = SymbolsOutput

    async def run(self, args: OutlineInput, env: ToolEnvironment) -> SymbolsOutput:
        rel = env.workspace.relative(args.path)
        if not env.workspace.resolve(rel).is_file():
            raise ToolError(f"{rel} is not a file")
        await asyncio.to_thread(env.symbols.refresh_paths, env.workspace, [rel])
        symbols = env.symbols.symbols(rel)
        rendered = [
            ("  " if s.parent else "") + s.render().split("  ", 1)[-1] + f"  (line {s.line})"
            for s in symbols
        ]
        return SymbolsOutput(query=rel, symbols=rendered)


# -- dependency_analysis ----------------------------------------------------------------------


class DependencyInput(ToolInput):
    path: str | None = Field(
        default=None,
        description="A source file: show what it imports and what imports it. Omit for a "
        "project-wide summary of declared and used dependencies.",
    )


class DependencyOutput(ToolOutput):
    summary: str

    def render(self) -> str:
        return self.summary


class DependencyAnalysis(Tool[DependencyInput, DependencyOutput]):
    name = "dependency_analysis"
    description = (
        "Without path: declared dependencies from manifests and the most-used external "
        "imports. With path: the file's internal imports, the files importing it (impact of "
        "a change), and its external imports."
    )
    input_model = DependencyInput
    output_model = DependencyOutput
    timeout_s = 180.0

    async def run(self, args: DependencyInput, env: ToolEnvironment) -> DependencyOutput:
        return await asyncio.to_thread(self._analyze, args, env)

    def _analyze(self, args: DependencyInput, env: ToolEnvironment) -> DependencyOutput:
        files = env.workspace.files()
        graph = build_import_graph(env.workspace, files)
        if args.path:
            rel = env.workspace.relative(args.path)
            if rel not in set(files):
                raise ToolError(f"{rel} is not a tracked source file")
            lines = [f"{rel}"]
            for title, items in (
                ("imports (internal)", graph.imports.get(rel, set())),
                ("imported by", graph.importers.get(rel, set())),
                ("imports (external)", graph.external.get(rel, set())),
            ):
                lines.append(f"{title}: " + (", ".join(sorted(items)) if items else "none"))
            return DependencyOutput(summary="\n".join(lines))
        manifests = parse_manifests(env.workspace, files)
        lines = ["Declared dependencies:"]
        lines += [m.render() for m in manifests] or ["  (no manifests found)"]
        usage = sorted(graph.external_usage().items(), key=lambda kv: (-kv[1], kv[0]))[:30]
        if usage:
            lines.append("Most-imported external modules (files importing):")
            lines.append("  " + ", ".join(f"{name} ({count})" for name, count in usage))
        hubs = sorted(graph.importers.items(), key=lambda kv: -len(kv[1]))[:10]
        if hubs:
            lines.append("Most-depended-on internal files (importers):")
            lines += [f"  {path} ({len(users)})" for path, users in hubs]
        return DependencyOutput(summary="\n".join(lines))
