"""Dependency manifests and the internal import graph."""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from forge.tools.analysis.languages import language_of
from forge.tools.workspace import Workspace, WorkspaceError

MANIFEST_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "package.json", "go.mod", "Cargo.toml",
    "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "composer.json", "Pipfile",
}  # fmt: skip


@dataclass
class Manifest:
    path: str
    ecosystem: str
    name: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    dev_dependencies: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        title = f"{self.path} ({self.ecosystem}{', ' + self.name if self.name else ''})"
        lines = [title]
        for label, deps in (("dependencies", self.dependencies), ("dev", self.dev_dependencies)):
            if deps:
                items = ", ".join(
                    f"{k}{v if v and v != '*' else ''}" for k, v in sorted(deps.items())
                )
                lines.append(f"  {label}: {items}")
        return "\n".join(lines)


def is_manifest(path: str) -> bool:
    name = PurePosixPath(path).name
    return name in MANIFEST_NAMES or bool(re.fullmatch(r"requirements[\w.-]*\.(txt|in)", name))


_ECOSYSTEM_BY_NAME = {
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "Gemfile": "ruby",
    "composer.json": "php",
    "Pipfile": "python",
    "setup.py": "python",
    "setup.cfg": "python",
}

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*([^;#]*)")


def _split_requirement(spec: str) -> tuple[str, str] | None:
    match = _REQ.match(spec)
    if not match:
        return None
    return match.group(1), match.group(3).strip().replace(" ", "")


def parse_manifest(path: str, text: str) -> Manifest | None:
    name = PurePosixPath(path).name
    try:
        if name == "pyproject.toml":
            data = tomllib.loads(text)
            manifest = Manifest(path, "python", data.get("project", {}).get("name"))
            for req in data.get("project", {}).get("dependencies", []) or []:
                if parsed := _split_requirement(req):
                    manifest.dependencies[parsed[0]] = parsed[1]
            groups = dict(data.get("project", {}).get("optional-dependencies", {}) or {})
            groups.update(data.get("dependency-groups", {}) or {})
            for reqs in groups.values():
                for req in reqs:
                    if isinstance(req, str) and (parsed := _split_requirement(req)):
                        manifest.dev_dependencies[parsed[0]] = parsed[1]
            poetry = data.get("tool", {}).get("poetry", {})
            for dep, spec in (poetry.get("dependencies", {}) or {}).items():
                if dep != "python":
                    manifest.dependencies[dep] = spec if isinstance(spec, str) else ""
            for group in (poetry.get("group", {}) or {}).values():
                for dep, spec in (group.get("dependencies", {}) or {}).items():
                    manifest.dev_dependencies[dep] = spec if isinstance(spec, str) else ""
            manifest.name = manifest.name or poetry.get("name")
            return manifest
        if re.fullmatch(r"requirements[\w.-]*\.(txt|in)", name):
            manifest = Manifest(path, "python")
            target = (
                manifest.dev_dependencies
                if "dev" in name or "test" in name
                else manifest.dependencies
            )
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "-")):
                    continue
                if parsed := _split_requirement(line):
                    target[parsed[0]] = parsed[1]
            return manifest
        if name == "package.json":
            data = json.loads(text)
            return Manifest(
                path,
                "node",
                data.get("name"),
                dict(data.get("dependencies", {}) or {}),
                {
                    **(data.get("devDependencies", {}) or {}),
                    **(data.get("peerDependencies", {}) or {}),
                },
            )
        if name == "go.mod":
            manifest = Manifest(path, "go")
            module = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
            manifest.name = module.group(1) if module else None
            for dep, version in re.findall(
                r"^\s*(?:require\s+)?([\w./-]+\.[\w./-]+)\s+(v[\w.+-]+)", text, re.MULTILINE
            ):
                manifest.dependencies[dep] = version
            return manifest
        if name == "Cargo.toml":
            data = tomllib.loads(text)
            manifest = Manifest(path, "rust", data.get("package", {}).get("name"))
            for key, target in (("dependencies", manifest.dependencies),
                                ("dev-dependencies", manifest.dev_dependencies)):  # fmt: skip
                for dep, spec in (data.get(key, {}) or {}).items():
                    target[dep] = spec if isinstance(spec, str) else str(spec.get("version", ""))
            return manifest
    except (ValueError, tomllib.TOMLDecodeError, AttributeError, TypeError):
        return Manifest(path, "unknown", name="(unparseable)")
    if name in MANIFEST_NAMES:
        return Manifest(path, _ECOSYSTEM_BY_NAME.get(name, "unknown"))
    return None


def parse_manifests(workspace: Workspace, files: Iterable[str]) -> list[Manifest]:
    manifests: list[Manifest] = []
    for rel in files:
        if not is_manifest(rel) or "node_modules/" in rel:
            continue
        try:
            text = workspace.read_text(rel)
        except WorkspaceError:
            continue
        manifest = parse_manifest(rel, text)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


# -- import graph -------------------------------------------------------------------------------

_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
_SOURCE_ROOTS = {"src", "lib", "python", "source", "app"}
_JS_IMPORT = re.compile(
    r"""(?:import\s+(?:[^'"]*?\s+from\s+)?|export\s+[^'"]*?\s+from\s+|require\s*\(\s*|import\s*\(\s*)['"]([^'"]+)['"]"""
)
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".d.ts")


@dataclass
class ImportGraph:
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    importers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    external: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, source: str, target: str) -> None:
        if source != target:
            self.imports[source].add(target)
            self.importers[target].add(source)

    def neighbors(self, path: str) -> set[str]:
        return set(self.imports.get(path, set())) | set(self.importers.get(path, set()))

    def external_usage(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for modules in self.external.values():
            for module in modules:
                counts[module] += 1
        return dict(counts)


def _python_module_names(path: str) -> list[str]:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    names = [".".join(parts)]
    if len(parts) > 1 and parts[0] in _SOURCE_ROOTS:
        names.append(".".join(parts[1:]))
    return names


def build_import_graph(workspace: Workspace, files: Iterable[str]) -> ImportGraph:
    file_list = [f for f in files if not workspace.is_secret(f)]
    graph = ImportGraph()
    modules: dict[str, str] = {}
    for rel in file_list:
        if rel.endswith(".py"):
            for name in _python_module_names(rel):
                modules.setdefault(name, rel)
    file_set = set(file_list)
    for rel in file_list:
        language = language_of(rel)
        if language not in {"python", "javascript", "typescript", "vue", "svelte"}:
            continue
        try:
            text = workspace.read_text(rel)
        except WorkspaceError:
            continue
        if language == "python":
            _python_edges(rel, text, modules, graph)
        else:
            _js_edges(rel, text, file_set, graph)
    return graph


def _python_edges(rel: str, text: str, modules: dict[str, str], graph: ImportGraph) -> None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return
    own = _python_module_names(rel)
    own_parts = own[-1].split(".") if own else []
    is_package = rel.endswith("__init__.py")
    package = own_parts if is_package else own_parts[:-1]

    def resolve(name: str) -> str | None:
        parts = name.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in modules:
                return modules[candidate]
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve(alias.name)
                if target:
                    graph.add(rel, target)
                elif alias.name.split(".")[0] not in _STDLIB:
                    graph.external[rel].add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = (
                    package[: len(package) - (node.level - 1)] if node.level > 1 else package
                )
                base = ".".join([*base_parts, *(node.module.split(".") if node.module else [])])
            else:
                base = node.module or ""
            resolved_any = False
            for alias in node.names:
                target = resolve(f"{base}.{alias.name}" if base else alias.name)
                if target:
                    graph.add(rel, target)
                    resolved_any = True
            if not resolved_any and base:
                target = resolve(base)
                if target:
                    graph.add(rel, target)
                elif not node.level and base.split(".")[0] not in _STDLIB:
                    graph.external[rel].add(base.split(".")[0])


def _js_edges(rel: str, text: str, files: set[str], graph: ImportGraph) -> None:
    directory = PurePosixPath(rel).parent
    for spec in _JS_IMPORT.findall(text):
        if spec.startswith("."):
            target = _resolve_js(str(directory / spec), files)
            if target:
                graph.add(rel, target)
        elif not spec.startswith(("http:", "https:", "node:")):
            parts = spec.split("/")
            package = "/".join(parts[:2]) if spec.startswith("@") else parts[0]
            graph.external[rel].add(package)


def _resolve_js(base: str, files: set[str]) -> str | None:
    parts: list[str] = []
    for part in base.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part and part != ".":
            parts.append(part)
    normalized = "/".join(parts)
    candidates = [normalized, *(normalized + ext for ext in _JS_EXTS),
                  *(f"{normalized}/index{ext}" for ext in _JS_EXTS)]  # fmt: skip
    for candidate in candidates:
        if candidate in files:
            return candidate
    return None
