"""Symbol extraction and search.

Python is parsed with ``ast`` (exact names, nesting and line ranges). Other languages use
per-language regular expressions: approximate, but dependency-free and good enough to
answer "where is X defined?" and to outline a file that is too large to show in full.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass

from forge.tools.analysis.languages import language_of
from forge.tools.workspace import Workspace, WorkspaceError

MAX_PARSE_BYTES = 1_000_000


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    path: str
    line: int
    end_line: int | None = None
    parent: str | None = None
    signature: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def render(self) -> str:
        span = (
            f"{self.line}-{self.end_line}"
            if self.end_line and self.end_line != self.line
            else str(self.line)
        )
        label = self.signature or f"{self.kind} {self.qualified_name}"
        return f"{self.path}:{span}  {label}"


# -- python ----------------------------------------------------------------------------------


def _python_symbols(path: str, text: str) -> list[Symbol]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _regex_symbols(path, text, "python")
    symbols: list[Symbol] = []

    def visit(nodes: Iterable[ast.stmt], parent: str | None, in_class: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        path=path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        parent=parent,
                        signature=f"class {node.name}({bases})" if bases else f"class {node.name}",
                    )
                )
                qualified = f"{parent}.{node.name}" if parent else node.name
                visit(node.body, qualified, True)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                signature = f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="method" if in_class else "function",
                        path=path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        parent=parent,
                        signature=_shorten(signature),
                    )
                )
            elif isinstance(node, ast.Assign | ast.AnnAssign) and parent is None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper() and len(target.id) > 1:
                        symbols.append(
                            Symbol(
                                name=target.id,
                                kind="constant",
                                path=path,
                                line=node.lineno,
                                end_line=node.end_lineno,
                            )
                        )

    visit(tree.body, None, False)
    return symbols


def _shorten(signature: str, limit: int = 160) -> str:
    return signature if len(signature) <= limit else signature[: limit - 3] + "..."


# -- regex languages --------------------------------------------------------------------------

_ID = r"[A-Za-z_$][\w$]*"
_RULES: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "javascript": [
        ("class", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+({_ID})")),
        ("function", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*({_ID})\s*[(<]")),
        ("function", re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+({_ID})\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|{_ID})\s*(?::[^=]+)?=>")),
        ("interface", re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+({_ID})")),
        ("type", re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?type\s+({_ID})\s*(?:<[^>]*>)?\s*=")),
        ("enum", re.compile(rf"^\s*(?:export\s+)?(?:const\s+)?enum\s+({_ID})")),
        ("method", re.compile(rf"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*({_ID})\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^{{]+)?\{{\s*$")),
    ],
    "go": [
        ("method", re.compile(r"^func\s+\([^)]*\)\s*(\w+)\s*[\[(]")),
        ("function", re.compile(r"^func\s+(\w+)\s*[\[(]")),
        ("struct", re.compile(r"^type\s+(\w+)\s+struct\b")),
        ("interface", re.compile(r"^type\s+(\w+)\s+interface\b")),
        ("type", re.compile(r"^type\s+(\w+)\s+\w")),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")),
        ("struct", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)")),
        ("enum", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)")),
        ("trait", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)")),
        ("impl", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)")),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:(?:public|private|protected|abstract|final|static|sealed|data|open|internal)\s+)*(?:class|record)\s+(\w+)")),
        ("interface", re.compile(r"^\s*(?:(?:public|private|protected|sealed|internal)\s+)*interface\s+(\w+)")),
        ("enum", re.compile(r"^\s*(?:(?:public|private|protected|internal)\s+)*enum\s+(?:class\s+)?(\w+)")),
        ("method", re.compile(r"^\s+(?:(?:public|private|protected|static|final|abstract|synchronized|override|async|virtual|internal|suspend)\s+)+[\w<>\[\],.?\s]*?\b(\w+)\s*\([^;]*$")),
        ("function", re.compile(r"^\s*(?:(?:private|public|internal|suspend|inline)\s+)*fun\s+(?:<[^>]*>\s*)?(?:\w+\.)?(\w+)\s*\(")),
    ],
    "ruby": [
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*(?:::\w+)*)")),
        ("method", re.compile(r"^\s*def\s+(?:self\.)?(\w+[?!=]?)")),
    ],
    "php": [
        ("class", re.compile(r"^\s*(?:(?:abstract|final)\s+)?(?:class|interface|trait|enum)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?(\w+)\s*\(")),
    ],
    "python": [
        ("class", re.compile(r"^\s*class\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")),
    ],
    "c": [
        ("struct", re.compile(r"^\s*(?:typedef\s+)?struct\s+(\w+)\s*\{")),
        ("function", re.compile(r"^(?!\s)(?:[\w*]+\s+)+\**(\w+)\s*\([^;]*\)\s*\{?\s*$")),
    ],
    "swift": [
        ("class", re.compile(r"^\s*(?:(?:public|private|internal|open|final)\s+)*(?:class|struct|enum|protocol|actor)\s+(\w+)")),
        ("function", re.compile(r"^\s*(?:(?:public|private|internal|open|static|override|mutating)\s+)*func\s+(\w+)")),
    ],
}  # fmt: skip
_RULES["typescript"] = _RULES["javascript"]
_RULES["vue"] = _RULES["javascript"]
_RULES["svelte"] = _RULES["javascript"]
_RULES["kotlin"] = _RULES["java"]
_RULES["csharp"] = _RULES["java"]
_RULES["scala"] = _RULES["java"]
_RULES["cpp"] = [*_RULES["c"], ("class", re.compile(r"^\s*(?:class|struct)\s+(\w+)\s*(?::|\{|$)"))]
_RULES["dart"] = _RULES["java"]

_NOT_NAMES = {"if", "for", "while", "switch", "catch", "return", "function", "else", "new",
              "constructor", "super", "await", "typeof", "sizeof", "do", "try"}  # fmt: skip


def _regex_symbols(path: str, text: str, language: str) -> list[Symbol]:
    rules = _RULES.get(language)
    if not rules:
        return []
    symbols: list[Symbol] = []
    current_class: str | None = None
    class_indent = -1
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith(("//", "#", "*", "/*")):
            continue
        indent = len(line) - len(line.lstrip())
        if current_class is not None and indent <= class_indent and line.strip() not in {"}", "};"}:
            current_class = None
        for kind, pattern in rules:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            if name in _NOT_NAMES:
                break
            if kind == "method" and current_class is None and language not in {"ruby", "go"}:
                break  # an indented call-like line outside a class is not a definition
            parent = (
                current_class if kind in {"method", "function"} and indent > class_indent else None
            )
            symbols.append(
                Symbol(
                    name=name,
                    kind="method" if parent and kind == "function" else kind,
                    path=path,
                    line=number,
                    parent=parent,
                    signature=_shorten(line.strip().rstrip("{").strip()),
                )
            )
            if kind in {"class", "struct", "interface", "trait", "impl", "enum"}:
                current_class, class_indent = name, indent
            break
    return symbols


def extract_symbols(path: str, text: str) -> list[Symbol]:
    language = language_of(path)
    if language == "python":
        return _python_symbols(path, text)
    if language is None:
        return []
    return _regex_symbols(path, text, language)


# -- index ----------------------------------------------------------------------------------


class SymbolIndex:
    """Lazily (re)built per file; files are re-parsed only when their mtime/size change."""

    def __init__(self) -> None:
        self._files: dict[str, tuple[tuple[int, int], list[Symbol]]] = {}

    def refresh(self, workspace: Workspace, files: Iterable[str]) -> None:
        """Bring the index in line with ``files`` (the full file list)."""
        seen = self.refresh_paths(workspace, files)
        for stale in set(self._files) - seen:
            del self._files[stale]

    def refresh_paths(self, workspace: Workspace, paths: Iterable[str]) -> set[str]:
        """Re-parse only ``paths`` whose content changed; returns the indexable ones."""
        seen: set[str] = set()
        for rel in paths:
            if language_of(rel) is None or workspace.is_secret(rel):
                continue
            seen.add(rel)
            try:
                stat = workspace.resolve(rel).stat()
            except OSError:
                self._files.pop(rel, None)
                continue
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self._files.get(rel)
            if cached is not None and cached[0] == key:
                continue
            if stat.st_size > MAX_PARSE_BYTES:
                self._files[rel] = (key, [])
                continue
            try:
                text = workspace.read_text(rel)
            except WorkspaceError:
                self._files[rel] = (key, [])
                continue
            self._files[rel] = (key, extract_symbols(rel, text))
        return seen

    def symbols(self, path: str) -> list[Symbol]:
        cached = self._files.get(path)
        return list(cached[1]) if cached else []

    def all(self) -> list[Symbol]:
        return [s for _, symbols in self._files.values() for s in symbols]

    def names_by_file(self) -> dict[str, set[str]]:
        return {
            path: {s.name.lower() for s in symbols} for path, (_, symbols) in self._files.items()
        }

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        path_prefix: str | None = None,
        limit: int = 50,
    ) -> list[Symbol]:
        q = query.strip().lower()
        if not q:
            return []
        q_tokens = set(split_identifier(query))
        scored: list[tuple[float, Symbol]] = []
        for symbol in self.all():
            if kind and symbol.kind != kind:
                continue
            if path_prefix and not symbol.path.startswith(path_prefix.strip("./")):
                continue
            name = symbol.name.lower()
            qualified = symbol.qualified_name.lower()
            if name == q or qualified == q:
                score = 100.0
            elif name.startswith(q):
                score = 80.0
            elif q in name:
                score = 60.0
            elif q in qualified:
                score = 50.0
            else:
                overlap = q_tokens & set(split_identifier(symbol.name))
                if not overlap:
                    continue
                score = 20.0 + 20.0 * len(overlap) / max(len(q_tokens), 1)
            if symbol.kind in {"class", "struct", "interface", "trait"}:
                score += 5
            scored.append((score, symbol))
        scored.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path, item[1].line))
        return [symbol for _, symbol in scored[:limit]]


_IDENT_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def split_identifier(name: str) -> list[str]:
    """``getUserByID`` / ``get_user_by_id`` -> ``['get', 'user', 'by', 'id']``."""
    return [part.lower() for part in _IDENT_SPLIT.findall(name)]
