from __future__ import annotations

import re
from pathlib import PurePosixPath

_BY_EXTENSION = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".m": "objective-c", ".dart": "dart", ".lua": "lua", ".r": "r", ".jl": "julia",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".hs": "haskell", ".clj": "clojure",
    ".vue": "vue", ".svelte": "svelte",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell", ".bat": "batch",
    ".sql": "sql", ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sass": "scss", ".less": "less",
    ".md": "markdown", ".rst": "restructuredtext", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".xml": "xml", ".proto": "protobuf", ".graphql": "graphql", ".tf": "terraform",
}  # fmt: skip
_BY_NAME = {"dockerfile": "docker", "makefile": "make", "justfile": "make", "rakefile": "ruby",
            "gemfile": "ruby", "procfile": "text", "jenkinsfile": "groovy"}  # fmt: skip

CODE_LANGUAGES = frozenset(
    {
        "python", "javascript", "typescript", "go", "rust", "java", "kotlin", "csharp", "ruby",
        "php", "swift", "scala", "c", "cpp", "objective-c", "dart", "lua", "r", "julia",
        "elixir", "erlang", "haskell", "clojure", "vue", "svelte", "shell", "powershell", "sql",
    }
)  # fmt: skip

_TEST_PATTERNS = [
    re.compile(p)
    for p in (
        r"(^|/)(tests?|__tests__|spec|specs|testing)/",
        r"(^|/)test_[^/]+\.py$",
        r"(^|/)[^/]+_test\.(py|go|rb|exs?)$",
        r"(^|/)[^/]+\.(test|spec)\.[jt]sx?$",
        r"(^|/)[^/]+(Test|Tests|Spec)\.(java|kt|cs|scala|swift)$",
        r"(^|/)conftest\.py$",
    )
]

_GENERATED = re.compile(
    r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|uv\.lock|Cargo\.lock|"
    r"go\.sum|composer\.lock|Gemfile\.lock|[^/]+\.min\.(js|css)|[^/]+\.map)$"
)


def language_of(path: str) -> str | None:
    pure = PurePosixPath(path)
    by_name = _BY_NAME.get(pure.name.lower())
    if by_name:
        return by_name
    return _BY_EXTENSION.get(pure.suffix.lower())


def is_code(path: str) -> bool:
    return language_of(path) in CODE_LANGUAGES


def is_test_path(path: str) -> bool:
    return any(p.search(path) for p in _TEST_PATTERNS)


def is_generated(path: str) -> bool:
    return bool(_GENERATED.search(path))
