"""The workspace: the only part of the filesystem tools may touch.

Escaping the workspace (``..``, absolute paths, symlinks pointing outside) is not a
policy decision; it is impossible through this API.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

from forge.core.errors import ToolError
from forge.core.globs import PathMatcher, normalize
from forge.policies.rules import DEFAULT_SECRET_PATHS

DEFAULT_IGNORES: tuple[str, ...] = (
    ".git/",
    ".hg/",
    ".svn/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".nox/",
    ".forge/",
    ".next/",
    ".nuxt/",
    ".gradle/",
    ".idea/",
    "*.egg-info/",
    ".DS_Store",
)

BINARY_SNIFF_BYTES = 8192


class WorkspaceError(ToolError):
    """A path is outside the workspace, missing, binary, or otherwise unusable."""


class Workspace:
    def __init__(
        self,
        root: Path | str,
        *,
        ignore: Iterable[str] = (),
        secret_paths: Iterable[str] = DEFAULT_SECRET_PATHS,
        max_file_bytes: int = 5_000_000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")
        self.ignore = PathMatcher([*DEFAULT_IGNORES, *ignore])
        self.secrets = PathMatcher(secret_paths)
        self.max_file_bytes = max_file_bytes

    # -- paths -----------------------------------------------------------------------

    def resolve(self, path: str | Path) -> Path:
        """Absolute path inside the workspace, or ``WorkspaceError``."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path {str(path)!r} is outside the workspace")
        return resolved

    def relative(self, path: str | Path) -> str:
        resolved = self.resolve(path)
        if resolved == self.root:
            return "."
        return resolved.relative_to(self.root).as_posix()

    def is_ignored(self, rel_path: str) -> bool:
        return self.ignore.matches(rel_path)

    def is_secret(self, rel_path: str) -> bool:
        return self.secrets.matches(rel_path)

    # -- file content ----------------------------------------------------------------

    def read_bytes(self, path: str | Path) -> bytes:
        resolved = self.resolve(path)
        if not resolved.exists():
            raise WorkspaceError(f"file not found: {self.relative(resolved)}")
        if resolved.is_dir():
            raise WorkspaceError(f"{self.relative(resolved)} is a directory")
        size = resolved.stat().st_size
        if size > self.max_file_bytes:
            raise WorkspaceError(
                f"{self.relative(resolved)} is {size} bytes (limit {self.max_file_bytes})"
            )
        return resolved.read_bytes()

    def read_text(self, path: str | Path) -> str:
        data = self.read_bytes(path)
        if is_binary(data):
            raise WorkspaceError(f"{self.relative(path)} looks like a binary file")
        return decode_text(data)

    def write_text(self, path: str | Path, text: str, *, create_dirs: bool = True) -> bool:
        """Write ``text`` exactly (no newline translation). Returns True if created."""
        resolved = self.resolve(path)
        if resolved.is_dir():
            raise WorkspaceError(f"{self.relative(resolved)} is a directory")
        created = not resolved.exists()
        if create_dirs:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            raise WorkspaceError(f"directory does not exist: {self.relative(resolved.parent)}")
        with open(resolved, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return created

    # -- listing ---------------------------------------------------------------------

    def files(self, start: str = ".", *, limit: int = 50_000) -> list[str]:
        """Workspace-relative files under ``start``, honouring .gitignore when in git."""
        base = self.resolve(start)
        listed = self._git_files(base)
        if listed is None:
            listed = list(self._walk(base))
        result: list[str] = []
        for rel in listed:
            if self.is_ignored(rel):
                continue
            result.append(rel)
            if len(result) >= limit:
                break
        result.sort()
        return result

    def _git_files(self, base: Path) -> list[str] | None:
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(base),
                    "ls-files",
                    "-co",
                    "--exclude-standard",
                    "-z",
                    "--full-name",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        top = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if top.returncode != 0:
            return None
        git_root = Path(top.stdout.strip()).resolve()
        out: list[str] = []
        for name in proc.stdout.decode("utf-8", "replace").split("\0"):
            if not name:
                continue
            absolute = (git_root / name).resolve()
            if absolute != self.root and self.root not in absolute.parents:
                continue
            if not absolute.is_file():
                continue  # deleted but still tracked
            out.append(absolute.relative_to(self.root).as_posix())
        return out

    def _walk(self, base: Path) -> Iterator[str]:
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            rel_dir = "" if rel_dir == "." else rel_dir
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not self.is_ignored(f"{rel_dir}/{d}".lstrip("/") + "/")
                and not self.is_ignored(f"{rel_dir}/{d}".lstrip("/"))
            )
            for name in filenames:
                yield normalize(f"{rel_dir}/{name}")


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:BINARY_SNIFF_BYTES]


def decode_text(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def detect_newline(text: str) -> str:
    crlf = text.count("\r\n")
    return "\r\n" if crlf and crlf >= text.count("\n") - crlf else "\n"
