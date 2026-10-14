"""Gitignore-style path patterns.

Used for protected/secret paths and ignore lists. Paths are workspace-relative and use
forward slashes.

* A pattern without a slash (``*.pem``, ``.env``) matches the name at any depth.
* A pattern with a slash (``.git/**``, ``src/*.py``) is anchored at the root.
* ``**`` matches any number of directories; ``*`` and ``?`` never cross ``/``.
* A match on a directory also covers everything inside it.
* ``!pattern`` re-includes; the last matching pattern wins (as in ``.gitignore``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache

_CASE_INSENSITIVE = os.name == "nt"


@lru_cache(maxsize=1024)
def _compile(pattern: str) -> re.Pattern[str]:
    body = pattern.strip()
    if body.endswith("/"):
        body = body.rstrip("/")
    anchored = "/" in body
    body = body.lstrip("/")
    out: list[str] = []
    i = 0
    while i < len(body):
        if body.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif body.startswith("/**", i) and i + 3 == len(body):
            out.append("(?:/.*)?")
            i += 3
        elif body.startswith("**", i):
            out.append(".*")
            i += 2
        elif body[i] == "*":
            out.append("[^/]*")
            i += 1
        elif body[i] == "?":
            out.append("[^/]")
            i += 1
        elif body[i] == "[":
            end = body.find("]", i + 1)
            if end == -1:
                out.append(re.escape(body[i]))
                i += 1
            else:
                out.append(body[i : end + 1])
                i = end + 1
        else:
            out.append(re.escape(body[i]))
            i += 1
    prefix = "" if anchored else "(?:.*/)?"
    flags = re.IGNORECASE if _CASE_INSENSITIVE else 0
    return re.compile(f"^{prefix}{''.join(out)}(?:/.*)?$", flags)


def normalize(path: str) -> str:
    """Workspace-relative POSIX form: forward slashes, no leading ``./``."""
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")


class PathMatcher:
    def __init__(self, patterns: Iterable[str]) -> None:
        self.patterns = [p for p in (p.strip() for p in patterns) if p and not p.startswith("#")]
        self._compiled = [
            (p.startswith("!"), _compile(p[1:] if p.startswith("!") else p)) for p in self.patterns
        ]

    def matches(self, path: str) -> bool:
        rel = normalize(path)
        matched = False
        for negated, regex in self._compiled:
            if regex.match(rel):
                matched = not negated
        return matched

    def __bool__(self) -> bool:
        return bool(self.patterns)
