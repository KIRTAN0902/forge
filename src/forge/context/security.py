"""Locating security-sensitive code for QA (spec §6 "Security testing", §13 QA context).

This is a triage aid, not a vulnerability scanner: it points QA's attention at places
where authentication, secrets, injection, deserialization and similar issues live, so a
model with limited context reviews the risky lines first.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

SENSITIVE_PATH = re.compile(
    r"(auth|login|logout|signup|password|passwd|token|secret|credential|session|oauth|"
    r"jwt|crypto|cipher|security|permission|acl|rbac|policy|admin|payment|billing|"
    r"checkout|upload|middleware|cors|csrf)",
    re.IGNORECASE,
)

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("dynamic code execution", re.compile(r"(?<![\w.])(eval|exec)\s*\(")),
    ("shell command with shell=True", re.compile(r"shell\s*=\s*True")),
    ("shell command via os.system/popen", re.compile(r"\bos\.(system|popen)\s*\(")),
    ("child_process exec", re.compile(r"\b(child_process|execSync|exec)\s*\(\s*[`'\"].*\$\{")),
    ("unsafe deserialization", re.compile(r"\b(pickle|cPickle|marshal|shelve)\.loads?\s*\(")),
    ("unsafe YAML load", re.compile(r"\byaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)")),
    (
        "SQL built from strings",
        re.compile(
            r"(execute|executemany|raw|query|exec)\s*\(\s*(f[\"']|[\"'][^\"']*[\"']\s*(%|\+|\.format))",
            re.IGNORECASE,
        ),
    ),
    ("HTML injection sink", re.compile(r"(innerHTML|outerHTML|dangerouslySetInnerHTML|v-html)")),
    ("TLS verification disabled", re.compile(r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false")),
    ("weak hash", re.compile(r"\b(md5|sha1)\s*\(", re.IGNORECASE)),
    (
        "hard-coded credential",
        re.compile(
            r"(password|passwd|secret|api_?key|token)\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']",
            re.IGNORECASE,
        ),
    ),
    (
        "permissive CORS",
        re.compile(r"Access-Control-Allow-Origin[\"']?\s*[:,]\s*[\"']\*|origins?\s*=\s*[\"']\*"),
    ),
    ("debug mode enabled", re.compile(r"\bDEBUG\s*=\s*True\b")),
    (
        "JWT verification disabled",
        re.compile(r"verify_signature[\"']?\s*:\s*False|algorithms?\s*=\s*\[?[\"']none"),
    ),
    (
        "path built from user input",
        re.compile(r"(open|send_file|sendFile|readFile)\s*\([^)]*(request|req)\."),
    ),
]


@dataclass(frozen=True)
class SecurityHit:
    path: str
    line: int
    label: str
    snippet: str

    def render(self) -> str:
        return f"{self.path}:{self.line}  [{self.label}]  {self.snippet}"


def is_sensitive_path(path: str) -> bool:
    return bool(SENSITIVE_PATH.search(path))


def scan_text(path: str, text: str, *, limit: int = 50) -> list[SecurityHit]:
    hits: list[SecurityHit] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        for label, pattern in _RULES:
            if pattern.search(line):
                hits.append(SecurityHit(path, number, label, stripped[:160]))
                break
        if len(hits) >= limit:
            break
    return hits


def scan_files(
    paths: Iterable[str], read: Callable[[str], str | None], *, limit: int = 80
) -> list[SecurityHit]:
    hits: list[SecurityHit] = []
    for path in paths:
        text = read(path)
        if text is None:
            continue
        hits.extend(scan_text(path, text, limit=limit - len(hits)))
        if len(hits) >= limit:
            break
    return hits
