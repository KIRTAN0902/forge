"""Test-command detection and result parsing for common ecosystems."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCommand:
    __test__ = False  # not a pytest class

    command: str
    framework: str
    accepts_paths: bool


def detect_test_command(root: Path) -> TestCommand | None:
    """Best guess at how this project runs its tests."""
    pyproject = root / "pyproject.toml"
    py_markers = [pyproject, root / "pytest.ini", root / "setup.cfg", root / "tox.ini",
                  root / "setup.py"]  # fmt: skip
    has_python_tests = any(m.exists() for m in py_markers) or any(
        (root / d).is_dir() for d in ("tests", "test")
    )
    if has_python_tests and _has_python_files(root):
        return TestCommand(f"{_python_runner(root)} -m pytest -q", "pytest", accepts_paths=True)

    package_json = root / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if isinstance(scripts, dict) and "test" in scripts:
            manager = "npm"
            if (root / "pnpm-lock.yaml").exists():
                manager = "pnpm"
            elif (root / "yarn.lock").exists():
                manager = "yarn"
            elif (root / "bun.lockb").exists() or (root / "bun.lock").exists():
                manager = "bun"
            command = "npm test --silent" if manager == "npm" else f"{manager} test"
            framework = "jest" if "jest" in str(scripts["test"]) else (
                "vitest" if "vitest" in str(scripts["test"]) else "npm")  # fmt: skip
            return TestCommand(command, framework, accepts_paths=False)
    if (root / "go.mod").exists():
        return TestCommand("go test ./...", "go", accepts_paths=False)
    if (root / "Cargo.toml").exists():
        return TestCommand("cargo test", "cargo", accepts_paths=False)
    if (root / "pom.xml").exists():
        return TestCommand("mvn -q test", "maven", accepts_paths=False)
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        wrapper = "./gradlew" if (root / "gradlew").exists() else "gradle"
        return TestCommand(f"{wrapper} test", "gradle", accepts_paths=False)
    return None


def _has_python_files(root: Path) -> bool:
    for path in root.rglob("*.py"):
        parts = set(path.relative_to(root).parts)
        if not parts & {".venv", "venv", "node_modules", ".git"}:
            return True
    return False


def _python_runner(root: Path) -> str:
    if (root / "uv.lock").exists() and shutil.which("uv"):
        return "uv run python"
    for candidate in (".venv", "venv"):
        exe = root / candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if exe.exists():
            return exe.relative_to(root).as_posix()
    return "python"


@dataclass
class TestCounts:
    __test__ = False

    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None


_PYTEST_SUMMARY = re.compile(r"=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s", re.MULTILINE)
_PYTEST_PART = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")


def parse_test_output(framework: str, output: str) -> TestCounts:
    counts = TestCounts()
    if framework == "pytest" or ("passed" in output and " in " in output):
        summaries = _PYTEST_SUMMARY.findall(output)
        if summaries:
            for number, kind in _PYTEST_PART.findall(summaries[-1]):
                value = int(number)
                if kind == "passed":
                    counts.passed = value
                elif kind == "failed":
                    counts.failed = value
                elif kind.startswith("error"):
                    counts.errors = value
                elif kind == "skipped":
                    counts.skipped = value
            counts.passed = counts.passed or 0
            counts.failed = counts.failed or 0
            return counts
    jest = re.search(r"Tests:\s+(.*?)\s*(\d+) total", output)
    if jest:
        for number, kind in re.findall(r"(\d+) (passed|failed|skipped|todo)", jest.group(1)):
            setattr(counts, kind if kind != "todo" else "skipped", int(number))
        counts.passed = counts.passed or 0
        counts.failed = counts.failed or 0
        return counts
    cargo = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed", output)
    if cargo:
        counts.passed = sum(int(p) for p, _ in cargo)
        counts.failed = sum(int(f) for _, f in cargo)
        return counts
    if framework == "go":
        counts.passed = len(re.findall(r"^ok\s", output, re.MULTILINE))
        counts.failed = len(re.findall(r"^(?:FAIL|--- FAIL)", output, re.MULTILINE))
    return counts
