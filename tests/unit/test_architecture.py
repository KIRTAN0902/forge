"""Enforces the spec's structural rules (§24) so they cannot silently erode.

Each rule is checked statically by parsing imports, so a violation fails the build
instead of relying on code review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "forge"

# package -> forge packages it must never import
FORBIDDEN: dict[str, set[str]] = {
    # The lowest layer depends on nothing else in Forge.
    "core": {
        "agents",
        "cli",
        "config",
        "context",
        "events",
        "models",
        "policies",
        "storage",
        "tools",
        "workflow",
    },
    # Rule 2 / 18: the engine is headless; only the CLI may import the CLI.
    **{
        pkg: {"cli"}
        for pkg in (
            "agents",
            "config",
            "context",
            "events",
            "models",
            "policies",
            "storage",
            "tools",
            "workflow",
        )
    },
    # Rule 1: agents use the model runtime's public API, never a concrete provider.
    "agents": {"cli", "models.providers"},
    # Policies are deterministic and must not depend on models.
    "policies": {"cli", "models", "agents", "workflow", "tools", "context"},
    # The model runtime knows nothing about tools, context or agents.
    "models": {"cli", "tools", "context", "agents", "workflow", "storage"},
    # Tools sit below context, agents and the workflow.
    "tools": {"cli", "context", "agents", "workflow"},
    # Context is built for agents, never from them.
    "context": {"cli", "agents", "workflow"},
}


def _forge_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        else:
            continue
        for name in names:
            if name == "forge" or name.startswith("forge."):
                found.add(name.removeprefix("forge").lstrip("."))
    return found


def _violations(package: str, forbidden: set[str]) -> list[str]:
    bad: list[str] = []
    for file in sorted((SRC / package).rglob("*.py")):
        for imported in _forge_imports(file):
            for target in forbidden:
                if imported == target or imported.startswith(target + "."):
                    bad.append(f"{file.relative_to(SRC)} imports forge.{imported}")
    return bad


@pytest.mark.parametrize(("package", "forbidden"), sorted(FORBIDDEN.items()))
def test_layering(package: str, forbidden: set[str]) -> None:
    assert _violations(package, forbidden) == []


def test_every_package_exists() -> None:
    expected = {
        "cli",
        "core",
        "agents",
        "models",
        "tools",
        "context",
        "workflow",
        "policies",
        "events",
        "storage",
    }
    present = {p.name for p in SRC.iterdir() if (p / "__init__.py").exists()}
    assert expected <= present
