"""Phase 2 exit condition with real models (opt-in): FORGE_LIVE_TESTS=1 pytest -m live.

A real model gets the sample repository (failing test, planted bug) and a toolset, and
must find and fix the bug through the tool runtime with no scripting.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from forge.models import Message, ModelProfile, ModelRegistry, ProviderError
from forge.policies import StaticApprover

from ..conftest import PYTHON, RuntimeFactory
from .agent_loop import run_tool_loop

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("FORGE_LIVE_TESTS") != "1", reason="FORGE_LIVE_TESTS!=1"),
]

TOOLS = ["repository_structure", "search", "read_file", "edit_file", "run_tests", "list_directory"]
SYSTEM = (
    "You are a software engineer working in a repository through tools. Investigate before "
    "changing anything, make the smallest correct fix, and verify it by running the tests. "
    "When the tests pass, reply with a one-sentence summary and no tool calls."
)


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


CASES = {
    "ollama": ("ollama", os.environ.get("FORGE_LIVE_OLLAMA_MODEL", "qwen2.5:7b"), _ollama_up),
    "openrouter": (
        "openrouter",
        os.environ.get("FORGE_LIVE_OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        lambda: bool(os.environ.get("OPENROUTER_API_KEY")),
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            "ollama",
            marks=pytest.mark.xfail(
                strict=False,
                reason="qwen2.5:7b tends to edit the failing test instead of the code; "
                "catching that is adversarial QA's job (Phase 4+), not the tool runtime's",
            ),
        ),
        "openrouter",
    ],
)
async def test_live_model_fixes_bug(
    case: str, sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    provider, model, available = CASES[case]
    if not available():
        pytest.skip(f"{case} not available")
    runtime = make_runtime(
        sample_repo,
        approver=StaticApprover(False),
        tools={"test_command": f'"{PYTHON}" -m pytest -q -p no:cacheprovider'},
    )
    registry = ModelRegistry(
        profiles={case: ModelProfile(provider=provider, model=model, params={"temperature": 0})}
    )
    try:
        adapter = await registry.adapter("executor")
        try:
            outcome = await run_tool_loop(
                adapter,
                runtime,
                [Message.system(SYSTEM), Message.user("The test suite fails. Fix the bug.")],
                tool_names=TOOLS,
                max_steps=14,
            )
        except ProviderError as exc:
            if exc.status == 429:
                pytest.skip(f"rate limited: {exc}")
            raise
        print(f"\n{case}: {adapter.describe()} [{adapter.tool_strategy()} tools]")
        for result in outcome.results:
            print(f"  {'ok ' if result.ok else 'ERR'} {result.tool}: {result.content[:90]!r}")
        print(f"  final: {outcome.final_text[:200]!r}")
        total = registry.tracker.total
        print(f"  steps={outcome.steps} calls={total.calls} tokens={total.usage.total_tokens}")

        check = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=sample_repo,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stdout[-2000:]
        assert "super-secret" not in "".join(m.text for m in outcome.transcript)
    finally:
        await registry.aclose()
