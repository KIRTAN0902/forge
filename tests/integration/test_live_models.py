"""Phase 1 exit condition against real models (opt-in).

Runs the same ``solve`` agent from ``test_model_swap`` against real endpoints:

    FORGE_LIVE_TESTS=1 uv run pytest -m live -v

* ``FORGE_LIVE_OLLAMA_MODEL`` (default ``qwen2.5:7b``) on a local Ollama server, once
  with native capabilities (auto-detected) and once forced onto the prompt protocols.
* ``FORGE_LIVE_OPENROUTER_MODEL`` (default a free model) when ``OPENROUTER_API_KEY`` is set.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from forge.models import ModelProfile, ModelRegistry, OptimizationConfig, ProviderError

from .test_model_swap import solve

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("FORGE_LIVE_TESTS") != "1", reason="FORGE_LIVE_TESTS!=1"),
]

OLLAMA_MODEL = os.environ.get("FORGE_LIVE_OLLAMA_MODEL", "qwen2.5:7b")
OPENROUTER_MODEL = os.environ.get(
    "FORGE_LIVE_OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)


def _ollama_available() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


CASES: dict[str, dict[str, Any]] = {
    "ollama-native": {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "optimization": OptimizationConfig(),
        "needs": _ollama_available,
    },
    "ollama-prompt-protocol": {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "optimization": OptimizationConfig(
            tool_call_strategy="prompt", structured_output_strategy="prompt"
        ),
        "needs": _ollama_available,
    },
    "openrouter": {
        "provider": "openrouter",
        "model": OPENROUTER_MODEL,
        "optimization": OptimizationConfig(),
        "needs": lambda: bool(os.environ.get("OPENROUTER_API_KEY")),
    },
}


@pytest.mark.parametrize("case", list(CASES))
async def test_live_model_swap(case: str) -> None:
    spec = CASES[case]
    if not spec["needs"]():
        pytest.skip(f"{case}: endpoint not available")
    registry = ModelRegistry(
        profiles={
            case: ModelProfile(
                provider=spec["provider"],
                model=spec["model"],
                optimization=spec["optimization"],
                params={"temperature": 0},
            )
        }
    )
    try:
        adapter = await registry.adapter("executor")
        print(f"\n{case}: {adapter.describe()}")
        print(f"  tool strategy: {adapter.tool_strategy()}")
        print(f"  structured strategy: {adapter.structured_strategy()}")
        try:
            answer = await solve(adapter)
        except ProviderError as exc:
            if exc.status == 429:
                pytest.skip(f"{case}: rate limited ({exc})")
            raise
        total = registry.tracker.total
        print(f"  answer: {answer!r}")
        print(
            f"  calls={total.calls} in={total.usage.input_tokens} "
            f"out={total.usage.output_tokens} latency={total.latency_s:.1f}s"
        )
        assert answer.result == 42
    finally:
        await registry.aclose()
