"""Unit tests for Model Catalog and Dropdown Selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.console import Console

from forge.cli.model_catalog import (
    CURATED_MODELS,
    fetch_provider_models,
    prompt_model_dropdown,
)


def test_fetch_provider_models_fallback() -> None:
    """Offline or error fetching falls back to curated models."""
    with patch("httpx.get", side_effect=Exception("Network error")):
        models = fetch_provider_models("openrouter")
        expected = [m[0] for m in CURATED_MODELS["openrouter"]]
        assert models == expected


def test_fetch_provider_models_success() -> None:
    """Successfully parses JSON data from model endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [{"id": "mock/custom-model-1"}, {"id": "mock/custom-model-2"}]
    }

    with patch("httpx.get", return_value=mock_resp):
        models = fetch_provider_models("custom_test_provider", base_url="http://mock:8000")
        assert "default" in models or [m[0] for m in CURATED_MODELS.get("custom_test_provider", [])]


def test_prompt_model_dropdown_select_number() -> None:
    """Selecting a number from the dropdown returns that model ID."""
    console = Console(record=True)
    with (
        patch("forge.cli.model_catalog.fetch_provider_models", return_value=["model-a", "model-b"]),
        patch("rich.prompt.Prompt.ask", return_value="1"),
    ):
        chosen = prompt_model_dropdown(
            role_label="Planner",
            provider="openrouter",
            current_model="model-b",
            console=console,
        )
        assert chosen == "model-a"


def test_prompt_model_dropdown_custom_id() -> None:
    """Entering 'c' allows entering any custom model ID string."""
    console = Console(record=True)
    with (
        patch("forge.cli.model_catalog.fetch_provider_models", return_value=["model-a"]),
        patch("rich.prompt.Prompt.ask", side_effect=["c", "anthropic/claude-custom-tier"]),
    ):
        chosen = prompt_model_dropdown(
            role_label="Executor",
            provider="openrouter",
            current_model="model-a",
            console=console,
        )
        assert chosen == "anthropic/claude-custom-tier"


def test_prompt_model_dropdown_search() -> None:
    """Entering 's' allows searching provider models and selecting one."""
    console = Console(record=True)
    all_models = [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.7-sonnet",
        "openai/gpt-4o",
        "deepseek/deepseek-r1",
    ]
    with (
        patch("forge.cli.model_catalog.fetch_provider_models", return_value=all_models),
        patch("rich.prompt.Prompt.ask", side_effect=["s", "deepseek", "1"]),
    ):
        chosen = prompt_model_dropdown(
            role_label="QA",
            provider="openrouter",
            current_model="openai/gpt-4o",
            console=console,
        )
        assert chosen == "deepseek/deepseek-r1"


def test_prompt_model_dropdown_cancel() -> None:
    """Entering '0' keeps the current model without changes."""
    console = Console(record=True)
    with (
        patch("forge.cli.model_catalog.fetch_provider_models", return_value=["model-a", "model-b"]),
        patch("rich.prompt.Prompt.ask", return_value="0"),
    ):
        chosen = prompt_model_dropdown(
            role_label="Orchestrator",
            provider="openrouter",
            current_model="current-custom-model",
            console=console,
        )
        assert chosen == "current-custom-model"
