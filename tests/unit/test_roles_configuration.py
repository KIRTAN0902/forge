"""Unit tests for multi-model, multi-provider agent role configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from forge.config.loader import load_config
from forge.config.roles import (
    get_roles_summary,
    save_multi_role_preset,
    save_single_model_preset,
    set_role_model,
)
from forge.models.registry import ModelRegistry


class TestRolesConfiguration:
    def test_single_model_preset(self, tmp_path: Path) -> None:
        """Test configuring one model and one provider for all agents."""
        save_single_model_preset(
            tmp_path,
            provider="nvidia",
            model="nvidia/nemotron-3-super-120b-a12b",
            profile_name="nemotron_shared",
        )

        config = load_config(tmp_path)
        assert "nemotron_shared" in config.models
        assert config.models["nemotron_shared"].provider == "nvidia"
        assert config.models["nemotron_shared"].model == "nvidia/nemotron-3-super-120b-a12b"

        # Verify all roles mapped to nemotron_shared
        assert config.roles["planner"].model == "nemotron_shared"
        assert config.roles["executor"].model == "nemotron_shared"
        assert config.roles["qa"].model == "nemotron_shared"
        assert config.roles["orchestrator"].model == "nemotron_shared"

    def test_same_provider_different_models(self, tmp_path: Path) -> None:
        """Test different models for different agents sharing the same provider/API key."""
        role_configs = {
            "planner": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            "executor": ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            "qa": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            "orchestrator": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
        }
        save_multi_role_preset(tmp_path, role_configs)

        config = load_config(tmp_path)
        assert config.roles["planner"].model == "planner_model"
        assert config.roles["executor"].model == "executor_model"
        assert config.models["planner_model"].model == "nvidia/nemotron-3-super-120b-a12b"
        assert config.models["executor_model"].model == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert config.models["planner_model"].provider == "nvidia"
        assert config.models["executor_model"].provider == "nvidia"

    def test_multi_provider_different_keys(self, tmp_path: Path) -> None:
        """Test different models and different providers for different agents."""
        role_configs = {
            "planner": ("openrouter", "anthropic/claude-3.7-sonnet"),
            "executor": ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            "qa": ("openai", "gpt-4o"),
            "orchestrator": ("openrouter", "anthropic/claude-3.7-sonnet"),
        }
        save_multi_role_preset(tmp_path, role_configs)

        config = load_config(tmp_path)
        assert config.models["planner_model"].provider == "openrouter"
        assert config.models["executor_model"].provider == "nvidia"
        assert config.models["qa_model"].provider == "openai"

        # Verify summary reflects the multi-provider setup
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "NVIDIA_API_KEY": "nvapi-test",
                "OPENAI_API_KEY": "sk-test",
            },
        ):
            summaries = {s.role: s for s in get_roles_summary(config)}
            assert summaries["planner"].provider == "openrouter"
            assert summaries["planner"].has_key is True
            assert summaries["executor"].provider == "nvidia"
            assert summaries["executor"].has_key is True
            assert summaries["qa"].provider == "openai"
            assert summaries["qa"].has_key is True

    def test_set_role_model_helper(self, tmp_path: Path) -> None:
        """Test dynamically assigning an individual role model."""
        save_single_model_preset(
            tmp_path, provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b"
        )
        set_role_model(
            tmp_path,
            role="executor",
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            provider="nvidia",
        )

        config = load_config(tmp_path)
        assert config.roles["planner"].model == "default_model"
        assert config.roles["executor"].model == "executor_model"
        assert config.models["executor_model"].model == "nvidia/nemotron-3.5-lightning-30b-a3b"

    async def test_registry_builds_separate_adapters(self, tmp_path: Path) -> None:
        """Verify ModelRegistry creates distinct adapters for different roles."""
        role_configs = {
            "planner": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
            "executor": ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            "qa": ("openai", "gpt-4o"),
        }
        save_multi_role_preset(tmp_path, role_configs)
        config = load_config(tmp_path)

        with patch.dict(
            os.environ,
            {"NVIDIA_API_KEY": "nvapi-test", "OPENAI_API_KEY": "sk-test"},
        ):
            registry = ModelRegistry(
                providers=config.providers,
                profiles=config.models,
                roles=config.roles,
            )
            try:
                planner_adapter = await registry.adapter("planner")
                executor_adapter = await registry.adapter("executor")
                qa_adapter = await registry.adapter("qa")

                assert planner_adapter.profile.model == "nvidia/nemotron-3-super-120b-a12b"
                assert executor_adapter.profile.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
                assert qa_adapter.profile.model == "gpt-4o"
                assert qa_adapter.provider.name == "openai"
                assert planner_adapter.provider.name == "nvidia"
            finally:
                await registry.aclose()

    def test_run_model_menu_exit(self, tmp_path: Path) -> None:
        """Test model menu prompt cleanly returns on 0."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_model_menu

        save_single_model_preset(tmp_path, provider="nvidia", model="test-model")
        console = Console(record=True)

        with patch("rich.prompt.Prompt.ask", return_value="0"):
            run_model_menu(tmp_path, console)

    def test_configure_single_role(self, tmp_path: Path) -> None:
        """Test configure_single_role updates orchestrator and planner models via dropdown."""
        from rich.console import Console

        from forge.cli.roles_wizard import configure_single_role

        save_single_model_preset(tmp_path, provider="nvidia", model="base-model")
        console = Console(record=True)

        with (
            patch("rich.prompt.Prompt.ask", side_effect=["1", "c", "nemotron-custom", ""]),
            patch("rich.prompt.Confirm.ask", return_value=True),
        ):
            configure_single_role(tmp_path, "orchestrator", console)

        config = load_config(tmp_path)
        assert config.roles["orchestrator"].model == "orchestrator_model"
        assert config.models["orchestrator_model"].model == "nemotron-custom"

    def test_set_provider_config_and_menu(self, tmp_path: Path) -> None:
        """Test set_provider_config helper and run_provider_menu."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_provider_menu
        from forge.config.roles import set_provider_config

        set_provider_config(
            tmp_path,
            provider="custom",
            base_url="http://custom-host:8080/v1",
            api_key_env="CUSTOM_API_KEY",
        )
        config = load_config(tmp_path)
        assert "custom" in config.providers
        assert config.providers["custom"].base_url == "http://custom-host:8080/v1"

        console = Console(record=True)
        with patch("rich.prompt.Prompt.ask", return_value="0"):
            changes = run_provider_menu(tmp_path, console)
            assert changes == []

    def test_dynamic_recent_models_tracking(self, tmp_path: Path) -> None:
        """Test that recently used models are tracked dynamically and in MRU order."""
        from forge.cli.credentials import get_recent_models, record_recent_model

        record_recent_model("model-alpha")
        record_recent_model("model-beta")
        record_recent_model("model-alpha")  # Should move to front

        recents = get_recent_models()
        assert recents[0] == "model-alpha"
        assert "model-beta" in recents

    def test_provider_modal_connected_available(self, tmp_path: Path) -> None:
        """Test that run_provider_modal shows only connected and available providers."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_provider_modal
        from forge.cli.tui_modal import TuiItem

        console = Console(record=True)
        captured_items: list[TuiItem] = []

        def mock_modal(title: str, items: list[TuiItem], **kwargs: Any) -> Any:
            captured_items.extend(items)
            return None  # cancel

        with patch("forge.cli.tui_modal.run_tui_modal", side_effect=mock_modal):
            changes = run_provider_modal(tmp_path, console)
            assert changes == []

        sections = {it.section for it in captured_items}
        # Sections should only be Connected Providers and/or Available Providers
        assert sections.issubset({"Connected Providers", "Available Providers"})
        prov_ids = {it.id for it in captured_items}
        assert "anthropic" in prov_ids
        assert "openrouter" in prov_ids
        assert "nvidia" in prov_ids
        assert "openai" in prov_ids
        assert "ollama" in prov_ids
        assert "custom" in prov_ids

    def test_provider_modal_connected_provider_only_asks_api_key(self, tmp_path: Path) -> None:
        """Connected provider does NOT ask for URL, only asks for API key."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_provider_modal
        from forge.cli.tui_modal import TuiItem

        console = Console(record=True)
        chosen_item = TuiItem(
            id="openai",
            title="OpenAI",
            subtitle="(https://api.openai.com/v1)",
            section="Connected Providers",
            badge="Connected",
        )

        with (
            patch("forge.cli.tui_modal.run_tui_modal", return_value=chosen_item),
            patch("forge.cli.roles_wizard.get_provider_api_key", return_value="sk-test-existing-12345"),
            patch("forge.cli.roles_wizard.Prompt.ask") as mock_ask,
            patch("forge.cli.roles_wizard.save_stored_credentials") as mock_save,
            patch("forge.config.roles.set_provider_config"),
        ):
            # User enters a new API key without being asked for URL
            mock_ask.return_value = "sk-new-key-67890"
            changes = run_provider_modal(tmp_path, console)
            assert len(changes) == 1
            assert "OpenAI" in changes[0]
            # Prompt.ask should have been called ONCE for API Key only, NEVER for Base URI / URL
            assert mock_ask.call_count == 1
            prompt_arg = mock_ask.call_args[0][0]
            assert "Base URI" not in prompt_arg
            assert "API Key" in prompt_arg
            # Verify URL used was preconfigured OpenAI endpoint
            mock_save.assert_called_once_with(
                "openai", api_key="sk-new-key-67890", base_url="https://api.openai.com/v1"
            )

    def test_provider_modal_custom_asks_both_url_and_key(self, tmp_path: Path) -> None:
        """Custom provider asks for BOTH URL and API key."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_provider_modal
        from forge.cli.tui_modal import TuiItem

        console = Console(record=True)
        chosen_item = TuiItem(
            id="custom",
            title="Custom Endpoint",
            subtitle="(http://localhost:8000/v1)",
            section="Available Providers",
            badge="",
        )

        with (
            patch("forge.cli.tui_modal.run_tui_modal", return_value=chosen_item),
            patch("forge.cli.roles_wizard.get_provider_api_key", return_value=None),
            patch("forge.cli.roles_wizard.Prompt.ask") as mock_ask,
            patch("forge.cli.roles_wizard.save_stored_credentials") as mock_save,
            patch("forge.config.roles.set_provider_config"),
        ):
            # Prompt 1: URL, Prompt 2: API key
            mock_ask.side_effect = ["http://my-custom-vllm:8000/v1", "my-secret-key"]
            changes = run_provider_modal(tmp_path, console)
            assert len(changes) == 1
            assert mock_ask.call_count == 2
            first_prompt = mock_ask.call_args_list[0][0][0]
            second_prompt = mock_ask.call_args_list[1][0][0]
            assert "Base URI" in first_prompt
            assert "API Key" in second_prompt
            mock_save.assert_called_once_with(
                "custom", api_key="my-secret-key", base_url="http://my-custom-vllm:8000/v1"
            )

    def test_provider_modal_ollama_does_not_ask_url_or_key(self, tmp_path: Path) -> None:
        """Ollama is pre-configured with URL and does not ask for URL or key."""
        from rich.console import Console

        from forge.cli.roles_wizard import run_provider_modal
        from forge.cli.tui_modal import TuiItem

        console = Console(record=True)
        chosen_item = TuiItem(
            id="ollama",
            title="Ollama (Local)",
            subtitle="(http://localhost:11434/v1)",
            section="Available Providers",
            badge="",
        )

        with (
            patch("forge.cli.tui_modal.run_tui_modal", return_value=chosen_item),
            patch("forge.cli.roles_wizard.Prompt.ask") as mock_ask,
            patch("forge.cli.roles_wizard.save_stored_credentials") as mock_save,
            patch("forge.config.roles.set_provider_config"),
        ):
            changes = run_provider_modal(tmp_path, console)
            assert len(changes) == 1
            assert "Ollama" in changes[0]
            # Prompt.ask should NEVER be called for Ollama
            assert mock_ask.call_count == 0
            mock_save.assert_called_once_with(
                "ollama", api_key="", base_url="http://localhost:11434/v1"
            )


