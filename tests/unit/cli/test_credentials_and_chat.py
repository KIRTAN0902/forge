"""Unit tests for credentials management and interactive chat interface."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from forge.cli.app import app
from forge.cli.chat import print_banner, print_help, read_user_input, run_chat_session
from forge.cli.credentials import (
    ensure_credentials_configured,
    init_credentials_env,
    load_stored_credentials,
    prompt_for_credentials,
    resolve_default_model_profile,
    save_stored_credentials,
)


class TestCredentialsManager:
    def test_save_and_load_credentials(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        with (
            patch("forge.cli.credentials.CREDENTIALS_DIR", tmp_path),
            patch("forge.cli.credentials.CREDENTIALS_FILE", creds_file),
        ):
            # Initially empty
            assert load_stored_credentials() is None

            # Save credentials
            save_stored_credentials(
                provider="openrouter",
                api_key="sk-or-test-key-123",
                model="anthropic/claude-3.7-sonnet",
            )
            assert creds_file.is_file()

            # Load credentials
            loaded = load_stored_credentials()
            assert loaded is not None
            assert loaded["provider"] == "openrouter"
            assert loaded["api_key"] == "sk-or-test-key-123"
            assert loaded["model"] == "anthropic/claude-3.7-sonnet"

    def test_init_credentials_env(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        with (
            patch("forge.cli.credentials.CREDENTIALS_DIR", tmp_path),
            patch("forge.cli.credentials.CREDENTIALS_FILE", creds_file),
            patch.dict(os.environ, {}, clear=True),
        ):
            save_stored_credentials(
                provider="openrouter",
                api_key="sk-or-env-test",
                model="test-model",
            )
            assert "OPENROUTER_API_KEY" not in os.environ
            active = init_credentials_env()
            assert active is True
            assert os.environ["OPENROUTER_API_KEY"] == "sk-or-env-test"
            assert os.environ["FORGE_DEFAULT_MODEL"] == "test-model"

    def test_resolve_default_model_profile(self) -> None:
        with (
            patch("forge.cli.credentials.load_stored_credentials", return_value=None),
            patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}, clear=True),
        ):
            name, profile = resolve_default_model_profile()
            assert name == "nvidia_default"
            assert profile is not None
            assert profile.provider == "nvidia"

        with (
            patch("forge.cli.credentials.load_stored_credentials", return_value=None),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}, clear=True),
        ):
            name, profile = resolve_default_model_profile()
            assert name == "openrouter_default"
            assert profile is not None
            assert profile.provider == "openrouter"

        with (
            patch("forge.cli.credentials.load_stored_credentials", return_value=None),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
        ):
            name, profile = resolve_default_model_profile()
            assert name == "openai_default"
            assert profile is not None
            assert profile.provider == "openai"

        # Stored credentials take priority
        with (
            patch(
                "forge.cli.credentials.load_stored_credentials",
                return_value={"provider": "nvidia", "api_key": "nvapi-key", "model": "test-m"},
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
        ):
            name, profile = resolve_default_model_profile()
            assert name == "nvidia_default"
            assert profile is not None
            assert profile.model == "test-m"

    def test_prompt_for_credentials(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        console = Console(record=True)

        with (
            patch("forge.cli.credentials.CREDENTIALS_DIR", tmp_path),
            patch("forge.cli.credentials.CREDENTIALS_FILE", creds_file),
            patch("forge.cli.credentials.Prompt.ask") as mock_ask,
            patch.dict(os.environ, {}, clear=True),
        ):
            # 1. choice="1" (nvidia), 2. api_key="nvapi-test"
            # 3. model="nvidia/nemotron-3-super-120b-a12b"
            mock_ask.side_effect = [
                "1",
                "nvapi-test-prompt",
                "nvidia/nemotron-3-super-120b-a12b",
            ]
            result = prompt_for_credentials(console)

            assert result["provider"] == "nvidia"
            assert result["api_key"] == "nvapi-test-prompt"
            assert load_stored_credentials() is not None

    def test_ensure_credentials_configured_skips_when_present(self) -> None:
        console = Console()
        with (
            patch("forge.cli.credentials.init_credentials_env", return_value=True),
            patch("forge.cli.credentials.prompt_for_credentials") as mock_prompt,
        ):
            ensure_credentials_configured(console, force_prompt=False)
            mock_prompt.assert_not_called()


class TestChatInterface:
    def test_print_banner_and_help(self, tmp_path: Path) -> None:
        console = Console(record=True)
        print_banner(console, tmp_path)
        output = console.export_text()
        assert "Interactive Agent Session" in output
        assert tmp_path.name in output

        console_help = Console(record=True)
        print_help(console_help)
        help_output = console_help.export_text()
        assert "/help" in help_output
        assert "/diff" in help_output
        assert "/status" in help_output
        assert "/rollback" in help_output

    def test_read_user_input_fallback(self) -> None:
        console = Console()
        with patch("rich.prompt.Prompt.ask", return_value="hello forge"):
            val = read_user_input(session=None, console=console)
            assert val == "hello forge"

    def test_run_chat_session_slash_commands(self, tmp_path: Path) -> None:
        # Mock inputs: /help, /models, /provider, /exit
        inputs = iter(["/help", "/models", "/provider", "/exit"])

        with (
            patch("forge.cli.chat.ensure_credentials_configured"),
            patch(
                "forge.cli.roles_wizard.run_model_menu", return_value=["Planner ➔ claude"]
            ) as mock_menu,
            patch(
                "forge.cli.roles_wizard.run_provider_menu", return_value=["Provider: NVIDIA"]
            ) as mock_prov,
            patch("forge.cli.chat.read_user_input", side_effect=lambda *args: next(inputs)),
        ):
            # Should run the commands and exit gracefully
            run_chat_session(path=tmp_path)
            assert (tmp_path / ".forge").is_dir()
            mock_menu.assert_called_once()
            mock_prov.assert_called_once()

    def test_cli_default_launches_split_tui(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("forge.ui.split_app.ForgeSplitPaneApp.run") as mock_split:
            result = runner.invoke(app, ["--path", str(tmp_path)])
            assert result.exit_code == 0
            mock_split.assert_called_once()

    def test_cli_classic_flag_launches_chat(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("forge.cli.chat.run_chat_session") as mock_chat:
            result = runner.invoke(app, ["--classic", "--path", str(tmp_path)])
            assert result.exit_code == 0
            mock_chat.assert_called_once()

    def test_cli_chat_command(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("forge.cli.chat.run_chat_session") as mock_chat:
            result = runner.invoke(app, ["chat", "--path", str(tmp_path)])
            assert result.exit_code == 0
            mock_chat.assert_called_once()

    def test_cli_login_command(self) -> None:
        runner = CliRunner()
        with patch("forge.cli.credentials.prompt_for_credentials") as mock_prompt:
            result = runner.invoke(app, ["login"])
            assert result.exit_code == 0
            mock_prompt.assert_called_once()

    def test_cli_provider_command(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch(
            "forge.cli.roles_wizard.run_provider_menu", return_value=["Provider: OpenRouter"]
        ) as mock_prov:
            result = runner.invoke(app, ["provider", "--path", str(tmp_path)])
            assert result.exit_code == 0
            mock_prov.assert_called_once()

    def test_chat_session_state_cycling_and_aliases(self) -> None:
        """Test default agent is orchestrator and cycling goes orchestrator -> executor -> qa."""
        from forge.cli.chat import ChatSessionState, _get_prompt_tokens

        state = ChatSessionState()
        assert state.active_agent == "orchestrator"

        # Tab cycle forward
        assert state.cycle_agent() == "executor"
        assert state.cycle_agent() == "qa"
        assert state.cycle_agent() == "orchestrator"

        # Shift+Tab cycle reverse
        assert state.cycle_agent_reverse() == "qa"
        assert state.cycle_agent_reverse() == "executor"

        # Aliases
        assert state.set_agent("anvil") is True
        assert state.active_agent == "orchestrator"
        assert state.set_agent("hammer") is True
        assert state.active_agent == "executor"
        assert state.set_agent("temper") is True
        assert state.active_agent == "qa"
        assert state.set_agent("plan") is True
        assert state.active_agent == "orchestrator"
        assert state.set_agent("planner") is True
        assert state.active_agent == "orchestrator"
        assert state.set_agent("exec") is True
        assert state.active_agent == "executor"
        assert state.set_agent("orch") is True
        assert state.active_agent == "orchestrator"
        assert state.set_agent("qa") is True
        assert state.active_agent == "qa"

        tokens = _get_prompt_tokens(state)
        token_text = "".join(t[1] for t in tokens)
        assert "TEMPER" in token_text

    def test_run_chat_session_switch_agents(self, tmp_path: Path) -> None:
        """Test switching agents via slash commands /anvil, /hammer, /temper, /planner, etc."""
        from forge.cli.chat import run_chat_session

        inputs = iter(
            [
                "/anvil",
                "/hammer",
                "/temper",
                "/planner",
                "/executor",
                "/qa",
                "/orch",
                "/agent",
                "/exit",
            ]
        )

        with (
            patch("forge.cli.chat.ensure_credentials_configured"),
            patch("forge.cli.chat.read_user_input", side_effect=lambda *args: next(inputs)),
        ):
            run_chat_session(path=tmp_path)
            assert (tmp_path / ".forge").is_dir()
