"""User credentials and first-time API key onboarding (spec §19).

Stores credentials persistently in ~/.forge/credentials.json so the user
only needs to insert their API key once.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from forge.models.capabilities import ModelCapabilities
from forge.models.profile import ModelProfile

CREDENTIALS_DIR = Path.home() / ".forge"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

PROVIDER_METADATA: dict[str, dict[str, Any]] = {
    "openrouter": {
        "name": "OpenRouter",
        "env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.7-sonnet",
        "base_url": "https://openrouter.ai/api/v1",
        "label": "OpenRouter (Recommended - access Claude 3.7, Sonnet 3.5, GPT-4o, DeepSeek)",
    },
    "openai": {
        "name": "OpenAI",
        "env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "label": "OpenAI (GPT-4o, GPT-4o-mini)",
    },
    "ollama": {
        "name": "Ollama",
        "env": None,
        "default_model": "qwen2.5-coder:7b",
        "base_url": "http://localhost:11434/v1",
        "label": "Ollama (Local, free, private - runs locally on port 11434)",
    },
    "custom": {
        "name": "Custom",
        "env": "CUSTOM_API_KEY",
        "default_model": "default",
        "base_url": "http://localhost:8000/v1",
        "label": "Custom OpenAI-compatible endpoint (vLLM, LM Studio, Groq, Mistral)",
    },
}


def load_stored_credentials() -> dict[str, Any] | None:
    """Load credentials from ~/.forge/credentials.json if available."""
    if not CREDENTIALS_FILE.is_file():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "provider" in data:
            return data
    except Exception:
        return None
    return None


def save_stored_credentials(
    provider: str,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
) -> None:
    """Save credentials securely to ~/.forge/credentials.json."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "api_key": api_key,
        "model": model or PROVIDER_METADATA.get(provider, {}).get("default_model", ""),
        "base_url": base_url or PROVIDER_METADATA.get(provider, {}).get("base_url", ""),
    }
    CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        CREDENTIALS_FILE.chmod(0o600)


def init_credentials_env() -> bool:
    """Populate os.environ from ~/.forge/credentials.json if not already set.

    Returns True if an active API key or provider is available in the environment.
    """
    # 1. Check if any known environment variable is already set
    for meta in PROVIDER_METADATA.values():
        env_var = meta.get("env")
        if env_var and os.environ.get(env_var):
            return True

    # 2. Check stored credentials
    stored = load_stored_credentials()
    if not stored:
        return False

    provider = stored.get("provider", "openrouter")
    api_key = stored.get("api_key", "")
    model = stored.get("model", "")
    base_url = stored.get("base_url", "")

    meta = PROVIDER_METADATA.get(provider, {})
    env_var = meta.get("env")
    if env_var and api_key and not os.environ.get(env_var):
        os.environ[env_var] = api_key

    if model and not os.environ.get("FORGE_DEFAULT_MODEL"):
        os.environ["FORGE_DEFAULT_MODEL"] = model

    if base_url and not os.environ.get("FORGE_DEFAULT_BASE_URL"):
        os.environ["FORGE_DEFAULT_BASE_URL"] = base_url

    if provider == "ollama":
        return True

    return bool(api_key or (env_var and os.environ.get(env_var)))


def resolve_default_model_profile() -> tuple[str, ModelProfile | None]:
    """Resolve a default ModelProfile based on the active environment/credentials."""
    # Check openrouter
    if os.environ.get("OPENROUTER_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "anthropic/claude-3.7-sonnet")
        return "openrouter_default", ModelProfile(
            provider="openrouter",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    # Check openai
    if os.environ.get("OPENAI_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "gpt-4o")
        return "openai_default", ModelProfile(
            provider="openai",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    # Check stored credentials for ollama or custom
    stored = load_stored_credentials()
    if stored and stored.get("provider") == "ollama":
        model_name = stored.get("model") or "qwen2.5-coder:7b"
        return "ollama_default", ModelProfile(
            provider="ollama",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    return "", None


def prompt_for_credentials(console: Console) -> dict[str, Any]:
    """Interactively prompt user for provider and API key on first launch."""
    banner = (
        "[bold cyan]⚡ Welcome to Forge Setup[/bold cyan]\n"
        "[dim]No API key detected. Let's configure your model provider (one-time setup).[/dim]\n\n"
        "[bold]Select a provider:[/bold]\n"
        "  [bold cyan]1[/bold cyan] OpenRouter (Recommended - Claude 3.7, GPT-4o, DeepSeek)\n"
        "  [bold cyan]2[/bold cyan] OpenAI (GPT-4o, GPT-4o-mini)\n"
        "  [bold cyan]3[/bold cyan] Ollama (Local, free, private - runs locally on port 11434)\n"
        "  [bold cyan]4[/bold cyan] Custom OpenAI-compatible endpoint"
    )
    console.print(Panel(banner, border_style="cyan", title="Forge Initial Setup"))

    choice = Prompt.ask(
        "[bold cyan]Enter choice[/bold cyan]",
        choices=["1", "2", "3", "4"],
        default="1",
        console=console,
    )

    mapping = {
        "1": "openrouter",
        "2": "openai",
        "3": "ollama",
        "4": "custom",
    }
    provider = mapping[choice]
    meta = PROVIDER_METADATA[provider]

    api_key = ""
    base_url = meta.get("base_url", "")
    default_model = meta.get("default_model", "")

    if provider in {"openrouter", "openai", "custom"}:
        console.print()
        console.print(
            f"[bold yellow]Paste your {meta['name']} API key[/bold yellow] "
            "[dim](input is hidden for security):[/dim]"
        )
        api_key = Prompt.ask("API Key", password=True, console=console).strip()
        while not api_key:
            console.print("[red]API key cannot be empty. Please enter a valid key:[/red]")
            api_key = Prompt.ask("API Key", password=True, console=console).strip()

    if provider == "custom":
        console.print()
        base_url = Prompt.ask(
            "Base URL", default="http://localhost:8000/v1", console=console
        ).strip()

    console.print()
    model = Prompt.ask(
        "Default Model",
        default=default_model,
        console=console,
    ).strip()

    save_stored_credentials(provider, api_key, model=model, base_url=base_url)
    init_credentials_env()

    console.print()
    console.print(
        f"[bold green]✔ Saved credentials to {CREDENTIALS_FILE}[/bold green]\n"
        "[dim]Stored securely. You will not be asked again on subsequent runs.[/dim]\n"
    )

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
    }


def ensure_credentials_configured(console: Console, force_prompt: bool = False) -> None:
    """Ensure an API key / credentials are ready, prompting if missing."""
    if not force_prompt and init_credentials_env():
        return
    prompt_for_credentials(console)
