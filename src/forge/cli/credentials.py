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
RECENT_MODELS_FILE = CREDENTIALS_DIR / "recent_models.json"

PROVIDER_METADATA: dict[str, dict[str, Any]] = {
    "anthropic": {
        "name": "Anthropic",
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-7-sonnet-20250219",
        "base_url": "https://api.anthropic.com/v1",
        "label": "Anthropic (Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3.5 Haiku)",
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "env": "NVIDIA_API_KEY",
        "default_model": "nvidia/nemotron-3-super-120b-a12b",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "label": "NVIDIA NIM (Nemotron 3 Super 120B, Nemotron 3.5 Lightning, Mistral)",
    },
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


def get_provider_api_key(provider: str) -> str | None:
    """Return the API key for a given provider from environment or stored credentials."""
    meta = PROVIDER_METADATA.get(provider, {})
    env_var = meta.get("env")
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    stored = load_stored_credentials()
    if stored:
        providers = stored.get("providers", {})
        if isinstance(providers, dict) and provider in providers:
            prov_data = providers[provider]
            if isinstance(prov_data, dict) and prov_data.get("api_key"):
                return str(prov_data["api_key"])

        if stored.get("provider") == provider and stored.get("api_key"):
            return str(stored["api_key"])

    return None


def save_stored_credentials(
    provider: str,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
) -> None:
    """Save credentials securely to ~/.forge/credentials.json, supporting multi-provider storage."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_stored_credentials() or {}
    providers = existing.get("providers", {})
    if not isinstance(providers, dict):
        providers = {}

    default_model = model or PROVIDER_METADATA.get(provider, {}).get("default_model", "")
    resolved_base_url = base_url or PROVIDER_METADATA.get(provider, {}).get("base_url", "")

    providers[provider] = {
        "api_key": api_key,
        "default_model": default_model,
        "base_url": resolved_base_url,
    }

    payload: dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
        "model": default_model,
        "base_url": resolved_base_url,
        "providers": providers,
    }
    if "roles" in existing:
        payload["roles"] = existing["roles"]

    CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        CREDENTIALS_FILE.chmod(0o600)


def get_recent_models() -> list[str]:
    """Return the list of recently used/selected models in MRU order."""
    if not RECENT_MODELS_FILE.is_file():
        stored = load_stored_credentials()
        if stored and isinstance(stored.get("recent_models"), list):
            return [str(m) for m in stored["recent_models"] if isinstance(m, str)]
        return []
    try:
        data = json.loads(RECENT_MODELS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(m) for m in data if m and isinstance(m, str)]
    except Exception:
        return []
    return []


def record_recent_model(model: str) -> None:
    """Record a model in the user's recent models history (MRU, max 10)."""
    if not model or not model.strip():
        return
    clean_model = model.strip()
    recents = get_recent_models()
    if clean_model in recents:
        recents.remove(clean_model)
    recents.insert(0, clean_model)
    recents = recents[:10]
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        RECENT_MODELS_FILE.write_text(json.dumps(recents, indent=2), encoding="utf-8")
        stored = load_stored_credentials() or {}
        stored["recent_models"] = recents
        if CREDENTIALS_FILE.is_file():
            CREDENTIALS_FILE.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    except Exception:
        pass


def init_credentials_env() -> bool:
    """Populate os.environ from ~/.forge/credentials.json if configured.

    Exports environment variables for all configured providers.
    Returns True if an active API key or provider is available.
    """
    has_credentials = False
    stored = load_stored_credentials()
    if stored:
        # 1. Populate all stored providers
        providers = stored.get("providers", {})
        if isinstance(providers, dict):
            for prov_name, prov_data in providers.items():
                if isinstance(prov_data, dict):
                    k = prov_data.get("api_key")
                    meta = PROVIDER_METADATA.get(prov_name, {})
                    env_var = meta.get("env")
                    if env_var and k and not os.environ.get(env_var):
                        os.environ[env_var] = str(k)
                        has_credentials = True
                    elif env_var and os.environ.get(env_var):
                        has_credentials = True

        # 2. Populate top-level active/default provider
        provider = stored.get("provider", "nvidia")
        api_key = stored.get("api_key", "")
        model = stored.get("model", "")
        base_url = stored.get("base_url", "")

        meta = PROVIDER_METADATA.get(provider, {})
        env_var = meta.get("env")
        if env_var and api_key and not os.environ.get(env_var):
            os.environ[env_var] = api_key
            has_credentials = True
        elif env_var and os.environ.get(env_var):
            has_credentials = True

        if model and not os.environ.get("FORGE_DEFAULT_MODEL"):
            os.environ["FORGE_DEFAULT_MODEL"] = model

        if base_url and not os.environ.get("FORGE_DEFAULT_BASE_URL"):
            os.environ["FORGE_DEFAULT_BASE_URL"] = base_url

        if provider == "ollama":
            has_credentials = True

    # 3. Check if any known environment variable is already set
    for meta in PROVIDER_METADATA.values():
        env_var = meta.get("env")
        if env_var and os.environ.get(env_var):
            has_credentials = True

    return has_credentials


def resolve_default_model_profile() -> tuple[str, ModelProfile | None]:
    """Resolve a default ModelProfile based on the active environment/credentials."""
    stored = load_stored_credentials()
    if stored:
        provider = stored.get("provider")
        model_name = stored.get("model") or PROVIDER_METADATA.get(str(provider), {}).get(
            "default_model", ""
        )
        if provider == "nvidia" and (os.environ.get("NVIDIA_API_KEY") or stored.get("api_key")):
            return "nvidia_default", ModelProfile(
                provider="nvidia",
                model=model_name or "nvidia/nemotron-3-super-120b-a12b",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )
        if provider == "anthropic" and (
            os.environ.get("ANTHROPIC_API_KEY") or stored.get("api_key")
        ):
            return "anthropic_default", ModelProfile(
                provider="anthropic",
                model=model_name or "claude-3-7-sonnet-20250219",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
                detect_capabilities=False,
            )
        if provider == "openrouter" and (
            os.environ.get("OPENROUTER_API_KEY") or stored.get("api_key")
        ):
            return "openrouter_default", ModelProfile(
                provider="openrouter",
                model=model_name or "anthropic/claude-3.7-sonnet",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
                detect_capabilities=False,
            )
        if provider == "openai" and (os.environ.get("OPENAI_API_KEY") or stored.get("api_key")):
            return "openai_default", ModelProfile(
                provider="openai",
                model=model_name or "gpt-4o",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
                detect_capabilities=False,
            )
        if provider == "ollama":
            return "ollama_default", ModelProfile(
                provider="ollama",
                model=model_name or "qwen2.5-coder:7b",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )
        if provider == "custom":
            return "custom_default", ModelProfile(
                provider="custom",
                model=model_name or "default",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )

    # Fallback to direct environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "claude-3-7-sonnet-20250219")
        return "anthropic_default", ModelProfile(
            provider="anthropic",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
            detect_capabilities=False,
        )

    if os.environ.get("NVIDIA_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        return "nvidia_default", ModelProfile(
            provider="nvidia",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    if os.environ.get("OPENROUTER_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "anthropic/claude-3.7-sonnet")
        return "openrouter_default", ModelProfile(
            provider="openrouter",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
            detect_capabilities=False,
        )

    if os.environ.get("OPENAI_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "gpt-4o")
        return "openai_default", ModelProfile(
            provider="openai",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
            detect_capabilities=False,
        )

    return "", None


def prompt_for_credentials(console: Console) -> dict[str, Any]:
    """Interactively prompt user for provider and API key on first launch."""
    banner = (
        "[bold cyan]⚡ Welcome to Forge Setup[/bold cyan]\n"
        "[dim]No API key detected. Let's configure your model provider (one-time setup).[/dim]\n\n"
        "[bold]Select a provider:[/bold]\n"
        "  [bold cyan]1[/bold cyan] NVIDIA NIM (Llama 3.1 Nemotron 70B, Mistral Large, DeepSeek)\n"
        "  [bold cyan]2[/bold cyan] OpenRouter (Claude 3.7, Sonnet 3.5, GPT-4o, DeepSeek)\n"
        "  [bold cyan]3[/bold cyan] OpenAI (GPT-4o, GPT-4o-mini)\n"
        "  [bold cyan]4[/bold cyan] Ollama (Local, free, private - runs locally on port 11434)\n"
        "  [bold cyan]5[/bold cyan] Custom OpenAI-compatible endpoint"
    )
    console.print(Panel(banner, border_style="cyan", title="Forge Initial Setup"))

    choice = Prompt.ask(
        "[bold cyan]Enter choice[/bold cyan]",
        choices=["1", "2", "3", "4", "5"],
        default="1",
        console=console,
    )

    mapping = {
        "1": "nvidia",
        "2": "openrouter",
        "3": "openai",
        "4": "ollama",
        "5": "custom",
    }
    provider = mapping[choice]
    meta = PROVIDER_METADATA[provider]

    api_key = ""
    base_url = meta.get("base_url", "")
    default_model = meta.get("default_model", "")

    if provider in {"nvidia", "openrouter", "openai", "custom"}:
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
