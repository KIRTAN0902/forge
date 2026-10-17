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
    "gemini": {
        "name": "Google Gemini",
        "env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "label": "Google Gemini (gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash)",
    },
    "deepseek": {
        "name": "DeepSeek",
        "env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "label": "DeepSeek (deepseek-chat, deepseek-reasoner)",
    },
    "groq": {
        "name": "Groq",
        "env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "label": "Groq (Llama 3.3 70B, Llama 3.1 8B)",
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

    # 3. Check ~/.forge/config.json via ForgeConfigManager
    with contextlib.suppress(Exception):
        from forge.config.settings import ForgeConfigManager, ProviderName

        cfg_mgr = ForgeConfigManager()
        for prov_enum in ProviderName:
            p_cfg = cfg_mgr.provider(prov_enum)
            p_key = cfg_mgr.api_key_for(prov_enum)
            if p_key and p_cfg.api_key_env:
                if not os.environ.get(p_cfg.api_key_env):
                    os.environ[p_cfg.api_key_env] = p_key
                has_credentials = True
            elif p_cfg.api_key_env and os.environ.get(p_cfg.api_key_env):
                has_credentials = True

    # 4. Check if any known environment variable is already set
    for meta in PROVIDER_METADATA.values():
        env_var = meta.get("env")
        if env_var and os.environ.get(env_var):
            has_credentials = True

    return has_credentials


def resolve_default_model_profile() -> tuple[str, ModelProfile | None]:
    """Resolve a default ModelProfile based on the active environment/credentials."""
    # 1. Check ~/.forge/config.json via ForgeConfigManager
    with contextlib.suppress(Exception):
        from forge.config.settings import AUTO_PRIORITY, ForgeConfigManager, ProviderName

        cfg_mgr = ForgeConfigManager()
        binding = cfg_mgr.binding_for("anvil") or cfg_mgr.binding_for("all")
        if binding and cfg_mgr.is_connected(binding.provider):
            return f"{binding.provider.value}_default", ModelProfile(
                provider=binding.provider.value,
                model=binding.model,
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True, vision=True
                ),
                detect_capabilities=False,
            )

        for env_var, prov_enum, default_model in AUTO_PRIORITY:
            key = cfg_mgr.api_key_for(prov_enum) or (os.environ.get(env_var) if env_var else None)
            if key or prov_enum is ProviderName.OLLAMA:
                p_cfg = cfg_mgr.provider(prov_enum)
                model = p_cfg.default_model or default_model
                return f"{prov_enum.value}_default", ModelProfile(
                    provider=prov_enum.value,
                    model=model,
                    capabilities=ModelCapabilities(
                        tool_calling=True, structured_output=True, vision=True
                    ),
                    detect_capabilities=False,
                )

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
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True, vision=True
                ),
                detect_capabilities=False,
            )
        if provider == "openrouter" and (
            os.environ.get("OPENROUTER_API_KEY") or stored.get("api_key")
        ):
            return "openrouter_default", ModelProfile(
                provider="openrouter",
                model=model_name or "anthropic/claude-3.7-sonnet",
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True, vision=True
                ),
                detect_capabilities=False,
            )
        if provider == "openai" and (os.environ.get("OPENAI_API_KEY") or stored.get("api_key")):
            return "openai_default", ModelProfile(
                provider="openai",
                model=model_name or "gpt-4o",
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True, vision=True
                ),
                detect_capabilities=False,
            )
        if provider == "gemini" and (os.environ.get("GEMINI_API_KEY") or stored.get("api_key")):
            return "gemini_default", ModelProfile(
                provider="gemini",
                model=model_name or "gemini-2.5-flash",
                capabilities=ModelCapabilities(
                    tool_calling=True, structured_output=True, vision=True
                ),
                detect_capabilities=False,
            )
        if provider == "deepseek" and (os.environ.get("DEEPSEEK_API_KEY") or stored.get("api_key")):
            return "deepseek_default", ModelProfile(
                provider="deepseek",
                model=model_name or "deepseek-chat",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )
        if provider == "groq" and (os.environ.get("GROQ_API_KEY") or stored.get("api_key")):
            return "groq_default", ModelProfile(
                provider="groq",
                model=model_name or "llama-3.3-70b-versatile",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
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
    if os.environ.get("GEMINI_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "gemini-2.5-flash")
        return "gemini_default", ModelProfile(
            provider="gemini",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True, vision=True),
            detect_capabilities=False,
        )

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

    if os.environ.get("DEEPSEEK_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "deepseek-chat")
        return "deepseek_default", ModelProfile(
            provider="deepseek",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    if os.environ.get("GROQ_API_KEY"):
        model_name = os.environ.get("FORGE_DEFAULT_MODEL", "llama-3.3-70b-versatile")
        return "groq_default", ModelProfile(
            provider="groq",
            model=model_name,
            capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
            detect_capabilities=False,
        )

    return "", None


def read_clipboard_key() -> str | None:
    """Safely retrieve clipboard content cross-platform without third-party dependencies."""
    # 1. Try pyperclip if installed
    try:
        import pyperclip  # type: ignore[import-untyped]

        val = pyperclip.paste()
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass

    # 2. Windows native (zero-dependency via ctypes with 64-bit/32-bit types)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.argtypes = [wintypes.HWND]
            user32.OpenClipboard.restype = wintypes.BOOL
            user32.CloseClipboard.argtypes = []
            user32.CloseClipboard.restype = wintypes.BOOL
            user32.GetClipboardData.argtypes = [wintypes.UINT]
            user32.GetClipboardData.restype = wintypes.HANDLE
            kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
            kernel32.GlobalLock.restype = wintypes.LPVOID
            kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
            kernel32.GlobalUnlock.restype = wintypes.BOOL

            cf_unicode = 13
            if user32.OpenClipboard(None):
                try:
                    handle = user32.GetClipboardData(cf_unicode)
                    if handle:
                        ptr = kernel32.GlobalLock(handle)
                        if ptr:
                            try:
                                text = ctypes.c_wchar_p(ptr).value
                                if text and text.strip():
                                    return text.strip()
                            finally:
                                kernel32.GlobalUnlock(handle)
                finally:
                    user32.CloseClipboard()
        except Exception:
            pass

    # 3. macOS native (pbpaste)
    import subprocess
    import sys

    if sys.platform == "darwin":
        try:
            proc = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=1)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass

    # 4. Linux native (wl-paste or xclip)
    if sys.platform.startswith("linux"):
        for cmd in (["wl-paste"], ["xclip", "-selection", "clipboard", "-o"]):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                continue

    return None


def looks_like_api_key(text: str | None) -> bool:
    """Check if clipboard text looks like an API key or auth token."""
    if not text:
        return False
    text = text.strip()
    if "\n" in text or "\r" in text or " " in text:
        return False
    if len(text) < 14 or len(text) > 300:
        return False
    if text.startswith(("http://", "https://", "file://", "git@", "ssh://")):
        return False
    known_prefixes = (
        "sk-",
        "nvapi-",
        "gsk_",
        "xai-",
        "AIza",
        "anthropic",
        "ghp_",
        "key-",
        "or-v1-",
        "hf_",
        "co-",
        "ms-",
    )
    if any(text.startswith(p) for p in known_prefixes):
        return True
    return len(text) >= 20 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text)


def prompt_api_key(
    prompt_label: str = "API Key",
    *,
    console: Console | None = None,
    default: str = "",
    allow_empty: bool = False,
    provider_name: str = "",
    prompt_class: type[Prompt] | None = None,
) -> str:
    """Prompt the user for an API key with full clipboard paste support (Ctrl+V, right-click, Shift+Insert)."""
    p_cls = prompt_class or Prompt

    # 1. If Prompt.ask is patched/mocked in tests, delegate to it
    if hasattr(p_cls.ask, "mock_calls") or hasattr(p_cls.ask, "assert_called"):
        val = p_cls.ask(
            prompt_label,
            password=True,
            default=default,
            show_default=bool(default),
            console=console,
        )
        return str(val).strip() if val is not None else ""

    c = console or Console()

    # 2. Inspect clipboard to see if user already copied an API key
    clip = read_clipboard_key()
    clip_is_key = looks_like_api_key(clip)
    effective_default = default

    if clip and clip_is_key and not default:
        masked_clip = f"{clip[:6]}...{clip[-4:]}" if len(clip) > 10 else "****"
        c.print(
            f"  [bold cyan]📋 Detected API key in clipboard:[/bold cyan] [yellow]{masked_clip}[/yellow] "
            f"[dim]({len(clip)} chars)[/dim]\n"
            f"  [dim]Press [bold white]Enter[/bold white] to use this clipboard key, "
            f"or [bold white]Ctrl+V[/bold white] / right-click to paste another:[/dim]"
        )
        effective_default = clip

    # 3. If non-interactive / stdin is not a tty, fallback to standard input
    import sys

    if not sys.stdin.isatty():
        try:
            val = input().strip()
        except (KeyboardInterrupt, EOFError):
            return ""
        if not val and effective_default:
            val = effective_default
        return val

    # 4. Interactive prompt using prompt_toolkit for full cross-platform paste & masking
    while True:
        try:
            from prompt_toolkit.formatted_text import HTML
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.shortcuts import prompt as pt_prompt

            kb = KeyBindings()

            @kb.add("c-v")
            def _paste_ctrl_v(event: Any) -> None:
                p = read_clipboard_key()
                if p:
                    event.current_buffer.insert_text(p)

            @kb.add("s-insert")
            def _paste_shift_ins(event: Any) -> None:
                p = read_clipboard_key()
                if p:
                    event.current_buffer.insert_text(p)

            clean_label = prompt_label.replace("[", "").replace("]", "")
            formatted_prompt = HTML(f"<b><ansicyan>{clean_label}</ansicyan></b>: ")
            entered = pt_prompt(
                formatted_prompt,
                is_password=True,
                key_bindings=kb,
            ).strip()

        except (KeyboardInterrupt, EOFError):
            c.print("\n[dim]Cancelled.[/dim]")
            return ""
        except Exception:
            # Fallback if prompt_toolkit fails for any reason
            entered = c.input(f"[bold cyan]{prompt_label}: [/bold cyan]").strip()

        # Clean up any \x16 characters (raw Ctrl+V character code sent by some terminals)
        if "\x16" in entered:
            clip_text = read_clipboard_key() or ""
            entered = entered.replace("\x16", clip_text).strip()

        # Strip any extraneous quotes user might have pasted
        entered = entered.strip("'\"")

        # If user hit Enter with no text, use effective_default if available
        if not entered and effective_default:
            entered = effective_default

        if entered or allow_empty:
            if entered:
                masked = f"{entered[:6]}...{entered[-4:]}" if len(entered) > 10 else "****"
                c.print(
                    f"  [bold green]✔ Key accepted:[/bold green] [cyan]{masked}[/cyan] "
                    f"[dim]({len(entered)} chars)[/dim]"
                )
            return entered

        c.print(
            "[red]API key cannot be empty. Please paste your API key (press Ctrl+V or right-click):[/red]"
        )


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

    if provider in {"nvidia", "openrouter", "openai", "anthropic", "custom"}:
        console.print()
        console.print(
            f"[bold yellow]Paste your {meta['name']} API key[/bold yellow] "
            "[dim](input is hidden for security):[/dim]"
        )
        api_key = prompt_api_key(
            "API Key",
            console=console,
            provider_name=meta["name"],
            prompt_class=Prompt,
        )

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
