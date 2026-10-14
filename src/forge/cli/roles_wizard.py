"""Interactive Role & Model Configuration Wizard.

Allows users to configure:
1. One model + one API key across all agents.
2. Different models across agents using the same API key / provider.
3. Different models across agents using different API keys / providers.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from forge.cli.credentials import (
    PROVIDER_METADATA,
    get_provider_api_key,
    init_credentials_env,
    load_stored_credentials,
    save_stored_credentials,
)
from forge.config.loader import load_config
from forge.config.roles import (
    get_roles_summary,
    save_multi_role_preset,
    save_single_model_preset,
)
from forge.config.schema import ForgeConfig

PROVIDER_CHOICES = ["1", "2", "3", "4", "5"]
PROVIDER_MAP = {
    "1": "nvidia",
    "2": "openrouter",
    "3": "openai",
    "4": "ollama",
    "5": "custom",
}


def print_roles_table(config: ForgeConfig, console: Console) -> None:
    """Render a rich table displaying current role bindings and API key status."""
    summaries = get_roles_summary(config, key_lookup=get_provider_api_key)
    table = Table(title="Agent Role & Model Configuration", border_style="cyan")
    table.add_column("Agent Role", style="bold white")
    table.add_column("Model Profile", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model ID", style="bold")
    table.add_column("API Key Status", justify="center")

    for s in summaries:
        key_status = "[green]✔ Configured[/green]" if s.has_key else "[red]✖ Missing[/red]"
        table.add_row(
            s.role.title(),
            s.profile,
            s.provider,
            s.model,
            f"{key_status} [dim]({s.key_source})[/dim]",
        )

    console.print(table)


def _prompt_provider_and_key(console: Console, role_label: str = "") -> tuple[str, str, str]:
    """Prompt user to select a provider and provide an API key if needed.

    Returns:
        (provider_name, api_key, default_model)
    """
    prefix = f"for {role_label} " if role_label else ""
    banner = (
        f"[bold]Select a model provider {prefix}:[/bold]\n"
        "  [cyan]1[/cyan] NVIDIA NIM (Nemotron 3 Super 120B, Nemotron 3.5 Lightning)\n"
        "  [cyan]2[/cyan] OpenRouter (Claude 3.7, Sonnet 3.5, GPT-4o, DeepSeek)\n"
        "  [cyan]3[/cyan] OpenAI (GPT-4o, GPT-4o-mini)\n"
        "  [cyan]4[/cyan] Ollama (Local, free, private on port 11434)\n"
        "  [cyan]5[/cyan] Custom OpenAI-compatible endpoint"
    )
    console.print(banner)

    choice = Prompt.ask(
        "Provider choice",
        choices=PROVIDER_CHOICES,
        default="1",
        console=console,
    )
    provider = PROVIDER_MAP[choice]
    meta = PROVIDER_METADATA[provider]

    existing_key = get_provider_api_key(provider)
    api_key = existing_key or ""

    if provider in {"nvidia", "openrouter", "openai", "custom"}:
        if existing_key:
            masked = (
                f"{existing_key[:7]}...{existing_key[-4:]}" if len(existing_key) > 12 else "****"
            )
            console.print(f"[dim]Existing {meta['name']} key detected: {masked}[/dim]")
            use_existing = Confirm.ask(
                f"Keep existing {meta['name']} API key?", default=True, console=console
            )
            if not use_existing:
                api_key = Prompt.ask(
                    f"New {meta['name']} API Key", password=True, console=console
                ).strip()
        else:
            api_key = Prompt.ask(
                f"Paste your {meta['name']} API Key", password=True, console=console
            ).strip()

        if api_key:
            save_stored_credentials(provider, api_key)
            init_credentials_env()

    return provider, api_key, meta.get("default_model", "")


def run_roles_wizard(project_dir: Path, console: Console) -> None:
    """Interactive wizard to configure roles: single model, same provider, or multi-provider."""
    banner = (
        "[bold cyan]⚡ Multi-Agent Model & Role Configuration[/bold cyan]\n"
        "[dim]Choose how you want to assign AI models and API keys to your agents:[/dim]\n\n"
        "  [bold cyan]1[/bold cyan] [bold]One Model & One API Key (All Agents)[/bold]\n"
        "      All agents (Planner, Executor, QA) share a single model and key.\n\n"
        "  [bold cyan]2[/bold cyan] [bold]Different Models, Same API Key / Provider[/bold]\n"
        "      e.g. Planner: Nemotron 120B (deep reasoning)\n"
        "           Executor: Nemotron 30B (fast code generation)\n"
        "           QA: Nemotron 120B (auditing)\n"
        "           [dim]Both use your existing NVIDIA NIM or OpenRouter key.[/dim]\n\n"
        "  [bold cyan]3[/bold cyan] [bold]Different Models & Different API Keys[/bold]\n"
        "      e.g. Planner: Claude 3.7 Sonnet (OpenRouter)\n"
        "           Executor: Nemotron 3 Super (NVIDIA NIM)\n"
        "           QA: GPT-4o (OpenAI)\n"
        "           [dim]Each agent gets its own specialized provider, model, and key.[/dim]"
    )
    console.print(Panel(banner, border_style="cyan", title="Forge Role Configuration"))

    mode = Prompt.ask(
        "[bold cyan]Select configuration mode[/bold cyan]",
        choices=["1", "2", "3"],
        default="1",
        console=console,
    )

    if mode == "1":
        # Mode 1: One model, one key for all agents
        console.print("\n[bold]Configuring Single Model for All Agents:[/bold]")
        provider, _, default_model = _prompt_provider_and_key(console)
        model = Prompt.ask("Model ID", default=default_model, console=console).strip()

        save_single_model_preset(project_dir, provider=provider, model=model)
        console.print(f"\n[bold green]✔ Configured all agents with {provider}/{model}[/bold green]")

    elif mode == "2":
        # Mode 2: Different models, same provider/key
        console.print("\n[bold]Configuring Shared Provider with Specialized Models:[/bold]")
        provider, _, default_model = _prompt_provider_and_key(console)

        # Smart defaults based on provider
        p_def = default_model
        e_def = default_model
        q_def = default_model

        if provider == "nvidia":
            p_def = "nvidia/nemotron-3-super-120b-a12b"
            e_def = "nvidia/nemotron-3.5-lightning-30b-a3b"
            q_def = "nvidia/nemotron-3-super-120b-a12b"
        elif provider == "openrouter":
            p_def = "anthropic/claude-3.7-sonnet"
            e_def = "anthropic/claude-3.5-sonnet"
            q_def = "openai/gpt-4o"

        console.print()
        orch_model = Prompt.ask(
            "Orchestrator Agent Model (Brainstorming & Tech Lead)", default=p_def, console=console
        ).strip()
        planner_model = Prompt.ask(
            "Planner Agent Model (Architect & Decomposition)", default=p_def, console=console
        ).strip()
        executor_model = Prompt.ask(
            "Executor Agent Model (Code Writing & Tools)", default=e_def, console=console
        ).strip()
        qa_model = Prompt.ask(
            "QA Agent Model (Testing & Auditing)", default=q_def, console=console
        ).strip()

        role_configs = {
            "orchestrator": (provider, orch_model),
            "planner": (provider, planner_model),
            "executor": (provider, executor_model),
            "qa": (provider, qa_model),
            "default": (provider, orch_model),
        }
        save_multi_role_preset(project_dir, role_configs)
        console.print(f"\n[bold green]✔ Configured specialized models on {provider}[/bold green]")

    else:
        # Mode 3: Different models and different API keys
        console.print("\n[bold]Configuring Multi-Provider / Multi-Key Agents:[/bold]")

        console.print(
            "\n[bold cyan]1. Configure Orchestrator Agent (Brainstorming & Tech Lead):[/bold cyan]"
        )
        o_prov, _, o_def = _prompt_provider_and_key(console, "Orchestrator Agent")
        o_model = Prompt.ask("Orchestrator Model ID", default=o_def, console=console).strip()

        console.print("\n[bold cyan]2. Configure Planner Agent (Architect & Strategy):[/bold cyan]")
        p_prov, _, p_def = _prompt_provider_and_key(console, "Planner Agent")
        p_model = Prompt.ask("Planner Model ID", default=p_def, console=console).strip()

        console.print(
            "\n[bold cyan]3. Configure Executor Agent (Code Generation & Tools):[/bold cyan]"
        )
        e_prov, _, e_def = _prompt_provider_and_key(console, "Executor Agent")
        e_model = Prompt.ask("Executor Model ID", default=e_def, console=console).strip()

        console.print("\n[bold cyan]4. Configure QA Agent (Testing & Auditing):[/bold cyan]")
        q_prov, _, q_def = _prompt_provider_and_key(console, "QA Agent")
        q_model = Prompt.ask("QA Model ID", default=q_def, console=console).strip()

        role_configs = {
            "orchestrator": (o_prov, o_model),
            "planner": (p_prov, p_model),
            "executor": (e_prov, e_model),
            "qa": (q_prov, q_model),
            "default": (o_prov, o_model),
        }
        save_multi_role_preset(project_dir, role_configs)
        console.print("\n[bold green]✔ Successfully configured multi-provider agents[/bold green]")

    init_credentials_env()
    config = load_config(project_dir)
    console.print()
    print_roles_table(config, console)


def configure_single_role(project_dir: Path, role: str, console: Console) -> str | None:
    """Prompt and update model and provider for a single agent role via dropdown."""
    from forge.cli.model_catalog import prompt_model_dropdown
    from forge.config.roles import set_role_model

    config = load_config(project_dir)
    role_title = role.title()
    current_profile = config.roles.get(role)
    current_model = "default"
    current_provider = "nvidia"
    if current_profile and current_profile.model in config.models:
        prof = config.models[current_profile.model]
        current_model = prof.model
        current_provider = prof.provider

    console.print(f"\n[bold cyan]1. Select Provider for {role_title} Agent:[/bold cyan]")
    provider, api_key, default_model = _prompt_provider_and_key(console, f"{role_title} Agent")

    # Get base_url if any
    base_url = None
    if provider in config.providers:
        base_url = config.providers[provider].base_url

    console.print(f"\n[bold cyan]2. Choose Model from {provider.upper()} Dropdown:[/bold cyan]")
    chosen_model = prompt_model_dropdown(
        role_label=role_title,
        provider=provider,
        current_model=current_model if provider == current_provider else default_model,
        console=console,
        api_key=api_key,
        base_url=base_url,
    )

    set_role_model(project_dir, role, model=chosen_model, provider=provider)
    console.print(f"\n[bold green]✔ Updated {role_title} model to {provider}/{chosen_model}[/bold green]\n")
    Prompt.ask("Press Enter to continue...", console=console)
    return f"{role_title} ➔ [cyan]{chosen_model}[/cyan] ({provider})"


def _quick_setup_all(project_dir: Path, console: Console) -> str | None:
    """Prompt provider and model dropdown to assign across all agents."""
    from forge.cli.model_catalog import prompt_model_dropdown

    console.print("\n[bold]Quick Setup: Single Model for All Agents[/bold]")
    provider, api_key, default_model = _prompt_provider_and_key(console, "All Agents")
    config = load_config(project_dir)
    base_url = config.providers[provider].base_url if provider in config.providers else None

    chosen_model = prompt_model_dropdown(
        role_label="All Agents",
        provider=provider,
        current_model=default_model,
        console=console,
        api_key=api_key,
        base_url=base_url,
    )

    save_single_model_preset(project_dir, provider=provider, model=chosen_model)
    console.print(f"\n[bold green]✔ Configured all agents with {provider}/{chosen_model}[/bold green]\n")
    Prompt.ask("Press Enter to continue...", console=console)
    return f"All Agents ➔ [cyan]{chosen_model}[/cyan] ({provider})"


def run_model_menu(project_dir: Path, console: Console) -> list[str]:
    """Interactive Model Configuration Menu (/model) running in an overlay screen.

    Returns the list of changes made during the session so chat can display them.
    """
    changes: list[str] = []

    def _menu_loop() -> None:
        while True:
            init_credentials_env()
            config = load_config(project_dir)
            console.clear()
            print_roles_table(config, console)

            menu_text = (
                "[bold cyan]Select an agent role to configure model:[/bold cyan]\n\n"
                "  [bold cyan]1[/bold cyan] 🎯 [bold]Orchestrator[/bold]  (Brainstorming, tech lead, routing)\n"
                "  [bold cyan]2[/bold cyan] 📐 [bold]Planner[/bold]       (Requirements, architecture, decomposition)\n"
                "  [bold cyan]3[/bold cyan] ⚡ [bold]Executor[/bold]      (Code writing, tool use, terminal)\n"
                "  [bold cyan]4[/bold cyan] 🛡️  [bold]QA[/bold]            (Test suites, code review, auditing)\n"
                "  [bold cyan]5[/bold cyan] ⚡ [bold]Quick Setup[/bold]   (Assign 1 model to ALL agents)\n"
                "  [bold cyan]6[/bold cyan] 🧙 [bold]Full Setup Wizard[/bold] (Step-by-step multi-role setup)\n"
                "  [bold cyan]0[/bold cyan] ↩️  [bold]Done / Return to Chat[/bold]"
            )
            console.print(
                Panel(menu_text, border_style="cyan", title="Forge Model Configuration Menu (/model)")
            )

            choice = Prompt.ask(
                "Select option",
                choices=["0", "1", "2", "3", "4", "5", "6", "exit", "quit", "q"],
                default="0",
                console=console,
            )

            if choice in {"0", "exit", "quit", "q"}:
                break

            if choice in {"1", "2", "3", "4"}:
                role_names = {"1": "orchestrator", "2": "planner", "3": "executor", "4": "qa"}
                role = role_names[choice]
                ch = configure_single_role(project_dir, role, console)
                if ch:
                    changes.append(ch)
            elif choice == "5":
                ch = _quick_setup_all(project_dir, console)
                if ch:
                    changes.append(ch)
            elif choice == "6":
                run_roles_wizard(project_dir, console)
                changes.append("Multi-role wizard completed")

    try:
        with console.screen():
            _menu_loop()
    except Exception:
        # Fallback if terminal does not support alternate screen buffer
        _menu_loop()

    return changes


def run_provider_menu(project_dir: Path, console: Console) -> list[str]:
    """Interactive Provider & URI/URL Configuration Menu (/provider).

    Allows changing providers, configuring API keys, and setting custom Base URIs/URLs.
    """
    changes: list[str] = []

    def _menu_loop() -> None:
        while True:
            init_credentials_env()
            config = load_config(project_dir)
            console.clear()

            table = Table(title="Configured Model Providers & Base URIs", border_style="cyan")
            table.add_column("No.", style="bold cyan", width=4, justify="right")
            table.add_column("Provider", style="bold white")
            table.add_column("Base URI / URL", style="cyan")
            table.add_column("API Key Status", justify="center")

            prov_items = [
                ("1", "nvidia", "NVIDIA NIM"),
                ("2", "openrouter", "OpenRouter"),
                ("3", "openai", "OpenAI"),
                ("4", "ollama", "Ollama (Local)"),
                ("5", "custom", "Custom OpenAI-Compatible Endpoint"),
            ]

            stored = load_stored_credentials() or {}
            stored_provs = stored.get("providers", {})

            for num, p_key, p_label in prov_items:
                api_key = get_provider_api_key(p_key)
                custom_url = None
                if p_key in config.providers:
                    custom_url = config.providers[p_key].base_url
                elif isinstance(stored_provs, dict) and p_key in stored_provs:
                    custom_url = stored_provs[p_key].get("base_url")

                base_url = custom_url or PROVIDER_METADATA.get(p_key, {}).get("base_url", "none")

                if p_key == "ollama":
                    key_stat = "[green]✔ Local (No key needed)[/green]"
                elif api_key:
                    masked = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "****"
                    key_stat = f"[green]✔ Configured ({masked})[/green]"
                else:
                    key_stat = "[red]✖ Not configured[/red]"

                table.add_row(num, p_label, base_url, key_stat)

            console.print(table)

            menu_panel = (
                "[bold cyan]Select a provider to configure API Key or custom URI/URL:[/bold cyan]\n\n"
                "  [cyan]1[/cyan] NVIDIA NIM (Nemotron 3 Super, Llama 3.3, DeepSeek)\n"
                "  [cyan]2[/cyan] OpenRouter (Claude 3.7, GPT-4o, Gemini, DeepSeek)\n"
                "  [cyan]3[/cyan] OpenAI (GPT-4o, o1, o3-mini)\n"
                "  [cyan]4[/cyan] Ollama (Local server on http://localhost:11434)\n"
                "  [cyan]5[/cyan] Custom Endpoint (vLLM, LMStudio, LiteLLM, Groq)\n"
                "  [cyan]0[/cyan] ↩️  Done / Return to Chat"
            )
            console.print(Panel(menu_panel, border_style="cyan", title="Forge Provider Menu (/provider)"))

            choice = Prompt.ask(
                "Choose provider to configure",
                choices=["0", "1", "2", "3", "4", "5", "exit", "quit", "q"],
                default="0",
                console=console,
            )

            if choice in {"0", "exit", "quit", "q"}:
                break

            p_map = {"1": "nvidia", "2": "openrouter", "3": "openai", "4": "ollama", "5": "custom"}
            provider = p_map[choice]
            meta = PROVIDER_METADATA.get(provider, {})
            name = meta.get("name", provider.title())

            console.print(f"\n[bold cyan]Configuring {name}:[/bold cyan]")

            # 1. Base URL / URI prompt
            default_url = meta.get("base_url", "http://localhost:8000/v1")
            existing_url = default_url
            if provider in config.providers:
                existing_url = config.providers[provider].base_url
            elif isinstance(stored_provs, dict) and provider in stored_provs:
                existing_url = stored_provs[provider].get("base_url", default_url)

            new_url = Prompt.ask(
                f"Base URI / URL for {name}",
                default=existing_url,
                console=console,
            ).strip()

            # 2. API Key prompt (except ollama)
            existing_key = get_provider_api_key(provider) or ""
            api_key = existing_key
            if provider != "ollama":
                if existing_key:
                    masked = f"{existing_key[:6]}...{existing_key[-4:]}" if len(existing_key) > 10 else "****"
                    console.print(f"[dim]Existing key detected: {masked}[/dim]")
                    keep = Confirm.ask("Keep existing API key?", default=True, console=console)
                    if not keep:
                        api_key = Prompt.ask("New API Key", password=True, console=console).strip()
                else:
                    api_key = Prompt.ask("Paste API Key", password=True, console=console).strip()

            from forge.config.roles import set_provider_config

            save_stored_credentials(provider, api_key=api_key, base_url=new_url)
            set_provider_config(project_dir, provider, base_url=new_url, api_key_env=meta.get("env"))
            init_credentials_env()

            change_str = f"Provider: [bold]{name}[/bold] (URL: {new_url})"
            changes.append(change_str)
            console.print(f"\n[bold green]✔ Successfully updated {change_str}[/bold green]\n")
            Prompt.ask("Press Enter to continue...", console=console)

    try:
        with console.screen():
            _menu_loop()
    except Exception:
        _menu_loop()

    return changes

