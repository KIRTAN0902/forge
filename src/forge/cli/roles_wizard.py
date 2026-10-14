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
            "planner": (provider, planner_model),
            "executor": (provider, executor_model),
            "qa": (provider, qa_model),
            "orchestrator": (provider, planner_model),
            "default": (provider, planner_model),
        }
        save_multi_role_preset(project_dir, role_configs)
        console.print(f"\n[bold green]✔ Configured specialized models on {provider}[/bold green]")

    else:
        # Mode 3: Different models and different API keys
        console.print("\n[bold]Configuring Multi-Provider / Multi-Key Agents:[/bold]")

        console.print("\n[bold cyan]1. Configure Planner Agent (Architect & Strategy):[/bold cyan]")
        p_prov, _, p_def = _prompt_provider_and_key(console, "Planner Agent")
        p_model = Prompt.ask("Planner Model ID", default=p_def, console=console).strip()

        console.print(
            "\n[bold cyan]2. Configure Executor Agent (Code Generation & Tools):[/bold cyan]"
        )
        e_prov, _, e_def = _prompt_provider_and_key(console, "Executor Agent")
        e_model = Prompt.ask("Executor Model ID", default=e_def, console=console).strip()

        console.print("\n[bold cyan]3. Configure QA Agent (Testing & Auditing):[/bold cyan]")
        q_prov, _, q_def = _prompt_provider_and_key(console, "QA Agent")
        q_model = Prompt.ask("QA Model ID", default=q_def, console=console).strip()

        role_configs = {
            "planner": (p_prov, p_model),
            "executor": (e_prov, e_model),
            "qa": (q_prov, q_model),
            "orchestrator": (p_prov, p_model),
            "default": (p_prov, p_model),
        }
        save_multi_role_preset(project_dir, role_configs)
        console.print("\n[bold green]✔ Successfully configured multi-provider agents[/bold green]")

    init_credentials_env()
    config = load_config(project_dir)
    console.print()
    print_roles_table(config, console)
