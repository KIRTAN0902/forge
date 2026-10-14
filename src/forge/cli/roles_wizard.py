"""Interactive Role & Model Configuration Wizard.

Allows users to configure:
1. One model + one API key across all agents.
2. Different models across agents using the same API key / provider.
3. Different models across agents using different API keys / providers.
"""

from __future__ import annotations

import os
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


def run_provider_modal(project_dir: Path, console: Console) -> list[str]:
    """Connect a provider modal (/provider) showing only connected and available providers."""
    from forge.cli.tui_modal import TuiItem, run_tui_modal

    init_credentials_env()
    config = load_config(project_dir)
    stored = load_stored_credentials() or {}
    stored_provs = stored.get("providers", {})

    all_providers = [
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1"),
        ("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1"),
        ("openai", "OpenAI", "https://api.openai.com/v1"),
        ("ollama", "Ollama (Local)", "http://localhost:11434/v1"),
        ("custom", "Custom Endpoint", "http://localhost:8000/v1"),
    ]

    connected_items: list[TuiItem] = []
    available_items: list[TuiItem] = []

    for p_id, p_title, default_url in all_providers:
        api_key = get_provider_api_key(p_id)
        custom_url = None
        if p_id in config.providers:
            custom_url = config.providers[p_id].base_url
        elif isinstance(stored_provs, dict) and p_id in stored_provs:
            custom_url = stored_provs[p_id].get("base_url")

        active_url = custom_url or default_url
        is_connected = (
            bool(api_key)
            or custom_url is not None
            or (p_id == "ollama" and p_id in config.providers)
        )

        if is_connected:
            connected_items.append(
                TuiItem(
                    id=p_id,
                    title=p_title,
                    subtitle=f"({active_url})",
                    section="Connected Providers",
                    badge="Connected",
                    is_checked=True,
                )
            )
        else:
            available_items.append(
                TuiItem(
                    id=p_id,
                    title=p_title,
                    subtitle=f"({active_url})",
                    section="Available Providers",
                    badge="",
                    is_checked=False,
                )
            )

    items = connected_items + available_items

    chosen = run_tui_modal(
        title="Connect a provider",
        items=items,
        search_placeholder="Search providers",
        footer_left="esc",
        footer_left_key="close",
        footer_right="enter",
        footer_right_key="select",
        allow_ctrl_a=False,
    )

    if not isinstance(chosen, TuiItem):
        return []

    provider = chosen.id
    meta = PROVIDER_METADATA.get(provider, {})
    name = chosen.title

    console.print(f"\n[bold cyan]Configuring {name}:[/bold cyan]")

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

    existing_key = get_provider_api_key(provider) or ""
    api_key = existing_key
    if provider != "ollama":
        if existing_key:
            masked = (
                f"{existing_key[:6]}...{existing_key[-4:]}"
                if len(existing_key) > 10
                else "****"
            )
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
    return [change_str]


def run_models_modal(project_dir: Path, console: Console) -> list[str]:
    """Select model modal (/models) with dynamic user recent models and provider catalog."""
    from forge.cli.credentials import get_recent_models, record_recent_model
    from forge.cli.model_catalog import fetch_provider_models
    from forge.cli.tui_modal import CTRL_A_TOKEN, TuiItem, run_tui_modal
    from forge.config.roles import set_role_model

    changes: list[str] = []

    init_credentials_env()
    config = load_config(project_dir)

    # 1. First step: Select agent role (or All Agents)
    role_items: list[TuiItem] = [
        TuiItem(
            id="all",
            title="All Agents",
            subtitle="(Assign 1 model to Orchestrator, Planner, Executor & QA)",
            section="Agent Roles",
        ),
    ]

    for r_name in ("orchestrator", "planner", "executor", "qa"):
        curr_binding = config.roles.get(r_name)
        curr_prof_name = curr_binding.model if curr_binding else None
        curr_m = "default"
        if curr_prof_name and curr_prof_name in config.models:
            curr_m = config.models[curr_prof_name].model.split("/")[-1]

        role_desc = {
            "orchestrator": "(Brainstorming, requirements, tech lead)",
            "planner": "(Task architecture & decomposition)",
            "executor": "(Code writing & tool runtime)",
            "qa": "(Verification, test suites & auditing)",
        }.get(r_name, "")

        role_items.append(
            TuiItem(
                id=r_name,
                title=r_name.title(),
                subtitle=role_desc,
                section="Agent Roles",
                badge=curr_m,
            )
        )

    role_pick = run_tui_modal(
        title="Select agent role",
        items=role_items,
        search_placeholder="Search roles",
        footer_left="Connect provider",
        footer_left_key="ctrl+a",
        allow_ctrl_a=True,
    )

    if role_pick == CTRL_A_TOKEN:
        prov_changes = run_provider_modal(project_dir, console)
        changes.extend(prov_changes)
        changes.extend(run_models_modal(project_dir, console))
        return changes

    if not isinstance(role_pick, TuiItem):
        return changes

    chosen_role = role_pick.id
    role_label = role_pick.title

    # 2. Second step: Select model for the chosen role
    while True:
        init_credentials_env()
        config = load_config(project_dir)

        # Detect active provider
        target_role = "orchestrator" if chosen_role == "all" else chosen_role
        target_binding = config.roles.get(target_role)
        curr_prof_name = target_binding.model if target_binding else None
        current_model = "default"
        current_prov = "openrouter"
        if curr_prof_name and curr_prof_name in config.models:
            prof = config.models[curr_prof_name]
            current_model = prof.model
            current_prov = prof.provider
        else:
            stored = load_stored_credentials()
            if stored and "provider" in stored:
                current_prov = stored["provider"]
            elif os.environ.get("OPENROUTER_API_KEY"):
                current_prov = "openrouter"
            elif os.environ.get("NVIDIA_API_KEY"):
                current_prov = "nvidia"

        api_key = get_provider_api_key(current_prov)
        base_url = (
            config.providers[current_prov].base_url if current_prov in config.providers else None
        )

        all_models = fetch_provider_models(current_prov, api_key=api_key, base_url=base_url)

        model_items: list[TuiItem] = []
        seen_ids = set()

        # Dynamic Recent Models: only actual recent models used by the user
        user_recents = get_recent_models()
        # If user has no recorded recents yet, populate from currently configured role models in forge.yaml
        if not user_recents and config.roles:
            for b in config.roles.values():
                if b.model in config.models:
                    m_val = config.models[b.model].model
                    if m_val not in user_recents:
                        user_recents.append(m_val)

        for m_id in user_recents:
            seen_ids.add(m_id)
            is_active = (m_id == current_model)
            m_short = m_id.split("/")[-1]
            prov_tag = m_id.split("/")[0].title() if "/" in m_id else current_prov.title()
            model_items.append(
                TuiItem(
                    id=m_id,
                    title=m_short,
                    subtitle=prov_tag,
                    section="Recent Models",
                    badge="Recent",
                    is_dot=is_active,
                )
            )

        # All models section from live provider API
        prov_section = f"Available Models ({current_prov.title()})"
        for m_id in all_models:
            if m_id not in seen_ids:
                is_active = (m_id == current_model)
                m_short = m_id.split("/")[-1]
                prov_tag = m_id.split("/")[0].title() if "/" in m_id else current_prov.title()
                model_items.append(
                    TuiItem(
                        id=m_id,
                        title=m_short,
                        subtitle=prov_tag,
                        section=prov_section,
                        is_dot=is_active,
                    )
                )

        model_pick = run_tui_modal(
            title=f"Select model ({role_label})",
            items=model_items,
            search_placeholder="Search models",
            footer_left="Connect provider",
            footer_left_key="ctrl+a",
            footer_right="Favorite",
            footer_right_key="ctrl+f",
            allow_ctrl_a=True,
            allow_custom=True,
        )

        if model_pick == CTRL_A_TOKEN:
            prov_changes = run_provider_modal(project_dir, console)
            changes.extend(prov_changes)
            continue  # loop to refresh models with newly configured provider

        if not isinstance(model_pick, TuiItem):
            break

        selected_model_id = model_pick.id
        record_recent_model(selected_model_id)

        if chosen_role == "all":
            save_single_model_preset(project_dir, provider=current_prov, model=selected_model_id)
            changes.append(f"All Agents ➔ [cyan]{selected_model_id}[/cyan] ({current_prov})")
        else:
            set_role_model(project_dir, chosen_role, model=selected_model_id, provider=current_prov)
            changes.append(f"{role_label} ➔ [cyan]{selected_model_id}[/cyan] ({current_prov})")
        break

    return changes


def run_model_menu(project_dir: Path, console: Console) -> list[str]:
    """Interactive Model Configuration Menu (/models)."""
    return run_models_modal(project_dir, console)


def run_provider_menu(project_dir: Path, console: Console) -> list[str]:
    """Interactive Provider & URI/URL Configuration Menu (/provider)."""
    return run_provider_modal(project_dir, console)

