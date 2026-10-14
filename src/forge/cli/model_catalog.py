"""Model Catalog and Interactive Dropdown Picker for Providers.

Enables fetching live models from providers (OpenRouter, NVIDIA NIM, OpenAI, Ollama, etc.)
and allows developers to select models via a clean, searchable dropdown menu.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

CURATED_MODELS: dict[str, list[tuple[str, str]]] = {
    "openrouter": [
        ("anthropic/claude-3.7-sonnet", "Claude 3.7 Sonnet (Frontier reasoning & coding)"),
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (High precision code generation)"),
        ("deepseek/deepseek-r1", "DeepSeek R1 (Deep reasoning & architecture)"),
        ("deepseek/deepseek-chat", "DeepSeek V3 (Fast, low-cost coding)"),
        ("openai/gpt-4o", "GPT-4o (Omni intelligence)"),
        ("openai/gpt-4o-mini", "GPT-4o Mini (Fast & lightweight)"),
        ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B (Open weights flagship)"),
        ("qwen/qwen-2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B (Coding specialist)"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro (Large context & reasoning)"),
        ("google/gemini-2.5-flash", "Gemini 2.5 Flash (Ultra-fast execution)"),
    ],
    "nvidia": [
        ("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B (Deep reasoning & planning)"),
        ("nvidia/nemotron-3.5-lightning-30b-a3b", "Nemotron 3.5 Lightning 30B (Fast tool & code execution)"),
        ("meta/llama-3.3-70b-instruct", "Llama 3.3 70B Instruct (General software tasks)"),
        ("meta/llama-3.1-405b-instruct", "Llama 3.1 405B Instruct (Frontier-scale intelligence)"),
        ("deepseek-ai/deepseek-r1", "DeepSeek R1 (Reasoning & algorithms)"),
        ("qwen/qwen2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B (Code generation specialist)"),
        ("mistralai/mistral-large-2-instruct", "Mistral Large 2 (Multilingual code & logic)"),
    ],
    "openai": [
        ("gpt-4o", "GPT-4o (Flagship reasoning & multimodal)"),
        ("gpt-4o-mini", "GPT-4o Mini (Fast & affordable)"),
        ("o1", "o1 (Deep analytical problem solving)"),
        ("o3-mini", "o3-mini (High-speed math & coding)"),
        ("gpt-4-turbo", "GPT-4 Turbo (Prior generation flagship)"),
    ],
    "ollama": [
        ("qwen2.5-coder:latest", "Qwen 2.5 Coder (Local code specialist)"),
        ("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B (Lightweight local)"),
        ("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B (High capacity local)"),
        ("llama3.3:latest", "Llama 3.3 (Local general intelligence)"),
        ("deepseek-r1:latest", "DeepSeek R1 (Local reasoning)"),
    ],
    "custom": [
        ("default", "Default model on custom server"),
    ],
}

# In-memory cache for fetched provider models
_PROVIDER_MODELS_CACHE: dict[str, list[str]] = {}


def fetch_provider_models(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Fetch all available models from provider API with caching and curated fallback."""
    cache_key = f"{provider}:{base_url or ''}"
    if cache_key in _PROVIDER_MODELS_CACHE:
        return _PROVIDER_MODELS_CACHE[cache_key]

    curated = [m[0] for m in CURATED_MODELS.get(provider, [])]
    models: list[str] = []

    try:
        if provider == "openrouter":
            url = "https://openrouter.ai/api/v1/models"
            resp = httpx.get(url, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])

        elif provider == "nvidia":
            url = f"{base_url.rstrip('/') if base_url else 'https://integrate.api.nvidia.com/v1'}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.get(url, headers=headers, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])

        elif provider == "openai":
            url = f"{base_url.rstrip('/') if base_url else 'https://api.openai.com/v1'}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.get(url, headers=headers, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])

        elif provider == "ollama":
            url = f"{base_url.rstrip('/') if base_url else 'http://localhost:11434'}/api/tags"
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("models", []):
                    if isinstance(item, dict) and "name" in item:
                        models.append(item["name"])

        elif provider == "custom" and base_url:
            url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = httpx.get(url, headers=headers, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])

    except Exception:
        # Fallback to curated if offline or network error
        pass

    # Merge curated models first, then newly fetched models
    combined: list[str] = list(curated)
    for m in models:
        if m not in combined:
            combined.append(m)

    if combined:
        _PROVIDER_MODELS_CACHE[cache_key] = combined
        return combined

    return curated or ["default"]


def prompt_model_dropdown(
    role_label: str,
    provider: str,
    current_model: str,
    console: Console,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Render an interactive dropdown to pick any model available from the provider."""
    console.print(f"\n[cyan]Fetching available models for {provider}...[/cyan]")
    all_models = fetch_provider_models(provider, api_key=api_key, base_url=base_url)
    curated_tuples = CURATED_MODELS.get(provider, [])

    # Separate top choices
    top_models: list[tuple[str, str]] = []
    seen = set()
    for m_id, desc in curated_tuples:
        if m_id in all_models or not all_models:
            top_models.append((m_id, desc))
            seen.add(m_id)

    # If few curated, add first few all_models
    for m_id in all_models[:10]:
        if m_id not in seen:
            top_models.append((m_id, ""))
            seen.add(m_id)
            if len(top_models) >= 8:
                break

    while True:
        table = Table(
            title=f"Select Model for {role_label.title()} ({provider})",
            border_style="cyan",
            show_header=True,
        )
        table.add_column("No.", style="bold cyan", width=4, justify="right")
        table.add_column("Model ID", style="bold white")
        table.add_column("Description / Notes", style="dim")

        choices_map: dict[str, str] = {}
        for idx, (m_id, desc) in enumerate(top_models, start=1):
            key = str(idx)
            choices_map[key] = m_id
            active_marker = " [bold green](current)[/bold green]" if m_id == current_model else ""
            table.add_row(key, f"{m_id}{active_marker}", desc or "[dim]available on provider[/dim]")

        console.print(table)
        console.print(
            f"[dim]Total models available on {provider}: [bold white]{len(all_models)}[/bold white]. "
            "Type a number, [cyan]'s'[/cyan] to search, [cyan]'c'[/cyan] for custom ID, or [cyan]'0'[/cyan] to keep current.[/dim]\n"
        )

        valid_choices = list(choices_map.keys()) + ["s", "c", "0", "exit", "quit", "q"]
        default_pick = "1"
        for k, v in choices_map.items():
            if v == current_model:
                default_pick = k
                break

        choice = Prompt.ask(
            f"Select model for {role_label.title()}",
            choices=valid_choices,
            default=default_pick,
            console=console,
        ).strip().lower()

        if choice in {"0", "exit", "quit", "q"}:
            return current_model

        if choice in choices_map:
            return choices_map[choice]

        if choice == "c":
            custom_id = Prompt.ask("Enter custom Model ID", default=current_model, console=console).strip()
            return custom_id or current_model

        if choice == "s":
            # Interactive search query
            keyword = Prompt.ask("Search models keyword (e.g. 'claude', 'deepseek', 'coder', 'llama')", console=console).strip().lower()
            if not keyword:
                continue

            matches = [m for m in all_models if keyword in m.lower()]
            if not matches:
                console.print(f"[yellow]No models found matching '{keyword}'.[/yellow]\n")
                continue

            search_table = Table(title=f"Search Results for '{keyword}' on {provider}", border_style="green")
            search_table.add_column("No.", style="bold green", width=4, justify="right")
            search_table.add_column("Model ID", style="bold white")

            search_map: dict[str, str] = {}
            for s_idx, m_id in enumerate(matches[:15], start=1):
                s_key = str(s_idx)
                search_map[s_key] = m_id
                active_marker = " [green](current)[/green]" if m_id == current_model else ""
                search_table.add_row(s_key, f"{m_id}{active_marker}")

            console.print(search_table)
            if len(matches) > 15:
                console.print(f"[dim]Showing top 15 of {len(matches)} matching models.[/dim]")

            s_choice = Prompt.ask(
                "Pick model number (or 0 to cancel)",
                choices=list(search_map.keys()) + ["0"],
                default="1",
                console=console,
            )
            if s_choice in search_map:
                return search_map[s_choice]
            continue
