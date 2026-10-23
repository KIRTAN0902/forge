"""Provider-Grade API Usage & Cost Telemetry Dashboard for Forge.

Tracks immutable usage entries in ``~/.forge/usage_ledger.json`` across all LLM
calls and generates rich accounting & credit dashboards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.config.settings import default_config_dir
from forge.core.atomic import atomic_write_text

# Pricing in USD per 1,000,000 tokens: (input_price_per_million, output_price_per_million)
PRICING_TABLE: dict[str, tuple[float, float]] = {
    # Gemini
    "gemini/gemini-2.5-pro": (1.25, 5.00),
    "gemini/gemini-2.5-flash": (0.075, 0.30),
    "gemini/gemini-2.0-flash": (0.10, 0.40),
    "gemini/gemini-1.5-pro": (1.25, 5.00),
    "gemini/gemini-1.5-flash": (0.075, 0.30),
    "gemini": (0.15, 0.60),
    # OpenRouter models
    "openrouter/anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "openrouter/google/gemini-2.5-pro": (1.25, 5.00),
    "openrouter/google/gemini-2.5-flash": (0.10, 0.40),
    "openrouter/openai/gpt-4o": (2.50, 10.00),
    "openrouter/deepseek/deepseek-chat": (0.14, 0.28),
    "openrouter/qwen/qwen-2.5-coder-32b-instruct": (0.18, 0.18),
    "openrouter": (1.00, 3.00),
    # OpenAI
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1": (2.50, 10.00),
    "openai/o3-mini": (1.10, 4.40),
    "openai/o1": (15.00, 60.00),
    "openai": (2.50, 10.00),
    # Anthropic
    "anthropic/claude-3-5-sonnet-latest": (3.00, 15.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-5-haiku-latest": (0.80, 4.00),
    "anthropic/claude-3-opus-latest": (15.00, 75.00),
    "anthropic": (3.00, 15.00),
    # DeepSeek
    "deepseek/deepseek-chat": (0.14, 0.28),
    "deepseek/deepseek-reasoner": (0.55, 2.19),
    "deepseek": (0.14, 0.28),
    # Groq
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "groq/mixtral-8x7b-32768": (0.24, 0.24),
    "groq": (0.20, 0.50),
    # NVIDIA
    "nvidia/meta/llama-3.1-70b-instruct": (0.70, 0.90),
    "nvidia/meta/llama-3.1-8b-instruct": (0.20, 0.20),
    "nvidia": (0.50, 0.80),
    # Ollama (Local - Free)
    "ollama": (0.0, 0.0),
    # Custom
    "custom": (0.0, 0.0),
}


def default_ledger_path() -> Path:
    """Return persistent global usage ledger path (~/.forge/usage_ledger.json)."""
    return default_config_dir() / "usage_ledger.json"


def calculate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate estimated cost in USD based on model pricing table."""
    prov = provider.strip().lower()
    mod = model.strip().lower()

    if prov == "ollama" or "ollama" in prov:
        return 0.0

    key = f"{prov}/{mod}"
    if key in PRICING_TABLE:
        in_rate, out_rate = PRICING_TABLE[key]
    else:
        # Check partial match on model
        matched = None
        for tbl_key, rates in PRICING_TABLE.items():
            if tbl_key.startswith(f"{prov}/") and mod in tbl_key:
                matched = rates
                break
        if matched is not None:
            in_rate, out_rate = matched
        elif prov in PRICING_TABLE:
            in_rate, out_rate = PRICING_TABLE[prov]
        else:
            in_rate, out_rate = (0.50, 1.50)

    cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
    return round(cost, 6)


@dataclass
class UsageEntry:
    """An immutable record of a single LLM invocation."""

    timestamp: str
    session_id: str
    provider: str
    model: str
    stage: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageEntry:
        return cls(
            timestamp=str(data.get("timestamp", "")),
            session_id=str(data.get("session_id", "default")),
            provider=str(data.get("provider", "unknown")),
            model=str(data.get("model", "unknown")),
            stage=str(data.get("stage", "anvil")),
            request_count=int(data.get("request_count", 1)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cached_tokens=int(data.get("cached_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
        )


class UsageTracker:
    """Manages persistent ledger recording and multi-dimensional aggregation."""

    def __init__(self, ledger_path: Path | None = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else default_ledger_path()
        self._entries: list[UsageEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.ledger_path.is_file():
            self._entries = []
            return
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._entries = [
                    UsageEntry.from_dict(item) for item in data if isinstance(item, dict)
                ]
        except Exception:
            self._entries = []

    def _save(self) -> None:
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([e.to_dict() for e in self._entries], indent=2)
            atomic_write_text(self.ledger_path, payload)
        except Exception:
            pass

    def record_call(
        self,
        *,
        provider: str,
        model: str,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
        cached_tokens: int = 0,
        session_id: str = "default",
        timestamp: str | None = None,
    ) -> UsageEntry:
        """Record an immutable usage entry to memory and disk."""
        if not timestamp:
            timestamp = datetime.now(UTC).isoformat()
        total_tokens = prompt_tokens + completion_tokens
        cost = calculate_cost(provider, model, prompt_tokens, completion_tokens)

        entry = UsageEntry(
            timestamp=timestamp,
            session_id=session_id,
            provider=provider.lower(),
            model=model,
            stage=stage.lower(),
            request_count=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=cost,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def get_entries(
        self,
        session_id: str | None = None,
        time_window: str = "all",
    ) -> list[UsageEntry]:
        """Filter entries by session and optional time window (all | today | session)."""
        filtered = list(self._entries)
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]
        if time_window == "today":
            today_str = datetime.now(UTC).strftime("%Y-%m-%d")
            filtered = [e for e in filtered if e.timestamp.startswith(today_str)]
        elif time_window not in {"all", "session"}:
            raise ValueError(
                f"unknown usage time window {time_window!r}; use 'all', 'today', or 'session'"
            )
        return filtered

    def aggregate_by_model(
        self, entries: list[UsageEntry] | None = None
    ) -> list[dict[str, Any]]:
        """Return per-model summary sorted by total cost descending."""
        data = entries if entries is not None else self._entries
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for e in data:
            key = (e.provider, e.model)
            if key not in groups:
                groups[key] = {
                    "provider": e.provider,
                    "model": e.model,
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                    "total_latency": 0.0,
                }
            g = groups[key]
            g["requests"] += e.request_count
            g["prompt_tokens"] += e.prompt_tokens
            g["completion_tokens"] += e.completion_tokens
            g["total_tokens"] += e.total_tokens
            g["estimated_cost"] += e.estimated_cost_usd
            g["total_latency"] += e.latency_ms

        results = list(groups.values())
        results.sort(key=lambda x: (x["estimated_cost"], x["total_tokens"]), reverse=True)
        return results

    def aggregate_by_provider(
        self, entries: list[UsageEntry] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return per-provider summary."""
        data = entries if entries is not None else self._entries
        providers: dict[str, dict[str, Any]] = {}
        for e in data:
            p = e.provider
            if p not in providers:
                providers[p] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                    "total_latency": 0.0,
                }
            item = providers[p]
            item["requests"] += e.request_count
            item["prompt_tokens"] += e.prompt_tokens
            item["completion_tokens"] += e.completion_tokens
            item["total_tokens"] += e.total_tokens
            item["estimated_cost"] += e.estimated_cost_usd
            item["total_latency"] += e.latency_ms

        for item in providers.values():
            reqs = item["requests"]
            item["avg_latency"] = (item["total_latency"] / reqs) if reqs > 0 else 0.0

        return providers

    def aggregate_by_stage(
        self, entries: list[UsageEntry] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return per-stage summary (anvil / hammer / temper)."""
        data = entries if entries is not None else self._entries
        stages: dict[str, dict[str, Any]] = {
            "anvil": {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
            "hammer": {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
            "temper": {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
        }
        for e in data:
            stg = e.stage if e.stage in stages else "anvil"
            stages[stg]["requests"] += e.request_count
            stages[stg]["prompt_tokens"] += e.prompt_tokens
            stages[stg]["completion_tokens"] += e.completion_tokens
            stages[stg]["total_tokens"] += e.total_tokens
            stages[stg]["estimated_cost"] += e.estimated_cost_usd
        return stages

    def render_dashboard_text(
        self,
        session_id: str | None = None,
        time_window: str = "all",
        live_credits: dict[str, Any] | None = None,
    ) -> str:
        """Render a formatted, provider-grade telemetry dashboard table."""
        entries = self.get_entries(session_id=session_id, time_window=time_window)
        by_model = self.aggregate_by_model(entries)
        by_prov = self.aggregate_by_provider(entries)
        by_stage = self.aggregate_by_stage(entries)

        tot_req = sum(e.request_count for e in entries)
        tot_prompt = sum(e.prompt_tokens for e in entries)
        tot_comp = sum(e.completion_tokens for e in entries)
        tot_tokens = tot_prompt + tot_comp
        tot_cost = sum(e.estimated_cost_usd for e in entries)

        lines: list[str] = []
        lines.append(f"FORGE // API USAGE & COST TELEMETRY DASHBOARD [{time_window.upper()}]")
        lines.append("═" * 105)

        # 1. Account & Provider Summary
        lines.append("ACCOUNT & PROVIDER SUMMARY:")
        for prov_name, stats in by_prov.items():
            reqs = stats["requests"]
            c = stats["estimated_cost"]
            t = stats["total_tokens"]
            lat = stats.get("avg_latency", 0.0)
            credit_info = ""
            if live_credits and prov_name in live_credits:
                info = live_credits[prov_name]
                if isinstance(info, dict) and "limit_remaining" in info:
                    credit_info = f" │ Remaining Credits: ${info['limit_remaining']:.2f}"
            summary_item = (
                f"  • {prov_name:<12} Requests: {reqs:<4} │ Tokens: {t:>8,} │ "
                f"Avg Latency: {lat:>5.0f}ms │ Est. Cost: ${c:.4f}{credit_info}"
            )
            lines.append(summary_item)
        if not by_prov:
            lines.append("  (No API calls recorded in this time window)")
        lines.append("")

        # 2. Per-Model Breakdown Table
        header_row = (
            f"{'PROVIDER':<12} {'MODEL':<30} {'REQUESTS':>10} {'INPUT TOKENS':>14} "
            f"{'OUTPUT TOKENS':>15} {'TOTAL TOKENS':>14} {'EST. COST':>11}"
        )
        lines.append(header_row)
        lines.append("─" * 105)
        for m in by_model:
            model_row = (
                f"{m['provider']:<12} {m['model']:<30} {m['requests']:>10} "
                f"{m['prompt_tokens']:>14,} {m['completion_tokens']:>15,} "
                f"{m['total_tokens']:>14,} ${m['estimated_cost']:>10.4f}"
            )
            lines.append(model_row)
        if not by_model:
            lines.append("  (No model invocations found)")
        lines.append("─" * 105)
        tot_row = (
            f"{'TOTAL':<43} {tot_req:>10} {tot_prompt:>14,} {tot_comp:>15,} "
            f"{tot_tokens:>14,} ${tot_cost:>10.4f}"
        )
        lines.append(tot_row)
        lines.append("")

        # 3. Per-Stage Breakdown
        lines.append("PER-STAGE ALLOCATION:")
        anvil_s = (
            f"  • Anvil  (Orchestrator) : {by_stage['anvil']['requests']} reqs, "
            f"{by_stage['anvil']['total_tokens']:,} tokens "
            f"(${by_stage['anvil']['estimated_cost']:.4f})"
        )
        hammer_s = (
            f"  • Hammer (Executor)     : {by_stage['hammer']['requests']} reqs, "
            f"{by_stage['hammer']['total_tokens']:,} tokens "
            f"(${by_stage['hammer']['estimated_cost']:.4f})"
        )
        temper_s = (
            f"  • Temper (QA Gatekeeper): {by_stage['temper']['requests']} reqs, "
            f"{by_stage['temper']['total_tokens']:,} tokens "
            f"(${by_stage['temper']['estimated_cost']:.4f})"
        )
        lines.append(anvil_s)
        lines.append(hammer_s)
        lines.append(temper_s)

        return "\n".join(lines)


# Singleton
_GLOBAL_TRACKER: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    """Return the global usage tracker singleton."""
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = UsageTracker()
    return _GLOBAL_TRACKER


async def fetch_openrouter_credits(api_key: str) -> dict[str, Any] | None:
    """Query OpenRouter live auth/key credit balance endpoint."""
    if not api_key:
        return None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "usage": data.get("usage", 0.0),
                    "limit": data.get("limit"),
                    "limit_remaining": data.get("limit_remaining"),
                    "is_free_tier": data.get("is_free_tier", False),
                }
    except Exception:
        pass
    return None
