"""Context budgets and section packing.

The budget comes from the model's actual window, so the same role context is lean on an
8K local model and generous on a 1M frontier model ("use large windows when useful").

Packing decides inclusion by priority and renders in declaration order. Each section may
offer a cheaper ``fallback`` (an outline instead of a full file, a diffstat instead of a
diff); when the full version does not fit, the fallback is used before anything is
dropped, so the model always knows a relevant file exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from forge.context.tokens import estimate_tokens, truncate_to_tokens
from forge.models.capabilities import ModelCapabilities, OptimizationConfig

# Share of the usable window kept free for the conversation that follows (tool calls,
# results, reasoning). The executor iterates most, so it keeps the most headroom.
CONVERSATION_SHARE = {
    "planner": 0.30,
    "executor": 0.50,
    "qa": 0.40,
    "orchestrator": 0.25,
    "default": 0.40,
}
# Upper bounds per strategy: "targeted" avoids stuffing huge windows with marginal files.
STRATEGY_CAP = {"minimal": 6_000, "targeted": 32_000, "full": None}


@dataclass(frozen=True)
class ContextBudget:
    window: int
    reserved_output: int
    reserved_conversation: int
    context_tokens: int


class ContextBudgetManager:
    def budget(
        self,
        capabilities: ModelCapabilities,
        optimization: OptimizationConfig,
        role: str,
    ) -> ContextBudget:
        window = capabilities.context_window
        output = min(capabilities.max_output_tokens, window // 4)
        usable = window - output
        conversation = int(usable * CONVERSATION_SHARE.get(role, CONVERSATION_SHARE["default"]))
        available = max(usable - conversation, 256)
        cap = STRATEGY_CAP[optimization.context_strategy]
        if cap is not None:
            available = min(available, cap)
        return ContextBudget(
            window=window,
            reserved_output=output,
            reserved_conversation=conversation,
            context_tokens=available,
        )


@dataclass
class Section:
    key: str
    title: str
    content: str
    priority: int = 2  # 0 = must include (truncated if necessary); higher = less important
    fallback: str | None = None
    truncatable: bool = True
    files: tuple[str, ...] = ()  # files this section shows (for reporting)


@dataclass
class PackedSection:
    key: str
    title: str
    content: str
    tokens: int
    mode: str  # "full" | "fallback" | "truncated"
    files: tuple[str, ...] = ()


@dataclass
class PackResult:
    sections: list[PackedSection]
    omitted: list[str]
    used_tokens: int
    budget_tokens: int
    notes: list[str] = field(default_factory=list)


_HEADER_TOKENS = 6


def pack(sections: Sequence[Section], budget_tokens: int) -> PackResult:
    remaining = budget_tokens
    chosen: dict[str, PackedSection] = {}
    omitted: list[str] = []
    order = sorted(range(len(sections)), key=lambda i: (sections[i].priority, i))

    for index in order:
        section = sections[index]
        if not section.content.strip():
            continue
        full_cost = estimate_tokens(section.content) + _HEADER_TOKENS
        if full_cost <= remaining:
            chosen[section.key] = PackedSection(
                section.key, section.title, section.content, full_cost, "full", section.files
            )
            remaining -= full_cost
            continue
        if section.fallback:
            fallback_cost = estimate_tokens(section.fallback) + _HEADER_TOKENS
            if fallback_cost <= remaining:
                chosen[section.key] = PackedSection(
                    section.key,
                    section.title,
                    section.fallback,
                    fallback_cost,
                    "fallback",
                    section.files,
                )
                remaining -= fallback_cost
                continue
        must = section.priority == 0
        if section.truncatable and (remaining > 150 or must):
            room = max(remaining - _HEADER_TOKENS, 60)
            source = section.content if must or not section.fallback else section.fallback
            text = truncate_to_tokens(
                source, room, marker="\n... [truncated to fit the context budget]"
            )
            cost = estimate_tokens(text) + _HEADER_TOKENS
            chosen[section.key] = PackedSection(
                section.key, section.title, text, cost, "truncated", section.files
            )
            remaining -= cost
            continue
        omitted.append(section.title)

    ordered = [chosen[s.key] for s in sections if s.key in chosen]
    return PackResult(
        sections=ordered,
        omitted=omitted,
        used_tokens=budget_tokens - remaining,
        budget_tokens=budget_tokens,
    )
