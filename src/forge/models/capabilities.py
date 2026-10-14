"""What a model can do, and how Forge should drive it."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.models.types import ModelInfo


class ModelCapabilities(BaseModel):
    """Capabilities Forge may rely on.

    Defaults are conservative so an unknown model still works (via prompt-based
    fallbacks); configuration or detection turns native features on. Forge prefers
    native features whenever they are available rather than reducing every model to
    the lowest common denominator.
    """

    model_config = ConfigDict(extra="forbid")

    context_window: int = Field(default=8192, ge=512)
    max_output_tokens: int = Field(default=4096, ge=1)
    tool_calling: bool = False
    parallel_tool_calls: bool = False
    structured_output: bool = Field(default=False, description="Native JSON-schema output.")
    json_mode: bool = Field(default=False, description="Native 'any valid JSON' output.")
    reasoning: bool = False
    streaming: bool = True
    vision: bool = False
    provider_features: dict[str, Any] = Field(default_factory=dict)

    def overlay(self, detected: ModelInfo) -> ModelCapabilities:
        """Return a copy where detected facts replace *unset* fields only.

        Values the user configured explicitly always win over detection.
        """
        explicit = self.model_fields_set
        updates: dict[str, Any] = {}
        mapping = {
            "context_window": detected.context_window,
            "max_output_tokens": detected.max_output_tokens,
            "tool_calling": detected.tool_calling,
            "structured_output": detected.structured_output,
            "json_mode": detected.json_mode,
            "reasoning": detected.reasoning,
            "vision": detected.vision,
        }
        for field, value in mapping.items():
            if value is not None and field not in explicit:
                updates[field] = value
        # parallel_tool_calls is deliberately never inferred: some endpoints reject the
        # parameter, and leaving it unset lets the server apply its own default.
        if (
            detected.structured_output
            and "json_mode" not in explicit
            and "json_mode" not in updates
        ):
            updates["json_mode"] = True
        return self.model_copy(update=updates)


class OptimizationConfig(BaseModel):
    """How Forge drives a model. ``auto`` means: native when capable, else fall back."""

    model_config = ConfigDict(extra="forbid")

    context_strategy: Literal["full", "targeted", "minimal"] = "targeted"
    tool_batching: Literal["sequential", "parallel", "adaptive"] = "adaptive"
    tool_call_strategy: Literal["auto", "native", "prompt"] = "auto"
    structured_output_strategy: Literal["auto", "native", "json_mode", "prompt"] = "auto"
    max_repair_attempts: int = Field(default=2, ge=0, le=5)

    def merged(self, override: dict[str, Any] | None) -> OptimizationConfig:
        if not override:
            return self.model_copy()
        return OptimizationConfig.model_validate({**self.model_dump(), **override})
