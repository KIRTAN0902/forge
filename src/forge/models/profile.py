"""Configuration-facing model types: providers, model profiles, and role bindings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.models.capabilities import ModelCapabilities, OptimizationConfig
from forge.models.types import GenerationParams

ROLES = ("default", "orchestrator", "planner", "executor", "qa")


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(_Config):
    type: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the API key."
    )
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    reasoning_style: Literal["openai", "openrouter", "none"] = Field(
        default="openai",
        description="How GenerationParams.reasoning_effort is sent on the wire.",
    )
    stream_usage: bool = Field(default=True, description="Request usage in the final stream chunk.")


BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        reasoning_style="openrouter",
        headers={"X-Title": "Forge"},
    ),
    "nvidia": ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        reasoning_style="openai",
    ),
    "ollama": ProviderConfig(base_url="http://localhost:11434/v1", reasoning_style="none"),
}


class Pricing(_Config):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class ModelProfile(_Config):
    """One configured model: which provider serves it, what it can do, how to drive it."""

    provider: str
    model: str = Field(description="Model id as the provider names it.")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    params: GenerationParams = Field(default_factory=GenerationParams)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    pricing: Pricing | None = None
    detect_capabilities: bool = Field(
        default=True,
        description="Ask the provider about the model and fill capabilities not set here.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_spec_shape(cls, data: Any) -> Any:
        # The spec writes `name:` for the model id; accept it as an alias for `model:`.
        if isinstance(data, dict) and "name" in data and "model" not in data:
            data = {**data, "model": data["name"]}
            data.pop("name")
        return data


class RoleBinding(_Config):
    """Binds a role (planner, executor, ...) to a model profile with role-specific tuning.

    The same model can be driven differently per role, e.g. a cooler temperature for
    the executor and a higher reasoning effort for the planner.
    """

    model: str
    params: GenerationParams = Field(default_factory=GenerationParams)
    optimization: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"model": data}
        return data
