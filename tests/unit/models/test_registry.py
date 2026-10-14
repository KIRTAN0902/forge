from __future__ import annotations

from pathlib import Path

import pytest

from forge.config import load_config
from forge.core.errors import ConfigError
from forge.models import (
    ModelCapabilities,
    ModelInfo,
    ModelProfile,
    ModelRegistry,
    ProviderError,
    RoleBinding,
)
from forge.models.providers.openai_compat import OpenAICompatibleProvider
from forge.models.providers.scripted import ScriptedProvider

YAML = """
providers:
  local-vllm:
    base_url: http://localhost:8000/v1
models:
  glm:
    provider: openrouter
    name: z-ai/glm-4.6          # spec-style `name:` accepted
    capabilities: {tool_calling: true}
    params: {temperature: 0.3}
    optimization: {context_strategy: targeted, tool_batching: adaptive}
  qwen:
    provider: local-vllm
    model: Qwen/Qwen2.5-Coder-32B-Instruct
roles:
  default: qwen
  planner: {model: glm, params: {reasoning_effort: high}}
  executor: {model: glm, params: {temperature: 0.0}, optimization: {max_repair_attempts: 4}}
"""


def write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "forge.yaml").write_text(text, encoding="utf-8")
    return tmp_path


class TestConfig:
    def test_parses_models_and_roles(self, tmp_path: Path) -> None:
        config = load_config(write(tmp_path, YAML), env={})
        assert config.models["glm"].model == "z-ai/glm-4.6"
        assert config.models["glm"].capabilities.tool_calling is True
        assert config.roles["default"] == RoleBinding(model="qwen")
        assert config.roles["planner"].params.reasoning_effort == "high"

    @pytest.mark.parametrize(
        ("snippet", "message"),
        [
            ("models:\n  m: {provider: nowhere, model: x}\n", "unknown provider"),
            ("models:\n  m: {provider: ollama, model: x}\nroles:\n  plannr: m\n", "unknown role"),
            ("models:\n  m: {provider: ollama, model: x}\nroles:\n  planner: zz\n", "zz"),
            ("models:\n  m: {provider: ollama, model: x, capabilities: {tools: true}}\n", "tools"),
        ],
    )
    def test_validation_errors(self, tmp_path: Path, snippet: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            load_config(write(tmp_path, snippet), env={})


def registry_from(tmp_path: Path, env: dict[str, str] | None = None) -> ModelRegistry:
    config = load_config(write(tmp_path, YAML), env={})
    return ModelRegistry(
        providers=config.providers, profiles=config.models, roles=config.roles, env=env or {}
    )


class TestResolution:
    async def test_role_bindings_and_param_layering(self, tmp_path: Path) -> None:
        registry = registry_from(tmp_path)
        registry.register_provider("openrouter", ScriptedProvider())
        executor = await registry.adapter("executor")
        assert executor.profile_name == "glm"
        assert executor.params.temperature == 0.0  # role overrides profile (0.3)
        assert executor.optimization.max_repair_attempts == 4
        assert executor.optimization.context_strategy == "targeted"
        planner = await registry.adapter("planner")
        assert planner.params.temperature == 0.3
        assert planner.params.reasoning_effort == "high"

    async def test_unbound_role_uses_default(self, tmp_path: Path) -> None:
        registry = registry_from(tmp_path)
        registry.register_provider("local-vllm", ScriptedProvider())
        qa = await registry.adapter("qa")
        assert qa.profile_name == "qwen"
        assert qa.role == "qa"

    def test_single_profile_needs_no_roles(self) -> None:
        registry = ModelRegistry(profiles={"only": ModelProfile(provider="ollama", model="m")})
        assert registry.binding("planner").model == "only"

    def test_no_models_is_a_clear_error(self) -> None:
        with pytest.raises(ConfigError, match="no models configured"):
            ModelRegistry().binding("planner")

    def test_missing_api_key(self, tmp_path: Path) -> None:
        registry = registry_from(tmp_path)
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            registry.provider("openrouter")

    async def test_builds_and_caches_real_provider(self, tmp_path: Path) -> None:
        registry = registry_from(tmp_path, env={"OPENROUTER_API_KEY": "sk"})
        provider = registry.provider("openrouter")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert registry.provider("openrouter") is provider
        assert provider.config.reasoning_style == "openrouter"
        await registry.aclose()


class TestCapabilityDetection:
    async def test_detection_fills_only_unset_fields(self) -> None:
        provider = ScriptedProvider(
            info=ModelInfo(id="m", context_window=128_000, tool_calling=False, vision=True)
        )
        registry = ModelRegistry(
            profiles={
                "p": ModelProfile(
                    provider="x", model="m", capabilities=ModelCapabilities(tool_calling=True)
                )
            }
        )
        registry.register_provider("x", provider)
        caps = await registry.capabilities("p")
        assert caps.context_window == 128_000  # detected
        assert caps.vision is True  # detected
        assert caps.tool_calling is True  # explicit config beats detection

    async def test_detection_can_be_disabled(self) -> None:
        registry = ModelRegistry(
            profiles={"p": ModelProfile(provider="x", model="m", detect_capabilities=False)}
        )
        registry.register_provider("x", ScriptedProvider(info=ModelInfo(id="m", vision=True)))
        assert (await registry.capabilities("p")).vision is False

    async def test_detection_failure_is_tolerated(self) -> None:
        class Failing(ScriptedProvider):
            async def describe_model(self, model: str) -> ModelInfo | None:
                raise ProviderError("listing down", status=503)

        registry = ModelRegistry(profiles={"p": ModelProfile(provider="x", model="m")})
        registry.register_provider("x", Failing())
        assert await registry.capabilities("p") == ModelCapabilities()
