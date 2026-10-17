"""Resolves roles to models and builds adapters.

The registry is the only place that knows which provider serves which model and how a
role is bound to a profile. Agents ask for ``await registry.adapter("planner")`` and
receive a ready :class:`ModelAdapter`; changing ``forge.yaml`` swaps the model.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from forge.core.errors import ConfigError
from forge.core.logging import get_logger
from forge.events.bus import EventBus
from forge.models.adapter import ModelAdapter
from forge.models.capabilities import ModelCapabilities
from forge.models.profile import BUILTIN_PROVIDERS, ModelProfile, ProviderConfig, RoleBinding
from forge.models.provider import ModelProvider, ProviderError
from forge.models.providers.openai_compat import OpenAICompatibleProvider
from forge.models.usage import UsageTracker

log = get_logger("models.registry")

ProviderFactory = Callable[[str, ProviderConfig, str | None], ModelProvider]


def _openai_compatible(name: str, config: ProviderConfig, api_key: str | None) -> ModelProvider:
    return OpenAICompatibleProvider(config, name=name, api_key=api_key)


class ModelRegistry:
    def __init__(
        self,
        *,
        providers: Mapping[str, ProviderConfig] | None = None,
        profiles: Mapping[str, ModelProfile] | None = None,
        roles: Mapping[str, RoleBinding | str] | None = None,
        env: Mapping[str, str] | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.provider_configs: dict[str, ProviderConfig] = {
            **BUILTIN_PROVIDERS,
            **(providers or {}),
        }
        self.profiles: dict[str, ModelProfile] = dict(profiles or {})
        self.roles: dict[str, RoleBinding] = {
            k: (v if isinstance(v, RoleBinding) else RoleBinding(model=v))
            for k, v in (roles or {}).items()
        }
        self.tracker = tracker or UsageTracker()
        self._env = os.environ if env is None else env
        self._factories: dict[str, ProviderFactory] = {"openai_compatible": _openai_compatible}
        self._providers: dict[str, ModelProvider] = {}
        self._capabilities: dict[str, ModelCapabilities] = {}

    # -- registration -------------------------------------------------------------------

    def register_provider_type(self, type_name: str, factory: ProviderFactory) -> None:
        self._factories[type_name] = factory

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        """Use a ready provider instance under ``name`` (tests, embedding, plugins)."""
        self._providers[name] = provider

    def add_profile(self, name: str, profile: ModelProfile) -> None:
        self.profiles[name] = profile
        self._capabilities.pop(name, None)

    def configure_providers(self, providers: Mapping[str, ProviderConfig]) -> None:
        """Merge provider configs (e.g. after a live wizard) and drop stale instances."""
        self.provider_configs.update(providers)
        for name in providers:
            self._providers.pop(name, None)

    def bind_role(self, role: str, binding: RoleBinding | str) -> None:
        self.roles[role] = (
            binding if isinstance(binding, RoleBinding) else RoleBinding(model=binding)
        )

    # -- resolution ---------------------------------------------------------------------

    def profile(self, name: str) -> ModelProfile:
        try:
            return self.profiles[name]
        except KeyError:
            known = ", ".join(sorted(self.profiles)) or "(none configured)"
            raise ConfigError(f"unknown model profile {name!r}; known: {known}") from None

    def binding(self, role: str) -> RoleBinding:
        if role in self.roles:
            return self.roles[role]
        if "default" in self.roles:
            return self.roles["default"]
        if len(self.profiles) == 1:
            return RoleBinding(model=next(iter(self.profiles)))
        if not self.profiles:
            raise ConfigError(
                "no models configured: add a `models:` section to forge.yaml (see `forge init`)"
            )
        raise ConfigError(f"no model bound to role {role!r} and no `roles.default` is configured")

    def provider(self, name: str) -> ModelProvider:
        if name in self._providers:
            return self._providers[name]
        config = self.provider_configs.get(name)
        if config is None:
            raise ConfigError(f"unknown provider {name!r}")
        factory = self._factories.get(config.type)
        if factory is None:
            raise ConfigError(f"no provider implementation for type {config.type!r}")
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = self._env.get(config.api_key_env)
        if not api_key:
            try:
                from forge.config.settings import ForgeConfigManager, ProviderName

                pname = ProviderName(name)
                api_key = ForgeConfigManager().api_key_for(pname)
            except Exception:
                pass
        if api_key and name == "nvidia":
            clean = api_key.strip()
            if clean.startswith("vapi-"):
                clean = "n" + clean
            api_key = clean
        if not api_key and config.api_key_env:
            raise ConfigError(f"provider {name!r} needs an API key in ${config.api_key_env}")
        provider = factory(name, config, api_key)
        self._providers[name] = provider
        return provider

    async def capabilities(self, profile_name: str) -> ModelCapabilities:
        """Configured capabilities, filled in by provider detection where enabled."""
        if profile_name in self._capabilities:
            return self._capabilities[profile_name]
        profile = self.profile(profile_name)
        capabilities = profile.capabilities
        if profile.detect_capabilities:
            try:
                info = await self.provider(profile.provider).describe_model(profile.model)
            except ProviderError as exc:
                log.warning("capability detection failed for %s: %s", profile_name, exc)
                info = None
            if info is not None:
                capabilities = capabilities.overlay(info)
                log.info(
                    "detected capabilities for %s: %s",
                    profile_name,
                    capabilities.model_dump(exclude={"provider_features"}),
                )
        self._capabilities[profile_name] = capabilities
        return capabilities

    async def adapter(
        self,
        role: str = "default",
        *,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> ModelAdapter:
        binding = self.binding(role)
        return await self.adapter_for_profile(
            binding.model, role=role, binding=binding, events=events, workflow_id=workflow_id
        )

    async def adapter_for_profile(
        self,
        profile_name: str,
        *,
        role: str = "default",
        binding: RoleBinding | None = None,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> ModelAdapter:
        profile = self.profile(profile_name)
        provider = self.provider(profile.provider)
        capabilities = await self.capabilities(profile_name)
        params = profile.params.merged(binding.params if binding else None)
        optimization = profile.optimization.merged(binding.optimization if binding else None)
        return ModelAdapter(
            profile_name=profile_name,
            profile=profile,
            provider=provider,
            capabilities=capabilities,
            role=role,
            params=params,
            optimization=optimization,
            tracker=self.tracker,
            events=events,
            workflow_id=workflow_id,
        )

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()
