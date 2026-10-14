"""Role and multi-agent model configuration helpers.

Enables configuring:
1. One model + one API key for all agents.
2. Different models for different agents sharing the same API key / provider.
3. Different models for different agents using different API keys / providers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from forge.config.loader import CONFIG_FILENAME
from forge.config.schema import ForgeConfig
from forge.models.profile import BUILTIN_PROVIDERS, ROLES


@dataclass
class RoleSummary:
    """Status summary of a single role binding."""

    role: str
    profile: str
    provider: str
    model: str
    has_key: bool
    key_source: str


def get_roles_summary(
    config: ForgeConfig,
    key_lookup: Callable[[str], str | None] | None = None,
) -> list[RoleSummary]:
    """Inspect and return the configured status of every agent role."""
    result: list[RoleSummary] = []
    agent_roles = [r for r in ROLES if r != "default"] + ["default"]

    for role in agent_roles:
        # Determine model profile
        profile_name = "default"
        if role in config.roles:
            profile_name = config.roles[role].model
        elif "default" in config.roles:
            profile_name = config.roles["default"].model
        elif config.models:
            profile_name = next(iter(config.models))

        # Determine profile details
        profile = config.models.get(profile_name)
        if profile:
            prov = profile.provider
            m_id = profile.model
        else:
            prov = "unknown"
            m_id = "default"

        # Check API key status
        key_source = ""
        has_key = False
        prov_cfg = config.providers.get(prov) or BUILTIN_PROVIDERS.get(prov)
        if prov == "ollama":
            has_key = True
            key_source = "local"
        elif prov_cfg and prov_cfg.api_key:
            has_key = True
            key_source = "direct"
        elif prov_cfg and prov_cfg.api_key_env:
            key = key_lookup(prov) if key_lookup else os.environ.get(prov_cfg.api_key_env)
            if key:
                has_key = True
                key_source = f"${prov_cfg.api_key_env}"
            else:
                key_source = f"missing (${prov_cfg.api_key_env})"
        else:
            key_source = "none"

        result.append(
            RoleSummary(
                role=role,
                profile=profile_name,
                provider=prov,
                model=m_id,
                has_key=has_key,
                key_source=key_source,
            )
        )

    return result


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    config_file = path / CONFIG_FILENAME if path.is_dir() else path
    if config_file.is_file():
        try:
            data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_yaml_dict(path: Path, data: dict[str, Any]) -> Path:
    config_file = path / CONFIG_FILENAME if path.is_dir() else path
    config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_file


def save_single_model_preset(
    project_dir: Path,
    provider: str,
    model: str,
    profile_name: str = "default_model",
) -> Path:
    """Preset 1: Configure all agent roles to use a single model and provider."""
    data = _load_yaml_dict(project_dir)
    data["version"] = 1
    if "project" not in data:
        data["project"] = {"name": project_dir.resolve().name, "data_dir": ".forge"}

    models = data.get("models", {})
    if not isinstance(models, dict):
        models = {}

    models[profile_name] = {
        "provider": provider,
        "model": model,
        "capabilities": {
            "tool_calling": True,
            "structured_output": True,
            "streaming": True,
        },
        "detect_capabilities": False,
    }
    data["models"] = models

    data["roles"] = {
        "default": profile_name,
        "orchestrator": profile_name,
        "planner": profile_name,
        "executor": profile_name,
        "qa": profile_name,
    }

    return _save_yaml_dict(project_dir, data)


def save_multi_role_preset(
    project_dir: Path,
    role_configs: dict[str, tuple[str, str]],
) -> Path:
    """Preset 2 & 3: Configure specific models and providers for different roles.

    role_configs format:
        {
            "planner": (provider, model_name),
            "executor": (provider, model_name),
            "qa": (provider, model_name),
            "orchestrator": (provider, model_name),
        }
    """
    data = _load_yaml_dict(project_dir)
    data["version"] = 1
    if "project" not in data:
        data["project"] = {"name": project_dir.resolve().name, "data_dir": ".forge"}

    models = data.get("models", {})
    if not isinstance(models, dict):
        models = {}

    roles: dict[str, str] = {}

    for role_name, (provider, model) in role_configs.items():
        # Create a clean profile slug
        slug = f"{role_name}_model"
        models[slug] = {
            "provider": provider,
            "model": model,
            "capabilities": {
                "tool_calling": True,
                "structured_output": True,
                "streaming": True,
            },
            "detect_capabilities": False,
        }
        roles[role_name] = slug

    # If default not specified, use orchestrator or planner
    if "default" not in roles:
        roles["default"] = (
            roles.get("orchestrator") or roles.get("planner") or next(iter(roles.values()))
        )

    data["models"] = models
    data["roles"] = roles

    return _save_yaml_dict(project_dir, data)


def set_role_model(
    project_dir: Path,
    role: str,
    model: str,
    provider: str | None = None,
) -> Path:
    """Assign or update a single role's model and provider in forge.yaml."""
    if role not in ROLES:
        raise ValueError(f"Invalid role {role!r}. Expected one of: {', '.join(ROLES)}")

    data = _load_yaml_dict(project_dir)
    data["version"] = 1
    models = data.get("models", {})
    if not isinstance(models, dict):
        models = {}

    # Check if this model/provider already matches an existing profile
    target_profile: str | None = None
    for name, prof in models.items():
        if isinstance(prof, dict):
            m_val = prof.get("model")
            p_val = prof.get("provider")
            if m_val == model and (provider is None or p_val == provider):
                target_profile = name
                break

    if target_profile is None:
        target_profile = f"{role}_model"
        resolved_provider = provider or "nvidia"
        models[target_profile] = {
            "provider": resolved_provider,
            "model": model,
            "capabilities": {
                "tool_calling": True,
                "structured_output": True,
                "streaming": True,
            },
            "detect_capabilities": False,
        }

    data["models"] = models
    roles = data.get("roles", {})
    if not isinstance(roles, dict):
        roles = {}
    roles[role] = target_profile
    data["roles"] = roles

    return _save_yaml_dict(project_dir, data)
