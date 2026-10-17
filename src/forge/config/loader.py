"""Configuration loading: defaults < forge.yaml < environment variables.

Environment overrides use ``FORGE__<SECTION>__<KEY>`` (double underscores separate
levels), e.g. ``FORGE__LIMITS__MAX_ITERATIONS=5``. Values are parsed as YAML scalars,
so ``true``, ``5`` and ``[read, write]`` all work.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from forge.config.schema import ForgeConfig
from forge.core.errors import ConfigError

CONFIG_FILENAME = "forge.yaml"
ENV_PREFIX = "FORGE__"


def find_config(start: Path) -> Path | None:
    """Search ``start`` and its parents for configuration.

    Checks .forge/config.json, .forge/config.yaml, or forge.yaml.
    """
    start = start.resolve()
    home = Path.home().resolve()
    for directory in (start, *start.parents):
        if directory == home:
            continue
        forge_dir = directory / ".forge"
        if (forge_dir / "config.json").is_file():
            return forge_dir / "config.json"
        if (forge_dir / "config.yaml").is_file():
            return forge_dir / "config.yaml"
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(
    start: Path | None = None,
    *,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ForgeConfig:
    """Load the effective configuration.

    ``path`` selects an explicit file; otherwise configuration is searched for from
    ``start`` (default: cwd) upward. The project root is the directory holding the
    config file, or ``start`` when there is none.
    """
    start = (start or Path.cwd()).resolve()
    source = path.resolve() if path is not None else find_config(start)

    data: dict[str, Any] = {}
    if source is not None:
        data = _read_yaml(source)

    _apply_env_overrides(data, os.environ if env is None else env)

    try:
        config = ForgeConfig.model_validate(data)
    except ValidationError as exc:
        where = str(source) if source else "defaults/environment"
        raise ConfigError(f"invalid configuration ({where}):\n{exc}") from exc

    if source is not None:
        config.root = source.parent.parent if source.parent.name == ".forge" else source.parent
    else:
        config.root = start
    config.source = source
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


def _apply_env_overrides(data: dict[str, Any], env: Mapping[str, str]) -> None:
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except yaml.YAMLError:
            node[parts[-1]] = value
