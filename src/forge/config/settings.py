"""Persistent Provider & Model configuration engine for Forge.

This module owns the *user-level* configuration that lives in ``~/.forge/config.json``
(created with ``0600`` permissions), merged with any ``.env`` file in the current working
directory and the process environment. It is the single source of truth for:

* which providers Forge knows about and how to reach them (base URLs, default models),
* the API keys for each provider (masked for display),
* which concrete ``provider/model`` each of the three stages (Anvil/Hammer/Temper) uses.

The engine can auto-attach the best available live model at boot so the developer is never
greeted with ``(no live model attached)`` when a usable key — or a local Ollama server — is
within reach. Real HTTP execution lives in :mod:`forge.providers.client`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STAGES: tuple[str, str, str] = ("anvil", "hammer", "temper")
StageTarget = Literal["anvil", "hammer", "temper", "all"]

# Fast probe timeout for a local Ollama server (spec: ~400ms).
OLLAMA_PROBE_TIMEOUT_S = 0.4
OLLAMA_TAGS_PATH = "/api/tags"


class ProviderName(StrEnum):
    """Every provider Forge can speak to natively."""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    NVIDIA = "nvidia"
    DEEPSEEK = "deepseek"
    GROQ = "groq"


class ModelTier(StrEnum):
    """Rough capability class used to reason about which stage a model belongs in."""

    PROSTHETIC_SLM = "PROSTHETIC_SLM"
    AMPLIFIER_FRONTIER = "AMPLIFIER_FRONTIER"


def default_config_dir() -> Path:
    """Return the user-level Forge configuration directory (``~/.forge``)."""
    return Path.home() / ".forge"


def default_config_path() -> Path:
    """Return the persistent user-level config file path (``~/.forge/config.json``)."""
    return default_config_dir() / "config.json"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ProviderConfig(_Model):
    """How to reach one provider plus its most useful default model & catalog."""

    provider: ProviderName
    api_key: str | None = None
    base_url: str
    default_model: str
    available_models: list[str] = Field(default_factory=list)
    api_key_env: str | None = None

    @property
    def key_configured(self) -> bool:
        return bool(self.api_key) or (self.provider is ProviderName.OLLAMA)

    def masked_key(self) -> str:
        return mask_api_key(self.api_key)


def mask_api_key(api_key: str | None) -> str:
    """Mask an API key for safe display, e.g. ``AIza...****``."""
    if not api_key:
        return "NOT SET"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...****"


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")
_SEGMENT_RE = re.compile(r"[^a-z0-9.]+")
_SLM_HINTS = ("mini", "lite", "nano", "small", "tiny", "haiku")
_FRONTIER_HINTS = ("opus", "gpt-4.5", "o1", "o3", "sonnet", "405b", "70b", "72b")


def classify_model_tier(provider: ProviderName | str, model: str) -> ModelTier:
    """Classify a model as a prosthetic SLM or an amplifier frontier model.

    Local Ollama models and small (``<= 32B``), ``mini``/``lite`` models are treated as
    prosthetic single-model helpers; everything else is a frontier amplifier.
    """
    provider_value = provider.value if isinstance(provider, ProviderName) else str(provider)
    if provider_value == ProviderName.OLLAMA.value:
        return ModelTier.PROSTHETIC_SLM

    low = model.lower()
    # Match capability hints on whole segments so ``ge-mini`` never trips ``mini``.
    segments = {segment for segment in _SEGMENT_RE.split(low) if segment}
    if segments & set(_SLM_HINTS):
        return ModelTier.PROSTHETIC_SLM

    sizes = [float(match) for match in _SIZE_RE.findall(low)]
    if sizes and max(sizes) <= 32:
        return ModelTier.PROSTHETIC_SLM

    if any(hint in low for hint in _FRONTIER_HINTS):
        return ModelTier.AMPLIFIER_FRONTIER

    return ModelTier.AMPLIFIER_FRONTIER


class StageModelBinding(_Model):
    """A concrete ``provider/model`` bound to a stage, with per-stage tuning."""

    provider: ProviderName
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096

    @property
    def spec_string(self) -> str:
        """Canonical ``provider/model`` string (e.g. ``gemini/gemini-2.5-flash``)."""
        return f"{self.provider.value}/{self.model}"

    @property
    def model_tier(self) -> ModelTier:
        return classify_model_tier(self.provider, self.model)


# -- built-in provider catalogue ---------------------------------------------------------

_CURATED_MODELS: dict[ProviderName, list[str]] = {
    ProviderName.GEMINI: [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ],
    ProviderName.OPENROUTER: [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-chat",
        "qwen/qwen-2.5-coder-32b-instruct",
    ],
    ProviderName.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3-mini"],
    ProviderName.ANTHROPIC: [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ],
    ProviderName.OLLAMA: [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "llama3.1:8b",
        "deepseek-coder-v2:16b",
    ],
    ProviderName.NVIDIA: [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "qwen/qwen2.5-coder-32b-instruct",
    ],
    ProviderName.DEEPSEEK: ["deepseek-chat", "deepseek-reasoner"],
    ProviderName.GROQ: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}

DEFAULT_PROVIDERS: dict[ProviderName, ProviderConfig] = {
    ProviderName.GEMINI: ProviderConfig(
        provider=ProviderName.GEMINI,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.GEMINI]),
    ),
    ProviderName.OPENROUTER: ProviderConfig(
        provider=ProviderName.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-flash",
        api_key_env="OPENROUTER_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.OPENROUTER]),
    ),
    ProviderName.OPENAI: ProviderConfig(
        provider=ProviderName.OPENAI,
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.OPENAI]),
    ),
    ProviderName.ANTHROPIC: ProviderConfig(
        provider=ProviderName.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-5-sonnet-latest",
        api_key_env="ANTHROPIC_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.ANTHROPIC]),
    ),
    ProviderName.OLLAMA: ProviderConfig(
        provider=ProviderName.OLLAMA,
        base_url="http://localhost:11434",
        default_model="qwen2.5-coder:7b",
        api_key_env=None,
        available_models=list(_CURATED_MODELS[ProviderName.OLLAMA]),
    ),
    ProviderName.NVIDIA: ProviderConfig(
        provider=ProviderName.NVIDIA,
        base_url="https://integrate.api.nvidia.com/v1",
        default_model="meta/llama-3.1-70b-instruct",
        api_key_env="NVIDIA_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.NVIDIA]),
    ),
    ProviderName.DEEPSEEK: ProviderConfig(
        provider=ProviderName.DEEPSEEK,
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.DEEPSEEK]),
    ),
    ProviderName.GROQ: ProviderConfig(
        provider=ProviderName.GROQ,
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        available_models=list(_CURATED_MODELS[ProviderName.GROQ]),
    ),
}

# Priority order for auto-attachment when the user has not chosen explicitly.
AUTO_PRIORITY: tuple[tuple[str, ProviderName, str], ...] = (
    ("GEMINI_API_KEY", ProviderName.GEMINI, "gemini-2.5-flash"),
    ("OPENROUTER_API_KEY", ProviderName.OPENROUTER, "google/gemini-2.5-flash"),
    ("ANTHROPIC_API_KEY", ProviderName.ANTHROPIC, "claude-3-5-sonnet-latest"),
    ("OPENAI_API_KEY", ProviderName.OPENAI, "gpt-4o"),
    ("DEEPSEEK_API_KEY", ProviderName.DEEPSEEK, "deepseek-chat"),
    ("NVIDIA_API_KEY", ProviderName.NVIDIA, "meta/llama-3.1-70b-instruct"),
    ("GROQ_API_KEY", ProviderName.GROQ, "llama-3.3-70b-versatile"),
)


def parse_provider_model(spec: str) -> tuple[ProviderName, str]:
    """Parse ``provider/model`` — or infer the provider from a bare model name.

    Examples::

        gemini/gemini-2.5-pro              -> (GEMINI, "gemini-2.5-pro")
        openrouter/anthropic/claude-3.5... -> (OPENROUTER, "anthropic/claude-3.5...")
        ollama/qwen2.5-coder:7b            -> (OLLAMA, "qwen2.5-coder:7b")
        claude-3-5-sonnet-latest           -> (ANTHROPIC, "claude-3-5-sonnet-latest")

    Raises :class:`ValueError` for an empty spec.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty model spec")

    head, sep, rest = spec.partition("/")
    if sep and head in ProviderName._value2member_map_:
        return ProviderName(head), rest.strip()

    low = spec.lower()
    if low.startswith("claude") or low.startswith("anthropic/"):
        return ProviderName.ANTHROPIC, spec
    if low.startswith(("gpt-", "gpt4", "o1", "o3", "o4")):
        return ProviderName.OPENAI, spec
    if low.startswith("gemini"):
        return ProviderName.GEMINI, spec
    if low.startswith("deepseek"):
        return ProviderName.DEEPSEEK, spec
    if low.startswith("nvidia/") or low.startswith("meta/llama"):
        return ProviderName.NVIDIA, spec
    if ":" in spec and "/" not in spec:
        # Ollama tags look like ``qwen2.5-coder:7b``.
        return ProviderName.OLLAMA, spec
    # Anything of the form ``vendor/model`` is almost always an OpenRouter id.
    return ProviderName.OPENROUTER, spec


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` ``.env`` file (no shell expansion)."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class ForgeConfigManager:
    """Loads, mutates and persists the user-level provider & model configuration."""

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        probe_ollama: bool = True,
    ) -> None:
        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._probe_ollama = probe_ollama
        self._base_env: dict[str, str] = dict(os.environ if env is None else env)

        self.providers: dict[ProviderName, ProviderConfig] = {
            name: cfg.model_copy(deep=True) for name, cfg in DEFAULT_PROVIDERS.items()
        }
        self.bindings: dict[str, StageModelBinding] = {}
        self._api_keys: dict[str, str] = {}
        self._overrides: dict[str, dict[str, str]] = {}
        self._dotenv: dict[str, str] = {}

        self._load_file()
        self._dotenv = _read_dotenv(self._cwd / ".env")
        self._apply_environment()

    # -- environment & file loading ------------------------------------------------------

    def _env(self, key: str) -> str | None:
        if key in self._base_env:
            return self._base_env[key]
        return self._dotenv.get(key)

    def _apply_environment(self) -> None:
        for cfg in self.providers.values():
            if cfg.api_key or cfg.api_key_env is None:
                continue
            value = self._env(cfg.api_key_env)
            if value:
                cfg.api_key = value

    def _load_file(self) -> None:
        if not self.config_path.is_file():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        for name, key in (data.get("api_keys") or {}).items():
            if name in ProviderName._value2member_map_ and isinstance(key, str):
                provider = ProviderName(name)
                self.providers[provider].api_key = key
                self._api_keys[name] = key

        for name, override in (data.get("provider_overrides") or {}).items():
            if name not in ProviderName._value2member_map_ or not isinstance(override, dict):
                continue
            cfg = self.providers[ProviderName(name)]
            if isinstance(override.get("base_url"), str):
                cfg.base_url = override["base_url"]
            if isinstance(override.get("default_model"), str):
                cfg.default_model = override["default_model"]
            self._overrides[name] = {k: str(v) for k, v in override.items()}

        for stage, spec in (data.get("bindings") or {}).items():
            if stage in STAGES and isinstance(spec, str):
                with contextlib.suppress(ValueError):
                    provider, model = parse_provider_model(spec)
                    self.bindings[stage] = StageModelBinding(provider=provider, model=model)

    # -- persistence ---------------------------------------------------------------------

    def save(self) -> Path:
        """Persist the current configuration to ``config.json`` with ``0600`` permissions."""
        overrides: dict[str, dict[str, str]] = {}
        for name, cfg in self.providers.items():
            default = DEFAULT_PROVIDERS[name]
            entry: dict[str, str] = {}
            if cfg.base_url != default.base_url:
                entry["base_url"] = cfg.base_url
            if cfg.default_model != default.default_model:
                entry["default_model"] = cfg.default_model
            if entry:
                overrides[name.value] = entry

        data = {
            "version": 1,
            "api_keys": dict(self._api_keys),
            "provider_overrides": overrides,
            "bindings": {stage: b.spec_string for stage, b in self.bindings.items()},
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, sort_keys=True)
        self.config_path.write_text(payload, encoding="utf-8")
        self._restrict_permissions()
        self._overrides = overrides
        return self.config_path

    def _restrict_permissions(self) -> None:
        with contextlib.suppress(OSError, NotImplementedError):
            self.config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # -- provider helpers ----------------------------------------------------------------

    def provider(self, name: ProviderName | str) -> ProviderConfig:
        return self.providers[ProviderName(name)]

    def api_key_for(self, name: ProviderName | str) -> str | None:
        return self.providers[ProviderName(name)].api_key

    def masked_key(self, name: ProviderName | str) -> str:
        return self.providers[ProviderName(name)].masked_key()

    def set_api_key(self, provider: ProviderName | str, api_key: str) -> None:
        """Store a provider API key, export it to the environment, and auto-attach a default.

        Any stage that is currently unbounded (or bound to a provider without a usable key)
        is pointed at this provider's default model so the user immediately has a live model.
        """
        name = ProviderName(provider)
        cfg = self.providers[name]
        cfg.api_key = api_key
        self._api_keys[name.value] = api_key
        if cfg.api_key_env:
            os.environ[cfg.api_key_env] = api_key
            self._base_env[cfg.api_key_env] = api_key
        for stage in STAGES:
            current = self.bindings.get(stage)
            if current is None or not self._usable(current):
                self.bindings[stage] = StageModelBinding(
                    provider=name, model=cfg.default_model
                )
        self.save()

    def set_base_url(self, provider: ProviderName | str, url: str) -> None:
        """Update a provider's endpoint (custom vLLM / LM Studio / Ollama / proxy URL)."""
        name = ProviderName(provider)
        self.providers[name].base_url = url
        self.save()

    def set_stage_model(self, target: str, spec: str) -> StageModelBinding:
        """Bind ``spec`` (``provider/model`` or bare model) to a stage — or to ``all`` stages."""
        provider, model = parse_provider_model(spec)
        binding = StageModelBinding(provider=provider, model=model)
        stages = STAGES if target == "all" else (target,)
        for stage in stages:
            self.bindings[stage] = binding.model_copy(deep=True)
        self.save()
        return binding

    def binding_for(self, stage: str) -> StageModelBinding | None:
        return self.bindings.get(stage)

    # -- auto-attachment -----------------------------------------------------------------

    def _usable(self, binding: StageModelBinding) -> bool:
        if binding.provider is ProviderName.OLLAMA:
            return True
        return bool(self.providers[binding.provider].api_key)

    def _ollama_model(self) -> str | None:
        if not self._probe_ollama:
            return None
        cfg = self.providers[ProviderName.OLLAMA]
        try:
            import httpx

            url = cfg.base_url.rstrip("/") + OLLAMA_TAGS_PATH
            resp = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT_S)
            if resp.status_code != 200:
                return None
            models = [
                str(m["name"])
                for m in resp.json().get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]
        except Exception:
            return None
        if not models:
            return None
        # Prefer a coding/general model, else the first reported model.
        for candidate in models:
            low = candidate.lower()
            if "coder" in low or "codellama" in low or "qwen" in low or "llama" in low:
                return candidate
        return models[0]

    def _first_available(self) -> StageModelBinding | None:
        for env_var, provider, model in AUTO_PRIORITY:
            key = self.providers[provider].api_key or self._env(env_var)
            if key:
                self.providers[provider].api_key = key
                return StageModelBinding(provider=provider, model=model)
        ollama_model = self._ollama_model()
        if ollama_model:
            return StageModelBinding(provider=ProviderName.OLLAMA, model=ollama_model)
        return None

    def auto_resolve_bindings(self) -> StageModelBinding | None:
        """Attach the best available live model to every stage that has no usable binding.

        Returns the binding that was chosen when at least one stage was (re)bound, else
        ``None`` so the caller knows no live model is reachable.
        """
        fallback = self._first_available()
        chosen: StageModelBinding | None = None
        changed = False
        for stage in STAGES:
            current = self.bindings.get(stage)
            if current is not None and self._usable(current):
                continue
            if fallback is None:
                continue
            self.bindings[stage] = fallback.model_copy(deep=True)
            chosen = fallback
            changed = True
        return chosen if changed else None

    # -- display -------------------------------------------------------------------------

    def describe(self) -> str:
        """Multi-line, masked rendering of the full configuration (for ``/config``)."""
        lines = [f"Forge Configuration  ({self.config_path})", "", "Providers:"]
        width = max(len(name.value) for name in self.providers)
        for name, cfg in self.providers.items():
            status = cfg.masked_key() if cfg.key_configured else "NOT SET"
            lines.append(f"  {name.value:<{width}}  {status}  {cfg.base_url}")
        lines.append("")
        lines.append("Stage bindings:")
        for stage in STAGES:
            binding = self.bindings.get(stage)
            if binding is None:
                lines.append(f"  {stage:<7}: (no model attached)")
            else:
                lines.append(
                    f"  {stage:<7}: {binding.spec_string}  [{binding.model_tier.value}]"
                )
        return "\n".join(lines)


__all__ = [
    "AUTO_PRIORITY",
    "DEFAULT_PROVIDERS",
    "OLLAMA_PROBE_TIMEOUT_S",
    "STAGES",
    "ForgeConfigManager",
    "ModelTier",
    "ProviderConfig",
    "ProviderName",
    "StageModelBinding",
    "StageTarget",
    "classify_model_tier",
    "default_config_dir",
    "default_config_path",
    "mask_api_key",
    "parse_provider_model",
]
