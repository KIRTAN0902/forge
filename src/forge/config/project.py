"""Project initialization (``forge init``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge.config.loader import CONFIG_FILENAME

CONFIG_TEMPLATE = """\
# Forge project configuration.
# Environment variables override any value: FORGE__<SECTION>__<KEY>=value
version: 1

project:
  name: {name}
  data_dir: .forge        # workflow database, logs (keep out of version control)

# Built-in providers: openai, openrouter (OPENROUTER_API_KEY), ollama (localhost).
# Declare others (vLLM, LM Studio, GLM, Qwen, DeepSeek...) under `providers:`:
# providers:
#   my-vllm:
#     base_url: http://localhost:8000/v1
#     api_key_env: VLLM_API_KEY
#
# Model profiles. Capabilities are auto-detected where the provider reports them;
# anything set here overrides detection. Missing native features (tool calling,
# JSON-schema output) fall back to prompt-level protocols automatically.
# models:
#   local-qwen:
#     provider: ollama
#     model: qwen2.5:7b
#     capabilities: {{context_window: 8192}}   # match OLLAMA_CONTEXT_LENGTH
#   glm:
#     provider: openrouter
#     model: z-ai/glm-4.6
#     params: {{temperature: 0.2}}
#
# Which model plays which role. The same model can be tuned differently per role.
# roles:
#   default: local-qwen
#   planner: {{model: glm, params: {{reasoning_effort: high}}}}
#   executor: {{model: glm, params: {{temperature: 0.1}}}}

limits:
  max_iterations: 10      # hard cap on plan/execute/QA cycles
  max_same_failure: 3     # block when the same failure repeats this many times
  max_wall_time_s: 3600
  max_total_tokens: null
  max_cost_usd: null
  max_tool_calls_per_task: 150

policy:
  approval_threshold: high          # low | medium | high | critical
  permissions: [read, write, execute, vcs]   # add 'network' to allow outbound access
  # protected_paths / secret_paths / forbidden_commands have safe built-in defaults;
  # set them here to replace those defaults (see `forge config` for current values).

tools:
  shell: auto             # auto | bash | sh | powershell | cmd | /path/to/shell
  command_timeout_s: 300
  max_output_chars: 30000 # per tool result shown to the model
  env_passthrough: []     # secret-looking env vars commands may see (scrubbed by default)
  test_command: null      # override auto-detection, e.g. "uv run pytest -q"

logging:
  level: INFO
  json_file: true
"""


@dataclass
class InitResult:
    config_path: Path
    created: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    gitignore_updated: bool = False


def init_project(root: Path, *, name: str | None = None, force: bool = False) -> InitResult:
    root = root.resolve()
    config_path = root / CONFIG_FILENAME
    result = InitResult(config_path=config_path)

    if config_path.exists() and not force:
        result.skipped.append(config_path)
    else:
        config_path.write_text(
            CONFIG_TEMPLATE.format(name=_yaml_str(name or root.name)), encoding="utf-8"
        )
        result.created.append(config_path)

    data_dir = root / ".forge"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        result.created.append(data_dir)

    gitignore = root / ".gitignore"
    if (root / ".git").exists() or gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if ".forge/" not in existing.splitlines():
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            gitignore.write_text(f"{existing}{prefix}.forge/\n", encoding="utf-8")
            result.gitignore_updated = True
    return result


def _yaml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
