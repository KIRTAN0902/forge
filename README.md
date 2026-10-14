# Forge

A CLI-first, model-agnostic, multi-agent coding harness. Forge doesn't train or replace
a model. It builds the environment around any capable model (Llama, Qwen, GLM, GPT,
Claude, ...) so that **the same model produces better software through Forge than when
used directly**.

See [`FORGE_SPECIFICATION.md`](FORGE_SPECIFICATION.md) for the product specification and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the implemented design and phase status.

---

## 🚀 Quick Install

No Node.js or NPM required. Install Forge globally in one command:

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/KIRTAN0902/forge/main/install.ps1 | iex
```

### Linux & macOS (Bash)
```bash
curl -fsSL https://raw.githubusercontent.com/KIRTAN0902/forge/main/install.sh | bash
```

### Or with `uv` (recommended package manager)
```bash
uv tool install --force git+https://github.com/KIRTAN0902/forge.git
```

### Zero-Install Run (`uvx` - Native Python `npx`)
Run Forge immediately without installing anything; always executes the latest commit on `main`:
```bash
uvx --from git+https://github.com/KIRTAN0902/forge.git@main --refresh forge chat
```

---

## 🔄 Instant Updates from GitHub

Forge is directly wired to GitHub. Whenever new changes or commits are pushed to `main`, you can update Forge instantly:

```bash
forge update
```

*(This pulls the latest commit from `https://github.com/KIRTAN0902/forge` and installs it in ~2 seconds.)*

---

## 🔑 Authentication & Providers

Set your API key interactively for any supported provider (NVIDIA NIM, OpenRouter, Anthropic, OpenAI, etc.):

```bash
forge login
```

Or configure via environment variables:
```bash
# NVIDIA NIM (Default / Fast inference)
export NVIDIA_API_KEY="nvapi-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# OpenAI / Anthropic
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 🎭 Multi-Role Agent Assignments

Forge allows you to configure different models for different agent roles (e.g. powerful frontier models for planning/architecture, fast models for execution, specialized models for QA), all with the same or different API keys:

```bash
# Interactive setup wizard
forge roles config

# List current agent role assignments
forge roles list

# Set a specific role to a model
forge roles set planner meta/llama-3.1-70b-instruct --provider nvidia
forge roles set executor qwen/qwen-2.5-coder-32b-instruct --provider openrouter
forge roles set reviewer anthropic/claude-3-7-sonnet --provider openrouter
```

---

## 💻 Quick Usage

```bash
# Start the interactive coding chat
forge chat

# Run an autonomous task end-to-end
forge run "Add health check endpoint and integration test"

# Inspect workflow status and tasks
forge status

# View git diff of changes made by Forge
forge diff

# Rollback changes if needed
forge rollback
```

---

## 🛠️ Development

```bash
uv sync                 # create .venv with runtime + dev dependencies
uv run pytest           # unit + integration tests (live model tests are opt-in)
uv run ruff check .     # lint
uv run mypy             # strict type checking
uv run forge --help     # CLI
```
