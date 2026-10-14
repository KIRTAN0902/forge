# @kirtan0902/forge

> Autonomous, Model-Agnostic, Multi-Agent Software Engineering CLI Harness

Forge is an open-source autonomous coding harness with specialized agents for Planning, Execution, and QA. It supports single-model or multi-model architectures with providers like NVIDIA NIM, OpenRouter, OpenAI, and Ollama.

---

## Installation

### Via NPM (Global CLI)
```bash
npm install -g @kirtan0902/forge
```
Or run directly without installation using `npx`:
```bash
npx @kirtan0902/forge
```

### Via UV (Recommended for Python)
```bash
uv tool install git+https://github.com/KIRTAN0902/forge.git
```
Or run instantly with `uvx`:
```bash
uvx --from git+https://github.com/KIRTAN0902/forge.git forge
```

### Via Pipx
```bash
pipx install git+https://github.com/KIRTAN0902/forge.git
```

---

## Quick Start

### 1. Launch Interactive Agent Chat
```bash
forge
```

### 2. Run Autonomous Tasks
```bash
forge run "Add a new feature or fix a bug in src/" -y
```

### 3. Inspect & Configure Agent Roles
```bash
# View active role assignments
forge roles

# Interactively configure roles (one model or specialized multi-model)
forge roles config

# Assign a specific model to an agent role
forge roles set executor nvidia/nemotron-3.5-lightning-30b-a3b --provider nvidia
```

### 4. Ping Test Agents
```bash
forge models ping planner
forge models ping executor
forge models ping qa
```

---

## Publishing to NPM Registry

To make `@kirtan0902/forge` installable by anyone worldwide via `npm install -g @kirtan0902/forge`:
```bash
cd npm
npm login
npm publish --access public
```

---

## License
MIT
