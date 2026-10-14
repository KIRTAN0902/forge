# Forge

A CLI-first, model-agnostic, multi-agent coding harness. Forge doesn't train or replace
a model. It builds the environment around any capable model (Llama, Qwen, GLM, GPT,
Claude, ...) so that **the same model produces better software through Forge than when
used directly**.

See [`FORGE_SPECIFICATION.md`](FORGE_SPECIFICATION.md) for the product specification and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the implemented design and phase status.

## Development

```bash
uv sync                 # create .venv with runtime + dev dependencies
uv run pytest           # unit + integration tests (live model tests are opt-in)
uv run ruff check .     # lint
uv run mypy             # strict type checking
uv run forge --help     # CLI
```
