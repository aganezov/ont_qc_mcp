# Contributing

Thanks for helping improve ONT QC MCP! This guide covers setup and common workflows.

## Setup
- Use Python 3.10+.
- Create a virtualenv in the repo root (`python -m venv .venv && source .venv/bin/activate`).
- Install deps: `pip install -e ".[all]"` (or `.[dev]` without matplotlib).
- Optional: install CLI tools (nanoq, chopper, cramino, mosdepth, samtools) or point to them via env vars `NANOQ`, `CHOPPER`, etc.

## Commands
- Run unit tests: `scripts/with-env.sh pytest`
- Run integration tests (requires CLIs): `scripts/with-env.sh pytest -m integration`
- Lint: `ruff check .`
- Type check: `mypy .`

## Development tips
- Use `scripts/with-env.sh <cmd>` to ensure consistent PATH/venv activation.
- Keep outputs JSON-first; prefer returning file paths for large artifacts.
- When adding new tools, update `flag_schemas.py`, `app_server.py` tool registry, and add tests/fixtures where possible.
- Please update `CHANGELOG.md` for user-visible changes.

