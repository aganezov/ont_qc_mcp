# Contributing

Thanks for helping improve ONT QC MCP! This guide covers **setup and commands**;
[AGENTS.md](AGENTS.md) covers **how changes are shipped and reviewed**.

## Setup

- Python 3.10+.
- Install [uv](https://docs.astral.sh/uv/), then create the environment from the
  lockfile: `uv sync --all-extras`.
- Optional: install the wrapped CLIs (nanoq, chopper, cramino, mosdepth, samtools,
  bcftools), or point to them via env vars `NANOQ`, `CHOPPER`, `CRAMINO`, etc.

## Commands

- **Full local gate** (lint + types + tests, as CI runs them): `scripts/ci-local.sh`
  (`--floors` also runs the min-deps floor job).
- Or individually, through `uv run`:
  - Lint — `uv run ruff check .`
  - Format — `uv run ruff format` (check only: `uv run ruff format --check`)
  - Types — `uv run mypy ont_qc_mcp tests`
  - Unit tests — `uv run pytest`
  - Integration tests (require the CLIs) — `uv run pytest -m integration`

## Development tips

- Keep outputs JSON-first; prefer returning file paths for large artifacts.
- When adding a tool, update `flag_schemas.py`, the `app_server.py` tool registry, and
  add tests/fixtures.
- Update `CHANGELOG.md` for user-visible changes.
