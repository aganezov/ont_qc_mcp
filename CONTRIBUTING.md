# Contributing

Thanks for helping improve ONT QC MCP! This guide covers **setup and commands**;
[AGENTS.md](AGENTS.md) covers **how changes are shipped and reviewed**.

## Setup

- Python 3.10+.
- CI tests Python 3.10–3.14. The minimum-dependency job uses 3.10; container integration uses 3.11.
- Install [uv](https://docs.astral.sh/uv/), then create the environment from the
  lockfile: `uv sync --all-extras`.
- For integration tests, install the [tested toolchain](docs/toolchain.md) from
  `environment.yml` and `scripts/install-rust-tools.sh`. Explicit paths can be
  supplied through `NANOQ`, `CHOPPER`, `CRAMINO`, and the other tool variables.

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

## Dependency audit updates

The CI lint tier and `scripts/ci-local.sh` install exact versions from `uv.lock`.
The dependency audit checks those versions against current advisory data, so an
unchanged commit can start failing when that data changes.

For a reported dependency with a published fix, use
`uv lock --upgrade-package <package>` and inspect the diff. Run
`scripts/ci-local.sh` to check the updated environment, then require green GitHub
checks and both advisory reviews before merging. An audit pass means no known
findings were reported for the audited packages at that time; it does not prove
that the application has no vulnerabilities.

The audit wrapper retries recognized temporary PyPI service failures only. It
fails on reported vulnerabilities and unknown errors. Keep those failure paths
enabled. Passing integration or minimum-dependency tests cannot substitute for
the audit because those jobs resolve dependencies separately from the lockfile.
