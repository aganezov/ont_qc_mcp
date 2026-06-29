#!/usr/bin/env bash
# Run the CI lint tier locally BEFORE pushing. uv installs the *locked* dependency
# set — the same versions CI uses — so this reproduces the lint-and-test job exactly.
#
#   scripts/ci-local.sh            # locked deps (mirrors the lint-and-test job)
#   scripts/ci-local.sh --fast     # checks only, reuse the current uv env (no sync)
#   scripts/ci-local.sh --floors   # lowest allowed versions (mirrors the min-deps job)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv not found — see https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

mode="${1:-}"
case "$mode" in
  --floors)
    echo "==> floors: lowest-direct resolution into a 3.10 venv (mirrors min-deps)"
    uv venv --python 3.10 >/dev/null || { echo "❌ uv venv failed"; exit 1; }
    uv pip install --resolution lowest-direct -e ".[all]" || { echo "❌ floor install failed"; exit 1; }
    ;;
  --fast)
    echo "==> fast: reusing the current uv environment (no sync)"
    ;;
  "")
    echo "==> uv sync --all-extras --locked  (exact locked deps — same as CI)"
    uv sync --all-extras --locked || { echo "❌ uv sync failed (is uv.lock in sync with pyproject.toml?)"; exit 1; }
    ;;
  *)
    echo "❌ unknown option: $mode (use --fast or --floors)"; exit 1 ;;
esac

fail=0
# --floors/--fast run in envs we don't want uv to re-sync from the lock; default just synced.
run() { case "$mode" in --floors | --fast) uv run --no-sync "$@" ;; *) uv run "$@" ;; esac; }
step() { echo; echo "==> $*"; run "$@" || { echo "   ^ FAILED"; fail=1; }; }

if [ "$mode" = "--floors" ]; then
  step pytest -m "not integration and not igv_integration" -q # min-deps runs only the unit suite
else
  # Mirrors the CI lint-and-test job.
  step ruff check .
  step mypy ont_qc_mcp tests
  step bandit -q -r ont_qc_mcp -x tests
  # CVE-2025-71176 is a pytest-only (test) advisory; see the ci.yml note. Suppress narrowly.
  step pip-audit --ignore-vuln CVE-2025-71176
  step pytest -q
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "❌ local CI FAILED — fix before pushing"
  exit 1
fi
echo "✅ local CI passed"
