#!/usr/bin/env bash
# Run the CI "lint-and-test" gate locally BEFORE pushing.
#
# Default: a FRESH virtualenv with fresh dependency resolution, so it reproduces
# "works on my machine" failures caused by unpinned-dependency drift (e.g. a new
# numpy/mypy combination breaking the type check) exactly the way CI does.
# Pass --fast to reuse the current environment for a quick check instead.
#
#   scripts/ci-local.sh          # faithful to CI (fresh venv; slower)
#   scripts/ci-local.sh --fast   # quick check in the current environment
set -uo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" != "--fast" ]; then
  # CI requires Python >=3.10; the bare `python3` may be older (macOS ships 3.9).
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    p="$(command -v "$c" 2>/dev/null)" || continue
    v="$("$p" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" || continue
    case "$v" in 3.1[0-9] | 3.[2-9][0-9]) PY="$p"; break ;; esac
  done
  if [ -z "$PY" ]; then echo "❌ no Python >=3.10 found on PATH"; exit 1; fi
  echo "==> interpreter: $PY ($("$PY" --version 2>&1))"
  VENV="$(mktemp -d)/ci-venv"
  echo "==> fresh venv: $VENV"
  "$PY" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --quiet --upgrade pip
  echo "==> pip install -e .[all]  (fresh resolution — this is what catches drift)"
  python -m pip install --quiet -e ".[all]" || { echo "❌ install failed"; exit 1; }
fi

fail=0
step() { echo; echo "==> $*"; "$@" || { echo "   ^ FAILED"; fail=1; }; }

# Mirrors the steps of the CI lint-and-test job.
step ruff check .
step mypy ont_qc_mcp tests
step bandit -q -r ont_qc_mcp -x tests
# CVE-2025-71176 is a pytest-only (test) advisory; see the ci.yml note. Suppress narrowly.
step pip-audit --ignore-vuln CVE-2025-71176
step pytest -q

echo
if [ "$fail" -ne 0 ]; then
  echo "❌ local CI FAILED — fix before pushing"
  exit 1
fi
echo "✅ local CI passed"
