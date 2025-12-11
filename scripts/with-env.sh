#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (this script is in scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prepend external toolchains (conda CLI tools + cargo nanoq) and activate venv
export PATH="/Users/saganezov/miniforge3/envs/ont-qc-mcp/bin:/Users/saganezov/.cargo/bin:${PATH}"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "Warning: .venv not found under ${ROOT_DIR}; continuing without venv" >&2
fi

cd "${ROOT_DIR}"
exec "$@"
