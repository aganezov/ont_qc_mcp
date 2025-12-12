#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (this script lives in scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_CONDA_ENV="${MCP_CONDA_ENV:-ont-qc-mcp}"

# Helper: prepend a path if it exists and is not already present.
prepend_path() {
  local dir="$1"
  if [[ -d "$dir" ]] && [[ ":$PATH:" != *":$dir:"* ]]; then
    PATH="$dir:$PATH"
  fi
}

# Activate a conda environment if available.
activate_conda_env() {
  local env_name="$1"
  if ! command -v conda >/dev/null 2>&1; then
    echo "Warning: conda not found; skipping conda activation" >&2
    return
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || {
    echo "Warning: failed to source conda.sh; skipping conda activation" >&2
    return
  }

  if conda env list | awk '{print $1}' | grep -Eq "^${env_name}$"; then
    conda activate "${env_name}"
  else
    echo "Warning: conda env '${env_name}' not found; skipping conda activation" >&2
  fi
}

# Ensure conda env (for CLI tools) is first on PATH when present.
activate_conda_env "${TARGET_CONDA_ENV}"

# Optional hooks for external toolchains.
# - MCP_TOOLCHAIN_PATH: explicit directory containing nanoq/chopper/cramino/mosdepth/samtools
# - CONDA_PREFIX: prepend <conda>/bin when active
# - CARGO_HOME: prepend cargo-installed binaries (defaults to ~/.cargo if unset)
if [[ -n "${MCP_TOOLCHAIN_PATH:-}" ]]; then
  prepend_path "${MCP_TOOLCHAIN_PATH}"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  prepend_path "${CONDA_PREFIX}/bin"
fi

if [[ -n "${CARGO_HOME:-}" ]]; then
  prepend_path "${CARGO_HOME}/bin"
elif [[ -d "${HOME}/.cargo/bin" ]]; then
  prepend_path "${HOME}/.cargo/bin"
fi

# Activate local venv when present.
if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "Warning: .venv not found under ${ROOT_DIR}; continuing without venv" >&2
fi

cd "${ROOT_DIR}"
exec "$@"
