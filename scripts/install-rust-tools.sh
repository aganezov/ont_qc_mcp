#!/usr/bin/env bash
set -euo pipefail

# Install into the activated toolchain, leaving ~/.cargo/bin unchanged.
: "${CONDA_PREFIX:?Activate the environment from environment.yml first}"
export RUSTC="${CONDA_PREFIX}/bin/rustc"
export LIBCLANG_PATH="${CONDA_PREFIX}/lib"
# Find Conda's compression/TLS libraries when the installed binaries run.
export CARGO_ENCODED_RUSTFLAGS="-C"$'\x1f'"link-arg=-Wl,-rpath,${CONDA_PREFIX}/lib"

"${CONDA_PREFIX}/bin/cargo" install --locked --root "$CONDA_PREFIX" \
  --git https://github.com/wdecoster/chopper.git \
  --rev 290608816ffb78b3c5f963567487a042ecc7de4d chopper
"${CONDA_PREFIX}/bin/cargo" install --locked --root "$CONDA_PREFIX" \
  --git https://github.com/wdecoster/cramino.git \
  --rev 67ae11739e62eaebbf61d2577b883e21460468a3 cramino
"${CONDA_PREFIX}/bin/cargo" install --locked --root "$CONDA_PREFIX" \
  --version '=0.10.0' nanoq

# Fail installation if a cached or newly built executable has the wrong version.
for expected in 'chopper 0.14.0' 'cramino 1.4.1' 'nanoq 0.10.0'; do
  tool="${expected%% *}"
  actual="$("${CONDA_PREFIX}/bin/${tool}" --version)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $expected; found $actual" >&2
    exit 1
  fi
  echo "$actual"
done
