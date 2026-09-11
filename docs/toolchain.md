# Tested bioinformatics toolchain

CI and local development use the same Conda environment and Rust installer.

| Tool | Version | Installation |
| --- | --- | --- |
| nanoq | 0.10.0 | crates.io, exact version and packaged Cargo lockfile |
| chopper | 0.14.0 | upstream Git revision and Cargo lockfile |
| cramino | 1.4.1 | upstream Git revision and Cargo lockfile |
| mosdepth | 0.3.14 | Bioconda |
| samtools | 1.24 | Bioconda |
| bcftools | 1.24 | Bioconda |
| HTSlib | 1.24 | Bioconda |

The HTSlib pin applies to the Conda-linked tools. Cramino embeds
[HTSlib 1.19.1](https://github.com/rust-bio/hts-sys/tree/64f51cc9c649df98d4d85b49c3ce242efe4aa6e6/htslib)
through its locked Rust dependencies; installing Conda HTSlib does not replace
that embedded copy. [HTSlib 1.24](https://github.com/samtools/htslib/releases/tag/1.24)
removes experimental CRAM 4 support.

## Installation

Install Conda/Mamba and a native C/C++ compiler first. On macOS, use Xcode Command
Line Tools. On Debian/Ubuntu, the compiler and standard build tools are provided
by `build-essential`. The environment supplies Rust 1.90.0, libclang, CMake, Make,
pkg-config, and compression/TLS libraries.

From the repository root:

```bash
mamba env create --file environment.yml --strict-channel-priority
conda activate ont-qc-mcp
bash scripts/install-rust-tools.sh
export NANOQ="$CONDA_PREFIX/bin/nanoq"
uv sync --all-extras --locked
```

Use a fresh environment name with `--name` if `ont-qc-mcp` already exists. Activate
that name in the following commands. The installer writes binaries into the
active environment, leaving `~/.cargo/bin` unchanged. `NANOQ` explicitly selects
the new environment when another Cargo-installed nanoq is present.

The first source build takes several minutes. Re-running the installer keeps
already installed versions and verifies all three Rust tool versions.

## Source and cache policy

As of September 11, 2026, Bioconda provides chopper 0.13.0 and cramino 1.3.0.
Current upstream binaries do not cover Linux ARM64. The shared source-build
recipe provides the same versions on Linux x86_64, Linux ARM64, and macOS ARM64:

- [chopper 0.14.0](https://github.com/wdecoster/chopper/releases/tag/v0.14.0),
  revision `290608816ffb78b3c5f963567487a042ecc7de4d`.
- [cramino 1.4.1](https://github.com/wdecoster/cramino/releases/tag/1.4.1),
  revision `67ae11739e62eaebbf61d2577b883e21460468a3`.

Cargo's `--locked` retains upstream Rust dependency selections. Conda package
builds and native build dependencies can still change within the environment
specification; this is a tested version baseline, not a bit-for-bit build lock.

CI caches the compiled binaries and Cargo installation metadata separately from
the Conda environment. The cache key includes platform, architecture, runner
image version, the resolved native package list, and the installer contents.
The installer verifies versions after every restore.

## Validation

With the environment active and `NANOQ` exported as above:

```bash
uv run pytest -q -m integration tests/test_toolchain_integration.py tests/test_mosdepth_thresholds.py
uv run pytest -q -m integration
```

The focused tests check full-length long reads through filtering and nanoq QC,
known cramino length/identity statistics and histogram weights, and exact
mosdepth coverage percentages. Cramino JSON retains read counts and base totals;
the tests also verify the TSV units selected by `--scaled`.
CI also runs the full suite with Docker and Apptainer on both Linux architectures.

The September 2026 upgrade audit compared prior and updated CLI behavior.
Chopper's split-read identifiers changed as intended, and newer bcftools rejects
malformed VCF input that previously produced successful partial statistics.
Ordinary filtering and the checked cramino metrics agreed across versions.
The long-read check protects output integrity but does not reliably reproduce
the old partial-write defect when stdout is a regular file.

See [the wrapper contract follow-up](https://github.com/aganezov/ont_qc_mcp/issues/50)
for advertised-option and histogram-unit changes. Upstream cramino's phased,
spliced, and uBAM modes are outside this MCP's tested interface.
