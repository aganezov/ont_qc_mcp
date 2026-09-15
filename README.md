# ONT QC MCP

Model Context Protocol server exposing lightweight QC/EDA helpers for Oxford Nanopore FASTQ and BAM/CRAM inputs. The server wraps common CLI tools and returns machine-readable summaries for educational and practical workflows.

## Features
- FASTQ and selected BAM/CRAM stored-read QC via `nanoq`.
- BAM/CRAM alignment-record QC via `samtools` and `cramino`.
- Exact depth and threshold breadth via `mosdepth`.
- Read filtering/trimming via `chopper`.
- Environment validation for required CLI tools.
- Per-tool runtime guidance via MCP resources to help LLM tool selection.
- Non-blocking execution: CLI calls are offloaded to worker threads with configurable timeouts and thread defaults.

## Requirements
- Python >= 3.10
- CLI tools on `PATH` (override via env vars):
  - `NANOQ` (default `nanoq`)
  - `CRAMINO` (default `cramino`)
  - `MOSDEPTH` (default `mosdepth`)
  - `CHOPPER` (default `chopper`)
  - `SAMTOOLS` (default `samtools`) for error profiling and BAM operations
  - `BCFTOOLS` (default `bcftools`) for VCF/BCF variant QC

### Installing CLI tools

**Tested toolchain (Linux x86_64/ARM64 and macOS ARM64):**

Use a native C/C++ compiler: the standard build tools on Linux, or Xcode Command
Line Tools on macOS. Then create and activate the environment:

```bash
mamba env create --file environment.yml --strict-channel-priority
conda activate ont-qc-mcp
bash scripts/install-rust-tools.sh
export NANOQ="$CONDA_PREFIX/bin/nanoq"
```

If `ont-qc-mcp` already exists, use a fresh name with `--name` and activate that
name instead. The first installation compiles three Rust tools and takes several
minutes. The explicit `NANOQ` path selects this environment's binary even when a
separate Cargo installation exists.

CI uses the same environment file and installer. See [toolchain versions and
validation](docs/toolchain.md) for the pins, source-build details, and test commands.

## IGV snapshot tool (optional)
- Container runtime: Docker (preferred) or Apptainer/Singularity
- Timeout/runtime controls: `MCP_TIMEOUT_IGV` (default 600s) and `MCP_IGV_CONTAINER_IMAGE` / `MCP_IGV_SIF_PATH`

### Pre-built multi-arch image (recommended)
```bash
docker pull aganezov/igv_snapper:0.2
```
This image supports both `linux/amd64` and `linux/arm64` natively. Docker automatically pulls the correct architecture.

### Build locally (optional)
```bash
cd docker/igv_snapper
./build-multiarch.sh igv_snapper 0.2
export MCP_IGV_CONTAINER_IMAGE=igv_snapper:0.2
```
The image uses Ubuntu 24.04 + OpenJDK 21 + IGV 2.19.7.

### Apptainer/HPC users
```bash
apptainer pull igv_snapper.sif docker://aganezov/igv_snapper:0.1
export MCP_IGV_SIF_PATH=/path/to/igv_snapper.sif
```

### Preserving IGV test snapshots
- Set `IGV_SNAPSHOT_DIR` to a writable directory and IGV integration tests will copy generated snapshots there.

## Quick start
```bash
uv sync               # reproducible venv from uv.lock  (or: pip install -e .)
uv run ont-qc-mcp     # launches the MCP stdio server
```

The server uses MCP Python SDK 2.2 or later in the 2.x series. Run `uv sync`
and restart the server process after upgrading.

### Consistent environment for tests/tools
- Use `scripts/with-env.sh` to set PATH and venv for all commands: `scripts/with-env.sh pytest`.
- It activates `.venv` (if present) and optionally prepends toolchain paths via:
  - `MCP_TOOLCHAIN_PATH` (directory with nanoq/chopper/cramino/mosdepth/samtools)
  - `CONDA_PREFIX` (prepends `<conda>/bin` when active)
  - `CARGO_HOME` or `~/.cargo/bin` (cargo-installed tools)
- Keep `.venv` in the repo root; if missing, the script warns and continues.

## Public API v2 tools

The server advertises exactly ten tools. Requests are strict: unknown fields and
implicit type coercions are rejected before a worker starts.

| Tool | Purpose |
| --- | --- |
| `read_qc` | Read length and quality QC for FASTQ, or complete stored sequences selected from BAM/CRAM. |
| `alignment_qc` | Alignment-record counts, MAPQ, aligned-base quality, identity, and NM/error evidence. |
| `coverage_qc` | Exact depth and threshold breadth for contigs, requested intervals, or fixed windows. |
| `variant_qc` | General, SNP, and indel summaries for VCF/BCF, with optional regional grouping. |
| `environment_status` | Availability and resolved paths for native tools and the IGV runtime. |
| `header_info` | BAM/CRAM/SAM/VCF header metadata. |
| `bed_qc` | BED structure and accepted-interval summary. |
| `run_summary` | ONT sequencing-summary yield, N50, Q-score, and one-hour yield windows. |
| `filter_reads` | Chopper FASTQ filtering or trimming with atomic output publication. |
| `igv_snapshots` | IGV PNG/SVG snapshots from regions or a caller-supplied batch file. |

The four numerical tools share `path`, optional `reference_path`, optional
`regions`, and optional `deadline_seconds`. Region sources can be an ordered list
of zero-based half-open intervals, one or more one-based inclusive samtools region
strings, BED text or a BED file, or a GFF3 gene source. BAM/CRAM regional requests
require an existing index. CRAM requests that need reference access require an
explicit indexed, uncompressed FASTA.

Numerical responses include the effective normalized request and backend
provenance. `metrics` selects response sections and only the required backends run.
Typed `selection` objects control record populations. Namespaced `extra_args`
arrays expose validated native options while server-owned output, population, and
security-sensitive arguments remain protected. See [the frozen contracts and
examples](docs/api-v2-contracts.md).

There is no composite alignment-summary endpoint. For alignment and reference
coverage over the same input, call both tools explicitly:

```json
[
  {"tool": "alignment_qc", "arguments": {"path": "/data/sample.bam"}},
  {"tool": "coverage_qc", "arguments": {"path": "/data/sample.bam"}}
]
```

Pass the same `regions`, `reference_path`, and compatible selection intent when
both reports must describe the same domain. The same recipe is available as
`tool://recipes/alignment_qc`. Each tool also has a guidance resource at
`tool://guidance/{tool}`.

`filter_reads` chooses output encoding from `output_fastq`: `.gz` writes gzip and
ordinary filenames write plain FASTQ. It rejects input/output aliases and
unsupported compressed suffixes, stages all work, and replaces the output only
after filtering and compression succeed. Use `read_qc` on the output when QC is
needed.

## Execution defaults and configurability
- CLI calls are executed in worker threads to avoid blocking the MCP event loop.
- Defaults are conservative and overridable via environment variables:
  - `MCP_THREADS_DEFAULT` / `MCP_THREADS_<TOOL>` (e.g., `MCP_THREADS_CRAMINO`)
  - `MCP_TIMEOUT_DEFAULT` / `MCP_TIMEOUT_<TOOL>` (seconds; e.g., `MCP_TIMEOUT_MOSDEPTH`)
  - `MCP_NANOQ_AUX_STATS=1` (default) to compute FASTQ/BAM length/qscore histograms via nanoq `--read-lengths/--read-qualities`; set to `0` to disable
  - `MCP_STDIO_TRANSPORT=anyio|compat` (default `anyio`) to control how the stdio MCP server reads/writes JSON-RPC (use `compat` in restricted/sandboxed environments that hang with async file wrappers)
  - `MCP_BLOCKING_MODE=auto|executor|sync` (default `auto`) to control how blocking work is executed; `auto` uses a threadpool when thread wakeups are reliable and falls back to `sync` otherwise
- Per-tool defaults are reflected in guidance resources and tool descriptions returned by `list_tools`. Nanoq 0.10.0 has no thread option, so `MCP_THREADS_NANOQ` is rejected.
- Nanoq auxiliary statistics use POSIX named pipes, with bounded reader buffers and no per-read disk files. Temporary FIFO paths are removed after each attempt, including failure and cancellation. Platforms without named pipes must disable auxiliary statistics explicitly. Exact length percentiles retain up to `MCP_NANOQ_PERCENTILES_EXACT_MAX_READS` values (default 200,000); larger inputs still receive histograms. Nanoq itself retains per-read values in memory, so this transport change does not bound the tool's total memory use.
- Most `MCP_*` environment variables are read at server startup; changing them requires restarting the MCP server. Per-call overrides use typed request fields and namespaced `extra_args`; `igv_snapshots` also supports `output_dir`. If multiple clients need different defaults, run separate server instances.

## Development
```bash
uv sync --all-extras    # dev + plot deps, exact versions from uv.lock
uv run pytest           # or: scripts/ci-local.sh  (mirrors the CI lint tier)
```
uv uses the committed `uv.lock` for a reproducible env. Plain pip works too:
`pip install -e ".[dev]"` (tests/lint) · `".[plots]"` (matplotlib) · `".[all]"`.

### Helpful wrappers
- `scripts/with-env.sh <cmd>`: activates `.venv` (if present) and prepends optional toolchain paths. It honors:
  - `MCP_TOOLCHAIN_PATH` (directory with nanoq/chopper/cramino/mosdepth/samtools)
  - `CONDA_PREFIX` (prepends `<conda>/bin` when active)
  - `CARGO_HOME` or `~/.cargo/bin` (cargo-installed tools)

### Common workflows
- Run the MCP server: `python -m ont_qc_mcp.app_server` (or `ont-qc-mcp` entrypoint)
- Unit tests only: `scripts/with-env.sh pytest`
- Full test suite with external CLIs on PATH: `scripts/with-env.sh pytest -m integration` (after CLIs are installed)
- Smoke-check real files via MCP (writes JSON to stdout or `--out`): `scripts/with-env.sh python scripts/mcp_smoke_real.py --dir /path/to/test_dir`
- Inspect contract examples and run real-file smoke calls: see `docs/tool-output-examples.md`

## Notes
- Outputs are JSON-first to play well with downstream pipelines.
- Plotting helpers emit file paths (PNG); no base64 payloads are returned.
