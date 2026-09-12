# ONT QC MCP

Model Context Protocol server exposing lightweight QC/EDA helpers for Oxford Nanopore FASTQ and BAM/CRAM inputs. The server wraps common CLI tools and returns machine-readable summaries for educational and practical workflows.

## Features
- FASTQ read-level QC via `nanoq` (read count, length/N50, GC, q-score/length histograms).
- BAM/CRAM alignment QC via `cramino` (read lengths, identity, alignment-accuracy histograms).
- Depth-of-coverage via `mosdepth`.
- Read filtering/trimming via `chopper`.
- Optional plotting helpers (length/qscore histograms) when `matplotlib` is installed.
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
and restart the server process after upgrading. The existing tool names, input
schemas, resource contents, and numerical result payloads are preserved.

### Consistent environment for tests/tools
- Use `scripts/with-env.sh` to set PATH and venv for all commands: `scripts/with-env.sh pytest`.
- It activates `.venv` (if present) and optionally prepends toolchain paths via:
  - `MCP_TOOLCHAIN_PATH` (directory with nanoq/chopper/cramino/mosdepth/samtools)
  - `CONDA_PREFIX` (prepends `<conda>/bin` when active)
  - `CARGO_HOME` or `~/.cargo/bin` (cargo-installed tools)
- Keep `.venv` in the repo root; if missing, the script warns and continues.

## MCP tools (high level)

### Environment & Metadata
- `env_status`: Check availability of required CLI tools.
- `header_metadata_tool`: Extract BAM/CRAM/VCF header metadata (contigs, samples, programs) plus a concise summary.

### Read-level QC (FASTQ)
- `qc_reads_fastq_tool`: nanoq read-level QC (counts, lengths, qscore histogram).
- `filter_reads_fastq_tool`: chopper filtering/trimming; returns the command and output path (use `qc_reads_fastq_tool` for output statistics).
  Output names ending in `.gz` (case-insensitive) write gzip-compressed FASTQ; automatic and other ordinary filenames write plain FASTQ.
  The supplied filename determines encoding, including when it is a symlink.
  Unsupported compressed endings are rejected: `.bgz`, `.bgzf`, `.bz`, `.bz2`, `.bzip2`, `.xz`, `.lzma`, `.zst`, `.zstd`, `.lz4`, `.zip`, `.z` (including `.Z`), and `.gzip`. Use `.gz` for gzip output.
  Gzip output uses a separate compression pass and temporarily needs space for both the raw and compressed filtered data.
  Rejects input/output aliases and replaces output only after filtering and any compression succeed.
  Existing file permission bits are retained; new output files are private to the current user.
- `read_length_distribution_fastq_tool`: percentiles + histogram from nanoq.
- `qscore_distribution_fastq_tool`: per-read q-score histogram from nanoq.

### Alignment QC (BAM/CRAM)
- `qc_alignment_tool`: cramino alignment QC (identity, read-length and alignment-accuracy Phred histograms).
- `coverage_stats_tool`: mosdepth coverage summary over reported contigs; see [field definitions and examples](docs/mosdepth-summary.md).
- `alignment_error_profile_tool`: NM-derived error rate and coverage distribution from `samtools stats`; see [field definitions and examples](docs/samtools-statistics.md).
- `alignment_summary_tool`: aggregates cramino + mosdepth (+ error profile).
- `read_length_distribution_bam_tool`: streaming samtools fastq -> nanoq length stats.
- `qscore_distribution_bam_tool`: streaming samtools fastq -> nanoq qscore histogram.
- `targeted_coverage_tool`: compute targeted coverage for genomic regions using mosdepth (supports gene names via GFF3, 0-based, end-exclusive location strings like `chr1:1000-2000`, or BED files; provides mean depth and coverage threshold percentages at 1x/10x/20x). Requires samtools to validate target contigs and coordinate bounds against the alignment header before running mosdepth.

Cramino 1.4.1 histogram bins contain `start`, `end`, `count` (reads), and `bases`
(total base pairs). `end: null` means an open-ended final bin. Both `length_histogram`
and `qscore_histogram` return these values together; `qscore_histogram` represents
Phred-scaled alignment identity, not MAPQ or FASTQ base quality. `include_hist: false`
returns `null` for both histograms; an explicitly empty bin array remains `[]`.
Plots retain open-bin counts in hatched bars labeled `≥ start`; the display width
of those bars does not indicate an upper bound.
Cramino flags accept only `threads`; output format and histogram switches are managed
by the wrapper. The `use_scaled` parameter, separate scaled histogram fields, MAPQ
histogram fields, and old Cramino recipes have been removed.

Chopper 0.14.0 writes FASTQ to stdout. The wrapper stages this output and atomically
publishes it on success. Nonzero crop flags require `trim_approach: "fixed-crop"`
and are rejected without it. The `aggressive_trim` recipe selects this mode and
trims 50 bases from each end.


### Variant QC (VCF/BCF)
- `qc_variants_tool`: VCF/BCF QC statistics via bcftools stats (SNP/indel counts, TS/TV ratio, singletons).

### Sequencing Run QC
- `sequencing_summary_tool`: parse ONT sequencing summary files (yield, N50, Q-scores, yield per hour windows).

### File Validation
- `qc_bed_tool`: validate and QC BED files (format validation, coordinate checks, issue reporting).

### IGV Snapshots
- `igv_snapshot_tool`: generate IGV screenshots for genomic regions (requires Docker or Apptainer).

### Resources & Guidance
- Guidance resource: `tool://guidance/{tool}` returns runtime hints, defaults (threads/timeouts), and links to flag schemas/recipes to help orchestration layers decide whether to call a tool.
- Null/empty semantics: histogram/percentile fields are `null` when the upstream tool omits them; empty lists mean the tool explicitly returned an empty block. Provenance is lightweight by default and can be expanded with `MCP_INCLUDE_PROVENANCE=1`.

## Execution defaults and configurability
- CLI calls are executed in worker threads to avoid blocking the MCP event loop.
- Defaults are conservative and overridable via environment variables:
  - `MCP_THREADS_DEFAULT` / `MCP_THREADS_<TOOL>` (e.g., `MCP_THREADS_CRAMINO`)
  - `MCP_TIMEOUT_DEFAULT` / `MCP_TIMEOUT_<TOOL>` (seconds; e.g., `MCP_TIMEOUT_MOSDEPTH`)
  - `MCP_NANOQ_AUX_STATS=1` (default) to compute FASTQ/BAM length/qscore histograms via nanoq `--read-lengths/--read-qualities` (may produce large temp files for huge inputs; set to `0` to disable)
  - `MCP_STDIO_TRANSPORT=anyio|compat` (default `anyio`) to control how the stdio MCP server reads/writes JSON-RPC (use `compat` in restricted/sandboxed environments that hang with async file wrappers)
  - `MCP_BLOCKING_MODE=auto|executor|sync` (default `auto`) to control how blocking work is executed; `auto` uses a threadpool when thread wakeups are reliable and falls back to `sync` otherwise
- Per-tool defaults are also reflected in the guidance resource and tool descriptions returned by `list_tools`. Threads are applied to all tools except nanoq. nanoq 0.10.0 has no thread option; explicit `threads` flags and `MCP_THREADS_NANOQ` settings are rejected.
- Most `MCP_*` environment variables are read at server startup; changing them requires restarting the MCP server. Per-call overrides are available via tool arguments/flags (e.g., `output_dir` for `igv_snapshot_tool`). If multiple clients need different defaults, run separate server instances.
- Coverage low-depth marking is opt-in via `low_cov_threshold`; error-profile collection in summaries is opt-in via `include_error_profile`.

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
- Regenerate documented tool outputs: see `docs/tool-output-examples.md` for the one-liner

## Notes
- Outputs are JSON-first to play well with downstream pipelines.
- Plotting helpers emit file paths (PNG); no base64 payloads are returned.
