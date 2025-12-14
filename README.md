# ONT QC MCP

Model Context Protocol server exposing lightweight QC/EDA helpers for Oxford Nanopore FASTQ and BAM/CRAM inputs. The server wraps common CLI tools and returns machine-readable summaries for educational and practical workflows.

## Features
- FASTQ read-level QC via `nanoq` (read count, length/N50, GC, q-score/length histograms).
- BAM/CRAM alignment QC via `cramino` (mapping breakdown, identity, MAPQ hist).
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
  - `SAMTOOLS` (default `samtools`) for error profiling

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

## Quick start
```bash
pip install -e .
ont-qc-mcp  # launches the MCP stdio server
```

### Consistent environment for tests/tools
- Use `scripts/with-env.sh` to set PATH and venv for all commands: `scripts/with-env.sh pytest`.
- It activates `.venv` (if present) and optionally prepends toolchain paths via:
  - `MCP_TOOLCHAIN_PATH` (directory with nanoq/chopper/cramino/mosdepth/samtools)
  - `CONDA_PREFIX` (prepends `<conda>/bin` when active)
  - `CARGO_HOME` or `~/.cargo/bin` (cargo-installed tools)
- Keep `.venv` in the repo root; if missing, the script warns and continues.

## MCP tools (high level)
- `env_status`
- `qc_reads_fastq_tool`: nanoq read-level QC (counts, lengths, qscore histogram).
- `filter_reads_fastq_tool`: chopper filtering/trimming; returns command + stats.
- `read_length_distribution_fastq_tool`: percentiles + histogram from nanoq.
- `qscore_distribution_fastq_tool`: per-read q-score histogram from nanoq.
- `read_length_distribution_bam_tool`: streaming samtools fastq -> nanoq length stats.
- `qscore_distribution_bam_tool`: streaming samtools fastq -> nanoq qscore histogram.
- `qc_alignment_tool`: cramino alignment QC (identity, MAPQ hist; mapped/unmapped may be `null` depending on cramino version; use `use_scaled` for base-weighted bins).
- `coverage_stats_tool`: mosdepth coverage summary.
- `alignment_error_profile_tool`: error rates parsed from `samtools stats`.
- `alignment_summary_tool`: aggregates cramino + mosdepth (+ error profile).
- `header_metadata_tool`: extract BAM/CRAM/VCF header metadata (contigs, samples, programs) plus a concise summary.
- Guidance resource: `tool://guidance/{tool}` returns runtime hints, defaults (threads/timeouts), and links to flag schemas/recipes to help orchestration layers decide whether to call a tool.
- Null/empty semantics: histogram/percentile fields are `null` when the upstream tool omits them; empty lists mean the tool explicitly returned an empty block. Provenance is lightweight by default and can be expanded with `MCP_INCLUDE_PROVENANCE=1`.

## Execution defaults and configurability
- CLI calls are executed in worker threads to avoid blocking the MCP event loop.
- Defaults are conservative and overridable via environment variables:
  - `MCP_THREADS_DEFAULT` / `MCP_THREADS_<TOOL>` (e.g., `MCP_THREADS_NANOQ`)
  - `MCP_TIMEOUT_DEFAULT` / `MCP_TIMEOUT_<TOOL>` (seconds; e.g., `MCP_TIMEOUT_MOSDEPTH`)
- Per-tool defaults are also reflected in the guidance resource and tool descriptions returned by `list_tools`.
- Coverage low-depth marking is opt-in via `low_cov_threshold`; error-profile collection in summaries is opt-in via `include_error_profile`.

## Development
```bash
pip install -e ".[dev]"           # tests + linting/coverage helpers
pip install -e ".[plots]"         # add matplotlib for PNG histograms
pip install -e ".[all]"           # everything above in one go
scripts/with-env.sh pytest
```

### Helpful wrappers
- `scripts/with-env.sh <cmd>`: activates `.venv` (if present) and prepends optional toolchain paths. It honors:
  - `MCP_TOOLCHAIN_PATH` (directory with nanoq/chopper/cramino/mosdepth/samtools)
  - `CONDA_PREFIX` (prepends `<conda>/bin` when active)
  - `CARGO_HOME` or `~/.cargo/bin` (cargo-installed tools)

### Common workflows
- Run the MCP server: `python -m ont_qc_mcp.app_server` (or `ont-qc-mcp` entrypoint)
- Unit tests only: `scripts/with-env.sh pytest`
- Full test suite with external CLIs on PATH: `scripts/with-env.sh pytest -m integration` (after CLIs are installed)
- Regenerate documented tool outputs: see `docs/tool-output-examples.md` for the one-liner

## Notes
- Outputs are JSON-first to play well with downstream pipelines.
- Plotting helpers emit file paths (PNG); no base64 payloads are returned.

