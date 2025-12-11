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

## Quick start
```bash
pip install -e .
ont-qc-mcp  # launches the MCP stdio server
```

### Consistent environment for tests/tools
- Use `scripts/with-env.sh` to set PATH and venv for all commands: `scripts/with-env.sh pytest`.
- It prepends conda CLI tools (`/Users/saganezov/miniforge3/envs/ont-qc-mcp/bin`) and cargo-installed `nanoq` (`/Users/saganezov/.cargo/bin`), then activates `.venv`.
- Keep `.venv` in the repo root; if missing, the script warns and continues.

## MCP tools (high level)
- `qc_reads`: nanoq read-level QC (counts, lengths, qscore histogram).
- `filter_reads`: chopper filtering/trimming; returns command + stats.
- `read_length_distribution`: percentiles + histogram from nanoq.
- `qscore_distribution`: per-read q-score histogram from nanoq.
- `qc_alignment`: cramino alignment QC (mapped/unmapped, identity, MAPQ hist; use `use_scaled` for base-weighted bins).
- `coverage_stats`: mosdepth coverage summary.
- `alignment_error_profile`: error rates parsed from `samtools stats`.
- `alignment_summary`: aggregates cramino + mosdepth (+ error profile).
- `env_status`: report availability and resolved paths for all tools.
- `header_metadata_tool`: extract BAM/CRAM/VCF header metadata (contigs, samples, programs) plus a concise summary.
- Guidance resource: `tool://guidance/{tool}` returns runtime hints, defaults (threads/timeouts), and links to flag schemas/recipes to help orchestration layers decide whether to call a tool.
- Defaults stay lightweight; heavier steps (error profiles, quantized/threshold coverage) are opt-in via flags.

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
pytest
```

## Notes
- Outputs are JSON-first to play well with downstream pipelines.
- Plotting helpers emit file paths (PNG); no base64 payloads are returned.

