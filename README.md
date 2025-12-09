# ONT QC MCP

Model Context Protocol server exposing lightweight QC/EDA helpers for Oxford Nanopore FASTQ and BAM/CRAM inputs. The server wraps common CLI tools and returns machine-readable summaries for educational and practical workflows.

## Features
- FASTQ basic stats via `seqkit stats` (read count, length/N50, GC).
- BAM/CRAM stats via `samtools stats`.
- Optional read filtering/trimming through `fastp` with JSON output passthrough.
- Simple environment validation for required CLI tools.

## Requirements
- Python >= 3.10
- CLI tools on `PATH` (override via env vars):
  - `SEQKIT` (default `seqkit`)
  - `SAMTOOLS` (default `samtools`)
  - `FASTP` (default `fastp`)
  - `NANOPLOT` (default `NanoPlot`, optional for plotting JSON export)

## Quick start
```bash
pip install -e .
ont-qc-mcp  # launches the MCP stdio server
```

## MCP tools (high level)
- `qc_fastq`: run `seqkit stats` and return parsed metrics.
- `qc_alignment`: run `samtools stats` and return parsed metrics.
- `fastp_filter`: optional filter/trim; returns the `fastp` JSON report.
- `fastq_eda`: aggregate `seqkit` stats with optional `NanoPlot` JSON export.
- `env_check`: report availability and versions of configured CLI tools.

## Development
```bash
pip install -e ".[dev]"
pytest
```

## Notes
- Outputs favor JSON/CLI-style data so they can be consumed by downstream workflows.
- Plotting hooks are scaffolded; NanoPlot JSON export can be wired in later without changing the MCP interface.

