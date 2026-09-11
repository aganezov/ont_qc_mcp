# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Four new MCP tools for enhanced QC workflows:
  - `qc_bed_tool`: Validate and QC BED files (format validation, coordinate checks, issue reporting)
  - `sequencing_summary_tool`: Parse ONT sequencing summary files (yield, N50, Q-scores, yield per hour windows)
  - `qc_variants_tool`: VCF QC statistics via bcftools stats (SNP/indel counts, TS/TV ratio, singletons)
  - `targeted_coverage_tool`: Compute targeted coverage for genomic regions using mosdepth (supports gene names via GFF3, location strings, or BED files; provides mean depth and coverage threshold percentages at 1x/10x/20x)
- New parsers: `parse_sequencing_summary`, `parse_bcftools_stats`, `parse_bed_qc`, `find_gene_coordinates`, `parse_mosdepth_regions_bed`, `parse_mosdepth_thresholds_bed`
- New data schemas: `SequencingSummaryStats`, `VCFStats`, `BedQCReport`, `TargetedCoverageReport`
- CLI wrapper for `bcftools stats` with flag validation support
- CLI wrapper for `mosdepth_targeted_coverage` using `--by` and `--thresholds` for richer coverage metrics
- `bcftools` added to tool configuration and CI workflow
- Synthetic test fixtures: sequencing summary mock, tiny VCF, GFF3 gene annotations, valid/invalid BED files
- Unit tests for QC parsers (`test_qc_parsers.py`)
- Integration tests for QC tools (`test_qc_tools_integration.py`)
- Structured logging across CLI wrappers, utils, tools, and MCP server
- Input validation with size limits and safer fallbacks
- Structured error payloads and optional verbose provenance
- Test expansion for plotting, utils, edge cases, concurrency, and protocol smoke checks

### Changed
- Use a shared pinned toolchain for CI and local setup, updating chopper to 0.14.0 and cramino to 1.4.1.
- **Breaking:** use the Cramino 1.4.1 JSON contract directly. Read-length and
  alignment-accuracy Phred histograms now retain both `count` (reads) and `bases`
  (base pairs), including open-ended bins. Remove `use_scaled`, separate scaled
  histogram fields, mislabeled MAPQ histogram fields, obsolete Cramino recipes,
  and user-controlled Cramino output flags; retain `threads` and `include_hist`.
- Preserve open-ended Cramino bins in histogram plots with explicit lower-bound labels.
- **Breaking:** reject nanoq thread settings because nanoq 0.10.0 has no thread option.
- Use Chopper 0.14.0 stdout output directly with atomic staging; remove the unsupported
  `filter --output --report-json` invocation. The aggressive trim recipe now selects
  fixed cropping, and direct crop flags require that mode. Remove unavailable
  Chopper report read-count fields; use FASTQ QC on the output for statistics.
- Update the CI mosdepth version from 0.3.12 to 0.3.14.
- Extend lightweight CI coverage to Python 3.13 and 3.14 while retaining Python 3.10 as the minimum.
- Bounded subprocess capture and safer streaming pipelines
- Parser semantics clarified for missing vs empty histogram blocks

### Fixed
- Convert GFF3 gene starts to zero-based BED coordinates for targeted coverage,
  preserving the first base and single-base features on either strand.
- Reject negative BED start coordinates and exclude those intervals from valid interval and base totals.
- Convert mosdepth threshold base counts to percentages using interval length, fixing values above 100% in targeted coverage reports.
- Preserve FASTQ files during filtering: reject input/output aliases, publish completed output atomically, and clean temporary files on failure.
- Run the CI formatting check in `scripts/ci-local.sh` and propagate failures.
- Refresh locked cryptography, Pillow, and pip dependencies while retaining MCP v1.
- Sanitized tool output example artifacts to remove machine-specific paths

### Security
- Reject control characters in IGV batch fields, fixing a command-injection issue:
  a newline in a region `chrom`/`name`, the `genome`, a track path, or
  `extra_commands`/`extra_preferences` could inject arbitrary IGV batch commands.
  All such fields are now validated and rejected fail-closed.
  Advisory: [GHSA-634p-vpv6-fxg8](https://github.com/aganezov/ont_qc_mcp/security/advisories/GHSA-634p-vpv6-fxg8).
- Neutralize CLI argument injection via `-`-leading file paths (CWE-88): an
  MCP-client-supplied path whose name begins with `-` (e.g. `-rf.bam` or
  `--reference=/etc/passwd`) could be parsed by a wrapped tool as an *option*
  instead of an input file. Every untrusted path passed to `nanoq`, `chopper`,
  `cramino`, `samtools`, `mosdepth`, and `bcftools` is now prefixed with `./`
  when needed, so it is always read as a file.

