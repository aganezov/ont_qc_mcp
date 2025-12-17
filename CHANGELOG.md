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
- Bounded subprocess capture and safer streaming pipelines
- Parser semantics clarified for missing vs empty histogram blocks

### Fixed
- Sanitized tool output example artifacts to remove machine-specific paths

