# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Migrate the local stdio adapter to MCP SDK 2.2, preserving existing tool schemas, input validation, resource contents, and numerical result payloads.

### Added
- Add shared, unregistered API v2 infrastructure for ordered region normalization and genomic unions, explicit
  BAM/CRAM index and reference resolution, protected native argument arrays, samtools selection/FASTQ conversion
  plans, and one-deadline subprocess pipelines that drain every stage before reporting success.
- Add `regional_alignment_stats_tool` for indexed BAM/CRAM interval batches, with explicit alignment-record counts, MAPQ missingness, and base-quality means for aligned bases inside each interval. Preserve repeated intervals, require existing indexes, and stream without alignment copies.
- Add `mismatch_counts_by_cycle` records with explicit cycle numbers, N-base counts, and per-quality mismatch counts when parsing samtools MPC output.
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
- Retire samtools GC-depth output for ONT QC. The legacy `ErrorProfile.gc_coverage` field remains present but is always null when parsed; it no longer exposes mislabeled histogram counts.
- Deprecate `ErrorProfile.mismatch_by_cycle`; the parser now leaves this unsupported rate field null. Use `mismatch_counts_by_cycle` for counts. Coverage bins retain inclusive integer bounds and use `end=null` for overflow.
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
- Stream nanoq length and quality auxiliary output through POSIX named pipes instead of per-read temporary files, preserving histograms and percentile limits across FASTQ and BAM QC.
- Suppress unused mosdepth per-base BED files and indexes for whole, windowed, and targeted coverage reports.
- Report missing configured executables before launching forced Docker or local IGV runtimes.
- Reject Boolean values for numeric CLI flags before command execution, while preserving accepted numeric values and null/unset behavior.
- Reject non-Boolean values for Boolean CLI flags instead of silently enabling or disabling them through Python truthiness; null still leaves a flag unset.
- Cooperatively stop owned subprocess groups on request cancellation, interrupt retry waits and streaming stderr readers, attempt removal of the invocation's Docker container, clean generated IGV and targeted mosdepth artifacts on failure, and release waiters when a shared read-QC owner is cancelled.
- Launch the configured Singularity executable when Apptainer is unavailable, while retaining the `apptainer` runtime category.
- Keep MCP concurrency capacity reserved until a cancelled running worker finishes, and prevent cancelled queued work from starting.
- Honor the caller's execution configuration for every `alignment_summary` component, including alignment and coverage input limits, thread counts, and timeouts.
- Validate mosdepth summary footers against the executed mode and cross-check aggregate length/base counts before returning QC results.
- Exclude mosdepth whole-genome and region aggregates from contig summaries, mean calculations, and low-coverage locations. Preserve real contigs with aggregate-like names or names beginning with `chrom`.
- Parse samtools SN numeric values separately from trailing comments and retain COV ranges and base counts, including underflow and overflow bins. Stop interpreting MPC N-base counts as mismatch rates.
- Include an hourly yield bin when sequencing-summary reads share one finite timestamp, including single-read summaries with zero run duration.
- Write valid gzip for filtered output paths ending in `.gz`, including empty results, while preserving atomic replacement and cleaning raw/compressed staging files on failure. Reject recognized unsupported compressed-output suffixes before filtering.
- Targeted coverage preserves distinct names for repeated coordinates, taking names from each mosdepth output record and retaining occurrence order for unnamed-output fallbacks.
- Normalize validated target BED rows before mosdepth, preventing blank lines and surrounding whitespace from dropping targets or causing coverage errors. Preserve the original BED file and remove temporary targets after each run.
- Reap streaming samtools/nanoq children and close their pipes on startup, timeout, and other failures; clean auxiliary files even when allocation fails. A descendant-held stderr pipe no longer blocks the caller during cleanup; its reader closes the pipe at EOF.
- Require ASCII decimal coordinate fields in BED QC, matching targeted coverage validation while preserving negative-coordinate diagnostics.
- Preserve BED intervals and target names on contigs beginning with `track` or `browser`, including exact keyword names.
- Convert GFF3 gene starts to zero-based BED coordinates for targeted coverage,
  preserving the first base and single-base features on either strand.
- Reject negative BED start coordinates and exclude those intervals from valid interval and base totals.
- Validate targeted coverage intervals against alignment contigs and lengths before running mosdepth; reject malformed, empty, or out-of-bounds targets.
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
