# API v2 integrated verification

This record checks the integrated ten-tool API against
[`api-v2-contracts.md`](api-v2-contracts.md) and
[`api-v2-regression-map.md`](api-v2-regression-map.md). The code under test was
`052af7c1f7526b3298fc2f862b23e47183c9f82f` (`origin/main` on
2026-09-19). This change adds only this record; it does not alter the tested
implementation.

## Numerical and wire evidence

| Contract or preserved failure | Check and expected result | Observed result |
| --- | --- | --- |
| Read QC, default/regional/grouped, empty and missing evidence, BAM/CRAM | `tests/test_v2_read_qc_real.py`: matching FASTQ/BAM/CRAM inputs yield 2 reads and 10 bases; two overlapping rows each count 2 while the union counts 2; an empty selection has zero counts and null means; missing SEQ is counted as an exclusion and missing QUAL aborts requested quality metrics | All 4 native tests passed |
| Alignment QC, default/regional/grouped, whole-record native metrics, BAM/CRAM | `tests/test_v2_alignment_qc_real.py`: default mask admits 3 records, MAPQ mean is 20 over 2 known records; overlapping rows count 2 and 3, while the union counts 3; identity and NM error retain their distinct meanings and CRAM agrees with BAM on checked fields | All 5 native tests passed |
| Coverage QC, four row plans, overlap union, zero depth, median, selection, BAM/CRAM | `tests/test_v2_coverage_qc_real.py`: hand-counted whole-reference depth sums are 8 and 4; the overlapping region rows have depth sums 8, 4, and 3, while the genomic union has 13 reference bases and depth sum 11; direct mosdepth output agrees with every checked row; empty-contig threshold zero, native selection and CRAM equivalence are checked separately | All 11 native test cases passed |
| Variant QC, overlap union, grouped rows, empty SNP evidence, advanced arguments | `tests/test_v2_variant_qc_real.py`: combined union counts 4 records and overlapping requested rows count 3 each; an indel whose POS is outside the interval still overlaps by record span; a zero-SNP selection has zero allele counts and null TS/TV; the typed expression and ordered native extras reach bcftools | All 6 native tests passed |
| Shared selection, lifecycle, catalog and regression map | The focused v2 contract, adapter, execution and MCP wire suite checks region normalization, native-option boundaries, upstream failure precedence, cancellation and child reaping, temporary-file cleanup, strict pre-worker validation, and exactly ten advertised tools. The regression map references 46 existing test files; source inspection found no missing file. | 344 focused unit tests passed; 46/46 mapped test files exist |

The native tests above were run locally with **older, non-pinned** binaries:
samtools 1.22.1, bcftools 1.22, mosdepth 0.3.12, chopper 0.11.0,
cramino 1.1.1, and nanoq 0.10.0. Their success is supplemental evidence.
The repository's pinned Linux integration matrix is required before accepting
native behavior at the documented versions, including mosdepth 0.3.14 and
cramino 1.4.1.

## Reproduction at the tested code revision

From the repository root after `uv sync --all-extras --locked`:

```bash
.venv/bin/pytest -q -m 'not integration and not igv_integration' tests/test_v2_contracts.py tests/test_v2_contract_regressions.py tests/test_v2_regions.py tests/test_v2_samtools.py tests/test_v2_read_qc.py tests/test_v2_alignment_qc.py tests/test_v2_coverage_qc.py tests/test_v2_variant_qc.py tests/test_v2_native_args.py tests/test_v2_execution.py tests/test_v2_supporting_tools.py tests/test_v2_public_api.py tests/test_mcp_wire_contract.py
.venv/bin/pytest -q -m 'not integration and not igv_integration'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy ont_qc_mcp tests
.venv/bin/bandit -q -r ont_qc_mcp -x tests
.venv/bin/pip-audit
```

Results: 344 focused tests passed; the full unit suite passed with 1,214 tests
and 154 integration tests deselected. Ruff check, Ruff format (113 files), mypy
(112 source files), Bandit, and pip-audit passed. Pip-audit found no known
vulnerabilities in published dependencies; it cannot audit this unpublished
local package against PyPI. Local native tests ran as follows after placing
the older Conda tool directory on `PATH`:

```bash
.venv/bin/pytest -q -m integration tests/test_v2_samtools_real.py tests/test_v2_read_qc_real.py tests/test_v2_variant_qc_real.py
.venv/bin/pytest -q -m integration tests/test_v2_coverage_qc_real.py tests/test_v2_alignment_qc_real.py tests/test_v2_supporting_tools_real.py
```

The two native commands passed 17 and 18 tests respectively, with no skips.
The source and unit checks found no missing decisive regression or demonstrated
supported-mode failure requiring an implementation change. Pinned native CI,
CodeQL, current-head advisory reviews, and merge verification remain separate
acceptance evidence for the pull request that carries this record.
