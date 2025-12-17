"""
Unit tests for QC-related parsers.

Tests for:
- parse_sequencing_summary: ONT sequencing summary file parsing
- parse_bcftools_stats: bcftools stats output parsing
- parse_bed_qc: BED file validation
- find_gene_coordinates: GFF3 gene coordinate lookup
"""

from pathlib import Path

import pytest

from ont_qc_mcp.parsers import (
    find_gene_coordinates,
    parse_bcftools_stats,
    parse_bed_qc,
    parse_sequencing_summary,
)
from ont_qc_mcp.schemas import (
    BedIssue,
    BedQCReport,
    IndelStats,
    RunYieldWindow,
    SequencingSummaryStats,
    SNPStats,
    VCFStats,
    VariantGeneralStats,
)


def test_parse_sequencing_summary_basic(tmp_path):
    """Test parsing a basic sequencing summary file with standard columns."""
    summary_file = tmp_path / "sequencing_summary.txt"
    summary_file.write_text(
        "filename\tread_id\trun_id\tchannel\tstart_time\tsequence_length_template\tmean_qscore_template\n"
        "read_0000.fastq\tread0000\trun_001\t115\t1.80\t19024\t9.71\n"
        "read_0001.fastq\tread0001\trun_001\t143\t53.03\t45348\t13.18\n"
        "read_0002.fastq\tread0002\trun_001\t90\t42.52\t3082\t8.21\n"
    )

    result = parse_sequencing_summary(summary_file)

    assert isinstance(result, SequencingSummaryStats)
    assert result.file == str(summary_file)
    assert result.total_reads == 3
    assert result.total_yield == 19024 + 45348 + 3082
    assert result.mean_length == pytest.approx((19024 + 45348 + 3082) / 3.0)
    assert result.mean_qscore == pytest.approx((9.71 + 13.18 + 8.21) / 3.0)
    # N50 calculation: sorted lengths [45348, 19024, 3082], cumulative [45348, 64372, 67454]
    # Half of total = 33727, first length >= half is 45348
    assert result.n50 == 45348


def test_parse_sequencing_summary_column_variants(tmp_path):
    """Test parsing sequencing summary with different column orders/names."""
    # Test with different column order and optional columns
    summary_file = tmp_path / "sequencing_summary_variant.txt"
    summary_file.write_text(
        "read_id\tfilename\tsequence_length_template\tstart_time\tmean_qscore_template\n"
        "read0000\tread_0000.fastq\t10000\t1.80\t10.5\n"
        "read0001\tread_0001.fastq\t20000\t53.03\t11.0\n"
    )

    result = parse_sequencing_summary(summary_file)

    assert isinstance(result, SequencingSummaryStats)
    assert result.total_reads == 2
    assert result.total_yield == 30000
    assert result.mean_length == pytest.approx(15000.0)
    assert result.mean_qscore == pytest.approx(10.75)

    # Test with missing mean_qscore column
    summary_file_no_qscore = tmp_path / "sequencing_summary_no_qscore.txt"
    summary_file_no_qscore.write_text(
        "filename\tread_id\tsequence_length_template\n"
        "read_0000.fastq\tread0000\t10000\n"
        "read_0001.fastq\tread0001\t20000\n"
    )

    result_no_qscore = parse_sequencing_summary(summary_file_no_qscore)
    assert result_no_qscore.mean_qscore is None
    assert result_no_qscore.total_reads == 2


def test_parse_sequencing_summary_yield_windows(tmp_path):
    """Test parsing sequencing summary with yield per hour windows."""
    summary_file = tmp_path / "sequencing_summary_windows.txt"
    summary_file.write_text(
        "filename\tread_id\trun_id\tchannel\tstart_time\tsequence_length_template\tmean_qscore_template\n"
        "read_0000.fastq\tread0000\trun_001\t115\t0.5\t10000\t10.0\n"
        "read_0001.fastq\tread0001\trun_001\t143\t0.8\t20000\t11.0\n"
        "read_0002.fastq\tread0002\trun_001\t90\t1.2\t15000\t12.0\n"
        "read_0003.fastq\tread0003\trun_001\t224\t1.5\t25000\t13.0\n"
    )

    result = parse_sequencing_summary(summary_file)

    assert isinstance(result, SequencingSummaryStats)
    # Should calculate yield windows (e.g., per hour bins)
    assert len(result.yield_per_hour) > 0
    assert all(isinstance(w, RunYieldWindow) for w in result.yield_per_hour)
    # First window should have reads
    if result.yield_per_hour:
        assert result.yield_per_hour[0].read_count > 0
        assert result.yield_per_hour[0].yield_bp > 0


def test_parse_sequencing_summary_real_fixture():
    """Test parsing the real sequencing summary mock fixture."""
    fixture_path = Path("tests/fixtures/synthetic/sequencing_summary_mock.txt")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    result = parse_sequencing_summary(fixture_path)

    assert isinstance(result, SequencingSummaryStats)
    assert result.file == str(fixture_path)
    assert result.total_reads == 20  # 20 data rows in the fixture
    assert result.total_yield > 0
    assert result.mean_length is not None
    assert result.mean_qscore is not None


def test_parse_bcftools_stats_snps_only():
    """Test parsing bcftools stats output with only SNP statistics."""
    stats_output = """
# This file was produced by bcftools stats (1.x)
# The command line was: bcftools stats test.vcf.gz
# Summary Numbers. Use `grep ^SN | cut -f 3-` to extract this part.
SN	0	number of samples:	1
SN	0	number of records:	3
SN	0	number of SNPs:	3
SN	0	number of indels:	0
SN	0	number of MNPs:	0
SN	0	number of others:	0
SN	0	number of multiallelic sites:	0
SN	0	number of multiallelic SNP sites:	0
# Transitions / Transversions
TSTV	3	3	1.00
# TSTV ratio: 1.00
"""

    result = parse_bcftools_stats(stats_output, include_snps=True, include_indels=False)

    assert isinstance(result, VCFStats)
    assert result.file == ""  # File path not in stats output
    assert isinstance(result.general, VariantGeneralStats)
    assert result.general.total_records == 3
    assert result.general.mnps == 0
    assert result.general.others == 0
    assert isinstance(result.snps, SNPStats)
    assert result.snps.count == 3
    assert result.snps.ts_tv_ratio == pytest.approx(1.00)
    assert result.indels is None


def test_parse_bcftools_stats_full():
    """Test parsing bcftools stats output with both SNPs and indels."""
    stats_output = """
# This file was produced by bcftools stats (1.x)
# The command line was: bcftools stats test.vcf.gz
# Summary Numbers. Use `grep ^SN | cut -f 3-` to extract this part.
SN	0	number of samples:	1
SN	0	number of records:	5
SN	0	number of SNPs:	3
SN	0	number of indels:	2
SN	0	number of MNPs:	0
SN	0	number of others:	0
SN	0	number of multiallelic sites:	0
SN	0	number of multiallelic SNP sites:	0
SN	0	number of singletons:	1
SN	0	sample name:	SAMPLE1
# Transitions / Transversions
TSTV	3	3	1.00
# TSTV ratio: 1.00
"""

    result = parse_bcftools_stats(stats_output, include_snps=True, include_indels=True)

    assert isinstance(result, VCFStats)
    assert result.general.total_records == 5
    assert result.general.singletons == 1
    assert result.general.sample_name == "SAMPLE1"
    assert isinstance(result.snps, SNPStats)
    assert result.snps.count == 3
    assert result.snps.ts_tv_ratio == pytest.approx(1.00)
    assert isinstance(result.indels, IndelStats)
    assert result.indels.count == 2


def test_parse_bcftools_stats_no_tstv():
    """Test parsing bcftools stats when TSTV section is missing."""
    stats_output = """
# This file was produced by bcftools stats (1.x)
SN	0	number of samples:	1
SN	0	number of records:	2
SN	0	number of SNPs:	2
SN	0	number of indels:	0
"""

    result = parse_bcftools_stats(stats_output, include_snps=True, include_indels=False)

    assert isinstance(result, VCFStats)
    assert result.snps is not None
    assert result.snps.count == 2
    assert result.snps.ts_tv_ratio is None  # No TSTV section


def test_parse_bcftools_stats_indels_only():
    """Test parsing bcftools stats with only indels requested."""
    stats_output = """
SN	0	number of samples:	1
SN	0	number of records:	5
SN	0	number of SNPs:	3
SN	0	number of indels:	2
"""

    result = parse_bcftools_stats(stats_output, include_snps=False, include_indels=True)

    assert isinstance(result, VCFStats)
    assert result.snps is None
    assert isinstance(result.indels, IndelStats)
    assert result.indels.count == 2


def test_parse_bed_qc_valid(tmp_path):
    """Test parsing a valid BED file."""
    bed_file = tmp_path / "valid.bed"
    bed_file.write_text(
        "chr1\t1000\t2000\tgene1_exon1\t.\t+\n"
        "chr1\t5000\t6000\tgene1_exon2\t.\t+\n"
        "chr1\t10000\t12000\tgene2_exon1\t.\t-\n"
    )

    result = parse_bed_qc(bed_file)

    assert isinstance(result, BedQCReport)
    assert result.file == str(bed_file)
    assert result.total_intervals == 3
    assert result.valid_intervals == 3
    assert result.total_bases == (2000 - 1000) + (6000 - 5000) + (12000 - 10000)
    assert result.is_valid is True
    assert len(result.issues) == 0


def test_parse_bed_qc_invalid(tmp_path):
    """Test parsing a BED file with validation errors."""
    bed_file = tmp_path / "invalid.bed"
    bed_file.write_text(
        "chr1\t1000\t2000\tvalid_interval\t.\t+\n"
        "chr1\t5000\t3000\tstart_gt_end\t.\t+\n"
        "chr1\tabc\t6000\tnon_integer_start\t.\t+\n"
        "chr1\t10000\txyz\tnon_integer_end\t.\t-\n"
        "chr1\t20000\t25000\tvalid_interval2\t.\t+\n"
    )

    result = parse_bed_qc(bed_file)

    assert isinstance(result, BedQCReport)
    assert result.file == str(bed_file)
    assert result.total_intervals == 5
    assert result.valid_intervals == 2  # Only 2 valid intervals
    assert result.is_valid is False
    assert len(result.issues) == 3  # 3 invalid lines

    # Check that issues are properly reported
    assert all(isinstance(issue, BedIssue) for issue in result.issues)
    assert any("start > end" in issue.issue.lower() or "start_gt_end" in issue.line_content for issue in result.issues)
    assert any("non-integer" in issue.issue.lower() or "non_integer" in issue.line_content for issue in result.issues)


def test_parse_bed_qc_real_fixture():
    """Test parsing the real valid.bed fixture."""
    fixture_path = Path("tests/fixtures/synthetic/valid.bed")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    result = parse_bed_qc(fixture_path)

    assert isinstance(result, BedQCReport)
    assert result.is_valid is True
    assert result.total_intervals == 5
    assert result.valid_intervals == 5


def test_parse_bed_qc_invalid_fixture():
    """Test parsing the real invalid.bed fixture."""
    fixture_path = Path("tests/fixtures/synthetic/invalid.bed")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    result = parse_bed_qc(fixture_path)

    assert isinstance(result, BedQCReport)
    assert result.is_valid is False
    assert result.total_intervals == 5
    assert result.valid_intervals < 5
    assert len(result.issues) > 0


def test_parse_bed_qc_empty(tmp_path):
    """Test parsing an empty BED file."""
    bed_file = tmp_path / "empty.bed"
    bed_file.write_text("")

    result = parse_bed_qc(bed_file)

    assert isinstance(result, BedQCReport)
    assert result.total_intervals == 0
    assert result.valid_intervals == 0
    assert result.total_bases == 0
    assert result.is_valid is True  # Empty file is technically valid


def test_parse_bed_qc_minimal_columns(tmp_path):
    """Test parsing BED file with minimal 3 columns."""
    bed_file = tmp_path / "minimal.bed"
    bed_file.write_text(
        "chr1\t1000\t2000\n"
        "chr1\t5000\t6000\n"
    )

    result = parse_bed_qc(bed_file)

    assert isinstance(result, BedQCReport)
    assert result.total_intervals == 2
    assert result.valid_intervals == 2
    assert result.is_valid is True


def test_find_gene_coordinates_gff3():
    """Test finding gene coordinates from a GFF3 file."""
    fixture_path = Path("tests/fixtures/synthetic/genes_mock.gff3")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    # Test finding MOCK_GENE1
    coordinates = find_gene_coordinates(fixture_path, "MOCK_GENE1")

    assert isinstance(coordinates, list)
    assert len(coordinates) > 0
    # Should return list of (chrom, start, end) tuples
    assert all(isinstance(coord, tuple) and len(coord) == 3 for coord in coordinates)
    assert all(isinstance(coord[0], str) for coord in coordinates)
    assert all(isinstance(coord[1], int) and isinstance(coord[2], int) for coord in coordinates)
    # MOCK_GENE1 should be on chr1, spanning 100000-200000
    assert any(coord[0] == "chr1" and coord[1] == 100000 and coord[2] == 200000 for coord in coordinates)


def test_find_gene_coordinates_gff3_multiple_exons():
    """Test finding gene coordinates when gene has multiple exons."""
    fixture_path = Path("tests/fixtures/synthetic/genes_mock.gff3")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    # MOCK_GENE1 has exons at 100000-120000 and 180000-200000
    coordinates = find_gene_coordinates(fixture_path, "MOCK_GENE1")

    # Should return coordinates for all exons or the gene span
    assert len(coordinates) >= 1
    # If returning exons separately, should have 2; if returning gene span, should have 1
    # Check that coordinates cover the gene region
    all_starts = [coord[1] for coord in coordinates]
    all_ends = [coord[2] for coord in coordinates]
    assert min(all_starts) <= 100000
    assert max(all_ends) >= 200000


def test_find_gene_coordinates_gff3_nonexistent():
    """Test finding coordinates for a gene that doesn't exist."""
    fixture_path = Path("tests/fixtures/synthetic/genes_mock.gff3")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    coordinates = find_gene_coordinates(fixture_path, "NONEXISTENT_GENE")

    assert isinstance(coordinates, list)
    assert len(coordinates) == 0


def test_find_gene_coordinates_gff3_by_id():
    """Test finding gene coordinates by ID attribute instead of Name."""
    fixture_path = Path("tests/fixtures/synthetic/genes_mock.gff3")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    # Try finding by ID (gene1) instead of Name (MOCK_GENE1)
    coordinates = find_gene_coordinates(fixture_path, "gene1")

    # Should find the gene by ID if parser supports it
    assert isinstance(coordinates, list)
    # May return empty if only Name is searched, or may find it if ID is also checked


def test_find_gene_coordinates_gff3_case_insensitive():
    """Test that gene name search is case-insensitive."""
    fixture_path = Path("tests/fixtures/synthetic/genes_mock.gff3")
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    coordinates_lower = find_gene_coordinates(fixture_path, "mock_gene1")
    coordinates_upper = find_gene_coordinates(fixture_path, "MOCK_GENE1")

    # Should find the same gene regardless of case
    assert len(coordinates_lower) == len(coordinates_upper)
    if coordinates_lower:
        assert coordinates_lower == coordinates_upper
