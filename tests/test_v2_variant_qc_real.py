"""Direct bcftools comparisons for the public API v2 variant backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.v2_variant_qc import variant_qc


@pytest.fixture
def indexed_variants(tmp_path: Path) -> tuple[Path, str]:
    require_executable_tools(["bcftools"])
    bcftools = shutil.which("bcftools")
    assert bcftools is not None
    source = tmp_path / "calls.vcf"
    source.write_text(
        "##fileformat=VCFv4.3\n"
        "##contig=<ID=chr1,length=20>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tG\t40\tPASS\t.\n"
        "chr1\t5\t.\tC\tA\t10\tPASS\t.\n"
        "chr1\t8\t.\tAT\tA\t50\tPASS\t.\n"
        "chr1\t12\t.\tA\tAT\t60\tPASS\t.\n"
        "chr1\t15\t.\tA\tG,C\t70\tPASS\t.\n"
    )
    compressed = tmp_path / "calls.vcf.gz"
    subprocess.run([bcftools, "view", "-Oz", "-o", str(compressed), str(source)], check=True, capture_output=True)
    subprocess.run([bcftools, "index", "--tbi", str(compressed)], check=True, capture_output=True)
    return compressed, bcftools


def _native_records(bcftools: str, vcf: Path, *args: str) -> int:
    result = subprocess.run(
        [bcftools, "view", "-H", *args, str(vcf)],
        check=True,
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line])


@pytest.mark.integration
def test_combined_and_per_region_counts_match_native_record_selection(indexed_variants, tmp_path: Path) -> None:
    vcf, bcftools = indexed_variants
    regions = [
        {"chrom": "chr1", "start": 0, "end": 10, "name": "left"},
        {"chrom": "chr1", "start": 4, "end": 13, "name": "middle"},
    ]
    combined = variant_qc({"path": str(vcf), "regions": regions}, tools=ToolPaths(bcftools=bcftools))
    grouped = variant_qc(
        {"path": str(vcf), "regions": regions, "group_by": "region"},
        tools=ToolPaths(bcftools=bcftools),
    )

    union_bed = tmp_path / "union.bed"
    union_bed.write_text("chr1\t0\t13\n")
    left_bed = tmp_path / "left.bed"
    left_bed.write_text("chr1\t0\t10\n")
    middle_bed = tmp_path / "middle.bed"
    middle_bed.write_text("chr1\t4\t13\n")
    assert combined.results[0].general is not None
    assert combined.results[0].general.total_records == _native_records(
        bcftools, vcf, "--regions-file", str(union_bed), "--regions-overlap", "1"
    )
    assert combined.results[0].general.total_records == 4
    assert [group.general.total_records if group.general is not None else None for group in grouped.results] == [
        _native_records(bcftools, vcf, "--regions-file", str(left_bed), "--regions-overlap", "1"),
        _native_records(bcftools, vcf, "--regions-file", str(middle_bed), "--regions-overlap", "1"),
    ]
    assert [group.general.total_records if group.general is not None else None for group in grouped.results] == [3, 3]


@pytest.mark.integration
def test_record_overlap_includes_an_indel_whose_pos_is_outside_the_interval(indexed_variants, tmp_path: Path) -> None:
    vcf, bcftools = indexed_variants
    region = {"chrom": "chr1", "start": 8, "end": 9, "name": "deleted-base"}
    result = variant_qc(
        {"path": str(vcf), "regions": [region], "metrics": ["general", "indels"]},
        tools=ToolPaths(bcftools=bcftools),
    )
    bed = tmp_path / "deleted-base.bed"
    bed.write_text("chr1\t8\t9\n")

    assert result.results[0].general is not None and result.results[0].indels is not None
    assert result.results[0].general.total_records == result.results[0].indels.count == 1
    assert _native_records(bcftools, vcf, "--regions-file", str(bed), "--regions-overlap", "1") == 1
    assert _native_records(bcftools, vcf, "--regions-file", str(bed), "--regions-overlap", "0") == 0


@pytest.mark.integration
def test_native_include_expression_and_advanced_arguments_are_effective(indexed_variants) -> None:
    vcf, bcftools = indexed_variants
    result = variant_qc(
        {
            "path": str(vcf),
            "selection": {"include_expression": "QUAL>=40"},
            "extra_args": {"bcftools_stats": ["--apply-filters", "PASS", "--1st-allele-only"]},
        },
        tools=ToolPaths(bcftools=bcftools),
    )

    assert result.results[0].general is not None
    assert result.results[0].general.total_records == _native_records(bcftools, vcf, "--include", "QUAL>=40") == 4
    assert result.provenance[0].native_options_used is True
    args = result.provenance[0].effective_args
    assert args[args.index("--include") + 1] == "QUAL>=40"
    assert args[-3:] == ["--apply-filters", "PASS", "--1st-allele-only"]


@pytest.mark.integration
def test_multiallelic_tstv_counts_are_allele_counts(indexed_variants) -> None:
    vcf, bcftools = indexed_variants
    result = variant_qc(
        {"path": str(vcf), "metrics": ["snps"]},
        tools=ToolPaths(bcftools=bcftools),
    )

    snps = result.results[0].snps
    assert snps is not None
    assert snps.model_dump() == {
        "count": 3,
        "transitions": 2,
        "transversions": 2,
        "ts_tv_ratio": 1.0,
    }
    assert snps.transitions is not None and snps.transversions is not None
    assert snps.transitions + snps.transversions > snps.count


@pytest.mark.integration
def test_zero_snp_selection_has_zero_allele_counts_and_null_ratio(indexed_variants) -> None:
    vcf, bcftools = indexed_variants
    result = variant_qc(
        {
            "path": str(vcf),
            "selection": {"include_expression": 'TYPE="indel"'},
            "metrics": ["general", "snps", "indels"],
        },
        tools=ToolPaths(bcftools=bcftools),
    )

    group = result.results[0]
    assert group.general is not None and group.snps is not None and group.indels is not None
    assert group.general.total_records == group.indels.count == 2
    assert group.snps.model_dump() == {
        "count": 0,
        "transitions": 0,
        "transversions": 0,
        "ts_tv_ratio": None,
    }


@pytest.mark.integration
def test_whole_file_reference_must_match_variant_header(indexed_variants, tmp_path: Path) -> None:
    vcf, bcftools = indexed_variants
    reference = tmp_path / "mismatched.fa"
    reference.write_text(">chr1\n" + "A" * 19 + "\n")
    Path(str(reference) + ".fai").write_text("chr1\t19\t6\t19\t20\n")

    with pytest.raises(ValueError, match="does not match variant contig"):
        variant_qc(
            {"path": str(vcf), "reference_path": str(reference)},
            tools=ToolPaths(bcftools=bcftools),
        )
