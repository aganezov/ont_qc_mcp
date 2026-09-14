"""Pinned samtools/cramino evidence for API v2 alignment QC."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.v2_alignment_qc import alignment_qc


pytestmark = pytest.mark.integration


@pytest.fixture
def alignment_qc_bam(tmp_path: Path) -> Path:
    require_executable_tools(["samtools", "cramino"])
    sam = tmp_path / "records.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:chr1\tLN:20\n"
        "known\t0\tchr1\t1\t30\t4M\t*\t0\t0\tAAAA\tIIII\tNM:i:1\tMD:Z:1C2\n"
        "missing\t2048\tchr1\t3\t255\t4M\t*\t0\t0\tCCCC\t*\tNM:i:0\tMD:Z:4\n"
        "secondary\t256\tchr1\t5\t20\t4M\t*\t0\t0\tGGGG\tIIII\tNM:i:0\tMD:Z:4\n"
        "duplicate\t1024\tchr1\t6\t20\t4M\t*\t0\t0\tTTTT\tIIII\tNM:i:0\tMD:Z:4\n"
        "qcfail\t512\tchr1\t7\t20\t4M\t*\t0\t0\tACAC\tIIII\tNM:i:0\tMD:Z:4\n"
        "complex\t0\tchr1\t10\t10\t1S2M1I1M1D1N2=1X1S\t*\t0\t0\tACGTACGTA\tBCDEFGHIJ\tNM:i:2\n"
        "unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tGGG\tIII\n"
    )
    bam = tmp_path / "records.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    return bam


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        per_tool_timeouts={"samtools": 30, "cramino": 30},
        per_tool_threads={},
        default_threads=None,
    )


def _tools() -> ToolPaths:
    return ToolPaths(samtools="samtools", cramino="cramino")


def test_whole_file_defaults_and_unmapped_eligibility_without_an_index(
    alignment_qc_bam: Path,
    tmp_path: Path,
) -> None:
    unindexed = tmp_path / "unindexed.bam"
    shutil.copyfile(alignment_qc_bam, unindexed)
    default = alignment_qc({"path": str(unindexed)}, tools=_tools(), exec_cfg=_config())
    [group] = default.results
    assert group.counts is not None
    assert group.counts.model_dump() == {
        "eligible_records": 3,
        "mapped_records": 3,
        "unmapped_records": 0,
        "secondary_records": 0,
        "supplementary_records": 1,
    }
    assert group.mapping_quality is not None
    assert group.mapping_quality.model_dump() == {
        "known_records": 2,
        "missing_records": 1,
        "mean_mapq": 20.0,
    }
    assert [value.backend for value in default.provenance] == [
        "samtools_view",
        "alignment_record_accumulator",
    ]

    with_unmapped = alignment_qc(
        {
            "path": str(unindexed),
            "selection": {"exclude_flags": 1792, "include_unmapped": True},
        },
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert with_unmapped.results[0].counts is not None
    assert with_unmapped.results[0].counts.eligible_records == 4
    assert with_unmapped.results[0].counts.unmapped_records == 1


def test_overlapping_region_groups_and_combined_union_preserve_base_denominators(
    alignment_qc_bam: Path,
) -> None:
    regions = [
        {"chrom": "chr1", "start": 0, "end": 5, "name": "left"},
        {"chrom": "chr1", "start": 2, "end": 12, "name": "right"},
    ]
    grouped = alignment_qc(
        {
            "path": str(alignment_qc_bam),
            "regions": regions,
            "group_by": "region",
            "metrics": ["counts", "mapping_quality", "aligned_base_quality"],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert [(value.region_id, value.region_name) for value in grouped.results] == [
        ("region_1", "left"),
        ("region_2", "right"),
    ]
    assert [value.counts.eligible_records for value in grouped.results if value.counts is not None] == [2, 3]
    assert [value.mapping_quality.mean_mapq for value in grouped.results if value.mapping_quality is not None] == [
        30,
        20,
    ]
    assert [
        value.aligned_base_quality.model_dump() for value in grouped.results if value.aligned_base_quality is not None
    ] == [
        {
            "aligned_query_bases": 7,
            "known_quality_bases": 4,
            "missing_quality_bases": 3,
            "mean_base_quality": 40.0,
        },
        {
            "aligned_query_bases": 9,
            "known_quality_bases": 5,
            "missing_quality_bases": 4,
            "mean_base_quality": pytest.approx(37.2),
        },
    ]

    combined = alignment_qc(
        {
            "path": str(alignment_qc_bam),
            "regions": regions,
            "metrics": ["counts", "aligned_base_quality"],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert combined.results[0].counts is not None
    assert combined.results[0].counts.eligible_records == 3
    assert combined.results[0].aligned_base_quality is not None
    assert combined.results[0].aligned_base_quality.model_dump() == {
        "aligned_query_bases": 11,
        "known_quality_bases": 7,
        "missing_quality_bases": 4,
        "mean_base_quality": 38.0,
    }


def test_native_identity_and_nm_error_use_the_same_selected_whole_records(
    alignment_qc_bam: Path,
    tmp_path: Path,
) -> None:
    request = {
        "regions": [{"chrom": "chr1", "start": 0, "end": 8}],
        "metrics": ["identity", "error_profile"],
    }
    response = alignment_qc(
        {
            "path": str(alignment_qc_bam),
            **request,
        },
        tools=_tools(),
        exec_cfg=_config(),
    )
    [group] = response.results
    assert group.identity is not None
    assert group.identity.mean_identity == pytest.approx(0.875)
    assert group.identity.scope == "whole_selected_record"
    assert group.error_profile is not None
    assert group.error_profile.nm_error_rate == pytest.approx(0.125)
    assert group.error_profile.mismatch_rate is None
    assert group.error_profile.scope == "whole_selected_record"
    assert [value.backend for value in response.provenance] == [
        "samtools_view",
        "cramino",
        "samtools_view",
        "samtools_stats",
    ]

    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\n" + "A" * 20 + "\n")
    subprocess.run(["samtools", "faidx", str(reference)], check=True, capture_output=True)
    cram = tmp_path / "records.cram"
    subprocess.run(
        ["samtools", "view", "-C", "-T", str(reference), "-o", str(cram), str(alignment_qc_bam)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["samtools", "index", str(cram)], check=True, capture_output=True)
    cram_response = alignment_qc(
        {"path": str(cram), "reference_path": str(reference), **request},
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert cram_response.results[0].identity == group.identity
    assert cram_response.results[0].error_profile is not None
    assert cram_response.results[0].error_profile.nm_error_rate == group.error_profile.nm_error_rate
    assert cram_response.results[0].error_profile.mismatch_counts_by_cycle is not None


def test_grouped_native_metrics_keep_requested_order_and_whole_record_scope(
    alignment_qc_bam: Path,
) -> None:
    response = alignment_qc(
        {
            "path": str(alignment_qc_bam),
            "regions": [
                {"chrom": "chr1", "start": 0, "end": 8, "name": "two-records"},
                {"chrom": "chr1", "start": 9, "end": 18, "name": "complex"},
            ],
            "group_by": "region",
            "metrics": ["identity", "error_profile"],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )

    assert [(value.region_id, value.region_name) for value in response.results] == [
        ("region_1", "two-records"),
        ("region_2", "complex"),
    ]
    assert response.results[0].identity is not None
    assert response.results[0].identity.mean_identity == pytest.approx(0.875)
    assert response.results[0].error_profile is not None
    assert response.results[0].error_profile.nm_error_rate == pytest.approx(0.125)
    assert all(
        value.identity is not None and value.identity.scope == "whole_selected_record" for value in response.results
    )
    assert all(
        value.error_profile is not None and value.error_profile.scope == "whole_selected_record"
        for value in response.results
    )


def test_error_profile_uses_reference_symlink_with_adjacent_index(
    alignment_qc_bam: Path,
    tmp_path: Path,
) -> None:
    reference_target = tmp_path / "reference-target.fa"
    reference_target.write_text(">chr1\n" + "A" * 20 + "\n")
    reference_alias = tmp_path / "reference-alias.fa"
    reference_alias.symlink_to(reference_target)
    subprocess.run(["samtools", "faidx", str(reference_alias)], check=True, capture_output=True)

    response = alignment_qc(
        {
            "path": str(alignment_qc_bam),
            "reference_path": str(reference_alias),
            "metrics": ["error_profile"],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )

    assert response.results[0].error_profile is not None
    assert response.results[0].error_profile.mismatch_counts_by_cycle is not None
    assert Path(str(reference_alias) + ".fai").exists()
    assert not Path(str(reference_target) + ".fai").exists()
