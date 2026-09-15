"""Direct mosdepth comparisons for the public API v2 coverage backend."""

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.v2_contracts import CoverageQCResponse
from ont_qc_mcp.v2_coverage_qc import coverage_qc


@pytest.fixture
def indexed_coverage_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    require_executable_tools(["samtools", "mosdepth"])
    samtools = shutil.which("samtools")
    assert samtools is not None
    reference = tmp_path / "reference.fa"
    reference.write_text(f">chr1\n{'A' * 12}\n>chr2\n{'A' * 5}\n")
    subprocess.run([samtools, "faidx", str(reference)], check=True, capture_output=True)
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:chr1\tLN:12\n"
        "@SQ\tSN:chr2\tLN:5\n"
        "@RG\tID:RG1\tSM:sample\n"
        "@RG\tID:RG2\tSM:sample\n"
        "hi\t0\tchr1\t1\t60\t4M\t*\t0\t0\tAAAA\tIIII\n"
        "low\t0\tchr1\t3\t5\t4M\t*\t0\t0\tAAAA\tIIII\n"
        "secondary\t256\tchr1\t9\t60\t2M\t*\t0\t0\tAA\tII\n"
        "rg1\t0\tchr2\t1\t60\t2M\t*\t0\t0\tAA\tII\tRG:Z:RG1\n"
        "rg2\t0\tchr2\t3\t60\t2M\t*\t0\t0\tAA\tII\tRG:Z:RG2\n"
    )
    bam = tmp_path / "reads.bam"
    cram = tmp_path / "reads.cram"
    subprocess.run([samtools, "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run([samtools, "index", str(bam)], check=True, capture_output=True)
    subprocess.run(
        [samtools, "view", "-C", "-T", str(reference), "-o", str(cram), str(sam)],
        check=True,
        capture_output=True,
    )
    subprocess.run([samtools, "index", str(cram)], check=True, capture_output=True)
    return bam, cram, reference


def _raw_mosdepth_rows(
    result: CoverageQCResponse,
    bam: Path,
    mosdepth: str,
    tmp_path: Path,
) -> tuple[dict[str, int], dict[str, float], dict[str, list[int]]]:
    bed = tmp_path / f"direct-{result.resolved_group_by}.bed"
    bed.write_text(
        "".join(f"{row.chrom}\t{row.start}\t{row.end}\t{row.row_id}\n" for row in result.rows),
        encoding="utf-8",
    )
    prefix = tmp_path / f"direct-{result.resolved_group_by}"
    thresholds = sorted(result.effective_request.thresholds)
    subprocess.run(
        [
            mosdepth,
            "--use-median",
            "--by",
            str(bed),
            "--thresholds",
            ",".join(map(str, thresholds)),
            str(prefix),
            str(bam),
        ],
        check=True,
        capture_output=True,
    )

    medians: dict[str, float] = {}
    with gzip.open(prefix.with_suffix(".regions.bed.gz"), "rt") as stream:
        for line in stream:
            fields = line.rstrip().split("\t")
            medians[fields[3]] = float(fields[4])

    arrays = {"chr1": [0] * 12, "chr2": [0] * 5}
    with gzip.open(prefix.with_suffix(".per-base.bed.gz"), "rt") as stream:
        for line in stream:
            chrom, raw_start, raw_end, raw_depth = line.rstrip().split("\t")
            start, end, depth = int(raw_start), int(raw_end), int(raw_depth)
            arrays[chrom][start:end] = [depth] * (end - start)
    depth_sums = {row.row_id: sum(arrays[row.chrom][row.start : row.end]) for row in result.rows}

    breadth: dict[str, list[int]] = {}
    with gzip.open(prefix.with_suffix(".thresholds.bed.gz"), "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            breadth[fields[3]] = [int(value) for value in fields[4:]]
    requested_order_counts = {
        row_id: [dict(zip(thresholds, counts, strict=True))[value] for value in result.effective_request.thresholds]
        for row_id, counts in breadth.items()
    }
    return depth_sums, medians, requested_order_counts


@pytest.mark.integration
@pytest.mark.parametrize(
    (
        "request_payload",
        "expected_ids",
        "expected_depths",
        "expected_medians",
        "expected_union_bases",
        "expected_union_depth",
    ),
    [
        ({}, ["contig.chr1", "contig.chr2"], [8, 4], [0.0, 1.0], 17, 12),
        (
            {
                "regions": [
                    {"chrom": "chr1", "start": 0, "end": 6, "name": "first"},
                    {"chrom": "chr1", "start": 3, "end": 9, "name": "overlap"},
                    {"chrom": "chr2", "start": 1, "end": 5, "name": "last"},
                ]
            },
            ["region_1", "region_2", "region_3"],
            [8, 4, 3],
            [1.0, 0.0, 1.0],
            13,
            11,
        ),
        (
            {"window_size": 5},
            ["chr1.window_1", "chr1.window_2", "chr1.window_3", "chr2.window_1"],
            [7, 1, 0, 4],
            [1.0, 0.0, 0.0, 1.0],
            17,
            12,
        ),
        (
            {
                "regions": [
                    {"chrom": "chr1", "start": 0, "end": 6, "name": "first"},
                    {"chrom": "chr1", "start": 3, "end": 9, "name": "overlap"},
                    {"chrom": "chr2", "start": 1, "end": 5, "name": "last"},
                ],
                "window_size": 4,
            },
            [
                "region_1.window_1",
                "region_1.window_2",
                "region_2.window_1",
                "region_2.window_2",
                "region_3.window_1",
            ],
            [6, 2, 4, 0, 3],
            [1.0, 1.0, 1.0, 0.0, 1.0],
            13,
            11,
        ),
    ],
)
def test_all_modes_match_separate_raw_mosdepth(
    indexed_coverage_inputs,
    tmp_path,
    request_payload,
    expected_ids,
    expected_depths,
    expected_medians,
    expected_union_bases,
    expected_union_depth,
) -> None:
    bam, _, _ = indexed_coverage_inputs
    mosdepth = shutil.which("mosdepth")
    samtools = shutil.which("samtools")
    assert mosdepth is not None and samtools is not None
    result = coverage_qc(
        {
            "path": str(bam),
            "depth_statistic": "median",
            "thresholds": [2, 0, 1, 10],
            **request_payload,
        },
        tools=ToolPaths(mosdepth=mosdepth, samtools=samtools),
    )

    assert [row.row_id for row in result.rows] == expected_ids
    assert [row.depth_sum for row in result.rows] == expected_depths
    assert [row.median_depth for row in result.rows] == expected_medians
    assert result.union_summary.reference_bases == expected_union_bases
    assert result.union_summary.depth_sum == expected_union_depth
    direct_depths, direct_medians, direct_breadth = _raw_mosdepth_rows(result, bam, mosdepth, tmp_path)
    for row in result.rows:
        assert row.depth_sum == direct_depths[row.row_id]
        assert row.median_depth == direct_medians[row.row_id]
        assert [value.bases_at_or_above for value in row.breadth] == direct_breadth[row.row_id]


@pytest.mark.integration
def test_native_median_uses_lower_middle_and_retains_zero_depth_bases(tmp_path: Path) -> None:
    require_executable_tools(["samtools", "mosdepth"])
    samtools = shutil.which("samtools")
    mosdepth = shutil.which("mosdepth")
    assert samtools is not None and mosdepth is not None
    sam = tmp_path / "median.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:odd\tLN:5\n"
        "@SQ\tSN:even\tLN:4\n"
        "@SQ\tSN:empty\tLN:3\n"
        "@SQ\tSN:skew\tLN:5\n"
        "odd-read\t0\todd\t1\t60\t3M\t*\t0\t0\tAAA\tIII\n"
        "even-read\t0\teven\t1\t60\t2M\t*\t0\t0\tAA\tII\n"
        + "".join(f"ten-{index}\t0\tskew\t4\t60\t1M\t*\t0\t0\tA\tI\n" for index in range(10))
        + "".join(f"forty-{index}\t0\tskew\t5\t60\t1M\t*\t0\t0\tA\tI\n" for index in range(40))
    )
    bam = tmp_path / "median.bam"
    subprocess.run([samtools, "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run([samtools, "index", str(bam)], check=True, capture_output=True)

    result = coverage_qc(
        {"path": str(bam), "metrics": ["depth"], "depth_statistic": "median"},
        tools=ToolPaths(mosdepth=mosdepth, samtools=samtools),
    )

    assert [row.depth_sum for row in result.rows] == [3, 2, 0, 50]
    assert [row.mean_depth for row in result.rows] == [pytest.approx(0.6), pytest.approx(0.5), 0.0, 10.0]
    # Pinned mosdepth 0.3.14 chooses the lower middle value for even-length regions.
    # The skew row has per-base depths [0, 0, 0, 10, 40], so its median is 0.
    assert [row.median_depth for row in result.rows] == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.integration
def test_threshold_zero_covers_a_contig_with_no_alignment_records(tmp_path: Path) -> None:
    require_executable_tools(["samtools", "mosdepth"])
    samtools = shutil.which("samtools")
    mosdepth = shutil.which("mosdepth")
    assert samtools is not None and mosdepth is not None
    sam = tmp_path / "empty.sam"
    sam.write_text("@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chrEmpty\tLN:5\n")
    bam = tmp_path / "empty.bam"
    subprocess.run([samtools, "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run([samtools, "index", str(bam)], check=True, capture_output=True)

    result = coverage_qc(
        {"path": str(bam), "metrics": ["breadth"], "thresholds": [0, 1]},
        tools=ToolPaths(mosdepth=mosdepth, samtools=samtools),
    )

    assert len(result.rows) == 1
    assert result.rows[0].reference_bases == 5
    assert [entry.bases_at_or_above for entry in result.rows[0].breadth] == [5, 0]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({"min_mapq": 10}, [4, 4]),
        ({"exclude_flags": 0}, [10, 4]),
        ({"include_flags": 256, "exclude_flags": 0}, [2, 0]),
        ({"read_group": "RG1"}, [0, 2]),
    ],
)
def test_mosdepth_native_selection_matches_hand_counted_depths(indexed_coverage_inputs, selection, expected) -> None:
    bam, _, _ = indexed_coverage_inputs
    mosdepth = shutil.which("mosdepth")
    samtools = shutil.which("samtools")
    assert mosdepth is not None and samtools is not None
    result = coverage_qc(
        {"path": str(bam), "metrics": ["depth"], "selection": selection},
        tools=ToolPaths(mosdepth=mosdepth, samtools=samtools),
    )
    assert [row.depth_sum for row in result.rows] == expected


@pytest.mark.integration
def test_indexed_cram_with_explicit_reference_matches_bam(indexed_coverage_inputs) -> None:
    bam, cram, reference = indexed_coverage_inputs
    mosdepth = shutil.which("mosdepth")
    samtools = shutil.which("samtools")
    assert mosdepth is not None and samtools is not None
    tools = ToolPaths(mosdepth=mosdepth, samtools=samtools)
    request = {
        "regions": [{"chrom": "chr1", "start": 0, "end": 9}, {"chrom": "chr2", "start": 1, "end": 5}],
        "window_size": 4,
    }
    bam_result = coverage_qc({"path": str(bam), **request}, tools=tools)
    cram_result = coverage_qc({"path": str(cram), "reference_path": str(reference), **request}, tools=tools)

    assert cram_result.rows == bam_result.rows
    assert cram_result.union_summary == bam_result.union_summary
    assert "--fasta" in cram_result.provenance[0].effective_args
