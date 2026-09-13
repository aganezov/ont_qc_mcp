"""Direct-native checks for shared v2 samtools selection."""

import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.v2_execution import PipelineStage, RequestDeadline, run_pipeline
from ont_qc_mcp.v2_regions import normalize_regions
from ont_qc_mcp.v2_samtools import SamtoolsSelection, build_fastq_stage, resolve_alignment_input, samtools_view_plan


pytestmark = pytest.mark.integration


@pytest.fixture
def selected_bam(tmp_path: Path) -> Path:
    require_executable_tools(["samtools"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:chr1\tLN:30\n"
        "@RG\tID:rg1\tSM:sample\n"
        "r1\t0\tchr1\t1\t0\t5M\t*\t0\t0\tAAAAA\tIIIII\tNM:i:0\tRG:Z:rg1\n"
        "secondary\t256\tchr1\t2\t60\t2M\t*\t0\t0\tGG\tII\tNM:i:0\tRG:Z:rg1\n"
        "r2\t0\tchr1\t6\t30\t5M\t*\t0\t0\tCCCCC\tIIIII\tNM:i:1\tRG:Z:rg1\n"
        "missing_mapq\t0\tchr1\t11\t255\t5M\t*\t0\t0\tACGTA\tIIIII\tNM:i:0\tRG:Z:rg1\n"
        "r3\t0\tchr1\t16\t20\t3M\t*\t0\t0\tTTT\tIII\tNM:i:0\tRG:Z:rg1\n"
        "unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tNN\tII\n"
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    return bam


def test_union_and_per_region_selection_match_direct_native_counts(selected_bam: Path, tmp_path: Path) -> None:
    regions = normalize_regions(
        [
            {"chrom": "chr1", "start": 0, "end": 6, "name": "left"},
            {"chrom": "chr1", "start": 4, "end": 10, "name": "overlap"},
            {"chrom": "chr1", "start": 15, "end": 20, "name": "right"},
        ],
        {"chr1": 30},
    )
    alignment = resolve_alignment_input(str(selected_bam), require_index=True)
    tools = ToolPaths(samtools="samtools")
    selection = SamtoolsSelection(exclude_flags=256, include_unmapped=False)
    counter = PipelineStage("count", ("samtools", "view", "-c", "-"))

    with samtools_view_plan(alignment, regions, selection, tools=tools) as plan:
        selected = run_pipeline([plan.stage, counter], RequestDeadline(10))
        union_bed = Path(plan.stage.command[plan.stage.command.index("-L") + 1])
        direct = subprocess.check_output(
            ["samtools", "view", "-c", "-M", "-L", str(union_bed), "-F", "260", str(selected_bam)],
            text=True,
        )
        assert int(selected.final.stdout) == int(direct) == 3

    counts = []
    for region in regions.per_region():
        with samtools_view_plan(alignment, region, selection, tools=tools) as plan:
            counts.append(int(run_pipeline([plan.stage, counter], RequestDeadline(10)).final.stdout))
    assert counts == [2, 2, 1]


@pytest.mark.parametrize(
    ("regions", "min_mapq", "expected"),
    [
        (None, 0, 4),
        (None, 20, 2),
        (None, 21, 1),
        ([{"chrom": "chr1", "start": 0, "end": 15}], 0, 3),
        ([{"chrom": "chr1", "start": 0, "end": 15}], 20, 1),
    ],
)
def test_mapq_threshold_treats_255_as_missing(
    selected_bam: Path, regions: object | None, min_mapq: int, expected: int
) -> None:
    normalized = normalize_regions(regions, {"chr1": 30})
    alignment = resolve_alignment_input(str(selected_bam), require_index=regions is not None)
    tools = ToolPaths(samtools="samtools")
    with samtools_view_plan(
        alignment,
        normalized,
        SamtoolsSelection(min_mapq=min_mapq, primary_only=True),
        tools=tools,
    ) as plan:
        result = run_pipeline(
            [plan.stage, PipelineStage("count", ("samtools", "view", "-c", "-"))],
            RequestDeadline(10),
        )
    assert int(result.final.stdout) == expected


def test_selection_preserves_tags_and_fastq_adds_no_second_filter(selected_bam: Path) -> None:
    regions = normalize_regions([{"chrom": "chr1", "start": 0, "end": 10}], {"chr1": 30})
    alignment = resolve_alignment_input(str(selected_bam), require_index=True)
    tools = ToolPaths(samtools="samtools")
    with samtools_view_plan(
        alignment,
        regions,
        SamtoolsSelection(primary_only=True, include_unmapped=True),
        tools=tools,
    ) as plan:
        sam = run_pipeline(
            [plan.stage, PipelineStage("sam", ("samtools", "view", "-h", "-"))], RequestDeadline(10)
        ).final.stdout
        records = [line for line in sam.splitlines() if line and not line.startswith("@")]
        assert len(records) == 2
        assert all("NM:i:" in line and "RG:Z:rg1" in line for line in records)

    with samtools_view_plan(
        alignment,
        regions,
        SamtoolsSelection(primary_only=True, include_unmapped=True),
        tools=tools,
    ) as plan:
        fastq = run_pipeline([plan.stage, build_fastq_stage(tools)], RequestDeadline(10)).final.stdout
        assert sum(line.startswith("@") for line in fastq.splitlines()) == 2
