import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.v2_execution import PipelineStageError
from ont_qc_mcp.v2_read_qc import read_qc


pytestmark = pytest.mark.integration


@pytest.fixture
def read_qc_inputs(tmp_path: Path) -> dict[str, Path]:
    require_executable_tools(["samtools", "nanoq"])
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\n" + "A" * 100 + "\n")
    subprocess.run(["samtools", "faidx", str(reference)], check=True, capture_output=True)

    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:chr1\tLN:100\n"
        "soft\t0\tchr1\t1\t30\t2S4M\t*\t0\t0\tTTAAAA\tIIIIII\n"
        "secondary\t256\tchr1\t2\t30\t4M\t*\t0\t0\tGGGG\tIIII\n"
        "supplementary\t2048\tchr1\t2\t30\t4M\t*\t0\t0\tTTTT\tIIII\n"
        "overlap\t0\tchr1\t4\t20\t4M\t*\t0\t0\tCCCC\tIIII\n"
        "unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tNNN\tIII\n"
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    cram = tmp_path / "reads.cram"
    subprocess.run(
        ["samtools", "view", "-C", "-T", str(reference), "-o", str(cram), str(sam)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["samtools", "index", str(cram)], check=True, capture_output=True)

    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@soft\nTTAAAA\n+\nIIIIII\n@overlap\nCCCC\n+\nIIII\n")
    return {"reference": reference, "bam": bam, "cram": cram, "fastq": fastq}


def _tools() -> ToolPaths:
    return ToolPaths(samtools="samtools", nanoq="nanoq")


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        per_tool_timeouts={"samtools": 30, "nanoq": 30},
        per_tool_threads={},
        nanoq_aux_stats=True,
    )


def test_fastq_bam_and_cram_metrics_agree_for_the_same_primary_sequences(
    read_qc_inputs: dict[str, Path],
) -> None:
    metrics = ["length", "read_quality", "length_distribution", "quality_distribution"]
    region = [{"chrom": "chr1", "start": 0, "end": 8}]
    fastq = read_qc(
        {"path": str(read_qc_inputs["fastq"]), "metrics": metrics},
        tools=_tools(),
        exec_cfg=_config(),
    )
    bam = read_qc(
        {"path": str(read_qc_inputs["bam"]), "regions": region, "metrics": metrics},
        tools=_tools(),
        exec_cfg=_config(),
    )
    cram = read_qc(
        {
            "path": str(read_qc_inputs["cram"]),
            "reference_path": str(read_qc_inputs["reference"]),
            "regions": region,
            "metrics": metrics,
        },
        tools=_tools(),
        exec_cfg=_config(),
    )

    assert bam.selected_records == bam.emitted_sequences == 2
    assert cram.selected_records == cram.emitted_sequences == 2
    for response in (bam, cram):
        assert response.results[0].length == fastq.results[0].length
        assert response.results[0].read_quality == fastq.results[0].read_quality
        assert response.results[0].length_distribution == fastq.results[0].length_distribution
        assert response.results[0].quality_distribution == fastq.results[0].quality_distribution
        assert response.results[0].length is not None
        assert response.results[0].length.total_bases == 10
        assert response.results[0].length.max_length == 6  # complete sequence includes the 2S soft clip


def test_overlapping_regions_keep_union_counts_unique_and_region_rows_ordered(
    read_qc_inputs: dict[str, Path],
) -> None:
    regions = [
        {"chrom": "chr1", "start": 0, "end": 5, "name": "left"},
        {"chrom": "chr1", "start": 3, "end": 8, "name": "right"},
    ]
    response = read_qc(
        {
            "path": str(read_qc_inputs["bam"]),
            "regions": regions,
            "group_by": "region",
            "metrics": ["length"],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )

    assert response.selected_records == response.emitted_sequences == 2
    assert [(result.region_id, result.region_name) for result in response.results] == [
        ("region_1", "left"),
        ("region_2", "right"),
    ]
    assert [result.length.read_count for result in response.results if result.length is not None] == [2, 2]


def test_empty_regional_selection_returns_zero_and_null_summaries(read_qc_inputs: dict[str, Path]) -> None:
    response = read_qc(
        {
            "path": str(read_qc_inputs["bam"]),
            "regions": [{"chrom": "chr1", "start": 90, "end": 100}],
        },
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert response.selected_records == response.emitted_sequences == response.conversion_exclusions == 0
    assert response.results[0].length is not None
    assert response.results[0].length.model_dump() == {
        "read_count": 0,
        "total_bases": 0,
        "min_length": None,
        "max_length": None,
        "mean_length": None,
        "median_length": None,
        "n50": None,
    }
    assert response.results[0].read_quality is not None
    assert response.results[0].read_quality.mean_qscore is None
    assert response.results[0].read_quality.median_qscore is None


def test_missing_sequence_is_counted_but_missing_quality_rejects_quality_metrics(tmp_path: Path) -> None:
    require_executable_tools(["samtools", "nanoq"])
    sam = tmp_path / "missing.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:chr1\tLN:20\n"
        "good\t0\tchr1\t1\t30\t4M\t*\t0\t0\tAAAA\tIIII\n"
        "missing_sequence\t0\tchr1\t5\t30\t4M\t*\t0\t0\t*\t*\n"
        "missing_quality\t0\tchr1\t9\t30\t4M\t*\t0\t0\tCCCC\t*\n"
    )
    bam = tmp_path / "missing.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)

    length_only = read_qc(
        {"path": str(bam), "metrics": ["length"]},
        tools=_tools(),
        exec_cfg=_config(),
    )
    assert length_only.selected_records == 3
    assert length_only.emitted_sequences == 2
    assert length_only.conversion_exclusions == 1
    assert length_only.results[0].length is not None
    assert length_only.results[0].length.total_bases == 8

    with pytest.raises(PipelineStageError) as caught:
        read_qc({"path": str(bam)}, tools=_tools(), exec_cfg=_config())
    assert caught.value.stage == "read_record_filter"
    assert "absent QUAL" in caught.value.result.stderr
