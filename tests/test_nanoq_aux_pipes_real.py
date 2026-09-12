"""Compare FIFO reports with real nanoq output written to ordinary files."""

import math
import shutil
import subprocess
from collections import Counter

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.cli_wrappers import nanoq_from_bam_streaming, nanoq_stats
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.parsers import parse_nanoq_json
from ont_qc_mcp.schemas import HistogramBin, LengthPercentiles


pytestmark = pytest.mark.integration


def file_histogram(values, width):
    # Independent reference, including dense zero bins and the existing float
    # floor-division convention at bin boundaries.
    counts = Counter(int(value // width) if value >= 0 else 0 for value in values)
    return [
        HistogramBin(start=i * width, end=(i + 1) * width, count=counts[i]) for i in range(max(counts, default=-1) + 1)
    ]


@pytest.mark.parametrize("bam_input", [False, True])
@pytest.mark.parametrize("percentile_limit", [2, 3])
def test_real_fifo_equals_independent_normal_file_baseline(tmp_path, bam_input, percentile_limit):
    require_executable_tools(["samtools", "nanoq"])
    samtools, nanoq = shutil.which("samtools"), shutil.which("nanoq")
    assert samtools is not None and nanoq is not None
    paths = ToolPaths(samtools=samtools, nanoq=nanoq)
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@r1\nAC\n+\n:;\n@r2\nACG\n+\n!!!\n@r3\nACGT\n+\nIIII\n")
    lengths_path, qualities_path = tmp_path / "lengths.txt", tmp_path / "qualities.txt"
    normal = subprocess.run(
        [
            nanoq,
            "--stats",
            "--json",
            "--input",
            str(fastq),
            "--read-lengths",
            str(lengths_path),
            "--read-qualities",
            str(qualities_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    expected = parse_nanoq_json(normal.stdout)
    lengths = [float(row) for row in lengths_path.read_text().splitlines()]
    qualities = [float(row) for row in qualities_path.read_text().splitlines()]
    # Nanoq emits one-decimal auxiliary Q-scores. The first crosses the 25.5
    # boundary after rounding; computing from raw qualities would differ.
    assert qualities == [-0.0, 25.5, 40.0]
    expected.length_histogram = file_histogram(lengths, 2)
    expected.qscore_histogram = file_histogram(qualities, 0.5)
    if len(lengths) <= percentile_limit:
        ordered = sorted(lengths)
        percentiles = {}
        for percentile in (1, 5, 25, 50, 75, 95, 99):
            position = percentile / 100 * (len(ordered) - 1)
            lower, upper = math.floor(position), math.ceil(position)
            percentiles[f"p{percentile}"] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        expected.length_percentiles = LengthPercentiles(**percentiles)
    else:
        expected.length_percentiles = None

    cfg = ExecutionConfig(
        nanoq_length_bin_width=2, nanoq_qscore_bin_width=0.5, nanoq_percentiles_exact_max_reads=percentile_limit
    )
    if bam_input:
        sam = tmp_path / "reads.sam"
        sam.write_text(
            "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:100\n"
            "r1\t4\t*\t0\t0\t*\t*\t0\t0\tAC\t:;\n"
            "r2\t4\t*\t0\t0\t*\t*\t0\t0\tACG\t!!!\n"
            "r3\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII\n"
        )
        bam = tmp_path / "reads.bam"
        subprocess.run([samtools, "view", "-b", "-o", str(bam), str(sam)], check=True)
        actual = nanoq_from_bam_streaming(bam, paths, exec_cfg=cfg)
        expected.file = str(bam)
    else:
        actual = nanoq_stats(fastq, paths, exec_cfg=cfg)
    assert actual.model_dump() == expected.model_dump()
