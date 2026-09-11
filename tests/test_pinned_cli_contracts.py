"""Regression checks against the supported nanoq, chopper, and cramino contracts."""

import json
from pathlib import Path

import pytest

from ont_qc_mcp import cli_wrappers
from ont_qc_mcp.cli_wrappers import FlagValidationError, build_cli_args
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.flag_schemas import get_tool_recipes
from ont_qc_mcp.parsers import parse_cramino_json


def test_cramino_json_keeps_read_and_base_counts_and_open_bins():
    payload = {
        "file_info": {"path": "known.bam"},
        "alignment_stats": {"num_reads": 3},
        "read_stats": {"mean_length": 2150, "median_length": 2100, "n50": 4300},
        "identity_stats": {"mean_identity": 99, "median_identity": 99},
        "histograms": {
            "read_length": {
                "bins": [
                    {"start": 0, "end": 2000, "count": 1, "bases": 50},
                    {"start": 2000, "end": 4000, "count": 1, "bases": 2100},
                    {"start": 4000, "end": 6000, "count": 1, "bases": 4300},
                    {"start": 10000, "count": 0, "bases": 0},
                ]
            },
            "q_score": {
                "bins": [
                    {"start": 16, "end": 17, "count": 1, "bases": 4300},
                    {"start": 19, "end": 20, "count": 1, "bases": 2100},
                    {"start": 40, "count": 1, "bases": 50},
                ]
            },
        },
    }
    parsed = parse_cramino_json(json.dumps(payload))
    assert parsed.length_histogram is not None
    assert parsed.qscore_histogram is not None
    assert [(b.count, b.bases) for b in parsed.length_histogram] == [(1, 50), (1, 2100), (1, 4300), (0, 0)]
    assert parsed.length_histogram[-1].end is None
    assert parsed.qscore_histogram[-1].end is None
    assert sum(b.count for b in parsed.qscore_histogram) == 3
    assert sum(b.bases for b in parsed.qscore_histogram) == 6450
    assert parsed.mean_identity == 99
    assert not {"mapq_histogram", "mapq_histogram_scaled", "length_histogram_scaled"} & parsed.model_dump().keys()


@pytest.mark.parametrize("flag", ["hist", "scaled", "mapq", "flags", "format"])
def test_cramino_rejects_output_controls(flag):
    with pytest.raises(FlagValidationError, match=flag):
        build_cli_args("cramino", {flag: "json" if flag == "format" else True})


@pytest.mark.parametrize("wrapper", [cli_wrappers.nanoq_stats, cli_wrappers.nanoq_from_bam_streaming])
@pytest.mark.parametrize("source", ["flags", "config", "environment"])
def test_nanoq_threads_rejected_before_start(monkeypatch, wrapper, source):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid flags must fail before starting any CLI")

    monkeypatch.setattr(cli_wrappers, "run_command_with_retry", unexpected)
    monkeypatch.setattr(cli_wrappers.subprocess, "Popen", unexpected)
    if source == "environment":
        monkeypatch.setenv("MCP_THREADS_NANOQ", "2")
    cfg = ExecutionConfig(per_tool_threads={"nanoq": 2}) if source == "config" else ExecutionConfig()
    with pytest.raises(FlagValidationError, match="nanoq.*threads|threads.*nanoq"):
        wrapper(Path("unused.bam"), ToolPaths(), flags={"threads": 2} if source == "flags" else None, exec_cfg=cfg)


@pytest.fixture
def known_reads(tmp_path):
    import subprocess
    from conftest import require_executable_tools

    require_executable_tools(["samtools"])
    sam = tmp_path / "known.sam"
    fastq = tmp_path / "known.fastq"
    records = [(50, 0), (2100, 21), (4300, 86)]
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:20000\n"
        + "".join(
            f"r{i}\t0\tchr1\t1\t60\t{n}M\t*\t0\t0\t{'A' * n}\t{'I' * n}\tNM:i:{nm}\n"
            for i, (n, nm) in enumerate(records)
        )
    )
    fastq.write_text("".join(f"@r{i}\n{'A' * n}\n+\n{'I' * n}\n" for i, (n, _) in enumerate(records)))
    bam = tmp_path / "known.bam"
    subprocess.run([ToolPaths().samtools, "view", "-b", "-o", str(bam), str(sam)], check=True)
    subprocess.run([ToolPaths().samtools, "index", str(bam)], check=True)
    return fastq, bam


@pytest.mark.integration
def test_cramino_real_known_lengths_identity_and_histograms(known_reads):
    from conftest import require_executable_tools

    require_executable_tools(["cramino"])
    _, bam = known_reads
    stats = cli_wrappers.cramino_stats(bam, ToolPaths())
    assert (stats.total_reads, stats.mean_length, stats.median_length, stats.n50) == (3, 2150, 2100, 4300)
    assert (stats.mean_identity, stats.median_identity) == (99, 99)
    assert stats.length_histogram is not None
    assert stats.qscore_histogram is not None
    assert [(b.start, b.end, b.count, b.bases) for b in stats.length_histogram if b.count] == [
        (0, 2000, 1, 50),
        (2000, 4000, 1, 2100),
        (4000, 6000, 1, 4300),
    ]
    assert sum(b.count for b in stats.qscore_histogram) == 3
    assert sum(b.bases for b in stats.qscore_histogram) == 6450
    # 98% identity gives Q~16.99, safely inside [16,17). All reads have MAPQ 60.
    bin_16 = next(b for b in stats.qscore_histogram if b.start == 16)
    assert (bin_16.count, bin_16.bases) == (1, 4300)
    no_hist = cli_wrappers.cramino_stats(bam, ToolPaths(), include_hist=False)
    assert no_hist.length_histogram is None
    assert no_hist.qscore_histogram is None


@pytest.mark.integration
def test_cramino_real_open_ended_bins(tmp_path):
    import subprocess
    from conftest import require_executable_tools

    require_executable_tools(["samtools", "cramino"])
    sam = tmp_path / "overflow.sam"
    sam.write_text(
        "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:20000\n"
        + f"r\t0\tchr1\t1\t60\t11000M\t*\t0\t0\t{'A' * 11000}\t{'I' * 11000}\tNM:i:0\n"
    )
    bam = tmp_path / "overflow.bam"
    subprocess.run([ToolPaths().samtools, "view", "-b", "-o", str(bam), str(sam)], check=True)
    stats = cli_wrappers.cramino_stats(bam, ToolPaths())
    # Cramino adapts the length maximum to 20000 for this input.
    assert stats.length_histogram is not None
    assert stats.qscore_histogram is not None
    assert stats.length_histogram[-1].end is None
    assert stats.length_histogram[-1].count == 0
    occupied = [b for b in stats.qscore_histogram if b.count]
    assert len(occupied) == 1
    assert (occupied[0].start, occupied[0].end, occupied[0].count, occupied[0].bases) == (40, None, 1, 11000)


@pytest.mark.integration
@pytest.mark.parametrize("recipe", list(get_tool_recipes("nanoq")))
@pytest.mark.parametrize("streaming", [False, True])
def test_nanoq_real_recipes(known_reads, recipe, streaming):
    from conftest import require_executable_tools

    require_executable_tools(["nanoq"])
    fastq, bam = known_reads
    wrapper = cli_wrappers.nanoq_from_bam_streaming if streaming else cli_wrappers.nanoq_stats
    stats = wrapper(bam if streaming else fastq, ToolPaths(), flags=get_tool_recipes("nanoq")[recipe])
    assert (stats.read_count, stats.total_bases) == (2, 6400)


@pytest.mark.integration
@pytest.mark.parametrize(
    "recipe,expected",
    [
        ("aggressive_trim", {"r1": 2000, "r2": 4200}),
        ("qual_trim", {"r0": 50, "r1": 2100, "r2": 4300}),
        ("inverse_short_reads", {"r0": 50}),
    ],
)
def test_chopper_real_recipes(known_reads, tmp_path, recipe, expected):
    from conftest import require_executable_tools

    require_executable_tools(["chopper"])
    fastq, _ = known_reads
    output = tmp_path / "filtered.fastq"
    cli_wrappers.chopper_filter(fastq, ToolPaths(), output, flags=get_tool_recipes("chopper")[recipe])
    lines = output.read_text().splitlines()
    # Parallel Chopper can emit complete records in a different order.
    assert len(lines) == 4 * len(expected)
    actual = {lines[i][1:]: (lines[i + 1], lines[i + 3]) for i in range(0, len(lines), 4)}
    assert actual == {name: ("A" * length, "I" * length) for name, length in expected.items()}


@pytest.mark.integration
@pytest.mark.parametrize("records", ["", "r\t4\t*\t0\t0\t*\t*\t0\t0\tAAAA\tIIII\n"])
@pytest.mark.parametrize("include_hist", [False, True])
def test_cramino_no_mapped_reads_has_no_identity(tmp_path, records, include_hist):
    import subprocess
    from conftest import require_executable_tools

    require_executable_tools(["samtools", "cramino"])
    sam = tmp_path / "no_mapped.sam"
    sam.write_text("@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n" + records)
    bam = tmp_path / "no_mapped.bam"
    subprocess.run([ToolPaths().samtools, "view", "-b", "-o", str(bam), str(sam)], check=True)
    stats = cli_wrappers.cramino_stats(bam, ToolPaths(), include_hist=include_hist)
    assert (stats.total_reads, stats.mean_length, stats.median_length, stats.n50) == (0, 0, 0, 0)
    assert stats.mean_identity is None
    assert stats.median_identity is None
