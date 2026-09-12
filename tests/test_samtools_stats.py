"""Literal samtools 1.24 statistics and independent parser controls."""

import importlib.util
import json
from pathlib import Path
import subprocess

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.parsers import parse_error_profile


FIXTURES = Path(__file__).parent / "fixtures" / "samtools_stats"


def fixture_profile(name):
    return parse_error_profile((FIXTURES / f"{name}.stats").read_text(), file_path="reads.sam")


def test_literal_sn_tab_comment_preserves_numeric_error_rate():
    stats = fixture_profile("default")
    assert stats.mismatch_rate == pytest.approx(0.023)


@pytest.mark.parametrize(
    "fixture,expected",
    [
        ("default", {"start": 20, "end": 20, "count": 50}),
        ("binned", {"start": 20, "end": None, "count": 50}),
        ("multiwidth", {"start": 20, "end": 24, "count": 50}),
        ("underflow", {"start": 0, "end": 24, "count": 50}),
    ],
)
def test_literal_cov_uses_interval_and_fourth_column(fixture, expected):
    bins = fixture_profile(fixture).coverage_histogram
    assert bins is not None
    assert [row.model_dump() for row in bins] == [expected]


def test_literal_mpc_separates_ns_and_quality_mismatches():
    stats = fixture_profile("reference")
    assert stats.mismatch_by_cycle is None
    cycles = stats.mismatch_counts_by_cycle
    assert cycles is not None
    # samtools 1.24 also emits an all-zero cycle 51 for these 50 bp reads.
    assert [row.cycle for row in cycles] == list(range(1, 52))
    assert cycles[-1].n_count == 0
    assert cycles[-1].mismatches_by_quality == [0] * 41
    assert cycles[0].n_count == 3
    assert cycles[0].mismatches_by_quality == [0] * 41
    assert cycles[1].n_count == 0
    assert cycles[1].mismatches_by_quality == [10] + [0] * 39 + [10]
    assert cycles[2].mismatches_by_quality == [0] * 41


@pytest.mark.parametrize("raw", ["0", "2.3e-2", "0.25"])
def test_sn_separate_comment_and_specific_rate_precedence(raw):
    stats = parse_error_profile(
        f"SN\terror rate:\t0.8\t# NM-derived fallback\nSN\tmismatches per base:\t{raw}\t# specific rate\n",
        "reads.bam",
    )
    assert stats.mismatch_rate == float(raw)


@pytest.mark.parametrize("raw", ["bad", "", "NaN", "inf", "-0.1"])
def test_malformed_sn_rates_are_unavailable(raw):
    stats = parse_error_profile(f"SN\terror rate:\t{raw}\t# not a usable rate\n", "reads.bam")
    assert stats.mismatch_rate is None


def test_absent_sections_remain_none():
    stats = parse_error_profile("# no statistics available\n", "reads.bam")
    assert stats.mismatch_rate is None
    assert stats.coverage_histogram is None
    assert stats.mismatch_by_cycle is None
    assert stats.mismatch_counts_by_cycle is None


def test_zero_bins_and_sparse_reordered_cycles_remain_explicit():
    stats = parse_error_profile(
        "COV\t[10-14]\t14\t0\nMPC\t5\t0\t0\t1\nMPC\t2\t7\t0\t0\n",
        "reads.bam",
    )
    assert stats.coverage_histogram is not None
    assert [row.model_dump() for row in stats.coverage_histogram] == [{"start": 10, "end": 14, "count": 0}]
    assert stats.mismatch_counts_by_cycle is not None
    assert [row.model_dump() for row in stats.mismatch_counts_by_cycle] == [
        {"cycle": 2, "n_count": 7, "mismatches_by_quality": [0, 0]},
        {"cycle": 5, "n_count": 0, "mismatches_by_quality": [0, 1]},
    ]
    assert stats.mismatch_by_cycle is None


@pytest.mark.parametrize(
    "line",
    [
        "COV\t20\t50",  # obsolete three-column invention
        "COV\t[5-2]\t3\t1",
        "COV\t[<0]\t0\t1",
        "COV\t[2-2]\t2\t-1",
        "COV\t[2-2]\t2\t1.5",
        "COV\t[2-2]\tnot-depth\t1",
        "COV\t[10-14]\t12\t1",
        "COV\t[2-2]\t2\tNaN",
        "MPC\t1\t0.01",  # obsolete invented rate row
        "MPC\t0\t0\t2",
        "MPC\t-1\t0\t2",
        "MPC\t2\t-1\t2",
        "MPC\t2\t0\t1.5",
        "MPC\t2\t0\tbad",
        "MPC\t2\t0\t-1",
    ],
)
def test_malformed_rows_do_not_create_partial_records(line):
    stats = parse_error_profile(line + "\n", "reads.bam")
    assert stats.coverage_histogram is None
    assert stats.mismatch_by_cycle is None
    assert stats.mismatch_counts_by_cycle is None


@pytest.mark.integration
def test_real_mcp_reads_sn_and_coverage(mcp_server_params, tmp_path):
    from conftest import require_executable_tools

    require_executable_tools(["samtools"])
    bam = tmp_path / "reads.bam"
    subprocess.run([ToolPaths().samtools, "view", "-b", "-o", str(bam), str(FIXTURES / "reads.sam")], check=True)

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("alignment_error_profile_tool", {"path": str(bam)})
                assert not result.isError, result.content
                content = result.content[0]
                assert isinstance(content, TextContent)
                stats = json.loads(content.text)
                assert stats["mismatch_rate"] == pytest.approx(0.023)
                assert stats["coverage_histogram"] == [{"start": 20, "end": 20, "count": 50}]
                assert stats["mismatch_by_cycle"] is None
                assert stats["mismatch_counts_by_cycle"] is None

    anyio.run(check)


@pytest.mark.integration
def test_generator_reproduces_literal_rows_and_reference_mpc(tmp_path):
    from conftest import require_executable_tools

    require_executable_tools(["samtools"])
    spec = importlib.util.spec_from_file_location("samtools_stats_generator", FIXTURES / "generate.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.generate(tmp_path, ToolPaths().samtools)
    for name in ("default", "reference", "binned", "multiwidth", "underflow"):
        assert (tmp_path / f"{name}.stats").read_bytes() == (FIXTURES / f"{name}.stats").read_bytes()
    stats = parse_error_profile((tmp_path / "reference.stats").read_text(), "reads.sam")
    assert stats.mismatch_by_cycle is None
    cycles = stats.mismatch_counts_by_cycle
    assert cycles is not None
    assert (cycles[0].cycle, cycles[0].n_count, sum(cycles[0].mismatches_by_quality)) == (1, 3, 0)
    assert (cycles[1].cycle, cycles[1].n_count, sum(cycles[1].mismatches_by_quality)) == (2, 0, 20)
