"""Finite timed reads retain hourly yield bins when their timestamps are equal."""

import json
import math
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.parsers import parse_sequencing_summary


@pytest.fixture(
    params=[
        pytest.param(([(3600, 100), (3600, 200)], [(1.0, 300, 2)], 0.0), id="identical-times"),
        pytest.param(([(0, 100)], [(0.0, 100, 1)], 0.0), id="single-read"),
        pytest.param(
            (
                [(1800, 100), (5399, 200), (5400, 300), (9000, 400)],
                [(0.5, 300, 2), (1.5, 300, 1), (2.5, 400, 1)],
                2.0,
            ),
            id="first-time-origin-and-boundaries",
        ),
        pytest.param(([(3600, 100), ("bad", 9000), (3600, 200)], [(1.0, 300, 2)], 0.0), id="invalid-time-row"),
        pytest.param(([("", 100), ("bad", 200)], [], None), id="no-parsed-times"),
    ]
)
def summary_case(tmp_path, request):
    rows, windows, duration = request.param
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "sequence_length_template\tstart_time\n" + "".join(f"{length}\t{time}\n" for time, length in rows)
    )
    expected = {
        "total_reads": len(rows),
        "total_yield": sum(length for _, length in rows),
        "run_duration_hours": duration,
        "yield_per_hour": [
            {"window_start_hours": start, "yield_bp": bases, "read_count": reads} for start, bases, reads in windows
        ],
    }
    return summary, expected


def test_hourly_bins_from_parser(summary_case):
    summary, expected = summary_case
    report = parse_sequencing_summary(summary).model_dump()
    assert {key: report[key] for key in expected} == expected


def test_hourly_bins_through_mcp(mcp_server_params, summary_case):
    summary, expected = summary_case

    async def check_report():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("sequencing_summary_tool", {"path": str(summary)})
                assert not result.is_error, result.content
                report = json.loads(cast(types.TextContent, result.content[0]).text)
                assert {key: report[key] for key in expected} == expected

    anyio.run(check_report)


@pytest.mark.parametrize("content", ["", "sequence_length_template\n100\n200\n"])
def test_summary_without_timing_has_no_hourly_bins(tmp_path, content):
    summary = tmp_path / "summary.txt"
    summary.write_text(content)
    report = parse_sequencing_summary(summary)
    assert report.run_duration_hours is None
    assert report.yield_per_hour == []


@pytest.mark.parametrize("timestamp", ["nan", "inf", "-inf"])
def test_single_nonfinite_time_keeps_existing_empty_bins(tmp_path, timestamp):
    summary = tmp_path / "summary.txt"
    summary.write_text(f"sequence_length_template\tstart_time\n100\t{timestamp}\n")
    report = parse_sequencing_summary(summary)
    assert report.total_reads == 1
    assert report.total_yield == 100
    assert report.run_duration_hours is not None and math.isnan(report.run_duration_hours)
    assert report.yield_per_hour == []
