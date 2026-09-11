"""Numerical checks for the pinned external tools through the MCP interface."""

import json
import os
from pathlib import Path
import subprocess
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


def test_mcp_uses_selected_toolchain(mcp_server_params):
    tools = ["nanoq", "chopper", "cramino", "mosdepth", "samtools", "bcftools"]
    require_executable_tools(tools)
    from ont_qc_mcp.config import ToolPaths

    expected = ToolPaths().resolved()

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("env_status", {})
                assert not result.isError, result.content
                status = json.loads(cast(types.TextContent, result.content[0]).text)
                for tool in tools:
                    assert Path(status["resolved_paths"][tool]).resolve() == Path(expected[tool]).resolve()
                    if override := os.getenv(tool.upper()):
                        assert Path(status["resolved_paths"][tool]).resolve() == Path(override).resolve()

    anyio.run(check)


def test_long_reads_survive_filtering_and_qc(mcp_server_params, tmp_path):
    require_executable_tools(["chopper", "nanoq"])
    reads = {
        f"r{i}": ("ACGT" * (length // 4) + "A" * (length % 4), "I" * length)
        for i, length in enumerate([100, 200_000, 1_000_001])
    }
    source = tmp_path / "reads.fastq"
    source.write_text("".join(f"@{name}\n{seq}\n+\n{qual}\n" for name, (seq, qual) in reads.items()))
    output = tmp_path / "filtered.fastq"

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                filtered = await session.call_tool(
                    "filter_reads_fastq_tool",
                    {"path": str(source), "output_fastq": str(output), "flags": {"minlength": 1000}},
                )
                assert not filtered.isError, filtered.content
                lines = output.read_text().splitlines()
                assert len(lines) == 8
                actual = {lines[i][1:]: (lines[i + 1], lines[i + 3]) for i in range(0, len(lines), 4)}
                assert actual == {name: reads[name] for name in ["r1", "r2"]}

                result = await session.call_tool("qc_reads_fastq_tool", {"path": str(output)})
                assert not result.isError, result.content
                stats = json.loads(cast(types.TextContent, result.content[0]).text)
                assert stats["read_count"] == 2
                assert stats["total_bases"] == 1_200_001
                assert stats["min_len"] == 200_000
                assert stats["max_len"] == 1_000_001

    anyio.run(check)


def test_cramino_read_and_histogram_counts(mcp_server_params, tmp_path):
    require_executable_tools(["samtools", "cramino"])
    source = tmp_path / "reads.sam"
    source.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:10000\n"
        + "".join(
            f"r{i}\t0\tchr1\t1\t60\t{length}M\t*\t0\t0\t{'A' * length}\t{'I' * length}\tNM:i:{nm}\n"
            for i, (length, nm) in enumerate([(50, 0), (2100, 21), (4300, 86)])
        )
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(source)], check=True, capture_output=True)

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_alignment_tool", {"path": str(bam), "include_hist": True})
                assert not result.isError, result.content
                stats = json.loads(cast(types.TextContent, result.content[0]).text)
                assert stats["total_reads"] == 3
                assert stats["mean_length"] == 2150
                assert stats["median_length"] == 2100
                assert stats["n50"] == 4300
                assert stats["mean_identity"] == pytest.approx(99)
                assert stats["median_identity"] == pytest.approx(99)
                assert [(b["start"], b["end"], b["count"]) for b in stats["length_histogram"] if b["count"]] == [
                    (0, 2000, 1),
                    (2000, 4000, 1),
                    (4000, 6000, 1),
                ]

    anyio.run(check)

    # Upstream JSON retains both weights; --scaled changes the TSV's units.
    for scaled in (False, True):
        counts = tmp_path / "histogram.tsv"
        command = [os.getenv("CRAMINO", "cramino"), "--format", "json", "--hist-count", str(counts)]
        if scaled:
            command.append("--scaled")
        result = subprocess.run([*command, str(bam)], check=True, capture_output=True, text=True)
        histograms = json.loads(result.stdout)["histograms"]
        assert [
            (b["start"], b["end"], b["count"], b["bases"]) for b in histograms["read_length"]["bins"] if b["count"]
        ] == [(0, 2000, 1, 50), (2000, 4000, 1, 2100), (4000, 6000, 1, 4300)]
        assert sum(b["count"] for b in histograms["q_score"]["bins"]) == 3
        assert sum(b["bases"] for b in histograms["q_score"]["bins"]) == 6450
        rows = [line.split("\t") for line in counts.read_text().splitlines() if line]
        assert rows[0] == ["bin_start", "bin_end", "bases" if scaled else "count"]
        assert [int(row[2]) for row in rows[1:] if int(row[2])] == ([50, 2100, 4300] if scaled else [1, 1, 1])
