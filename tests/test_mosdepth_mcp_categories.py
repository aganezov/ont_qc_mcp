"""Real MCP coverage summaries keep aggregate rows out of genomic locations."""

import json
import subprocess
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


@pytest.fixture(params=["ordinary", "colliding_names"])
def coverage_category_bam(tmp_path, request):
    require_executable_tools(["samtools", "mosdepth", "cramino"])
    if request.param == "ordinary":
        contigs = [("chrA", 10000, 10), ("chrB", 30000, 1)]
    else:
        names = ["total", "total_region", "chrA", "chrA_region", "chrom", "chromosome1"]
        contigs = [(name, 1000, i + 1) for i, name in enumerate(names)]
    rows = ["@HD\tVN:1.6\tSO:coordinate"]
    rows.extend(f"@SQ\tSN:{name}\tLN:{length}" for name, length, _ in contigs)
    for name, length, depth in contigs:
        for i in range(depth):
            rows.append(f"{name}_{i}\t0\t{name}\t1\t60\t{length}M\t*\t0\t0\t{'A' * length}\t{'I' * length}\tNM:i:0")
    sam = tmp_path / "reads.sam"
    bam = tmp_path / "reads.bam"
    sam.write_text("\n".join(rows) + "\n")
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    return bam, contigs


@pytest.mark.parametrize("mode", ["whole", "window", "by_alias", "summary"])
def test_mosdepth_categories_through_mcp(mcp_server_params, coverage_category_bam, mode):
    bam, contigs = coverage_category_bam

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                arguments = {"path": str(bam), "low_cov_threshold": 5}
                tool_name = "coverage_stats_tool"
                if mode == "window":
                    arguments["window"] = 400
                elif mode == "by_alias":
                    arguments["flags"] = {"by": 400}
                elif mode == "summary":
                    tool_name = "alignment_summary_tool"
                    arguments = {
                        "path": str(bam),
                        "include_hist": False,
                        "coverage_window": 400,
                        "coverage_low_cov_threshold": 5,
                    }
                result = await session.call_tool(tool_name, arguments)
                assert not result.isError, result.content
                payload = json.loads(cast(types.TextContent, result.content[0]).text)
                if mode == "summary":
                    payload = payload["coverage"]
                assert [
                    (row["contig"], row["length"], row["mean_depth"]) for row in payload["coverage_by_contig"]
                ] == contigs
                assert payload["mean_depth_unweighted"] == pytest.approx(
                    sum(depth for _, _, depth in contigs) / len(contigs)
                )
                assert payload["mean_depth"] == pytest.approx(
                    sum(length * depth for _, length, depth in contigs) / sum(length for _, length, _ in contigs)
                )
                assert [
                    (row["contig"], row["start"], row["end"], row["mean_depth"])
                    for row in payload["low_coverage_regions"]
                ] == [(name, 0, length, depth) for name, length, depth in contigs if depth < 5]

    anyio.run(check)
