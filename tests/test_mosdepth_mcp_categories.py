"""Real MCP coverage summaries keep aggregate rows out of genomic locations."""

import json
import subprocess
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


@pytest.fixture(params=["ordinary", "colliding_names"])
def coverage_category_bam(tmp_path, request):
    require_executable_tools(["samtools", "mosdepth"])
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


@pytest.mark.parametrize("mode", ["whole", "window"])
def test_mosdepth_categories_through_mcp(mcp_server_params, coverage_category_bam, mode):
    bam, contigs = coverage_category_bam

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                arguments: dict[str, object] = {"path": str(bam)}
                if mode == "window":
                    arguments["window_size"] = 400
                result = await session.call_tool("coverage_qc", arguments)
                assert not result.is_error, result.content
                payload = json.loads(cast(types.TextContent, result.content[0]).text)
                assert payload["resolved_group_by"] == ("window" if mode == "window" else "contig")
                assert {row["chrom"] for row in payload["rows"]} == {name for name, _, _ in contigs}
                assert payload["union_summary"]["mean_depth"] == pytest.approx(
                    sum(length * depth for _, length, depth in contigs) / sum(length for _, length, _ in contigs)
                )
                for row in payload["rows"]:
                    expected_depth = next(depth for name, _, depth in contigs if name == row["chrom"])
                    assert row["mean_depth"] == pytest.approx(expected_depth)

    anyio.run(check)
