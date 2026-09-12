import json
import subprocess
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


@pytest.mark.integration
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize(
    "start,end,expected_depth,expected_percentages",
    [(1, 2, 10.0, [50.0, 50.0, 50.0]), (1, 1, 20.0, [100.0, 100.0, 100.0]), (3, 3, 10.0, [100.0, 100.0, 0.0])],
)
def test_gene_coverage_converts_gff_coordinates(
    mcp_server_params, tmp_path, strand, start, end, expected_depth, expected_percentages
):
    """Twenty reads cover base 1, ten cover base 3; base 2 is uncovered."""
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:10\n"
        + "".join(
            f"r{i}\t0\tchr1\t{position}\t60\t1M\t*\t0\t0\tA\tI\n" for i, position in enumerate([1] * 20 + [3] * 10)
        )
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    gff = tmp_path / "genes.gff3"
    gff.write_text(f"##gff-version 3\nchr1\tfixture\tgene\t{start}\t{end}\t.\t{strand}\t.\tID=gene1;Name=Target\n")

    async def check_report():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(
                    "targeted_coverage_tool",
                    {"bam_path": str(bam), "gene_name": "Target", "annotation_path": str(gff)},
                )
                assert not response.is_error, response.content
                reports = json.loads(cast(types.TextContent, response.content[0]).text)
                assert len(reports) == 1
                report = reports[0]
                assert (report["chrom"], report["start"], report["end"]) == ("chr1", start - 1, end)
                assert report["mean_depth"] == pytest.approx(expected_depth)
                for threshold, percentage in zip([1, 10, 20], expected_percentages):
                    assert report[f"pct_coverage_{threshold}x"] == pytest.approx(percentage)

    anyio.run(check_report)
