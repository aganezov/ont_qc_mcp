"""Coverage must consume exactly the target intervals accepted by preflight."""

import json
import subprocess
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp import tools


@pytest.mark.integration
@pytest.mark.parametrize(
    "content",
    [
        "\nchr1\t0\t1\nchr1\t9\t10\n",
        "chr1\t0\t1\n\nchr1\t9\t10\n",
        "chr1\t0\t1\nchr1\t9\t10\n\n",
        " \t \nchr1\t0\t1\n \t \nchr1\t9\t10\n \t \n",
        "  chr1\t0\t1\n  chr1\t9\t10\n",
        "chr1\t0\t1  \nchr1\t9\t10  \n",
    ],
    ids=["leading-blank", "internal-blank", "trailing-blank", "whitespace-lines", "contig-spaces", "end-spaces"],
)
def test_normalized_targets_through_mcp(mcp_server_params, tmp_path, content):
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:10\n"
        "r1\t0\tchr1\t1\t60\t1M\t*\t0\t0\tA\tI\n"
        "r2\t0\tchr1\t10\t60\t1M\t*\t0\t0\tA\tI\n"
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    bed = tmp_path / "targets.bed"
    bed.write_bytes(content.encode())

    async def check_targets():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("targeted_coverage_tool", {"bam_path": str(bam), "bed_path": str(bed)})
                assert not result.isError, result.content
                reports = json.loads(cast(types.TextContent, result.content[0]).text)
                assert [(r["chrom"], r["start"], r["end"]) for r in reports] == [("chr1", 0, 1), ("chr1", 9, 10)]
                assert all(r["mean_depth"] == 1.0 and r["pct_coverage_1x"] == 100.0 for r in reports)

    try:
        anyio.run(check_targets)
    finally:
        assert bed.read_bytes() == content.encode()


@pytest.mark.parametrize("outcome", ["success", "validation", "cli", "parse"])
@pytest.mark.parametrize("mode", ["bed", "location", "gene"])
def test_normalized_bed_is_removed_on_every_exit(tmp_path, monkeypatch, outcome, mode):
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"test alignment")
    bed = tmp_path / "targets.bed"
    content = b'\n# targets\ntrack name="targets"\nbrowser position track:1-10\n  track\t0\t10\tname\t0\t+  \n\n'
    if outcome == "validation":
        content += b"track\t9\t11\n"
    bed.write_bytes(content)
    end = 11 if outcome == "validation" else 10
    target_args: dict[str, Any]
    if mode == "bed":
        target_args = {"bed_path": str(bed)}
        expected = b"track\t0\t10\tname\t0\t+\n"
    elif mode == "location":
        target_args = {"location": f"track:0-{end}"}
        expected = b"track\t0\t10\ttrack:0-10\n"
    else:
        gff = tmp_path / "genes.gff3"
        gff.write_text(f"track\ttest\tgene\t1\t{end}\t.\t+\t.\tID=g;Name=GENE\n")
        target_args = {"gene_name": "GENE", "annotation_path": str(gff)}
        expected = b"track\t0\t10\tGENE\n"
    monkeypatch.setattr(tools.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tools, "_read_alignment_header_text", lambda *args: "@SQ\tSN:track\tLN:10\n")

    def run_targets(**kwargs):
        used_bed = kwargs["bed_path"]
        assert used_bed != bed
        assert used_bed.read_bytes() == expected
        if outcome == "cli":
            raise RuntimeError("cli failed")
        return tmp_path / "regions.bed.gz", None, None

    runner = Mock(side_effect=run_targets)
    monkeypatch.setattr(tools, "run_mosdepth_targeted", runner)
    parser = Mock(side_effect=RuntimeError("parse failed")) if outcome == "parse" else Mock(return_value=[])
    monkeypatch.setattr(tools, "parse_mosdepth_regions_bed", parser)

    if outcome == "success":
        assert tools.targeted_coverage(str(bam), **target_args) == []
    else:
        error = ValueError if outcome == "validation" else RuntimeError
        line_number = 7 if mode == "bed" else 1
        message = f"line {line_number}.*exceeds" if outcome == "validation" else f"{outcome} failed"
        with pytest.raises(error, match=message):
            tools.targeted_coverage(str(bam), **target_args)
    if outcome == "validation":
        runner.assert_not_called()
    else:
        runner.assert_called_once()
        if outcome != "cli":
            parser.assert_called_once_with(tmp_path / "regions.bed.gz", runner.call_args.kwargs["bed_path"])
    assert bed.read_bytes() == content
    assert list(tmp_path.glob("*.bed")) == [bed]
