"""Target intervals must fit the alignment reference before coverage runs."""

import json
import subprocess
from typing import Any, cast
from unittest.mock import Mock

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp import tools
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.utils import CommandResult


@pytest.fixture
def target_runner(tmp_path, monkeypatch):
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"test alignment")
    header = Mock(return_value="@SQ\tSN:chr1\tLN:100\n@SQ\tSN:track1\tLN:100\n")
    runner = Mock(return_value=(tmp_path / "regions.bed.gz", None, None))
    monkeypatch.setattr(tools, "_read_alignment_header_text", header)
    monkeypatch.setattr(tools, "run_mosdepth_targeted", runner)
    monkeypatch.setattr(tools, "parse_mosdepth_regions_bed", lambda *args: [])
    # Put generated BED files in the test directory so cleanup can be checked.
    monkeypatch.setattr(tools.tempfile, "tempdir", str(tmp_path))
    return bam, header, runner


@pytest.mark.parametrize(
    "row,message",
    [
        ("chr1\t0\n", "3 columns"),
        ("chr1\tzero\t10\n", "integer"),
        ("chr1\t0\t1.5\n", "integer"),
        ("chr1\t0\t1_0\n", "integer"),
        ("chr1\t0\t１０\n", "integer"),
        ("chr1\t-1\t10\n", "0 <= start < end"),
        ("chr1\t10\t10\n", "0 <= start < end"),
        ("chr1\t20\t10\n", "0 <= start < end"),
        ("chr2\t0\t10\n", "absent"),
        ("chr1\t90\t101\n", "exceeds"),
        ("chr1\t100\t101\n", "exceeds"),
        ("track1\t90\t101\n", "exceeds"),
        ("chr1\t0\t1\nchr1\t99\t101\n", "line 2.*exceeds"),
        ("track\tname=targets\nchr1\t0\t1\n", "3 columns"),
        ("browser\tposition\tchr1:1-100\nchr1\t0\t1\n", "integer"),
        ("# comment\n\ntrack name=targets\nbrowser position chr1:1-100\n", "No target intervals"),
    ],
)
def test_invalid_bed_never_runs_mosdepth(tmp_path, target_runner, row, message):
    bam, _header, runner = target_runner
    bed = tmp_path / "targets.bed"
    bed.write_text(row)
    with pytest.raises(ValueError, match=message):
        tools.targeted_coverage(str(bam), bed_path=str(bed))
    runner.assert_not_called()
    assert bed.read_text() == row


@pytest.mark.parametrize("length", [None, "unknown", "0", "-1"])
def test_target_requires_usable_reference_length(tmp_path, target_runner, length):
    bam, header, runner = target_runner
    header.return_value = "@SQ\tSN:chr1" + (f"\tLN:{length}" if length is not None else "") + "\n"
    with pytest.raises(ValueError, match="reference length"):
        tools.targeted_coverage(str(bam), location="chr1:0-1")
    runner.assert_not_called()
    assert not list(tmp_path.glob("*.bed"))


def test_target_header_uses_configured_samtools_and_timeout(tmp_path, monkeypatch):
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"test alignment")
    command = Mock(return_value=CommandResult([], 0, "@SQ\tSN:chr1\tLN:100\n", ""))
    runner = Mock()
    monkeypatch.setattr(tools, "run_command", command)
    monkeypatch.setattr(tools, "run_mosdepth_targeted", runner)
    cfg = ExecutionConfig(per_tool_threads={"samtools": 2}, per_tool_timeouts={"samtools": 7})
    with pytest.raises(ValueError, match="exceeds"):
        tools.targeted_coverage(
            str(bam), location="chr1:90-101", tools=ToolPaths(samtools="/custom/samtools"), exec_cfg=cfg
        )
    command.assert_called_once_with(["/custom/samtools", "view", "-H", "-@", "2", str(bam)], timeout=7)
    runner.assert_not_called()


@pytest.mark.parametrize("mode", ["location", "gene"])
@pytest.mark.parametrize("failure", ["bounds", "contig", "header"])
def test_generated_targets_reject_and_clean_up(tmp_path, target_runner, mode, failure):
    bam, header, runner = target_runner
    chrom = "chr2" if failure == "contig" else "chr1"
    end = 101 if failure == "bounds" else 10
    kwargs: dict[str, Any]
    expected_error: type[Exception]
    if mode == "location":
        kwargs = {"location": f"{chrom}:1-{end}"}
    else:
        gff = tmp_path / "genes.gff3"
        gff.write_text(f"##gff-version 3\n{chrom}\ttest\tgene\t1\t{end}\t.\t+\t.\tID=g;Name=GENE\n")
        kwargs = {"gene_name": "GENE", "annotation_path": str(gff)}
    if failure == "header":
        header.side_effect = RuntimeError("header unavailable")
        expected_error = RuntimeError
        message = "header unavailable"
    else:
        expected_error = ValueError
        message = "exceeds" if failure == "bounds" else "absent"
    with pytest.raises(expected_error, match=message):
        tools.targeted_coverage(str(bam), **kwargs)
    runner.assert_not_called()
    assert not list(tmp_path.glob("*.bed"))


@pytest.mark.parametrize("start", [0, -1])
def test_invalid_gff_start_is_rejected_after_conversion(tmp_path, target_runner, start):
    bam, _header, runner = target_runner
    gff = tmp_path / "invalid.gff3"
    gff.write_text(f"##gff-version 3\nchr1\ttest\tgene\t{start}\t2\t.\t+\t.\tID=g;Name=GENE\n")
    with pytest.raises(ValueError, match="0 <= start < end"):
        tools.targeted_coverage(str(bam), gene_name="GENE", annotation_path=str(gff))
    runner.assert_not_called()
    assert not list(tmp_path.glob("*.bed"))


@pytest.mark.parametrize("start,end", [(0, 1), (99, 100), (0, 100)])
@pytest.mark.parametrize("mode", ["bed", "location"])
def test_valid_boundary_targets_are_unchanged(tmp_path, target_runner, mode, start, end):
    bam, _header, runner = target_runner
    kwargs: dict[str, Any]
    if mode == "bed":
        content = f"chr1\t{start}\t{end}\tname\t0\t+\n"
        bed = tmp_path / "targets.bed"
        original = "# targets\ntrack name=targets\nbrowser position chr1:1-100\n\n" + content
        bed.write_text(original)
        kwargs = {"bed_path": str(bed)}
    else:
        location = f"chr1:{start}-{end}"
        content = f"chr1\t{start}\t{end}\t{location}\n"
        kwargs = {"location": location}

    def capture_bed(**kwargs):
        assert kwargs["bed_path"].read_text() == content
        return tmp_path / "regions.bed.gz", None, None

    runner.side_effect = capture_bed
    tools.targeted_coverage(str(bam), **kwargs)
    runner.assert_called_once()
    used_bed = runner.call_args.kwargs["bed_path"]
    assert not used_bed.exists()
    if mode == "bed":
        assert bed.read_text() == original


def test_metadata_keyword_contig_is_validated(tmp_path, target_runner):
    bam, header, runner = target_runner
    chrom = "track"
    header.return_value = f"@SQ\tSN:{chrom}\tLN:100\n"
    bed = tmp_path / "targets.bed"
    bed.write_text(f"{chrom}\t0\t100\n")
    tools.targeted_coverage(str(bam), bed_path=str(bed))
    runner.assert_called_once()
    runner.reset_mock()
    bed.write_text(f"{chrom}\t0\t101\n")
    with pytest.raises(ValueError, match="exceeds"):
        tools.targeted_coverage(str(bam), bed_path=str(bed))
    runner.assert_not_called()


@pytest.mark.integration
def test_target_bounds_through_mcp(mcp_server_params, tmp_path):
    """An end past LN must error; first, last and full intervals retain known depth."""
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

    async def check_targets():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                invalid = await session.call_tool(
                    "targeted_coverage_tool", {"bam_path": str(bam), "location": "chr1:9-11"}
                )
                assert invalid.is_error, invalid.content
                assert "exceeds" in cast(types.TextContent, invalid.content[0]).text
                for start, end, depth, percent in [(0, 1, 1.0, 100.0), (9, 10, 1.0, 100.0), (0, 10, 0.2, 20.0)]:
                    valid = await session.call_tool(
                        "targeted_coverage_tool", {"bam_path": str(bam), "location": f"chr1:{start}-{end}"}
                    )
                    assert not valid.is_error, valid.content
                    reports = json.loads(cast(types.TextContent, valid.content[0]).text)
                    assert len(reports) == 1
                    assert (reports[0]["start"], reports[0]["end"]) == (start, end)
                    assert reports[0]["mean_depth"] == pytest.approx(depth)
                    assert reports[0]["pct_coverage_1x"] == pytest.approx(percent)

    anyio.run(check_targets)
