"""BED QC and coverage validation must agree on coordinate number syntax."""

import json
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.parsers import parse_bed_qc
from ont_qc_mcp.tools import _validate_target_intervals


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("value", ["1_0", "１０", "١٠", "+10", "-0", " 10", "10 "])
def test_bed_qc_rejects_non_decimal_coordinate_spellings(tmp_path, field, value):
    bed = tmp_path / "targets.bed"
    start, end = (value, "20") if field == "start" else ("0", value)
    row = f"chr1\t{start}\t{end}\tTarget"
    bed.write_text("# targets\n" + row + "\nchr1\t20\t30\tValid\n")

    report = parse_bed_qc(bed)

    assert not report.is_valid
    assert report.total_intervals == 2
    assert report.valid_intervals == 1
    assert report.total_bases == 10
    assert len(report.issues) == 1
    assert report.issues[0].line_number == 2
    assert report.issues[0].line_content == row
    assert field in report.issues[0].issue.lower()
    with pytest.raises(ValueError, match="ASCII decimal"):
        _validate_target_intervals(bed, {"chr1": 100})


@pytest.mark.parametrize("start,end,bases", [("0", "10", 10), ("00", "010", 10), ("001", "010", 9), ("10", "20", 10)])
def test_bed_qc_preserves_ascii_decimal_coordinates(tmp_path, start, end, bases):
    bed = tmp_path / "targets.bed"
    bed.write_text(f"chr1\t{start}\t{end}\tTarget\n")

    report = parse_bed_qc(bed)

    assert report.is_valid
    assert report.total_intervals == report.valid_intervals == 1
    assert report.total_bases == bases
    assert not report.issues
    _validate_target_intervals(bed, {"chr1": 100})


@pytest.mark.parametrize(
    "start,end,message",
    [("-1", "10", "negative"), ("-20", "-10", "negative"), ("0", "-1", ">= end"), ("10", "10", ">= end")],
)
def test_bed_qc_preserves_coordinate_range_diagnostics(tmp_path, start, end, message):
    bed = tmp_path / "targets.bed"
    row = f"chr1\t{start}\t{end}\tTarget"
    bed.write_text(row + "\n")

    report = parse_bed_qc(bed)

    assert not report.is_valid
    assert report.total_intervals == 1
    assert report.valid_intervals == report.total_bases == 0
    assert len(report.issues) == 1
    assert report.issues[0].line_content == row
    assert message in report.issues[0].issue


@pytest.mark.integration
def test_bed_coordinate_syntax_through_mcp(mcp_server_params, tmp_path):
    bed = tmp_path / "targets.bed"
    rows = ["chr1\t0\t10", "track\t001\t010", "chr1\t0\t1_0", "chr1\t0\t１０", "chr1\t-1\t10"]
    content = "# targets\n" + "\n".join(rows) + "\n"
    bed.write_text(content)

    async def check_report():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("bed_qc", {"path": str(bed)})
                assert not result.is_error, result.content
                report = json.loads(cast(types.TextContent, result.content[0]).text)
                assert not report["is_valid"]
                assert report["total_intervals"] == 5
                assert report["valid_intervals"] == 2
                assert report["total_bases"] == 19
                assert [(issue["line_number"], issue["line_content"]) for issue in report["issues"]] == list(
                    enumerate(rows[2:], start=4)
                )
                assert "negative" in report["issues"][-1]["issue"]

    anyio.run(check_report)
    assert bed.read_text() == content
