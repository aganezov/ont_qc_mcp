"""Coverage records retain names even when target coordinates repeat."""

import gzip
import json
import subprocess
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp.parsers import parse_mosdepth_regions_bed


@pytest.mark.parametrize("columns", [4, 5])
def test_repeated_coordinate_names_are_not_collapsed(tmp_path, columns):
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t10\tone\nchr1\t0\t10\ttwo\nchr2\t0\t10\tthree\n")
    regions = tmp_path / "regions.bed.gz"
    # Output may group contigs differently from the input BED.
    rows = [("chr2", "three", 3.0), ("chr1", "one", 1.0), ("chr1", "two", 1.0)]
    with gzip.open(regions, "wt") as output:
        for chrom, name, depth in rows:
            name_field = f"{name}\t" if columns == 5 else ""
            output.write(f"{chrom}\t0\t10\t{name_field}{depth}\n")
    result = parse_mosdepth_regions_bed(regions, bed)
    assert [r["region_name"] for r in result] == ["three", "one", "two"]
    assert [r["mean_depth"] for r in result] == [3.0, 1.0, 1.0]


def test_named_output_is_authoritative_in_output_order(tmp_path):
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t10\tone\nchr1\t0\t10\ttwo\n")
    regions = tmp_path / "regions.bed.gz"
    with gzip.open(regions, "wt") as output:
        output.write("chr1\t0\t10\ttwo\t1.0\nchr1\t0\t10\tone\t1.0\n")
    assert [r["region_name"] for r in parse_mosdepth_regions_bed(regions, bed)] == ["two", "one"]


@pytest.mark.parametrize("name", ["123", ".", "unknown", "target name"])
def test_output_name_spelling_is_preserved(tmp_path, name):
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t10\told_name\n")
    regions = tmp_path / "regions.bed.gz"
    with gzip.open(regions, "wt") as output:
        output.write(f"chr1\t0\t10\t{name}\t2.5\n")
    result = parse_mosdepth_regions_bed(regions, bed)
    assert result[0]["region_name"] == name
    assert result[0]["mean_depth"] == 2.5


def test_unnamed_output_consumes_bed_names_per_coordinate(tmp_path):
    bed = tmp_path / "targets.bed"
    bed.write_text("# targets\ntrack name=targets\ntrack\t0\t10\none\t0\t10\tNamed\ntrack\t0\t10\tLast\n")
    regions = tmp_path / "regions.bed.gz"
    with gzip.open(regions, "wt") as output:
        output.write("track\t0\t10\t1.0\none\t0\t10\t2.0\ntrack\t0\t10\t1.0\ntrack\t0\t10\t1.0\n")
    assert [r["region_name"] for r in parse_mosdepth_regions_bed(regions, bed)] == [
        "track:0-10",
        "Named",
        "Last",
        "track:0-10",
    ]


@pytest.mark.integration
def test_duplicate_target_names_through_mcp(mcp_server_params, tmp_path):
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:20\nr1\t0\tchr1\t1\t60\t10M\t*\t0\t0\tAAAAAAAAAA\tIIIIIIIIII\n"
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    bed = tmp_path / "targets.bed"
    original = b"chr1\t0\t10\tone\nchr1\t0\t10\ttwo\nchr1\t10\t20\tuncovered\n"
    bed.write_bytes(original)

    async def check_reports():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("targeted_coverage_tool", {"bam_path": str(bam), "bed_path": str(bed)})
                assert not result.isError, result.content
                reports = json.loads(cast(types.TextContent, result.content[0]).text)
                assert len(reports) == 3
                by_name = {report["region_name"]: report for report in reports}
                assert set(by_name) == {"one", "two", "uncovered"}
                for name, depth, pct in [("one", 1.0, 100.0), ("two", 1.0, 100.0), ("uncovered", 0.0, 0.0)]:
                    assert by_name[name]["mean_depth"] == pytest.approx(depth)
                    assert by_name[name]["pct_coverage_1x"] == pytest.approx(pct)
                    assert by_name[name]["pct_coverage_10x"] == pytest.approx(0.0)
                    assert by_name[name]["pct_coverage_20x"] == pytest.approx(0.0)
                assert (by_name["one"]["start"], by_name["one"]["end"]) == (0, 10)
                assert (by_name["two"]["start"], by_name["two"]["end"]) == (0, 10)

    anyio.run(check_reports)
    assert bed.read_bytes() == original
