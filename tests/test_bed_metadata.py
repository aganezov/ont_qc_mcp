"""BED metadata keywords must not hide contigs or their validation errors."""

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
from ont_qc_mcp.parsers import parse_bed_qc, parse_mosdepth_regions_bed

CONTIGS = ["track1", "browser1", "track", "browser", "chr1"]
METADATA = '# targets\n\ntrack name="targets"\nbrowser position chr1:1-10\n'


@pytest.mark.parametrize("chrom", CONTIGS)
def test_bed_qc_keeps_metadata_keyword_contigs(tmp_path, chrom):
    bed = tmp_path / "targets.bed"
    bed.write_text(METADATA + f"{chrom}\t0\t10\tTarget\n")

    report = parse_bed_qc(bed)

    assert report.is_valid
    assert report.total_intervals == report.valid_intervals == 1
    assert report.total_bases == 10
    assert report.issues == []


def test_bed_qc_reports_errors_on_metadata_keyword_contigs(tmp_path):
    bed = tmp_path / "targets.bed"
    rows = ["track1\t-1\t10", "browser1\t10\t10", "track\tbad\t10", "browser\t0"]
    bed.write_text("# targets\n" + "\n".join(rows) + "\nchr1\t0\t10\n")

    report = parse_bed_qc(bed)

    assert not report.is_valid
    assert report.total_intervals == 5
    assert report.valid_intervals == 1
    assert report.total_bases == 10
    assert [(issue.line_number, issue.line_content) for issue in report.issues] == list(enumerate(rows, start=2))


@pytest.mark.parametrize("chrom", CONTIGS)
@pytest.mark.parametrize("output_columns", [4, 5])
def test_coverage_keeps_metadata_keyword_contig_names(tmp_path, chrom, output_columns):
    bed = tmp_path / "targets.bed"
    bed.write_text(METADATA + f"{chrom}\t0\t10\tTarget\n")
    regions = tmp_path / "regions.bed.gz"
    name_field = "Target\t" if output_columns == 5 else ""
    with gzip.open(regions, "wt") as output:
        output.write(f"{chrom}\t0\t10\t{name_field}2.5\n")

    assert parse_mosdepth_regions_bed(regions, bed) == [
        {"chrom": chrom, "start": 0, "end": 10, "region_name": "Target", "mean_depth": 2.5}
    ]


def test_coverage_unnamed_bed_keeps_coordinate_fallback(tmp_path):
    bed = tmp_path / "targets.bed"
    bed.write_text(METADATA + "track\t0\t10\n")
    regions = tmp_path / "regions.bed.gz"
    with gzip.open(regions, "wt") as output:
        output.write("track\t0\t10\t2.5\n")

    assert parse_mosdepth_regions_bed(regions, bed) == [
        {"chrom": "track", "start": 0, "end": 10, "region_name": "track:0-10", "mean_depth": 2.5}
    ]


@pytest.mark.integration
def test_metadata_keyword_contigs_through_mcp(mcp_server_params, tmp_path):
    """Real coverage retains all named targets and agrees with BED QC totals."""
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n"
        + "".join(f"@SQ\tSN:{chrom}\tLN:10\n" for chrom in CONTIGS)
        + "".join(
            f"r{idx}\t0\t{chrom}\t1\t60\t10M\t*\t0\t0\tAAAAAAAAAA\tIIIIIIIIII\n" for idx, chrom in enumerate(CONTIGS)
        )
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    bed = tmp_path / "targets.bed"
    # Blank-line handling by mosdepth is tracked separately in #60.
    bed.write_text(
        METADATA.replace("\n\n", "\n") + "".join(f"{chrom}\t0\t10\tTarget_{idx}\n" for idx, chrom in enumerate(CONTIGS))
    )

    async def check_reports():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                qc = await session.call_tool("qc_bed_tool", {"path": str(bed)})
                assert not qc.isError, qc.content
                qc_report = json.loads(cast(types.TextContent, qc.content[0]).text)
                assert qc_report["is_valid"]
                assert qc_report["total_intervals"] == qc_report["valid_intervals"] == len(CONTIGS)
                assert qc_report["total_bases"] == 10 * len(CONTIGS)

                coverage = await session.call_tool(
                    "targeted_coverage_tool", {"bam_path": str(bam), "bed_path": str(bed)}
                )
                assert not coverage.isError, coverage.content
                reports = json.loads(cast(types.TextContent, coverage.content[0]).text)
                assert len(reports) == len(CONTIGS)
                by_chrom = {report["chrom"]: report for report in reports}
                assert set(by_chrom) == set(CONTIGS)
                for idx, chrom in enumerate(CONTIGS):
                    report = by_chrom[chrom]
                    assert report["region_name"] == f"Target_{idx}"
                    assert (report["start"], report["end"]) == (0, 10)
                    assert report["mean_depth"] == pytest.approx(1.0)
                    assert report["pct_coverage_1x"] == pytest.approx(100.0)

    anyio.run(check_reports)
