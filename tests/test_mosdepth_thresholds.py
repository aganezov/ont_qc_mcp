import gzip
import json
import subprocess
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp.parsers import parse_mosdepth_thresholds_bed


@pytest.mark.parametrize("name", ["gene", "unknown", None])
def test_threshold_counts_are_normalized_by_interval_length(tmp_path, name):
    path = tmp_path / "thresholds.bed.gz"
    rows = [
        (0, 100, [50, 25, 10], [50.0, 25.0, 10.0]),
        (0, 50, [50, 25, 10], [100.0, 50.0, 20.0]),
        (50, 100, [0, 0, 0], [0.0, 0.0, 0.0]),
        (0, 1, [1, 1, 1], [100.0, 100.0, 100.0]),
        (0, 3, [1, 0, 0], [100.0 / 3, 0.0, 0.0]),
    ]
    with gzip.open(path, "wt") as stream:
        stream.write("#chrom\tstart\tend\tregion\t1X\t10X\t20X\n")
        for start, end, counts, _expected in rows:
            fields = ["chr1", str(start), str(end)]
            if name is not None:
                fields.append(name)
            stream.write("\t".join(fields + [str(count) for count in counts]) + "\n")

    result = parse_mosdepth_thresholds_bed(path, [1, 10, 20])
    for start, end, _counts, expected in rows:
        assert result[("chr1", start, end)] == pytest.approx(
            dict(zip(["pct_coverage_1x", "pct_coverage_10x", "pct_coverage_20x"], expected))
        )


@pytest.mark.parametrize("start,end", [(5, 5), (5, 4)])
def test_thresholds_reject_nonpositive_interval_length(tmp_path, start, end):
    path = tmp_path / "thresholds.bed.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(f"chr1\t{start}\t{end}\tregion\t0\t0\t0\n")
    with pytest.raises(ValueError, match="positive"):
        parse_mosdepth_thresholds_bed(path, [1, 10, 20])


@pytest.mark.integration
@pytest.mark.parametrize("named", [False, True])
def test_targeted_coverage_known_depth_through_mcp(mcp_server_params, tmp_path, named):
    """Depth is 20 on [0,25), 10 on [25,50), and zero on [50,200)."""
    require_executable_tools(["samtools", "mosdepth"])
    sam = tmp_path / "reads.sam"
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:200\n"
        + "".join(
            f"r{i}\t0\tchr1\t1\t60\t{length}M\t*\t0\t0\t{'A' * length}\t{'I' * length}\n"
            for i, length in enumerate([50] * 10 + [25] * 10)
        )
    )
    bam = tmp_path / "reads.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True, capture_output=True)
    # start, end, mean depth, percentages at >=1x, >=10x, >=20x
    rows = [
        (0, 1, 20.0, [100.0, 100.0, 100.0]),
        (0, 50, 15.0, [100.0, 100.0, 50.0]),
        (0, 100, 7.5, [50.0, 50.0, 25.0]),
        (25, 50, 10.0, [100.0, 100.0, 0.0]),
        (50, 100, 0.0, [0.0, 0.0, 0.0]),
    ]
    bed = tmp_path / "regions.bed"
    bed.write_text(
        "".join(
            f"chr1\t{start}\t{end}" + (f"\tregion{i}" if named else "") + "\n"
            for i, (start, end, _mean, _pcts) in enumerate(rows)
        )
    )

    async def check_reports():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(
                    "coverage_qc", {"path": str(bam), "regions": {"format": "bed", "path": str(bed)}}
                )
                assert not response.is_error, response.content
                payload = json.loads(cast(types.TextContent, response.content[0]).text)
                reports = {(report["start"], report["end"]): report for report in payload["rows"]}
                assert len(reports) == len(rows)
                for start, end, mean, percentages in rows:
                    report = reports[(start, end)]
                    assert report["mean_depth"] == pytest.approx(mean)
                    breadth = {item["threshold"]: item["fraction_at_or_above"] for item in report["breadth"]}
                    for threshold, percentage in zip([1, 10, 20], percentages):
                        assert breadth[threshold] == pytest.approx(percentage / 100)

    anyio.run(check_reports)
