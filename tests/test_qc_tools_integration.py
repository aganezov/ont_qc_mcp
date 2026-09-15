"""End-to-end MCP checks for the run, variant, coverage, and BED QC tools."""

import json
from typing import Any, cast

import anyio
import mcp_types as types
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


def _payload(result: types.CallToolResult) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(cast(types.TextContent, result.content[0]).text))


def test_run_summary(mcp_server_params, synthetic_sequencing_summary):
    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("run_summary", {"path": str(synthetic_sequencing_summary)})
                assert not result.is_error, result.content
                payload = _payload(result)
                assert payload["file"] == str(synthetic_sequencing_summary)
                assert payload["total_reads"] == 20
                assert payload["total_yield"] > 0
                assert payload["mean_length"] is not None
                assert payload["mean_qscore"] is not None
                assert isinstance(payload["yield_per_hour"], list)

    anyio.run(check)


@pytest.mark.parametrize("fixture_name", ["sample_vcf", "synthetic_vcf"])
def test_variant_qc(mcp_server_params, request, fixture_name):
    require_executable_tools(["bcftools"])
    vcf = request.getfixturevalue(fixture_name)

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("variant_qc", {"path": str(vcf)})
                assert not result.is_error, result.content
                payload = _payload(result)
                assert payload["effective_request"]["path"] == str(vcf)
                group = payload["results"][0]
                assert group["general"]["total_records"] >= 0
                assert group["snps"]["count"] >= 0
                assert group["indels"]["count"] >= 0

    anyio.run(check)


def test_variant_qc_requested_sections(mcp_server_params, sample_vcf):
    require_executable_tools(["bcftools"])

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "variant_qc", {"path": str(sample_vcf), "metrics": ["general", "snps"]}
                )
                assert not result.is_error, result.content
                group = _payload(result)["results"][0]
                assert group["general"] is not None
                assert group["snps"] is not None
                assert group["indels"] is None

    anyio.run(check)


@pytest.mark.parametrize(
    "regions,expected_count",
    [
        ({"format": "bed", "path": "{bed}"}, None),
        ({"format": "samtools", "values": ["chr1:1000-2000"]}, 1),
        ({"format": "gff3", "path": "{gff}", "feature_type": "gene", "ids": ["MOCK_GENE1"]}, None),
    ],
    ids=["bed", "samtools", "gff3"],
)
def test_coverage_qc_regions(
    mcp_server_params,
    sample_bam,
    synthetic_bed_valid,
    synthetic_gff3,
    regions,
    expected_count,
):
    require_executable_tools(["samtools", "mosdepth"])
    rendered = {
        key: (str(synthetic_bed_valid) if value == "{bed}" else str(synthetic_gff3) if value == "{gff}" else value)
        for key, value in regions.items()
    }

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("coverage_qc", {"path": str(sample_bam), "regions": rendered})
                assert not result.is_error, result.content
                payload = _payload(result)
                assert payload["resolved_group_by"] == "region"
                if expected_count is not None:
                    assert len(payload["rows"]) == expected_count
                for row in payload["rows"]:
                    assert row["mean_depth"] >= 0
                    assert [item["threshold"] for item in row["breadth"]] == [1, 10, 20]
                if rendered["format"] == "bed":
                    assert any(row["name"] for row in payload["rows"])

    anyio.run(check)


@pytest.mark.parametrize("fixture_name,is_valid", [("synthetic_bed_valid", True), ("synthetic_bed_invalid", False)])
def test_bed_qc(mcp_server_params, request, fixture_name, is_valid):
    bed = request.getfixturevalue(fixture_name)

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("bed_qc", {"path": str(bed)})
                assert not result.is_error, result.content
                payload = _payload(result)
                assert payload["file"] == str(bed)
                assert payload["is_valid"] is is_valid
                assert payload["total_intervals"] > 0
                assert (payload["valid_intervals"] == payload["total_intervals"]) is is_valid
                assert bool(payload["issues"]) is (not is_valid)

    anyio.run(check)


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("coverage_qc", {"path": "missing.bam", "regions": {"format": "samtools", "values": ["chr1:1-2"]}}),
        ("variant_qc", {"path": "missing.vcf.gz"}),
        ("run_summary", {"path": "missing.txt"}),
    ],
)
def test_missing_input_is_structured_execution_error(mcp_server_params, tmp_path, tool, arguments):
    arguments["path"] = str(tmp_path / arguments["path"])

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                assert result.is_error
                payload = _payload(result)
                assert payload["kind"] == "execution_error"
                assert payload["partial_result_returned"] is False

    anyio.run(check)


def test_coverage_qc_rejects_legacy_region_fields(mcp_server_params, sample_bam, synthetic_bed_valid):
    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "coverage_qc",
                    {"path": str(sample_bam), "bed_path": str(synthetic_bed_valid), "location": "chr1:1-2"},
                )
                assert result.is_error
                assert _payload(result)["kind"] == "validation_error"

    anyio.run(check)
