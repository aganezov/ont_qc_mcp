"""
Integration tests for QC tools: sequencing_summary_tool, qc_variants_tool,
targeted_coverage_tool, and qc_bed_tool.

These tests spawn the MCP server as a subprocess and communicate via MCP protocol.
Tests use synthetic fixtures from tests/fixtures/synthetic/.
"""

import json
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools


pytestmark = pytest.mark.integration


def _text_content(content: types.Content) -> types.TextContent:
    return cast(types.TextContent, content)


# Note: synthetic_* fixtures are defined in conftest.py and auto-generated if missing


def test_sequencing_summary_tool(mcp_server_params, synthetic_sequencing_summary):
    """Test sequencing_summary_tool with synthetic sequencing summary file."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("sequencing_summary_tool", {"path": str(synthetic_sequencing_summary)})
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(synthetic_sequencing_summary)
                assert payload["total_reads"] == 20
                assert payload["total_yield"] > 0
                assert payload.get("mean_length") is not None
                assert payload.get("mean_qscore") is not None
                # Check yield_per_hour is a list
                assert isinstance(payload.get("yield_per_hour", []), list)

    anyio.run(_test)


def test_qc_variants_tool_real_vcf(mcp_server_params, sample_vcf):
    """Test qc_variants_tool with real VCF fixture."""

    require_executable_tools(["bcftools"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "qc_variants_tool", {"path": str(sample_vcf), "include_snps": True, "include_indels": True}
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(sample_vcf)
                assert "general" in payload
                assert payload["general"]["total_records"] >= 0
                # SNP and indel stats may or may not be present depending on VCF content
                if payload.get("snps"):
                    assert payload["snps"]["count"] >= 0
                if payload.get("indels"):
                    assert payload["indels"]["count"] >= 0

    anyio.run(_test)


def test_qc_variants_tool_synthetic_vcf(mcp_server_params, synthetic_vcf):
    """Test qc_variants_tool with synthetic VCF."""

    require_executable_tools(["bcftools"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "qc_variants_tool", {"path": str(synthetic_vcf), "include_snps": True, "include_indels": True}
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(synthetic_vcf)
                assert "general" in payload
                assert payload["general"]["total_records"] >= 0

    anyio.run(_test)


def test_qc_variants_tool_snps_only(mcp_server_params, sample_vcf):
    """Test qc_variants_tool with SNPs only (no indels)."""

    require_executable_tools(["bcftools"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "qc_variants_tool", {"path": str(sample_vcf), "include_snps": True, "include_indels": False}
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(sample_vcf)
                assert "general" in payload
                # Indels should not be present when include_indels=False
                assert payload.get("indels") is None

    anyio.run(_test)


def test_targeted_coverage_tool_bed(mcp_server_params, sample_bam, synthetic_bed_valid):
    """Test targeted_coverage_tool with BED file input.

    Uses mosdepth with --by and --thresholds to compute coverage.
    This test specifically verifies that multi-column BED files work correctly,
    which caught a bug in the original samtools bedcov implementation where
    the coverage column was incorrectly assumed to be at index 3.
    """
    require_executable_tools(["mosdepth"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "targeted_coverage_tool", {"bam_path": str(sample_bam), "bed_path": str(synthetic_bed_valid)}
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert isinstance(payload, list), "Expected list of coverage reports"
                assert len(payload) > 0, "Expected at least one coverage report"

                # Check structure of first report
                report = payload[0]
                assert "region_name" in report
                assert "chrom" in report
                assert "start" in report
                assert "end" in report
                assert "mean_depth" in report
                assert isinstance(report["mean_depth"], (int, float))
                assert report["mean_depth"] >= 0

                # Verify that the region_name is properly extracted from BED (not coordinates)
                # This would have failed with the original bedcov bug that tried to parse
                # "gene1_exon1" as an integer coverage value
                assert report["region_name"] != f"{report['chrom']}:{report['start']}-{report['end']}", (
                    "region_name should be extracted from BED column 4, not generated from coordinates"
                )

                # Check that coverage threshold percentages are present (mosdepth feature)
                assert "pct_coverage_1x" in report
                assert "pct_coverage_10x" in report
                assert "pct_coverage_20x" in report

    anyio.run(_test)


def test_targeted_coverage_tool_location(mcp_server_params, sample_bam):
    """Test targeted_coverage_tool with location string input.

    This verifies that location strings like 'chr1:1000-2000' are correctly
    parsed and coverage is computed using mosdepth.
    """
    require_executable_tools(["mosdepth"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Use a location string format: "chr:start-end"
                result = await session.call_tool(
                    "targeted_coverage_tool",
                    {"bam_path": str(sample_bam), "location": "chr1:1000-2000"},
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert isinstance(payload, list)
                assert len(payload) == 1
                report = payload[0]
                assert report["chrom"] == "chr1"
                assert report["start"] == 1000
                assert report["end"] == 2000
                assert "mean_depth" in report
                assert isinstance(report["mean_depth"], (int, float))

                # Mosdepth provides threshold coverage percentages
                assert "pct_coverage_1x" in report
                assert "pct_coverage_10x" in report
                assert "pct_coverage_20x" in report

    anyio.run(_test)


def test_targeted_coverage_tool_gene_name(mcp_server_params, sample_bam, synthetic_gff3):
    """Test targeted_coverage_tool with gene name and annotation file.

    Verifies that gene coordinates are correctly looked up from GFF3 and
    coverage is computed using mosdepth.
    """
    require_executable_tools(["mosdepth"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "targeted_coverage_tool",
                    {
                        "bam_path": str(sample_bam),
                        "gene_name": "MOCK_GENE1",
                        "annotation_path": str(synthetic_gff3),
                    },
                )
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert isinstance(payload, list)
                assert len(payload) > 0

                # Check that reports have expected structure
                for report in payload:
                    assert "region_name" in report
                    assert "chrom" in report
                    assert "start" in report
                    assert "end" in report
                    assert "mean_depth" in report
                    assert isinstance(report["mean_depth"], (int, float))

                    # Mosdepth provides threshold coverage percentages
                    assert "pct_coverage_1x" in report
                    assert "pct_coverage_10x" in report
                    assert "pct_coverage_20x" in report

    anyio.run(_test)


def test_qc_bed_tool_valid(mcp_server_params, synthetic_bed_valid):
    """Test qc_bed_tool with valid BED file."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_bed_tool", {"path": str(synthetic_bed_valid)})
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(synthetic_bed_valid)
                assert payload["is_valid"] is True
                assert payload["total_intervals"] > 0
                assert payload["valid_intervals"] == payload["total_intervals"]
                assert payload["total_bases"] > 0
                assert isinstance(payload.get("issues", []), list)
                assert len(payload.get("issues", [])) == 0

    anyio.run(_test)


def test_qc_bed_tool_invalid(mcp_server_params, synthetic_bed_invalid):
    """Test qc_bed_tool with invalid BED file (should report issues)."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_bed_tool", {"path": str(synthetic_bed_invalid)})
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"] == str(synthetic_bed_invalid)
                assert payload["is_valid"] is False
                assert payload["total_intervals"] > 0
                assert payload["valid_intervals"] < payload["total_intervals"]
                assert isinstance(payload.get("issues", []), list)
                assert len(payload.get("issues", [])) > 0
                # Check that issues have expected structure
                for issue in payload["issues"]:
                    assert "line_number" in issue
                    assert "line_content" in issue
                    assert "issue" in issue

    anyio.run(_test)


def test_targeted_coverage_tool_missing_bam(mcp_server_params, tmp_path):
    """Test targeted_coverage_tool with missing BAM file (should return error)."""

    missing_bam = tmp_path / "does_not_exist.bam"

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "targeted_coverage_tool", {"bam_path": str(missing_bam), "location": "chr1:1000-2000"}
                )
                assert result.isError
                assert result.content
                assert "not_found" in _text_content(result.content[0]).text

    anyio.run(_test)


def test_qc_variants_tool_missing_vcf(mcp_server_params, tmp_path):
    """Test qc_variants_tool with missing VCF file (should return error)."""

    missing_vcf = tmp_path / "does_not_exist.vcf.gz"

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_variants_tool", {"path": str(missing_vcf)})
                assert result.isError
                assert result.content
                assert "not_found" in _text_content(result.content[0]).text

    anyio.run(_test)


def test_sequencing_summary_tool_missing_file(mcp_server_params, tmp_path):
    """Test sequencing_summary_tool with missing file (should return error)."""

    missing_file = tmp_path / "does_not_exist.txt"

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("sequencing_summary_tool", {"path": str(missing_file)})
                assert result.isError
                assert result.content
                assert "not_found" in _text_content(result.content[0]).text

    anyio.run(_test)


def test_targeted_coverage_tool_invalid_input_modes(mcp_server_params, sample_bam, synthetic_bed_valid):
    """Test targeted_coverage_tool with invalid input mode combinations."""
    require_executable_tools(["mosdepth"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Test: providing both bed_path and location (should fail)
                result = await session.call_tool(
                    "targeted_coverage_tool",
                    {"bam_path": str(sample_bam), "bed_path": str(synthetic_bed_valid), "location": "chr1:1000-2000"},
                )
                assert result.isError
                assert result.content
                assert "validation" in _text_content(result.content[0]).text

    anyio.run(_test)
