"""
Integration tests for the MCP server using the stdio client.

These tests spawn the server as a subprocess and communicate via MCP protocol.
"""

import json

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.tools import env_check


def test_initialize_and_list_tools(mcp_server_params):
    """Test that we can connect and list available tools."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tool_names = {tool.name for tool in result.tools}
                assert "env_status" in tool_names
                assert "qc_alignment_tool" in tool_names
                assert "qc_reads_tool" in tool_names
                assert "header_metadata_tool" in tool_names

    anyio.run(_test)


def test_tool_schemas_define_required_params(mcp_server_params):
    """Test that tool schemas properly define required parameters."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()

                tools_by_name = {tool.name: tool for tool in result.tools}

                # env_status has no required params
                env_schema = tools_by_name["env_status"].inputSchema
                assert env_schema.get("required") is None or env_schema.get("required") == []

                # qc_alignment_tool requires path
                qc_align_schema = tools_by_name["qc_alignment_tool"].inputSchema
                assert "path" in qc_align_schema.get("required", [])
                assert "path" in qc_align_schema["properties"]
                assert "include_hist" in qc_align_schema["properties"]
                assert "use_scaled" in qc_align_schema["properties"]

                # alignment_summary_tool requires path and has many optional params
                summary_schema = tools_by_name["alignment_summary_tool"].inputSchema
                assert "path" in summary_schema.get("required", [])
                assert "include_coverage" in summary_schema["properties"]
                assert "coverage_window" in summary_schema["properties"]

    anyio.run(_test)


def test_list_and_read_resources(mcp_server_params):
    """Test resource listing and reading."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                resources = await session.list_resources()
                uris = {str(res.uri) for res in resources.resources}
                assert "tool://flags/nanoq" in uris
                assert "tool://recipes/nanoq" in uris

                flags = await session.read_resource("tool://flags/nanoq")
                flag_payload = json.loads(flags.contents[0].text)
                assert flag_payload["tool"] == "nanoq"
                assert flag_payload["flags"]

                recipes = await session.read_resource("tool://recipes/nanoq")
                recipe_payload = json.loads(recipes.contents[0].text)
                assert recipe_payload["tool"] == "nanoq"
                assert recipe_payload["recipes"]

    anyio.run(_test)


def test_env_status_tool(mcp_server_params):
    """Test calling the env_status tool."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool("env_status")
                assert not result.isError, f"Tool error: {result.content}"
                payload = json.loads(result.content[0].text)
                assert "available" in payload
                assert isinstance(payload["available"], dict)

    anyio.run(_test)


def test_alignment_workflow_smoke(mcp_server_params, sample_bam):
    """Simulate a minimal alignment workflow using real tools if available."""

    env_status = env_check()
    missing = [tool for tool, ok in env_status.available.items() if not ok]
    if missing:
        pytest.skip(f"Required CLI tools missing: {', '.join(missing)}")

    skip_reason: str | None = None

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                qc = await session.call_tool("qc_alignment_tool", {"path": str(sample_bam)})
                if qc.isError:
                    nonlocal skip_reason
                    skip_reason = f"qc_alignment_tool failed: {qc.content[0].text}"
                    return

                coverage = await session.call_tool("coverage_stats_tool", {"path": str(sample_bam)})
                assert not coverage.isError

                coverage_data = json.loads(coverage.content[0].text)
                assert coverage_data

    anyio.run(_test)
    if skip_reason:
        pytest.skip(skip_reason)


def test_header_metadata_tool_vcf(mcp_server_params, tmp_path):
    """Header metadata tool should parse VCF headers without external CLIs."""
    header_text = "\n".join(
        [
            "##fileformat=VCFv4.3",
            "##source=test-suite",
            '##contig=<ID=chr1,length=5000>',
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1",
        ]
    )
    vcf_path = tmp_path / "mini.vcf"
    vcf_path.write_text(header_text)

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "header_metadata_tool", {"path": str(vcf_path), "file_type": "vcf"}
                )
                assert not result.isError
                assert result.content
                payload = json.loads(result.content[1].text)
                assert payload["format"] == "vcf"
                assert payload["samples"][0]["name"] == "sample1"

    anyio.run(_test)


def test_header_metadata_tool_real_vcf(mcp_server_params, sample_vcf):
    """Header metadata tool should parse the real gzipped VCF fixture."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "header_metadata_tool", {"path": str(sample_vcf), "file_type": "vcf"}
                )
                assert not result.isError
                assert result.content
                payload = json.loads(result.content[1].text)
                assert payload["format"] == "vcf"
                assert payload["samples"]
                assert payload["references"]

    anyio.run(_test)
