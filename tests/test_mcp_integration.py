"""
Integration tests for the MCP server using the stdio client.

These tests spawn the server as a subprocess and communicate via MCP protocol.
"""

import json
import importlib
import ont_qc_mcp
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from conftest import require_executable_tools


REQUIRED_TOOLS = ["nanoq", "chopper", "cramino", "mosdepth", "samtools"]
pytestmark = pytest.mark.integration


def _text_content(content: types.Content) -> types.TextContent:
    return cast(types.TextContent, content)


def _text_resource(content: types.ResourceContents) -> types.TextResourceContents:
    return cast(types.TextResourceContents, content)


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
                assert "qc_reads_fastq_tool" in tool_names
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

                flags = await session.read_resource(cast(AnyUrl, "tool://flags/nanoq"))
                flag_payload = json.loads(_text_resource(flags.contents[0]).text)
                assert flag_payload["tool"] == "nanoq"
                assert flag_payload["flags"]

                recipes = await session.read_resource(cast(AnyUrl, "tool://recipes/nanoq"))
                recipe_payload = json.loads(_text_resource(recipes.contents[0]).text)
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
                payload = json.loads(_text_content(result.content[0]).text)
                assert "available" in payload
                assert isinstance(payload["available"], dict)

    anyio.run(_test)


def test_alignment_workflow_smoke(mcp_server_params, sample_bam):
    """Simulate a minimal alignment workflow using real tools if available."""

    require_executable_tools(REQUIRED_TOOLS)

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                qc = await session.call_tool("qc_alignment_tool", {"path": str(sample_bam)})
                assert not qc.isError, f"qc_alignment_tool failed: {_text_content(qc.content[0]).text}"
                qc_payload = json.loads(_text_content(qc.content[0]).text)
                assert qc_payload["length_histogram"], "Expected length_histogram from cramino"
                assert qc_payload.get("total_reads", 0) > 0
                assert qc_payload.get("mean_identity") is not None

                coverage = await session.call_tool("coverage_stats_tool", {"path": str(sample_bam)})
                assert not coverage.isError

                coverage_data = json.loads(_text_content(coverage.content[0]).text)
                assert coverage_data

    anyio.run(_test)


def test_header_metadata_tool_vcf(mcp_server_params, tmp_path):
    """Header metadata tool should parse VCF headers without external CLIs."""
    header_text = "\n".join(
        [
            "##fileformat=VCFv4.3",
            "##source=test-suite",
            "##contig=<ID=chr1,length=5000>",
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
                result = await session.call_tool("header_metadata_tool", {"path": str(vcf_path), "file_type": "vcf"})
                assert not result.isError
                assert result.content
                payload = json.loads(_text_content(result.content[1]).text)
                assert payload["format"] == "vcf"
                assert payload["samples"][0]["name"] == "sample1"

    anyio.run(_test)


def test_header_metadata_tool_real_vcf(mcp_server_params, sample_vcf):
    """Header metadata tool should parse the real gzipped VCF fixture."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("header_metadata_tool", {"path": str(sample_vcf), "file_type": "vcf"})
                assert not result.isError
                assert result.content
                payload = json.loads(_text_content(result.content[1]).text)
                assert payload["format"] == "vcf"
                assert payload["samples"]
                assert payload["references"]

    anyio.run(_test)


def test_qc_reads_tool_real_fastq(mcp_server_params, sample_fastq):
    """qc_reads_fastq_tool should operate on the real FASTQ fixture."""

    require_executable_tools(["nanoq"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_reads_fastq_tool", {"path": str(sample_fastq)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"]
                assert payload["read_count"] > 0
                assert payload.get("mean_len", 0) > 0

    anyio.run(_test)


def test_read_length_distribution_tool_real_fastq(mcp_server_params, sample_fastq):
    """read_length_distribution_fastq_tool should operate on the real FASTQ fixture."""

    require_executable_tools(["nanoq"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_length_distribution_fastq_tool", {"path": str(sample_fastq)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["percentiles"]
                assert payload["histogram"] is not None

    anyio.run(_test)


def test_qscore_distribution_tool_real_fastq(mcp_server_params, sample_fastq):
    """qscore_distribution_fastq_tool should operate on the real FASTQ fixture."""

    require_executable_tools(["nanoq"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qscore_distribution_fastq_tool", {"path": str(sample_fastq)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["histogram"] is not None

    anyio.run(_test)


def test_read_length_distribution_bam_tool(mcp_server_params, sample_bam):
    """read_length_distribution_bam_tool should stream BAM -> nanoq and return histogram."""

    require_executable_tools(["samtools", "nanoq"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_length_distribution_bam_tool", {"path": str(sample_bam)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["file"]
                assert payload["percentiles"] is not None
                assert payload["histogram"] is not None

    anyio.run(_test)


def test_qscore_distribution_bam_tool(mcp_server_params, sample_bam):
    """qscore_distribution_bam_tool should stream BAM -> nanoq and return qscore histogram."""

    require_executable_tools(["samtools", "nanoq"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qscore_distribution_bam_tool", {"path": str(sample_bam)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["histogram"] is not None

    anyio.run(_test)


def test_filter_reads_tool_real_fastq(mcp_server_params, sample_fastq, tmp_path):
    """filter_reads_fastq_tool should process the real FASTQ fixture and produce output."""

    require_executable_tools(["chopper"])

    output_fastq = tmp_path / "filtered.fastq"

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "filter_reads_fastq_tool", {"path": str(sample_fastq), "output_fastq": str(output_fastq)}
                )
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["command"]
                assert output_fastq.exists()

    anyio.run(_test)


def test_header_metadata_tool_real_bam(mcp_server_params, sample_bam):
    """Header metadata tool should parse BAM headers."""

    require_executable_tools(["samtools"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("header_metadata_tool", {"path": str(sample_bam), "file_type": "bam"})
                assert not result.isError
                payload = json.loads(_text_content(result.content[1]).text)
                assert payload["format"] == "bam"
                assert payload["references"]
                assert payload["programs"] is not None

    anyio.run(_test)


def test_alignment_error_profile_tool_real_bam(mcp_server_params, sample_bam):
    """alignment_error_profile_tool should run samtools stats on the real BAM."""

    require_executable_tools(["samtools"])

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("alignment_error_profile_tool", {"path": str(sample_bam)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert "mismatch_rate" in payload

    anyio.run(_test)


def test_alignment_summary_tool_real_bam(mcp_server_params, sample_bam):
    """alignment_summary_tool should aggregate cramino + mosdepth on real BAM."""

    require_executable_tools(REQUIRED_TOOLS)

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("alignment_summary_tool", {"path": str(sample_bam)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["alignment"]
                assert payload["coverage"]
                assert payload["alignment"].get("total_reads", 0) > 0

    anyio.run(_test)


def test_alignment_summary_tool_highdepth_bam(mcp_server_params, sample_bam_highdepth):
    """alignment_summary_tool should show non-zero coverage on the high-depth synthetic BAM."""

    require_executable_tools(REQUIRED_TOOLS)

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("alignment_summary_tool", {"path": str(sample_bam_highdepth)})
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["coverage"]["mean_depth"] > 0
                assert payload["alignment"].get("total_reads", 0) >= 50

    anyio.run(_test)


def test_missing_fastq_returns_not_found_error(mcp_server_params, tmp_path):
    """FASTQ tools should surface not_found errors for missing paths."""

    missing_fastq = tmp_path / "does_not_exist.fastq"

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("qc_reads_fastq_tool", {"path": str(missing_fastq)})
                assert result.isError
                assert result.content
                assert "not_found" in _text_content(result.content[0]).text

    anyio.run(_test)


def test_invalid_flags_return_validation_error(mcp_server_params, sample_fastq):
    """Invalid flag types should be returned as validation errors."""

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "qc_reads_fastq_tool", {"path": str(sample_fastq), "flags": {"threads": "bad"}}
                )
                assert result.isError
                assert result.content
                assert "validation" in _text_content(result.content[0]).text

    anyio.run(_test)


def test_bam_streaming_timeout_surface_runtime_error(mcp_server_params, tmp_path, monkeypatch):
    """Streaming pipeline should fail cleanly on timeout."""

    sleep_script = tmp_path / "sleep_tool.py"
    sleep_script.write_text("#!/usr/bin/env python3\nimport time\nimport sys\ntime.sleep(5)\n")
    sleep_script.chmod(0o755)

    # Force very short timeouts and redirect samtools/nanoq to the sleeping stub.
    monkeypatch.setenv("SAMTOOLS", str(sleep_script))
    monkeypatch.setenv("NANOQ", str(sleep_script))
    monkeypatch.setenv("MCP_TIMEOUT_SAMTOOLS", "1")
    monkeypatch.setenv("MCP_TIMEOUT_NANOQ", "1")

    dummy_bam = tmp_path / "dummy.bam"
    dummy_bam.write_text("bam")

    async def _test():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_length_distribution_bam_tool", {"path": str(dummy_bam)})
                assert result.isError
                assert result.content
                payload = _text_content(result.content[0]).text
                # Accept runtime/timeout errors and environments lacking samtools/nanoq.
                assert "runtime" in payload or "Timeout" in payload or "not_found" in payload

    anyio.run(_test)


def test_alignment_summary_serial_with_concurrency_1(monkeypatch):
    """Ensure MCP_MAX_CONCURRENCY=1 serializes calls without deadlock."""
    monkeypatch.setenv("MCP_MAX_CONCURRENCY", "1")
    srv = importlib.reload(ont_qc_mcp.app_server)

    async def _run():
        results: list[types.CallToolResult] = []

        async def call_tool():
            res = await srv.dispatch_tool("env_status", {})
            results.append(res)

        async with anyio.create_task_group() as tg:
            tg.start_soon(call_tool)
            tg.start_soon(call_tool)

        return results

    outputs = anyio.run(_run)
    assert len(outputs) == 2
    assert all(not r.isError for r in outputs)
