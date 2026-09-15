"""End-to-end MCP SDK checks for the public API v2 catalog."""

from __future__ import annotations

import json
from typing import cast

import anyio
import mcp_types as types
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp.v2_contracts import api_v2_contracts

pytestmark = pytest.mark.integration


def _text(content: types.ContentBlock) -> types.TextContent:
    return cast(types.TextContent, content)


def _resource(content: types.ResourceContents) -> types.TextResourceContents:
    return cast(types.TextResourceContents, content)


def test_initialize_lists_exact_public_catalog(mcp_server_params) -> None:
    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == set(api_v2_contracts())

    anyio.run(check)


def test_schemas_and_alignment_coverage_recipe(mcp_server_params) -> None:
    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = {tool.name: tool for tool in (await session.list_tools()).tools}
                for name, contract in api_v2_contracts().items():
                    assert listed[name].input_schema == contract.request_model.model_json_schema(mode="validation")
                resource = await session.read_resource("tool://recipes/alignment_qc")
                payload = json.loads(_resource(resource.contents[0]).text)
                calls = payload["recipes"]["alignment_and_coverage"]["calls"]
                assert [call["tool"] for call in calls] == ["alignment_qc", "coverage_qc"]

    anyio.run(check)


def test_environment_status(mcp_server_params) -> None:
    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("environment_status", {})
                assert not result.is_error
                payload = json.loads(_text(result.content[0]).text)
                assert isinstance(payload["available"], dict)
                assert isinstance(payload["resolved_paths"], dict)

    anyio.run(check)


def test_read_qc_fastq(mcp_server_params, sample_fastq) -> None:
    require_executable_tools(["nanoq"])

    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "read_qc",
                    {
                        "path": str(sample_fastq),
                        "metrics": ["length", "read_quality", "length_distribution", "quality_distribution"],
                    },
                )
                assert not result.is_error, _text(result.content[0]).text
                payload = json.loads(_text(result.content[0]).text)
                group = payload["results"][0]
                assert group["length"]["read_count"] > 0
                assert group["length_distribution"]["histogram"]
                assert group["quality_distribution"]["histogram"]

    anyio.run(check)


def test_alignment_and_coverage_recipe_calls(mcp_server_params, sample_bam) -> None:
    require_executable_tools(["samtools", "cramino", "mosdepth"])

    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                alignment = await session.call_tool("alignment_qc", {"path": str(sample_bam)})
                assert not alignment.is_error, _text(alignment.content[0]).text
                alignment_payload = json.loads(_text(alignment.content[0]).text)
                assert alignment_payload["results"][0]["counts"]["eligible_records"] >= 0

                coverage = await session.call_tool("coverage_qc", {"path": str(sample_bam)})
                assert not coverage.is_error, _text(coverage.content[0]).text
                coverage_payload = json.loads(_text(coverage.content[0]).text)
                assert coverage_payload["rows"]
                assert coverage_payload["union_summary"]["reference_bases"] > 0

    anyio.run(check)


@pytest.mark.parametrize("kind", ["bam", "vcf"])
def test_header_info(mcp_server_params, sample_bam, sample_vcf, kind: str) -> None:
    if kind == "bam":
        require_executable_tools(["samtools"])
    path = sample_bam if kind == "bam" else sample_vcf

    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("header_info", {"path": str(path)})
                assert not result.is_error, _text(result.content[0]).text
                payload = json.loads(_text(result.content[0]).text)
                assert payload["format"] == kind
                assert payload["references"]

    anyio.run(check)


def test_filter_reads_then_read_qc(mcp_server_params, sample_fastq, tmp_path) -> None:
    require_executable_tools(["chopper", "nanoq"])
    output = tmp_path / "filtered.fastq.gz"

    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                filtered = await session.call_tool(
                    "filter_reads",
                    {
                        "path": str(sample_fastq),
                        "output_fastq": str(output),
                        "selection": {"minlength": 1},
                    },
                )
                assert not filtered.is_error, _text(filtered.content[0]).text
                assert output.is_file()
                qc = await session.call_tool("read_qc", {"path": str(output)})
                assert not qc.is_error, _text(qc.content[0]).text
                assert json.loads(_text(qc.content[0]).text)["results"][0]["length"]["read_count"] > 0

    anyio.run(check)


def test_unknown_legacy_alias_is_rejected(mcp_server_params) -> None:
    async def check() -> None:
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("alignment_summary_tool", {"path": "unused.bam"})
                assert result.is_error
                payload = json.loads(_text(result.content[0]).text)
                assert payload["kind"] == "validation_error"
                assert "Unknown tool" in payload["issues"][0]["message"]

    anyio.run(check)
