"""Boolean CLI flags reject implicit truthiness at the shared and MCP boundaries."""

import json
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.cli_wrappers import FlagValidationError, build_cli_args
from ont_qc_mcp.flag_schemas import TOOL_FLAGS


BOOLEAN_FLAGS = [
    (tool, key, flag.name)
    for tool, flags in TOOL_FLAGS.items()
    for flag in flags
    if flag.type == "bool"
    for key in flag.all_keys()
]
INVALID_VALUES = ["false", "true", "", 0, 1, 0.0, 1.0, [], [1], {}, {"enabled": False}]


@pytest.mark.parametrize("tool,key,cli_name", BOOLEAN_FLAGS)
@pytest.mark.parametrize("value", INVALID_VALUES)
def test_boolean_flags_reject_non_boolean_values(tool, key, cli_name, value):
    with pytest.raises(FlagValidationError, match=f"Flag {key} expects bool, got {type(value).__name__}"):
        build_cli_args(tool, {key: value})


@pytest.mark.parametrize("tool,key,cli_name", BOOLEAN_FLAGS)
@pytest.mark.parametrize("value", [True, False, None])
def test_boolean_flags_preserve_boolean_and_unset_values(tool, key, cli_name, value):
    assert build_cli_args(tool, {key: value}) == ([cli_name] if value is True else [])
    assert build_cli_args(tool, {}) == []


@pytest.mark.integration
def test_boolean_validation_through_mcp(tmp_path, mcp_server_params):
    missing_bam = tmp_path / "missing.bam"

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for value in [True, False, *INVALID_VALUES]:
                    result = await session.call_tool(
                        "alignment_qc",
                        {
                            "path": str(missing_bam),
                            "selection": {"include_unmapped": value, "exclude_flags": 1792},
                        },
                    )
                    content = cast(types.TextContent, result.content[0]).text
                    payload = json.loads(content)
                    if isinstance(value, bool):
                        assert result.is_error, content
                        assert payload["kind"] == "execution_error"
                    else:
                        assert result.is_error, content
                        assert payload["kind"] == "validation_error"
                        assert any(
                            issue["location"] == ["selection", "include_unmapped"] for issue in payload["issues"]
                        )

    anyio.run(check)
