"""Numeric CLI flags reject booleans without changing accepted numeric values."""

import json
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.cli_wrappers import FlagValidationError, build_cli_args
from ont_qc_mcp.flag_schemas import TOOL_FLAGS


NUMERIC_FLAGS = [
    pytest.param(tool, key, flag, id=f"{tool}-{key}")
    for tool, flags in TOOL_FLAGS.items()
    for flag in flags
    if flag.type in {"int", "float"}
    for key in flag.all_keys()
]


@pytest.mark.parametrize("tool,key,flag", NUMERIC_FLAGS)
@pytest.mark.parametrize("value", [True, False])
def test_numeric_flags_reject_booleans(tool, key, flag, value):
    flags = {key: value}
    if tool == "chopper" and key in {"headcrop", "tailcrop"}:
        flags["trim_approach"] = "fixed-crop"
    with pytest.raises(FlagValidationError, match=f"Flag {key} expects {flag.type}, got bool"):
        build_cli_args(tool, flags)


@pytest.mark.parametrize("tool,key,flag", NUMERIC_FLAGS)
@pytest.mark.parametrize("value", [None, 0, 7])
def test_numeric_flags_keep_numeric_and_unset_values(tool, key, flag, value):
    flags = {key: value}
    if tool == "chopper" and key in {"headcrop", "tailcrop"}:
        flags["trim_approach"] = "fixed-crop"
    args = build_cli_args(tool, flags)
    if value is None:
        assert flag.name not in args
    else:
        assert args[args.index(flag.name) + 1] == str(value)


@pytest.mark.parametrize("key", ["min_qual", "max_qual"])
def test_float_flags_keep_fractional_values(key):
    assert build_cli_args("nanoq", {key: 7.5}) == [f"--{key.replace('_', '-')}", "7.5"]


@pytest.mark.integration
def test_numeric_validation_through_mcp(tmp_path, mcp_server_params):
    missing_bam = tmp_path / "missing.bam"

    async def check():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for value in [0, 1, 7, True, False, 7.5, None]:
                    result = await session.call_tool(
                        "alignment_qc",
                        {"path": str(missing_bam), "selection": {"min_mapq": value}},
                    )
                    content = cast(types.TextContent, result.content[0]).text
                    payload = json.loads(content)
                    if type(value) is int:
                        assert result.is_error, content
                        assert payload["kind"] == "execution_error"
                    else:
                        assert result.is_error, content
                        assert payload["kind"] == "validation_error"
                        assert any(issue["location"] == ["selection", "min_mapq"] for issue in payload["issues"])

    anyio.run(check)
