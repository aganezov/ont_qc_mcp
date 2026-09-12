"""Numeric CLI flags reject booleans without changing accepted numeric values."""

import json
import sys
from typing import cast

import anyio
import pytest
from mcp import types
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
@pytest.mark.parametrize(
    "key,cli_name,valid_values",
    [("min_len", "--min-len", [0, 1, 7, None]), ("min_qual", "--min-qual", [0, 1, 7, 7.5, None])],
)
def test_numeric_validation_through_mcp(tmp_path, mcp_server_params, key, cli_name, valid_values):
    fastq = tmp_path / "input.fastq"
    fastq.write_text("@read\nACGT\n+\nIIII\n")
    receipt = tmp_path / "args.json"
    executable = tmp_path / "nanoq"
    # This controlled producer verifies argument validation and command execution,
    # not the bioinformatics program's interpretation of these numeric options.
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'reads': 1, 'bases': 4, 'shortest': 4, 'longest': 4, 'mean_length': 4,\n"
        "    'median_length': 4, 'n50': 4, 'mean_quality': 40, 'median_quality': 40}))\n"
    )
    executable.chmod(0o755)
    params = mcp_server_params.model_copy(
        update={
            "env": {
                **(mcp_server_params.env or {}),
                "NANOQ": str(executable),
                "MCP_NANOQ_AUX_STATS": "0",
            }
        }
    )

    async def check():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for value in [*valid_values, True, False]:
                    receipt.unlink(missing_ok=True)
                    result = await session.call_tool("qc_reads_fastq_tool", {"path": str(fastq), "flags": {key: value}})
                    content = cast(types.TextContent, result.content[0]).text
                    payload = json.loads(content)
                    if isinstance(value, bool):
                        assert result.isError, content
                        assert payload["kind"] == "validation"
                        assert f"Flag {key} expects" in payload["message"]
                        assert "got bool" in payload["message"]
                        assert not receipt.exists()
                    else:
                        assert not result.isError, content
                        assert payload["read_count"] == 1
                        args = json.loads(receipt.read_text())
                        if value is None:
                            assert cli_name not in args
                        else:
                            assert args[args.index(cli_name) + 1] == str(value)

    anyio.run(check)
