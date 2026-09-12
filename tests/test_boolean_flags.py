"""Boolean CLI flags reject implicit truthiness at the shared and MCP boundaries."""

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
    # A controlled producer records command execution and emits a valid summary.
    # The BAM contents are immaterial to this argument-validation regression.
    bam = tmp_path / "input.bam"
    bam.write_bytes(b"controlled input")
    receipt = tmp_path / "args.json"
    executable = tmp_path / "mosdepth"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "Path(sys.argv[-2] + '.mosdepth.summary.txt').write_text(\n"
        "    'chrom\\tlength\\tbases\\tmean\\tmin\\tmax\\n'\n"
        "    'chrA\\t100\\t200\\t2\\t2\\t2\\n'\n"
        "    'total\\t100\\t200\\t2\\t2\\t2\\n')\n"
    )
    executable.chmod(0o755)
    params = mcp_server_params.model_copy(
        update={"env": {**(mcp_server_params.env or {}), "MOSDEPTH": str(executable)}}
    )

    async def check():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for value in [True, False, None, *INVALID_VALUES]:
                    receipt.unlink(missing_ok=True)
                    result = await session.call_tool(
                        "coverage_stats_tool", {"path": str(bam), "flags": {"fast_mode": value}}
                    )
                    content = cast(types.TextContent, result.content[0]).text
                    payload = json.loads(content)
                    if value is None or isinstance(value, bool):
                        assert not result.isError, content
                        assert ("--fast-mode" in json.loads(receipt.read_text())) == (value is True)
                        assert payload["coverage_by_contig"][0]["mean_depth"] == 2
                    else:
                        assert result.isError, content
                        assert payload["kind"] == "validation"
                        assert "Flag fast_mode expects bool" in payload["message"]
                        assert not receipt.exists()

    anyio.run(check)
