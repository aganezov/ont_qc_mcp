"""Check the subprocess environment used by MCP integration tests."""

import json
from typing import cast

import anyio
import pytest
import mcp_types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ont_qc_mcp.cli_wrappers import detect_container_runtime
from ont_qc_mcp.config import ToolPaths


@pytest.mark.parametrize(
    "key", ["NANOQ", "CHOPPER", "CRAMINO", "MOSDEPTH", "SAMTOOLS", "BCFTOOLS", "DOCKER", "APPTAINER", "SINGULARITY"]
)
def test_mcp_server_receives_explicit_tool_path(monkeypatch, request, key):
    expected = f"/selected/toolchain/bin/{key.lower()}"
    monkeypatch.setenv(key, expected)
    params = request.getfixturevalue("mcp_server_params")
    assert params.env.get(key) == expected


@pytest.mark.integration
@pytest.mark.parametrize("selected", ["docker", "apptainer", "singularity"])
def test_mcp_server_preserves_container_runtime(monkeypatch, request, tmp_path, selected):
    # Keep PATH fallbacks distinct from the selected executable so dropping an
    # override changes the child's runtime or resolved path on every host.
    selected_path = tmp_path / f"selected-{selected}"
    selected_path.write_text("#!/bin/sh\nexit 0\n")
    selected_path.chmod(0o755)
    for name in ("docker", "apptainer", "singularity"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        monkeypatch.setenv(name.upper(), str(selected_path) if name == selected else "__disabled_runtime__")
    monkeypatch.setenv("PATH", str(tmp_path))
    expected = "docker" if selected == "docker" else "apptainer"
    assert detect_container_runtime(ToolPaths()) == expected
    params = request.getfixturevalue("mcp_server_params")

    async def check_child():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("env_status", {})
                assert not result.is_error
                payload = json.loads(cast(types.TextContent, result.content[0]).text)
                assert payload["igv_runtime"] == expected
                assert payload["resolved_paths"][selected] == str(selected_path)

    anyio.run(check_child)
