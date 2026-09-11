"""Check the subprocess environment used by MCP integration tests."""

import pytest


@pytest.mark.parametrize("key", ["NANOQ", "CHOPPER", "CRAMINO", "MOSDEPTH", "SAMTOOLS", "BCFTOOLS"])
def test_mcp_server_receives_explicit_tool_path(monkeypatch, request, key):
    expected = f"/selected/toolchain/bin/{key.lower()}"
    monkeypatch.setenv(key, expected)
    params = request.getfixturevalue("mcp_server_params")
    assert params.env.get(key) == expected
