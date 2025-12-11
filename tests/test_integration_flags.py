import asyncio
import json
from pathlib import Path

import pytest

from ont_qc_mcp.cli_wrappers import FlagValidationError, build_cli_args, cramino_stats
from ont_qc_mcp.config import ToolPaths


def test_build_cli_args_validation():
    args = build_cli_args("nanoq", {"min_len": 500, "threads": 2, "min_qual": 7.5})
    assert args == ["--min-len", "500", "--threads", "2", "--min-qual", "7.5"]

    with pytest.raises(FlagValidationError):
        build_cli_args("nanoq", {"min_len": "bad"})


def test_cramino_flags_applied_once(monkeypatch):
    captured = {}

    def fake_run(cmd, timeout=None):
        captured["cmd"] = cmd

        class Result:
            stdout = "{}"

        return Result()

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command", fake_run)
    cramino_stats(Path("dummy.bam"), ToolPaths(), include_hist=True, use_scaled=False, flags={"threads": 4})
    cmd = captured["cmd"]
    assert cmd[0] == "cramino"
    assert cmd.count("--hist-count") == 1
    hist_idx = cmd.index("--hist-count")
    assert hist_idx + 1 < len(cmd) and cmd[hist_idx + 1].endswith(".cramino.hist.tsv")
    assert "--format" in cmd
    assert "json" in cmd
    assert "--threads" in cmd and "4" in cmd


def test_resources_exposed():
    from ont_qc_mcp import app_server as srv

    resources = asyncio.run(srv.list_resources())
    uris = {str(r.uri) for r in resources}
    assert "tool://flags/nanoq" in uris
    assert "tool://recipes/nanoq" in uris


def test_read_resource_flags_and_recipes():
    from ont_qc_mcp import app_server as srv

    flag_contents = asyncio.run(srv.read_resource("tool://flags/nanoq"))
    flag_payload = json.loads(flag_contents[0].content)
    assert flag_payload["tool"] == "nanoq"
    assert flag_payload["flags"]

    recipe_contents = asyncio.run(srv.read_resource("tool://recipes/nanoq"))
    recipe_payload = json.loads(recipe_contents[0].content)
    assert recipe_payload["tool"] == "nanoq"
    assert "strict_qc" in recipe_payload["recipes"]
