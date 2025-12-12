import asyncio
import io
import json
import subprocess
from pathlib import Path

import pytest

from ont_qc_mcp.cli_wrappers import (
    FlagValidationError,
    build_cli_args,
    chopper_filter,
    cramino_stats,
    nanoq_from_bam_streaming,
)
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.utils import CommandError, CommandResult


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
    assert Path(cmd[0]).name == "cramino"
    assert cmd.count("--hist-count") == 1
    hist_idx = cmd.index("--hist-count")
    assert hist_idx + 1 < len(cmd) and cmd[hist_idx + 1].endswith(".cramino.hist.tsv")
    assert "--format" in cmd
    assert "json" in cmd
    assert "--threads" in cmd and "4" in cmd


def test_chopper_dual_failure_surfaces_both_errors(monkeypatch, tmp_path):
    fastq_path = tmp_path / "reads.fastq"
    fastq_path.write_text("@r1\nACGT\n+\n!!!!\n")

    def fake_run_command_with_retry(*args, **kwargs):
        raise CommandError(CommandResult(cmd=["chopper"], returncode=1, stdout="", stderr="disk full"))

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run_command_with_retry)

    with pytest.raises(RuntimeError) as exc_info:
        chopper_filter(fastq_path, ToolPaths())

    assert "chopper failed" in str(exc_info.value)
    assert "disk full" in str(exc_info.value)


def test_streaming_timeout_identifies_hung_stage(monkeypatch, tmp_path):
    bam_path = tmp_path / "dummy.bam"
    bam_path.write_text("bam")

    class FakeProc:
        def __init__(self, name: str, timeout_on_communicate: bool = False, running: bool = True, stderr: str = ""):
            self.name = name
            self._timeout_on_communicate = timeout_on_communicate
            self._running = running
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(stderr.encode("utf-8")) if stderr else io.BytesIO(b"")
            self.returncode = 0

        def poll(self):
            return None if self._running else self.returncode

        def communicate(self, timeout=None):
            if self._timeout_on_communicate:
                raise subprocess.TimeoutExpired(cmd=[self.name], timeout=timeout or 0)
            return b"", b""

        def terminate(self):
            self._running = False

        def kill(self):
            self._running = False

        def wait(self, timeout=None):
            self._running = False
            return self.returncode

    procs = [
        FakeProc("samtools", timeout_on_communicate=False, running=False, stderr="sam err line 1\nsam err line 2"),
        FakeProc("nanoq", timeout_on_communicate=True, running=True),
    ]

    def fake_popen(cmd, stdout=None, stderr=None, stdin=None):
        assert procs, "No more fake processes available"
        return procs.pop(0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as exc_info:
        nanoq_from_bam_streaming(bam_path, ToolPaths())

    msg = str(exc_info.value)
    assert "Timeout" in msg
    assert "hung at nanoq" in msg or "hung at both" in msg or "hung at nanoq" in msg
    assert "samtools stderr tail" in msg


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
