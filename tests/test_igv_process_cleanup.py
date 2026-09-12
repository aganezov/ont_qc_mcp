"""Docker cleanup uses an operation-specific identity even after cancellation."""

import asyncio
from threading import Event

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.utils import CommandError, CommandResult


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_docker_run_removes_only_its_container(tmp_path, monkeypatch, cancelled):
    calls = []
    tools = ToolPaths(docker="fixture-docker")
    failure = asyncio.CancelledError() if cancelled else CommandError(CommandResult(["docker"], 124, "", "timeout"))

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "run":
            raise failure
        return CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(cli, "run_command", run)
    monkeypatch.setattr(cli, "which", lambda cmd: cmd)
    expected = asyncio.CancelledError if cancelled else RuntimeError
    with pytest.raises(expected):
        cli.run_igv_snapshot(tmp_path / "batch", tmp_path, tools, force_runtime="docker")
    assert len(calls) == 2
    name = calls[0][calls[0].index("--name") + 1]
    assert name.startswith("ont-qc-igv-")
    assert calls[0][0] == tools.docker
    assert calls[1] == [tools.docker, "rm", "--force", name]


def test_docker_cleanup_failure_preserves_cancellation_and_reports_uncertainty(tmp_path, monkeypatch, caplog):
    def run(cmd, **kwargs):
        if cmd[1] == "run":
            raise asyncio.CancelledError()
        raise CommandError(CommandResult(cmd, 1, "", "daemon unreachable"))

    monkeypatch.setattr(cli, "run_command", run)
    monkeypatch.setattr(cli, "which", lambda cmd: cmd)
    with pytest.raises(asyncio.CancelledError):
        cli.run_igv_snapshot(tmp_path / "batch", tmp_path, ToolPaths(), force_runtime="docker")
    assert "cleanup" in caplog.text and "daemon unreachable" in caplog.text


def test_docker_cleanup_runs_after_cancellation_event_is_set(tmp_path, monkeypatch):
    from ont_qc_mcp.process_control import CANCEL_EVENT, check_cancelled

    event = Event()
    token = CANCEL_EVENT.set(event)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "run":
            event.set()
        check_cancelled()
        assert kwargs["timeout"] <= 5
        return CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(cli, "run_command", run)
    monkeypatch.setattr(cli, "which", lambda cmd: cmd)
    try:
        with pytest.raises(asyncio.CancelledError):
            cli.run_igv_snapshot(tmp_path / "batch", tmp_path, ToolPaths(), force_runtime="docker")
        assert len(calls) == 2 and calls[-1][1:3] == ["rm", "--force"]
        assert event.is_set() and CANCEL_EVENT.get() is event
    finally:
        CANCEL_EVENT.reset(token)
