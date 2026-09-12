"""Targeted mosdepth owns its directory until a successful return transfers it."""

import asyncio
import shutil
from pathlib import Path
from threading import Event

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.process_control import CANCEL_EVENT
from ont_qc_mcp.utils import CommandError, CommandResult


@pytest.mark.parametrize("failure", ["command", "cancel", "missing-binary", "missing-output", "late-cancel"])
def test_targeted_failure_cleans_unreturned_directory(tmp_path, monkeypatch, failure):
    created = tmp_path / "mosdepth_targeted_owned"

    def allocate(**kwargs):
        created.mkdir()
        return str(created)

    event = Event()
    token = CANCEL_EVENT.set(event)

    def run(cmd, **kwargs):
        (created / "partial-output").write_text("partial")
        if failure == "command":
            raise CommandError(CommandResult(cmd, 1, "", "bad input"))
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "missing-binary":
            raise FileNotFoundError("mosdepth")
        if failure == "late-cancel":
            (created / "coverage.regions.bed.gz").touch()
            event.set()

    monkeypatch.setattr(cli.tempfile, "mkdtemp", allocate)
    monkeypatch.setattr(cli, "run_command", run)
    expected = (
        asyncio.CancelledError
        if failure in {"cancel", "late-cancel"}
        else FileNotFoundError
        if failure == "missing-binary"
        else RuntimeError
    )
    try:
        with pytest.raises(expected):
            cli.run_mosdepth_targeted(tmp_path / "input.bam", tmp_path / "regions.bed", ToolPaths())
        assert not created.exists()
    finally:
        CANCEL_EVENT.reset(token)
        shutil.rmtree(created, ignore_errors=True)


@pytest.mark.parametrize("with_thresholds", [False, True])
def test_targeted_success_transfers_output_directory(tmp_path, monkeypatch, with_thresholds):
    def run(cmd, **kwargs):
        prefix = Path(cmd[-2])
        prefix.with_suffix(".regions.bed.gz").touch()
        if with_thresholds:
            prefix.with_suffix(".thresholds.bed.gz").touch()

    monkeypatch.setattr(cli, "run_command", run)
    regions, thresholds, output = cli.run_mosdepth_targeted(
        tmp_path / "input.bam", tmp_path / "regions.bed", ToolPaths()
    )
    try:
        assert regions.exists() and output.exists()
        assert (thresholds is not None) == with_thresholds
        if thresholds is not None:
            assert thresholds.exists()
    finally:
        shutil.rmtree(output)
