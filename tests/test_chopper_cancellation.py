"""Cancellation before publication leaves existing output and no staging files."""

import asyncio
import tempfile
from threading import Event
from pathlib import Path

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.process_control import CANCEL_EVENT


@pytest.mark.parametrize("output_name", [None, "filtered.fastq", "filtered.fastq.gz"])
def test_cancellation_before_publication(tmp_path, monkeypatch, output_name):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = tmp_path / "input.fastq"
    source.write_text("@r\nACGT\n+\nIIII\n")
    output = tmp_path / output_name if output_name else None
    if output is not None:
        output.write_bytes(b"original")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    event = Event()
    token = CANCEL_EVENT.set(event)

    def run(cmd, **kwargs):
        Path(kwargs["stdout_path"]).write_bytes(source.read_bytes())
        if output_name != "filtered.fastq.gz":
            event.set()

    original_copy = cli.copyfileobj

    def copy(*args, **kwargs):
        original_copy(*args, **kwargs)
        event.set()

    monkeypatch.setattr(cli, "run_command_with_retry", run)
    monkeypatch.setattr(cli, "copyfileobj", copy)
    try:
        with pytest.raises(asyncio.CancelledError):
            cli.chopper_filter(source, ToolPaths(), output)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    finally:
        CANCEL_EVENT.reset(token)
