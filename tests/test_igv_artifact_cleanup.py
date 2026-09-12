"""Only server-owned IGV artifacts are removed when a request fails."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ont_qc_mcp import tools


@pytest.fixture
def igv_inputs(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setattr(tools.tempfile, "tempdir", str(private))
    monkeypatch.delenv("MCP_IGV_MOCK", raising=False)
    track = tmp_path / "reads.bam"
    track.write_bytes(b"test alignment")
    args: dict[str, Any] = {
        "genome": "hg38",
        "tracks": [str(track)],
        "regions": [{"chrom": "chr1", "start": 1, "end": 10, "name": "target"}],
    }
    return private, args


def test_validation_failure_cleans_allocated_igv_directories(igv_inputs):
    private, args = igv_inputs
    args["tracks"] = [str(private / "missing.bam")]
    with pytest.raises(FileNotFoundError):
        tools.generate_igv_snapshots(**args)
    assert list(private.iterdir()) == []


def test_second_allocation_failure_cleans_first_directory(igv_inputs, monkeypatch):
    private, args = igv_inputs
    mkdtemp = tools.tempfile.mkdtemp

    def allocate(*, prefix):
        if prefix == "igv_batch_":
            raise OSError("allocation failed")
        return mkdtemp(prefix=prefix)

    monkeypatch.setattr(tools.tempfile, "mkdtemp", allocate)
    with pytest.raises(OSError, match="allocation failed"):
        tools.generate_igv_snapshots(**args)
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("failure", ["cli", "cancel", "result"])
@pytest.mark.parametrize("ownership", ["automatic", "caller-batch", "caller-output", "caller-both"])
def test_failed_igv_preserves_caller_files_only(igv_inputs, tmp_path, monkeypatch, failure, ownership):
    private, args = igv_inputs
    batch = tmp_path / "caller.batch"
    batch.write_text("exit\n")
    output = tmp_path / "caller-output"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_bytes(b"original")
    if ownership in {"caller-batch", "caller-both"}:
        args = {"batch_file": str(batch)}
    if ownership in {"caller-output", "caller-both"}:
        args["output_dir"] = str(output)

    def run(**kwargs):
        partial = kwargs["output_dir"] / "partial.png"
        partial.write_bytes(b"partial")
        if failure == "result":
            return [partial], "invalid-runtime", []
        if failure == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("CLI failed")

    monkeypatch.setattr(tools, "run_igv_snapshot", run)
    error = asyncio.CancelledError if failure == "cancel" else ValueError if failure == "result" else RuntimeError
    with pytest.raises(error):
        tools.generate_igv_snapshots(**args)
    assert list(private.iterdir()) == []
    assert batch.read_text() == "exit\n"
    assert existing.read_bytes() == b"original"
    if ownership in {"caller-output", "caller-both"}:
        # Explicit output directories remain caller-owned, including partial writes.
        assert (output / "partial.png").read_bytes() == b"partial"


def test_cancelled_igv_does_not_allocate(igv_inputs, monkeypatch):
    private, args = igv_inputs

    def cancelled():
        raise asyncio.CancelledError()

    monkeypatch.setattr(tools, "check_cancelled", cancelled, raising=False)
    monkeypatch.setattr(tools, "run_igv_snapshot", lambda **kwargs: ([], "docker", ["igv"]))
    with pytest.raises(asyncio.CancelledError):
        tools.generate_igv_snapshots(**args)
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("mock", [False, True])
@pytest.mark.parametrize("cancel_after_run", [False, True])
def test_igv_keeps_successful_artifacts_unless_cancelled(igv_inputs, monkeypatch, mock, cancel_after_run):
    private, args = igv_inputs
    cancelled = False

    def check():
        if cancelled:
            raise asyncio.CancelledError()

    def snapshots(batch_path, output_root, snapshot_format):
        nonlocal cancelled
        snapshot = output_root / "target.png"
        snapshot.write_bytes(b"snapshot")
        cancelled = cancel_after_run
        return [snapshot]

    def run(**kwargs):
        return snapshots(kwargs["batch_file"], kwargs["output_dir"], kwargs["snapshot_format"]), "docker", ["igv"]

    monkeypatch.setattr(tools, "check_cancelled", check, raising=False)
    monkeypatch.setattr(tools, "run_igv_snapshot", run)
    if mock:
        monkeypatch.setenv("MCP_IGV_MOCK", "1")
        monkeypatch.setattr(tools, "_mock_snapshot_files", snapshots)
    if cancel_after_run:
        with pytest.raises(asyncio.CancelledError):
            tools.generate_igv_snapshots(**args)
        assert list(private.iterdir()) == []
    else:
        result = tools.generate_igv_snapshots(**args)
        assert Path(result.batch_file).is_file()
        assert Path(result.output_directory).is_dir()
        assert len(result.snapshot_files) == 1
        assert Path(result.snapshot_files[0]).read_bytes() == b"snapshot"
        assert result.execution_mode == "docker"
        assert result.command == (["mock_igv_snapshot"] if mock else ["igv"])
