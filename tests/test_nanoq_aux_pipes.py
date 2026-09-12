"""Nanoq auxiliary statistics use bounded pipes rather than per-read disk files."""

import json
import asyncio
import os
import stat
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp import nanoq_aux_pipes as transport
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.process_control import CANCEL_EVENT
from ont_qc_mcp.utils import CommandError, CommandResult


def test_fastq_uses_fifos_and_retains_histograms(tmp_path, monkeypatch):
    paths = []

    def run(cmd, **kwargs):
        for flag, content in [("--read-lengths", "1\n2\n4\n"), ("--read-qualities", "0.0\n1.9\n2.0\n")]:
            path = Path(cmd[cmd.index(flag) + 1])
            paths.append(path)
            assert stat.S_ISFIFO(path.stat().st_mode)
            path.write_text(content)
        return CommandResult(cmd, 0, json.dumps({"reads": {"count": 3}}), "")

    monkeypatch.setattr(cli, "run_command_with_retry", run)
    result = cli.nanoq_stats(tmp_path / "reads.fastq", ToolPaths(), exec_cfg=ExecutionConfig(nanoq_length_bin_width=2))
    assert result.length_histogram is not None and result.qscore_histogram is not None
    assert result.length_percentiles is not None
    assert [row.count for row in result.length_histogram] == [1, 1, 1]
    assert result.length_percentiles.p50 == 2
    assert [row.count for row in result.qscore_histogram] == [1, 1, 1]
    assert all(not path.exists() for path in paths)


@pytest.fixture
def fifo_probe(tmp_path, monkeypatch):
    directory = tmp_path / "transport"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    yield directory
    assert list(directory.iterdir()) == []
    assert not any(thread.name.startswith("nanoq-aux-") for thread in threading.enumerate())


@pytest.mark.parametrize("payload", [b"", b"\ninvalid\n", b"0\n2\n8", b"1\n2\n3\n4\n5\n"])
@pytest.mark.parametrize("limit", [4, 5])
def test_empty_tail_bins_and_percentile_boundary(fifo_probe, payload, limit):
    with transport.NanoqAuxPipes(
        ExecutionConfig(nanoq_length_bin_width=2, nanoq_percentiles_exact_max_reads=limit)
    ) as aux:
        for reader in aux.readers:
            reader.path.write_bytes(payload)
        stats = cli.parse_nanoq_json({"reads": {"count": 0}})
        aux.augment(stats)
        if not payload:
            assert stats.length_histogram is None and stats.qscore_histogram is None
        elif payload == b"\ninvalid\n":
            assert stats.length_histogram == [] and stats.qscore_histogram == []
        elif payload == b"0\n2\n8":
            assert stats.length_histogram is not None and stats.length_percentiles is not None
            assert [row.count for row in stats.length_histogram] == [1, 1, 0, 0, 1]
            assert stats.length_percentiles.p50 == 2
            assert stats.length_percentiles.p25 == 1
        elif limit == 4:
            assert stats.length_percentiles is None
            assert aux.readers[0].accumulator.values is None
        else:
            assert stats.length_percentiles is not None
            assert stats.length_percentiles.p50 == 3
            assert stats.length_percentiles.p1 == pytest.approx(1.04)


def test_json_fields_take_precedence(fifo_probe):
    stats = cli.parse_nanoq_json(
        {"reads": {"length": {"hist": [], "percentiles": {"p50": 17}}, "qscore": {"hist": []}}}
    )
    expected = stats.model_dump()
    with transport.NanoqAuxPipes(ExecutionConfig()) as aux:
        for reader in aux.readers:
            reader.path.write_text("1\n2\n3\n")
        aux.augment(stats)
    assert stats.model_dump() == expected


def test_retry_uses_fresh_pipes_and_accumulators(fifo_probe, tmp_path, monkeypatch):
    attempts = []

    def run(cmd, **kwargs):
        assert kwargs["max_attempts"] == 1
        paths = [Path(cmd[cmd.index(flag) + 1]) for flag in ("--read-lengths", "--read-qualities")]
        attempts.append(paths)
        for path in paths:
            path.write_text("999\n" if len(attempts) == 1 else "1\n2\n")
        if len(attempts) == 1:
            raise CommandError(CommandResult(cmd, 1, "", "partial output"))
        assert all(not path.exists() for path in attempts[0])
        return CommandResult(cmd, 0, '{"reads": {"count": 2}}', "")

    monkeypatch.setattr(cli, "run_command_with_retry", run)
    monkeypatch.setattr(cli, "wait_or_cancel", lambda _: None)
    stats = cli.nanoq_stats(tmp_path / "reads.fastq", ToolPaths(), exec_cfg=ExecutionConfig(nanoq_length_bin_width=1))
    assert attempts[0] != attempts[1]
    assert stats.length_histogram is not None and stats.qscore_histogram is not None
    assert stats.length_percentiles is not None
    assert [row.count for row in stats.length_histogram] == [0, 1, 1]
    assert stats.length_percentiles.p50 == 1.5
    assert [row.count for row in stats.qscore_histogram] == [0, 1, 1]


@pytest.mark.parametrize("failure", ["mkfifo", "open", "thread"])
@pytest.mark.parametrize("bam", [False, True])
def test_second_reader_startup_failure_cleans_every_resource(fifo_probe, monkeypatch, tmp_path, failure, bam):
    fds = []
    original_open, original_mkfifo, original_start = os.open, os.mkfifo, threading.Thread.start

    def open_fifo(path, flags, *args, **kwargs):
        is_fifo = Path(path).name in {"lengths", "qualities"}
        if failure == "open" and Path(path).name == "qualities":
            raise OSError("second auxiliary startup failed")
        fd = original_open(path, flags, *args, **kwargs)
        if is_fifo:
            fds.append(fd)
        return fd

    def mkfifo(path, mode):
        if failure == "mkfifo" and path.name == "qualities":
            raise OSError("second auxiliary startup failed")
        original_mkfifo(path, mode)

    def start(thread):
        if failure == "thread" and thread.name == "nanoq-aux-qualities":
            raise OSError("second auxiliary startup failed")
        original_start(thread)

    monkeypatch.setattr(os, "open", open_fifo)
    monkeypatch.setattr(os, "mkfifo", mkfifo)
    monkeypatch.setattr(threading.Thread, "start", start)
    call = cli.nanoq_from_bam_streaming if bam else cli.nanoq_stats
    with pytest.raises(OSError, match="second auxiliary startup failed"):
        call(tmp_path / "unused", ToolPaths())
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_unsupported_platform_fails_before_launch(fifo_probe, monkeypatch, tmp_path):
    monkeypatch.delattr(os, "mkfifo")
    with pytest.raises(RuntimeError, match="require POSIX named pipes"):
        cli.nanoq_stats(tmp_path / "unused", ToolPaths())


def test_producer_exit_racing_an_empty_read_does_not_lose_tail(fifo_probe, monkeypatch):
    original_read = os.read
    empty_read, release = threading.Event(), threading.Event()
    lock = threading.Lock()

    def read(fd, size):
        chunk = original_read(fd, size)
        with lock:
            pause = not chunk and not empty_read.is_set()
            if pause:
                empty_read.set()
        if pause:
            assert release.wait(2)
        return chunk

    monkeypatch.setattr(os, "read", read)
    try:
        with transport.NanoqAuxPipes(ExecutionConfig()) as aux:
            assert empty_read.wait(2)
            for reader in aux.readers:
                reader.path.write_text("1\n2\n3\n")
            aux.done.set()
            release.set()
            stats = cli.parse_nanoq_json({"reads": {"count": 3}})
            aux.augment(stats)
            assert stats.length_histogram is not None and stats.qscore_histogram is not None
            assert sum(row.count for row in stats.length_histogram) == 3
            assert sum(row.count for row in stats.qscore_histogram) == 3
    finally:
        release.set()


def make_tools(tmp_path, body):
    nanoq = tmp_path / "nanoq"
    nanoq.write_text(f"#!{sys.executable}\nimport sys, os, time\nfrom pathlib import Path\n" + body + "\n")
    nanoq.chmod(0o755)
    samtools = tmp_path / "samtools"
    samtools.write_text(f"#!{sys.executable}\nprint('@r\\nACGT\\n+\\nIIII')\n")
    samtools.chmod(0o755)
    return ToolPaths(nanoq=str(nanoq), samtools=str(samtools))


@pytest.mark.parametrize("bam", [False, True])
@pytest.mark.parametrize("bad_line", [False, True])
def test_producer_exceeding_pipe_capacity_finishes_and_propagates_reader_errors(fifo_probe, tmp_path, bam, bad_line):
    prefix = "x" * 4097 + "\n" if bad_line else ""
    body = (
        'print(\'{"reads": {"count": 250000}}\', flush=True)\n'
        "for flag in ('--read-lengths', '--read-qualities'):\n"
        "    p = Path(sys.argv[sys.argv.index(flag) + 1])\n"
        f"    p.write_text({prefix!r} + '2\\n' * 250000)\n"
        "    assert p.stat().st_size == 0\n"
    )
    paths = make_tools(tmp_path, body)
    cfg = ExecutionConfig(per_tool_timeouts={"nanoq": 3, "samtools": 3}, nanoq_length_bin_width=1)
    call = cli.nanoq_from_bam_streaming if bam else cli.nanoq_stats
    if bad_line:
        with pytest.raises(RuntimeError, match="auxiliary.*exceeds 4096"):
            call(tmp_path / "unused", paths, exec_cfg=cfg)
    else:
        stats = call(tmp_path / "unused", paths, exec_cfg=cfg)
        assert stats.length_histogram is not None and stats.qscore_histogram is not None
        assert [row.count for row in stats.length_histogram] == [0, 0, 250000]
        assert [row.count for row in stats.qscore_histogram] == [0, 0, 250000]
        assert stats.length_percentiles is None


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), asyncio.CancelledError(), ValueError("bad JSON")])
def test_failures_and_cancellation_remove_pipes(fifo_probe, monkeypatch, tmp_path, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "run_command_with_retry", run)
    with pytest.raises(type(error)):
        cli.nanoq_stats(tmp_path / "unused", ToolPaths())


@pytest.mark.parametrize("bam", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_partial_writer_timeout_or_cancellation_joins_readers(fifo_probe, tmp_path, bam, cancel):
    marker = tmp_path / "partial-written"
    paths = make_tools(
        tmp_path,
        "p = Path(sys.argv[sys.argv.index('--read-lengths') + 1])\n"
        "with p.open('w') as stream:\n"
        "    stream.write('1\\n' * 1000)\n"
        "    stream.flush()\n"
        f"    Path({str(marker)!r}).touch()\n"
        "    time.sleep(30)\n",
    )
    cfg = ExecutionConfig(per_tool_timeouts={"nanoq": 1, "samtools": 1})
    call = cli.nanoq_from_bam_streaming if bam else cli.nanoq_stats
    event = threading.Event()
    token = CANCEL_EVENT.set(event)
    stop = threading.Event()

    def cancel_after_write():
        while not stop.wait(0.01):
            if marker.exists():
                event.set()
                break

    thread = threading.Thread(target=cancel_after_write)
    if cancel:
        thread.start()
    try:
        expected = asyncio.CancelledError if cancel else RuntimeError if bam else CommandError
        with pytest.raises(expected):
            call(tmp_path / "unused", paths, exec_cfg=cfg)
        assert marker.exists()
    finally:
        stop.set()
        if cancel:
            thread.join()
        CANCEL_EVENT.reset(token)


@pytest.mark.parametrize("bam", [False, True])
def test_reader_io_error_is_reported_after_discard_draining(fifo_probe, monkeypatch, tmp_path, bam):
    original_read = os.read
    failed = threading.Event()

    def read(fd, size):
        if threading.current_thread().name == "nanoq-aux-lengths" and not failed.is_set():
            failed.set()
            raise OSError("injected FIFO read failure")
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", read)
    paths = make_tools(
        tmp_path,
        'print(\'{"reads": {"count": 100000}}\', flush=True)\n'
        "for flag in ('--read-lengths', '--read-qualities'):\n"
        "    Path(sys.argv[sys.argv.index(flag) + 1]).write_text('2\\n' * 100000)\n",
    )
    call = cli.nanoq_from_bam_streaming if bam else cli.nanoq_stats
    with pytest.raises(RuntimeError, match="auxiliary lengths.*injected FIFO read failure"):
        call(tmp_path / "unused", paths, exec_cfg=ExecutionConfig(per_tool_timeouts={"nanoq": 3, "samtools": 3}))
