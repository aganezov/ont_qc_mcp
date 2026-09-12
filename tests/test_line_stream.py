"""Exercise bounded streaming and failure cleanup with real child processes."""

import asyncio
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ont_qc_mcp import line_stream
from ont_qc_mcp.process_control import CANCEL_EVENT, cleanup_processes
from ont_qc_mcp.utils import CommandError


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[subprocess.Popen[Any]]]:
    captured: list[subprocess.Popen[Any]] = []
    original = line_stream.start_process

    def start(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = original(*args, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(line_stream, "start_process", start)
    try:
        yield captured
        for process in captured:
            assert process.poll() is not None
            assert process.stdout is not None and process.stdout.closed
            assert process.stderr is not None and process.stderr.closed
    finally:
        cleanup_processes(captured)


def command(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_floods_both_pipes_without_retaining_stdout(processes: list[subprocess.Popen[Any]]) -> None:
    code = "import os; [(os.write(1, b'row\\n' * 20000), os.write(2, b'e' * 80000)) for _ in range(8)]"
    count = 0

    def consume(line: str) -> None:
        nonlocal count
        assert line == "row"
        count += 1

    result = line_stream.run_line_stream(command(code), consume, timeout=5, max_stderr_bytes=101)
    assert count == 160000
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr.startswith("e" * 101)
    assert "truncated" in result.stderr
    assert len(result.stderr) < 160


def test_utf8_partial_final_line_environment_and_stdin_eof(processes: list[subprocess.Popen[Any]]) -> None:
    code = (
        "import os,sys; assert sys.stdin.read() == ''; "
        "os.write(1, ('héllo\\n\\n' + os.environ['STREAM_TEST']).encode())"
    )
    rows: list[str] = []
    result = line_stream.run_line_stream(command(code), rows.append, timeout=5, env={"STREAM_TEST": "last"})
    assert rows == ["héllo", "", "last"]
    assert result.stderr == ""


@pytest.mark.parametrize("ending", ["b'\\n'", "b''"])
def test_overlong_line_rejected_before_callback(processes: list[subprocess.Popen[Any]], ending: str) -> None:
    code = f"import os,time; os.write(1, b'x' * 65536 + {ending}); time.sleep(30)"
    rows: list[str] = []
    with pytest.raises(ValueError, match="max_line_bytes"):
        line_stream.run_line_stream(command(code), rows.append, timeout=5, max_line_bytes=100)
    assert rows == []


def test_exact_line_limit_and_split_utf8(processes: list[subprocess.Popen[Any]]) -> None:
    code = "import os,time; os.write(1, b'a' * 65535 + b'\\xc3'); time.sleep(.05); os.write(1, b'\\xa9\\n')"
    rows: list[str] = []
    line_stream.run_line_stream(command(code), rows.append, timeout=5, max_line_bytes=65537)
    assert rows == ["a" * 65535 + "é"]


def test_invalid_utf8_fails_without_replacement(processes: list[subprocess.Popen[Any]]) -> None:
    with pytest.raises(UnicodeDecodeError):
        line_stream.run_line_stream(command("import os; os.write(1, b'\\xff\\n')"), lambda _: None, timeout=5)


def test_parser_error_is_preserved_even_if_cleanup_raises(
    monkeypatch: pytest.MonkeyPatch, processes: list[subprocess.Popen[Any]]
) -> None:
    original_error = ValueError("parser failed")

    def consume(_: str) -> None:
        raise original_error

    def bad_cleanup(owned: Any) -> None:
        cleanup_processes(owned)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(line_stream, "cleanup_processes", bad_cleanup)
    with pytest.raises(ValueError) as caught:
        line_stream.run_line_stream(
            command("import time; print('row', flush=True); time.sleep(30)"), consume, timeout=5
        )
    assert caught.value is original_error


def test_nonzero_exit_retains_bounded_diagnostics(processes: list[subprocess.Popen[Any]]) -> None:
    with pytest.raises(CommandError) as caught:
        line_stream.run_line_stream(
            command("import os,sys; os.write(2, b'e' * 100000); sys.exit(7)"),
            lambda _: None,
            timeout=5,
            max_stderr_bytes=20,
        )
    assert caught.value.result.returncode == 7
    assert caught.value.result.stdout == ""
    assert caught.value.result.stderr.startswith("e" * 20)
    assert "truncated" in caught.value.result.stderr


def test_timeout_retains_diagnostics(processes: list[subprocess.Popen[Any]]) -> None:
    code = "import os,time; os.write(2, b'diagnostic'); time.sleep(30)"
    before = time.monotonic()
    with pytest.raises(CommandError, match="timed out") as caught:
        line_stream.run_line_stream(command(code), lambda _: None, timeout=1)
    assert caught.value.result.returncode == 124
    assert caught.value.result.stderr == "diagnostic"
    assert time.monotonic() - before < 4


def test_timeout_includes_parser_processing(processes: list[subprocess.Popen[Any]]) -> None:
    rows: list[str] = []

    def consume(line: str) -> None:
        rows.append(line)
        time.sleep(1.1)

    with pytest.raises(CommandError) as caught:
        line_stream.run_line_stream(command("print('first'); print('second')"), consume, timeout=1)
    assert caught.value.result.returncode == 124
    assert rows == ["first"]


@pytest.mark.parametrize("code", ["import time; time.sleep(30)", "import os\nwhile True: os.write(1, b'x\\n' * 20000)"])
def test_cancellation_when_idle_or_flooding(processes: list[subprocess.Popen[Any]], code: str) -> None:
    event = threading.Event()
    token = CANCEL_EVENT.set(event)
    timer = threading.Timer(0.1, event.set)
    timer.start()
    before = time.monotonic()
    try:
        with pytest.raises(asyncio.CancelledError):
            line_stream.run_line_stream(command(code), lambda _: None, timeout=10)
        assert time.monotonic() - before < 3
    finally:
        timer.cancel()
        timer.join()
        CANCEL_EVENT.reset(token)


def test_precancelled_does_not_start(processes: list[subprocess.Popen[Any]]) -> None:
    event = threading.Event()
    event.set()
    token = CANCEL_EVENT.set(event)
    try:
        with pytest.raises(asyncio.CancelledError):
            line_stream.run_line_stream(command("raise AssertionError('must not start')"), lambda _: None, timeout=5)
    finally:
        CANCEL_EVENT.reset(token)
    assert processes == []


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
def test_exited_leader_descendant_holding_pipes_times_out(
    processes: list[subprocess.Popen[Any]], tmp_path: Path
) -> None:
    marker = tmp_path / "survived"
    child_code = f"import pathlib,time; time.sleep(2); pathlib.Path({str(marker)!r}).touch(); time.sleep(30)"
    code = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child_code!r}])"
    before = time.monotonic()
    with pytest.raises(CommandError) as caught:
        line_stream.run_line_stream(command(code), lambda _: None, timeout=1)
    assert caught.value.result.returncode == 124
    assert processes[0].returncode == 0
    assert time.monotonic() - before < 3
    time.sleep(max(0, 2.3 - (time.monotonic() - before)))
    assert not marker.exists()


def test_startup_failure_preserves_os_error(processes: list[subprocess.Popen[Any]], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        line_stream.run_line_stream([str(tmp_path / "does-not-exist")], lambda _: None, timeout=5)
    assert processes == []


def test_closed_pipes_with_live_process_still_times_out(processes: list[subprocess.Popen[Any]]) -> None:
    code = "import os,time; os.close(1); os.close(2); time.sleep(30)"
    with pytest.raises(CommandError) as caught:
        line_stream.run_line_stream(command(code), lambda _: None, timeout=0.15)
    assert caught.value.result.returncode == 124


@pytest.mark.parametrize("failure", ["timeout", "cancel", "startup", "nonzero"])
def test_cleanup_error_preserves_other_primary_failures(
    processes: list[subprocess.Popen[Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    def bad_cleanup(owned: Any) -> None:
        cleanup_processes(owned)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(line_stream, "cleanup_processes", bad_cleanup)
    event = threading.Event()
    token = CANCEL_EVENT.set(event)
    timer = threading.Timer(0.05, event.set)
    try:
        if failure == "startup":
            with pytest.raises(FileNotFoundError):
                line_stream.run_line_stream([str(tmp_path / "absent")], lambda _: None, timeout=5)
        elif failure == "cancel":
            timer.start()
            with pytest.raises(asyncio.CancelledError):
                line_stream.run_line_stream(command("import time; time.sleep(30)"), lambda _: None, timeout=5)
        else:
            code = "import time; time.sleep(30)" if failure == "timeout" else "raise SystemExit(7)"
            with pytest.raises(CommandError) as caught:
                line_stream.run_line_stream(command(code), lambda _: None, timeout=0.15)
            assert caught.value.result.returncode == (124 if failure == "timeout" else 7)
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        CANCEL_EVENT.reset(token)


def test_cleanup_error_after_success_is_not_hidden(
    processes: list[subprocess.Popen[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_cleanup(owned: Any) -> None:
        cleanup_processes(owned)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(line_stream, "cleanup_processes", bad_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        line_stream.run_line_stream(command("pass"), lambda _: None, timeout=5)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_rejected_before_start(processes: list[subprocess.Popen[Any]], timeout: int | float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        line_stream.run_line_stream(command("pass"), lambda _: None, timeout=timeout)
    assert processes == []


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limits_rejected_before_start(processes: list[subprocess.Popen[Any]], limit: int) -> None:
    with pytest.raises(ValueError, match="max_line_bytes"):
        line_stream.run_line_stream(command("pass"), lambda _: None, timeout=5, max_line_bytes=limit)
    with pytest.raises(ValueError, match="max_stderr_bytes"):
        line_stream.run_line_stream(command("pass"), lambda _: None, timeout=5, max_stderr_bytes=limit)
    assert processes == []
