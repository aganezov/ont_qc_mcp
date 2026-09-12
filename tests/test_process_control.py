"""Real subprocess checks for cooperative cancellation and owned-group cleanup."""

import asyncio
import contextvars
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from ont_qc_mcp import threadpool, utils


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setenv("MCP_BLOCKING_MODE", "executor")
        monkeypatch.setattr(threadpool, "get_executor", lambda: pool)
        yield


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[subprocess.Popen[Any]]]:
    original = subprocess.Popen
    captured: list[subprocess.Popen[Any]] = []

    def start(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = original(*args, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    try:
        yield captured
    finally:
        with ExitStack() as cleanup:
            for process in captured:
                cleanup.callback(process.wait, timeout=3)
                if process.poll() is None:
                    cleanup.callback(process.kill)


async def wait_for_path(path: Path) -> None:
    async def poll() -> None:
        while not path.exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), 5)


def wait_for_path_sync(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"Child did not create {path}")
        time.sleep(0.01)


@pytest.mark.asyncio
async def test_running_command_cancellation_stops_child(
    executor: None, processes: list[subprocess.Popen[Any]], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ready = tmp_path / "ready"
    code = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    task = asyncio.create_task(utils.run_command_async([sys.executable, "-c", code, str(ready)], timeout=60))
    try:
        await wait_for_path(ready)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=2)
        assert task in done, "Cancellation did not stop the running command"
        assert task.cancelled()
        assert processes[0].poll() is not None
        assert "Worker failed after request cancellation" not in caplog.text
    finally:
        try:
            for process in processes:
                if process.poll() is None:
                    process.kill()
        finally:
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_retry_backoff_is_interrupted(executor: None, tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open('a').write('attempt\\n'); sys.exit(2)"
    task = asyncio.create_task(
        threadpool.run_sync(
            utils.run_command_with_retry,
            [sys.executable, "-c", code, str(attempts)],
            max_attempts=2,
            backoff_seconds=1.5,
        )
    )
    try:
        await wait_for_path(attempts)
        await asyncio.sleep(0.1)
        task.cancel()
        done, _ = await asyncio.wait([task], timeout=0.7)
        assert task in done, "Cancellation waited for retry backoff"
        assert task.cancelled()
        assert attempts.read_text().splitlines() == ["attempt"]
    finally:
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_executor_copies_context_and_isolates_request_events(executor: None) -> None:
    import ont_qc_mcp.process_control as control

    marker = contextvars.ContextVar("process_control_test_marker", default="missing")
    token = marker.set("request-value")
    try:
        first = await threadpool.run_sync(lambda: (marker.get(), control.CANCEL_EVENT.get()))
        second = await threadpool.run_sync(control.CANCEL_EVENT.get)
        assert first[0] == "request-value"
        assert isinstance(first[1], threading.Event)
        assert first[1] is not second
        assert control.CANCEL_EVENT.get() is None
    finally:
        marker.reset(token)


def test_cancellation_disabled_restores_the_original_event() -> None:
    import ont_qc_mcp.process_control as control

    event = threading.Event()
    event.set()
    token = control.CANCEL_EVENT.set(event)
    try:
        with control.cancellation_disabled():
            assert control.CANCEL_EVENT.get() is None
            control.check_cancelled()
            control.wait_or_cancel(0)
        assert control.CANCEL_EVENT.get() is event and event.is_set()
        with pytest.raises(asyncio.CancelledError):
            control.check_cancelled()
        with pytest.raises(asyncio.CancelledError):
            control.wait_or_cancel(10)
    finally:
        control.CANCEL_EVENT.reset(token)


def test_precancelled_commands_do_not_start_or_truncate_output(tmp_path: Path) -> None:
    import ont_qc_mcp.process_control as control

    output = tmp_path / "existing-output"
    output.write_text("keep")
    event = threading.Event()
    event.set()
    token = control.CANCEL_EVENT.set(event)
    try:
        with pytest.raises(asyncio.CancelledError):
            utils.run_command([sys.executable, "-c", "raise AssertionError('must not run')"], stdout_path=output)
        assert output.read_text() == "keep"
    finally:
        control.CANCEL_EVENT.reset(token)


def test_command_stdio_file_streaming_encoding_and_bounds(tmp_path: Path) -> None:
    source = tmp_path / "stdin"
    source.write_text("héllo\n", encoding="utf-8")
    code = "import sys; print(sys.stdin.read().strip() + ' café'); print('diagnostic', file=sys.stderr)"
    with source.open(encoding="utf-8") as stdin:
        result = utils.run_command([sys.executable, "-c", code], stdin=stdin)
    assert result.stdout == "héllo café\n"
    assert result.stderr == "diagnostic\n"
    output = tmp_path / "stdout"
    with source.open(encoding="utf-8") as stdin:
        streamed = utils.run_command([sys.executable, "-c", code], stdin=stdin, stdout_path=output, max_stderr_chars=4)
    assert output.read_text(encoding="utf-8") == result.stdout
    assert streamed.stdout == f"<streamed to {output}>"
    assert streamed.stderr == "diag... (truncated 7 chars)"
    bounded = utils.run_command([sys.executable, "-c", "print('abcdef')"], max_stdout_chars=3)
    assert bounded.stdout == "abc... (truncated 4 chars)"


def test_timeout_retains_partial_output_and_returncode(processes: list[subprocess.Popen[Any]]) -> None:
    code = (
        "import sys,time; print('partial-out', flush=True); "
        "print('partial-error',file=sys.stderr,flush=True); time.sleep(30)"
    )
    with pytest.raises(utils.CommandError) as caught:
        utils.run_command([sys.executable, "-c", code], timeout=1)
    assert caught.value.result.returncode == 124
    assert "partial-out" in caught.value.result.stdout
    assert "partial-error" in caught.value.result.stderr
    assert "after 1s" in str(caught.value)
    assert processes[0].poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="Owned process groups require POSIX")
@pytest.mark.parametrize("leader_exits", [False, True], ids=["running-leader", "exited-leader-pipe-holder"])
def test_cleanup_kills_resistant_descendants_only_in_owned_group(tmp_path: Path, leader_exits: bool) -> None:
    import ont_qc_mcp.process_control as control

    child_ready = tmp_path / "child-ready"
    leader_ready = tmp_path / "leader-ready"
    child_code = (
        "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader_code = (
        "import pathlib,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
        "p=pathlib.Path(sys.argv[2]); "
        "\nwhile not p.exists(): time.sleep(.01)\n"
        "pathlib.Path(sys.argv[3]).write_text('ready'); " + ("sys.exit(0)" if leader_exits else "time.sleep(30)")
    )
    independent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    process = control.start_process(
        [sys.executable, "-c", leader_code, child_code, str(child_ready), str(leader_ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    try:
        wait_for_path_sync(leader_ready)
        child_pid = int(child_ready.read_text())
        assert os.getpgid(child_pid) == process.pid
        if leader_exits:
            process.wait(timeout=3)
            with pytest.raises(subprocess.TimeoutExpired):
                control.communicate_process(process, timeout=0.1)
        started = time.monotonic()
        control.cleanup_processes([None, process])
        assert time.monotonic() - started < 3
        assert process.poll() is not None
        assert independent.poll() is None
        assert process.stdout is not None and process.stdout.closed
        assert process.stderr is not None and process.stderr.closed
        deadline = time.monotonic() + 2
        while True:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            if sys.platform.startswith("linux"):
                try:
                    state = Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if state == "Z":  # Only an orphan's new parent can reap it.
                    break
            assert time.monotonic() < deadline, f"Descendant {child_pid} is still running"
            time.sleep(0.01)
    finally:
        try:
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            try:
                control.cleanup_processes([process])
            finally:
                try:
                    independent.kill()
                finally:
                    independent.wait(timeout=3)


def test_wait_process_observes_cancellation(tmp_path: Path) -> None:
    import ont_qc_mcp.process_control as control

    ready = tmp_path / "ready"
    process = control.start_process(
        [sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)", str(ready)]
    )
    event = threading.Event()
    token = control.CANCEL_EVENT.set(event)
    try:
        wait_for_path_sync(ready)
        timer = threading.Timer(0.05, event.set)
        timer.start()
        with pytest.raises(asyncio.CancelledError):
            control.wait_process(process, timeout=10)
        timer.join()
    finally:
        try:
            control.cleanup_processes([process])
        finally:
            control.CANCEL_EVENT.reset(token)


@pytest.mark.skipif(os.name != "posix", reason="Owned process groups require POSIX")
def test_failed_term_still_attempts_kill_and_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import ont_qc_mcp.process_control as control

    ready = tmp_path / "ready"
    process = control.start_process(
        [sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(30)", str(ready)]
    )
    original_killpg = os.killpg
    signals: list[int] = []

    def fail_term(pgid: int, sig: int) -> None:
        signals.append(sig)
        if sig == signal.SIGTERM:
            raise OSError("injected TERM failure")
        original_killpg(pgid, sig)

    try:
        wait_for_path_sync(ready)
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", fail_term)
            with pytest.raises(ValueError, match="primary command error"):
                try:
                    raise ValueError("primary command error")
                finally:
                    control.cleanup_processes([process])
        assert signal.SIGKILL in signals
        assert process.poll() is not None
        assert "injected TERM failure" in caplog.text
    finally:
        try:
            try:
                original_killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            process.wait(timeout=3)


def test_sustained_output_polling_sanity() -> None:
    import ont_qc_mcp.process_control as control

    # Compare the same paced 64 MiB producer with one communicate() call and
    # cancellation polling. This catches gross repeated-buffer-copy overhead.
    code = "import os,time; data=b'x'*262144\nfor _ in range(256): os.write(1,data); time.sleep(.004)"
    command = [sys.executable, "-c", code]
    started = time.monotonic()
    baseline = subprocess.run(command, capture_output=True, timeout=10)
    baseline_elapsed = time.monotonic() - started
    process = control.start_process(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        started = time.monotonic()
        output, error = control.communicate_process(process, timeout=10)
        polling_elapsed = time.monotonic() - started
        assert output == baseline.stdout == b"x" * (64 * 1024 * 1024)
        assert error == b""
        assert polling_elapsed < baseline_elapsed * 3 + 0.5
        print(f"64 MiB sustained output: baseline={baseline_elapsed:.3f}s polling={polling_elapsed:.3f}s")
    finally:
        control.cleanup_processes([process])
