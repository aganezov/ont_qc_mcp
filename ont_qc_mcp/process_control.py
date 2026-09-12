"""Cooperative cancellation and cleanup for subprocesses owned by a worker."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess  # nosec B404: intentional CLI process ownership
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)
CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar("command_cancel_event", default=None)
_POLL_SECONDS = 0.2
_TERM_GRACE_SECONDS = 0.5


def check_cancelled() -> None:
    event = CANCEL_EVENT.get()
    if event is not None and event.is_set():
        raise asyncio.CancelledError("Blocking operation cancelled")


def wait_or_cancel(seconds: float) -> None:
    check_cancelled()
    event = CANCEL_EVENT.get()
    if event is None:
        time.sleep(seconds)
    elif event.wait(seconds):
        check_cancelled()


@contextmanager
def cancellation_disabled() -> Iterator[None]:
    """Allow bounded cleanup without clearing the original cancellation event."""
    token = CANCEL_EVENT.set(None)
    try:
        yield
    finally:
        CANCEL_EVENT.reset(token)


def start_process(cmd: Sequence[str], **kwargs: Any) -> subprocess.Popen[Any]:
    check_cancelled()
    if os.name == "posix":
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)  # nosec B603: argument vector, no shell


def communicate_process(process: subprocess.Popen[Any], timeout: float | None) -> tuple[Any, Any]:
    """Drain both pipes with cancellation checkpoints and one overall timeout."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        check_cancelled()
        interval = _POLL_SECONDS if deadline is None else min(_POLL_SECONDS, max(0, deadline - time.monotonic()))
        try:
            result = process.communicate(timeout=interval)
        except subprocess.TimeoutExpired as error:
            if timeout is not None and deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    process.args, timeout, output=error.output, stderr=error.stderr
                ) from error
        else:
            check_cancelled()
            return result


def wait_process(process: subprocess.Popen[Any], timeout: float | None) -> int:
    """Wait without reading pipes; callers must drain any piped output."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        check_cancelled()
        interval = _POLL_SECONDS if deadline is None else min(_POLL_SECONDS, max(0, deadline - time.monotonic()))
        try:
            result = process.wait(timeout=interval)
        except subprocess.TimeoutExpired as error:
            if timeout is not None and deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(process.args, timeout) from error
        else:
            check_cancelled()
            return result


def cleanup_processes(processes: Sequence[subprocess.Popen[Any] | None]) -> None:
    """Stop owned groups, reap their leaders, and close pipes without masking an error.

    Only pass processes created by start_process(). A group may still contain
    descendants after its leader has exited. Descendants receive TERM then KILL;
    only direct children can be reaped here. Callers must first stop pipe readers.
    """
    owned = [process for process in processes if process is not None]
    failures: list[str] = []

    def send(process: subprocess.Popen[Any], sig: int) -> bool:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            return True
        except ProcessLookupError:
            return False
        except OSError as error:
            failures.append(f"signal {sig} to process group {process.pid}: {error}")
            return True  # A failed TERM must still reach the KILL fallback.

    pending = [process for process in owned if send(process, signal.SIGTERM)]
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while pending and time.monotonic() < deadline:
        remaining = []
        for process in pending:
            process.poll()  # Reap a finished direct child before checking its group.
            try:
                if os.name == "posix":
                    os.killpg(process.pid, 0)
                elif process.returncode is not None:
                    continue
            except ProcessLookupError:
                continue
            except OSError as error:
                failures.append(f"checking process group {process.pid}: {error}")
            remaining.append(process)
        pending = remaining
        if pending:
            time.sleep(0.02)
    for process in pending:
        send(process, signal.SIGKILL)
    for process in owned:
        try:
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"reaping process {process.pid}: {error}")
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError as error:
                    failures.append(f"closing pipe for process {process.pid}: {error}")
    if failures:
        message = "Process cleanup failed: " + "; ".join(failures)
        logger.error(message)
        if sys.exc_info()[0] is None:
            raise RuntimeError(message)
