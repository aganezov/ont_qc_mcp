"""Bounded, cancellable line consumption for CLI output on POSIX systems."""

from __future__ import annotations

import logging
import math
import os
import selectors
import subprocess  # nosec B404: intentional CLI process ownership
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .process_control import check_cancelled, cleanup_processes, start_process
from .utils import CommandError, CommandResult, format_cmd

logger = logging.getLogger(__name__)
_READ_BYTES = 64 * 1024
_POLL_SECONDS = 0.2
_STDERR_TRUNCATED = "\n... (stderr truncated)"


def run_line_stream(
    cmd: Sequence[str],
    consume: Callable[[str], None],
    *,
    timeout: int | float,
    env: Mapping[str, str] | None = None,
    max_line_bytes: int = 16 * 1024 * 1024,
    max_stderr_bytes: int = 65536,
) -> CommandResult:
    """Consume UTF-8 stdout lines without retaining command output.

    Lines exclude LF; an unterminated final line is also consumed. A line may
    contain at most ``max_line_bytes`` bytes, excluding LF. Stderr retains only
    its first ``max_stderr_bytes`` bytes, plus a fixed marker if truncated.
    Both pipes are drained concurrently in bounded reads. The deadline includes
    callback processing, checked before and after each callback; callbacks must
    return for cancellation and timeout checks to proceed.
    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    for name, value in (("max_line_bytes", max_line_bytes), ("max_stderr_bytes", max_stderr_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    deadline = time.monotonic() + timeout
    process: subprocess.Popen[Any] | None = None
    pending_line = bytearray()
    stderr_prefix = bytearray()
    stderr_truncated = False

    def result(returncode: int) -> CommandResult:
        stderr = stderr_prefix.decode("utf-8", errors="replace")
        if stderr_truncated:
            stderr += _STDERR_TRUNCATED
        return CommandResult(cmd=cmd, returncode=returncode, stdout="", stderr=stderr)

    def checkpoint() -> None:
        check_cancelled()
        if time.monotonic() >= deadline:
            timed_out = result(124)
            message = f"Command timed out after {timeout}s: {format_cmd(cmd)}"
            if timed_out.stderr:
                message += f"\n{timed_out.stderr}"
            raise CommandError(timed_out, message_override=message) from subprocess.TimeoutExpired(cmd, timeout)

    def emit_line() -> None:
        checkpoint()
        consume(pending_line.decode("utf-8", errors="strict"))
        pending_line.clear()
        checkpoint()

    def accept_stdout(chunk: bytes) -> None:
        offset = 0
        while offset < len(chunk):
            checkpoint()
            newline = chunk.find(b"\n", offset)
            stop = len(chunk) if newline < 0 else newline
            if len(pending_line) + stop - offset > max_line_bytes:
                raise ValueError(f"stdout line exceeds max_line_bytes={max_line_bytes}")
            pending_line.extend(chunk[offset:stop])
            if newline < 0:
                break
            emit_line()
            offset = newline + 1

    try:
        checkpoint()
        process = start_process(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Streaming command requires stdout and stderr pipes")
        with selectors.DefaultSelector() as selector:
            for pipe in (process.stdout, process.stderr):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                checkpoint()
                for key, _ in selector.select(min(_POLL_SECONDS, max(0, deadline - time.monotonic()))):
                    checkpoint()
                    try:
                        chunk = os.read(key.fd, _READ_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if key.fileobj is process.stdout and pending_line:
                            emit_line()
                    elif key.fileobj is process.stdout:
                        accept_stdout(chunk)
                    else:
                        available = max_stderr_bytes - len(stderr_prefix)
                        stderr_prefix.extend(chunk[:available])
                        stderr_truncated |= len(chunk) > available
            checkpoint()
        if process.returncode is None:
            raise RuntimeError("Streaming command did not exit")
        completed = result(process.returncode)
        if completed.returncode != 0:
            raise CommandError(completed)
        return completed
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            cleanup_processes([process])
        except BaseException:
            if not primary_error:
                raise
            logger.exception("Process cleanup failed while preserving the original stream error")
        finally:
            # Close our descriptors even if cleanup itself unexpectedly fails.
            if process is not None:
                for owned_pipe in (process.stdout, process.stderr):
                    if owned_pipe is not None:
                        try:
                            owned_pipe.close()
                        except OSError:
                            if not primary_error:
                                raise
                            logger.exception("Pipe close failed while preserving the original stream error")
