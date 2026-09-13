"""One-deadline, fully drained subprocess pipelines for API v2 adapters."""

from __future__ import annotations

import math
import os
import selectors
import subprocess  # nosec B404: intentional CLI process ownership
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .process_control import check_cancelled, cleanup_processes, start_process
from .utils import CommandResult, format_cmd


_POLL_SECONDS = 0.2
_READ_BYTES = 64 * 1024
_TRUNCATED = b"\n... (output truncated)"


class RequestDeadline:
    """A monotonic deadline shared by every stage of one request."""

    def __init__(self, seconds: int | float) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("request deadline must be finite and positive")
        if seconds <= 0:
            raise ValueError("request deadline must be finite and positive")
        self.seconds = float(seconds)
        self.started_at = time.monotonic()
        self.expires_at = self.started_at + self.seconds

    def remaining(self) -> float:
        check_cancelled()
        value = self.expires_at - time.monotonic()
        if value <= 0:
            raise TimeoutError(f"request deadline exceeded after {self.seconds:g}s")
        return value

    def checkpoint(self) -> None:
        self.remaining()


@dataclass(frozen=True)
class PipelineStage:
    name: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline stage name must not be empty")
        if not self.command or any(not isinstance(token, str) or not token for token in self.command):
            raise ValueError("pipeline stage command must contain nonempty strings")


class PipelineStageError(RuntimeError):
    """A nonzero stage result, checked from upstream to downstream."""

    def __init__(self, stage: str, result: CommandResult) -> None:
        detail = result.stderr or f"exit {result.returncode}"
        super().__init__(f"Pipeline stage {stage!r} failed: {format_cmd(result.cmd)}\n{detail}")
        self.stage = stage
        self.result = result


class PipelineOutputLimitError(RuntimeError):
    """A successful stage produced output that could not be returned intact."""

    def __init__(self, stage: str, stream: str, limit: int) -> None:
        super().__init__(f"Pipeline stage {stage!r} {stream} exceeded output limit of {limit} bytes")
        self.stage = stage
        self.stream = stream
        self.limit = limit


@dataclass(frozen=True)
class PipelineResult:
    stages: tuple[CommandResult, ...]

    @property
    def final(self) -> CommandResult:
        return self.stages[-1]


class _BoundedBytes:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.value = bytearray()
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        available = self.limit - len(self.value)
        self.value.extend(chunk[:available])
        self.truncated |= len(chunk) > available

    def text(self) -> str:
        payload = bytes(self.value)
        if self.truncated:
            payload += _TRUNCATED
        return payload.decode("utf-8", errors="replace")


def run_pipeline(
    stages: Sequence[PipelineStage],
    deadline: RequestDeadline,
    *,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 1024 * 1024,
) -> PipelineResult:
    """Run a shell-free pipeline and fully drain all observable output.

    Every process is started in an owned process group through
    :func:`start_process`. Intermediate stdout is consumed only by the next
    stage. All stderr pipes and final stdout are drained concurrently. Results
    are checked in pipeline order, so a successful downstream parser cannot
    mask an upstream selection or conversion failure.
    """
    if not stages:
        raise ValueError("pipeline must contain at least one stage")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer")

    processes: list[subprocess.Popen[Any]] = []
    stdout = _BoundedBytes(max_output_bytes)
    stderrs = [_BoundedBytes(max_output_bytes) for _ in stages]
    try:
        previous_stdout = None
        for index, stage in enumerate(stages):
            deadline.checkpoint()
            process = start_process(
                stage.command,
                stdin=previous_stdout if previous_stdout is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env,
            )
            processes.append(process)
            if previous_stdout is not None:
                previous_stdout.close()
            previous_stdout = process.stdout

        with selectors.DefaultSelector() as selector:
            for index, process in enumerate(processes):
                if process.stderr is None:
                    raise RuntimeError(f"Pipeline stage {stages[index].name!r} has no stderr pipe")
                os.set_blocking(process.stderr.fileno(), False)
                selector.register(process.stderr, selectors.EVENT_READ, (index, "stderr"))
            final_stdout = processes[-1].stdout
            if final_stdout is None:
                raise RuntimeError("Final pipeline stage has no stdout pipe")
            os.set_blocking(final_stdout.fileno(), False)
            selector.register(final_stdout, selectors.EVENT_READ, (len(stages) - 1, "stdout"))

            while selector.get_map() or any(process.poll() is None for process in processes):
                remaining = deadline.remaining()
                for key, _ in selector.select(min(_POLL_SECONDS, remaining)):
                    deadline.checkpoint()
                    try:
                        chunk = os.read(key.fd, _READ_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stage_index, stream_name = key.data
                    if stream_name == "stdout":
                        stdout.add(chunk)
                    else:
                        stderrs[stage_index].add(chunk)
            deadline.checkpoint()

        results: list[CommandResult] = []
        for index, (stage, process) in enumerate(zip(stages, processes, strict=True)):
            if process.returncode is None:
                raise RuntimeError(f"Pipeline stage {stage.name!r} did not exit")
            results.append(
                CommandResult(
                    cmd=stage.command,
                    returncode=process.returncode,
                    stdout=stdout.text() if index == len(stages) - 1 else "",
                    stderr=stderrs[index].text(),
                )
            )
        for stage, result in zip(stages, results, strict=True):
            if result.returncode != 0:
                raise PipelineStageError(stage.name, result)
        if stdout.truncated:
            raise PipelineOutputLimitError(stages[-1].name, "stdout", max_output_bytes)
        for stage, stderr in zip(stages, stderrs, strict=True):
            if stderr.truncated:
                raise PipelineOutputLimitError(stage.name, "stderr", max_output_bytes)
        return PipelineResult(tuple(results))
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            cleanup_processes(processes)
        except BaseException:
            if not primary_error:
                raise
        finally:
            for process in processes:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError:
                            if not primary_error:
                                raise


__all__ = [
    "PipelineResult",
    "PipelineOutputLimitError",
    "PipelineStage",
    "PipelineStageError",
    "RequestDeadline",
    "run_pipeline",
]
