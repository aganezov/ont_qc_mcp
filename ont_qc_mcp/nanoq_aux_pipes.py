"""Own and drain nanoq's two per-read auxiliary streams without data files."""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread
from types import TracebackType
from typing import TYPE_CHECKING

from .config import ExecutionConfig
from .nanoq_aux import _HistogramAccumulator, _length_percentiles
from .process_control import check_cancelled
from .schemas import NanoqStats

if TYPE_CHECKING:
    from .v2_execution import RequestDeadline


class _AuxReader:
    def __init__(self, path: Path, accumulator: _HistogramAccumulator, done: Event, stop: Event):
        self.path = path
        self.accumulator = accumulator
        self.done = done
        self.stop = stop
        self.had_bytes = False
        self.error: Exception | None = None
        self.thread = Thread(target=self._read, name=f"nanoq-aux-{path.name}", daemon=True)
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    def _read(self) -> None:
        pending = b""
        while not self.stop.is_set():
            # Sample before read: a producer can finish between an empty read and
            # the next check. Reading once after done prevents losing its tail.
            producer_done = self.done.is_set()
            try:
                chunk = os.read(self.fd, 65536)
            except BlockingIOError:
                chunk = b""
            except OSError as exc:
                self.error = self.error or exc
                chunk = b""
            if not chunk:
                if producer_done:
                    break
                self.stop.wait(0.01)
                continue
            self.had_bytes = True
            if self.error is not None:
                continue  # Drain after a parser error so the writer cannot block.
            try:
                lines = (pending + chunk).split(b"\n")
                pending = lines.pop()
                if len(pending) > 4096 or any(len(line) > 4096 for line in lines):
                    raise ValueError("Nanoq auxiliary line exceeds 4096 bytes")
                for line in lines:
                    self.accumulator.add_line(line.decode("utf-8"))
            except Exception as exc:
                self.error = exc
                pending = b""
        if pending and self.error is None and not self.stop.is_set():
            try:
                self.accumulator.add_line(pending.decode("utf-8"))
            except Exception as exc:
                self.error = exc


class NanoqAuxPipes:
    """Fresh FIFO paths and bounded reader state for exactly one nanoq attempt."""

    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.readers: list[_AuxReader] = []
        self.done = Event()
        self.stop = Event()

    def __enter__(self) -> NanoqAuxPipes:
        if not hasattr(os, "mkfifo"):
            raise RuntimeError(
                "Nanoq auxiliary statistics require POSIX named pipes; set MCP_NANOQ_AUX_STATS=0 to disable"
            )
        try:
            self.directory = tempfile.TemporaryDirectory(prefix="nanoq_aux_")
            accumulators = [
                _HistogramAccumulator(
                    float(self.cfg.nanoq_length_bin_width), int, self.cfg.nanoq_percentiles_exact_max_reads
                ),
                _HistogramAccumulator(self.cfg.nanoq_qscore_bin_width, float),
            ]
            for name, accumulator in zip(("lengths", "qualities"), accumulators):
                path = Path(self.directory.name) / name
                os.mkfifo(path, 0o600)
                reader = _AuxReader(path, accumulator, self.done, self.stop)
                self.readers.append(reader)
                reader.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    @property
    def args(self) -> list[str]:
        return ["--read-lengths", str(self.readers[0].path), "--read-qualities", str(self.readers[1].path)]

    def finish(self, deadline: RequestDeadline | None = None) -> None:
        """Call only after the producer has exited; drain all buffered output."""
        self.done.set()
        local_deadline = time.monotonic() + 5
        for reader in self.readers:
            while reader.thread.is_alive():
                if deadline is None:
                    check_cancelled()
                else:
                    deadline.checkpoint()
                if deadline is None and time.monotonic() >= local_deadline:
                    raise RuntimeError("Nanoq auxiliary readers did not finish after producer exit")
                reader.thread.join(timeout=0.05)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        for reader in self.readers:
            if reader.error is not None:
                raise RuntimeError(f"Cannot read nanoq auxiliary {reader.path.name}: {reader.error}") from reader.error

    def augment(self, stats: NanoqStats, deadline: RequestDeadline | None = None) -> None:
        self.finish(deadline)
        lengths, qualities = self.readers
        if lengths.had_bytes:
            if stats.length_histogram is None:
                stats.length_histogram = lengths.accumulator.histogram()
            if stats.length_percentiles is None:
                stats.length_percentiles = _length_percentiles(lengths.accumulator.values)
        if qualities.had_bytes and stats.qscore_histogram is None:
            stats.qscore_histogram = qualities.accumulator.histogram()

    def close(self) -> None:
        self.stop.set()
        with ExitStack() as cleanup:
            if self.directory is not None:
                cleanup.callback(self.directory.cleanup)
            for reader in self.readers:
                cleanup.callback(os.close, reader.fd)
            for reader in self.readers:
                if reader.thread.ident is not None:
                    reader.thread.join()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
        if exc is not None and isinstance(exc, Exception):
            self.raise_if_failed()
