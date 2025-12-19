from __future__ import annotations

import asyncio
import os
import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_EXECUTOR: ThreadPoolExecutor | None = None
_LOCK = threading.Lock()
_THREADSAFE_WAKEUP_OK: bool | None = None


def get_executor() -> ThreadPoolExecutor:
    """
    Return a process-wide ThreadPoolExecutor.

    In this environment, asyncio's default executor can hang; using an explicit
    executor avoids that issue and keeps blocking work off the event loop.
    """
    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR

    with _LOCK:
        if _EXECUTOR is not None:
            return _EXECUTOR

        raw_workers = os.getenv("MCP_THREADPOOL_WORKERS")
        max_workers: int | None = None
        if raw_workers:
            try:
                parsed = int(raw_workers)
                if parsed > 0:
                    max_workers = parsed
            except ValueError:
                max_workers = None

        if max_workers is None:
            cpu = os.cpu_count() or 1
            max_workers = min(32, cpu + 4)

        _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ont-qc-mcp")
        return _EXECUTOR


def _threadsafe_wakeup_supported(loop: asyncio.AbstractEventLoop) -> bool:
    """
    Best-effort check whether loop thread wakeups work.

    Some sandboxed environments break asyncio's internal self-pipe, causing
    `loop.call_soon_threadsafe(...)` to not wake a sleeping loop. When this
    happens, awaiting `run_in_executor()` can hang indefinitely unless another
    timer/IO event wakes the loop.
    """
    ssock = getattr(loop, "_ssock", None)
    if ssock is None:
        return True

    try:
        ssock.setblocking(False)
    except (AttributeError, OSError, ValueError):
        return True

    # Drain any existing wake bytes (avoid false positives).
    try:
        while ssock.recv(4096):
            continue
    except (BlockingIOError, AttributeError, OSError):
        return True

    finished = threading.Event()

    def worker() -> None:
        try:
            loop.call_soon_threadsafe(lambda: None)
        finally:
            finished.set()

    threading.Thread(target=worker, daemon=True).start()
    finished.wait(timeout=0.2)
    time.sleep(0.01)

    try:
        readable, _, _ = select.select([ssock], [], [], 0)
    except Exception:
        return True

    # Drain wake bytes we generated (or any others) to keep the loop clean.
    try:
        while ssock.recv(4096):
            pass
    except (BlockingIOError, AttributeError, OSError):
        pass

    return bool(readable)


async def run_sync(func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a blocking callable without hanging the event loop.

    By default, this tries to offload work to a shared ThreadPoolExecutor. If the
    environment does not support reliable thread-safe loop wakeups, it falls
    back to running the callable synchronously in the event loop thread.
    """
    global _THREADSAFE_WAKEUP_OK
    loop = asyncio.get_running_loop()

    mode = os.getenv("MCP_BLOCKING_MODE", "auto").lower()
    if mode in {"sync", "direct"}:
        return func(*args, **kwargs)

    if mode not in {"auto", "executor", "threadpool", "sync", "direct"}:
        mode = "auto"

    if mode == "auto":
        if _THREADSAFE_WAKEUP_OK is None:
            _THREADSAFE_WAKEUP_OK = _threadsafe_wakeup_supported(loop)
        if not _THREADSAFE_WAKEUP_OK:
            return func(*args, **kwargs)

    return await loop.run_in_executor(get_executor(), partial(func, *args, **kwargs))


__all__ = ["get_executor", "run_sync"]
