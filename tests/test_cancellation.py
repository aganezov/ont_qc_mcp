"""Cancellation must follow the lifetime of real executor work."""

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest
import mcp_types as types

from ont_qc_mcp import app_server, threadpool


async def wait_for_event(event: threading.Event) -> None:
    async def poll() -> None:
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=5)


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(max_workers=getattr(request, "param", 2))
    monkeypatch.setenv("MCP_BLOCKING_MODE", "executor")
    monkeypatch.setattr(threadpool, "get_executor", lambda: executor)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("cancel_kind", ["asyncio", "asyncio-repeated", "anyio"])
@pytest.mark.parametrize("worker_error", [False, True], ids=["worker-returns", "worker-raises"])
@pytest.mark.asyncio
async def test_running_cancelled_dispatch_retains_capacity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pool: ThreadPoolExecutor,
    cancel_kind: str,
    worker_error: bool,
) -> None:
    started = [threading.Event(), threading.Event()]
    finished = [threading.Event(), threading.Event()]
    release = threading.Event()
    cancellation_seen = threading.Event()
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def worker(index: int) -> None:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        started[index].set()
        try:
            if not release.wait(5):
                raise RuntimeError("Test cleanup deadline exceeded")
            if index == 0 and worker_error:
                raise RuntimeError("Worker failed after cancellation")
        finally:
            with lock:
                active -= 1
            finished[index].set()

    async def handler(index: int) -> list[types.TextContent]:
        await threadpool.run_sync(worker, index)
        return []

    name = "test_cancelled_worker_capacity"
    semaphore = anyio.Semaphore(1)
    monkeypatch.setattr(app_server, "_CONCURRENCY_SEM", semaphore)
    monkeypatch.setitem(app_server.TOOL_SPECS, name, app_server.ToolSpec(name, "test", handler, {}, {}))
    scopes: list[anyio.CancelScope] = []

    async def first_request() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await app_server.dispatch_tool(name, {"index": 0})
            except asyncio.CancelledError:
                cancellation_seen.set()
                raise

    first = asyncio.create_task(first_request())
    tasks: list[asyncio.Task] = [first]
    try:
        await wait_for_event(started[0])
        if cancel_kind == "anyio":
            scopes[0].cancel()
        else:
            first.cancel()
        second = asyncio.create_task(app_server.dispatch_tool(name, {"index": 1}))
        tasks.append(second)
        for _ in range(3):
            await asyncio.sleep(0.01)
            if cancel_kind == "asyncio-repeated":
                first.cancel()
        await asyncio.sleep(0)
        assert not finished[0].is_set()
        assert not first.done(), "Cancellation released dispatch before its worker completed"
        assert not cancellation_seen.is_set()
        assert semaphore.value == 0
        assert not started[1].is_set(), "A second worker bypassed the concurrency limit"
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
        assert cancellation_seen.is_set(), "Worker results must not replace cancellation"
        assert first.cancelled() if cancel_kind != "anyio" else scopes[0].cancelled_caught
        assert finished[0].is_set() and finished[1].is_set()
        assert not second.result().is_error
        assert peak_active == 1
        assert semaphore.value == 1
        worker_warnings = [record for record in caplog.records if record.name == "ont_qc_mcp.threadpool"]
        assert len(worker_warnings) == int(worker_error)
        if worker_error:
            assert "Worker failed after cancellation" in worker_warnings[0].getMessage()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_dispatch_capacity_prevents_start(
    monkeypatch: pytest.MonkeyPatch, pool: ThreadPoolExecutor
) -> None:
    started = threading.Event()

    async def handler() -> list[types.TextContent]:
        await threadpool.run_sync(started.set)
        return []

    name = "test_cancelled_capacity_waiter"
    semaphore = anyio.Semaphore(1)
    monkeypatch.setattr(app_server, "_CONCURRENCY_SEM", semaphore)
    monkeypatch.setitem(app_server.TOOL_SPECS, name, app_server.ToolSpec(name, "test", handler, {}, {}))
    await semaphore.acquire()
    task = asyncio.create_task(app_server.dispatch_tool(name, {}))
    try:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not started.is_set()
        assert semaphore.value == 0
    finally:
        semaphore.release()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("pool", [1], indirect=True)
@pytest.mark.asyncio
async def test_cancelled_executor_queue_entry_never_starts(
    monkeypatch: pytest.MonkeyPatch, pool: ThreadPoolExecutor
) -> None:
    occupied = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    queued_started = threading.Event()

    def occupy_worker() -> None:
        occupied.set()
        if not release.wait(5):
            raise RuntimeError("Test cleanup deadline exceeded")

    blocker = pool.submit(occupy_worker)
    original_submit = pool.submit

    def record_submission(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        submitted.set()
        return future

    monkeypatch.setattr(pool, "submit", record_submission)
    task = asyncio.create_task(threadpool.run_sync(queued_started.set))
    try:
        await wait_for_event(occupied)
        await wait_for_event(submitted)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert not release.is_set(), "Queued cancellation must not wait for an unrelated worker"
        release.set()
        await asyncio.wrap_future(blocker)
        # A sentinel after the canceled queue entry proves the executor passed it.
        await asyncio.wrap_future(original_submit(lambda: None))
        assert not queued_started.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wrap_future(blocker)


@pytest.mark.parametrize("cancel_kind", ["asyncio", "anyio"])
@pytest.mark.asyncio
async def test_already_cancelled_executor_call_is_not_submitted(
    monkeypatch: pytest.MonkeyPatch, pool: ThreadPoolExecutor, cancel_kind: str
) -> None:
    submitted = threading.Event()
    original_submit = pool.submit

    def record_submission(*args, **kwargs):
        submitted.set()
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(pool, "submit", record_submission)

    async def request() -> None:
        with anyio.CancelScope() as scope:
            if cancel_kind == "anyio":
                scope.cancel()
            else:
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
            await threadpool.run_sync(lambda: None)

    task = asyncio.create_task(request())
    await asyncio.gather(task, return_exceptions=True)
    assert not submitted.is_set()


@pytest.mark.asyncio
async def test_executor_preserves_normal_return_and_error(pool: ThreadPoolExecutor) -> None:
    caller_thread = threading.get_ident()

    def calculate(value: int, *, offset: int) -> tuple[int, int]:
        return value + offset, threading.get_ident()

    result, worker_thread = await threadpool.run_sync(calculate, 3, offset=4)
    assert result == 7
    assert worker_thread != caller_thread

    def fail() -> None:
        raise ValueError("original worker error")

    with pytest.raises(ValueError, match="original worker error"):
        await threadpool.run_sync(fail)


@pytest.mark.parametrize("mode", ["direct", "sync", "auto"])
@pytest.mark.asyncio
async def test_synchronous_modes_keep_execution_and_deferred_cancellation(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("MCP_BLOCKING_MODE", mode)
    monkeypatch.setattr(threadpool, "_THREADSAFE_WAKEUP_OK", False)
    caller_thread = threading.get_ident()
    returned: list[int] = []

    async def request() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        returned.append(await threadpool.run_sync(threading.get_ident))
        await asyncio.sleep(0)

    task = asyncio.create_task(request())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert returned == [caller_thread]
