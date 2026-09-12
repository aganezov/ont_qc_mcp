"""Cancellation must not strand or cancel unrelated deduplicated FASTQ QC work."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ont_qc_mcp import threadpool, tools
from ont_qc_mcp.process_control import CANCEL_EVENT, check_cancelled
from ont_qc_mcp.schemas import NanoqStats


async def wait_for_event(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["owner-cancelled", "waiter-cancelled", "owner-timeout", "success"])
async def test_deduplicated_qc_cancellation(tmp_path, monkeypatch, outcome):
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@r\nACGT\n+\n!!!!\n")
    monkeypatch.setattr(tools, "_NANOQ_CACHE", {})
    monkeypatch.setattr(tools, "_NANOQ_INFLIGHT", {})
    monkeypatch.setenv("MCP_BLOCKING_MODE", "executor")
    executor = ThreadPoolExecutor(max_workers=3)
    monkeypatch.setattr(threadpool, "get_executor", lambda: executor)
    started = threading.Event()
    release = threading.Event()
    waiter_entered = threading.Event()
    calls = []

    def nanoq(path, *args, **kwargs):
        calls.append(path)
        started.set()
        while not release.wait(0.01):
            check_cancelled()
        check_cancelled()
        if outcome == "owner-timeout":
            raise TimeoutError("nanoq timed out")
        return NanoqStats(
            file=str(path), read_count=1, total_bases=4, min_len=4, max_len=4, mean_len=4.0, median_len=4.0
        )

    monkeypatch.setattr(tools, "nanoq_stats", nanoq)
    owner = asyncio.create_task(threadpool.run_sync(tools.qc_reads, str(fastq)))
    waiter = None
    shared = None
    try:
        await wait_for_event(started)
        with tools._NANOQ_CACHE_LOCK:
            shared = next(iter(tools._NANOQ_INFLIGHT.values()))
        original_result = shared.result

        def observed_result(*args, **kwargs):
            waiter_entered.set()
            return original_result(*args, **kwargs)

        monkeypatch.setattr(shared, "result", observed_result)
        waiter = asyncio.create_task(threadpool.run_sync(tools.qc_reads, str(fastq)))
        await wait_for_event(waiter_entered)
        assert len(calls) == 1

        if outcome == "owner-cancelled":
            owner.cancel()
            done, _ = await asyncio.wait([owner, waiter], timeout=2)
            assert owner in done
            assert waiter in done, "Owner cancellation stranded a duplicate QC worker"
            assert owner.cancelled()
            with pytest.raises(RuntimeError, match="Shared FASTQ QC.*cancelled.*retry"):
                waiter.result()
            assert not tools._NANOQ_CACHE
            assert shared.done()
            assert not tools._NANOQ_INFLIGHT
            release.set()
            retried = await threadpool.run_sync(tools.qc_reads, str(fastq))
            assert retried.read_count == 1
            assert len(calls) == 2
        elif outcome == "waiter-cancelled":
            waiter.cancel()
            done, _ = await asyncio.wait([waiter], timeout=2)
            assert waiter in done, "A cancelled duplicate QC worker remained blocked on its owner"
            assert waiter.cancelled()
            assert not owner.done()
            assert not shared.done()
            release.set()
            result = await owner
            assert result.read_count == 1
        elif outcome == "owner-timeout":
            release.set()
            done, _ = await asyncio.wait([owner, waiter], timeout=2)
            assert owner in done and waiter in done
            for task in (owner, waiter):
                with pytest.raises(TimeoutError, match="nanoq timed out"):
                    task.result()
            assert not tools._NANOQ_CACHE
            assert not tools._NANOQ_INFLIGHT
            return
        else:
            release.set()
            first, second = await asyncio.gather(owner, waiter)
            assert first is second
            assert first.read_count == 1

        cached = await threadpool.run_sync(tools.qc_reads, str(fastq))
        assert cached.read_count == 1
        assert len(calls) == (2 if outcome == "owner-cancelled" else 1)
        assert not tools._NANOQ_INFLIGHT
    finally:
        # Unblock the known pre-fix leak so a failing regression cannot strand pytest.
        release.set()
        if shared is not None and owner.done() and not shared.done():
            shared.set_exception(RuntimeError("Test cleanup released an abandoned future"))
        await asyncio.gather(owner, *([waiter] if waiter else []), return_exceptions=True)
        executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("checkpoint", ["cached-result", "publication"])
def test_qc_cache_does_not_publish_success_after_cancellation(tmp_path, monkeypatch, checkpoint):
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@r\nACGT\n+\n!!!!\n")
    monkeypatch.setattr(tools, "_NANOQ_CACHE", {})
    monkeypatch.setattr(tools, "_NANOQ_INFLIGHT", {})
    event = threading.Event()
    stats = NanoqStats(file=str(fastq), read_count=1, total_bases=4, min_len=4, max_len=4, mean_len=4.0, median_len=4.0)

    def nanoq(*args, **kwargs):
        if checkpoint == "publication":
            event.set()
        return stats

    monkeypatch.setattr(tools, "nanoq_stats", nanoq)
    if checkpoint == "cached-result":
        tools.qc_reads(str(fastq))
        event.set()
    token = CANCEL_EVENT.set(event)
    try:
        with pytest.raises(asyncio.CancelledError):
            tools.qc_reads(str(fastq))
    finally:
        CANCEL_EVENT.reset(token)
    assert not tools._NANOQ_INFLIGHT
    if checkpoint == "publication":
        assert not tools._NANOQ_CACHE
