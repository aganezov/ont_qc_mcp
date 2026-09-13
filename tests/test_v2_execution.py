"""Shared multi-stage subprocess lifecycle for API v2 adapters."""

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ont_qc_mcp.process_control import CANCEL_EVENT
from ont_qc_mcp.v2_execution import PipelineStage, PipelineStageError, RequestDeadline, run_pipeline


def python_stage(name: str, code: str) -> PipelineStage:
    return PipelineStage(name=name, command=(sys.executable, "-c", code))


def test_pipeline_drains_all_stages_and_returns_final_output() -> None:
    result = run_pipeline(
        [
            python_stage("source", "import sys; sys.stderr.write('source warning'); print('payload')"),
            python_stage(
                "sink",
                "import sys; data=sys.stdin.read(); sys.stderr.write('sink warning'); print(data.strip().upper())",
            ),
        ],
        RequestDeadline(5),
    )
    assert result.final.stdout == "PAYLOAD\n"
    assert result.stages[0].stderr == "source warning"
    assert result.stages[1].stderr == "sink warning"


def test_successful_downstream_output_never_masks_upstream_failure() -> None:
    with pytest.raises(PipelineStageError) as caught:
        run_pipeline(
            [
                python_stage("selection", "import sys; print('partial'); sys.exit(7)"),
                python_stage("json", "import sys; sys.stdin.read(); print('{\"ok\": true}')"),
            ],
            RequestDeadline(5),
        )
    assert caught.value.stage == "selection"
    assert caught.value.result.returncode == 7


def test_successful_pipeline_rejects_truncated_final_output() -> None:
    with pytest.raises(RuntimeError, match="output limit"):
        run_pipeline(
            [python_stage("large-output", "print('x' * 100)")],
            RequestDeadline(5),
            max_output_bytes=10,
        )


def test_one_request_deadline_covers_every_stage() -> None:
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="request deadline"):
        run_pipeline(
            [
                python_stage("source", "import time; time.sleep(.7); print('payload')"),
                python_stage("sink", "import sys,time; sys.stdin.read(); time.sleep(.7); print('done')"),
            ],
            RequestDeadline(1),
        )
    assert time.monotonic() - started < 3


def test_start_failure_cleans_up_already_started_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_execution

    processes: list[subprocess.Popen[Any]] = []
    original = v2_execution.start_process

    def fail_second(*args, **kwargs):
        if processes:
            raise FileNotFoundError("missing sink")
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(v2_execution, "start_process", fail_second)
    with pytest.raises(FileNotFoundError, match="missing sink"):
        run_pipeline(
            [
                python_stage("source", "import time; time.sleep(30)"),
                python_stage("sink", "print('never starts')"),
            ],
            RequestDeadline(5),
        )
    assert len(processes) == 1
    process = processes[0]
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)
    assert process.returncode is not None
    for pipe in (process.stdin, process.stdout, process.stderr):
        assert pipe is None or pipe.closed


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
def test_cancellation_kills_and_reaps_all_started_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_execution

    processes: list[subprocess.Popen[Any]] = []
    original = v2_execution.start_process

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(v2_execution, "start_process", capture)
    event = threading.Event()
    token = CANCEL_EVENT.set(event)
    timer = threading.Timer(0.2, event.set)
    timer.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            run_pipeline(
                [
                    python_stage("source", "import time; print('ready', flush=True); time.sleep(30)"),
                    python_stage("sink", "import sys,time; sys.stdin.read(); time.sleep(30)"),
                ],
                RequestDeadline(10),
            )
    finally:
        timer.cancel()
        timer.join()
        CANCEL_EVENT.reset(token)
    assert len(processes) == 2
    for process in processes:
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        assert process.returncode is not None
        for pipe in (process.stdin, process.stdout, process.stderr):
            assert pipe is None or pipe.closed
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, signal.SIGCONT)
