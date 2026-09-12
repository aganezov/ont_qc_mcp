"""Direct child-process ownership in the samtools -> nanoq pipeline."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from ont_qc_mcp.cli_wrappers import nanoq_from_bam_streaming
from ont_qc_mcp.config import ExecutionConfig, ToolPaths


NANOQ_JSON = json.dumps(
    {
        "reads": 1,
        "bases": 4,
        "shortest": 4,
        "longest": 4,
        "mean_length": 4,
        "median_length": 4,
        "n50": 4,
        "mean_quality": 40,
        "median_quality": 40,
    }
)


@pytest.fixture
def pipeline_probe(tmp_path, monkeypatch):
    """Record real children after a startup handshake; clean them even on red tests."""
    original_popen = subprocess.Popen
    processes = []
    original_poll, original_kill, original_wait = original_popen.poll, original_popen.kill, original_popen.wait
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def tracked_popen(cmd, **kwargs):
        proc = original_popen(cmd, **kwargs)
        processes.append(proc)
        marker = Path(cmd[0]).with_suffix(".started")
        deadline = time.monotonic() + 5
        while not marker.exists():
            if time.monotonic() >= deadline:
                pytest.fail(f"Child failed to reach startup handshake: {cmd}")
            time.sleep(0.01)
        return proc

    monkeypatch.setattr(subprocess, "Popen", tracked_popen)

    def make_tools(sam_body="signal.pause()", nano_body="signal.pause()"):
        paths = {}
        for name, body in (("samtools", sam_body), ("nanoq", nano_body)):
            script = tmp_path / f"{name}.py"
            script.write_text(
                f"#!{sys.executable}\nimport os, signal, sys\nfrom pathlib import Path\n"
                "Path(__file__).with_suffix('.started').write_text(str(os.getpid()))\n" + body + "\n"
            )
            script.chmod(0o755)
            paths[name] = str(script)
        return ToolPaths(**paths)

    yield make_tools, processes, original_popen

    # Separate from wrapper teardown so failures cannot leave test children behind.
    for proc in processes:
        if original_poll(proc) is None:
            original_kill(proc)
        original_wait(proc, timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def config():
    return ExecutionConfig(per_tool_timeouts={"samtools": 1, "nanoq": 1}, per_tool_threads={}, nanoq_aux_stats=True)


def assert_released(processes, tmp_path):
    for proc in processes:
        # Check waitpid before poll(): poll itself could conceal a missing reap.
        with pytest.raises(ChildProcessError):
            os.waitpid(proc.pid, os.WNOHANG)
        assert proc.returncode is not None
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            assert stream is None or stream.closed
    assert not list(tmp_path.glob("*.nanoq.*.txt"))


def test_second_child_start_failure_reaps_first_and_preserves_error(pipeline_probe, tmp_path):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools()
    tools.nanoq = str(tmp_path / "missing-nanoq")
    with pytest.raises(FileNotFoundError, match="missing-nanoq"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert len(children) == 1
    assert_released(children, tmp_path)


@pytest.mark.parametrize("stage", ["nanoq", "samtools"])
def test_each_timeout_reaps_children_and_closes_pipes(pipeline_probe, tmp_path, stage):
    make_tools, children, popen_class = pipeline_probe
    tools = (
        make_tools("sys.exit(0)", "signal.pause()")
        if stage == "nanoq"
        else make_tools("os.close(1)\nsignal.pause()", f"print({NANOQ_JSON!r})")
    )
    message = (
        "Timeout while running samtools\\|nanoq pipeline.*likely hung at nanoq"
        if stage == "nanoq"
        else "Timeout waiting for samtools fastq to exit"
    )
    with pytest.raises(RuntimeError, match=message):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert len(children) == 2
    assert_released(children, tmp_path)


def test_unexpected_communicate_failure_still_reaps_children(pipeline_probe, tmp_path, monkeypatch):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools()
    original_communicate = popen_class.communicate

    def fail_nanoq(proc, *args, **kwargs):
        if proc is children[-1] and len(children) == 2:
            raise OSError("nanoq pipe failure")
        return original_communicate(proc, *args, **kwargs)

    monkeypatch.setattr(popen_class, "communicate", fail_nanoq)
    with pytest.raises(OSError, match="nanoq pipe failure"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert len(children) == 2
    assert_released(children, tmp_path)


def test_cleanup_failure_does_not_replace_start_error(pipeline_probe, tmp_path, monkeypatch):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools()
    tools.nanoq = str(tmp_path / "missing-nanoq")

    def failed_terminate(proc):
        raise OSError("termination failed")

    monkeypatch.setattr(popen_class, "terminate", failed_terminate)
    with pytest.raises(FileNotFoundError, match="missing-nanoq"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert_released(children, tmp_path)


def test_second_aux_file_failure_removes_first(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    original = tempfile.NamedTemporaryFile
    created: list[Path] = []

    def create_once(*args, **kwargs):
        if created:
            raise OSError("cannot create qualities file")
        result = original(*args, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", create_once)
    with pytest.raises(OSError, match="cannot create qualities file"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", ToolPaths(), exec_cfg=config())
    assert len(created) == 1
    assert not created[0].exists()


def test_success_reaps_closes_and_does_not_terminate(pipeline_probe, tmp_path, monkeypatch):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools("print('@r\\nACGT\\n+\\nIIII')", f"sys.stdin.read()\nprint({NANOQ_JSON!r})")

    def unexpected_signal(proc):
        pytest.fail("Successful children must not be terminated or killed")

    monkeypatch.setattr(popen_class, "terminate", unexpected_signal)
    monkeypatch.setattr(popen_class, "kill", unexpected_signal)
    stats = nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert (stats.read_count, stats.total_bases) == (1, 4)
    assert_released(children, tmp_path)


def test_samtools_failure_preserves_stderr(pipeline_probe, tmp_path):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools(
        "print('alignment corrupt', file=sys.stderr)\nsys.exit(7)", f"sys.stdin.read()\nprint({NANOQ_JSON!r})"
    )
    with pytest.raises(RuntimeError, match="samtools fastq failed:.*\\nalignment corrupt"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert_released(children, tmp_path)


def test_unexpected_wait_failure_still_reaps_children(pipeline_probe, tmp_path, monkeypatch):
    make_tools, children, popen_class = pipeline_probe
    tools = make_tools("os.close(1)\nsignal.pause()", f"print({NANOQ_JSON!r})")
    original_wait = popen_class.wait
    failed = False

    def fail_samtools_once(proc, *args, **kwargs):
        nonlocal failed
        if children and proc is children[0] and not failed:
            failed = True
            raise OSError("samtools wait failed")
        return original_wait(proc, *args, **kwargs)

    monkeypatch.setattr(popen_class, "wait", fail_samtools_once)
    with pytest.raises(OSError, match="samtools wait failed"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert_released(children, tmp_path)


def test_stderr_thread_start_failure_reaps_both_children(pipeline_probe, tmp_path, monkeypatch):
    from ont_qc_mcp import cli_wrappers

    make_tools, children, _ = pipeline_probe
    tools = make_tools()

    def failed_start(thread):
        raise RuntimeError("stderr thread startup failed")

    monkeypatch.setattr(cli_wrappers.Thread, "start", failed_start)
    with pytest.raises(RuntimeError, match="stderr thread startup failed"):
        nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
    assert len(children) == 2
    assert_released(children, tmp_path)


def test_descendant_held_stderr_does_not_extend_wrapper_timeout(pipeline_probe, tmp_path, monkeypatch):
    import signal

    make_tools, children, _ = pipeline_probe
    holder = tmp_path / "stderr_holder.py"
    holder.write_text(
        "import os, signal\nfrom pathlib import Path\n"
        "signal.alarm(8)\n"
        "Path(__file__).with_suffix('.started').write_text(str(os.getpid()))\n"
        "signal.pause()\n"
    )
    marker = holder.with_suffix(".started")
    tools = make_tools(
        "import subprocess\n"
        f"subprocess.Popen([{sys.executable!r}, {str(holder)!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)\n"
        "signal.pause()"
    )
    tracked_start = subprocess.Popen
    pipeline_started_at = []

    def wait_for_holder(cmd, **kwargs):
        proc = tracked_start(cmd, **kwargs)
        if cmd[0] == tools.samtools:
            deadline = time.monotonic() + 5
            while not marker.exists():
                assert time.monotonic() < deadline, "Descendant never reached startup handshake"
                time.sleep(0.01)
        else:
            pipeline_started_at.append(time.monotonic())
        return proc

    monkeypatch.setattr(subprocess, "Popen", wait_for_holder)
    try:
        with pytest.raises(RuntimeError, match="Timeout while running samtools\\|nanoq pipeline"):
            nanoq_from_bam_streaming(tmp_path / "unused.bam", tools, exec_cfg=config())
        assert time.monotonic() - pipeline_started_at[0] < 4
        assert len(children) == 2
        for proc in children:
            with pytest.raises(ChildProcessError):
                os.waitpid(proc.pid, os.WNOHANG)
        # The caller returns before the descendant releases the reader's stream.
        os.kill(int(marker.read_text()), 0)
        assert not children[0].stderr.closed
        assert children[0].stdout.closed
        assert children[1].stdout.closed and children[1].stderr.closed
        assert not list(tmp_path.glob("*.nanoq.*.txt"))
    finally:
        # The descendant is outside wrapper ownership; release it independently.
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while not children[0].stderr.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert_released(children, tmp_path)
