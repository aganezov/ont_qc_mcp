"""Forced Python API runtimes validate their configured executables before launch."""

import json
import sys
from pathlib import Path
from typing import Literal

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp.config import ExecutionConfig, ToolPaths


def executable(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ToolPaths:
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setenv("TEST_RUNTIME_LOG", str(tmp_path / "runtime.jsonl"))
    monkeypatch.delenv("TEST_DOCKER_FAIL", raising=False)
    record = (
        "import json,os,pathlib,sys\n"
        "with open(os.environ['TEST_RUNTIME_LOG'],'a') as log: log.write(json.dumps(sys.argv)+'\\n')\n"
    )
    runtime = record + (
        "if sys.argv[1:] == ['info']: sys.exit(7)\n"
        "if sys.argv[1] == 'rm': sys.exit(0)\n"
        "if sys.argv[1] == 'run' and os.getenv('TEST_DOCKER_FAIL'):\n"
        "    print('injected launch failure',file=sys.stderr); sys.exit(2)\n"
        "batch=pathlib.Path(sys.argv[sys.argv.index('-b')+1])\n"
        "output=None\n"
        "for line in batch.read_text().splitlines():\n"
        "    if line.startswith('snapshotDirectory '): output=pathlib.Path(line.split(' ',1)[1])\n"
        "    if line.startswith('snapshot '):\n"
        "        (output/line.split(' ',1)[1]).write_text(pathlib.Path(sys.argv[0]).name)\n"
    )
    return ToolPaths(
        docker=executable(directory / "site-docker", runtime),
        apptainer=executable(directory / "site-apptainer", runtime),
        singularity=executable(directory / "site-singularity", runtime),
        igv=executable(directory / "site-igv", runtime),
        xvfb_run=executable(directory / "site-xvfb", record + "os.execvp(sys.argv[3],sys.argv[3:])\n"),
    )


def snapshot(tmp_path: Path, tools: ToolPaths, runtime: Literal["docker", "local"]):
    output = tmp_path / "output"
    batch = tmp_path / "batch"
    batch.write_text(f"snapshotDirectory {output}\nsnapshot region.png\nexit\n", encoding="utf-8")
    return cli.run_igv_snapshot(
        batch,
        output,
        tools,
        force_runtime=runtime,
        exec_cfg=ExecutionConfig(igv_container_image="stub:image"),
    )


@pytest.mark.parametrize(("runtime", "field"), [("docker", "docker"), ("local", "xvfb_run"), ("local", "igv")])
@pytest.mark.parametrize("unavailable", ["missing", "not-executable"])
def test_forced_runtime_unavailable_command_fails_before_launch_or_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tools: ToolPaths,
    runtime: Literal["docker", "local"],
    field: str,
    unavailable: str,
) -> None:
    configured = Path(getattr(tools, field))
    if unavailable == "missing":
        configured.unlink()
    else:
        configured.chmod(0o644)
    # An executable under the default name must not override the configured path.
    fallback = {"docker": "docker", "xvfb_run": "xvfb-run", "igv": "igv.sh"}[field]
    executable(configured.parent / fallback, "raise AssertionError('must not fall back')\n")
    calls: list[list[str]] = []
    original = cli.run_command

    def record(cmd, **kwargs):
        calls.append(list(cmd))
        return original(cmd, **kwargs)

    monkeypatch.setattr(cli, "run_command", record)
    with pytest.raises(RuntimeError, match=f"Requested '{runtime}' runtime") as caught:
        snapshot(tmp_path, tools, runtime)
    assert str(configured) in str(caught.value)
    assert calls == [], "Unavailable runtimes must not launch commands or attempt Docker rm"
    assert not (tmp_path / "runtime.jsonl").exists()


@pytest.mark.parametrize("runtime", ["docker", "local"])
def test_available_forced_runtime_launches_configured_commands(
    tmp_path: Path, tools: ToolPaths, runtime: Literal["docker", "local"]
) -> None:
    snapshots, actual, command = snapshot(tmp_path, tools, runtime)
    assert actual == runtime
    assert len(snapshots) == 1
    calls = [json.loads(line) for line in (tmp_path / "runtime.jsonl").read_text().splitlines()]
    if runtime == "docker":
        assert command[:3] == [tools.docker, "run", "--rm"]
        assert snapshots[0].read_text() == "site-docker"
        assert [call[0] for call in calls] == [tools.docker]
    else:
        assert command[0] == tools.xvfb_run and command[3] == tools.igv
        assert snapshots[0].read_text() == "site-igv"
        assert [call[0] for call in calls] == [tools.xvfb_run, tools.igv]
    # Available Apptainer/Singularity do not replace the forced choice, and
    # forced Docker proceeds to run without an automatic-detection info probe.
    assert not any("info" in call for call in calls)


def test_docker_failure_after_launch_still_cleans_its_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tools: ToolPaths
) -> None:
    monkeypatch.setenv("TEST_DOCKER_FAIL", "1")
    with pytest.raises(RuntimeError, match="injected launch failure") as caught:
        snapshot(tmp_path, tools, "docker")
    assert isinstance(caught.value.__cause__, cli.CommandError)
    calls = [json.loads(line) for line in (tmp_path / "runtime.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[0][0] == tools.docker and calls[0][1] == "run"
    assert calls[1] == [tools.docker, "rm", "--force", name]
