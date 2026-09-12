"""Runtime detection must select an executable that IGV can actually launch."""

import json
import sys
from pathlib import Path
from typing import Literal

import pytest

from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.tools import env_check


@pytest.fixture
def runtime_bin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    return directory


def make_runtime(directory: Path, name: str) -> Path:
    executable = directory / name
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        "if sys.argv[1:] == ['info']: sys.exit(0)\n"
        "batch=pathlib.Path(sys.argv[sys.argv.index('-b')+1])\n"
        "output=None\n"
        "for line in batch.read_text().splitlines():\n"
        "    if line.startswith('snapshotDirectory '): output=pathlib.Path(line.split(' ',1)[1])\n"
        "    if line.startswith('snapshot '):\n"
        "        (output/line.split(' ',1)[1]).write_text(pathlib.Path(sys.argv[0]).name)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def run_snapshot(
    tmp_path: Path, tools: ToolPaths, force_runtime: Literal["apptainer"] | None = None
) -> tuple[list[Path], str, list[str]]:
    output = tmp_path / "snapshot output"
    batch = tmp_path / "igv.batch"
    batch.write_text(f"snapshotDirectory {output}\nsnapshot region.png\nexit\n", encoding="utf-8")
    return cli.run_igv_snapshot(
        batch,
        output,
        tools,
        exec_cfg=ExecutionConfig(igv_container_image="stub:image", igv_sif_path=None),
        force_runtime=force_runtime,
    )


@pytest.mark.parametrize("force_runtime", [None, "apptainer"], ids=["auto", "forced-category"])
@pytest.mark.parametrize("configured", ["singularity", "custom-name", "absolute-path"])
def test_singularity_only_launches_the_detected_executable(
    runtime_bin: Path, tmp_path: Path, configured: str, force_runtime: Literal["apptainer"] | None
) -> None:
    name = "singularity" if configured == "singularity" else "site-container-runtime"
    executable = make_runtime(runtime_bin, name)
    selected = str(executable) if configured == "absolute-path" else name
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity=selected)

    assert cli.detect_container_runtime(tools) == "apptainer"
    status = env_check(tools)
    assert status.igv_runtime == "apptainer"
    assert status.available["igv_snapshot"] and status.available["singularity"]
    assert not status.available["apptainer"]
    assert status.resolved_paths["singularity"] == str(executable)
    assert json.loads(status.model_dump_json())["igv_runtime"] == "apptainer"

    snapshots, runtime, command = run_snapshot(tmp_path, tools, force_runtime)
    assert runtime == "apptainer"
    assert command[:2] == [str(executable), "exec"]
    assert len(snapshots) == 1
    assert snapshots[0].read_text() == name
    assert "docker://stub:image" in command


def test_apptainer_is_preferred_to_singularity(runtime_bin: Path, tmp_path: Path) -> None:
    apptainer = make_runtime(runtime_bin, "custom-apptainer")
    singularity = make_runtime(runtime_bin, "singularity")
    tools = ToolPaths(docker="docker", apptainer=str(apptainer), singularity=str(singularity))
    snapshots, runtime, command = run_snapshot(tmp_path, tools)
    assert runtime == "apptainer"
    assert command[:2] == [str(apptainer), "exec"]
    assert snapshots[0].read_text() == "custom-apptainer"


def test_available_docker_keeps_priority(runtime_bin: Path, tmp_path: Path) -> None:
    docker = make_runtime(runtime_bin, "custom-docker")
    apptainer = make_runtime(runtime_bin, "apptainer")
    singularity = make_runtime(runtime_bin, "singularity")
    tools = ToolPaths(docker=str(docker), apptainer=str(apptainer), singularity=str(singularity))
    snapshots, runtime, command = run_snapshot(tmp_path, tools)
    assert runtime == "docker"
    assert command[:3] == [str(docker), "run", "--rm"]
    assert "--name" in command
    assert snapshots[0].read_text() == "custom-docker"


def test_forced_apptainer_category_uses_configured_singularity_fallback(runtime_bin: Path, tmp_path: Path) -> None:
    # An invalid explicit APPTAINER path must not silently select a different
    # apptainer on PATH. Forced category selection also bypasses available Docker.
    make_runtime(runtime_bin, "apptainer")
    make_runtime(runtime_bin, "docker")
    singularity = make_runtime(runtime_bin, "custom-singularity")
    tools = ToolPaths(docker="docker", apptainer=str(tmp_path / "missing-apptainer"), singularity=str(singularity))
    snapshots, runtime, command = run_snapshot(tmp_path, tools, force_runtime="apptainer")
    assert runtime == "apptainer"
    assert command[:2] == [str(singularity), "exec"]
    assert snapshots[0].read_text() == "custom-singularity"


def test_forced_apptainer_category_errors_when_both_configured_commands_are_missing(
    runtime_bin: Path, tmp_path: Path
) -> None:
    make_runtime(runtime_bin, "apptainer")
    make_runtime(runtime_bin, "singularity")
    tools = ToolPaths(
        docker="docker",
        apptainer=str(tmp_path / "missing-apptainer"),
        singularity=str(tmp_path / "missing-singularity"),
    )
    assert cli.detect_container_runtime(tools) is None
    with pytest.raises(RuntimeError, match="[Aa]pptainer.*[Ss]ingularity"):
        run_snapshot(tmp_path, tools, force_runtime="apptainer")
