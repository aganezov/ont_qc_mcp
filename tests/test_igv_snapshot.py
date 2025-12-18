import json
import os
import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

import ont_qc_mcp.cli_wrappers as cli
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.tools import generate_igv_snapshots


def _is_arm64_mac() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=8)
def _apptainer_exec_works(apptainer_cmd: str, sif_path: str) -> bool:
    """
    Best-effort probe for whether apptainer can execute a local SIF on this host.

    Some environments have an `apptainer` binary on PATH but still cannot execute
    containers (e.g. NSS/user lookup issues in sandboxes, or unprivileged user
    namespaces disabled on shared systems). In such cases, IGV snapshot tests
    should fall back to mock mode instead of hard-failing.
    """
    if not Path(sif_path).exists():
        return False

    try:
        with tempfile.TemporaryDirectory(prefix="ont-qc-mcp-apptainer-") as tmp:
            env = os.environ.copy()
            env.setdefault("APPTAINER_CACHEDIR", str(Path(tmp) / "cache"))
            env.setdefault("APPTAINER_TMPDIR", str(Path(tmp) / "tmp"))
            env.setdefault("APPTAINER_DISABLE_CACHE", "1")
            probe = subprocess.run(
                [apptainer_cmd, "exec", "--no-home", sif_path, "/bin/true"],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
                check=False,
            )
            return probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _should_use_mock() -> str:
    """
    Determine mock mode based on environment and platform.
    
    - MCP_IGV_MOCK=0: Force real execution (even on ARM64)
    - MCP_IGV_MOCK=1: Force mock execution
    - Unset: Auto-detect (mock on ARM64 Mac; otherwise real only when a runnable container runtime is available)
    """
    explicit = os.getenv("MCP_IGV_MOCK")
    if explicit is not None:
        return explicit
    # Default: mock on ARM64 Mac (x86 image is slow), real elsewhere
    if _is_arm64_mac():
        return "1"

    runtime = cli.detect_container_runtime(ToolPaths())
    if runtime == "docker":
        return "0"
    if runtime == "apptainer":
        # Avoid implicit pulls from docker:// in test environments; require an explicit local SIF to run "real".
        sif_path = os.getenv("MCP_IGV_SIF_PATH")
        if sif_path and Path(sif_path).exists():
            apptainer_cmd = cli.which(ToolPaths().apptainer) or ToolPaths().apptainer
            if _apptainer_exec_works(apptainer_cmd, sif_path):
                return "0"
        return "1"

    return "1"


def _preserve_snapshot(snapshot_path: Path, test_name: str) -> None:
    """Copy snapshot to IGV_SNAPSHOT_DIR for CI artifact preservation."""
    snapshot_dir = os.getenv("IGV_SNAPSHOT_DIR")
    if snapshot_dir and snapshot_path.exists():
        dest_dir = Path(snapshot_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{test_name}_{snapshot_path.name}"
        shutil.copy(snapshot_path, dest)


def test_detect_runtime_docker_available(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")
    monkeypatch.setattr(cli, "which", lambda cmd: f"/usr/bin/{cmd}" if cmd == "docker" else None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0))

    assert cli.detect_container_runtime(tools) == "docker"


def test_detect_runtime_docker_daemon_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")
    monkeypatch.setattr(
        cli,
        "which",
        lambda cmd: f"/usr/bin/{cmd}" if cmd in {"docker", "apptainer"} else None,
    )

    def _raise(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["docker"])

    monkeypatch.setattr(cli.subprocess, "run", _raise)

    assert cli.detect_container_runtime(tools) == "apptainer"


def test_detect_runtime_none_available(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")
    monkeypatch.setattr(cli, "which", lambda cmd: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0))

    assert cli.detect_container_runtime(tools) is None


def test_build_docker_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batch_file = tmp_path / "igv.batch"
    batch_file.write_text("exit\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    mount_dir = tmp_path / "data"
    output_dir.mkdir()
    mount_dir.mkdir()
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")

    def _fake_run_command(cmd, timeout):
        (output_dir / "snapshot.png").write_text("test", encoding="utf-8")
        return None

    monkeypatch.setattr(cli, "run_command", _fake_run_command)

    snapshots, runtime, cmd = cli.run_igv_snapshot(
        batch_file=batch_file,
        output_dir=output_dir,
        tools=tools,
        exec_cfg=ExecutionConfig(),
        snapshot_format="png",
        force_runtime="docker",
        mount_paths=[mount_dir],
    )

    assert runtime == "docker"
    assert snapshots == [output_dir / "snapshot.png"]
    assert tools.docker in cmd[0]
    assert "--rm" in cmd
    assert any(str(output_dir) in part for part in cmd)
    assert str(batch_file) in cmd


def test_build_apptainer_command_with_sif(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batch_file = tmp_path / "igv.batch"
    batch_file.write_text("exit\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sif_path = tmp_path / "igv.sif"
    sif_path.write_text("image", encoding="utf-8")
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")

    def _fake_run_command(cmd, timeout):
        (output_dir / "snapshot.svg").write_text("svg", encoding="utf-8")
        return None

    monkeypatch.setattr(cli, "run_command", _fake_run_command)

    cfg = ExecutionConfig(igv_sif_path=str(sif_path))
    snapshots, runtime, cmd = cli.run_igv_snapshot(
        batch_file=batch_file,
        output_dir=output_dir,
        tools=tools,
        exec_cfg=cfg,
        snapshot_format="svg",
        force_runtime="apptainer",
        mount_paths=[tmp_path],
    )

    assert runtime == "apptainer"
    assert snapshots == [output_dir / "snapshot.svg"]
    assert sif_path.as_posix() in cmd
    assert "--bind" in cmd
    assert any(part.endswith("xvfb-run") for part in cmd)


def test_run_igv_snapshot_no_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batch_file = tmp_path / "igv.batch"
    batch_file.write_text("exit\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    tools = ToolPaths(docker="docker", apptainer="apptainer", singularity="singularity")

    monkeypatch.setattr(cli, "detect_container_runtime", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError):
        cli.run_igv_snapshot(
            batch_file=batch_file,
            output_dir=output_dir,
            tools=tools,
            exec_cfg=ExecutionConfig(),
            snapshot_format="png",
        )


def _text_content(content: types.Content) -> types.TextContent:
    return cast(types.TextContent, content)


@pytest.mark.igv_integration
def test_igv_snapshot_from_bed(
    sample_bam_highdepth: Path,
    sample_reference: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_flag = _should_use_mock()
    monkeypatch.setenv("MCP_IGV_MOCK", mock_flag)
    bed_path = tmp_path / "regions.bed"
    bed_path.write_text("chrHD\t1\t150\tbed_region\n", encoding="utf-8")

    result = generate_igv_snapshots(
        genome=str(sample_reference),
        tracks=[str(sample_bam_highdepth)],
        regions=str(bed_path),
        output_dir=str(tmp_path),
    )

    assert result.execution_mode in {"docker", "apptainer"}
    assert len(result.snapshot_files) == 1
    snapshot_path = Path(result.snapshot_files[0])
    assert snapshot_path.exists()
    assert snapshot_path.name == "bed_region.png"
    _preserve_snapshot(snapshot_path, "bed_region")


@pytest.mark.igv_integration
def test_igv_snapshot_custom_genome(
    sample_bam_highdepth: Path,
    sample_reference: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_flag = _should_use_mock()
    monkeypatch.setenv("MCP_IGV_MOCK", mock_flag)

    result = generate_igv_snapshots(
        genome=str(sample_reference),
        tracks=[str(sample_bam_highdepth)],
        regions=[{"chrom": "chrHD", "start": 1, "end": 200, "name": "custom"}],
        output_dir=str(tmp_path),
        snapshot_format="svg",
    )

    assert result.execution_mode in {"docker", "apptainer"}
    assert len(result.snapshot_files) == 1
    snapshot_path = Path(result.snapshot_files[0])
    assert snapshot_path.exists()
    assert snapshot_path.suffix == ".svg"
    _preserve_snapshot(snapshot_path, "custom_genome")


@pytest.mark.igv_integration
def test_igv_snapshot_bam_and_vcf(
    sample_bam: Path,
    sample_vcf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test snapshot generation with both BAM and VCF tracks (haplotag data)."""
    mock_flag = _should_use_mock()
    monkeypatch.setenv("MCP_IGV_MOCK", mock_flag)

    result = generate_igv_snapshots(
        genome="hg38",
        tracks=[str(sample_bam), str(sample_vcf)],
        regions=[
            {"chrom": "chr1", "start": 4140000, "end": 4145000, "name": "haplotag_wide"},
            {"chrom": "chr1", "start": 4142200, "end": 4143000, "name": "haplotag_zoomed"},
        ],
        output_dir=str(tmp_path),
    )

    assert result.execution_mode in {"docker", "apptainer"}
    assert len(result.snapshot_files) == 2
    for snap_file in result.snapshot_files:
        snapshot_path = Path(snap_file)
        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".png"
        _preserve_snapshot(snapshot_path, "bam_vcf")


@pytest.mark.igv_integration
def test_igv_snapshot_tool_mcp_protocol(
    mcp_server_params,
    sample_bam_highdepth: Path,
    sample_reference: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_flag = _should_use_mock()
    monkeypatch.setenv("MCP_IGV_MOCK", mock_flag)
    server_params = mcp_server_params.model_copy()
    # Pass relevant IGV env vars to the subprocess server
    base_env: dict[str, str] = dict(getattr(server_params, "env", None) or {})
    base_env.update(
        {
            "MCP_IGV_MOCK": mock_flag,
            "MCP_IGV_CONTAINER_IMAGE": os.getenv("MCP_IGV_CONTAINER_IMAGE", "aganezov/igv_snapper:0.2"),
        }
    )
    if (sif := os.getenv("MCP_IGV_SIF_PATH")):
        base_env["MCP_IGV_SIF_PATH"] = sif
    server_params.env = base_env

    async def _test():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "igv_snapshot_tool",
                    {
                        "genome": str(sample_reference),
                        "tracks": [str(sample_bam_highdepth)],
                        "regions": [{"chrom": "chrHD", "start": 1, "end": 200}],
                        "output_dir": str(tmp_path),
                    },
                )
                assert not result.isError
                payload = json.loads(_text_content(result.content[0]).text)
                assert payload["snapshot_files"]
                assert payload["execution_mode"] in {"docker", "apptainer"}
                for snap_file in payload["snapshot_files"]:
                    _preserve_snapshot(Path(snap_file), "mcp_protocol")

    anyio.run(_test)
