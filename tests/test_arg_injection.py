"""Regression tests for CLI argument-injection hardening.

An MCP client supplies the file paths we hand to wrapped tools. A path whose
string form starts with ``-`` (e.g. ``-rf.bam`` or ``--reference=/etc/passwd``)
would otherwise be parsed by the tool as an *option* rather than a *file*. The
``safe_path_arg`` helper neutralizes this by prefixing such paths with ``./``;
these tests pin both the helper's logic and its application at the positional
call sites that pass an untrusted path as a bare argument.
"""

import shutil
from pathlib import Path

import pytest

from ont_qc_mcp.cli_wrappers import (
    chopper_filter,
    cramino_stats,
    mosdepth_coverage,
    run_bcftools_stats,
    run_mosdepth_targeted,
    run_samtools_bedcov,
)
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.tools import (
    _maybe_quickcheck,
    _read_alignment_header_text,
    alignment_error_profile,
)
from ont_qc_mcp.utils import safe_path_arg


# --------------------------------------------------------------------------- #
# Helper logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("-evil.bam", "./-evil.bam"),  # short-option-looking name
        ("--reference=/etc/passwd", "./--reference=/etc/passwd"),  # long option w/ value
        ("-", "./-"),  # bare dash (stdin sentinel for many tools)
        ("normal.bam", "normal.bam"),  # plain relative path — untouched
        ("sub/dir/-weird.bam", "sub/dir/-weird.bam"),  # dash not leading — untouched
        ("/abs/-weird.bam", "/abs/-weird.bam"),  # absolute — never starts with '-'
        ("./already.bam", "./already.bam"),  # already explicit-relative — untouched
    ],
)
def testsafe_path_arg_neutralizes_only_leading_dash(raw: str, expected: str) -> None:
    result = safe_path_arg(raw)
    assert result == expected
    # Must return a str, not a Path: Path("./-x") collapses back to "-x", which
    # would re-expose the leading dash. The string form is load-bearing.
    assert isinstance(result, str)


def testsafe_path_arg_accepts_path_objects() -> None:
    # str(Path(...)) preserves a leading dash, so Path inputs are handled too.
    assert safe_path_arg(Path("-evil.bam")) == "./-evil.bam"
    assert safe_path_arg(Path("/data/sample.bam")) == "/data/sample.bam"


# --------------------------------------------------------------------------- #
# Application at the positional call sites
# --------------------------------------------------------------------------- #
class _StopRun(Exception):
    """Raised by the fake ``run_command`` to abort a wrapper immediately after
    the command list is assembled, so the test can inspect it without executing
    a real tool."""


def _capture_run_command(
    monkeypatch: pytest.MonkeyPatch, target: str = "ont_qc_mcp.cli_wrappers.run_command"
) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], *args: object, **kwargs: object) -> None:
        captured["cmd"] = list(cmd)
        raise _StopRun

    monkeypatch.setattr(target, fake_run)
    return captured


def test_cramino_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_command(monkeypatch)
    with pytest.raises(_StopRun):
        cramino_stats(Path("-rf.bam"), ToolPaths(), include_hist=False)
    cmd = captured["cmd"]
    assert cmd[-1] == "./-rf.bam"
    assert "-rf.bam" not in cmd  # never a bare token a tool could read as a flag


def test_mosdepth_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_command(monkeypatch)
    with pytest.raises(_StopRun):
        mosdepth_coverage(Path("-rf.bam"), ToolPaths())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-rf.bam"
    assert "-rf.bam" not in cmd
    # The output prefix is an internal tempdir path and must stay untouched.
    assert cmd[-2].endswith("/mosdepth")


def test_bcftools_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_command(monkeypatch)
    with pytest.raises(_StopRun):
        run_bcftools_stats(Path("-rf.vcf"), ToolPaths())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-rf.vcf"
    assert "-rf.vcf" not in cmd


def test_samtools_bedcov_neutralizes_dash_leading_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_command(monkeypatch)
    with pytest.raises(_StopRun):
        run_samtools_bedcov(Path("-rf.bam"), Path("-Qjunk.bed"), ToolPaths())
    cmd = captured["cmd"]
    # Both positional paths are client-supplied; "-Qjunk.bed" would otherwise
    # inject samtools' -Q quality-filter flag.
    assert "./-rf.bam" in cmd
    assert "./-Qjunk.bed" in cmd
    assert "-rf.bam" not in cmd
    assert "-Qjunk.bed" not in cmd


def test_mosdepth_targeted_neutralizes_dash_leading_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_run_command(monkeypatch)
    with pytest.raises(_StopRun):
        run_mosdepth_targeted(Path("-rf.bam"), Path("-Qjunk.bed"), ToolPaths())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-rf.bam"  # positional BAM
    assert "./-Qjunk.bed" in cmd  # --by value (bound, neutralized for consistency)
    assert "-rf.bam" not in cmd
    assert "-Qjunk.bed" not in cmd
    assert cmd[-2].endswith("/coverage")  # internal mkdtemp prefix stays untouched
    shutil.rmtree(Path(cmd[-2]).parent, ignore_errors=True)  # clean the mkdtemp dir


def test_chopper_neutralizes_dash_leading_output_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # chopper's preferred path uses run_command_with_retry, and output_fastq is an
    # MCP-tool input (exposed by app_server), so the output path needs neutralizing too.
    captured: dict[str, list[str]] = {}

    def fake_retry(cmd: list[str], *args: object, **kwargs: object) -> None:
        captured["cmd"] = list(cmd)
        raise _StopRun

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_retry)
    with pytest.raises(_StopRun):
        chopper_filter(Path("reads.fastq"), ToolPaths(), output_fastq=Path("-evil.fastq"))
    cmd = captured["cmd"]
    assert "./-evil.fastq" in cmd  # --output value neutralized
    assert "-evil.fastq" not in cmd
    Path(cmd[cmd.index("--report-json") + 1]).unlink(missing_ok=True)  # clean json temp


# --------------------------------------------------------------------------- #
# tools.py builds some samtools commands directly (not via cli_wrappers), so the
# shared guard must cover those MCP-exposed sinks too.
# --------------------------------------------------------------------------- #
def test_samtools_stats_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # alignment_error_profile_tool — samtools stats <BAM>; validation resolves the
    # path but the command receives the raw token, so the guard must apply here.
    monkeypatch.setattr("ont_qc_mcp.tools._validate_input_file", lambda *a, **k: None)
    captured = _capture_run_command(monkeypatch, "ont_qc_mcp.tools.run_command")
    with pytest.raises(_StopRun):
        alignment_error_profile("-x.bam", ToolPaths())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-x.bam"
    assert "-x.bam" not in cmd


def test_samtools_quickcheck_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # header_metadata_tool -> _maybe_quickcheck — samtools quickcheck -v <BAM>.
    monkeypatch.setattr("ont_qc_mcp.tools.shutil.which", lambda _x: "/usr/bin/samtools")
    captured = _capture_run_command(monkeypatch, "ont_qc_mcp.tools.run_command")
    with pytest.raises(_StopRun):
        _maybe_quickcheck(Path("-x.bam"), ToolPaths(), "bam", ExecutionConfig())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-x.bam"
    assert "-x.bam" not in cmd


def test_samtools_view_header_neutralizes_dash_leading_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # header_metadata_tool -> _read_alignment_header_text — samtools view -H <BAM>.
    captured = _capture_run_command(monkeypatch, "ont_qc_mcp.tools.run_command")
    with pytest.raises(_StopRun):
        _read_alignment_header_text(Path("-x.bam"), ToolPaths(), None, ExecutionConfig())
    cmd = captured["cmd"]
    assert cmd[-1] == "./-x.bam"
    assert "-x.bam" not in cmd
