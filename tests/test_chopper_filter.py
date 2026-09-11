import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from ont_qc_mcp.cli_wrappers import chopper_filter
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.utils import CommandError, CommandResult


FASTQ = "@r1\nACGT\n+\nIIII\n"


def command_error(cmd, message, code=1):
    return CommandError(CommandResult(cmd=cmd, returncode=code, stdout="", stderr=message))


@pytest.fixture
def input_fastq(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = tmp_path / "reads.fastq"
    path.write_text(FASTQ)
    return path


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_reject_input_output_alias_before_command(input_fastq, tmp_path, monkeypatch, alias):
    output = input_fastq
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("reads.fastq")
    elif alias == "symlink":
        output = tmp_path / "link.fastq"
        output.symlink_to(input_fastq)
    elif alias == "hardlink":
        output = tmp_path / "link.fastq"
        output.hardlink_to(input_fastq)
    before = set(tmp_path.iterdir())

    def unexpected_command(*args, **kwargs):
        pytest.fail("chopper must not run when the input and output alias")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", unexpected_command)
    with pytest.raises(ValueError, match="same file"):
        chopper_filter(input_fastq, ToolPaths(), output)
    assert input_fastq.read_text() == FASTQ
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("contents", [FASTQ, ""])
@pytest.mark.parametrize("destination", ["existing", "absent", "automatic", "symlink", "dash"])
def test_publish_complete_output(input_fastq, tmp_path, monkeypatch, legacy, contents, destination):
    output = tmp_path / "filtered.fastq.gz"
    target = output
    if destination in ("existing", "symlink"):
        target.write_text("previous output")
        target.chmod(0o640)
    if destination == "symlink":
        output = tmp_path / "link.fastq.gz"
        output.symlink_to(target)
    elif destination == "automatic":
        output = None
    elif destination == "dash":
        monkeypatch.chdir(tmp_path)
        output = Path("-filtered.fastq")
        target = tmp_path / output
    before = set(tmp_path.iterdir())

    def fake_run(cmd, **kwargs):
        if output is not None and target.exists():
            assert target.read_text() == "previous output"
        if "filter" in cmd:
            if legacy:
                raise command_error(cmd, "unexpected argument 'filter'", 2)
            staged = Path(cmd[cmd.index("--output") + 1])
            Path(cmd[cmd.index("--report-json") + 1]).write_text(json.dumps({"reads": {"input": 1}}))
        else:
            staged = Path(kwargs["stdout_path"])
        if output is not None:
            assert staged != target
            assert staged.parent == target.parent
            assert staged.name.endswith("".join(target.suffixes))
        staged.write_text(contents)

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run)
    report = chopper_filter(input_fastq, ToolPaths(), output)
    assert report.output_fastq is not None
    result = Path(report.output_fastq)
    assert result.read_text() == contents
    assert report.input_reads == (None if legacy else 1)
    assert input_fastq.read_text() == FASTQ
    if output is not None:
        assert report.output_fastq == str(output)
    if destination == "symlink":
        assert output.is_symlink()
        assert target.read_text() == contents
    if destination in ("existing", "symlink"):
        assert target.stat().st_mode & 0o777 == 0o640
    assert set(tmp_path.iterdir()) == before | {result.resolve()}


@pytest.mark.parametrize(
    "destination,failure",
    [
        (destination, failure)
        for destination in ("existing", "absent", "automatic")
        for failure in ("preferred", "legacy", "missing", "json", "report", "replace")
        if (destination, failure) != ("automatic", "replace")
    ],
)
def test_failure_preserves_files_and_cleans_temps(input_fastq, tmp_path, monkeypatch, destination, failure):
    output = tmp_path / "filtered.fastq"
    if destination == "existing":
        output.write_text("previous output")
    elif destination == "automatic":
        output = None
    before = set(tmp_path.iterdir())

    def fake_run(cmd, **kwargs):
        if "filter" in cmd:
            staged = Path(cmd[cmd.index("--output") + 1])
            staged.write_text("partial output")
            if failure == "legacy":
                raise command_error(cmd, "unexpected argument 'filter'", 2)
            if failure == "missing":
                raise FileNotFoundError("chopper")
            if failure in ("json", "report", "replace"):
                report = {
                    "json": "invalid JSON",
                    "report": '{"reads": {"input": "invalid count"}}',
                    "replace": "{}",
                }[failure]
                Path(cmd[cmd.index("--report-json") + 1]).write_text(report)
                return
        else:
            Path(kwargs["stdout_path"]).write_text("partial legacy output")
        raise command_error(cmd, "disk full")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run)
    if failure == "replace":

        def failed_replace(*args):
            raise PermissionError("Cannot replace destination")

        monkeypatch.setattr("ont_qc_mcp.cli_wrappers.os.replace", failed_replace)
    expected = {
        "missing": FileNotFoundError,
        "json": json.JSONDecodeError,
        "report": ValidationError,
        "replace": PermissionError,
    }
    with pytest.raises(expected.get(failure, RuntimeError)):
        chopper_filter(input_fastq, ToolPaths(), output)
    assert input_fastq.read_text() == FASTQ
    if destination == "existing":
        assert output.read_text() == "previous output"
    assert set(tmp_path.iterdir()) == before
