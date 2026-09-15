import gzip
import tempfile
from pathlib import Path

import pytest

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


def test_reject_directory_output_before_command(input_fastq, tmp_path, monkeypatch):
    before = set(tmp_path.iterdir())

    def unexpected_command(*args, **kwargs):
        pytest.fail("chopper must not run with a directory output")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", unexpected_command)
    with pytest.raises(ValueError, match="regular file"):
        chopper_filter(input_fastq, ToolPaths(), tmp_path)
    assert input_fastq.read_text() == FASTQ
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("crop", ["headcrop", "tailcrop"])
@pytest.mark.parametrize("mode", [None, "trim-by-quality"])
def test_reject_ineffective_crop_before_command(input_fastq, tmp_path, monkeypatch, crop, mode):
    output = tmp_path / "filtered.fastq"
    output.write_text("previous output")
    before = set(tmp_path.iterdir())
    flags = {crop: 50}
    if mode is not None:
        flags["trim_approach"] = mode

    def unexpected_command(*args, **kwargs):
        pytest.fail("Ineffective crop settings must fail before Chopper starts")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", unexpected_command)
    with pytest.raises(ValueError, match="fixed-crop"):
        chopper_filter(input_fastq, ToolPaths(), output, flags=flags)
    assert input_fastq.read_text() == FASTQ
    assert output.read_text() == "previous output"
    assert set(tmp_path.iterdir()) == before


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


@pytest.mark.parametrize("contents", [FASTQ, ""])
@pytest.mark.parametrize("destination", ["existing", "absent", "automatic", "symlink", "dash"])
def test_publish_complete_output(input_fastq, tmp_path, monkeypatch, contents, destination):
    output = tmp_path / "filtered.fastq.gz"
    target = output
    if destination == "symlink":
        target = tmp_path / "data.fastq"
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
        assert cmd[1] == "--input"
        assert "--output" not in cmd
        assert "--report-json" not in cmd
        if output is not None and target.exists():
            assert target.read_text() == "previous output"
        staged = Path(kwargs["stdout_path"])
        if output is not None:
            assert staged != target
            assert staged.parent == target.parent
            assert staged.name.endswith("".join(output.suffixes))
        staged.write_text(contents)

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run)
    report = chopper_filter(input_fastq, ToolPaths(), output)
    assert report.output_fastq is not None
    result = Path(report.output_fastq)
    if output is not None and output.suffix.lower() == ".gz":
        assert result.read_bytes()[:2] == b"\x1f\x8b"
        assert gzip.decompress(result.read_bytes()).decode() == contents
    else:
        assert result.read_text() == contents
    assert not {"input_reads", "output_reads", "filtered_reads"} & report.model_dump().keys()
    assert input_fastq.read_text() == FASTQ
    if output is not None:
        assert report.output_fastq == str(output)
    if destination == "symlink":
        assert output.is_symlink()
        assert gzip.decompress(target.read_bytes()).decode() == contents
    if destination in ("existing", "symlink"):
        assert target.stat().st_mode & 0o777 == 0o640
    assert set(tmp_path.iterdir()) == before | {result.resolve()}


def test_native_arguments_follow_typed_chopper_arguments(input_fastq, tmp_path, monkeypatch):
    output = tmp_path / "filtered.fastq"
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        Path(kwargs["stdout_path"]).write_text(FASTQ)

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run)
    report = chopper_filter(
        input_fastq,
        ToolPaths(),
        output,
        flags={"minlength": 4},
        extra_args=["--future-option=value"],
    )

    command = commands[0]
    assert command[command.index("--minlength") + 1] == "4"
    assert command[-1] == "--future-option=value"
    assert command.index("--minlength") < len(command) - 1
    assert report.command == commands[0]


@pytest.mark.parametrize(
    "destination,failure",
    [
        (destination, failure)
        for destination in ("existing", "absent", "automatic")
        for failure in ("command", "missing", "replace")
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
        Path(kwargs["stdout_path"]).write_text("partial output")
        if failure == "missing":
            raise FileNotFoundError("chopper")
        if failure == "replace":
            return
        raise command_error(cmd, "disk full")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_run)
    if failure == "replace":

        def failed_replace(*args):
            raise PermissionError("Cannot replace destination")

        monkeypatch.setattr("ont_qc_mcp.cli_wrappers.os.replace", failed_replace)
    expected = {
        "missing": FileNotFoundError,
        "replace": PermissionError,
    }
    with pytest.raises(expected.get(failure, RuntimeError)):
        chopper_filter(input_fastq, ToolPaths(), output)
    assert input_fastq.read_text() == FASTQ
    if destination == "existing":
        assert output.read_text() == "previous output"
    assert set(tmp_path.iterdir()) == before
