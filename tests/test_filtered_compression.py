"""Filtered output encoding follows the supplied output filename."""

import gzip
import json
import tempfile
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from ont_qc_mcp.cli_wrappers import chopper_filter
from ont_qc_mcp.config import ToolPaths


SHORT_READ = "@short\nACGT\n+\nIIII\n"
LONG_READ = "@long\nACGTACGT\n+\nIIIIIIII\n"
FASTQ = SHORT_READ + LONG_READ


@pytest.mark.parametrize(
    "supplied,target,gzipped",
    [
        ("filtered.fastq.gz", None, True),
        ("filtered.fastq.GZ", None, True),
        (".gz", None, True),
        ("link.fastq.gz", "target.fastq", True),
        ("link.fastq", "target.fastq.gz", False),
        ("filtered.fastq", None, False),
        ("filtered.data", None, False),
    ],
)
def test_supplied_filename_selects_encoding(tmp_path, monkeypatch, supplied, target, gzipped):
    input_path = tmp_path / "reads.fastq"
    input_path.write_text(FASTQ)
    output = tmp_path / supplied
    if target:
        output.symlink_to(tmp_path / target)

    def fake_chopper(cmd, **kwargs):
        Path(kwargs["stdout_path"]).write_text(LONG_READ)

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_chopper)
    report = chopper_filter(input_path, ToolPaths(), output)
    assert report.output_fastq == str(output)
    raw = output.read_bytes()
    if gzipped:
        assert raw[:2] == b"\x1f\x8b"
        assert gzip.decompress(raw).decode() == LONG_READ
        # No timestamp or temporary filename leaks into gzip metadata.
        assert raw[3] & 8 == 0
        assert raw[4:8] == b"\0\0\0\0"
    else:
        assert raw.decode() == LONG_READ
    if target:
        assert output.is_symlink()
    assert input_path.read_text() == FASTQ


@pytest.mark.integration
@pytest.mark.parametrize("input_compressed", [False, True])
@pytest.mark.parametrize(
    "output_name,minlength,expected",
    [
        ("filtered.fastq.gz", 5, LONG_READ),
        ("filtered.fastq.GZ", 5, LONG_READ),
        ("empty.fastq.gz", 100, ""),
        ("filtered.fastq", 5, LONG_READ),
        (None, 5, LONG_READ),
    ],
)
def test_mcp_filter_encoding_roundtrip(mcp_server_params, tmp_path, input_compressed, output_name, minlength, expected):
    from conftest import require_executable_tools

    require_executable_tools(["chopper", "nanoq"])
    input_path = tmp_path / ("reads.fastq.gz" if input_compressed else "reads.fastq")
    input_path.write_bytes(gzip.compress(FASTQ.encode()) if input_compressed else FASTQ.encode())
    arguments = {"path": str(input_path), "flags": {"minlength": minlength}}
    if output_name is not None:
        arguments["output_fastq"] = str(tmp_path / output_name)

    async def check_output():
        async with stdio_client(mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("filter_reads_fastq_tool", arguments)
                assert not result.is_error, result.content
                content = result.content[0]
                assert isinstance(content, TextContent)
                report = json.loads(content.text)
                output = Path(report["output_fastq"])
                try:
                    encoded = output.read_bytes()
                    if output_name and output_name.lower().endswith(".gz"):
                        assert encoded[:2] == b"\x1f\x8b"
                        assert gzip.decompress(encoded).decode() == expected
                    else:
                        assert encoded.decode() == expected
                    if expected:
                        qc = await session.call_tool("qc_reads_fastq_tool", {"path": str(output)})
                        assert not qc.is_error, qc.content
                        qc_content = qc.content[0]
                        assert isinstance(qc_content, TextContent)
                        stats = json.loads(qc_content.text)
                        assert (stats["read_count"], stats["total_bases"]) == (1, 8)
                finally:
                    if output_name is None:
                        output.unlink(missing_ok=True)

    anyio.run(check_output)


@pytest.mark.parametrize(
    "suffix",
    [
        ".bgz",
        ".bgzf",
        ".bz",
        ".bz2",
        ".bzip2",
        ".xz",
        ".lzma",
        ".zst",
        ".zstd",
        ".lz4",
        ".zip",
        ".z",
        ".gzip",
        ".XZ",
        ".Z",
    ],
)
def test_unsupported_compression_rejected_before_acquisition(tmp_path, monkeypatch, suffix):
    input_path = tmp_path / "reads.fastq"
    input_path.write_text(FASTQ)
    output = tmp_path / f"filtered.fastq{suffix}"
    output.write_bytes(b"existing output")
    before = set(tmp_path.iterdir())

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported encoding must fail before a CLI or stage is acquired")

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", unexpected)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", unexpected)
    with pytest.raises(ValueError, match="Unsupported.*compression.*\\.gz"):
        chopper_filter(input_path, ToolPaths(), output)
    assert output.read_bytes() == b"existing output"
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "destination,failure",
    [
        (destination, failure)
        for destination in ("existing", "absent", "symlink")
        for failure in ("allocation", "copy", "finalization", "raw_cleanup", "chmod", "replace")
        if (destination, failure) != ("absent", "chmod")
    ],
)
def test_compression_failure_preserves_destination_and_cleans_stages(tmp_path, monkeypatch, failure, destination):
    input_path = tmp_path / "reads.fastq"
    input_path.write_text(FASTQ)
    output = tmp_path / "filtered.fastq.gz"
    target = tmp_path / "target.fastq" if destination == "symlink" else output
    if destination != "absent":
        target.write_bytes(b"previous output\0")
        target.chmod(0o640)
    if destination == "symlink":
        output.symlink_to(target)
    before = set(tmp_path.iterdir())
    allocated: list[Path] = []
    original_temp = tempfile.NamedTemporaryFile

    def tracked_temp(*args, **kwargs):
        if failure == "allocation" and allocated:
            raise OSError("compression stage allocation failed")
        result = original_temp(*args, **kwargs)
        allocated.append(Path(result.name))
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracked_temp)

    def fake_chopper(cmd, **kwargs):
        Path(kwargs["stdout_path"]).write_text(LONG_READ)

    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_chopper)
    if failure == "copy":
        original_write = gzip.GzipFile.write

        def failed_write(stream, data):
            original_write(stream, data[:8])
            raise OSError("compression copy failed")

        monkeypatch.setattr(gzip.GzipFile, "write", failed_write)
    elif failure == "finalization":
        original_close = gzip.GzipFile.close

        def failed_close(stream):
            was_open = stream.fileobj is not None
            original_close(stream)
            if was_open:
                raise OSError("compression finalization failed")

        monkeypatch.setattr(gzip.GzipFile, "close", failed_close)
    elif failure == "raw_cleanup":
        original_unlink = Path.unlink
        failed = False

        def failed_unlink(path, *args, **kwargs):
            nonlocal failed
            if allocated and path == allocated[0] and not failed:
                failed = True
                raise OSError("compression raw_cleanup failed")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failed_unlink)
    elif failure == "chmod":
        original_chmod = Path.chmod

        def failed_chmod(path, *args, **kwargs):
            if path in allocated:
                raise OSError("compression chmod failed")
            return original_chmod(path, *args, **kwargs)

        monkeypatch.setattr(Path, "chmod", failed_chmod)
    elif failure == "replace":

        def failed_replace(source, destination):
            # Publication receives a complete gzip stream, including its trailer.
            assert gzip.decompress(Path(source).read_bytes()).decode() == LONG_READ
            raise OSError("compression replace failed")

        monkeypatch.setattr("ont_qc_mcp.cli_wrappers.os.replace", failed_replace)

    with pytest.raises(OSError, match=f"compression.*{failure}"):
        chopper_filter(input_path, ToolPaths(), output)
    assert input_path.read_text() == FASTQ
    if destination != "absent":
        assert target.read_bytes() == b"previous output\0"
        assert target.stat().st_mode & 0o777 == 0o640
    if destination == "symlink":
        assert output.is_symlink()
    assert set(tmp_path.iterdir()) == before
    assert all(not path.exists() for path in allocated)


def test_cleanup_attempts_both_stages_without_masking_copy_error(tmp_path, monkeypatch, caplog):
    input_path = tmp_path / "reads.fastq"
    input_path.write_text(FASTQ)
    output = tmp_path / "filtered.fastq.gz"
    output.write_bytes(b"previous output")
    allocated: list[Path] = []
    original_temp = tempfile.NamedTemporaryFile
    original_unlink = Path.unlink

    def tracked_temp(*args, **kwargs):
        result = original_temp(*args, **kwargs)
        allocated.append(Path(result.name))
        return result

    def fake_chopper(cmd, **kwargs):
        Path(kwargs["stdout_path"]).write_text(LONG_READ)

    def failed_write(stream, data):
        raise OSError("original compression failure")

    def failed_raw_unlink(path, *args, **kwargs):
        if allocated and path == allocated[0]:
            raise PermissionError("raw cleanup failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracked_temp)
    monkeypatch.setattr("ont_qc_mcp.cli_wrappers.run_command_with_retry", fake_chopper)
    monkeypatch.setattr(gzip.GzipFile, "write", failed_write)
    monkeypatch.setattr(Path, "unlink", failed_raw_unlink)
    try:
        with pytest.raises(OSError, match="original compression failure"):
            chopper_filter(input_path, ToolPaths(), output)
        assert output.read_bytes() == b"previous output"
        assert len(allocated) == 2
        assert not allocated[1].exists()
        assert "raw cleanup failed" in caplog.text
    finally:
        for path in allocated:
            original_unlink(path, missing_ok=True)
