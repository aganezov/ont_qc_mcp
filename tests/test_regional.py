"""Regional orchestration contracts: strict inputs, indexed access and complete results."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ont_qc_mcp import app_server, regional
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.utils import CommandError, CommandResult

REGIONS = [{"chrom": "chr1", "start": 0, "end": 2, "name": "first"}]


@pytest.fixture
def fake_region(tmp_path, monkeypatch):
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"alignment")
    Path(str(bam) + ".bai").write_bytes(b"index")
    commands = []
    beds = []

    def run(cmd, consume, **kwargs):
        commands.append((cmd, kwargs))
        if "--version" in cmd:
            consume("samtools 1.24")
        elif "-H" in cmd:
            consume("@SQ\tSN:chr1\tLN:100")
        else:
            bed = Path(cmd[cmd.index("-L") + 1])
            beds.append(bed)
            assert bed.read_text() == "chr1\t0\t2\n"
            consume("r\t0\tchr1\t1\t20\t2M\t*\t0\t0\tAC\t!I")
        return CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(regional, "run_line_stream", run)
    monkeypatch.setattr(regional.tempfile, "tempdir", str(tmp_path))
    return bam, commands, beds


def test_indexed_result_identity_and_cleanup(fake_region):
    bam, commands, beds = fake_region
    cfg = ExecutionConfig(per_tool_threads={"samtools": 2}, per_tool_timeouts={"samtools": 7})
    result = regional.regional_alignment_stats(str(bam), REGIONS, tools=ToolPaths(samtools="custom"), exec_cfg=cfg)
    assert result["complete"] is True
    assert result["input"]["path"] == str(bam)
    assert result["index"]["path"] == str(bam) + ".bai"
    assert result["regions"][0]["mean_base_quality"] == 20
    assert result["regions"][0]["quality_known_bases"] == 2
    assert result["regions"][0]["mean_mapq"] == 20
    assert result["selection"]["exclude_flags"] == 1796
    assert len(commands) == 3
    cmd, kwargs = commands[-1]
    assert "-M" in cmd and "-X" in cmd
    assert cmd[cmd.index("-F") + 1] == "1796"
    assert cmd[cmd.index("-q") + 1] == "0"
    assert cmd[-2:] == [str(bam), str(bam) + ".bai"]
    assert cmd[cmd.index("-@") + 1] == "2"
    assert 0 < kwargs["timeout"] <= 7
    assert not any(p.exists() for p in beds)


@pytest.mark.parametrize(
    "changes",
    [
        {"regions": []},
        {"regions": REGIONS * 1025},
        {"regions": [{"chrom": "chr1", "start": True, "end": 2}]},
        {"regions": [{"chrom": "chr1", "start": 0.0, "end": 2}]},
        {"regions": [{"chrom": "chr1", "start": 2, "end": 2}]},
        {"regions": [{"chrom": "chr1\nchr2", "start": 0, "end": 2}]},
        {"exclude_flags": True},
        {"min_mapq": True},
        {"min_mapq": 255},
        {"exclude_flags": -1},
    ],
)
def test_invalid_request_before_filesystem(monkeypatch, changes):
    validate = Mock(side_effect=AssertionError("filesystem reached"))
    monkeypatch.setattr(regional, "_validate_input_file", validate)
    with pytest.raises(ValueError):
        regional.regional_alignment_stats(**{"path": "absent.bam", "regions": REGIONS, **changes})
    validate.assert_not_called()


@pytest.mark.parametrize(
    "region,message",
    [
        ({"chrom": "missing", "start": 0, "end": 1}, "absent"),
        ({"chrom": "chr1", "start": 99, "end": 101}, "exceeds"),
    ],
)
def test_reference_bounds_before_scan(fake_region, region, message):
    bam, commands, beds = fake_region
    with pytest.raises(ValueError, match=message):
        regional.regional_alignment_stats(str(bam), [region])
    assert len(commands) == 2 and not beds


@pytest.mark.parametrize(
    "failure", [ValueError("bad SAM"), CommandError(CommandResult([], 2, "", "failed")), asyncio.CancelledError()]
)
def test_failure_never_returns_partial_and_cleans_bed(fake_region, monkeypatch, failure):
    bam, _commands, beds = fake_region
    original = regional.run_line_stream

    def failing(cmd, consume, **kwargs):
        result = original(cmd, consume, **kwargs)
        if "-M" in cmd:
            raise failure
        return result

    monkeypatch.setattr(regional, "run_line_stream", failing)
    with pytest.raises(type(failure)):
        regional.regional_alignment_stats(str(bam), REGIONS)
    assert beds and not any(p.exists() for p in beds)


def test_input_changes_invalidate_result(fake_region, monkeypatch):
    bam, _commands, _beds = fake_region
    original = regional.run_line_stream

    def mutate(cmd, consume, **kwargs):
        result = original(cmd, consume, **kwargs)
        if "-M" in cmd:
            bam.write_bytes(b"different alignment")
        return result

    monkeypatch.setattr(regional, "run_line_stream", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        regional.regional_alignment_stats(str(bam), REGIONS)


def test_missing_index_never_runs_command(fake_region, monkeypatch):
    bam, _commands, _beds = fake_region
    Path(str(bam) + ".bai").unlink()
    runner = Mock()
    monkeypatch.setattr(regional, "run_line_stream", runner)
    with pytest.raises(FileNotFoundError, match="index"):
        regional.regional_alignment_stats(str(bam), REGIONS)
    runner.assert_not_called()
    assert not Path(str(bam) + ".bai").exists()


@pytest.mark.asyncio
async def test_adapter_retains_explicit_configuration_and_result(monkeypatch):
    result = {"complete": True, "regions": [{"mean_base_quality": 20}]}
    runner = Mock(return_value=result)
    monkeypatch.setattr(app_server, "regional_alignment_stats", runner)
    content = await app_server.regional_alignment_stats_tool(
        "reads.bam", REGIONS, min_mapq=10, exclude_flags=3844, reference_path="ref.fa"
    )
    payload = json.loads(content[0].text)
    assert payload["complete"] is True and payload["regions"] == result["regions"]
    assert runner.call_args.kwargs["exec_cfg"] is app_server.EXEC_CFG
    assert runner.call_args.kwargs["min_mapq"] == 10
    assert runner.call_args.kwargs["exclude_flags"] == 3844
    assert runner.call_args.kwargs["reference_path"] == "ref.fa"
    assert payload["provenance"]["effective_timeout"] == app_server.EXEC_CFG.timeout_for("samtools")


def test_disguised_compressed_reference_rejected_before_commands(fake_region, monkeypatch):
    bam, _commands, _beds = fake_region
    reference = bam.parent / "compressed.fa"
    reference.write_bytes(b"\x1f\x8b\x08\x00compressed")
    Path(str(reference) + ".fai").write_text("chr1\t100\t6\t100\t101\n")
    runner = Mock()
    monkeypatch.setattr(regional, "run_line_stream", runner)
    with pytest.raises(ValueError, match="uncompressed FASTA"):
        regional.regional_alignment_stats(str(bam), REGIONS, reference_path=str(reference))
    runner.assert_not_called()


@pytest.mark.parametrize("mask,expected", [(2048, 2052), (0, 4)])
def test_requested_filters_reach_samtools(fake_region, mask, expected):
    bam, commands, _beds = fake_region
    result = regional.regional_alignment_stats(str(bam), REGIONS, exclude_flags=mask, min_mapq=30)
    cmd, _kwargs = commands[-1]
    assert cmd[cmd.index("-F") + 1] == str(expected)  # mapped-only eligibility is unconditional
    assert cmd[cmd.index("-q") + 1] == "30"
    assert result["selection"]["exclude_flags"] == mask
    assert result["execution"]["samtools_filters"] == {"exclude_flags": expected, "min_mapq": 30}
    assert result["regions"][0]["span_overlapping_alignments"] == 0


def test_missing_package_metadata_does_not_discard_measurement(fake_region, monkeypatch):
    bam, _commands, _beds = fake_region
    monkeypatch.setattr(regional, "version", Mock(side_effect=regional.PackageNotFoundError()))
    report = regional.regional_alignment_stats(str(bam), REGIONS)
    assert report["complete"] is True
    assert report["regions"][0]["mean_base_quality"] == 20
    assert report["execution"]["server_version"] is None
