"""Strict API v2 adapters for the six supporting tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY

import pytest
from pydantic import ValidationError

from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.schemas import (
    BedQCReport,
    ChopperReport,
    EnvStatus,
    HeaderMetadata,
    IgvSnapshotResult,
    SequencingSummaryStats,
)
from ont_qc_mcp.v2_supporting_tools import (
    bed_qc,
    environment_status,
    filter_reads,
    header_info,
    igv_snapshots,
    run_summary,
)
from ont_qc_mcp.utils import CommandResult


def test_supporting_adapters_validate_requests_before_calling_existing_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_supporting_tools as module

    calls: list[tuple[str, object]] = []

    def fake_environment(tools):
        calls.append(("environment_status", tools))
        return EnvStatus(available={}, resolved_paths={}, missing=[])

    def fake_header(**kwargs):
        calls.append(("header_info", kwargs))
        return HeaderMetadata(file=kwargs["path"], format="vcf", raw_header="##fileformat=VCFv4.3")

    def fake_bed(path, **kwargs):
        calls.append(("bed_qc", path))
        return BedQCReport(file=path, total_intervals=0, valid_intervals=0, total_bases=0, is_valid=True)

    def fake_summary(path, **kwargs):
        calls.append(("run_summary", path))
        return SequencingSummaryStats(file=path, total_yield=0, total_reads=0)

    monkeypatch.setattr(module, "env_check", fake_environment)
    monkeypatch.setattr(module, "header_metadata_lookup", fake_header)
    monkeypatch.setattr(module, "qc_bed", fake_bed)
    monkeypatch.setattr(module, "sequencing_summary", fake_summary)

    tools = ToolPaths()
    assert environment_status({}, tools=tools).missing == []
    assert header_info({"path": "calls.vcf", "reference_path": "reference.fa"}, tools=tools).format == "vcf"
    assert bed_qc({"path": "targets.bed"}, tools=tools).is_valid
    assert run_summary({"path": "summary.txt"}, tools=tools).total_reads == 0
    assert calls[1][1] == {
        "path": "calls.vcf",
        "reference_path": "reference.fa",
        "tools": tools,
        "exec_cfg": ANY,
    }

    with pytest.raises(ValidationError):
        environment_status({"unexpected": True}, tools=tools)


def test_filter_reads_maps_typed_selection_and_preserves_native_argument_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ont_qc_mcp import v2_supporting_tools as module

    captured: dict[str, object] = {}

    def fake_filter(path, **kwargs):
        captured.update(path=path, **kwargs)
        return ChopperReport(command=["chopper"], output_fastq=kwargs["output_fastq"])

    monkeypatch.setattr(module, "filter_reads_core", fake_filter)
    result = filter_reads(
        {
            "path": "reads.fastq.gz",
            "output_fastq": "filtered.fastq.gz",
            "selection": {
                "headcrop": 10,
                "tailcrop": 5,
                "minlength": 100,
                "trim_approach": "fixed-crop",
                "threads": 2,
            },
            "extra_args": {"chopper": ["--future-option=value"]},
        }
    )

    assert captured["path"] == "reads.fastq.gz"
    assert captured["flags"] == {
        "headcrop": 10,
        "tailcrop": 5,
        "minlength": 100,
        "trim_approach": "fixed-crop",
        "inverse": False,
        "threads": 2,
    }
    assert captured["extra_args"] == ["--future-option=value"]
    assert result.params == {
        "selection": captured["flags"],
        "extra_args": {"chopper": ["--future-option=value"]},
    }


def test_igv_snapshots_preserves_dynamic_and_batch_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_supporting_tools as module

    calls: list[dict[str, object]] = []

    def fake_snapshots(**kwargs):
        calls.append(kwargs)
        return IgvSnapshotResult(
            snapshot_files=["snapshot.png"],
            batch_file="generated.batch",
            output_directory="snapshots",
            execution_mode="docker",
            command=["igv"],
        )

    monkeypatch.setattr(module, "generate_igv_snapshots", fake_snapshots)
    dynamic = igv_snapshots(
        {
            "genome": "reference.fa",
            "tracks": ["reads.bam"],
            "regions": [
                {
                    "chrom": "chr1",
                    "start": 0,
                    "end": 10,
                    "name": "target",
                    "extra_commands": ["sort BASE"],
                }
            ],
            "extra_commands": ["viewaspairs"],
            "compact": "collapse",
        }
    )
    batch = igv_snapshots({"batch_file": "prepared.batch", "output_dir": "snapshots"})

    assert dynamic.snapshot_files == ["snapshot.png"] and batch.snapshot_files == ["snapshot.png"]
    assert calls[0]["regions"] == [
        {
            "chrom": "chr1",
            "start": 0,
            "end": 10,
            "name": "target",
            "extra_commands": ["sort BASE"],
        }
    ]
    assert calls[0]["extra_commands"] == ["viewaspairs"]
    assert calls[0]["regions_are_zero_based_half_open"] is True
    assert calls[1]["batch_file"] == "prepared.batch"
    assert calls[1]["tracks"] is None and calls[1]["regions"] is None


def test_pure_python_supporting_tools_keep_existing_results(tmp_path: Path) -> None:
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t5\ttarget\n")
    summary = tmp_path / "summary.txt"
    summary.write_text("sequence_length_template\tstart_time\n4\t3600\n8\t3600\n")
    vcf = tmp_path / "calls.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=chr1,length=10>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )

    bed_result = bed_qc({"path": str(bed)})
    summary_result = run_summary({"path": str(summary)})
    header_result = header_info({"path": str(vcf)})

    assert bed_result.model_dump() == {
        "file": str(bed),
        "total_intervals": 1,
        "valid_intervals": 1,
        "total_bases": 5,
        "is_valid": True,
        "issues": [],
    }
    assert summary_result.total_reads == 2 and summary_result.total_yield == 12
    assert summary_result.yield_per_hour[0].window_start_hours == 1
    assert header_result.references[0].name == "chr1" and header_result.references[0].length == 10


def test_header_info_passes_reference_to_samtools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import tools as core_tools

    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"BAM")
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nACGT\n")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        stdout = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:4\n" if "view" in command else ""
        return CommandResult(command, 0, stdout, "")

    monkeypatch.setattr(core_tools, "run_command", fake_run)
    result = header_info(
        {"path": str(bam), "reference_path": str(reference)},
        tools=ToolPaths(samtools="samtools"),
    )

    view_command = next(command for command in commands if "view" in command)
    assert view_command[view_command.index("-T") + 1] == str(reference.resolve())
    assert result.references[0].name == "chr1" and result.references[0].length == 4


@pytest.mark.parametrize("region_source", ["list", "bed"])
def test_igv_v2_regions_become_one_based_in_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    region_source: str,
) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nA\n")
    track = tmp_path / "reads.bam"
    track.write_bytes(b"BAM")
    output = tmp_path / region_source
    if region_source == "bed":
        bed = tmp_path / "targets.bed"
        bed.write_text("chr1\t0\t1\ttarget\n")
        regions: object = str(bed)
    else:
        regions = [{"chrom": "chr1", "start": 0, "end": 1, "name": "target"}]

    monkeypatch.setenv("MCP_IGV_MOCK", "1")
    result = igv_snapshots(
        {
            "genome": str(reference),
            "tracks": [str(track)],
            "regions": regions,
            "output_dir": str(output),
        }
    )

    assert "goto chr1:1-1\n" in Path(result.batch_file).read_text()
