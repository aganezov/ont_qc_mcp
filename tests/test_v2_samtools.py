"""Samtools selection and conversion planning for later API v2 adapters."""

from pathlib import Path

import pytest

from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.v2_contracts import CoverageQCRequest
from ont_qc_mcp.v2_regions import normalize_regions
from ont_qc_mcp.v2_samtools import (
    SamtoolsSelection,
    build_fastq_stage,
    read_alignment_reference_lengths,
    resolve_alignment_input,
    resolve_and_normalize_regions,
    samtools_fastq_plan,
    samtools_view_plan,
)
from ont_qc_mcp.v2_execution import RequestDeadline
from ont_qc_mcp.utils import CommandResult


def alignment_files(tmp_path: Path, suffix: str = ".bam") -> tuple[Path, Path | None]:
    alignment = tmp_path / f"reads{suffix}"
    alignment.write_bytes(b"alignment")
    index_suffix = ".crai" if suffix == ".cram" else ".bai"
    Path(str(alignment) + index_suffix).write_bytes(b"index")
    if suffix != ".cram":
        return alignment, None
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\n" + "A" * 100 + "\n")
    Path(str(reference) + ".fai").write_text("chr1\t100\t6\t100\t101\n")
    return alignment, reference


def test_union_plan_uses_explicit_index_and_preserves_auxiliary_tags(tmp_path: Path) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=True, exec_cfg=ExecutionConfig())
    regions = normalize_regions(
        [
            {"chrom": "chr1", "start": 10, "end": 20, "name": "one"},
            {"chrom": "chr1", "start": 15, "end": 25, "name": "two"},
            {"chrom": "chr1", "start": 40, "end": 45, "name": "three"},
        ],
        {"chr1": 100},
    )
    cfg = ExecutionConfig(per_tool_threads={"samtools": 2})
    with samtools_view_plan(
        resolved,
        regions,
        SamtoolsSelection(min_mapq=10, exclude_flags=1796, include_unmapped=False),
        tools=ToolPaths(samtools="/samtools"),
        exec_cfg=cfg,
        native_args=["--subsample", "0.5"],
    ) as plan:
        assert plan.regional is True
        assert plan.required_tags_preserved is True
        assert plan.output_format == "uncompressed_bam"
        assert "-u" in plan.stage.command and "-h" in plan.stage.command
        assert "-x" not in plan.stage.command and "--remove-tag" not in plan.stage.command
        assert plan.stage.command[-2:] == (str(bam), str(bam) + ".bai")
        assert plan.stage.command[plan.stage.command.index("-F") + 1] == "1796"
        assert plan.stage.command[plan.stage.command.index("-q") + 1] == "10"
        assert plan.stage.command[plan.stage.command.index("-@") + 1] == "2"
        bed = Path(plan.stage.command[plan.stage.command.index("-L") + 1])
        assert bed.read_text() == "chr1\t10\t25\nchr1\t40\t45\n"
        assert plan.native_args.reuse_safe is False
    assert not bed.exists()


@pytest.mark.parametrize(
    ("include_unmapped", "exclude_flags", "effective"),
    [(False, 0, 4), (False, 2048, 2052), (True, 0, 0)],
)
def test_include_unmapped_is_independent_of_custom_exclusion_mask(
    tmp_path: Path, include_unmapped: bool, exclude_flags: int, effective: int
) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=False)
    with samtools_view_plan(
        resolved,
        normalize_regions(None, {"chr1": 100}),
        SamtoolsSelection(exclude_flags=exclude_flags, include_unmapped=include_unmapped),
        tools=ToolPaths(samtools="samtools"),
    ) as plan:
        assert plan.effective_exclude_flags == effective
        assert plan.stage.command[plan.stage.command.index("-F") + 1] == str(effective)


@pytest.mark.parametrize("min_mapq", [1, 20, 254])
@pytest.mark.parametrize(
    "native_expression",
    [["--expr=mapq >= 0"], ["--exp=mapq >= 0"], ["-emapq >= 0"]],
)
def test_positive_mapq_threshold_excludes_missing_255_and_protects_predicate(
    tmp_path: Path,
    min_mapq: int,
    native_expression: list[str],
) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=False)
    with samtools_view_plan(
        resolved,
        normalize_regions(None, {"chr1": 100}),
        SamtoolsSelection(min_mapq=min_mapq),
        tools=ToolPaths(samtools="samtools"),
    ) as plan:
        expression_index = plan.stage.command.index("-e")
        assert plan.stage.command[expression_index + 1] == "mapq != 255"
    with pytest.raises(ValueError, match="wrapper-owned"):
        with samtools_view_plan(
            resolved,
            normalize_regions(None, {"chr1": 100}),
            SamtoolsSelection(min_mapq=min_mapq),
            tools=ToolPaths(samtools="samtools"),
            native_args=native_expression,
        ):
            pass


def test_zero_mapq_threshold_retains_missing_255_and_allows_native_expression(tmp_path: Path) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=False)
    with samtools_view_plan(
        resolved,
        normalize_regions(None, {"chr1": 100}),
        SamtoolsSelection(min_mapq=0),
        tools=ToolPaths(samtools="samtools"),
        native_args=["-e", "mapq >= 0"],
    ) as plan:
        assert plan.stage.command.count("-e") == 1
        assert "mapq != 255" not in plan.stage.command


def test_primary_read_selection_and_fastq_conversion_apply_one_policy(tmp_path: Path) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=False)
    with samtools_view_plan(
        resolved,
        normalize_regions(None, {"chr1": 100}),
        SamtoolsSelection(primary_only=True, include_unmapped=True),
        tools=ToolPaths(samtools="/samtools"),
    ) as plan:
        assert plan.effective_exclude_flags == 0x900
        fastq = build_fastq_stage(ToolPaths(samtools="/samtools"), exec_cfg=ExecutionConfig())
        assert fastq.command[fastq.command.index("-F") + 1] == "0"
        assert fastq.command[-1] == "-"
        fastq_plan = samtools_fastq_plan(ToolPaths(samtools="/samtools"), native_args=["-n"])
        assert fastq_plan.native_args.effective_args[-2:] == ("-n", "-")
        assert fastq_plan.native_args.native_options_used is True
        assert fastq_plan.native_args.reuse_safe is False


def test_regional_selection_requires_index_and_cram_requires_reference(tmp_path: Path) -> None:
    bam, _ = alignment_files(tmp_path)
    Path(str(bam) + ".bai").unlink()
    with pytest.raises(FileNotFoundError, match="index"):
        resolve_alignment_input(str(bam), require_index=True)

    cram, reference = alignment_files(tmp_path, ".cram")
    with pytest.raises(ValueError, match="CRAM requires"):
        resolve_alignment_input(str(cram), require_index=True)
    assert reference is not None
    resolved = resolve_alignment_input(str(cram), reference_path=str(reference), require_index=True)
    assert resolved.reference == reference.resolve()
    assert resolved.reference_index == Path(str(reference) + ".fai").resolve()


def test_input_identity_change_is_detected(tmp_path: Path) -> None:
    bam, _ = alignment_files(tmp_path)
    resolved = resolve_alignment_input(str(bam), require_index=True)
    bam.write_bytes(b"changed alignment")
    with pytest.raises(RuntimeError, match="changed"):
        resolved.assert_unchanged()


def test_header_bounds_and_normalization_share_one_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_samtools

    bam, _ = alignment_files(tmp_path)
    timeouts: list[float] = []

    def header(command, consume, *, timeout, **kwargs):
        timeouts.append(timeout)
        consume("@SQ\tSN:chr1\tLN:100")
        return CommandResult(command, 0, "", "")

    monkeypatch.setattr(v2_samtools, "run_line_stream", header)
    deadline = RequestDeadline(5)
    alignment, regions = resolve_and_normalize_regions(
        str(bam),
        [{"chrom": "chr1", "start": 99, "end": 100}],
        reference_path=None,
        tools=ToolPaths(samtools="/samtools"),
        exec_cfg=ExecutionConfig(),
        deadline=deadline,
    )
    assert alignment.index == Path(str(bam) + ".bai")
    assert [(r.start, r.end) for r in regions.requested] == [(99, 100)]
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 5

    with pytest.raises(ValueError, match="exceeds reference length"):
        resolve_and_normalize_regions(
            str(bam),
            [{"chrom": "chr1", "start": 99, "end": 101}],
            reference_path=None,
            tools=ToolPaths(samtools="/samtools"),
            exec_cfg=ExecutionConfig(),
            deadline=RequestDeadline(5),
        )


def test_gff_normalization_observes_shared_request_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_samtools

    bam, _ = alignment_files(tmp_path)
    gff = tmp_path / "genes.gff3"
    gff.write_text(
        "##gff-version 3\nchr1\ttest\tgene\t1\t2\t.\t+\t.\tID=first\nchr1\ttest\tgene\t3\t4\t.\t+\t.\tID=target\n"
    )

    def header(command, consume, **kwargs):
        consume("@SQ\tSN:chr1\tLN:100")
        return CommandResult(command, 0, "", "")

    class ExpiringDeadline:
        def __init__(self) -> None:
            self.checkpoints = 0

        def remaining(self) -> float:
            return 5

        def checkpoint(self) -> None:
            self.checkpoints += 1
            if self.checkpoints >= 3:
                raise TimeoutError("request deadline exceeded")

    monkeypatch.setattr(v2_samtools, "run_line_stream", header)
    request = CoverageQCRequest.model_validate(
        {
            "path": str(bam),
            "regions": {"format": "gff3", "path": str(gff), "feature_type": "gene", "ids": ["target"]},
        }
    )
    with pytest.raises(TimeoutError, match="request deadline exceeded"):
        resolve_and_normalize_regions(
            str(bam),
            request.regions,
            reference_path=None,
            tools=ToolPaths(samtools="/samtools"),
            exec_cfg=ExecutionConfig(),
            deadline=ExpiringDeadline(),  # type: ignore[arg-type]
        )


def test_whole_file_normalization_does_not_require_reference_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ont_qc_mcp import v2_samtools

    bam = tmp_path / "unmapped-only.bam"
    bam.write_bytes(b"alignment")

    def unexpected_header_read(*args, **kwargs):
        raise AssertionError("whole-file normalization must not read reference lengths")

    monkeypatch.setattr(v2_samtools, "read_alignment_reference_lengths", unexpected_header_read)
    alignment, regions = resolve_and_normalize_regions(
        str(bam),
        None,
        reference_path=None,
        tools=ToolPaths(samtools="/samtools"),
        exec_cfg=ExecutionConfig(),
        deadline=RequestDeadline(5),
    )
    assert alignment.alignment == bam.resolve()
    assert regions.requested == regions.union == ()


def test_header_parser_rejects_missing_or_duplicate_reference_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ont_qc_mcp import v2_samtools

    bam, _ = alignment_files(tmp_path)
    alignment = resolve_alignment_input(str(bam), require_index=False)

    def duplicate(command, consume, **kwargs):
        consume("@SQ\tSN:chr1\tLN:100")
        consume("@SQ\tSN:chr1\tLN:100")
        return CommandResult(command, 0, "", "")

    monkeypatch.setattr(v2_samtools, "run_line_stream", duplicate)
    with pytest.raises(ValueError, match="repeats reference"):
        read_alignment_reference_lengths(alignment, ToolPaths(), ExecutionConfig(), RequestDeadline(5))


def test_header_parser_enforces_total_size_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_samtools

    bam, _ = alignment_files(tmp_path)
    alignment = resolve_alignment_input(str(bam), require_index=False)

    def oversized(command, consume, **kwargs):
        consume("@SQ\tSN:chr1\tLN:100")
        return CommandResult(command, 0, "", "")

    monkeypatch.setattr(v2_samtools, "MAX_HEADER_BYTES", 10)
    monkeypatch.setattr(v2_samtools, "run_line_stream", oversized)
    with pytest.raises(ValueError, match="16 MiB"):
        read_alignment_reference_lengths(alignment, ToolPaths(), ExecutionConfig(), RequestDeadline(5))


def test_reference_fai_must_match_alignment_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_samtools

    cram, reference = alignment_files(tmp_path, ".cram")
    assert reference is not None
    alignment = resolve_alignment_input(str(cram), reference_path=str(reference), require_index=True)

    def header(command, consume, **kwargs):
        consume("@SQ\tSN:chr1\tLN:100")
        return CommandResult(command, 0, "", "")

    monkeypatch.setattr(v2_samtools, "run_line_stream", header)
    Path(str(reference) + ".fai").write_text("chr1\t99\t6\t99\t100\n")
    with pytest.raises(ValueError, match="does not match alignment contig"):
        read_alignment_reference_lengths(alignment, ToolPaths(), ExecutionConfig(), RequestDeadline(5))
