"""API v2 indexed-mosdepth coverage planning and exact arithmetic."""

from __future__ import annotations

import asyncio
import gzip
from dataclasses import dataclass
from pathlib import Path

import pytest

from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.regional_metrics import RegionalInterval
from ont_qc_mcp.v2_contracts import NormalizedInterval
from ont_qc_mcp.v2_coverage_qc import _plan_rows, _read_per_base_depths, _read_threshold_counts, coverage_qc
from ont_qc_mcp.v2_execution import PipelineResult, RequestDeadline
from ont_qc_mcp.v2_regions import NormalizedRegionSet
from ont_qc_mcp.utils import CommandResult


REFERENCE_LENGTHS = {"chr1": 10, "chr2": 5}
REGIONS = NormalizedRegionSet(
    requested=(
        NormalizedInterval(chrom="chr1", start=0, end=5, name="first", region_id="region_1"),
        NormalizedInterval(chrom="chr1", start=3, end=8, name="overlap", region_id="region_2"),
        NormalizedInterval(chrom="chr2", start=2, end=5, name="last", region_id="region_3"),
    ),
    union=(
        RegionalInterval(chrom="chr1", start=0, end=8),
        RegionalInterval(chrom="chr2", start=2, end=5),
    ),
)


def test_all_four_row_plans_preserve_requested_identity_and_partial_windows() -> None:
    whole = _plan_rows(REFERENCE_LENGTHS, NormalizedRegionSet((), ()), None)
    assert [(row.row_id, row.chrom, row.start, row.end, row.name) for row in whole] == [
        ("contig.chr1", "chr1", 0, 10, None),
        ("contig.chr2", "chr2", 0, 5, None),
    ]

    requested = _plan_rows(REFERENCE_LENGTHS, REGIONS, None)
    assert [(row.row_id, row.chrom, row.start, row.end, row.name) for row in requested] == [
        ("region_1", "chr1", 0, 5, "first"),
        ("region_2", "chr1", 3, 8, "overlap"),
        ("region_3", "chr2", 2, 5, "last"),
    ]

    windows = _plan_rows(REFERENCE_LENGTHS, NormalizedRegionSet((), ()), 4)
    assert [(row.row_id, row.chrom, row.start, row.end) for row in windows] == [
        ("chr1.window_1", "chr1", 0, 4),
        ("chr1.window_2", "chr1", 4, 8),
        ("chr1.window_3", "chr1", 8, 10),
        ("chr2.window_1", "chr2", 0, 4),
        ("chr2.window_2", "chr2", 4, 5),
    ]

    region_windows = _plan_rows(REFERENCE_LENGTHS, REGIONS, 4)
    assert [(row.row_id, row.chrom, row.start, row.end, row.name) for row in region_windows] == [
        ("region_1.window_1", "chr1", 0, 4, "first"),
        ("region_1.window_2", "chr1", 4, 5, "first"),
        ("region_2.window_1", "chr1", 3, 7, "overlap"),
        ("region_2.window_2", "chr1", 7, 8, "overlap"),
        ("region_3.window_1", "chr2", 2, 5, "last"),
    ]


def test_window_plan_rejects_an_unsafe_projected_row_count_before_materialization(monkeypatch) -> None:
    from ont_qc_mcp import v2_coverage_qc as coverage

    monkeypatch.setattr(coverage, "MAX_COVERAGE_ROWS", 2)
    no_regions = NormalizedRegionSet(requested=(), union=(), external_dependencies=())

    with pytest.raises(ValueError, match="projected coverage row count 3 exceeds the limit of 2"):
        _plan_rows({"chr1": 3}, no_regions, 1)


def test_exact_depth_sums_use_integer_per_base_evidence_and_union(tmp_path) -> None:
    per_base = tmp_path / "coverage.per-base.bed.gz"
    with gzip.open(per_base, "wt") as output:
        output.write("chr1\t0\t2\t2\nchr1\t2\t4\t1\nchr1\t4\t10\t0\nchr2\t0\t5\t0\n")

    rows = _plan_rows(REFERENCE_LENGTHS, REGIONS, None)
    row_sums, union_sum = _read_per_base_depths(per_base, REFERENCE_LENGTHS, rows, REGIONS.union)

    # The first two rows overlap, so their sums are 6 and 1 while the chr1 union is 6.
    assert row_sums == [6, 1, 0]
    assert union_sum == 6


def test_per_base_output_must_cover_every_reference_base(tmp_path) -> None:
    per_base = tmp_path / "truncated.per-base.bed.gz"
    with gzip.open(per_base, "wt") as output:
        output.write("chr1\t0\t10\t0\n")
    rows = _plan_rows(REFERENCE_LENGTHS, NormalizedRegionSet((), ()), None)
    union = tuple(RegionalInterval(chrom=chrom, start=0, end=length) for chrom, length in REFERENCE_LENGTHS.items())

    with pytest.raises(ValueError, match="incomplete for contig.*chr2"):
        _read_per_base_depths(per_base, REFERENCE_LENGTHS, rows, union)


def test_threshold_counts_follow_internal_row_ids_not_native_output_order(tmp_path) -> None:
    thresholds_path = tmp_path / "coverage.thresholds.bed.gz"
    with gzip.open(thresholds_path, "wt") as output:
        output.write("#chrom\tstart\tend\tregion\t1X\t10X\t20X\n")
        output.write("chr2\t2\t5\tregion_3\t0\t0\t0\n")
        output.write("chr1\t3\t8\tregion_2\t1\t0\t0\n")
        output.write("chr1\t0\t5\tregion_1\t4\t0\t0\n")

    rows = _plan_rows(REFERENCE_LENGTHS, REGIONS, None)
    counts = _read_threshold_counts(thresholds_path, rows, [1, 10, 20])

    assert counts == [[4, 0, 0], [1, 0, 0], [0, 0, 0]]


def test_threshold_zero_uses_reference_domain_when_mosdepth_reports_empty_contig_as_zero(tmp_path) -> None:
    thresholds_path = tmp_path / "coverage.thresholds.bed.gz"
    with gzip.open(thresholds_path, "wt") as output:
        output.write("#chrom\tstart\tend\tregion\t0X\t1X\n")
        output.write("chr2\t0\t5\tcontig.chr2\t0\t0\n")

    rows = _plan_rows({"chr2": 5}, NormalizedRegionSet(requested=(), union=(), external_dependencies=()), None)
    counts = _read_threshold_counts(thresholds_path, rows, [0, 1])

    assert counts == [[5, 0]]


def test_threshold_output_must_contain_each_planned_occurrence(tmp_path) -> None:
    thresholds_path = tmp_path / "truncated.thresholds.bed.gz"
    with gzip.open(thresholds_path, "wt") as output:
        output.write("#chrom\tstart\tend\tregion\t1X\n")
        output.write("chr1\t0\t5\tregion_1\t4\n")

    rows = _plan_rows(REFERENCE_LENGTHS, REGIONS, None)
    with pytest.raises(ValueError, match="missing row.*region_2.*region_3"):
        _read_threshold_counts(thresholds_path, rows, [1])


def _alignment_files(tmp_path: Path) -> Path:
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"BAM")
    bam.with_suffix(".bam.bai").write_bytes(b"index")
    return bam


@dataclass
class _FakeState:
    deadlines: list[RequestDeadline]
    commands: list[list[str]]
    directories: list[Path]


def _install_fake_mosdepth(monkeypatch, *, failure: BaseException | None = None):
    from ont_qc_mcp import v2_coverage_qc as coverage

    segments = {
        "chr1": [(0, 2, 2), (2, 4, 1), (4, 10, 0)],
        "chr2": [(0, 5, 0)],
    }
    state = _FakeState([], [], [])

    def header(alignment, tools, cfg, deadline):
        state.deadlines.append(deadline)
        return REFERENCE_LENGTHS

    def run(stages, deadline, **kwargs):
        command = list(stages[0].command)
        state.deadlines.append(deadline)
        state.commands.append(command)
        prefix = Path(command[-2])
        state.directories.append(prefix.parent)
        (prefix.parent / "partial").write_text("owned")
        if failure is not None:
            raise failure

        target_path = Path(command[command.index("--by") + 1])
        targets: list[tuple[str, int, int, str]] = []
        for line in target_path.read_text().splitlines():
            chrom, raw_start, raw_end, row_id = line.split("\t")
            targets.append((chrom, int(raw_start), int(raw_end), row_id))

        with gzip.open(prefix.with_suffix(".regions.bed.gz"), "wt") as output:
            for chrom, start, end, row_id in reversed(targets):
                output.write(f"{chrom}\t{start}\t{end}\t{row_id}\t0.00\n")
        if "--no-per-base" not in command:
            with gzip.open(prefix.with_suffix(".per-base.bed.gz"), "wt") as output:
                for chrom, values in segments.items():
                    for start, end, depth in values:
                        output.write(f"{chrom}\t{start}\t{end}\t{depth}\n")
        if "--thresholds" in command:
            thresholds = [int(value) for value in command[command.index("--thresholds") + 1].split(",")]
            with gzip.open(prefix.with_suffix(".thresholds.bed.gz"), "wt") as output:
                output.write("#chrom\tstart\tend\tregion")
                output.write("".join(f"\t{threshold}X" for threshold in thresholds) + "\n")
                for chrom, start, end, row_id in reversed(targets):
                    counts = []
                    for threshold in thresholds:
                        count = sum(
                            max(0, min(end, segment_end) - max(start, segment_start))
                            for segment_start, segment_end, depth in segments[chrom]
                            if depth >= threshold
                        )
                        counts.append(count)
                    output.write(
                        f"{chrom}\t{start}\t{end}\t{row_id}" + "".join(f"\t{count}" for count in counts) + "\n"
                    )
        result = CommandResult(command, 0, "", "")
        return PipelineResult((result,))

    monkeypatch.setattr(coverage, "read_alignment_reference_lengths", header)
    monkeypatch.setattr(coverage, "run_pipeline", run)
    return state


@pytest.mark.parametrize(
    ("request_payload", "row_ids", "depth_sums", "union_bases"),
    [
        ({}, ["contig.chr1", "contig.chr2"], [6, 0], 15),
        (
            {"regions": [region.model_dump(exclude={"region_id"}) for region in REGIONS.requested]},
            ["region_1", "region_2", "region_3"],
            [6, 1, 0],
            11,
        ),
        (
            {"window_size": 4},
            ["chr1.window_1", "chr1.window_2", "chr1.window_3", "chr2.window_1", "chr2.window_2"],
            [6, 0, 0, 0, 0],
            15,
        ),
        (
            {
                "regions": [region.model_dump(exclude={"region_id"}) for region in REGIONS.requested],
                "window_size": 4,
            },
            [
                "region_1.window_1",
                "region_1.window_2",
                "region_2.window_1",
                "region_2.window_2",
                "region_3.window_1",
            ],
            [6, 0, 1, 0, 0],
            11,
        ),
    ],
)
def test_coverage_qc_all_four_modes(tmp_path, monkeypatch, request_payload, row_ids, depth_sums, union_bases) -> None:
    bam = _alignment_files(tmp_path)
    state = _install_fake_mosdepth(monkeypatch)

    result = coverage_qc(
        {"path": str(bam), "thresholds": [2, 0, 1], **request_payload},
        tools=ToolPaths(mosdepth="mosdepth", samtools="samtools"),
    )

    assert [row.row_id for row in result.rows] == row_ids
    assert [row.depth_sum for row in result.rows] == depth_sums
    assert result.union_summary.reference_bases == union_bases
    assert result.union_summary.depth_sum == 6
    assert [[entry.threshold for entry in row.breadth] for row in result.rows] == [[2, 0, 1]] * len(row_ids)
    assert all(row.breadth[1].bases_at_or_above == row.reference_bases for row in result.rows)
    assert state.deadlines[0] is state.deadlines[1]
    assert all(not directory.exists() for directory in state.directories)


def test_metric_subsets_and_native_selection_shape_the_command(tmp_path, monkeypatch) -> None:
    bam = _alignment_files(tmp_path)
    state = _install_fake_mosdepth(monkeypatch)
    tools = ToolPaths(mosdepth="mosdepth", samtools="samtools")

    breadth = coverage_qc(
        {
            "path": str(bam),
            "metrics": ["breadth"],
            "thresholds": [1, 2],
            "selection": {"min_mapq": 10, "include_flags": 1, "exclude_flags": 260, "read_group": "RG1"},
            "extra_args": {"mosdepth": ["--fast-mode"]},
        },
        tools=tools,
    )
    breadth_command = state.commands[0]
    assert "--no-per-base" in breadth_command
    assert breadth_command[breadth_command.index("--mapq") + 1] == "10"
    assert breadth_command[breadth_command.index("--include-flag") + 1] == "1"
    assert breadth_command[breadth_command.index("--flag") + 1] == "260"
    assert breadth_command[breadth_command.index("--read-groups") + 1] == "RG1"
    assert all(row.depth_sum is None and row.mean_depth is None for row in breadth.rows)
    assert breadth.union_summary.depth_sum is breadth.union_summary.mean_depth is None
    assert breadth.provenance[0].native_options_used is True
    assert "fast mode" in breadth.provenance[0].measurement_scope

    depth = coverage_qc({"path": str(bam), "metrics": ["depth"]}, tools=tools)
    depth_command = state.commands[1]
    assert "--thresholds" not in depth_command and "--no-per-base" not in depth_command
    assert depth_command[depth_command.index("--flag") + 1] == "1796"
    assert all(not row.breadth for row in depth.rows)


def test_combined_fragment_and_fast_modes_are_both_recorded_in_provenance(tmp_path, monkeypatch) -> None:
    bam = _alignment_files(tmp_path)
    _install_fake_mosdepth(monkeypatch)

    result = coverage_qc(
        {
            "path": str(bam),
            "metrics": ["depth"],
            "extra_args": {"mosdepth": ["--fragment-mode", "--fast-mode"]},
        },
        tools=ToolPaths(mosdepth="mosdepth", samtools="samtools"),
    )

    scope = result.provenance[0].measurement_scope
    assert "fragment mode combined with fast mode" in scope
    assert "without internal CIGAR or mate-overlap correction" in scope


def test_samtools_expression_is_rejected_before_input_or_native_execution(monkeypatch) -> None:
    from ont_qc_mcp import v2_coverage_qc as coverage

    def unexpected(*args, **kwargs):
        raise AssertionError("input resolution or native execution should not start")

    monkeypatch.setattr(coverage, "resolve_alignment_input", unexpected)
    monkeypatch.setattr(coverage, "run_pipeline", unexpected)
    with pytest.raises(ValueError, match="does not accept samtools-style filter expressions for mosdepth"):
        coverage_qc({"path": "missing.bam", "extra_args": {"mosdepth": ["--expr=mapq>10"]}})


def test_mosdepth_uses_alignment_access_path_with_adjacent_symlink_index(tmp_path, monkeypatch) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "reads.bam"
    target.write_bytes(b"BAM")
    access_path = tmp_path / "linked.bam"
    access_path.symlink_to(target)
    access_path.with_suffix(".bam.bai").write_bytes(b"index")
    state = _install_fake_mosdepth(monkeypatch)

    coverage_qc(
        {"path": str(access_path), "metrics": ["depth"]},
        tools=ToolPaths(mosdepth="mosdepth", samtools="samtools"),
    )

    assert state.commands[0][-1] == str(access_path.absolute())


@pytest.mark.parametrize("failure", [RuntimeError("upstream failure"), asyncio.CancelledError("cancelled")])
def test_failure_and_cancellation_remove_all_temporary_outputs(tmp_path, monkeypatch, failure) -> None:
    bam = _alignment_files(tmp_path)
    state = _install_fake_mosdepth(monkeypatch, failure=failure)

    with pytest.raises(type(failure), match=str(failure)):
        coverage_qc({"path": str(bam)}, tools=ToolPaths(mosdepth="mosdepth", samtools="samtools"))

    assert all(not directory.exists() for directory in state.directories)
