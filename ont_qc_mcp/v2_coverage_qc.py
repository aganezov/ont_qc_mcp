"""Unregistered API v2 coverage-QC backend built on indexed mosdepth."""

from __future__ import annotations

import gzip
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ExecutionConfig, ToolPaths
from .process_control import check_cancelled
from .regional_metrics import RegionalInterval
from .utils import safe_path_arg
from .v2_contracts import (
    CoverageBreadth,
    CoverageEffectiveRequest,
    CoverageQCRequest,
    CoverageQCResponse,
    CoverageRow,
    CoverageUnionSummary,
    Provenance,
)
from .v2_execution import PipelineStage, RequestDeadline, run_pipeline
from .v2_native_args import ValidatedNativeArgs, validate_native_args
from .v2_regions import NormalizedRegionSet
from .v2_regions import normalize_regions
from .v2_samtools import file_identity, read_alignment_reference_lengths, resolve_alignment_input


_INTEGER = re.compile(r"[0-9]+")
MAX_COVERAGE_ROWS = 100_000
MAX_COVERAGE_BREADTH_CELLS = 300_000


@dataclass(frozen=True)
class _PlannedRow:
    row_id: str
    chrom: str
    start: int
    end: int
    name: str | None = None


def _checkpoint(deadline: RequestDeadline | None) -> None:
    if deadline is None:
        check_cancelled()
    else:
        deadline.checkpoint()


def _plan_rows(
    reference_lengths: Mapping[str, int],
    regions: NormalizedRegionSet,
    window_size: int | None,
    deadline: RequestDeadline | None = None,
) -> list[_PlannedRow]:
    """Resolve contig, requested-region, or requested-window rows in stable order."""
    if window_size is not None and (
        isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer or unset")

    domains = (
        ((region.start, region.end) for region in regions.requested)
        if regions.requested
        else ((0, length) for length in reference_lengths.values())
    )
    projected_rows = 0
    for start, end in domains:
        _checkpoint(deadline)
        projected_rows += 1 if window_size is None else (end - start + window_size - 1) // window_size
    if projected_rows > MAX_COVERAGE_ROWS:
        raise ValueError(f"projected coverage row count {projected_rows} exceeds the limit of {MAX_COVERAGE_ROWS}")

    if regions.requested and window_size is None:
        return [
            _PlannedRow(region.region_id, region.chrom, region.start, region.end, region.name)
            for region in regions.requested
        ]

    rows: list[_PlannedRow] = []
    if regions.requested:
        if window_size is None:  # Kept explicit because optimized Python removes assertions.
            raise RuntimeError("requested-region window planning requires window_size")
        for region in regions.requested:
            for window_index, start in enumerate(range(region.start, region.end, window_size), start=1):
                _checkpoint(deadline)
                rows.append(
                    _PlannedRow(
                        f"{region.region_id}.window_{window_index}",
                        region.chrom,
                        start,
                        min(start + window_size, region.end),
                        region.name,
                    )
                )
        return rows

    for chrom, length in reference_lengths.items():
        _checkpoint(deadline)
        if window_size is None:
            rows.append(_PlannedRow(f"contig.{chrom}", chrom, 0, length))
            continue
        for window_index, start in enumerate(range(0, length, window_size), start=1):
            _checkpoint(deadline)
            rows.append(_PlannedRow(f"{chrom}.window_{window_index}", chrom, start, min(start + window_size, length)))
    return rows


def _intervals_by_chrom(
    intervals: Sequence[tuple[int, RegionalInterval]],
) -> dict[str, list[tuple[int, RegionalInterval]]]:
    grouped: dict[str, list[tuple[int, RegionalInterval]]] = {}
    for index, interval in intervals:
        grouped.setdefault(interval.chrom, []).append((index, interval))
    for values in grouped.values():
        values.sort(key=lambda value: (value[1].start, value[1].end, value[0]))
    return grouped


def _add_segment(
    grouped: Mapping[str, Sequence[tuple[int, RegionalInterval]]],
    first_live: dict[str, int],
    totals: list[int],
    chrom: str,
    start: int,
    end: int,
    depth: int,
) -> None:
    intervals = grouped.get(chrom, ())
    first = first_live.get(chrom, 0)
    while first < len(intervals) and intervals[first][1].end <= start:
        first += 1
    first_live[chrom] = first
    index = first
    while index < len(intervals) and intervals[index][1].start < end:
        output_index, interval = intervals[index]
        overlap = min(end, interval.end) - max(start, interval.start)
        if overlap > 0:
            totals[output_index] += overlap * depth
        index += 1


def _read_per_base_depths(
    path: Path,
    reference_lengths: Mapping[str, int],
    rows: Sequence[_PlannedRow],
    union: Sequence[RegionalInterval],
    deadline: RequestDeadline | None = None,
) -> tuple[list[int], int]:
    """Read complete integer run-length depth evidence and aggregate rows plus union."""
    row_intervals = _intervals_by_chrom(
        [(index, RegionalInterval(chrom=row.chrom, start=row.start, end=row.end)) for index, row in enumerate(rows)]
    )
    union_intervals = _intervals_by_chrom(list(enumerate(union)))
    row_totals = [0] * len(rows)
    union_totals = [0] * len(union)
    row_first: dict[str, int] = {}
    union_first: dict[str, int] = {}
    next_start = {chrom: 0 for chrom in reference_lengths}
    closed_contigs: set[str] = set()
    current_chrom: str | None = None

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            _checkpoint(deadline)
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 4 or not all(_INTEGER.fullmatch(value) for value in fields[1:]):
                raise ValueError(f"Malformed mosdepth per-base output at line {line_number}")
            chrom = fields[0]
            if chrom not in reference_lengths:
                raise ValueError(f"Mosdepth per-base output contains unknown contig {chrom!r}")
            start, end, depth = map(int, fields[1:])
            if current_chrom != chrom:
                if chrom in closed_contigs:
                    raise ValueError(f"Mosdepth per-base output revisits contig {chrom!r}")
                if current_chrom is not None:
                    closed_contigs.add(current_chrom)
                current_chrom = chrom
            if start != next_start[chrom] or start >= end or end > reference_lengths[chrom]:
                raise ValueError(f"Mosdepth per-base output is incomplete or overlapping for {chrom!r}")
            next_start[chrom] = end
            _add_segment(row_intervals, row_first, row_totals, chrom, start, end, depth)
            _add_segment(union_intervals, union_first, union_totals, chrom, start, end, depth)

    # Mosdepth 0.3.14 omits a contig from per-base output when it has no
    # selected alignments. A wholly absent contig therefore represents exact
    # zero depth. Once a contig appears, however, its run-length rows must span
    # the complete reference domain without gaps or overlaps.
    incomplete = [chrom for chrom, length in reference_lengths.items() if next_start[chrom] not in {0, length}]
    if incomplete:
        raise ValueError(f"Mosdepth per-base output is incomplete for contig(s): {', '.join(incomplete)}")
    return row_totals, sum(union_totals)


def _read_threshold_counts(
    path: Path,
    rows: Sequence[_PlannedRow],
    thresholds: Sequence[int],
    deadline: RequestDeadline | None = None,
) -> list[list[int]]:
    """Read exact mosdepth threshold counts and restore planned request order."""
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    by_id = {row.row_id: (index, row) for index, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("planned coverage row IDs must be unique")
    counts: list[list[int] | None] = [None] * len(rows)
    expected_header = ["#chrom", "start", "end", "region", *(f"{threshold}X" for threshold in thresholds)]
    header_seen = False

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            _checkpoint(deadline)
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if fields[0].startswith("#"):
                if fields != expected_header or header_seen:
                    raise ValueError("Mosdepth threshold output has an unexpected header")
                header_seen = True
                continue
            if len(fields) != 4 + len(thresholds) or not all(_INTEGER.fullmatch(value) for value in fields[1:3]):
                raise ValueError(f"Malformed mosdepth threshold output at line {line_number}")
            entry = by_id.get(fields[3])
            if entry is None:
                raise ValueError(f"Mosdepth threshold output contains unknown row {fields[3]!r}")
            output_index, row = entry
            if counts[output_index] is not None:
                raise ValueError(f"Mosdepth threshold output repeats row {row.row_id!r}")
            start, end = int(fields[1]), int(fields[2])
            if (fields[0], start, end) != (row.chrom, row.start, row.end):
                raise ValueError(f"Mosdepth threshold coordinates do not match planned row {row.row_id!r}")
            values = fields[4:]
            if not all(_INTEGER.fullmatch(value) for value in values):
                raise ValueError(f"Mosdepth threshold row {row.row_id!r} contains a non-integer count")
            parsed = [int(value) for value in values]
            if any(value > row.end - row.start for value in parsed):
                raise ValueError(f"Mosdepth threshold row {row.row_id!r} exceeds its reference length")
            # Mosdepth 0.3.14 writes zero for every threshold when a contig has
            # no alignment records. Depth is nonnegative, so threshold zero is
            # definitionally the complete requested reference domain.
            for index, threshold in enumerate(thresholds):
                if threshold == 0:
                    parsed[index] = row.end - row.start
            counts[output_index] = parsed

    if not header_seen:
        raise ValueError("Mosdepth threshold output is missing its header")
    missing = [row.row_id for index, row in enumerate(rows) if counts[index] is None]
    if missing:
        raise ValueError(f"Mosdepth threshold output is missing row(s): {', '.join(missing)}")
    return [value for value in counts if value is not None]


def _read_median_depths(
    path: Path,
    rows: Sequence[_PlannedRow],
    deadline: RequestDeadline | None = None,
) -> list[float]:
    """Read native per-row medians and restore planned request order."""
    by_id = {row.row_id: (index, row) for index, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("planned coverage row IDs must be unique")
    medians: list[float | None] = [None] * len(rows)

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            _checkpoint(deadline)
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 5 or not all(_INTEGER.fullmatch(value) for value in fields[1:3]):
                raise ValueError(f"Malformed mosdepth median output at line {line_number}")
            entry = by_id.get(fields[3])
            if entry is None:
                raise ValueError(f"Mosdepth median output contains unknown row {fields[3]!r}")
            output_index, row = entry
            if medians[output_index] is not None:
                raise ValueError(f"Mosdepth median output repeats row {row.row_id!r}")
            start, end = int(fields[1]), int(fields[2])
            if (fields[0], start, end) != (row.chrom, row.start, row.end):
                raise ValueError(f"Mosdepth median coordinates do not match planned row {row.row_id!r}")
            try:
                median = float(fields[4])
            except ValueError as error:
                raise ValueError(
                    f"Mosdepth median row {row.row_id!r} must contain a nonnegative finite median"
                ) from error
            if not math.isfinite(median) or median < 0:
                raise ValueError(f"Mosdepth median row {row.row_id!r} must contain a nonnegative finite median")
            medians[output_index] = median

    missing = [row.row_id for index, row in enumerate(rows) if medians[index] is None]
    if missing:
        raise ValueError(f"Mosdepth median output is missing row(s): {', '.join(missing)}")
    return [value for value in medians if value is not None]


def _deadline(request: CoverageQCRequest, cfg: ExecutionConfig) -> RequestDeadline:
    if request.deadline_seconds is not None:
        return RequestDeadline(request.deadline_seconds)
    return RequestDeadline(max(cfg.timeout_for("samtools"), cfg.timeout_for("mosdepth")))


def _native(request: CoverageQCRequest) -> ValidatedNativeArgs:
    for argument in request.extra_args.mosdepth:
        if argument.split("=", 1)[0] in {"-e", "--expr", "--expression"}:
            raise ValueError(
                "coverage_qc does not accept samtools-style filter expressions for mosdepth; "
                "use mosdepth-native selection controls"
            )
    return validate_native_args("mosdepth", request.extra_args.mosdepth)


def _threads(cfg: ExecutionConfig) -> list[str]:
    threads = cfg.threads_for("mosdepth")
    if threads is None:
        return []
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError("mosdepth threads must be a nonnegative integer or unset")
    return ["--threads", str(threads)]


def _selection_args(request: CoverageQCRequest) -> list[str]:
    selection = request.selection
    # Make the pinned mosdepth default explicit so provenance records the actual
    # exclusion population even when the typed request leaves it unset.
    exclude_flags = 1796 if selection.exclude_flags is None else selection.exclude_flags
    args = ["--mapq", str(selection.min_mapq), "--flag", str(exclude_flags)]
    if selection.include_flags is not None:
        args.extend(("--include-flag", str(selection.include_flags)))
    if selection.read_group is not None:
        args.extend(("--read-groups", selection.read_group))
    return args


def _write_plan(path: Path, rows: Sequence[_PlannedRow], deadline: RequestDeadline) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            deadline.checkpoint()
            if row.chrom.startswith("#"):
                raise ValueError("Contig names beginning with '#' are unsupported by mosdepth BED selection")
            output.write(f"{row.chrom}\t{row.start}\t{row.end}\t{row.row_id}\n")


def _union_domain(reference_lengths: Mapping[str, int], regions: NormalizedRegionSet) -> tuple[RegionalInterval, ...]:
    if regions.requested:
        return regions.union
    return tuple(RegionalInterval(chrom=chrom, start=0, end=length) for chrom, length in reference_lengths.items())


def _counting_mode(native_args: Sequence[str]) -> str:
    short_flags = "".join(
        argument[1:] for argument in native_args if argument.startswith("-") and not argument.startswith("--")
    )
    fragment_mode = "--fragment-mode" in native_args or "a" in short_flags
    fast_mode = "--fast-mode" in native_args or "x" in short_flags
    if fragment_mode and fast_mode:
        mode = "mosdepth fragment mode combined with fast mode without internal CIGAR or mate-overlap correction"
    elif fragment_mode:
        mode = "mosdepth fragment mode"
    elif fast_mode:
        mode = "mosdepth fast mode without internal CIGAR or mate-overlap correction"
    else:
        mode = "mosdepth default CIGAR-aware aligned-base depth with mate-overlap correction"
    if native_args:
        mode += "; native options may alter selection or measurement and no default-mode equivalence is claimed"
    return mode


def _measurement_scope(request: CoverageQCRequest, regions: NormalizedRegionSet, native_args: Sequence[str]) -> str:
    if regions.requested:
        domain = "requested interval union; zero-depth reference bases retained"
    elif request.window_size is not None:
        domain = "windows tiled within reference contigs; zero-depth reference bases retained"
    else:
        domain = "whole reference; zero-depth reference bases retained"
    return f"{domain}; {_counting_mode(native_args)}"


def coverage_qc(
    request: CoverageQCRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> CoverageQCResponse:
    """Execute one validated coverage request without registering the v2 catalog."""
    validated = request if isinstance(request, CoverageQCRequest) else CoverageQCRequest.model_validate(request)
    cfg = exec_cfg or ExecutionConfig()
    tool_paths = tools or ToolPaths()
    native = _native(validated)
    deadline = _deadline(validated, cfg)
    alignment = resolve_alignment_input(
        validated.path,
        reference_path=validated.reference_path,
        require_index=True,
        exec_cfg=cfg,
    )
    reference_lengths = read_alignment_reference_lengths(alignment, tool_paths, cfg, deadline)
    regions = normalize_regions(validated.regions, reference_lengths, exec_cfg=cfg, deadline=deadline)
    dependency_identities = tuple(file_identity(path) for path in regions.external_dependencies)
    rows = _plan_rows(reference_lengths, regions, validated.window_size, deadline)
    if "breadth" in validated.metrics:
        projected_breadth_cells = len(rows) * len(validated.thresholds)
        if projected_breadth_cells > MAX_COVERAGE_BREADTH_CELLS:
            raise ValueError(
                f"projected breadth cell count {projected_breadth_cells} exceeds the limit of "
                f"{MAX_COVERAGE_BREADTH_CELLS}"
            )
    union = _union_domain(reference_lengths, regions)
    resolved_group_by = validated.group_by or (
        "window" if validated.window_size is not None else "region" if regions.requested else "contig"
    )
    sorted_thresholds = sorted(validated.thresholds)

    with tempfile.TemporaryDirectory(prefix="ont-qc-v2-coverage-") as directory:
        output_dir = Path(directory)
        target_path = output_dir / "targets.bed"
        prefix = output_dir / "coverage"
        _write_plan(target_path, rows, deadline)
        command = [
            tool_paths.mosdepth,
            *_threads(cfg),
            *_selection_args(validated),
            *native.supplied_args,
            *(["--use-median"] if validated.depth_statistic == "median" and "depth" in validated.metrics else []),
            "--by",
            safe_path_arg(target_path),
        ]
        if "breadth" in validated.metrics:
            command.extend(("--thresholds", ",".join(str(value) for value in sorted_thresholds)))
        if "depth" not in validated.metrics:
            command.append("--no-per-base")
        if alignment.reference_access_path is not None:
            command.extend(("--fasta", safe_path_arg(alignment.reference_access_path)))
        command.extend((str(prefix), safe_path_arg(alignment.alignment_access_path)))
        environment = dict(os.environ, REF_PATH=os.devnull, REF_CACHE=os.devnull)
        run_pipeline([PipelineStage("mosdepth", tuple(command))], deadline, env=environment)
        deadline.checkpoint()

        regions_output = prefix.with_suffix(".regions.bed.gz")
        if not regions_output.is_file():
            raise RuntimeError(f"mosdepth did not produce expected output: {regions_output}")

        depth_sums: Sequence[int | None]
        median_depths: Sequence[float | None]
        union_depth_sum: int | None
        if "depth" in validated.metrics:
            per_base_output = prefix.with_suffix(".per-base.bed.gz")
            if not per_base_output.is_file():
                raise RuntimeError(f"mosdepth did not produce expected output: {per_base_output}")
            depth_sums, union_depth_sum = _read_per_base_depths(
                per_base_output, reference_lengths, rows, union, deadline
            )
            median_depths = (
                _read_median_depths(regions_output, rows, deadline)
                if validated.depth_statistic == "median"
                else [None] * len(rows)
            )
        else:
            depth_sums = [None] * len(rows)
            median_depths = [None] * len(rows)
            union_depth_sum = None

        if "breadth" in validated.metrics:
            threshold_output = prefix.with_suffix(".thresholds.bed.gz")
            if not threshold_output.is_file():
                raise RuntimeError(f"mosdepth did not produce expected output: {threshold_output}")
            sorted_counts = _read_threshold_counts(threshold_output, rows, sorted_thresholds, deadline)
            breadth_counts = [
                {threshold: count for threshold, count in zip(sorted_thresholds, counts, strict=True)}
                for counts in sorted_counts
            ]
        else:
            breadth_counts = [{} for _ in rows]

        alignment.assert_unchanged()
        if dependency_identities != tuple(file_identity(path) for path in regions.external_dependencies):
            raise RuntimeError("A BED or GFF3 region source changed during coverage analysis; retry with stable files")

    response_rows: list[CoverageRow] = []
    for row, depth_sum, median_depth, counts in zip(rows, depth_sums, median_depths, breadth_counts, strict=True):
        deadline.checkpoint()
        reference_bases = row.end - row.start
        response_rows.append(
            CoverageRow(
                row_id=row.row_id,
                chrom=row.chrom,
                start=row.start,
                end=row.end,
                name=row.name,
                reference_bases=reference_bases,
                depth_sum=depth_sum,
                mean_depth=depth_sum / reference_bases if depth_sum is not None else None,
                median_depth=median_depth,
                breadth=[
                    CoverageBreadth(
                        threshold=threshold,
                        bases_at_or_above=counts[threshold],
                        fraction_at_or_above=counts[threshold] / reference_bases,
                    )
                    for threshold in validated.thresholds
                ]
                if "breadth" in validated.metrics
                else [],
            )
        )

    union_reference_bases = sum(interval.end - interval.start for interval in union)
    effective = CoverageEffectiveRequest(
        path=str(alignment.alignment),
        reference_path=str(alignment.reference) if alignment.reference is not None else None,
        region_scope="normalized_intervals" if regions.requested else "whole_reference",
        normalized_regions=list(regions.requested),
        resolved_group_by=resolved_group_by,
        window_size=validated.window_size,
        metrics=validated.metrics,
        depth_statistic=validated.depth_statistic,
        thresholds=validated.thresholds,
        selection=validated.selection,
        extra_args=validated.extra_args,
    )
    return CoverageQCResponse(
        resolved_group_by=resolved_group_by,
        effective_request=effective,
        rows=response_rows,
        union_summary=CoverageUnionSummary(
            reference_bases=union_reference_bases,
            depth_sum=union_depth_sum,
            mean_depth=union_depth_sum / union_reference_bases if union_depth_sum is not None else None,
        ),
        provenance=[
            Provenance(
                backend="mosdepth",
                effective_args=command[1:],
                native_options_used=native.native_options_used,
                measurement_scope=_measurement_scope(validated, regions, native.supplied_args),
            )
        ],
    )


__all__ = ["coverage_qc"]
