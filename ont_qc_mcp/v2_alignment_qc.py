"""Unregistered API v2 alignment-QC backend."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .config import ExecutionConfig, ToolPaths
from .parsers import parse_cramino_json, parse_error_profile
from .regional_metrics import RegionalInterval
from .v2_contracts import (
    AlignedBaseQualitySection,
    AlignmentCounts,
    AlignmentEffectiveRequest,
    AlignmentQCGroup,
    AlignmentQCRequest,
    AlignmentQCResponse,
    ErrorProfileSection,
    IdentitySection,
    MappingQualitySection,
    Provenance,
    V2CoverageBin,
    V2CycleMismatchCounts,
    V2HistogramBin,
)
from .v2_execution import PipelineStage, RequestDeadline, run_pipeline
from .v2_native_args import ValidatedNativeArgs, validate_native_args
from .v2_regions import NormalizedRegionSet
from .v2_samtools import ResolvedAlignmentInput, SamtoolsSelection, resolve_and_normalize_regions, samtools_view_plan


_SAMTOOLS_SUMMARY = re.compile(r"^SN\t(.+?):\t([^\t]*)")
_DEFAULT_PIPELINE_OUTPUT_BYTES = 1024 * 1024
_RECORD_GROUP_OUTPUT_HEADROOM_BYTES = 1024
MAX_SAMTOOLS_STATS_BYTES = 64 * 1024 * 1024


def _deadline(request: AlignmentQCRequest, cfg: ExecutionConfig) -> RequestDeadline:
    if request.deadline_seconds is not None:
        return RequestDeadline(request.deadline_seconds)
    timeouts = [cfg.timeout_for("samtools")]
    if "identity" in request.metrics:
        timeouts.append(cfg.timeout_for("cramino"))
    return RequestDeadline(max(timeouts))


def _threads(cfg: ExecutionConfig, backend: str, flag: str) -> tuple[str, ...]:
    value = cfg.threads_for(backend)
    if value is None:
        return ()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{backend} threads must be a nonnegative integer or unset")
    return (flag, str(value))


def _selection(request: AlignmentQCRequest) -> SamtoolsSelection:
    return SamtoolsSelection(
        min_mapq=request.selection.min_mapq,
        exclude_flags=request.selection.exclude_flags,
        include_unmapped=request.selection.include_unmapped,
    )


def _environment() -> dict[str, str]:
    return dict(os.environ, REF_PATH=os.devnull, REF_CACHE=os.devnull)


def _provenance(backend: str, args: Sequence[str], *, native: bool, scope: str) -> Provenance:
    return Provenance(
        backend=backend,
        effective_args=list(args),
        native_options_used=native,
        measurement_scope=scope,
    )


def _population_scope(regions: NormalizedRegionSet) -> str:
    return "whole records selected by interval overlap" if regions.requested else "whole selected alignment records"


def _record_output_limit(regions: NormalizedRegionSet, group_by: str) -> int:
    """Bound valid grouped JSON while retaining the shared pipeline's default floor."""
    if group_by != "region":
        return _DEFAULT_PIPELINE_OUTPUT_BYTES
    identity_bytes = sum(
        len(json.dumps((region.region_id, region.name), separators=(",", ":")).encode("utf-8"))
        for region in regions.requested
    )
    return (
        _DEFAULT_PIPELINE_OUTPUT_BYTES + len(regions.requested) * _RECORD_GROUP_OUTPUT_HEADROOM_BYTES + identity_bytes
    )


def _selected_populations(
    request: AlignmentQCRequest,
    regions: NormalizedRegionSet,
) -> list[tuple[str | None, str | None, NormalizedRegionSet]]:
    if request.group_by == "combined":
        return [(None, None, regions)]
    return [
        (region.region_id, region.name, population)
        for region, population in zip(regions.requested, regions.per_region(), strict=True)
    ]


def _run_selected_pipeline(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: AlignmentQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
    final_stages: Sequence[PipelineStage],
    *,
    max_output_bytes: int = 1024 * 1024,
) -> tuple[str, ValidatedNativeArgs]:
    with samtools_view_plan(
        alignment,
        regions,
        _selection(request),
        tools=tools,
        exec_cfg=cfg,
        native_args=request.extra_args.samtools_view,
    ) as view:
        result = run_pipeline(
            [view.stage, *final_stages],
            deadline,
            env=_environment(),
            max_output_bytes=max_output_bytes,
        )
        view_native = view.native_args
    alignment.assert_unchanged()
    deadline.checkpoint()
    return result.final.stdout, view_native


def _record_stage(
    *,
    region_path: Path | None,
    group_by: str,
    include_base_quality: bool,
) -> PipelineStage:
    command = [sys.executable, "-m", "ont_qc_mcp.v2_alignment_records", "--group-by", group_by]
    if region_path is not None:
        command.extend(("--regions", str(region_path)))
    if include_base_quality:
        command.append("--aligned-base-quality")
    return PipelineStage("alignment_record_metrics", tuple(command))


def _sam_to_text_stage(tools: ToolPaths) -> PipelineStage:
    return PipelineStage("samtools_sam", (tools.samtools, "view", "-h", "--no-PG", "-"))


def _record_groups(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: AlignmentQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> tuple[list[AlignmentQCGroup], list[Provenance]]:
    include_base_quality = "aligned_base_quality" in request.metrics
    metric_regions: Sequence[RegionalInterval]
    if request.group_by == "region":
        metric_regions = regions.requested
    else:
        metric_regions = regions.union

    with tempfile.TemporaryDirectory(prefix="ont-qc-v2-alignment-") as directory:
        region_path = None
        if metric_regions:
            region_path = Path(directory) / "regions.json"
            region_path.write_text(
                json.dumps(
                    [
                        {
                            "chrom": region.chrom,
                            "start": region.start,
                            "end": region.end,
                            "name": region.name,
                        }
                        for region in metric_regions
                    ],
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        output, view_native = _run_selected_pipeline(
            alignment,
            regions,
            request,
            tools,
            cfg,
            deadline,
            (
                _sam_to_text_stage(tools),
                _record_stage(
                    region_path=region_path,
                    group_by=request.group_by,
                    include_base_quality=include_base_quality,
                ),
            ),
            max_output_bytes=_record_output_limit(regions, request.group_by),
        )
    payload = json.loads(output)
    raw_groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(raw_groups, list):
        raise ValueError("alignment record metrics returned an invalid group payload")

    groups: list[AlignmentQCGroup] = []
    for value in raw_groups:
        if not isinstance(value, dict):
            raise ValueError("alignment record metrics returned an invalid group")
        groups.append(
            AlignmentQCGroup(
                region_id=value.get("region_id"),
                region_name=value.get("region_name"),
                counts=AlignmentCounts.model_validate(value["counts"]) if "counts" in request.metrics else None,
                mapping_quality=(
                    MappingQualitySection.model_validate(value["mapping_quality"])
                    if "mapping_quality" in request.metrics
                    else None
                ),
                aligned_base_quality=(
                    AlignedBaseQualitySection.model_validate(value["aligned_base_quality"])
                    if include_base_quality
                    else None
                ),
            )
        )
    expected_groups = 1 if request.group_by == "combined" else len(regions.requested)
    if len(groups) != expected_groups:
        raise ValueError("alignment record metrics returned the wrong number of groups")
    return groups, [
        _provenance(
            "samtools_view",
            view_native.effective_args,
            native=view_native.native_options_used,
            scope="alignment record selection",
        ),
        _provenance(
            "alignment_record_accumulator",
            (),
            native=False,
            scope=(
                "selected record counts, MAPQ, and aligned query bases inside requested intervals"
                if regions.requested
                else "selected record counts, MAPQ, and aligned query bases"
            ),
        ),
    ]


def _identity_section(payload: str) -> IdentitySection:
    stats = parse_cramino_json(payload)
    mean_identity = stats.mean_identity / 100 if stats.mean_identity is not None else None
    return IdentitySection(
        records_with_identity=None,
        records_missing_identity=None,
        mean_identity=mean_identity,
    )


def _run_identity(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: AlignmentQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> tuple[IdentitySection, list[Provenance]]:
    native = validate_native_args("cramino", request.extra_args.cramino)
    args = ["--format", "json", *_threads(cfg, "cramino", "--threads"), *native.supplied_args, "-"]
    payload, view_native = _run_selected_pipeline(
        alignment,
        regions,
        request,
        tools,
        cfg,
        deadline,
        (PipelineStage("cramino", (tools.cramino, *args)),),
    )
    return _identity_section(payload), [
        _provenance(
            "samtools_view",
            view_native.effective_args,
            native=view_native.native_options_used,
            scope="alignment record selection",
        ),
        _provenance(
            "cramino",
            args,
            native=native.native_options_used,
            scope=_population_scope(regions),
        ),
    ]


def _summary_rate(payload: str, key: str) -> float | None:
    values: dict[str, str] = {}
    for line in payload.splitlines():
        match = _SAMTOOLS_SUMMARY.match(line)
        if match:
            values[match.group(1).strip().lower()] = match.group(2).strip()
    raw = values.get(key)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _error_section(payload: str) -> ErrorProfileSection:
    parsed = parse_error_profile(payload, "-")
    return ErrorProfileSection(
        records_with_nm=None,
        records_missing_nm=None,
        nm_error_rate=_summary_rate(payload, "error rate"),
        mismatch_rate=_summary_rate(payload, "mismatches per base"),
        insertion_rate=parsed.insertion_rate,
        deletion_rate=parsed.deletion_rate,
        coverage_histogram=(
            [V2CoverageBin(start=value.start, end=value.end, count=value.count) for value in parsed.coverage_histogram]
            if parsed.coverage_histogram is not None
            else None
        ),
        mismatch_counts_by_cycle=(
            [
                V2CycleMismatchCounts(
                    cycle=value.cycle,
                    n_count=value.n_count,
                    mismatches_by_quality=value.mismatches_by_quality,
                )
                for value in parsed.mismatch_counts_by_cycle
            ]
            if parsed.mismatch_counts_by_cycle is not None
            else None
        ),
        insert_size_histogram=(
            [
                V2HistogramBin(start=value.start, end=value.end, count=value.count)
                for value in parsed.insert_size_histogram
                if value.end is not None
            ]
            if parsed.insert_size_histogram is not None
            else None
        ),
    )


def _run_error_profile(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: AlignmentQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> tuple[ErrorProfileSection, list[Provenance]]:
    native = validate_native_args("samtools_stats", request.extra_args.samtools_stats)
    reference_args = ("-r", str(alignment.reference_access_path)) if alignment.reference_access_path is not None else ()
    args = [*_threads(cfg, "samtools", "-@"), "-F", "0", *reference_args, *native.supplied_args, "-"]
    payload, view_native = _run_selected_pipeline(
        alignment,
        regions,
        request,
        tools,
        cfg,
        deadline,
        (
            PipelineStage("samtools_stats", (tools.samtools, "stats", *args)),
            PipelineStage(
                "samtools_stats_filter",
                (sys.executable, "-m", "ont_qc_mcp.v2_samtools_stats_filter"),
            ),
        ),
        max_output_bytes=MAX_SAMTOOLS_STATS_BYTES,
    )
    return _error_section(payload), [
        _provenance(
            "samtools_view",
            view_native.effective_args,
            native=view_native.native_options_used,
            scope="alignment record selection",
        ),
        _provenance(
            "samtools_stats",
            args,
            native=native.native_options_used,
            scope=_population_scope(regions),
        ),
    ]


def alignment_qc(
    request: AlignmentQCRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> AlignmentQCResponse:
    """Execute one validated public alignment-QC request."""
    validated = request if isinstance(request, AlignmentQCRequest) else AlignmentQCRequest.model_validate(request)
    cfg = exec_cfg or ExecutionConfig()
    tool_paths = tools or ToolPaths()
    deadline = _deadline(validated, cfg)
    alignment, regions = resolve_and_normalize_regions(
        validated.path,
        validated.regions,
        reference_path=validated.reference_path,
        tools=tool_paths,
        exec_cfg=cfg,
        deadline=deadline,
    )

    needs_records = any(metric in validated.metrics for metric in ("counts", "mapping_quality", "aligned_base_quality"))
    if needs_records:
        groups, provenance = _record_groups(alignment, regions, validated, tool_paths, cfg, deadline)
    else:
        groups = [
            AlignmentQCGroup(region_id=region_id, region_name=region_name)
            for region_id, region_name, _ in _selected_populations(validated, regions)
        ]
        provenance = []

    for index, (_, _, population) in enumerate(_selected_populations(validated, regions)):
        if "identity" in validated.metrics:
            groups[index].identity, additions = _run_identity(
                alignment,
                population,
                validated,
                tool_paths,
                cfg,
                deadline,
            )
            provenance.extend(additions)
        if "error_profile" in validated.metrics:
            groups[index].error_profile, additions = _run_error_profile(
                alignment,
                population,
                validated,
                tool_paths,
                cfg,
                deadline,
            )
            provenance.extend(additions)

    effective = AlignmentEffectiveRequest(
        path=str(alignment.alignment),
        reference_path=str(alignment.reference) if alignment.reference is not None else None,
        region_scope="normalized_intervals" if regions.requested else "whole_file",
        normalized_regions=list(regions.requested),
        resolved_group_by=validated.group_by,
        metrics=validated.metrics,
        selection=validated.selection,
        extra_args=validated.extra_args,
    )
    return AlignmentQCResponse(
        resolved_group_by=validated.group_by,
        effective_request=effective,
        results=groups,
        provenance=provenance,
    )


__all__ = ["MAX_SAMTOOLS_STATS_BYTES", "alignment_qc"]
