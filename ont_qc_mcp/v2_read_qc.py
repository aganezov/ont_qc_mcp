"""API v2 read-QC adapter for FASTQ, BAM, and CRAM inputs."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ExecutionConfig, ToolPaths
from .nanoq_aux_pipes import NanoqAuxPipes
from .parsers import parse_nanoq_json
from .process_control import wait_or_cancel
from .schemas import HistogramBin, NanoqStats
from .tools import _validate_input_file
from .v2_contracts import (
    LengthDistributionSection,
    Provenance,
    QualityDistributionSection,
    ReadEffectiveRequest,
    ReadLengthSection,
    ReadQCGroup,
    ReadQCRequest,
    ReadQCResponse,
    ReadQualitySection,
    V2HistogramBin,
    V2LengthPercentiles,
)
from .v2_execution import PipelineResult, PipelineStage, PipelineStageError, RequestDeadline, run_pipeline
from .v2_native_args import READ_QC_NANOQ_POPULATION_ARGS, ValidatedNativeArgs, validate_native_args
from .v2_read_records import ReadRecordReport, parse_report
from .v2_regions import NormalizedRegionSet
from .v2_samtools import (
    ResolvedAlignmentInput,
    SamtoolsSelection,
    resolve_and_normalize_regions,
    samtools_fastq_plan,
    samtools_view_plan,
)


FASTQ_SUFFIXES = (".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.bgz", ".fq.bgz")


@dataclass(frozen=True)
class _PopulationResult:
    accounting: ReadRecordReport
    stats: NanoqStats
    provenance: tuple[Provenance, ...]


def _input_format(path: str, requested: str | None) -> str:
    lowered = path.lower()
    inferred = (
        "fastq"
        if any(lowered.endswith(suffix) for suffix in FASTQ_SUFFIXES)
        else "bam"
        if lowered.endswith(".bam")
        else "cram"
        if lowered.endswith(".cram")
        else None
    )
    if inferred is None:
        raise ValueError("read_qc input must use a FASTQ, BAM, or CRAM filename extension")
    if requested is not None and requested != inferred:
        raise ValueError(f"input_format={requested!r} does not match the {inferred.upper()} path")
    return inferred


def _deadline(request: ReadQCRequest, cfg: ExecutionConfig, input_format: str) -> RequestDeadline:
    if request.deadline_seconds is not None:
        return RequestDeadline(request.deadline_seconds)
    seconds = cfg.timeout_for("nanoq")
    if input_format != "fastq":
        seconds = max(seconds, cfg.timeout_for("samtools"))
    return RequestDeadline(seconds)


def _nanoq_native(request: ReadQCRequest) -> ValidatedNativeArgs:
    return validate_native_args(
        "nanoq",
        request.extra_args.nanoq,
        additionally_protected=READ_QC_NANOQ_POPULATION_ARGS,
    )


def _uses_distributions(request: ReadQCRequest) -> bool:
    return any(metric in request.metrics for metric in ("length_distribution", "quality_distribution"))


def _nanoq_stage(
    tools: ToolPaths,
    native: ValidatedNativeArgs,
    aux_args: Sequence[str],
    *,
    path: Path | None,
) -> tuple[PipelineStage, list[str]]:
    args = ["--stats", "--json"]
    if path is not None:
        args.extend(("--input", str(path)))
    args.extend(aux_args)
    args.extend(native.supplied_args)
    if path is not None:
        return PipelineStage("nanoq", (tools.nanoq, *args)), args
    command = (sys.executable, "-m", "ont_qc_mcp.v2_nanoq_stream", "--", tools.nanoq, *args)
    return PipelineStage("nanoq", command), args


def _provenance(backend: str, args: Sequence[str], *, native: bool, scope: str) -> Provenance:
    return Provenance(
        backend=backend,
        effective_args=list(args),
        native_options_used=native,
        measurement_scope=scope,
    )


def _run_nanoq_pipeline(
    request: ReadQCRequest,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
    build_stages: Callable[[Sequence[str]], tuple[Sequence[PipelineStage], list[str]]],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[PipelineResult, NanoqStats, list[str]]:
    """Run nanoq at most twice, with fresh auxiliary pipes for each attempt."""
    for attempt in range(2):
        try:
            with ExitStack() as cleanup:
                aux = (
                    cleanup.enter_context(NanoqAuxPipes(cfg))
                    if _uses_distributions(request) and cfg.nanoq_aux_stats
                    else None
                )
                stages, effective_args = build_stages(aux.args if aux is not None else ())
                result = run_pipeline(stages, deadline, env=env)
                stats = parse_nanoq_json(result.final.stdout)
                if aux is not None:
                    aux.augment(stats, deadline=deadline)
                return result, stats, effective_args
        except PipelineStageError as error:
            if attempt == 1 or error.stage != "nanoq":
                raise
            wait_or_cancel(min(0.5, deadline.remaining()))
    raise RuntimeError("nanoq retry exhausted")  # Unreachable: the second failure is raised above.


def _run_fastq(
    path: Path,
    request: ReadQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> _PopulationResult:
    native = _nanoq_native(request)

    def build_stages(aux_args: Sequence[str]) -> tuple[Sequence[PipelineStage], list[str]]:
        stage, effective_args = _nanoq_stage(tools, native, aux_args, path=path)
        return [stage], effective_args

    _, stats, effective_args = _run_nanoq_pipeline(request, cfg, deadline, build_stages)
    deadline.checkpoint()
    accounting = ReadRecordReport(stats.read_count, stats.read_count, 0)
    return _PopulationResult(
        accounting=accounting,
        stats=stats,
        provenance=(
            _provenance(
                "nanoq",
                effective_args,
                native=native.native_options_used,
                scope="complete FASTQ sequences",
            ),
        ),
    )


def _sam_to_text_stage(tools: ToolPaths) -> PipelineStage:
    return PipelineStage("samtools_sam", (tools.samtools, "view", "-h", "--no-PG", "-"))


def _record_stage(*, quality_required: bool, discard: bool = False) -> PipelineStage:
    command = [sys.executable, "-m", "ont_qc_mcp.v2_read_records"]
    if quality_required:
        command.append("--quality-required")
    if discard:
        command.append("--discard")
    return PipelineStage("read_record_filter", tuple(command))


def _selection(request: ReadQCRequest, regional: bool) -> SamtoolsSelection:
    return SamtoolsSelection(
        min_mapq=request.selection.min_mapq or 0,
        include_unmapped=not regional,
        primary_only=True,
    )


def _alignment_environment() -> dict[str, str]:
    return dict(os.environ, REF_PATH=os.devnull, REF_CACHE=os.devnull)


def _inspect_alignment_population(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: ReadQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
    *,
    quality_required: bool,
) -> ReadRecordReport:
    with samtools_view_plan(
        alignment,
        regions,
        _selection(request, bool(regions.requested)),
        tools=tools,
        exec_cfg=cfg,
        native_args=request.extra_args.samtools_view,
    ) as view:
        result = run_pipeline(
            [view.stage, _sam_to_text_stage(tools), _record_stage(quality_required=quality_required, discard=True)],
            deadline,
            env=_alignment_environment(),
        )
    alignment.assert_unchanged()
    return parse_report(result.final.stderr)


def _run_alignment_population(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    request: ReadQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
    *,
    quality_required: bool,
) -> _PopulationResult:
    native_nanoq = _nanoq_native(request)
    fastq = samtools_fastq_plan(tools, exec_cfg=cfg, native_args=request.extra_args.samtools_fastq)
    scope = (
        "complete stored sequences from region-overlapping primary records"
        if regions.requested
        else "complete stored sequences from primary alignment records"
    )
    with samtools_view_plan(
        alignment,
        regions,
        _selection(request, bool(regions.requested)),
        tools=tools,
        exec_cfg=cfg,
        native_args=request.extra_args.samtools_view,
    ) as view:

        def build_stages(aux_args: Sequence[str]) -> tuple[Sequence[PipelineStage], list[str]]:
            nanoq, nanoq_args = _nanoq_stage(tools, native_nanoq, aux_args, path=None)
            stages = [
                view.stage,
                _sam_to_text_stage(tools),
                _record_stage(quality_required=quality_required),
                fastq.stage,
                nanoq,
            ]
            return stages, nanoq_args

        result, stats, nanoq_args = _run_nanoq_pipeline(
            request,
            cfg,
            deadline,
            build_stages,
            env=_alignment_environment(),
        )
        accounting = parse_report(result.stages[2].stderr)
        provenance = (
            _provenance(
                "samtools_view",
                view.native_args.effective_args,
                native=view.native_args.native_options_used,
                scope="primary alignment record selection",
            ),
            _provenance(
                "samtools_fastq",
                fastq.native_args.effective_args,
                native=fastq.native_args.native_options_used,
                scope="stored-sequence conversion",
            ),
            _provenance(
                "nanoq",
                nanoq_args,
                native=native_nanoq.native_options_used,
                scope=scope,
            ),
        )
    alignment.assert_unchanged()
    deadline.checkpoint()
    if stats.read_count != accounting.emitted_sequences:
        raise RuntimeError(
            "nanoq retained a different read count than samtools emitted; "
            "native filtering cannot preserve v2 accounting"
        )
    return _PopulationResult(accounting=accounting, stats=stats, provenance=provenance)


def _histogram(values: Sequence[HistogramBin] | None) -> list[V2HistogramBin]:
    return [V2HistogramBin(start=value.start, end=value.end, count=value.count) for value in values or ()]


def _group(
    stats: NanoqStats,
    request: ReadQCRequest,
    *,
    region_id: str | None,
    region_name: str | None,
) -> ReadQCGroup:
    if (
        stats.read_count
        and (stats.mean_qscore is None or stats.median_qscore is None)
        and any(metric in request.metrics for metric in ("read_quality", "quality_distribution"))
    ):
        raise RuntimeError("nanoq did not report quality summaries for emitted sequences")
    length = (
        ReadLengthSection(
            read_count=stats.read_count,
            total_bases=stats.total_bases,
            min_length=stats.min_len if stats.read_count else None,
            max_length=stats.max_len if stats.read_count else None,
            mean_length=stats.total_bases / stats.read_count if stats.read_count else None,
            median_length=stats.median_len if stats.read_count else None,
            n50=stats.n50 if stats.read_count else None,
        )
        if "length" in request.metrics
        else None
    )
    read_quality = (
        ReadQualitySection(
            reads_with_quality=stats.read_count,
            reads_missing_quality=0,
            mean_qscore=stats.mean_qscore,
            median_qscore=stats.median_qscore,
        )
        if "read_quality" in request.metrics
        else None
    )
    percentiles = stats.length_percentiles
    length_distribution = (
        LengthDistributionSection(
            percentiles=V2LengthPercentiles(
                p1=percentiles.p1 if percentiles else None,
                p5=percentiles.p5 if percentiles else None,
                p25=percentiles.p25 if percentiles else None,
                p50=percentiles.p50 if percentiles else None,
                p75=percentiles.p75 if percentiles else None,
                p95=percentiles.p95 if percentiles else None,
                p99=percentiles.p99 if percentiles else None,
            ),
            histogram=_histogram(stats.length_histogram),
        )
        if "length_distribution" in request.metrics
        else None
    )
    quality_distribution = (
        QualityDistributionSection(histogram=_histogram(stats.qscore_histogram), per_position_mean=None)
        if "quality_distribution" in request.metrics
        else None
    )
    return ReadQCGroup(
        region_id=region_id,
        region_name=region_name,
        length=length,
        read_quality=read_quality,
        length_distribution=length_distribution,
        quality_distribution=quality_distribution,
    )


def read_qc(
    request: ReadQCRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ReadQCResponse:
    """Execute one validated read-QC request without registering the v2 catalog."""
    validated = request if isinstance(request, ReadQCRequest) else ReadQCRequest.model_validate(request)
    cfg = exec_cfg or ExecutionConfig()
    tool_paths = tools or ToolPaths()
    input_format = _input_format(validated.path, validated.input_format)
    deadline = _deadline(validated, cfg, input_format)
    quality_required = any(metric in validated.metrics for metric in ("read_quality", "quality_distribution"))

    if input_format == "fastq":
        if validated.regions is not None or validated.reference_path is not None:
            raise ValueError("FASTQ input does not support regions or reference_path")
        if (
            validated.selection.min_mapq is not None
            or validated.extra_args.samtools_view
            or validated.extra_args.samtools_fastq
        ):
            raise ValueError("FASTQ input does not support alignment selection or samtools arguments")
        path = Path(validated.path).resolve()
        _validate_input_file(path, cfg, allowed_exts=FASTQ_SUFFIXES)
        population = _run_fastq(path, validated, tool_paths, cfg, deadline)
        regions = NormalizedRegionSet((), ())
        groups = [_group(population.stats, validated, region_id=None, region_name=None)]
        accounting = population.accounting
        provenance = list(population.provenance)
        effective_path = str(path)
        effective_reference = None
    else:
        alignment, regions = resolve_and_normalize_regions(
            validated.path,
            validated.regions,
            reference_path=validated.reference_path,
            tools=tool_paths,
            exec_cfg=cfg,
            deadline=deadline,
        )
        effective_path = str(alignment.alignment)
        effective_reference = str(alignment.reference) if alignment.reference is not None else None
        if validated.group_by == "combined":
            population = _run_alignment_population(
                alignment,
                regions,
                validated,
                tool_paths,
                cfg,
                deadline,
                quality_required=quality_required,
            )
            accounting = population.accounting
            groups = [_group(population.stats, validated, region_id=None, region_name=None)]
            provenance = list(population.provenance)
        else:
            accounting = _inspect_alignment_population(
                alignment,
                regions,
                validated,
                tool_paths,
                cfg,
                deadline,
                quality_required=quality_required,
            )
            groups = []
            provenance = []
            for region, population_regions in zip(regions.requested, regions.per_region(), strict=True):
                population = _run_alignment_population(
                    alignment,
                    population_regions,
                    validated,
                    tool_paths,
                    cfg,
                    deadline,
                    quality_required=quality_required,
                )
                groups.append(_group(population.stats, validated, region_id=region.region_id, region_name=region.name))
                provenance.extend(population.provenance)

    effective = ReadEffectiveRequest(
        path=effective_path,
        reference_path=effective_reference,
        region_scope="normalized_intervals" if regions.requested else "whole_file",
        normalized_regions=list(regions.requested),
        resolved_group_by=validated.group_by,
        metrics=validated.metrics,
        selection=validated.selection,
        extra_args=validated.extra_args,
    )
    return ReadQCResponse(
        resolved_group_by=validated.group_by,
        selected_records=accounting.selected_records,
        emitted_sequences=accounting.emitted_sequences,
        conversion_exclusions=accounting.conversion_exclusions,
        effective_request=effective,
        results=groups,
        provenance=provenance,
    )


__all__ = ["FASTQ_SUFFIXES", "read_qc"]
