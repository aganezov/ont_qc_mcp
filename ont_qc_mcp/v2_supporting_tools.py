"""Strict request adapters for the six unregistered API v2 supporting tools."""

from __future__ import annotations

from .config import ExecutionConfig, ToolPaths
from .schemas import BedQCReport, ChopperReport, EnvStatus, HeaderMetadata, IgvSnapshotResult, SequencingSummaryStats
from .tools import (
    _parse_bed_regions,
    env_check,
    filter_reads as filter_reads_core,
    generate_igv_snapshots,
    header_metadata_lookup,
    qc_bed,
    sequencing_summary,
)
from .v2_contracts import (
    BedQCRequest,
    EnvironmentStatusRequest,
    FilterReadsRequest,
    HeaderInfoRequest,
    IgvSnapshotsRequest,
    RunSummaryRequest,
    V2IgvRegion,
)
from .v2_regions import MAX_REGIONS
from .v2_samtools import file_identity, local_file


def environment_status(
    request: EnvironmentStatusRequest | dict[str, object] | None = None,
    *,
    tools: ToolPaths | None = None,
) -> EnvStatus:
    """Return the existing executable/runtime inspection through the strict v2 request."""
    if not isinstance(request, EnvironmentStatusRequest):
        EnvironmentStatusRequest.model_validate({} if request is None else request)
    return env_check(tools or ToolPaths())


def header_info(
    request: HeaderInfoRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> HeaderMetadata:
    """Return inferred BAM/CRAM/SAM/VCF header metadata with optional reference access."""
    validated = request if isinstance(request, HeaderInfoRequest) else HeaderInfoRequest.model_validate(request)
    return header_metadata_lookup(
        path=validated.path,
        reference_path=validated.reference_path,
        tools=tools or ToolPaths(),
        exec_cfg=exec_cfg or ExecutionConfig(),
        max_lines=None,
    )


def bed_qc(
    request: BedQCRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> BedQCReport:
    """Run the existing BED validator through the strict v2 request."""
    validated = request if isinstance(request, BedQCRequest) else BedQCRequest.model_validate(request)
    return qc_bed(validated.path, tools=tools or ToolPaths(), exec_cfg=exec_cfg or ExecutionConfig())


def run_summary(
    request: RunSummaryRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> SequencingSummaryStats:
    """Retain fixed one-hour sequencing-summary bins through the path-only v2 request."""
    validated = request if isinstance(request, RunSummaryRequest) else RunSummaryRequest.model_validate(request)
    return sequencing_summary(validated.path, tools=tools or ToolPaths(), exec_cfg=exec_cfg or ExecutionConfig())


def filter_reads(
    request: FilterReadsRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ChopperReport:
    """Run the existing atomic Chopper transformation with typed and native v2 arguments."""
    validated = request if isinstance(request, FilterReadsRequest) else FilterReadsRequest.model_validate(request)
    selection = validated.selection.model_dump(exclude_none=True)
    native_args = validated.extra_args.get("chopper", [])
    report = filter_reads_core(
        validated.path,
        tools=tools or ToolPaths(),
        output_fastq=validated.output_fastq,
        flags=selection,
        exec_cfg=exec_cfg or ExecutionConfig(),
        extra_args=native_args,
    )
    report.params = {
        "selection": selection,
        "extra_args": {"chopper": list(native_args)},
    }
    return report


def igv_snapshots(
    request: IgvSnapshotsRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> IgvSnapshotResult:
    """Retain dynamic and prebuilt-batch IGV execution through the strict v2 request."""
    validated = request if isinstance(request, IgvSnapshotsRequest) else IgvSnapshotsRequest.model_validate(request)
    cfg = exec_cfg or ExecutionConfig()
    regions: list[dict[str, object]] | None
    if isinstance(validated.regions, str):
        bed_path = local_file(validated.regions, cfg, (".bed",))
        identity = file_identity(bed_path)
        parsed = _parse_bed_regions(
            bed_path,
            snapshot_format=validated.snapshot_format,
            min_snapshot_width=validated.min_snapshot_width,
            max_regions=MAX_REGIONS,
        )
        if identity != file_identity(bed_path):
            raise RuntimeError("The IGV BED region source changed during parsing; retry with a stable file")
        regions = [V2IgvRegion.model_validate(region.model_dump()).model_dump() for region in parsed]
    else:
        if isinstance(validated.regions, list) and len(validated.regions) > MAX_REGIONS:
            raise ValueError(f"regions must contain 1 to {MAX_REGIONS} intervals")
        regions = [region.model_dump() for region in validated.regions] if validated.regions is not None else None
    return generate_igv_snapshots(
        genome=validated.genome,
        tracks=validated.tracks,
        regions=regions,
        output_dir=validated.output_dir,
        batch_file=validated.batch_file,
        compact=validated.compact,
        color_by=validated.color_by,
        group_by=validated.group_by,
        snapshot_format=validated.snapshot_format,
        min_snapshot_width=validated.min_snapshot_width,
        extra_commands=validated.extra_commands,
        extra_preferences=validated.extra_preferences,
        small_indels_show=validated.small_indels_show,
        small_indels_threshold=validated.small_indels_threshold,
        allele_threshold=validated.allele_threshold,
        tools=tools or ToolPaths(),
        exec_cfg=cfg,
        regions_are_zero_based_half_open=True,
    )


__all__ = [
    "bed_qc",
    "environment_status",
    "filter_reads",
    "header_info",
    "igv_snapshots",
    "run_summary",
]
