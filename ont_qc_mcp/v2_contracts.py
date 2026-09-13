"""Unregistered request/response contracts for the proposed API v2 catalog.

This module intentionally has no import path from :mod:`ont_qc_mcp.app_server`.
Later implementation slices can bind these contracts to handlers without changing
the currently advertised MCP catalog.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from .regional_metrics import RegionalInterval
from .schemas import (
    BedQCReport,
    ChopperReport,
    CoverageBin,
    CycleMismatchCounts,
    EnvStatus,
    HeaderMetadata,
    HistogramBin,
    IgvSnapshotResult,
    LengthPercentiles,
    SequencingSummaryStats,
)


class ContractModel(BaseModel):
    """Strict base for wire contracts."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class SamtoolsRegions(ContractModel):
    format: Literal["samtools"]
    values: list[str] = Field(min_length=1, description="One-based inclusive samtools region strings")

    @field_validator("values")
    @classmethod
    def nonempty_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("samtools region strings must not be blank")
        return values


class BedTextRegions(ContractModel):
    format: Literal["bed"]
    text: str = Field(min_length=1, description="Inline zero-based half-open BED text")

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("BED text must contain at least one interval")
        return value


class BedFileRegions(ContractModel):
    format: Literal["bed"]
    path: str = Field(min_length=1, description="Path to a BED file")


class Gff3Regions(ContractModel):
    format: Literal["gff3"]
    path: str = Field(min_length=1, description="Path to a GFF/GFF3 annotation file")
    feature_type: Literal["gene"] = Field(description="The narrow gene lookup supported by the current server")
    ids: list[str] | None = Field(default=None, min_length=1, description="Optional gene identifiers or names")


TaggedRegionSource: TypeAlias = SamtoolsRegions | BedTextRegions | BedFileRegions | Gff3Regions
RegionSource: TypeAlias = list[RegionalInterval] | TaggedRegionSource


class ReadSelection(ContractModel):
    primary_only: Literal[True] = Field(
        default=True,
        description="For BAM/CRAM, select primary alignment records before complete stored-sequence conversion",
    )
    min_mapq: int | None = Field(default=None, ge=0, le=254)

    @field_validator("primary_only", mode="before")
    @classmethod
    def primary_only_is_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("primary_only must be a boolean")
        return value


class AlignmentSelection(ContractModel):
    min_mapq: int = Field(default=0, ge=0, le=254)
    exclude_flags: int = Field(default=1796, ge=0, le=65535)
    include_unmapped: bool = Field(default=False, description="Eligible only for whole-file reporting")

    @model_validator(mode="after")
    def unmapped_flag_is_consistent(self) -> "AlignmentSelection":
        if self.include_unmapped and self.exclude_flags & 0x4:
            raise ValueError("include_unmapped=True requires exclude_flags without the unmapped bit 0x4")
        return self


class CoverageSelection(ContractModel):
    min_mapq: int = Field(default=0, ge=0, le=254)
    include_flags: int | None = Field(default=None, ge=0, le=65535)
    exclude_flags: int | None = Field(default=None, ge=0, le=65535)
    read_group: str | None = Field(default=None, min_length=1)


class VariantSelection(ContractModel):
    include_expression: str | None = Field(default=None, min_length=1)
    exclude_expression: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def expressions_are_exclusive(self) -> "VariantSelection":
        if self.include_expression is not None and self.exclude_expression is not None:
            raise ValueError("include_expression and exclude_expression are mutually exclusive")
        return self


class ReadExtraArgs(ContractModel):
    samtools_view: list[str] = Field(default_factory=list)
    samtools_fastq: list[str] = Field(default_factory=list)
    nanoq: list[str] = Field(default_factory=list)


class AlignmentExtraArgs(ContractModel):
    samtools_view: list[str] = Field(default_factory=list)
    samtools_stats: list[str] = Field(default_factory=list)
    cramino: list[str] = Field(default_factory=list)


class CoverageExtraArgs(ContractModel):
    mosdepth: list[str] = Field(default_factory=list)


class VariantExtraArgs(ContractModel):
    bcftools_stats: list[str] = Field(default_factory=list)


class NumericalRequest(ContractModel):
    path: str = Field(min_length=1)
    reference_path: str | None = Field(default=None, min_length=1)
    regions: RegionSource | None = None
    deadline_seconds: float | None = Field(default=None, gt=0)

    @field_validator("regions", mode="before")
    @classmethod
    def reject_empty_region_list(cls, value: object) -> object:
        if isinstance(value, list) and not value:
            raise ValueError("regions must be omitted for whole-file scope, not an empty list")
        return value


ReadMetric = Literal["length", "read_quality", "length_distribution", "quality_distribution"]


def _default_read_metrics() -> list[ReadMetric]:
    return ["length", "read_quality"]


class ReadQCRequest(NumericalRequest):
    input_format: Literal["fastq", "bam", "cram"] | None = None
    selection: ReadSelection = Field(default_factory=ReadSelection)
    group_by: Literal["combined", "region"] = "combined"
    metrics: list[ReadMetric] = Field(default_factory=_default_read_metrics, min_length=1)
    extra_args: ReadExtraArgs = Field(default_factory=ReadExtraArgs)

    @model_validator(mode="after")
    def fastq_has_no_alignment_scope(self) -> "ReadQCRequest":
        if self.input_format == "fastq" and (self.regions is not None or self.reference_path is not None):
            raise ValueError("FASTQ input does not support regions or reference_path")
        if self.group_by == "region" and self.regions is None:
            raise ValueError("group_by='region' requires regions")
        if self.input_format == "fastq" and (
            self.selection.min_mapq is not None or self.extra_args.samtools_view or self.extra_args.samtools_fastq
        ):
            raise ValueError("FASTQ input does not support alignment selection or samtools arguments")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        return self


AlignmentMetric = Literal["counts", "mapping_quality", "aligned_base_quality", "identity", "error_profile"]


def _default_alignment_metrics() -> list[AlignmentMetric]:
    return ["counts", "mapping_quality"]


class AlignmentQCRequest(NumericalRequest):
    selection: AlignmentSelection = Field(default_factory=AlignmentSelection)
    group_by: Literal["combined", "region"] = "combined"
    metrics: list[AlignmentMetric] = Field(default_factory=_default_alignment_metrics, min_length=1)
    extra_args: AlignmentExtraArgs = Field(default_factory=AlignmentExtraArgs)

    @model_validator(mode="after")
    def valid_scope(self) -> "AlignmentQCRequest":
        if self.group_by == "region" and self.regions is None:
            raise ValueError("group_by='region' requires regions")
        if self.selection.include_unmapped and self.regions is not None:
            raise ValueError("unmapped records are ineligible for regional reporting")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        return self


CoverageGroup = Literal["contig", "region", "window"]
CoverageMetric = Literal["depth", "breadth"]


def _default_coverage_metrics() -> list[CoverageMetric]:
    return ["depth", "breadth"]


class CoverageQCRequest(NumericalRequest):
    selection: CoverageSelection = Field(default_factory=CoverageSelection)
    group_by: CoverageGroup | None = Field(
        default=None,
        description="Natural default resolves to contig, region, or window from regions/window_size",
    )
    window_size: PositiveInt | None = None
    metrics: list[CoverageMetric] = Field(default_factory=_default_coverage_metrics, min_length=1)
    thresholds: list[NonNegativeInt] = Field(default_factory=lambda: [1, 10, 20], min_length=1)
    extra_args: CoverageExtraArgs = Field(default_factory=CoverageExtraArgs)

    @model_validator(mode="after")
    def valid_coverage_shape(self) -> "CoverageQCRequest":
        expected: CoverageGroup = (
            "window" if self.window_size is not None else "region" if self.regions is not None else "contig"
        )
        if self.group_by is not None and self.group_by != expected:
            raise ValueError(f"group_by must be '{expected}' for this regions/window_size combination")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("thresholds must not contain duplicates")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        return self


VariantMetric = Literal["general", "snps", "indels"]


def _default_variant_metrics() -> list[VariantMetric]:
    return ["general", "snps", "indels"]


class VariantQCRequest(NumericalRequest):
    selection: VariantSelection = Field(default_factory=VariantSelection)
    group_by: Literal["combined", "region"] = "combined"
    metrics: list[VariantMetric] = Field(default_factory=_default_variant_metrics, min_length=1)
    extra_args: VariantExtraArgs = Field(default_factory=VariantExtraArgs)

    @model_validator(mode="after")
    def valid_scope(self) -> "VariantQCRequest":
        if self.group_by == "region" and self.regions is None:
            raise ValueError("group_by='region' requires regions")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must not contain duplicates")
        return self


class Provenance(ContractModel):
    backend: str = Field(min_length=1)
    backend_version: str | None = None
    effective_args: list[str] = Field(default_factory=list)
    native_options_used: bool = False
    measurement_scope: str = Field(min_length=1)


class NormalizedInterval(RegionalInterval):
    region_id: str = Field(min_length=1, description="Stable request-order identifier")


def _validate_normalized_regions(scope: str, regions: list[NormalizedInterval]) -> None:
    interval_scope = scope == "normalized_intervals"
    if interval_scope != bool(regions):
        raise ValueError("normalized_intervals scope requires regions; whole scope requires none")
    ids = [region.region_id for region in regions]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized region IDs must be unique")


class ReadEffectiveRequest(ContractModel):
    path: str
    reference_path: str | None = None
    region_scope: Literal["whole_file", "normalized_intervals"]
    normalized_regions: list[NormalizedInterval]
    resolved_group_by: Literal["combined", "region"]
    metrics: list[ReadMetric] = Field(min_length=1)
    selection: ReadSelection
    extra_args: ReadExtraArgs

    @model_validator(mode="after")
    def scope_matches_regions(self) -> "ReadEffectiveRequest":
        _validate_normalized_regions(self.region_scope, self.normalized_regions)
        return self


class AlignmentEffectiveRequest(ContractModel):
    path: str
    reference_path: str | None = None
    region_scope: Literal["whole_file", "normalized_intervals"]
    normalized_regions: list[NormalizedInterval]
    resolved_group_by: Literal["combined", "region"]
    metrics: list[AlignmentMetric] = Field(min_length=1)
    selection: AlignmentSelection
    extra_args: AlignmentExtraArgs

    @model_validator(mode="after")
    def scope_matches_regions(self) -> "AlignmentEffectiveRequest":
        _validate_normalized_regions(self.region_scope, self.normalized_regions)
        return self


class CoverageEffectiveRequest(ContractModel):
    path: str
    reference_path: str | None = None
    region_scope: Literal["whole_reference", "normalized_intervals"]
    normalized_regions: list[NormalizedInterval]
    resolved_group_by: CoverageGroup
    window_size: PositiveInt | None = None
    metrics: list[CoverageMetric] = Field(min_length=1)
    thresholds: list[NonNegativeInt] = Field(min_length=1)
    selection: CoverageSelection
    extra_args: CoverageExtraArgs

    @model_validator(mode="after")
    def scope_matches_regions(self) -> "CoverageEffectiveRequest":
        _validate_normalized_regions(self.region_scope, self.normalized_regions)
        if len(self.thresholds) != len(set(self.thresholds)):
            raise ValueError("thresholds must not contain duplicates")
        return self


class VariantEffectiveRequest(ContractModel):
    path: str
    reference_path: str | None = None
    region_scope: Literal["whole_file", "normalized_intervals"]
    normalized_regions: list[NormalizedInterval]
    resolved_group_by: Literal["combined", "region"]
    metrics: list[VariantMetric] = Field(min_length=1)
    selection: VariantSelection
    extra_args: VariantExtraArgs

    @model_validator(mode="after")
    def scope_matches_regions(self) -> "VariantEffectiveRequest":
        _validate_normalized_regions(self.region_scope, self.normalized_regions)
        return self


class ReadLengthSection(ContractModel):
    read_count: NonNegativeInt
    total_bases: NonNegativeInt
    min_length: NonNegativeInt | None = None
    max_length: NonNegativeInt | None = None
    mean_length: float | None = Field(default=None, ge=0)
    median_length: float | None = Field(default=None, ge=0)
    n50: NonNegativeInt | None = None

    @model_validator(mode="after")
    def empty_means_are_missing(self) -> "ReadLengthSection":
        values = (self.min_length, self.max_length, self.mean_length, self.median_length, self.n50)
        if self.read_count == 0 and any(value is not None for value in values):
            raise ValueError("length summaries must be null when read_count is zero")
        if self.read_count > 0 and any(value is None for value in values[:4]):
            raise ValueError("nonempty length summaries require min, max, mean, and median")
        return self


class ReadQualitySection(ContractModel):
    reads_with_quality: NonNegativeInt
    reads_missing_quality: NonNegativeInt
    mean_qscore: float | None = Field(default=None, ge=0)
    median_qscore: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def known_denominator_controls_mean(self) -> "ReadQualitySection":
        summaries_missing = self.mean_qscore is None and self.median_qscore is None
        summaries_complete = self.mean_qscore is not None and self.median_qscore is not None
        if self.reads_with_quality == 0 and not summaries_missing:
            raise ValueError("quality summaries must be null when no reads have quality")
        if self.reads_with_quality > 0 and not summaries_complete:
            raise ValueError("quality summaries require both mean and median when quality is present")
        return self


class LengthDistributionSection(ContractModel):
    percentiles: LengthPercentiles
    histogram: list[HistogramBin]


class QualityDistributionSection(ContractModel):
    histogram: list[HistogramBin]
    per_position_mean: list[float] | None = None


class ReadQCGroup(ContractModel):
    region_id: str | None = None
    region_name: str | None = None
    length: ReadLengthSection | None = None
    read_quality: ReadQualitySection | None = None
    length_distribution: LengthDistributionSection | None = None
    quality_distribution: QualityDistributionSection | None = None


class ReadQCResponse(ContractModel):
    tool: Literal["read_qc"] = "read_qc"
    resolved_group_by: Literal["combined", "region"]
    selected_records: NonNegativeInt
    emitted_sequences: NonNegativeInt
    conversion_exclusions: NonNegativeInt
    missing_sequence_policy: Literal["exclude_and_count"] = "exclude_and_count"
    missing_quality_policy: Literal["reject_when_quality_requested"] = "reject_when_quality_requested"
    effective_request: ReadEffectiveRequest
    results: list[ReadQCGroup] = Field(min_length=1)
    provenance: list[Provenance] = Field(min_length=1)

    @model_validator(mode="after")
    def requested_sections_and_grouping_match(self) -> "ReadQCResponse":
        if self.resolved_group_by != self.effective_request.resolved_group_by:
            raise ValueError("resolved grouping must match effective_request")
        if self.resolved_group_by == "combined" and len(self.results) != 1:
            raise ValueError("combined grouping requires exactly one result")
        if self.resolved_group_by == "region":
            expected_ids = [region.region_id for region in self.effective_request.normalized_regions]
            if [result.region_id for result in self.results] != expected_ids:
                raise ValueError("result region IDs and order must match normalized requested intervals")
        requested = set(self.effective_request.metrics)
        for result in self.results:
            if self.resolved_group_by == "region" and result.region_id is None:
                raise ValueError("region grouping requires region_id on every result")
            for metric in ("length", "read_quality", "length_distribution", "quality_distribution"):
                present = getattr(result, metric) is not None
                if present != (metric in requested):
                    raise ValueError(f"result section '{metric}' must match requested metrics")
        return self


class AlignmentCounts(ContractModel):
    eligible_records: NonNegativeInt
    mapped_records: NonNegativeInt
    unmapped_records: NonNegativeInt
    secondary_records: NonNegativeInt
    supplementary_records: NonNegativeInt


class MappingQualitySection(ContractModel):
    known_records: NonNegativeInt
    missing_records: NonNegativeInt
    mean_mapq: float | None = Field(default=None, ge=0, le=254)

    @model_validator(mode="after")
    def known_denominator_controls_mean(self) -> "MappingQualitySection":
        if (self.known_records == 0) != (self.mean_mapq is None):
            raise ValueError("mean_mapq is null exactly when no records have known MAPQ")
        return self


class AlignedBaseQualitySection(ContractModel):
    aligned_query_bases: NonNegativeInt
    known_quality_bases: NonNegativeInt
    missing_quality_bases: NonNegativeInt
    mean_base_quality: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def known_denominator_controls_mean(self) -> "AlignedBaseQualitySection":
        if self.known_quality_bases + self.missing_quality_bases != self.aligned_query_bases:
            raise ValueError("quality base denominators must sum to aligned_query_bases")
        if (self.known_quality_bases == 0) != (self.mean_base_quality is None):
            raise ValueError("mean_base_quality is null exactly when no aligned bases have quality")
        return self


class IdentitySection(ContractModel):
    records_with_identity: NonNegativeInt | None = Field(
        default=None, description="Unavailable when the backend reports only an aggregate"
    )
    records_missing_identity: NonNegativeInt | None = Field(
        default=None, description="Unavailable when the backend reports only an aggregate"
    )
    mean_identity: float | None = Field(default=None, ge=0, le=1)
    scope: Literal["whole_selected_record"] = "whole_selected_record"
    aggregation: Literal["cramino_backend_mean_normalized_to_fraction"] = "cramino_backend_mean_normalized_to_fraction"

    @model_validator(mode="after")
    def known_denominator_controls_mean(self) -> "IdentitySection":
        if self.records_with_identity == 0 and self.mean_identity is not None:
            raise ValueError("mean_identity must be null when no records have identity")
        if self.records_with_identity is not None and self.records_with_identity > 0 and self.mean_identity is None:
            raise ValueError("mean_identity is required when records have identity")
        return self


class ErrorProfileSection(ContractModel):
    records_with_nm: NonNegativeInt | None = Field(
        default=None, description="Unavailable from the current samtools stats aggregate"
    )
    records_missing_nm: NonNegativeInt | None = Field(
        default=None, description="Unavailable from the current samtools stats aggregate"
    )
    nm_error_rate: float | None = Field(default=None, ge=0)
    mismatch_rate: float | None = Field(
        default=None, ge=0, description="Specific mismatches-per-base value when samtools reports it"
    )
    insertion_rate: float | None = Field(default=None, ge=0)
    deletion_rate: float | None = Field(default=None, ge=0)
    coverage_histogram: list[CoverageBin] | None = None
    mismatch_counts_by_cycle: list[CycleMismatchCounts] | None = None
    insert_size_histogram: list[HistogramBin] | None = None
    scope: Literal["whole_selected_record"] = "whole_selected_record"

    @model_validator(mode="after")
    def known_denominator_controls_rate(self) -> "ErrorProfileSection":
        if self.records_with_nm == 0 and self.nm_error_rate is not None:
            raise ValueError("nm_error_rate must be null when no records have NM evidence")
        if self.records_with_nm is not None and self.records_with_nm > 0 and self.nm_error_rate is None:
            raise ValueError("nm_error_rate is required when records have NM evidence")
        return self


class AlignmentQCGroup(ContractModel):
    region_id: str | None = None
    region_name: str | None = None
    counts: AlignmentCounts | None = None
    mapping_quality: MappingQualitySection | None = None
    aligned_base_quality: AlignedBaseQualitySection | None = None
    identity: IdentitySection | None = None
    error_profile: ErrorProfileSection | None = None


class AlignmentQCResponse(ContractModel):
    tool: Literal["alignment_qc"] = "alignment_qc"
    resolved_group_by: Literal["combined", "region"]
    effective_request: AlignmentEffectiveRequest
    results: list[AlignmentQCGroup] = Field(min_length=1)
    provenance: list[Provenance] = Field(min_length=1)

    @model_validator(mode="after")
    def requested_sections_and_grouping_match(self) -> "AlignmentQCResponse":
        if self.resolved_group_by != self.effective_request.resolved_group_by:
            raise ValueError("resolved grouping must match effective_request")
        if self.resolved_group_by == "combined" and len(self.results) != 1:
            raise ValueError("combined grouping requires exactly one result")
        if self.resolved_group_by == "region":
            expected_ids = [region.region_id for region in self.effective_request.normalized_regions]
            if [result.region_id for result in self.results] != expected_ids:
                raise ValueError("result region IDs and order must match normalized requested intervals")
        requested = set(self.effective_request.metrics)
        for result in self.results:
            if self.resolved_group_by == "region" and result.region_id is None:
                raise ValueError("region grouping requires region_id on every result")
            for metric in ("counts", "mapping_quality", "aligned_base_quality", "identity", "error_profile"):
                if (getattr(result, metric) is not None) != (metric in requested):
                    raise ValueError(f"result section '{metric}' must match requested metrics")
        return self


class CoverageBreadth(ContractModel):
    threshold: NonNegativeInt
    bases_at_or_above: NonNegativeInt
    fraction_at_or_above: float = Field(ge=0, le=1)


class CoverageRow(ContractModel):
    row_id: str = Field(min_length=1)
    chrom: str = Field(min_length=1)
    start: NonNegativeInt
    end: PositiveInt
    name: str | None = None
    reference_bases: PositiveInt
    depth_sum: NonNegativeInt | None = None
    mean_depth: float | None = Field(default=None, ge=0)
    breadth: list[CoverageBreadth]

    @model_validator(mode="after")
    def interval_and_mean_are_consistent(self) -> "CoverageRow":
        if self.start >= self.end or self.reference_bases != self.end - self.start:
            raise ValueError("coverage row must describe one nonempty half-open interval")
        if (self.depth_sum is None) != (self.mean_depth is None):
            raise ValueError("depth_sum and mean_depth must be present or absent together")
        if (
            self.depth_sum is not None
            and self.mean_depth is not None
            and abs(self.mean_depth - self.depth_sum / self.reference_bases) > 1e-9
        ):
            raise ValueError("mean_depth must equal depth_sum/reference_bases")
        for value in self.breadth:
            if value.bases_at_or_above > self.reference_bases:
                raise ValueError("breadth count must not exceed reference_bases")
            expected_fraction = value.bases_at_or_above / self.reference_bases
            if value.fraction_at_or_above is None or abs(value.fraction_at_or_above - expected_fraction) > 1e-9:
                raise ValueError("breadth fraction must equal bases_at_or_above/reference_bases")
        return self


class CoverageUnionSummary(ContractModel):
    reference_bases: NonNegativeInt
    depth_sum: NonNegativeInt | None = None
    mean_depth: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def mean_matches_denominator(self) -> "CoverageUnionSummary":
        if (self.depth_sum is None) != (self.mean_depth is None):
            if not (self.reference_bases == 0 and self.depth_sum == 0 and self.mean_depth is None):
                raise ValueError("depth_sum and mean_depth must be present or absent together")
        if self.depth_sum is None and self.mean_depth is None:
            return self
        if self.reference_bases == 0:
            if self.depth_sum != 0 or self.mean_depth is not None:
                raise ValueError("empty union requires zero depth_sum and null mean_depth")
        elif (
            self.depth_sum is None
            or self.mean_depth is None
            or abs(self.mean_depth - self.depth_sum / self.reference_bases) > 1e-9
        ):
            raise ValueError("union mean_depth must equal depth_sum/reference_bases")
        return self


class CoverageQCResponse(ContractModel):
    tool: Literal["coverage_qc"] = "coverage_qc"
    resolved_group_by: CoverageGroup
    effective_request: CoverageEffectiveRequest
    rows: list[CoverageRow]
    union_summary: CoverageUnionSummary
    zero_depth_bases_retained: Literal[True] = True
    provenance: list[Provenance] = Field(min_length=1)

    @model_validator(mode="after")
    def requested_sections_and_grouping_match(self) -> "CoverageQCResponse":
        if self.resolved_group_by != self.effective_request.resolved_group_by:
            raise ValueError("resolved grouping must match effective_request")
        breadth_requested = "breadth" in self.effective_request.metrics
        depth_requested = "depth" in self.effective_request.metrics
        for row in self.rows:
            if breadth_requested != bool(row.breadth):
                raise ValueError("breadth rows must match requested metrics")
            if breadth_requested and [value.threshold for value in row.breadth] != self.effective_request.thresholds:
                raise ValueError("breadth thresholds and order must match effective_request")
            if depth_requested != (row.depth_sum is not None and row.mean_depth is not None):
                raise ValueError("depth fields must match requested metrics")
        union_has_depth = self.union_summary.depth_sum is not None or self.union_summary.mean_depth is not None
        if depth_requested != union_has_depth:
            raise ValueError("depth fields must match requested metrics")
        return self


class VariantGeneralSection(ContractModel):
    total_records: NonNegativeInt
    mnps: NonNegativeInt = 0
    others: NonNegativeInt = 0


class VariantSnpSection(ContractModel):
    count: NonNegativeInt
    transitions: NonNegativeInt | None = None
    transversions: NonNegativeInt | None = None
    ts_tv_ratio: float | None = Field(default=None, ge=0)


class VariantIndelSection(ContractModel):
    count: NonNegativeInt


class VariantQCGroup(ContractModel):
    region_id: str | None = None
    region_name: str | None = None
    general: VariantGeneralSection | None = None
    snps: VariantSnpSection | None = None
    indels: VariantIndelSection | None = None


class VariantQCResponse(ContractModel):
    tool: Literal["variant_qc"] = "variant_qc"
    resolved_group_by: Literal["combined", "region"]
    effective_request: VariantEffectiveRequest
    results: list[VariantQCGroup] = Field(min_length=1)
    provenance: list[Provenance] = Field(min_length=1)

    @model_validator(mode="after")
    def requested_sections_and_grouping_match(self) -> "VariantQCResponse":
        if self.resolved_group_by != self.effective_request.resolved_group_by:
            raise ValueError("resolved grouping must match effective_request")
        if self.resolved_group_by == "combined" and len(self.results) != 1:
            raise ValueError("combined grouping requires exactly one result")
        if self.resolved_group_by == "region":
            expected_ids = [region.region_id for region in self.effective_request.normalized_regions]
            if [result.region_id for result in self.results] != expected_ids:
                raise ValueError("result region IDs and order must match normalized requested intervals")
        requested = set(self.effective_request.metrics)
        for result in self.results:
            if self.resolved_group_by == "region" and result.region_id is None:
                raise ValueError("region grouping requires region_id on every result")
            for metric in ("general", "snps", "indels"):
                if (getattr(result, metric) is not None) != (metric in requested):
                    raise ValueError(f"result section '{metric}' must match requested metrics")
        return self


class PathRequest(ContractModel):
    path: str = Field(min_length=1)


class EnvironmentStatusRequest(ContractModel):
    pass


class HeaderInfoRequest(PathRequest):
    reference_path: str | None = Field(default=None, min_length=1)


class BedQCRequest(PathRequest):
    pass


class RunSummaryRequest(PathRequest):
    pass


class FilterReadSelection(ContractModel):
    headcrop: NonNegativeInt | None = None
    tailcrop: NonNegativeInt | None = None
    minlength: NonNegativeInt | None = None
    maxlength: NonNegativeInt | None = None
    quality: NonNegativeInt | None = None
    cutoff: NonNegativeInt | None = None
    trim_approach: Literal["fixed-crop", "trim-by-quality"] | None = None
    inverse: bool = False
    threads: PositiveInt | None = None

    @model_validator(mode="after")
    def valid_lengths_and_trim(self) -> "FilterReadSelection":
        if self.minlength is not None and self.maxlength is not None and self.minlength > self.maxlength:
            raise ValueError("minlength must not exceed maxlength")
        if (self.headcrop is not None or self.tailcrop is not None) and self.trim_approach != "fixed-crop":
            raise ValueError("headcrop and tailcrop require trim_approach='fixed-crop'")
        if self.cutoff is not None and self.trim_approach != "trim-by-quality":
            raise ValueError("cutoff requires trim_approach='trim-by-quality'")
        return self


class FilterReadsRequest(PathRequest):
    output_fastq: str | None = Field(default=None, min_length=1)
    selection: FilterReadSelection = Field(default_factory=FilterReadSelection)
    extra_args: dict[Literal["chopper"], list[str]] = Field(default_factory=dict)


class IgvSnapshotsRequest(ContractModel):
    batch_file: str | None = Field(default=None, min_length=1)
    genome: str | None = Field(default=None, min_length=1)
    tracks: list[str] | None = Field(default=None, min_length=1)
    regions: list[RegionalInterval] | str | None = None
    output_dir: str | None = Field(default=None, min_length=1)
    snapshot_format: Literal["png", "svg"] = "png"
    compact: Literal["expand", "collapse", "squish"] = "squish"
    color_by: str | None = Field(default=None, min_length=1)
    group_by: str | None = Field(default=None, min_length=1)
    min_snapshot_width: NonNegativeInt = 0
    small_indels_show: bool = False
    small_indels_threshold: NonNegativeInt = 100
    allele_threshold: float = Field(default=0.2, ge=0, le=1)
    extra_commands: list[str] = Field(default_factory=list)
    extra_preferences: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exactly_one_input_mode(self) -> "IgvSnapshotsRequest":
        dynamic_complete = self.genome is not None and self.tracks is not None and self.regions is not None
        dynamic_any = self.genome is not None or self.tracks is not None or self.regions is not None
        if self.batch_file is not None and dynamic_any:
            raise ValueError("batch_file cannot be combined with genome, tracks, or regions")
        if self.batch_file is None and not dynamic_complete:
            raise ValueError("provide batch_file or all of genome, tracks, and regions")
        if isinstance(self.regions, list) and not self.regions:
            raise ValueError("regions must not be empty")
        return self


class ValidationIssue(ContractModel):
    location: list[str | int]
    message: str = Field(min_length=1)
    code: str = Field(min_length=1)


class ValidationErrorResponse(ContractModel):
    kind: Literal["validation_error"] = "validation_error"
    tool: str = Field(min_length=1)
    message: str = Field(min_length=1)
    issues: list[ValidationIssue] = Field(min_length=1)


class ExecutionErrorResponse(ContractModel):
    kind: Literal["execution_error"] = "execution_error"
    tool: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    exit_code: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    partial_result_returned: Literal[False] = False


class ToolContract(ContractModel):
    name: str
    request_model: type[BaseModel]
    response_model: type[BaseModel]
    status: Literal["unregistered"] = "unregistered"

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


API_V2_CATALOG: tuple[ToolContract, ...] = (
    ToolContract(name="read_qc", request_model=ReadQCRequest, response_model=ReadQCResponse),
    ToolContract(name="alignment_qc", request_model=AlignmentQCRequest, response_model=AlignmentQCResponse),
    ToolContract(name="coverage_qc", request_model=CoverageQCRequest, response_model=CoverageQCResponse),
    ToolContract(name="variant_qc", request_model=VariantQCRequest, response_model=VariantQCResponse),
    ToolContract(name="environment_status", request_model=EnvironmentStatusRequest, response_model=EnvStatus),
    ToolContract(name="header_info", request_model=HeaderInfoRequest, response_model=HeaderMetadata),
    ToolContract(name="bed_qc", request_model=BedQCRequest, response_model=BedQCReport),
    ToolContract(name="run_summary", request_model=RunSummaryRequest, response_model=SequencingSummaryStats),
    ToolContract(name="filter_reads", request_model=FilterReadsRequest, response_model=ChopperReport),
    ToolContract(name="igv_snapshots", request_model=IgvSnapshotsRequest, response_model=IgvSnapshotResult),
)


def api_v2_contracts() -> dict[str, ToolContract]:
    """Return the proposed catalog keyed by its exact public names."""

    return {contract.name: contract for contract in API_V2_CATALOG}
