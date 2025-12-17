from typing import Literal

from pydantic import BaseModel, Field


class EnvStatus(BaseModel):
    available: dict[str, bool]
    resolved_paths: dict[str, str]
    missing: list[str]
    igv_runtime: str | None = Field(default=None, description="Detected IGV runtime (docker/apptainer/local)")


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class LengthPercentiles(BaseModel):
    p1: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None


class NanoqStats(BaseModel):
    file: str
    read_count: int
    total_bases: int
    min_len: int
    max_len: int
    mean_len: float
    median_len: float
    n50: int | None = None
    mean_qscore: float | None = None
    median_qscore: float | None = None
    gc_content: float | None = Field(default=None, description="GC proportion (0-1)")
    length_percentiles: LengthPercentiles | None = None
    length_histogram: list[HistogramBin] | None = None
    qscore_histogram: list[HistogramBin] | None = None


class ReadLengthDistribution(BaseModel):
    file: str
    percentiles: LengthPercentiles
    histogram: list[HistogramBin]


class QScoreDistribution(BaseModel):
    file: str
    mean_qscore: float | None = None
    median_qscore: float | None = None
    histogram: list[HistogramBin]
    per_position_mean: list[float] | None = Field(default=None, description="Mean Q per cycle (sampled)")


class ChopperReport(BaseModel):
    input_reads: int | None = None
    output_reads: int | None = None
    filtered_reads: int | None = None
    command: list[str]
    params: dict[str, object] = Field(default_factory=dict)
    output_fastq: str | None = None


class CraminoStats(BaseModel):
    file: str
    total_reads: int
    mapped: int | None = None
    unmapped: int | None = None
    primary: int | None = None
    secondary: int | None = None
    supplementary: int | None = None
    mean_length: float | None = None
    median_length: float | None = None
    n50: int | None = None
    mean_identity: float | None = None
    median_identity: float | None = None
    length_histogram: list[HistogramBin] | None = None  # count-based bins
    length_histogram_scaled: list[HistogramBin] | None = None  # reserved for future scaled support
    mapq_histogram: list[HistogramBin] | None = None  # count-based bins
    mapq_histogram_scaled: list[HistogramBin] | None = None  # base-weighted bins when --scaled


class CoverageByContig(BaseModel):
    contig: str
    mean_depth: float
    median_depth: float | None = None
    length: int | None = None


class LowCoverageRegion(BaseModel):
    contig: str
    start: int
    end: int
    mean_depth: float


class MosdepthStats(BaseModel):
    file: str
    mean_depth: float
    mean_depth_unweighted: float | None = None
    median_depth: float | None = None
    coverage_distribution: list[HistogramBin] = Field(default_factory=list)
    coverage_by_contig: list[CoverageByContig] = Field(default_factory=list)
    low_coverage_regions: list[LowCoverageRegion] = Field(default_factory=list)


class ErrorProfile(BaseModel):
    file: str
    mismatch_rate: float | None = None
    insertion_rate: float | None = None
    deletion_rate: float | None = None
    error_by_position: list[float] | None = Field(default=None, description="Error rate per position (sampled)")
    coverage_histogram: list[HistogramBin] | None = Field(
        default=None, description="Coverage distribution from samtools stats (depth -> bases)"
    )
    gc_coverage: list[HistogramBin] | None = Field(
        default=None, description="GC coverage distribution (GC% -> depth/bases)"
    )
    mismatch_by_cycle: list[float] | None = Field(
        default=None, description="Mismatch rate per cycle from samtools stats"
    )
    insert_size_histogram: list[HistogramBin] | None = Field(
        default=None, description="Insert size distribution when available"
    )


class ProgramRecord(BaseModel):
    id: str
    name: str | None = None
    version: str | None = None
    command: str | None = None
    previous_id: str | None = None
    other: dict[str, str] = Field(default_factory=dict)


class ReferenceRecord(BaseModel):
    name: str
    length: int | None = None
    assembly: str | None = None
    uri: str | None = None
    md5: str | None = None
    other: dict[str, str] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    name: str
    read_group_id: str | None = None
    library: str | None = None
    platform: str | None = None
    platform_unit: str | None = None
    sequencing_center: str | None = None
    run_date: str | None = None
    flowcell_id: str | None = None
    platform_model: str | None = None
    description: str | None = None
    other: dict[str, str] = Field(default_factory=dict)


class VCFFieldDef(BaseModel):
    id: str
    number: str | None = None
    type: str | None = None
    description: str | None = None
    category: Literal["INFO", "FORMAT", "FILTER"]
    other: dict[str, str] = Field(default_factory=dict)


class HeaderMetadata(BaseModel):
    file: str
    format: str
    references: list[ReferenceRecord] = Field(default_factory=list)
    programs: list[ProgramRecord] = Field(default_factory=list)
    samples: list[SampleRecord] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    info_fields: list[VCFFieldDef] = Field(default_factory=list)
    format_fields: list[VCFFieldDef] = Field(default_factory=list)
    filter_fields: list[VCFFieldDef] = Field(default_factory=list)
    header_other: dict[str, str] = Field(default_factory=dict)
    comments: list[str] = Field(default_factory=list)
    raw_header: str
    summary: str | None = None


class QCReport(BaseModel):
    reads: NanoqStats | None = None
    read_lengths: ReadLengthDistribution | None = None
    qscores: QScoreDistribution | None = None
    chopper: ChopperReport | None = None
    alignment: CraminoStats | None = None
    coverage: MosdepthStats | None = None
    errors: ErrorProfile | None = None


class IgvRegion(BaseModel):
    """A genomic region to snapshot, with optional per-region IGV commands."""

    chrom: str
    start: int
    end: int
    name: str | None = None  # Optional filename (without extension)
    extra_commands: list[str] = Field(
        default_factory=list,
        description="IGV commands to execute before this snapshot (e.g., 'sort BASE', 'viewaspairs')",
    )


class IgvSnapshotResult(BaseModel):
    """Result of an IGV snapshot run."""

    snapshot_files: list[str]
    batch_file: str
    output_directory: str
    execution_mode: Literal["docker", "apptainer", "local"]
    command: list[str]


# Sequencing Summary
class RunYieldWindow(BaseModel):
    window_start_hours: float
    yield_bp: int
    read_count: int


class SequencingSummaryStats(BaseModel):
    file: str
    total_yield: int
    total_reads: int
    n50: int | None = None
    mean_length: float | None = None
    mean_qscore: float | None = None
    active_channels: int | None = None
    run_duration_hours: float | None = None
    yield_per_hour: list[RunYieldWindow] = Field(default_factory=list)


# VCF QC
class SNPStats(BaseModel):
    count: int
    ts_tv_ratio: float | None = None


class IndelStats(BaseModel):
    count: int


class VariantGeneralStats(BaseModel):
    total_records: int
    mnps: int = 0
    others: int = 0
    singletons: int | None = None
    sample_name: str | None = None


class VCFStats(BaseModel):
    file: str
    general: VariantGeneralStats
    snps: SNPStats | None = None
    indels: IndelStats | None = None


# Targeted Coverage
class TargetedCoverageReport(BaseModel):
    region_name: str
    chrom: str
    start: int
    end: int
    mean_depth: float
    min_depth: int | None = None
    max_depth: int | None = None
    pct_coverage_1x: float | None = None
    pct_coverage_10x: float | None = None
    pct_coverage_20x: float | None = None


# BED QC
class BedIssue(BaseModel):
    line_number: int
    line_content: str
    issue: str


class BedQCReport(BaseModel):
    file: str
    total_intervals: int
    valid_intervals: int
    total_bases: int
    is_valid: bool
    issues: list[BedIssue] = Field(default_factory=list)


__all__ = [
    "ChopperReport",
    "CoverageByContig",
    "CraminoStats",
    "EnvStatus",
    "ErrorProfile",
    "HeaderMetadata",
    "HistogramBin",
    "LengthPercentiles",
    "LowCoverageRegion",
    "MosdepthStats",
    "NanoqStats",
    "ProgramRecord",
    "QCReport",
    "QScoreDistribution",
    "ReadLengthDistribution",
    "ReferenceRecord",
    "SampleRecord",
    "VCFFieldDef",
    "IgvRegion",
    "IgvSnapshotResult",
    "RunYieldWindow",
    "SequencingSummaryStats",
    "SNPStats",
    "IndelStats",
    "VariantGeneralStats",
    "VCFStats",
    "TargetedCoverageReport",
    "BedIssue",
    "BedQCReport",
]
