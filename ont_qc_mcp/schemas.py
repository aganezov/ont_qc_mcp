from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EnvStatus(BaseModel):
    available: Dict[str, bool]
    resolved_paths: Dict[str, str]
    missing: List[str]


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class LengthPercentiles(BaseModel):
    p1: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None


class NanoqStats(BaseModel):
    file: str
    read_count: int
    total_bases: int
    min_len: int
    max_len: int
    mean_len: float
    median_len: float
    n50: Optional[int] = None
    mean_qscore: Optional[float] = None
    median_qscore: Optional[float] = None
    gc_content: Optional[float] = Field(default=None, description="GC proportion (0-1)")
    length_percentiles: Optional[LengthPercentiles] = None
    length_histogram: Optional[List[HistogramBin]] = None
    qscore_histogram: Optional[List[HistogramBin]] = None


class ReadLengthDistribution(BaseModel):
    file: str
    percentiles: LengthPercentiles
    histogram: List[HistogramBin]


class QScoreDistribution(BaseModel):
    file: str
    mean_qscore: Optional[float] = None
    median_qscore: Optional[float] = None
    histogram: List[HistogramBin]
    per_position_mean: Optional[List[float]] = Field(default=None, description="Mean Q per cycle (sampled)")


class ChopperReport(BaseModel):
    input_reads: Optional[int] = None
    output_reads: Optional[int] = None
    filtered_reads: Optional[int] = None
    command: List[str]
    params: Dict[str, object] = Field(default_factory=dict)
    output_fastq: Optional[str] = None


class CraminoStats(BaseModel):
    file: str
    total_reads: int
    mapped: Optional[int] = None
    unmapped: Optional[int] = None
    primary: Optional[int] = None
    secondary: Optional[int] = None
    supplementary: Optional[int] = None
    mean_length: Optional[float] = None
    median_length: Optional[float] = None
    n50: Optional[int] = None
    mean_identity: Optional[float] = None
    median_identity: Optional[float] = None
    length_histogram: Optional[List[HistogramBin]] = None  # count-based bins
    length_histogram_scaled: Optional[List[HistogramBin]] = None  # reserved for future scaled support
    mapq_histogram: Optional[List[HistogramBin]] = None  # count-based bins
    mapq_histogram_scaled: Optional[List[HistogramBin]] = None  # base-weighted bins when --scaled


class CoverageByContig(BaseModel):
    contig: str
    mean_depth: float
    median_depth: Optional[float] = None
    length: Optional[int] = None


class LowCoverageRegion(BaseModel):
    contig: str
    start: int
    end: int
    mean_depth: float


class MosdepthStats(BaseModel):
    file: str
    mean_depth: float
    mean_depth_unweighted: Optional[float] = None
    median_depth: Optional[float] = None
    coverage_distribution: List[HistogramBin] = Field(default_factory=list)
    coverage_by_contig: List[CoverageByContig] = Field(default_factory=list)
    low_coverage_regions: List[LowCoverageRegion] = Field(default_factory=list)


class ErrorProfile(BaseModel):
    file: str
    mismatch_rate: Optional[float] = None
    insertion_rate: Optional[float] = None
    deletion_rate: Optional[float] = None
    error_by_position: Optional[List[float]] = Field(default=None, description="Error rate per position (sampled)")
    coverage_histogram: Optional[List[HistogramBin]] = Field(
        default=None, description="Coverage distribution from samtools stats (depth -> bases)"
    )
    gc_coverage: Optional[List[HistogramBin]] = Field(
        default=None, description="GC coverage distribution (GC% -> depth/bases)"
    )
    mismatch_by_cycle: Optional[List[float]] = Field(
        default=None, description="Mismatch rate per cycle from samtools stats"
    )
    insert_size_histogram: Optional[List[HistogramBin]] = Field(
        default=None, description="Insert size distribution when available"
    )


class ProgramRecord(BaseModel):
    id: str
    name: Optional[str] = None
    version: Optional[str] = None
    command: Optional[str] = None
    previous_id: Optional[str] = None
    other: Dict[str, str] = Field(default_factory=dict)


class ReferenceRecord(BaseModel):
    name: str
    length: Optional[int] = None
    assembly: Optional[str] = None
    uri: Optional[str] = None
    md5: Optional[str] = None
    other: Dict[str, str] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    name: str
    read_group_id: Optional[str] = None
    library: Optional[str] = None
    platform: Optional[str] = None
    platform_unit: Optional[str] = None
    sequencing_center: Optional[str] = None
    run_date: Optional[str] = None
    flowcell_id: Optional[str] = None
    platform_model: Optional[str] = None
    description: Optional[str] = None
    other: Dict[str, str] = Field(default_factory=dict)


class VCFFieldDef(BaseModel):
    id: str
    number: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    category: Literal["INFO", "FORMAT", "FILTER"]
    other: Dict[str, str] = Field(default_factory=dict)


class HeaderMetadata(BaseModel):
    file: str
    format: str
    references: List[ReferenceRecord] = Field(default_factory=list)
    programs: List[ProgramRecord] = Field(default_factory=list)
    samples: List[SampleRecord] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
    info_fields: List[VCFFieldDef] = Field(default_factory=list)
    format_fields: List[VCFFieldDef] = Field(default_factory=list)
    filter_fields: List[VCFFieldDef] = Field(default_factory=list)
    header_other: Dict[str, str] = Field(default_factory=dict)
    comments: List[str] = Field(default_factory=list)
    raw_header: str
    summary: Optional[str] = None


class QCReport(BaseModel):
    reads: Optional[NanoqStats] = None
    read_lengths: Optional[ReadLengthDistribution] = None
    qscores: Optional[QScoreDistribution] = None
    chopper: Optional[ChopperReport] = None
    alignment: Optional[CraminoStats] = None
    coverage: Optional[MosdepthStats] = None
    errors: Optional[ErrorProfile] = None


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
]

