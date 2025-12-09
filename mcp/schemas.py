from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SeqkitStats(BaseModel):
    file: str
    num_seqs: int
    total_bases: int
    min_len: int
    avg_len: float
    max_len: int
    n50: Optional[int] = Field(default=None)
    gc: Optional[float] = Field(default=None, description="GC proportion (0-1)")


class SamtoolsStats(BaseModel):
    raw_total_sequences: Optional[int] = None
    filtered_sequences: Optional[int] = None
    reads_mapped: Optional[int] = None
    reads_properly_paired: Optional[int] = None
    bases_mapped: Optional[int] = None
    error_rate: Optional[float] = None
    average_length: Optional[float] = None
    insert_size_average: Optional[float] = None
    other_metrics: Dict[str, str] = Field(default_factory=dict)


class FastpReport(BaseModel):
    summary: Dict[str, object]
    command: List[str] = Field(description="Executed fastp command")
    output_fastq: Optional[str] = Field(default=None, description="Path to filtered reads if kept")


class NanoPlotReport(BaseModel):
    summary: Dict[str, object]
    command: List[str] = Field(description="Executed NanoPlot command")


class FastqEdaReport(BaseModel):
    seqkit: SeqkitStats
    nanoplot: Optional[NanoPlotReport] = None


class EnvStatus(BaseModel):
    available: Dict[str, bool]
    resolved_paths: Dict[str, str]
    missing: List[str]

