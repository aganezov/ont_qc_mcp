import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cli_wrappers import build_cli_args, chopper_filter, cramino_stats, mosdepth_coverage, nanoq_stats
from .config import ExecutionConfig, ToolPaths
from .parsers import parse_alignment_header, parse_error_profile, parse_vcf_header, summarize_header
from .schemas import (
    ChopperReport,
    CraminoStats,
    EnvStatus,
    ErrorProfile,
    HeaderMetadata,
    MosdepthStats,
    NanoqStats,
    QScoreDistribution,
    QCReport,
    ReadLengthDistribution,
    LengthPercentiles,
)
from .utils import CommandError, format_cmd, run_command


def env_check(tools: Optional[ToolPaths] = None) -> EnvStatus:
    tools = tools or ToolPaths()
    missing = tools.missing()
    resolved = tools.as_dict()
    available = {k: k not in missing for k in resolved.keys()}
    return EnvStatus(available=available, resolved_paths=resolved, missing=missing)


_EXEC_CFG = ExecutionConfig()


def qc_reads(
    path: str,
    tools: Optional[ToolPaths] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> NanoqStats:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    fastq_path = Path(path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ not found: {fastq_path}")
    return nanoq_stats(fastq_path, tools, flags=flags, exec_cfg=cfg)


def filter_reads(
    path: str,
    tools: Optional[ToolPaths] = None,
    output_fastq: Optional[str] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> ChopperReport:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    fastq_path = Path(path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ not found: {fastq_path}")
    output_path = Path(output_fastq) if output_fastq else None
    return chopper_filter(fastq_path, tools, output_fastq=output_path, flags=flags, exec_cfg=cfg)


def read_length_distribution(
    path: str,
    tools: Optional[ToolPaths] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> ReadLengthDistribution:
    cfg = exec_cfg or _EXEC_CFG
    stats = qc_reads(path, tools=tools, flags=flags, exec_cfg=cfg)
    return ReadLengthDistribution(
        file=stats.file,
        percentiles=stats.length_percentiles or LengthPercentiles(),
        histogram=stats.length_histogram or [],
    )


def qscore_distribution(
    path: str,
    tools: Optional[ToolPaths] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> QScoreDistribution:
    cfg = exec_cfg or _EXEC_CFG
    stats = qc_reads(path, tools=tools, flags=flags, exec_cfg=cfg)
    return QScoreDistribution(
        file=stats.file,
        mean_qscore=stats.mean_qscore,
        median_qscore=stats.median_qscore,
        histogram=stats.qscore_histogram or [],
        per_position_mean=None,
    )


def qc_alignment(
    path: str,
    tools: Optional[ToolPaths] = None,
    include_hist: bool = True,
    use_scaled: bool = False,
    flags: Optional[Dict[str, Any]] = None,
) -> CraminoStats:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment not found: {aln_path}")
    return cramino_stats(
        aln_path,
        tools,
        include_hist=include_hist,
        use_scaled=use_scaled,
        flags=flags,
        exec_cfg=_EXEC_CFG,
    )


def coverage_stats(
    path: str,
    tools: Optional[ToolPaths] = None,
    window: Optional[int] = None,
    flags: Optional[Dict[str, Any]] = None,
) -> MosdepthStats:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment not found: {aln_path}")
    return mosdepth_coverage(aln_path, tools, window=window, flags=flags, exec_cfg=_EXEC_CFG)


def alignment_error_profile(
    path: str,
    tools: Optional[ToolPaths] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> ErrorProfile:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    aln_path = Path(path)
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment not found: {aln_path}")

    flag_data: Dict[str, Any] = dict(flags or {})
    flag_data.setdefault("threads", cfg.threads_for("samtools"))
    flag_args = build_cli_args("samtools", flag_data)
    cmd = [tools.samtools, *flag_args, "stats", str(aln_path)]
    try:
        result = run_command(cmd, timeout=cfg.timeout_for("samtools"))
    except CommandError as exc:
        raise RuntimeError(f"samtools stats failed: {format_cmd(exc.result.cmd)}\n{exc.result.stderr}") from exc
    return parse_error_profile(result.stdout, file_path=str(aln_path))


def alignment_summary(
    path: str,
    include_coverage: bool = True,
    include_hist: bool = True,
    use_scaled: bool = False,
    coverage_window: Optional[int] = None,
    coverage_flags: Optional[Dict[str, Any]] = None,
    cramino_flags: Optional[Dict[str, Any]] = None,
    error_profile_flags: Optional[Dict[str, Any]] = None,
    tools: Optional[ToolPaths] = None,
) -> QCReport:
    tools = tools or ToolPaths()
    aln_stats = qc_alignment(path, tools=tools, include_hist=include_hist, use_scaled=use_scaled, flags=cramino_flags)
    coverage = coverage_stats(path, tools=tools, window=coverage_window, flags=coverage_flags) if include_coverage else None
    errors = alignment_error_profile(path, tools=tools, flags=error_profile_flags, exec_cfg=_EXEC_CFG)
    return QCReport(alignment=aln_stats, coverage=coverage, errors=errors)


def _detect_header_format(file_path: Path, file_type: Optional[str]) -> str:
    """Infer the header format from the provided type or filename."""
    if file_type:
        normalized = file_type.lower()
        if normalized not in {"bam", "cram", "sam", "vcf"}:
            raise ValueError(f"Unsupported file type: {file_type}")
        return normalized

    name = file_path.name.lower()
    if name.endswith(".bam"):
        return "bam"
    if name.endswith(".cram"):
        return "cram"
    if name.endswith(".sam"):
        return "sam"
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return "vcf"
    raise ValueError(f"Unable to infer file type from extension: {file_path}")


def _read_alignment_header_text(
    path: Path,
    tools: ToolPaths,
    flags: Optional[Dict[str, Any]],
    exec_cfg: ExecutionConfig,
) -> str:
    flag_data: Dict[str, Any] = dict(flags or {})
    flag_data.setdefault("threads", exec_cfg.threads_for("samtools"))
    flag_args = build_cli_args("samtools", flag_data)
    cmd = [tools.samtools, *flag_args, "view", "-H", str(path)]
    try:
        result = run_command(cmd, timeout=exec_cfg.timeout_for("samtools"))
    except CommandError as exc:
        raise RuntimeError(f"samtools view -H failed: {format_cmd(exc.result.cmd)}\n{exc.result.stderr}") from exc
    return result.stdout


DEFAULT_VCF_HEADER_MAX_LINES = 2000


def _read_vcf_header_text(path: Path, max_lines: Optional[int] = DEFAULT_VCF_HEADER_MAX_LINES) -> str:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    header_lines: List[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if max_lines is not None and len(header_lines) >= max_lines:
                break
            if not line.startswith("#"):
                break
            header_lines.append(line.rstrip("\n"))
            if line.startswith("#CHROM"):
                break
    if not header_lines:
        raise ValueError(f"No VCF header lines found in {path}")
    return "\n".join(header_lines)


def header_metadata_lookup(
    path: str,
    file_type: Optional[str] = None,
    flags: Optional[Dict[str, Any]] = None,
    tools: Optional[ToolPaths] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
    max_lines: Optional[int] = DEFAULT_VCF_HEADER_MAX_LINES,
) -> HeaderMetadata:
    """
    Extract header metadata for BAM/CRAM/VCF inputs and return structured data with a summary.
    """
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    fmt = _detect_header_format(file_path, file_type)
    if fmt in {"bam", "cram", "sam"}:
        header_text = _read_alignment_header_text(file_path, tools, flags, cfg)
        metadata = parse_alignment_header(header_text, file_path=str(file_path), fmt=fmt)
    elif fmt == "vcf":
        header_text = _read_vcf_header_text(file_path, max_lines=max_lines)
        metadata = parse_vcf_header(header_text, file_path=str(file_path))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    summarize_header(metadata)
    return metadata


def serialize_model(model) -> Dict[str, object]:
    """Return a JSON-serializable dict from a pydantic model."""
    return model.model_dump()

