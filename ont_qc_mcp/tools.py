import gzip
import logging
import shutil
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel

from .cli_wrappers import (
    build_cli_args,
    chopper_filter,
    cramino_stats,
    mosdepth_coverage,
    nanoq_from_bam_streaming,
    nanoq_stats,
)
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
from .utils import CommandError, _truncate_stderr, format_cmd, report_progress, run_command


logger = logging.getLogger(__name__)


def env_check(tools: ToolPaths | None = None) -> EnvStatus:
    tools = tools or ToolPaths()
    missing = tools.missing()
    resolved = tools.resolved()
    available = {k: k not in missing for k in resolved.keys()}
    return EnvStatus(available=available, resolved_paths=resolved, missing=missing)


_EXEC_CFG = ExecutionConfig()
_NANOQ_CACHE: dict[tuple, NanoqStats] = {}
_NANOQ_CACHE_MAX = 8
_NANOQ_CACHE_LOCK = threading.Lock()
_NANOQ_INFLIGHT: dict[tuple, Future[NanoqStats]] = {}


def _nanoq_cache_key(path: Path, flags: dict[str, Any] | None, cfg: ExecutionConfig) -> tuple:
    stat = path.stat()
    normalized_flags = tuple(sorted((flags or {}).items()))
    return (
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        normalized_flags,
        cfg.timeout_for("nanoq"),
        cfg.threads_for("nanoq"),
    )


def _validate_input_file(path: Path, cfg: ExecutionConfig, allowed_exts: tuple[str, ...] | None = None) -> None:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")
    size = resolved.stat().st_size
    if size == 0:
        raise ValueError(f"File is empty: {resolved}")
    if cfg.max_file_size_bytes and size > cfg.max_file_size_bytes:
        raise ValueError(f"File exceeds configured size limit ({cfg.max_file_size_bytes} bytes): {resolved}")
    if allowed_exts:
        lowered_name = resolved.name.lower()
        if not any(lowered_name.endswith(ext) for ext in allowed_exts):
            raise ValueError(f"Unexpected file extension for {resolved}; expected one of {allowed_exts}")

def _cached_nanoq_stats(
    path: Path, tools: ToolPaths, flags: dict[str, Any] | None, cfg: ExecutionConfig
) -> NanoqStats:
    key = _nanoq_cache_key(path, flags, cfg)
    with _NANOQ_CACHE_LOCK:
        cached = _NANOQ_CACHE.get(key)
        if cached:
            logger.debug("nanoq cache hit for %s", path)
            return cached

        future = _NANOQ_INFLIGHT.get(key)
        if future is None:
            future = Future[NanoqStats]()
            _NANOQ_INFLIGHT[key] = future
            is_owner = True
        else:
            is_owner = False

    if not is_owner:
        # Another thread is computing this key; wait for it and re-use the result/exception.
        logger.debug("nanoq cache inflight wait for %s", path)
        return future.result()

    try:
        logger.debug("nanoq cache miss, computing for %s", path)
        stats = nanoq_stats(path, tools, flags=flags, exec_cfg=cfg)
        if not stats.file or stats.file == "unknown":
            stats.file = str(path)

        # Simple bounded cache to avoid unbounded growth.
        with _NANOQ_CACHE_LOCK:
            if len(_NANOQ_CACHE) >= _NANOQ_CACHE_MAX:
                _NANOQ_CACHE.pop(next(iter(_NANOQ_CACHE)))
            _NANOQ_CACHE[key] = stats

        future.set_result(stats)
        return stats
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        with _NANOQ_CACHE_LOCK:
            if _NANOQ_INFLIGHT.get(key) is future:
                _NANOQ_INFLIGHT.pop(key, None)


def qc_reads(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> NanoqStats:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    fastq_path = Path(path)
    _validate_input_file(fastq_path, cfg, allowed_exts=(".fastq", ".fq", ".fastq.gz", ".fq.gz"))
    report_progress(f"qc_reads start: {fastq_path}")
    logger.debug("qc_reads on %s", fastq_path)
    result = _cached_nanoq_stats(fastq_path, tools, flags=flags, cfg=cfg)
    report_progress(f"qc_reads done: {fastq_path}")
    return result


def filter_reads(
    path: str,
    tools: ToolPaths | None = None,
    output_fastq: str | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ChopperReport:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    fastq_path = Path(path)
    _validate_input_file(fastq_path, cfg, allowed_exts=(".fastq", ".fq", ".fastq.gz", ".fq.gz"))
    output_path = Path(output_fastq) if output_fastq else None
    logger.debug("filter_reads on %s -> %s", fastq_path, output_path or "<temp>")
    report_progress(f"filter_reads start: {fastq_path}")
    result = chopper_filter(fastq_path, tools, output_fastq=output_path, flags=flags, exec_cfg=cfg)
    report_progress(f"filter_reads done: {fastq_path}")
    return result


def read_length_distribution(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
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
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
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


async def read_length_distribution_bam(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ReadLengthDistribution:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    _validate_input_file(aln_path, exec_cfg or _EXEC_CFG, allowed_exts=(".bam", ".cram", ".sam"))
    stats = await anyio.to_thread.run_sync(
        nanoq_from_bam_streaming, aln_path, tools, flags or {}, exec_cfg or _EXEC_CFG
    )
    return ReadLengthDistribution(
        file=stats.file,
        percentiles=stats.length_percentiles or LengthPercentiles(),
        histogram=stats.length_histogram or [],
    )


async def qscore_distribution_bam(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> QScoreDistribution:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    _validate_input_file(aln_path, exec_cfg or _EXEC_CFG, allowed_exts=(".bam", ".cram", ".sam"))
    stats = await anyio.to_thread.run_sync(
        nanoq_from_bam_streaming, aln_path, tools, flags or {}, exec_cfg or _EXEC_CFG
    )
    return QScoreDistribution(
        file=stats.file,
        mean_qscore=stats.mean_qscore,
        median_qscore=stats.median_qscore,
        histogram=stats.qscore_histogram or [],
        per_position_mean=None,
    )


def qc_alignment(
    path: str,
    tools: ToolPaths | None = None,
    include_hist: bool = True,
    use_scaled: bool = False,
    flags: dict[str, Any] | None = None,
) -> CraminoStats:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    _validate_input_file(aln_path, _EXEC_CFG, allowed_exts=(".bam", ".cram", ".sam"))
    report_progress(f"qc_alignment start: {aln_path}")
    result = cramino_stats(
        aln_path,
        tools,
        include_hist=include_hist,
        use_scaled=use_scaled,
        flags=flags,
        exec_cfg=_EXEC_CFG,
    )
    report_progress(f"qc_alignment done: {aln_path}")
    return result


def coverage_stats(
    path: str,
    tools: ToolPaths | None = None,
    window: int | None = None,
    low_cov_threshold: float | None = None,
    flags: dict[str, Any] | None = None,
) -> MosdepthStats:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    _validate_input_file(aln_path, _EXEC_CFG, allowed_exts=(".bam", ".cram", ".sam"))
    report_progress(f"coverage_stats start: {aln_path}")
    result = mosdepth_coverage(
        aln_path,
        tools,
        window=window,
        low_cov_threshold=low_cov_threshold,
        flags=flags,
        exec_cfg=_EXEC_CFG,
    )
    report_progress(f"coverage_stats done: {aln_path}")
    return result


def alignment_error_profile(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ErrorProfile:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    aln_path = Path(path)
    _validate_input_file(aln_path, cfg, allowed_exts=(".bam", ".cram", ".sam"))

    flag_data: dict[str, Any] = dict(flags or {})
    flag_data.setdefault("threads", cfg.threads_for("samtools"))
    flag_args = build_cli_args("samtools", flag_data)
    cmd = [tools.samtools, "stats", *flag_args, str(aln_path)]
    report_progress(f"alignment_error_profile start: {aln_path}")
    try:
        logger.debug("alignment_error_profile via samtools stats: %s", format_cmd(cmd))
        result = run_command(cmd, timeout=cfg.timeout_for("samtools"))
    except CommandError as exc:
        raise RuntimeError(
            f"samtools stats failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
        ) from exc
    parsed = parse_error_profile(result.stdout, file_path=str(aln_path))
    report_progress(f"alignment_error_profile done: {aln_path}")
    return parsed


def alignment_summary(
    path: str,
    include_coverage: bool = True,
    include_hist: bool = True,
    use_scaled: bool = False,
    include_error_profile: bool = False,
    coverage_window: int | None = None,
    coverage_low_cov_threshold: float | None = None,
    coverage_flags: dict[str, Any] | None = None,
    cramino_flags: dict[str, Any] | None = None,
    error_profile_flags: dict[str, Any] | None = None,
    tools: ToolPaths | None = None,
) -> QCReport:
    tools = tools or ToolPaths()
    report_progress(f"alignment_summary start: {path}")
    aln_stats = qc_alignment(path, tools=tools, include_hist=include_hist, use_scaled=use_scaled, flags=cramino_flags)
    coverage = coverage_stats(
        path,
        tools=tools,
        window=coverage_window,
        low_cov_threshold=coverage_low_cov_threshold,
        flags=coverage_flags,
    ) if include_coverage else None
    errors = (
        alignment_error_profile(path, tools=tools, flags=error_profile_flags, exec_cfg=_EXEC_CFG)
        if include_error_profile
        else None
    )
    report_progress(f"alignment_summary done: {path}")
    return QCReport(alignment=aln_stats, coverage=coverage, errors=errors)


def _normalize_declared_format(file_type: str | None) -> str | None:
    if not file_type:
        return None
    normalized = file_type.lower()
    if normalized not in {"bam", "cram", "sam", "vcf"}:
        raise ValueError(f"Unsupported file type: {file_type}")
    return normalized


def _detect_header_format(file_path: Path) -> str | None:
    """Infer the header format from the filename extension only."""
    name = file_path.name.lower()
    if name.endswith(".bam"):
        return "bam"
    if name.endswith(".cram"):
        return "cram"
    if name.endswith(".sam"):
        return "sam"
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return "vcf"
    return None


def _sniff_header_format(file_path: Path) -> str | None:
    """
    Use cheap magic-byte or header sniffing to guess format.
    Returns None when inconclusive.
    """
    try:
        with open(file_path, "rb") as fh:
            prefix = fh.read(32)
    except OSError:
        return None

    # CRAM starts with literal CRAM magic.
    if prefix.startswith(b"CRAM"):
        return "cram"

    is_gzip = prefix.startswith(b"\x1f\x8b")

    # BAM is BGZF; decompress a few bytes to check for BAM\\1 magic.
    if is_gzip or prefix.startswith(b"BAM\1"):
        try:
            with gzip.open(file_path, "rb") as gf:
                inner = gf.read(4)
                if inner.startswith(b"BAM\1"):
                    return "bam"
        except OSError:
            # Could still be another gzip; keep sniffing.
            pass

    # For gzip VCF, inspect first few text lines.
    if is_gzip:
        try:
            with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as gf:
                for _ in range(8):
                    line = gf.readline()
                    if not line:
                        break
                    if line.startswith("##fileformat=VCF") or line.startswith("#CHROM"):
                        return "vcf"
        except OSError:
            return None
    else:
        # Plain text inputs: check a small prefix.
        try:
            text = prefix.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return None
        for line in text.splitlines():
            if line.startswith("@HD") or line.startswith("@SQ"):
                return "sam"
            if line.startswith("##fileformat=VCF") or line.startswith("#CHROM"):
                return "vcf"

    return None


def _infer_header_format(file_path: Path, file_type: str | None) -> str:
    """
    Combine caller hint, sniffing, and extension to decide format.
    Raise friendly errors when mismatched.
    """
    declared = _normalize_declared_format(file_type)
    sniffed = _sniff_header_format(file_path)
    ext_guess = _detect_header_format(file_path)

    if declared:
        if sniffed and sniffed != declared:
            raise ValueError(
                f"Provided file_type '{declared}' disagrees with detected format '{sniffed}' for {file_path}"
            )
        return declared

    if sniffed:
        if ext_guess and sniffed != ext_guess:
            raise ValueError(
                f"File extension suggests '{ext_guess}' but contents look like '{sniffed}' for {file_path}. "
                "Pass file_type to override if this is intentional."
            )
        return sniffed

    if ext_guess:
        return ext_guess

    raise ValueError(f"Unable to determine file type for {file_path}; provide file_type explicitly.")


def _maybe_quickcheck(path: Path, tools: ToolPaths, fmt: str, exec_cfg: ExecutionConfig) -> None:
    """Optionally validate BAM/CRAM with samtools quickcheck when available."""
    if fmt not in {"bam", "cram"}:
        return
    samtools_path = shutil.which(tools.samtools)
    if not samtools_path:
        return

    cmd = [samtools_path, "quickcheck", "-v", str(path)]
    try:
        run_command(cmd, timeout=exec_cfg.timeout_for("samtools"))
    except CommandError as exc:
        raise RuntimeError(
            f"samtools quickcheck failed for {fmt}: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
        ) from exc


def _read_alignment_header_text(
    path: Path,
    tools: ToolPaths,
    flags: dict[str, Any] | None,
    exec_cfg: ExecutionConfig,
) -> str:
    flag_data: dict[str, Any] = dict(flags or {})
    flag_data.setdefault("threads", exec_cfg.threads_for("samtools"))
    flag_args = build_cli_args("samtools", flag_data)
    cmd = [tools.samtools, "view", "-H", *flag_args, str(path)]
    try:
        result = run_command(cmd, timeout=exec_cfg.timeout_for("samtools"))
    except CommandError as exc:
        raise RuntimeError(
            f"samtools view -H failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
        ) from exc
    return result.stdout


DEFAULT_VCF_HEADER_MAX_LINES = 2000


def _read_vcf_header_text(path: Path, max_lines: int | None = DEFAULT_VCF_HEADER_MAX_LINES) -> str:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    header_lines: list[str] = []
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
    file_type: str | None = None,
    flags: dict[str, Any] | None = None,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
    max_lines: int | None = DEFAULT_VCF_HEADER_MAX_LINES,
) -> HeaderMetadata:
    """
    Extract header metadata for BAM/CRAM/VCF inputs and return structured data with a summary.
    """
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    file_path = Path(path)
    _validate_input_file(file_path, cfg)

    fmt = _infer_header_format(file_path, file_type)
    if fmt in {"bam", "cram", "sam"}:
        _maybe_quickcheck(file_path, tools, fmt, cfg)
        header_text = _read_alignment_header_text(file_path, tools, flags, cfg)
        metadata = parse_alignment_header(header_text, file_path=str(file_path), fmt=fmt)
    elif fmt == "vcf":
        header_text = _read_vcf_header_text(file_path, max_lines=max_lines)
        metadata = parse_vcf_header(header_text, file_path=str(file_path))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    summarize_header(metadata)
    return metadata


def serialize_model(model: BaseModel) -> dict[str, object]:
    """Return a JSON-serializable dict from a pydantic model."""
    return model.model_dump()


__all__ = [
    "alignment_error_profile",
    "alignment_summary",
    "coverage_stats",
    "env_check",
    "filter_reads",
    "header_metadata_lookup",
    "qc_alignment",
    "qc_reads",
    "qscore_distribution",
    "qscore_distribution_bam",
    "read_length_distribution",
    "read_length_distribution_bam",
    "serialize_model",
]

