import gzip
import os
import logging
import shutil
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import BaseModel

from .cli_wrappers import (
    build_cli_args,
    chopper_filter,
    cramino_stats,
    mosdepth_coverage,
    nanoq_from_bam_streaming,
    nanoq_stats,
    detect_container_runtime,
    run_igv_snapshot,
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
    IgvRegion,
    IgvSnapshotResult,
)
from .utils import CommandError, _truncate_stderr, format_cmd, report_progress, run_command
from .igv_batch import generate_igv_batch


logger = logging.getLogger(__name__)


def env_check(tools: ToolPaths | None = None) -> EnvStatus:
    tools = tools or ToolPaths()
    missing = tools.missing()
    resolved = tools.resolved()
    available = {k: k not in missing for k in resolved.keys()}
    runtime = detect_container_runtime(tools)
    available["igv_snapshot"] = runtime is not None
    return EnvStatus(available=available, resolved_paths=resolved, missing=missing, igv_runtime=runtime)


_EXEC_CFG = ExecutionConfig()
_NANOQ_CACHE: dict[tuple, NanoqStats] = {}
_NANOQ_CACHE_MAX = 8
_NANOQ_CACHE_LOCK = threading.Lock()
_NANOQ_INFLIGHT: dict[tuple, Future[NanoqStats]] = {}
_NANOQ_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}


def _mock_snapshot_files(batch_path: Path, output_root: Path, snapshot_format: str) -> list[Path]:
    """
    Generate placeholder snapshot files based on the batch file contents.

    Used when MCP_IGV_MOCK=1 is set (test environments without a container runtime).
    """
    names: list[str] = []
    try:
        for line in batch_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("snapshot "):
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2 and parts[1]:
                    names.append(parts[1])
    except FileNotFoundError:
        pass

    if not names:
        names = [f"mock_snapshot.{snapshot_format}"]

    snapshots: list[Path] = []
    for name in names:
        snap_path = output_root / name
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text("mock igv snapshot", encoding="utf-8")
        snapshots.append(snap_path)
    return snapshots


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


def _cached_nanoq_stats(path: Path, tools: ToolPaths, flags: dict[str, Any] | None, cfg: ExecutionConfig) -> NanoqStats:
    key = _nanoq_cache_key(path, flags, cfg)
    with _NANOQ_CACHE_LOCK:
        cached = _NANOQ_CACHE.get(key)
        if cached:
            _NANOQ_CACHE_STATS["hits"] += 1
            logger.debug("nanoq cache hit for %s (hits=%d)", path, _NANOQ_CACHE_STATS["hits"])
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
        with _NANOQ_CACHE_LOCK:
            _NANOQ_CACHE_STATS["misses"] += 1
        stats = nanoq_stats(path, tools, flags=flags, exec_cfg=cfg)
        if not stats.file or stats.file == "unknown":
            stats.file = str(path)

        # Simple bounded cache to avoid unbounded growth.
        with _NANOQ_CACHE_LOCK:
            if len(_NANOQ_CACHE) >= _NANOQ_CACHE_MAX:
                evicted_key = next(iter(_NANOQ_CACHE))
                _NANOQ_CACHE.pop(evicted_key)
                _NANOQ_CACHE_STATS["evictions"] += 1
                logger.debug("nanoq cache eviction (evictions=%d)", _NANOQ_CACHE_STATS["evictions"])
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


def get_nanoq_cache_stats() -> dict[str, int]:
    """Return a snapshot of cache hit/miss/eviction counters and sizes."""
    with _NANOQ_CACHE_LOCK:
        return {
            "hits": _NANOQ_CACHE_STATS.get("hits", 0),
            "misses": _NANOQ_CACHE_STATS.get("misses", 0),
            "evictions": _NANOQ_CACHE_STATS.get("evictions", 0),
            "size": len(_NANOQ_CACHE),
            "inflight": len(_NANOQ_INFLIGHT),
            "max_size": _NANOQ_CACHE_MAX,
        }


def qc_reads(
    path: str,
    tools: ToolPaths | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> NanoqStats:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    fastq_path = Path(path)
    _validate_input_file(
        fastq_path,
        cfg,
        allowed_exts=(".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.bgz", ".fq.bgz"),
    )
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
    _validate_input_file(
        fastq_path,
        cfg,
        allowed_exts=(".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.bgz", ".fq.bgz"),
    )
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
    exec_cfg: ExecutionConfig | None = None,
) -> QCReport:
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG
    report_progress(f"alignment_summary start: {path}")
    aln_stats = qc_alignment(path, tools=tools, include_hist=include_hist, use_scaled=use_scaled, flags=cramino_flags)
    coverage = (
        coverage_stats(
            path,
            tools=tools,
            window=coverage_window,
            low_cov_threshold=coverage_low_cov_threshold,
            flags=coverage_flags,
        )
        if include_coverage
        else None
    )
    errors = (
        alignment_error_profile(path, tools=tools, flags=error_profile_flags, exec_cfg=cfg)
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
                for _ in range(4):
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
                f"File type mismatch for {file_path}: you specified file_type='{declared}' "
                f"but magic bytes/header indicate '{sniffed}'. Verify the file or pass the correct file_type."
            )
        return declared

    if sniffed:
        if ext_guess and sniffed != ext_guess:
            raise ValueError(
                f"File extension suggests '{ext_guess}' but contents look like '{sniffed}' for {file_path}'. "
                "Pass file_type to override if intentional, or check the input for corruption."
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


def _parse_bed_regions(bed_path: Path, snapshot_format: str, min_snapshot_width: int) -> list[IgvRegion]:
    regions: list[IgvRegion] = []
    with open(bed_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Invalid BED entry (expected at least 3 columns): {line}")
            chrom, start_str, end_str = parts[:3]
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"Invalid start/end in BED entry: {line}") from exc

            name = parts[3] if len(parts) > 3 else None
            extra_cmds: list[str] = []
            if len(parts) > 4:
                for entry in parts[4:]:
                    entry = entry.lstrip("#")
                    extra_cmds.extend(cmd.strip() for cmd in entry.split(";") if cmd.strip())

            regions.append(
                IgvRegion(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name,
                    extra_commands=extra_cmds,
                )
            )
    if not regions:
        raise ValueError(f"No regions found in BED file: {bed_path}")
    return regions


def _coerce_regions(
    regions: list[dict[str, Any]] | list[IgvRegion] | str,
    snapshot_format: str,
    min_snapshot_width: int,
) -> list[IgvRegion]:
    if isinstance(regions, str):
        bed_path = Path(regions)
        return _parse_bed_regions(bed_path, snapshot_format=snapshot_format, min_snapshot_width=min_snapshot_width)

    coerced: list[IgvRegion] = []
    for region in regions:
        if isinstance(region, IgvRegion):
            coerced.append(region)
        elif isinstance(region, dict):
            coerced.append(IgvRegion(**region))
        else:
            raise TypeError(f"Unsupported region type: {type(region)}")
    if not coerced:
        raise ValueError("At least one region is required")
    return coerced


def generate_igv_snapshots(
    genome: str | None = None,
    tracks: list[str] | None = None,
    regions: list[dict[str, Any]] | list[IgvRegion] | str | None = None,
    output_dir: str | None = None,
    batch_file: str | None = None,
    compact: str = "squish",
    color_by: str | None = None,
    group_by: str | None = None,
    snapshot_format: Literal["png", "svg"] = "png",
    min_snapshot_width: int = 0,
    extra_commands: list[str] | None = None,
    extra_preferences: dict[str, str] | None = None,
    small_indels_show: bool = False,
    small_indels_threshold: int = 100,
    allele_threshold: float = 0.2,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> IgvSnapshotResult:
    """
    Generate IGV snapshots via containerized IGV. Supports pre-made batch files or dynamic generation from regions.
    """
    tools = tools or ToolPaths()
    cfg = exec_cfg or _EXEC_CFG

    if batch_file:
        batch_path = Path(batch_file).resolve()
        _validate_input_file(batch_path, cfg)
        output_root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="igv_snapshots_"))
        output_root.mkdir(parents=True, exist_ok=True)
        region_objs: list[IgvRegion] = []
        track_paths: list[Path] = []
        bed_path: Path | None = None
        genome_path: Path | None = None
    else:
        if not genome or not tracks or not regions:
            raise ValueError("genome, tracks, and regions are required when batch_file is not provided")

        output_root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="igv_snapshots_"))
        output_root.mkdir(parents=True, exist_ok=True)

        batch_dir = Path(tempfile.mkdtemp(prefix="igv_batch_"))
        batch_path = batch_dir / "igv.batch"

        region_objs = _coerce_regions(regions, snapshot_format=snapshot_format, min_snapshot_width=min_snapshot_width)
        bed_path = Path(regions) if isinstance(regions, str) else None

        track_paths = []
        for track in tracks:
            track_path = Path(track)
            _validate_input_file(track_path, cfg)
            track_paths.append(track_path.resolve())

        genome_path = Path(genome)
        genome_arg = genome
        if genome_path.exists():
            _validate_input_file(genome_path, cfg)
            genome_arg = str(genome_path.resolve())
            genome_path = genome_path.resolve()

        generate_igv_batch(
            genome=genome_arg,
            tracks=[str(t) for t in track_paths],
            regions=region_objs,
            output_path=batch_path,
            compact=compact,
            color_by=color_by,
            group_by=group_by,
            snapshot_dir=output_root,
            snapshot_format=snapshot_format,
            min_snapshot_width=min_snapshot_width,
            small_indels_show=small_indels_show,
            small_indels_threshold=small_indels_threshold,
            allele_threshold=allele_threshold,
            extra_commands=extra_commands,
            extra_preferences=extra_preferences,
        )

    mount_paths = set()
    if not batch_file:
        for track_path in track_paths:
            mount_paths.add(track_path.parent)
        if genome_path and genome_path.exists():
            mount_paths.add(genome_path.parent)
        if bed_path and bed_path.exists():
            mount_paths.add(bed_path.parent)
    mount_paths.add(batch_path.parent)

    if os.getenv("MCP_IGV_MOCK") == "1":
        mock_runtime = os.getenv("MCP_IGV_MOCK_RUNTIME", "docker")
        snapshots = _mock_snapshot_files(batch_path, output_root, snapshot_format)
        return IgvSnapshotResult(
            snapshot_files=[str(p) for p in snapshots],
            batch_file=str(batch_path),
            output_directory=str(output_root),
            execution_mode=mock_runtime,  # type: ignore[arg-type]
            command=["mock_igv_snapshot"],
        )

    snapshots, runtime, cmd = run_igv_snapshot(
        batch_file=batch_path,
        output_dir=output_root,
        tools=tools,
        exec_cfg=cfg,
        snapshot_format=snapshot_format,
        mount_paths=list(mount_paths),
    )

    return IgvSnapshotResult(
        snapshot_files=[str(p) for p in snapshots],
        batch_file=str(batch_path),
        output_directory=str(output_root),
        execution_mode=runtime,
        command=cmd,
    )


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
    "get_nanoq_cache_stats",
    "generate_igv_snapshots",
]
