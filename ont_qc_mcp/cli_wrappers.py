import gzip
import logging
import os
import select
import subprocess  # nosec B404
import tempfile
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from shutil import copyfileobj, rmtree, which
from threading import Event, Lock, Thread
from typing import Any, Literal
from uuid import uuid4

from .config import ExecutionConfig, ToolPaths
from .flag_schemas import FlagDef, get_tool_flags
from .process_control import (
    cancellation_disabled,
    check_cancelled,
    cleanup_processes,
    communicate_process,
    start_process,
    wait_process,
)
from .parsers import parse_cramino_json, parse_mosdepth_summary, parse_nanoq_json
from .schemas import (
    ChopperReport,
    CraminoStats,
    MosdepthStats,
    NanoqStats,
)
from .nanoq_aux import length_histogram_and_percentiles, qscore_histogram
from .utils import (
    CommandError,
    CommandResult,
    _truncate_stderr,
    format_cmd,
    report_progress,
    run_command,
    run_command_with_retry,
    safe_path_arg,
)


class FlagValidationError(ValueError):
    """Raised when MCP flags fail validation."""


def _select_flag_name(flag: FlagDef) -> str:
    if flag.name:
        return flag.name
    if flag.short:
        return flag.short
    raise FlagValidationError(f"Flag {flag.param} has no CLI name defined")


def _prepare_execution(
    tool: str, flags: dict[str, Any] | None, exec_cfg: ExecutionConfig | None
) -> tuple[dict[str, Any], int]:
    """Merge user flags with defaults and return timeout."""
    cfg = exec_cfg or ExecutionConfig()
    merged = dict(flags or {})
    threads = cfg.threads_for(tool)
    if threads is not None:
        merged.setdefault("threads", threads)
    return merged, cfg.timeout_for(tool)


logger = logging.getLogger(__name__)


def build_cli_args(tool: str, flags: dict[str, Any] | None) -> list[str]:
    """
    Validate and convert an MCP flags dict into CLI args.

    Unknown keys raise FlagValidationError. Values are type-checked conservatively.
    """
    if not flags:
        return []

    flag_defs = get_tool_flags(tool)
    lookup: dict[str, FlagDef] = {}
    for fd in flag_defs:
        for key in fd.all_keys():
            lookup[key] = fd

    args: list[str] = []
    for key, value in flags.items():
        if key not in lookup:
            raise FlagValidationError(f"Unknown flag for {tool}: {key}")
        flag = lookup[key]
        if value is None:
            continue

        match flag.type:
            case "bool":
                if not isinstance(value, bool):
                    raise FlagValidationError(f"Flag {key} expects bool, got {type(value).__name__}")
                if value:
                    args.append(_select_flag_name(flag))
                continue
            case "int":
                if not isinstance(value, int):
                    raise FlagValidationError(f"Flag {key} expects int, got {type(value).__name__}")
            case "float":
                if not isinstance(value, (int, float)):
                    raise FlagValidationError(f"Flag {key} expects float, got {type(value).__name__}")
            case "path":
                value = str(Path(value))
            case "str":
                if not isinstance(value, str):
                    raise FlagValidationError(f"Flag {key} expects str, got {type(value).__name__}")
            case _:
                raise FlagValidationError(f"Unsupported flag type for {key}: {flag.type}")

        args.extend([_select_flag_name(flag), str(value)])

    if (
        tool == "chopper"
        and any(flags.get(crop) for crop in ("headcrop", "tailcrop"))
        and flags.get("trim_approach") != "fixed-crop"
    ):
        raise FlagValidationError("Chopper headcrop/tailcrop require trim_approach='fixed-crop'")

    return args


def nanoq_stats(
    path: Path,
    tools: ToolPaths,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> NanoqStats:
    """
    Run nanoq --stats --json for fast read-level metrics.
    """
    cfg = exec_cfg or ExecutionConfig()
    merged_flags, timeout = _prepare_execution("nanoq", flags, cfg)
    flag_args = build_cli_args("nanoq", merged_flags)
    read_lengths_path: Path | None = None
    read_qualities_path: Path | None = None

    cmd: list[str] = [tools.nanoq, "--stats", "--json", "--input", safe_path_arg(path)]
    if cfg.nanoq_aux_stats:
        with tempfile.NamedTemporaryFile(suffix=".nanoq.lengths.txt", delete=False) as tmp:
            read_lengths_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".nanoq.quals.txt", delete=False) as tmp:
            read_qualities_path = Path(tmp.name)
        cmd += ["--read-lengths", str(read_lengths_path), "--read-qualities", str(read_qualities_path)]
    cmd += [*flag_args]

    report_progress(f"nanoq stats start: {path}")
    logger.debug("Executing nanoq stats: %s", format_cmd(cmd))
    try:
        result = run_command_with_retry(cmd, timeout=timeout, max_attempts=2, backoff_seconds=0.5)
        report_progress(f"nanoq stats done: {path}")
        stats = parse_nanoq_json(result.stdout)

        if cfg.nanoq_aux_stats:
            if (stats.length_histogram is None or stats.length_percentiles is None) and read_lengths_path:
                if read_lengths_path.exists() and read_lengths_path.stat().st_size > 0:
                    hist, percentiles = length_histogram_and_percentiles(
                        read_lengths_path,
                        bin_width=cfg.nanoq_length_bin_width,
                        percentiles_exact_max_reads=cfg.nanoq_percentiles_exact_max_reads,
                    )
                    if stats.length_histogram is None:
                        stats.length_histogram = hist
                    if stats.length_percentiles is None and percentiles is not None:
                        stats.length_percentiles = percentiles

            if stats.qscore_histogram is None and read_qualities_path:
                if read_qualities_path.exists() and read_qualities_path.stat().st_size > 0:
                    stats.qscore_histogram = qscore_histogram(
                        read_qualities_path,
                        bin_width=cfg.nanoq_qscore_bin_width,
                    )

        return stats
    finally:
        if read_lengths_path:
            read_lengths_path.unlink(missing_ok=True)
        if read_qualities_path:
            read_qualities_path.unlink(missing_ok=True)


def chopper_filter(
    input_fastq: Path,
    tools: ToolPaths,
    output_fastq: Path | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> ChopperReport:
    """
    Run chopper for ONT-oriented filtering/trimming.
    Stream stdout into an atomic staging file; the CLI does not emit JSON stats.
    """
    supplied_name = output_fastq.name.lower() if output_fastq is not None else ""
    gzip_output = supplied_name.endswith(".gz")
    unsupported_suffix = next(
        (
            suffix
            for suffix in {
                ".bgz",
                ".bgzf",
                ".bz",
                ".bz2",
                ".bzip2",
                ".xz",
                ".lzma",
                ".zst",
                ".zstd",
                ".lz4",
                ".zip",
                ".z",
                ".gzip",
            }
            if supplied_name.endswith(suffix)
        ),
        None,
    )
    if unsupported_suffix:
        raise ValueError(f"Unsupported output compression suffix {unsupported_suffix!r}; use .gz for gzip output")
    merged_flags, timeout = _prepare_execution("chopper", flags, exec_cfg)
    report_progress(f"chopper start: {input_fastq}")
    flag_args = build_cli_args("chopper", merged_flags)
    destination = output_fastq.resolve() if output_fastq is not None else None
    if destination is not None:
        if destination == input_fastq.resolve() or (destination.exists() and destination.samefile(input_fastq)):
            raise ValueError("Input and output FASTQ paths refer to the same file")
        if destination.exists() and not destination.is_file():
            raise ValueError("Output FASTQ must be a regular file")

    # Stage beside the resolved destination so replacement is atomic and symlinks
    # retain their existing write-through behavior.
    staged_output: Path | None = None
    compressed_output: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent if destination is not None else None,
            suffix="".join(output_fastq.suffixes) if output_fastq is not None else ".fastq",
            delete=False,
        ) as output_fp:
            staged_output = Path(output_fp.name)
        if output_fastq is None:
            output_fastq = staged_output

        cmd = [tools.chopper, "--input", safe_path_arg(input_fastq), *flag_args]
        logger.debug("Executing chopper: %s", format_cmd(cmd))
        try:
            run_command_with_retry(cmd, timeout=timeout, stdout_path=staged_output, max_attempts=2, backoff_seconds=0.5)
        except CommandError as exc:
            raise RuntimeError(
                f"chopper failed (exit {exc.result.returncode}): {format_cmd(exc.result.cmd)}\n"
                f"{_truncate_stderr(exc.result.stderr)}"
            ) from exc
        report = ChopperReport(
            command=cmd,
            params={"flags": flags or {}},
            output_fastq=str(output_fastq),
        )
        publish_stage = staged_output
        if gzip_output and destination is not None:
            # Chopper writes plain FASTQ. Compress in Python after it succeeds;
            # a GzipFile passed as subprocess stdout would bypass compression.
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".gz", delete=False) as compressed_fp:
                compressed_output = Path(compressed_fp.name)
                with staged_output.open("rb") as raw_fp:
                    with gzip.GzipFile(
                        filename="", mode="wb", fileobj=compressed_fp, compresslevel=6, mtime=0
                    ) as gzip_fp:
                        copyfileobj(raw_fp, gzip_fp, length=1024 * 1024)
            # Close the gzip trailer and both files, then remove the raw stage
            # before publication so a cleanup failure leaves the destination intact.
            staged_output.unlink()
            publish_stage = compressed_output
        check_cancelled()
        if destination is not None:
            if destination.exists():
                publish_stage.chmod(destination.stat().st_mode & 0o777)
            os.replace(publish_stage, destination)
        published = True
        report_progress(f"chopper done: {input_fastq}")
        return report
    finally:
        for stage in (compressed_output, staged_output if gzip_output or not published else None):
            if stage is not None:
                try:
                    stage.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to remove Chopper staging file %s: %s", stage, exc)


def cramino_stats(
    path: Path,
    tools: ToolPaths,
    include_hist: bool = True,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> CraminoStats:
    """
    Run cramino with JSON output for alignment-level stats.
    """
    flag_data, timeout = _prepare_execution("cramino", flags, exec_cfg)
    flag_args = build_cli_args("cramino", flag_data)
    # A histogram switch populates both JSON histograms. Discard its separate TSV
    # output so stdout contains only JSON; no second parser or temporary file is needed.
    hist_args = ["--hist-count", os.devnull] if include_hist else []
    cmd = [tools.cramino, "--format", "json", *flag_args, *hist_args, safe_path_arg(path)]
    report_progress(f"cramino start: {path}")
    logger.debug("Executing cramino: %s", format_cmd(cmd))
    result = run_command(cmd, timeout=timeout)
    report_progress(f"cramino done: {path}")
    return parse_cramino_json(result.stdout)


def nanoq_from_bam_streaming(
    path: Path,
    tools: ToolPaths,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> NanoqStats:
    """
    Stream BAM/CRAM through samtools fastq into nanoq --stats --json.
    Avoids temp FASTQ; relies on piping via subprocess stdout/stdin.
    """
    report_progress(f"nanoq streaming start: {path}")
    cfg = exec_cfg or ExecutionConfig()
    samtools_flags, sam_timeout = _prepare_execution("samtools", {}, cfg)
    nanoq_flags, nano_timeout = _prepare_execution("nanoq", flags, cfg)
    sam_threads = samtools_flags.get("threads")
    nanoq_args = build_cli_args("nanoq", nanoq_flags)

    sam_cmd: list[str] = [tools.samtools, "fastq"]
    if sam_threads is not None:
        sam_cmd += ["-@", str(sam_threads)]
    sam_cmd += [safe_path_arg(path)]

    nano_cmd: list[str] = [tools.nanoq, "--stats", "--json"]
    read_lengths_path: Path | None = None
    read_qualities_path: Path | None = None
    sam_proc: subprocess.Popen | None = None
    nano_proc: subprocess.Popen | None = None
    stderr_thread: Thread | None = None
    stop_reader = Event()
    stderr_lock = Lock()
    try:
        if cfg.nanoq_aux_stats:
            with tempfile.NamedTemporaryFile(suffix=".nanoq.lengths.txt", delete=False) as tmp:
                read_lengths_path = Path(tmp.name)
            with tempfile.NamedTemporaryFile(suffix=".nanoq.quals.txt", delete=False) as tmp:
                read_qualities_path = Path(tmp.name)
            nano_cmd += ["--read-lengths", str(read_lengths_path), "--read-qualities", str(read_qualities_path)]
        nano_cmd += nanoq_args
        logger.debug("Starting samtools|nanoq streaming pipeline: %s | %s", format_cmd(sam_cmd), format_cmd(nano_cmd))
        sam_proc = start_process(sam_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        nano_proc = start_process(
            nano_cmd,
            stdin=sam_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if sam_proc.stdout:
            sam_proc.stdout.close()

        overall_timeout = max(sam_timeout, nano_timeout)
        start_time = time.monotonic()

        stderr_tail: deque[str] = deque(maxlen=200)
        stderr_head: list[str] = []
        sam_stderr = sam_proc.stderr

        def _record_stderr(line: bytes) -> None:
            line_text = line.decode("utf-8", errors="replace")
            with stderr_lock:
                if len(stderr_head) < 20:
                    stderr_head.append(line_text)
                stderr_tail.append(line_text)

        def _drain_sam_stderr():
            if not sam_stderr:
                return
            pending = b""
            try:
                fd = sam_stderr.fileno()
                os.set_blocking(fd, False)
                while not stop_reader.is_set():
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if not ready:
                        continue
                    try:
                        chunk = os.read(fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        break
                    lines = (pending + chunk).split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        _record_stderr(line)
                if pending:
                    _record_stderr(pending)
            except OSError as exc:
                logger.warning("Cannot read samtools stderr: %s", exc)
            finally:
                sam_stderr.close()

        if sam_proc.stderr:
            stderr_thread = Thread(target=_drain_sam_stderr, daemon=True)
            stderr_thread.start()

        try:
            nano_out, nano_err = communicate_process(nano_proc, timeout=overall_timeout)
        except subprocess.TimeoutExpired:
            sam_running = sam_proc.poll() is None
            nano_running = nano_proc.poll() is None
            hung_stage = (
                "samtools"
                if sam_running and not nano_running
                else "nanoq"
                if nano_running and not sam_running
                else "both"
            )
            with stderr_lock:
                stderr_context = "\n".join(list(stderr_tail)[-10:]) if stderr_tail else "(no stderr captured)"
            raise RuntimeError(
                f"Timeout while running samtools|nanoq pipeline (>{overall_timeout}s); "
                f"likely hung at {hung_stage}. "
                f"samtools cmd: {format_cmd(sam_cmd)}; nanoq cmd: {format_cmd(nano_cmd)}; "
                f"samtools stderr tail:\n{stderr_context}"
            )

        remaining = max(0.0, overall_timeout - (time.monotonic() - start_time))
        try:
            wait_process(sam_proc, timeout=remaining or 0.1)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Timeout waiting for samtools fastq to exit (>{overall_timeout}s). cmd: {format_cmd(sam_cmd)}"
            )
        if stderr_thread:
            stderr_thread.join(timeout=0.2)

        sam_rc = sam_proc.returncode
        with stderr_lock:
            sam_err_lines = list(stderr_tail)
            if len(sam_err_lines) == stderr_tail.maxlen:
                sam_err_lines = stderr_head + ["... (truncated) ..."] + sam_err_lines[-20:]
        sam_err_text = "\n".join(sam_err_lines)
        nano_out_text = (
            nano_out.decode("utf-8", errors="replace") if isinstance(nano_out, (bytes, bytearray)) else nano_out
        )
        nano_err_text = (
            nano_err.decode("utf-8", errors="replace") if isinstance(nano_err, (bytes, bytearray)) else nano_err
        )

        if sam_rc not in (0, 141):  # samtools may exit with SIGPIPE (141) if downstream closes early
            raise RuntimeError(f"samtools fastq failed: {format_cmd(sam_cmd)}\n{_truncate_stderr(sam_err_text)}")
        if nano_proc.returncode != 0:
            raise CommandError(
                CommandResult(
                    cmd=nano_cmd,
                    returncode=nano_proc.returncode or 1,
                    stdout=nano_out_text,
                    stderr=nano_err_text,
                )
            )

        stats = parse_nanoq_json(nano_out_text)
        if not stats.file or stats.file == "unknown":
            stats.file = str(path)
        if cfg.nanoq_aux_stats:
            if (stats.length_histogram is None or stats.length_percentiles is None) and read_lengths_path:
                if read_lengths_path.exists() and read_lengths_path.stat().st_size > 0:
                    hist, percentiles = length_histogram_and_percentiles(
                        read_lengths_path,
                        bin_width=cfg.nanoq_length_bin_width,
                        percentiles_exact_max_reads=cfg.nanoq_percentiles_exact_max_reads,
                    )
                    if stats.length_histogram is None:
                        stats.length_histogram = hist
                    if stats.length_percentiles is None and percentiles is not None:
                        stats.length_percentiles = percentiles
            if stats.qscore_histogram is None and read_qualities_path:
                if read_qualities_path.exists() and read_qualities_path.stat().st_size > 0:
                    stats.qscore_histogram = qscore_histogram(
                        read_qualities_path,
                        bin_width=cfg.nanoq_qscore_bin_width,
                    )
        logger.debug("Completed streaming nanoq for %s", path)
        report_progress(f"nanoq streaming done: {path}")
        return stats
    finally:
        # Stop the reader before closing pipes; inherited stderr need not reach EOF.
        stop_reader.set()
        if stderr_thread and stderr_thread.ident is not None:
            stderr_thread.join()
        try:
            cleanup_processes([nano_proc, sam_proc])
        finally:
            for aux_path in (read_lengths_path, read_qualities_path):
                if aux_path is not None:
                    try:
                        aux_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.debug("Ignore auxiliary file cleanup error for %s: %s", aux_path, exc)


def mosdepth_coverage(
    path: Path,
    tools: ToolPaths,
    window: int | None = None,
    low_cov_threshold: float | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> MosdepthStats:
    """
    Run mosdepth to compute depth-of-coverage summaries.
    Returns parsed summary.txt metrics.
    """
    flag_data: dict[str, Any] = dict(flags or {})
    report_progress(f"mosdepth start: {path}")
    if window is not None:
        flag_data.setdefault("window", window)
    flag_data, timeout = _prepare_execution("mosdepth", flag_data, exec_cfg)
    flag_args = build_cli_args("mosdepth", flag_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "mosdepth"
        summary_path = Path(f"{prefix}.mosdepth.summary.txt")

        cmd: list[str] = [
            tools.mosdepth,
        ]
        cmd += flag_args
        cmd += [str(prefix), safe_path_arg(path)]

        try:
            run_command(cmd, timeout=timeout)
        except CommandError as exc:
            raise RuntimeError(
                f"mosdepth failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
            ) from exc

        with open(summary_path, "r", encoding="utf-8") as fh:
            summary_text = fh.read()

    report_progress(f"mosdepth done: {path}")
    return parse_mosdepth_summary(
        summary_text,
        file_path=str(path),
        threshold=low_cov_threshold,
        expected_region_mode="--by" in flag_args,
    )


def detect_container_runtime(tools: ToolPaths) -> Literal["docker", "apptainer", None]:
    """
    Detect available container runtime in priority order.
    Returns None if no container runtime is available.
    """
    docker_path = which(tools.docker)
    if docker_path:
        try:
            run_command([docker_path, "info"], timeout=5)
            return "docker"
        except CommandError:
            pass

    for cmd in (tools.apptainer, tools.singularity):
        if which(cmd):
            return "apptainer"

    return None


def run_igv_snapshot(
    batch_file: Path,
    output_dir: Path,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig | None = None,
    snapshot_format: str = "png",
    force_runtime: Literal["docker", "apptainer", "local"] | None = None,
    mount_paths: list[Path] | None = None,
) -> tuple[list[Path], Literal["docker", "apptainer", "local"], list[str]]:
    """
    Execute IGV via container runtime or local xvfb-run.

    Fallback chain: Docker → Apptainer → Local IGV

    Returns: (snapshot_paths, execution_mode, command_used)
    """
    cfg = exec_cfg or ExecutionConfig()
    batch_file = Path(batch_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = force_runtime or detect_container_runtime(tools)
    if force_runtime and runtime != force_runtime:
        raise RuntimeError(f"Requested runtime '{force_runtime}' not available (detected: {runtime})")
    if runtime is None:
        raise RuntimeError("No container runtime available. Install Docker or Apptainer, or set MCP_IGV_SIF_PATH.")

    read_mounts = {batch_file.parent, *(Path(p).resolve() for p in (mount_paths or []))}
    write_mounts = {output_dir}
    # Avoid duplicate mount points (e.g., output_dir also listed as a read-only mount).
    read_mounts -= write_mounts
    timeout = cfg.timeout_for("igv")
    image = cfg.igv_container_image

    container_name = f"ont-qc-igv-{uuid4().hex}"
    if runtime == "docker":
        cmd: list[str] = [
            tools.docker,
            "run",
            "--rm",
            "--name",
            container_name,
        ]
        for mount in sorted(read_mounts):
            cmd += ["-v", f"{mount}:{mount}:ro"]
        for mount in sorted(write_mounts):
            cmd += ["-v", f"{mount}:{mount}"]
        cmd += [
            image,
            "/IGV_Linux_2.16.2/igv.sh",
            "-b",
            str(batch_file),
        ]
    elif runtime == "apptainer":
        image_ref = cfg.igv_sif_path or f"docker://{image}"
        cmd = [tools.apptainer, "exec"]
        for mount in sorted(read_mounts):
            cmd += ["--bind", f"{mount}:{mount}:ro"]
        for mount in sorted(write_mounts):
            cmd += ["--bind", f"{mount}:{mount}"]
        cmd += [
            image_ref,
            # Use xvfb-run inside the container to ensure Xvfb is cleaned up on exit.
            "/usr/bin/xvfb-run",
            "-a",
            "/IGV_Linux_2.16.2/igv.sh",
            "-b",
            str(batch_file),
        ]
    else:  # local
        cmd = [
            tools.xvfb_run,
            "--auto-servernum",
            "--server-num=1",
            tools.igv,
            "-b",
            str(batch_file),
        ]

    try:
        try:
            run_command(cmd, timeout=timeout)
        except CommandError as exc:
            raise RuntimeError(
                f"igv snapshot failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
            ) from exc
    except BaseException:
        if runtime == "docker":
            # The Docker daemon owns the container separately from the CLI process.
            with cancellation_disabled():
                try:
                    run_command([tools.docker, "rm", "--force", container_name], timeout=5)
                except Exception as exc:
                    logger.warning("Docker cleanup could not confirm removal of %s: %s", container_name, exc)
        raise

    snapshots = sorted(output_dir.glob(f"*.{snapshot_format}"))
    return snapshots, runtime, cmd


def run_bcftools_stats(
    path: Path,
    tools: ToolPaths,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> str:
    """
    Run bcftools stats and return stdout.

    Args:
        path: Path to VCF/BCF file
        tools: ToolPaths instance
        flags: Optional flags dict (threads, samples, regions)
        exec_cfg: Optional ExecutionConfig for timeout/threads

    Returns:
        stdout text from bcftools stats
    """
    merged_flags, timeout = _prepare_execution("bcftools", flags, exec_cfg)
    flag_args = build_cli_args("bcftools", merged_flags)
    cmd: list[str] = [tools.bcftools, "stats", *flag_args, safe_path_arg(path)]
    report_progress(f"bcftools stats start: {path}")
    logger.debug("Executing bcftools stats: %s", format_cmd(cmd))
    try:
        result = run_command(cmd, timeout=timeout)
    except CommandError as exc:
        raise RuntimeError(
            f"bcftools stats failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
        ) from exc
    report_progress(f"bcftools stats done: {path}")
    return result.stdout


def run_samtools_bedcov(
    bam_path: Path,
    bed_path: Path,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig | None = None,
) -> str:
    """
    Run samtools bedcov and return stdout.

    Args:
        bam_path: Path to BAM/CRAM file
        bed_path: Path to BED file
        tools: ToolPaths instance
        exec_cfg: Optional ExecutionConfig for timeout/threads

    Returns:
        stdout text from samtools bedcov
    """
    cfg = exec_cfg or ExecutionConfig()
    samtools_flags, timeout = _prepare_execution("samtools", {}, cfg)
    sam_threads = samtools_flags.get("threads")

    cmd: list[str] = [tools.samtools, "bedcov"]
    if sam_threads is not None:
        cmd += ["-@", str(sam_threads)]
    cmd += [safe_path_arg(bed_path), safe_path_arg(bam_path)]

    report_progress(f"samtools bedcov start: {bam_path} x {bed_path}")
    logger.debug("Executing samtools bedcov: %s", format_cmd(cmd))
    try:
        result = run_command(cmd, timeout=timeout)
    except CommandError as exc:
        raise RuntimeError(
            f"samtools bedcov failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
        ) from exc
    report_progress(f"samtools bedcov done: {bam_path} x {bed_path}")
    return result.stdout


def run_mosdepth_targeted(
    bam_path: Path,
    bed_path: Path,
    tools: ToolPaths,
    thresholds: list[int] | None = None,
    flags: dict[str, Any] | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> tuple[Path, Path | None, Path]:
    """
    Run mosdepth with --by for targeted coverage with optional thresholds.

    Args:
        bam_path: Path to BAM/CRAM file
        bed_path: Path to BED file with target regions
        tools: ToolPaths instance
        thresholds: Coverage thresholds for percentage calculation (e.g., [1, 10, 20])
        flags: Optional mosdepth flags
        exec_cfg: Optional ExecutionConfig for timeout/threads

    Returns:
        Tuple of (regions_bed_path, thresholds_bed_path or None, output_dir)
        Note: Caller is responsible for cleaning up output_dir
    """
    flag_data: dict[str, Any] = dict(flags or {})
    flag_data, timeout = _prepare_execution("mosdepth", flag_data, exec_cfg)
    flag_args = build_cli_args("mosdepth", flag_data)

    # Create temp directory for mosdepth output
    output_dir = Path(tempfile.mkdtemp(prefix="mosdepth_targeted_"))
    with ExitStack() as cleanup:
        cleanup.callback(rmtree, output_dir, ignore_errors=True)
        prefix = output_dir / "coverage"

        cmd: list[str] = [tools.mosdepth]
        cmd += flag_args
        cmd += ["--by", safe_path_arg(bed_path)]

        if thresholds:
            threshold_str = ",".join(str(t) for t in thresholds)
            cmd += ["--thresholds", threshold_str]

        cmd += [str(prefix), safe_path_arg(bam_path)]

        report_progress(f"mosdepth targeted start: {bam_path} x {bed_path}")
        logger.debug("Executing mosdepth targeted: %s", format_cmd(cmd))
        try:
            run_command(cmd, timeout=timeout)
        except CommandError as exc:
            raise RuntimeError(
                f"mosdepth targeted failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
            ) from exc

        regions_bed = prefix.with_suffix(".regions.bed.gz")
        thresholds_bed = prefix.with_suffix(".thresholds.bed.gz")

        if not regions_bed.exists():
            raise RuntimeError(f"mosdepth did not produce expected output: {regions_bed}")

        report_progress(f"mosdepth targeted done: {bam_path} x {bed_path}")
        result = regions_bed, thresholds_bed if thresholds_bed.exists() else None, output_dir
        check_cancelled()
        cleanup.pop_all()
        return result


__all__ = [
    "FlagValidationError",
    "build_cli_args",
    "chopper_filter",
    "cramino_stats",
    "mosdepth_coverage",
    "nanoq_from_bam_streaming",
    "nanoq_stats",
    "detect_container_runtime",
    "run_bcftools_stats",
    "run_igv_snapshot",
    "run_mosdepth_targeted",
    "run_samtools_bedcov",
]
