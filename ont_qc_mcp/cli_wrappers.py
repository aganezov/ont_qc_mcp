import json
import logging
import tempfile
import time
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Any, Deque, Dict, List, Optional, Sequence

from .config import ExecutionConfig, ToolPaths
from .flag_schemas import FlagDef, get_tool_flags
from .parsers import parse_cramino_json, parse_mosdepth_summary, parse_nanoq_json
from .schemas import (
    ChopperReport,
    CraminoStats,
    HistogramBin,
    MosdepthStats,
    NanoqStats,
)
from .utils import (
    CommandError,
    CommandResult,
    _truncate_stderr,
    format_cmd,
    report_progress,
    run_command,
    run_command_with_retry,
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
    tool: str, flags: Optional[Dict[str, Any]], exec_cfg: Optional[ExecutionConfig]
) -> tuple[Dict[str, Any], int]:
    """Merge user flags with defaults and return timeout."""
    cfg = exec_cfg or ExecutionConfig()
    merged = dict(flags or {})
    threads = cfg.threads_for(tool)
    if threads is not None:
        merged.setdefault("threads", threads)
    return merged, cfg.timeout_for(tool)


logger = logging.getLogger(__name__)


def build_cli_args(tool: str, flags: Optional[Dict[str, Any]]) -> List[str]:
    """
    Validate and convert an MCP flags dict into CLI args.

    Unknown keys raise FlagValidationError. Values are type-checked conservatively.
    """
    if not flags:
        return []

    flag_defs = get_tool_flags(tool)
    lookup: Dict[str, FlagDef] = {}
    for fd in flag_defs:
        for key in fd.all_keys():
            lookup[key] = fd

    args: List[str] = []
    for key, value in flags.items():
        if key not in lookup:
            raise FlagValidationError(f"Unknown flag for {tool}: {key}")
        flag = lookup[key]
        if value is None:
            continue

        match flag.type:
            case "bool":
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

    return args


def nanoq_stats(
    path: Path,
    tools: ToolPaths,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> NanoqStats:
    """
    Run nanoq --stats --json for fast read-level metrics.
    """
    merged_flags, timeout = _prepare_execution("nanoq", flags, exec_cfg)
    flag_args = build_cli_args("nanoq", merged_flags)
    cmd: Sequence[str] = [tools.nanoq, "--stats", "--json", "--input", str(path), *flag_args]
    report_progress(f"nanoq stats start: {path}")
    logger.debug("Executing nanoq stats: %s", format_cmd(cmd))
    result = run_command_with_retry(cmd, timeout=timeout, max_attempts=2, backoff_seconds=0.5)
    report_progress(f"nanoq stats done: {path}")
    return parse_nanoq_json(result.stdout)


def chopper_filter(
    input_fastq: Path,
    tools: ToolPaths,
    output_fastq: Optional[Path] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> ChopperReport:
    """
    Run chopper for ONT-oriented filtering/trimming.
    Attempts to capture JSON stats when supported by the installed version.
    """
    merged_flags, timeout = _prepare_execution("chopper", flags, exec_cfg)
    report_progress(f"chopper start: {input_fastq}")
    flag_args = build_cli_args("chopper", merged_flags)
    created_temp_output = False
    if output_fastq is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".fastq", delete=False)
        output_fastq = Path(temp_file.name)
        temp_file.close()
        created_temp_output = True

    report_data: dict = {}
    json_path: Optional[Path] = None

    # Preferred path: newer chopper with filter/report-json support.
    command_executed: List[str] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_fp:
            json_path = Path(json_fp.name)
        cmd: List[str] = [
            tools.chopper,
            "filter",
            "--input",
            str(input_fastq),
            "--output",
            str(output_fastq),
            "--report-json",
            str(json_path),
            *flag_args,
        ]
        command_executed = cmd
        logger.debug("Executing chopper filter with JSON: %s", format_cmd(cmd))
        run_command_with_retry(cmd, timeout=timeout, max_attempts=2, backoff_seconds=0.5)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as fh:
                report_data = json.load(fh)
    except CommandError as exc:
        # Fallback for older chopper versions: no subcommand, no JSON; stream stdout directly to output_fastq.
        stderr_text = exc.result.stderr or ""
        capability_error = "report-json" in stderr_text or "filter" in stderr_text.lower() or "unknown command" in stderr_text.lower()
        if not capability_error:
            if created_temp_output and output_fastq.exists():
                output_fastq.unlink(missing_ok=True)
            raise RuntimeError(
                f"chopper failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(stderr_text)}"
            ) from exc

        fallback_cmd: List[str] = [tools.chopper, "--input", str(input_fastq), *flag_args]
        try:
            command_executed = fallback_cmd
            logger.warning("Falling back to legacy chopper invocation: %s", format_cmd(fallback_cmd))
            run_command_with_retry(
                fallback_cmd, timeout=timeout, stdout_path=output_fastq, max_attempts=2, backoff_seconds=0.5
            )
        except CommandError as inner_exc:
            if created_temp_output and output_fastq.exists():
                output_fastq.unlink(missing_ok=True)
            raise RuntimeError(
                f"chopper failed: {format_cmd(inner_exc.result.cmd)}\n{_truncate_stderr(inner_exc.result.stderr)}"
            ) from inner_exc
    finally:
        if json_path:
            json_path.unlink(missing_ok=True)

    reads_section = report_data.get("reads", {}) if isinstance(report_data, dict) else {}
    report_progress(f"chopper done: {input_fastq}")
    return ChopperReport(
        input_reads=reads_section.get("input"),
        output_reads=reads_section.get("output"),
        filtered_reads=reads_section.get("filtered"),
        command=list(command_executed),
        params={"flags": flags or {}},
        output_fastq=str(output_fastq),
    )


def cramino_stats(
    path: Path,
    tools: ToolPaths,
    include_hist: bool = True,
    use_scaled: bool = False,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> CraminoStats:
    """
    Run cramino with JSON output for alignment-level stats.
    """
    flag_data: Dict[str, Any] = dict(flags or {})
    hist_count_path: Optional[Path] = None
    hist_args: List[str] = []
    length_bins: List[HistogramBin] = []
    report_progress(f"cramino start: {path}")

    if include_hist:
        # Collect bin counts in a TSV so stdout stays pure JSON.
        with tempfile.NamedTemporaryFile(suffix=".cramino.hist.tsv", delete=False) as tmp:
            hist_count_path = Path(tmp.name)
        hist_args = ["--hist-count", str(hist_count_path)]
    if use_scaled:
        flag_data.setdefault("scaled", True)
    # Force JSON output regardless of user-supplied flags; parser expects JSON.
    flag_data.setdefault("format", "json")

    flag_data, timeout = _prepare_execution("cramino", flag_data, exec_cfg)
    flag_args = build_cli_args("cramino", flag_data)
    cmd: List[str] = [tools.cramino, *flag_args, *hist_args, str(path)]
    try:
        logger.debug("Executing cramino: %s", format_cmd(cmd))
        result = run_command(cmd, timeout=timeout)
        if hist_count_path and hist_count_path.exists():
            with open(hist_count_path, "r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
            # Expect header then rows: bin_start bin_end count
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    start, end, count = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                length_bins.append(HistogramBin(start=start, end=end, count=int(count)))
    finally:
        if hist_count_path:
            hist_count_path.unlink(missing_ok=True)
    report_progress(f"cramino done: {path}")
    return parse_cramino_json(result.stdout, length_bins=length_bins)


def nanoq_from_bam_streaming(
    path: Path,
    tools: ToolPaths,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
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
    nanoq_threads = nanoq_flags.get("threads")

    sam_cmd: List[str] = [tools.samtools, "fastq"]
    if sam_threads is not None:
        sam_cmd += ["-@", str(sam_threads)]
    sam_cmd += [str(path)]

    nano_cmd: List[str] = [tools.nanoq, "--stats", "--json"]
    if nanoq_threads is not None:
        nano_cmd += ["--threads", str(nanoq_threads)]
    nano_cmd += build_cli_args("nanoq", {k: v for k, v in nanoq_flags.items() if k != "threads"})

    # Run samtools fastq -> nanoq via async to avoid deadlocks on pipes.
    import subprocess

    logger.debug("Starting samtools|nanoq streaming pipeline: %s | %s", format_cmd(sam_cmd), format_cmd(nano_cmd))
    sam_proc = subprocess.Popen(sam_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nano_proc = subprocess.Popen(
        nano_cmd,
        stdin=sam_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if sam_proc.stdout:
        sam_proc.stdout.close()

    overall_timeout = max(sam_timeout, nano_timeout)
    start_time = time.monotonic()

    stderr_tail: Deque[str] = deque(maxlen=200)
    stderr_thread: Optional[Thread] = None

    def _drain_sam_stderr():
        if not sam_proc.stderr:
            return
        for line in sam_proc.stderr:
            try:
                decoded = line.decode("utf-8", errors="replace")
            except Exception:
                decoded = str(line)
            stderr_tail.append(decoded.rstrip("\n"))

    if sam_proc.stderr:
        stderr_thread = Thread(target=_drain_sam_stderr, daemon=True)
        stderr_thread.start()

    def _terminate_pipeline(reason: str) -> RuntimeError:
        for proc in (nano_proc, sam_proc):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in (nano_proc, sam_proc):
            try:
                proc.kill()
            except Exception:
                pass
        for proc in (nano_proc, sam_proc):
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        return RuntimeError(reason)

    try:
        nano_out, nano_err = nano_proc.communicate(timeout=overall_timeout)
    except subprocess.TimeoutExpired:
        raise _terminate_pipeline(
            f"Timeout while running samtools|nanoq pipeline (>{overall_timeout}s). "
            f"samtools cmd: {format_cmd(sam_cmd)}; nanoq cmd: {format_cmd(nano_cmd)}"
        )

    remaining = max(0.0, overall_timeout - (time.monotonic() - start_time))
    try:
        _, sam_err = sam_proc.communicate(timeout=remaining or 0.1)
    except subprocess.TimeoutExpired:
        raise _terminate_pipeline(
            f"Timeout waiting for samtools fastq to exit (>{overall_timeout}s). "
            f"cmd: {format_cmd(sam_cmd)}"
        )
    finally:
        if stderr_thread:
            stderr_thread.join(timeout=0.2)

    sam_rc = sam_proc.returncode
    sam_err_text = sam_err.decode("utf-8", errors="replace") if isinstance(sam_err, (bytes, bytearray)) else (sam_err or "")
    if stderr_tail:
        tail_text = "\n".join(stderr_tail)
        sam_err_text = tail_text if sam_err_text == "" else sam_err_text or tail_text
    nano_out_text = nano_out.decode("utf-8", errors="replace") if isinstance(nano_out, (bytes, bytearray)) else nano_out
    nano_err_text = nano_err.decode("utf-8", errors="replace") if isinstance(nano_err, (bytes, bytearray)) else nano_err

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
    logger.debug("Completed streaming nanoq for %s", path)
    report_progress(f"nanoq streaming done: {path}")
    return stats


def mosdepth_coverage(
    path: Path,
    tools: ToolPaths,
    window: Optional[int] = None,
    low_cov_threshold: Optional[float] = None,
    flags: Optional[Dict[str, Any]] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> MosdepthStats:
    """
    Run mosdepth to compute depth-of-coverage summaries.
    Returns parsed summary.txt metrics.
    """
    flag_data: Dict[str, Any] = dict(flags or {})
    report_progress(f"mosdepth start: {path}")
    if window is not None:
        flag_data.setdefault("window", window)
    flag_data, timeout = _prepare_execution("mosdepth", flag_data, exec_cfg)
    flag_args = build_cli_args("mosdepth", flag_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "mosdepth"
        summary_path = Path(f"{prefix}.mosdepth.summary.txt")

        cmd: List[str] = [
            tools.mosdepth,
        ]
        cmd += flag_args
        cmd += [str(prefix), str(path)]

        try:
            run_command(cmd, timeout=timeout)
        except CommandError as exc:
            raise RuntimeError(
                f"mosdepth failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
            ) from exc

        with open(summary_path, "r", encoding="utf-8") as fh:
            summary_text = fh.read()

    report_progress(f"mosdepth done: {path}")
    return parse_mosdepth_summary(summary_text, file_path=str(path), threshold=low_cov_threshold)


__all__ = [
    "FlagValidationError",
    "build_cli_args",
    "chopper_filter",
    "cramino_stats",
    "mosdepth_coverage",
    "nanoq_from_bam_streaming",
    "nanoq_stats",
]
