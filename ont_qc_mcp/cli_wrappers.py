import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ExecutionConfig, ToolPaths
from .flag_schemas import FlagDef, get_tool_flags
from .parsers import parse_cramino_json, parse_mosdepth_summary, parse_nanoq_json
from .schemas import (
    ChopperReport,
    CraminoStats,
    MosdepthStats,
    NanoqStats,
)
from .utils import CommandError, _truncate_stderr, format_cmd, run_command


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
    result = run_command(cmd, timeout=timeout)
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
    flag_args = build_cli_args("chopper", merged_flags)
    temp_output = None
    if output_fastq is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".fastq", delete=False)
        temp_output = Path(temp_file.name)
        temp_file.close()
        output_fastq = temp_output

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
        run_command(cmd, timeout=timeout)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as fh:
                report_data = json.load(fh)
    except CommandError:
        # Fallback for older chopper versions: no subcommand, no JSON; capture stdout to output_fastq.
        fallback_cmd: List[str] = [tools.chopper, "--input", str(input_fastq), *flag_args]
        try:
            command_executed = fallback_cmd
            result = run_command(fallback_cmd, timeout=timeout)
            output_fastq.write_text(result.stdout)
        except CommandError as exc:
            raise RuntimeError(
                f"chopper failed: {format_cmd(exc.result.cmd)}\n{_truncate_stderr(exc.result.stderr)}"
            ) from exc
    finally:
        if json_path:
            json_path.unlink(missing_ok=True)
        if temp_output:
            temp_output.unlink(missing_ok=True)

    reads_section = report_data.get("reads", {}) if isinstance(report_data, dict) else {}
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
    hist_path: Optional[Path] = None
    hist_arg: List[str] = []

    if include_hist:
        # Direct histogram output to a temp file so stdout remains pure JSON.
        with tempfile.NamedTemporaryFile(suffix=".cramino.hist", delete=False) as tmp:
            hist_path = Path(tmp.name)
        hist_arg = ["--hist", str(hist_path)]
    if use_scaled:
        flag_data.setdefault("scaled", True)
    # Force JSON output regardless of user-supplied flags; parser expects JSON.
    flag_data.setdefault("format", "json")

    flag_data, timeout = _prepare_execution("cramino", flag_data, exec_cfg)
    # Handle --hist separately so we can pass a path even though the flag is defined as boolean.
    flag_data_no_hist = {k: v for k, v in flag_data.items() if k != "hist"}
    flag_args = build_cli_args("cramino", flag_data_no_hist)
    cmd: List[str] = [tools.cramino, *flag_args, *hist_arg, str(path)]
    try:
        result = run_command(cmd, timeout=timeout)
    finally:
        if hist_path:
            hist_path.unlink(missing_ok=True)
    return parse_cramino_json(result.stdout)


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

    return parse_mosdepth_summary(summary_text, file_path=str(path), threshold=low_cov_threshold)

