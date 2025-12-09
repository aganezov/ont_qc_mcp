import json
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .config import ToolPaths
from .parsers import parse_samtools_stats, parse_seqkit_stats
from .schemas import FastpReport, NanoPlotReport, SamtoolsStats, SeqkitStats
from .utils import CommandError, CommandResult, format_cmd, run_command


def seqkit_stats(path: Path, tools: ToolPaths) -> SeqkitStats:
    cmd: Sequence[str] = [tools.seqkit, "stats", "--tabular", str(path)]
    result = run_command(cmd)
    return parse_seqkit_stats(result.stdout)


def samtools_stats(path: Path, tools: ToolPaths) -> SamtoolsStats:
    cmd: Sequence[str] = [tools.samtools, "stats", str(path)]
    result = run_command(cmd)
    return parse_samtools_stats(result.stdout)


def fastp_filter(
    input_fastq: Path,
    tools: ToolPaths,
    output_fastq: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> FastpReport:
    """
    Run fastp for optional filtering/trimming and capture its JSON report.
    When output_fastq is None, results are written to a temporary file that is removed afterwards.
    """
    args = extra_args or []
    temp_output = None
    if output_fastq is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".fastq", delete=False)
        temp_output = Path(temp_file.name)
        temp_file.close()
        output_fastq = temp_output

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_fp:
        json_path = Path(json_fp.name)

    cmd: List[str] = [
        tools.fastp,
        "-i",
        str(input_fastq),
        "-o",
        str(output_fastq),
        "--json",
        str(json_path),
        "-w",
        "1",
        "--dont_overwrite",
    ] + args

    result: CommandResult
    try:
        result = run_command(cmd, timeout=300)
    except CommandError as exc:
        raise RuntimeError(f"fastp failed: {format_cmd(exc.result.cmd)}\n{exc.result.stderr}") from exc

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            report_data = json.load(fh)
    finally:
        json_path.unlink(missing_ok=True)
        if temp_output:
            temp_output.unlink(missing_ok=True)

    return FastpReport(summary=report_data, command=list(cmd), output_fastq=str(output_fastq))


def nanoplot_json(
    input_fastq: Path,
    tools: ToolPaths,
    extra_args: Optional[List[str]] = None,
) -> NanoPlotReport:
    args = extra_args or []
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "nanoplot.json"
        cmd: List[str] = [
            tools.nanoplot,
            "--fastq",
            str(input_fastq),
            "--json",
            str(json_path),
            "--threads",
            "1",
        ] + args
        try:
            run_command(cmd, timeout=600)
        except CommandError as exc:
            raise RuntimeError(f"NanoPlot failed: {format_cmd(exc.result.cmd)}\n{exc.result.stderr}") from exc

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    return NanoPlotReport(summary=data, command=list(cmd))

