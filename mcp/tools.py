import json
from pathlib import Path
from typing import Dict, List, Optional

from .cli_wrappers import fastp_filter, nanoplot_json, samtools_stats, seqkit_stats
from .config import ToolPaths
from .schemas import EnvStatus, FastpReport, FastqEdaReport, SamtoolsStats, SeqkitStats


def env_check(tools: Optional[ToolPaths] = None) -> EnvStatus:
    tools = tools or ToolPaths()
    missing = tools.missing()
    resolved = tools.as_dict()
    available = {k: k not in missing for k in resolved.keys()}
    return EnvStatus(available=available, resolved_paths=resolved, missing=missing)


def qc_fastq(path: str, tools: Optional[ToolPaths] = None) -> SeqkitStats:
    tools = tools or ToolPaths()
    fastq_path = Path(path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ not found: {fastq_path}")
    return seqkit_stats(fastq_path, tools)


def qc_alignment(path: str, tools: Optional[ToolPaths] = None) -> SamtoolsStats:
    tools = tools or ToolPaths()
    aln_path = Path(path)
    if not aln_path.exists():
        raise FileNotFoundError(f"Alignment not found: {aln_path}")
    return samtools_stats(aln_path, tools)


def fastp_qc(
    path: str,
    tools: Optional[ToolPaths] = None,
    output_fastq: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> FastpReport:
    tools = tools or ToolPaths()
    fastq_path = Path(path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ not found: {fastq_path}")
    output_path = Path(output_fastq) if output_fastq else None
    return fastp_filter(fastq_path, tools, output_fastq=output_path, extra_args=extra_args)


def fastq_eda(
    path: str,
    tools: Optional[ToolPaths] = None,
    use_nanoplot: bool = True,
    nanoplot_args: Optional[List[str]] = None,
) -> FastqEdaReport:
    tools = tools or ToolPaths()
    fastq_path = Path(path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ not found: {fastq_path}")

    seqkit = seqkit_stats(fastq_path, tools)
    nanoplot_report = None
    if use_nanoplot:
        try:
            nanoplot_report = nanoplot_json(fastq_path, tools, extra_args=nanoplot_args)
        except FileNotFoundError:
            # NanoPlot not installed; still return seqkit stats
            nanoplot_report = None

    return FastqEdaReport(seqkit=seqkit, nanoplot=nanoplot_report)


def serialize_model(model) -> Dict[str, object]:
    """Return a JSON-serializable dict from a pydantic model."""
    return json.loads(model.model_dump_json())

