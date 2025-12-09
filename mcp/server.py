import json
from typing import List, Optional

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import ToolPaths
from .tools import env_check, fastp_qc, fastq_eda, qc_alignment, qc_fastq, serialize_model

server = Server("ont-qc-mcp")


def _json_content(payload) -> List[types.TextContent]:
    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
    ]


@server.tool()
async def env_status() -> List[types.TextContent]:
    """Check availability of required CLI tools."""
    status = env_check()
    return _json_content(serialize_model(status))


@server.tool()
async def qc_fastq_tool(path: str) -> List[types.TextContent]:
    """Run seqkit stats on a FASTQ file."""
    stats = qc_fastq(path, tools=ToolPaths())
    return _json_content(serialize_model(stats))


@server.tool()
async def qc_alignment_tool(path: str) -> List[types.TextContent]:
    """Run samtools stats on a BAM/CRAM alignment."""
    stats = qc_alignment(path, tools=ToolPaths())
    return _json_content(serialize_model(stats))


@server.tool()
async def fastp_filter_tool(
    path: str,
    output_fastq: Optional[str] = None,
    extra_args: Optional[str] = None,
) -> List[types.TextContent]:
    """
    Run fastp; returns JSON report. extra_args is a space-separated string appended to fastp.
    """
    args = extra_args.split() if extra_args else None
    report = fastp_qc(path, tools=ToolPaths(), output_fastq=output_fastq, extra_args=args)
    return _json_content(serialize_model(report))


@server.tool()
async def fastq_eda_tool(
    path: str,
    use_nanoplot: bool = True,
    nanoplot_args: Optional[str] = None,
) -> List[types.TextContent]:
    """Aggregate FASTQ EDA (seqkit + optional NanoPlot JSON)."""
    args = nanoplot_args.split() if nanoplot_args else None
    report = fastq_eda(path, tools=ToolPaths(), use_nanoplot=use_nanoplot, nanoplot_args=args)
    return _json_content(serialize_model(report))


def main():
    with stdio_server() as (read, write):
        server.run(read, write)


if __name__ == "__main__":
    main()

