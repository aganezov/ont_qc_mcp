import json
from typing import List, Optional

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server

from .config import ExecutionConfig, ToolPaths
from .flag_schemas import TOOL_FLAGS, get_tool_flags, get_tool_recipes
from .tools import (
    alignment_error_profile,
    alignment_summary,
    coverage_stats,
    env_check,
    filter_reads,
    qc_alignment,
    qc_reads,
    header_metadata_lookup,
    qscore_distribution,
    read_length_distribution,
    serialize_model,
)

server = Server("ont-qc-mcp")
EXEC_CFG = ExecutionConfig()


def _json_content(payload) -> List[types.TextContent]:
    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
    ]


async def env_status() -> List[types.TextContent]:
    """Check availability of required CLI tools."""
    status = env_check()
    return _json_content(serialize_model(status))


async def qc_alignment_tool(
    path: str,
    include_hist: bool = True,
    use_scaled: bool = False,
    flags: Optional[dict] = None,
) -> List[types.TextContent]:
    """Run cramino stats on a BAM/CRAM alignment."""
    stats = await anyio.to_thread.run_sync(
        lambda: qc_alignment(
            path,
            tools=ToolPaths(),
            include_hist=include_hist,
            use_scaled=use_scaled,
            flags=flags,
        )
    )
    return _json_content(serialize_model(stats))


async def qc_reads_tool(path: str, flags: Optional[dict] = None) -> List[types.TextContent]:
    """Run nanoq stats on a FASTQ file."""
    stats = await anyio.to_thread.run_sync(lambda: qc_reads(path, tools=ToolPaths(), flags=flags))
    return _json_content(serialize_model(stats))


async def filter_reads_tool(
    path: str,
    output_fastq: Optional[str] = None,
    flags: Optional[dict] = None,
) -> List[types.TextContent]:
    """Filter/trim reads with chopper."""
    report = await anyio.to_thread.run_sync(
        lambda: filter_reads(path, tools=ToolPaths(), output_fastq=output_fastq, flags=flags)
    )
    return _json_content(serialize_model(report))


async def read_length_distribution_tool(path: str, flags: Optional[dict] = None) -> List[types.TextContent]:
    """Return length percentiles/histogram from nanoq."""
    report = await anyio.to_thread.run_sync(
        lambda: read_length_distribution(path, tools=ToolPaths(), flags=flags)
    )
    return _json_content(serialize_model(report))


async def qscore_distribution_tool(path: str, flags: Optional[dict] = None) -> List[types.TextContent]:
    """Return q-score distribution/histogram from nanoq."""
    report = await anyio.to_thread.run_sync(lambda: qscore_distribution(path, tools=ToolPaths(), flags=flags))
    return _json_content(serialize_model(report))


async def coverage_stats_tool(
    path: str, window: Optional[int] = None, flags: Optional[dict] = None
) -> List[types.TextContent]:
    """Compute coverage with mosdepth."""
    report = await anyio.to_thread.run_sync(
        lambda: coverage_stats(path, tools=ToolPaths(), window=window, flags=flags)
    )
    return _json_content(serialize_model(report))


async def alignment_error_profile_tool(path: str, flags: Optional[dict] = None) -> List[types.TextContent]:
    """Parse error profile from samtools stats."""
    report = await anyio.to_thread.run_sync(
        lambda: alignment_error_profile(path, tools=ToolPaths(), flags=flags)
    )
    return _json_content(serialize_model(report))


async def alignment_summary_tool(
    path: str,
    include_coverage: bool = True,
    include_hist: bool = True,
    use_scaled: bool = False,
    coverage_window: Optional[int] = None,
    coverage_flags: Optional[dict] = None,
    cramino_flags: Optional[dict] = None,
) -> List[types.TextContent]:
    """Aggregate cramino stats + mosdepth + samtools error profile."""
    report = await anyio.to_thread.run_sync(
        lambda: alignment_summary(
            path,
            include_coverage=include_coverage,
            include_hist=include_hist,
            use_scaled=use_scaled,
            coverage_window=coverage_window,
            coverage_flags=coverage_flags,
            cramino_flags=cramino_flags,
            tools=ToolPaths(),
        )
    )
    return _json_content(serialize_model(report))


async def header_metadata_tool(
    path: str,
    file_type: Optional[str] = None,
    flags: Optional[dict] = None,
    max_lines: Optional[int] = None,
) -> List[types.TextContent]:
    """Extract header metadata from BAM/CRAM/VCF and return JSON + summary."""
    meta = await anyio.to_thread.run_sync(
        lambda: header_metadata_lookup(path, file_type=file_type, flags=flags, tools=ToolPaths(), max_lines=max_lines)
    )
    payload = serialize_model(meta)
    summary = meta.summary or ""
    return [
        types.TextContent(type="text", text=summary),
        types.TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
        ),
    ]


# Common schema fragments
_PATH_PROP = {"type": "string", "description": "Path to the input file"}
_FLAGS_PROP = {"type": "object", "description": "Optional CLI flags to pass to the underlying tool"}

TOOL_SCHEMAS: dict[str, dict] = {
    "env_status": {
        "type": "object",
        "properties": {},
    },
    "qc_alignment_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
            "include_hist": {"type": "boolean", "description": "Include histogram data", "default": True},
            "use_scaled": {"type": "boolean", "description": "Use scaled histogram bins", "default": False},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "qc_reads_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to FASTQ file"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "filter_reads_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to input FASTQ file"},
            "output_fastq": {"type": "string", "description": "Path for filtered output FASTQ"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "read_length_distribution_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to FASTQ file"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "qscore_distribution_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to FASTQ file"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "coverage_stats_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
            "window": {"type": "integer", "description": "Window size for coverage calculation"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "alignment_error_profile_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
            "flags": _FLAGS_PROP,
        },
        "required": ["path"],
    },
    "alignment_summary_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
            "include_coverage": {"type": "boolean", "description": "Include coverage stats", "default": True},
            "include_hist": {"type": "boolean", "description": "Include histogram data", "default": True},
            "use_scaled": {"type": "boolean", "description": "Use scaled histogram bins", "default": False},
            "coverage_window": {"type": "integer", "description": "Window size for coverage calculation"},
            "coverage_flags": {"type": "object", "description": "Flags for mosdepth coverage tool"},
            "cramino_flags": {"type": "object", "description": "Flags for cramino alignment tool"},
        },
        "required": ["path"],
    },
    "header_metadata_tool": {
        "type": "object",
        "properties": {
            "path": {**_PATH_PROP, "description": "Path to BAM/CRAM/VCF file"},
            "file_type": {
                "type": "string",
                "enum": ["bam", "cram", "sam", "vcf"],
                "description": "Override detected file type",
            },
            "flags": {
                "type": "object",
                "description": "Optional flags for samtools when reading BAM/CRAM headers",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of VCF header lines to read (guards huge headers)",
                "minimum": 1,
                "default": 2000,
            },
        },
        "required": ["path"],
    },
}

TOOL_HANDLERS: dict[str, tuple[str, callable]] = {
    "env_status": ("Check availability of required CLI tools", env_status),
    "qc_alignment_tool": ("Run cramino stats on a BAM/CRAM alignment", qc_alignment_tool),
    "qc_reads_tool": ("Run nanoq stats on a FASTQ file", qc_reads_tool),
    "filter_reads_tool": ("Filter/trim reads with chopper", filter_reads_tool),
    "read_length_distribution_tool": ("Return length percentiles/histogram from nanoq", read_length_distribution_tool),
    "qscore_distribution_tool": ("Return q-score distribution/histogram from nanoq", qscore_distribution_tool),
    "coverage_stats_tool": ("Compute coverage with mosdepth", coverage_stats_tool),
    "alignment_error_profile_tool": ("Parse error profile from samtools stats", alignment_error_profile_tool),
    "alignment_summary_tool": ("Aggregate cramino + mosdepth + samtools stats", alignment_summary_tool),
    "header_metadata_tool": ("Extract BAM/CRAM/VCF header metadata", header_metadata_tool),
}

_SUMMARY_TIMEOUT = max(
    EXEC_CFG.timeout_for("cramino"),
    EXEC_CFG.timeout_for("mosdepth"),
    EXEC_CFG.timeout_for("samtools"),
)
_SUMMARY_THREADS = max(
    EXEC_CFG.threads_for("cramino"),
    EXEC_CFG.threads_for("mosdepth"),
    EXEC_CFG.threads_for("samtools"),
)

TOOL_METADATA: dict[str, dict] = {
    "env_status": {
        "runtime_hint": "instant (<1s)",
        "io_hint": "No inputs; checks PATH for required CLI tools",
        "timeout_seconds": 30,
    },
    "qc_alignment_tool": {
        "runtime_hint": "medium (≈1-3 min for 1-5 GB BAM/CRAM)",
        "io_hint": "Reads BAM/CRAM; optional histograms",
        "default_threads": EXEC_CFG.threads_for("cramino"),
        "timeout_seconds": EXEC_CFG.timeout_for("cramino"),
        "when_to_use": "Alignment-level stats via cramino; use when you need read-length/identity summaries.",
    },
    "qc_reads_tool": {
        "runtime_hint": "fast-medium (tens of seconds for 1-2 GB FASTQ)",
        "io_hint": "Reads FASTQ; returns read-level QC metrics",
        "default_threads": EXEC_CFG.threads_for("nanoq"),
        "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
        "when_to_use": "Quick QC for raw reads with nanoq; combine with recipes for strict/lenient QC.",
    },
    "filter_reads_tool": {
        "runtime_hint": "medium (minutes for multi-GB FASTQ, depends on flags)",
        "io_hint": "Reads FASTQ, writes filtered FASTQ, emits JSON stats",
        "default_threads": EXEC_CFG.threads_for("chopper"),
        "timeout_seconds": EXEC_CFG.timeout_for("chopper"),
        "when_to_use": "Trim/filter ONT reads with chopper; specify output_fastq if persistence is needed.",
    },
    "read_length_distribution_tool": {
        "runtime_hint": "fast-medium (reuses nanoq stats; tens of seconds)",
        "io_hint": "Reads FASTQ; returns percentiles + histogram",
        "default_threads": EXEC_CFG.threads_for("nanoq"),
        "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
        "when_to_use": "Get length percentiles/histogram without full QC payload.",
    },
    "qscore_distribution_tool": {
        "runtime_hint": "fast-medium (reuses nanoq stats; tens of seconds)",
        "io_hint": "Reads FASTQ; returns q-score histogram",
        "default_threads": EXEC_CFG.threads_for("nanoq"),
        "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
        "when_to_use": "Retrieve q-score distribution quickly; same cost as qc_reads.",
    },
    "coverage_stats_tool": {
        "runtime_hint": "medium-heavy (minutes; depends on BAM/CRAM size and window)",
        "io_hint": "Reads BAM/CRAM; writes temporary outputs only",
        "default_threads": EXEC_CFG.threads_for("mosdepth"),
        "timeout_seconds": EXEC_CFG.timeout_for("mosdepth"),
        "when_to_use": "Depth-of-coverage summaries via mosdepth; tune window to control cost.",
    },
    "alignment_error_profile_tool": {
        "runtime_hint": "medium (1-3 min; scales with BAM/CRAM size)",
        "io_hint": "Reads BAM/CRAM; uses samtools stats",
        "default_threads": EXEC_CFG.threads_for("samtools"),
        "timeout_seconds": EXEC_CFG.timeout_for("samtools"),
        "when_to_use": "Base error profile and indel/substitution rates from samtools stats.",
    },
    "alignment_summary_tool": {
        "runtime_hint": "composite (bounded by cramino + mosdepth + samtools)",
        "io_hint": "Reads BAM/CRAM; aggregates multiple tools",
        "default_threads": _SUMMARY_THREADS,
        "timeout_seconds": _SUMMARY_TIMEOUT,
        "when_to_use": "One-shot QC combining alignment, coverage, and error profile.",
    },
    "header_metadata_tool": {
        "runtime_hint": "fast (header-only; seconds)",
        "io_hint": "Reads header via samtools (BAM/CRAM) or text (VCF)",
        "default_threads": EXEC_CFG.threads_for("samtools"),
        "timeout_seconds": EXEC_CFG.timeout_for("samtools"),
        "when_to_use": "Summarize contigs/samples/programs without full QC run.",
    },
}


def _tool_description(name: str, base_desc: str) -> str:
    meta = TOOL_METADATA.get(name)
    if not meta:
        return base_desc

    parts = []
    if runtime := meta.get("runtime_hint"):
        parts.append(f"runtime {runtime}")
    if threads := meta.get("default_threads"):
        parts.append(f"default threads={threads}")
    if timeout := meta.get("timeout_seconds"):
        parts.append(f"timeout≈{timeout}s")
    suffix = "; ".join(parts)
    return f"{base_desc} ({suffix})" if suffix else base_desc

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=_tool_description(name, description),
            inputSchema=TOOL_SCHEMAS[name],
        )
        for name, (description, _) in TOOL_HANDLERS.items()
    ]


@server.call_tool(validate_input=False)
async def dispatch_tool(name: str, arguments: Optional[dict]) -> types.CallToolResult:
    handler_entry = TOOL_HANDLERS.get(name)
    if handler_entry is None:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )

    _, handler = handler_entry
    try:
        result = await handler(**(arguments or {}))
    except TypeError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Invalid arguments for {name}: {exc}")],
            isError=True,
        )
    except Exception as exc:  # pragma: no cover
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(exc))],
            isError=True,
        )

    if isinstance(result, types.CallToolResult):
        return result

    return types.CallToolResult(content=result, isError=False)


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            uriTemplate="tool://flags/{tool}",
            description="Flag schemas for supported CLI tools",
            mimeType="application/json",
        ),
        types.ResourceTemplate(
            uriTemplate="tool://recipes/{tool}",
            description="Flag recipes/presets for supported CLI tools",
            mimeType="application/json",
        ),
        types.ResourceTemplate(
            uriTemplate="tool://guidance/{tool}",
            description="Runtime guidance and defaults for supported tools",
            mimeType="application/json",
        ),
    ]


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    resources: list[types.Resource] = []
    for tool in TOOL_FLAGS.keys():
        resources.append(
            types.Resource(
                name=f"{tool} flags",
                uri=f"tool://flags/{tool}",
                description=f"{tool} flag schema",
                mimeType="application/json",
            )
        )
        if get_tool_recipes(tool):
            resources.append(
                types.Resource(
                    name=f"{tool} recipes",
                    uri=f"tool://recipes/{tool}",
                    description=f"{tool} flag recipes",
                    mimeType="application/json",
                )
            )
    for tool_name in TOOL_HANDLERS.keys():
        resources.append(
            types.Resource(
                name=f"{tool_name} guidance",
                uri=f"tool://guidance/{tool_name}",
                description="Runtime guidance and defaults for tool selection",
                mimeType="application/json",
            )
        )
    return resources


@server.read_resource()
async def read_resource(uri: str):
    uri_str = str(uri)
    if uri_str.startswith("tool://flags/"):
        tool = uri_str.split("tool://flags/", 1)[1]
        payload = json.dumps({"tool": tool, "flags": [flag.model_dump() for flag in get_tool_flags(tool)]}, indent=2)
        return [ReadResourceContents(content=payload, mime_type="application/json")]

    if uri_str.startswith("tool://recipes/"):
        tool = uri_str.split("tool://recipes/", 1)[1]
        payload = json.dumps({"tool": tool, "recipes": get_tool_recipes(tool)}, indent=2)
        return [ReadResourceContents(content=payload, mime_type="application/json")]

    if uri_str.startswith("tool://guidance/"):
        tool = uri_str.split("tool://guidance/", 1)[1]
        meta = TOOL_METADATA.get(tool, {})
        base_desc = TOOL_HANDLERS.get(tool, ("", None))[0] if tool in TOOL_HANDLERS else ""
        payload = json.dumps(
            {
                "tool": tool,
                "description": base_desc,
                "runtime_hint": meta.get("runtime_hint"),
                "io_hint": meta.get("io_hint"),
                "defaults": {
                    "threads": meta.get("default_threads"),
                    "timeout_seconds": meta.get("timeout_seconds"),
                },
                "when_to_use": meta.get("when_to_use"),
                "schema_uri": f"tool://flags/{tool}" if tool in TOOL_FLAGS else None,
                "recipes_uri": f"tool://recipes/{tool}" if get_tool_recipes(tool) else None,
            },
            indent=2,
        )
        return [ReadResourceContents(content=payload, mime_type="application/json")]

    raise FileNotFoundError(f"Unknown resource URI: {uri_str}")


async def _async_main():
    async with stdio_server() as (read, write):
        init_options = server.create_initialization_options()
        await server.run(read, write, init_options, raise_exceptions=True)


def main():
    anyio.run(_async_main)


if __name__ == "__main__":
    main()
