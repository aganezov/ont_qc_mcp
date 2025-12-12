import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, cast

import anyio
from importlib import metadata
from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from .cli_wrappers import FlagValidationError
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
    get_nanoq_cache_stats,
    qscore_distribution,
    qscore_distribution_bam,
    read_length_distribution,
    read_length_distribution_bam,
    serialize_model,
)

server = Server("ont-qc-mcp")
EXEC_CFG = ExecutionConfig()
logger = logging.getLogger(__name__)
_USE_JSON_LOG = os.getenv("MCP_LOG_FORMAT", "0").lower() in {"1", "true", "json", "structured"}
_ENABLE_CACHE_STATS = os.getenv("MCP_CACHE_STATS", "0").lower() not in {"", "0", "false", "False"}
_INCLUDE_PROVENANCE_VERBOSE = os.getenv("MCP_INCLUDE_PROVENANCE", "0").lower() not in {"", "0", "false", "False"}
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_REQUEST_START: ContextVar[float] = ContextVar("request_start", default=0.0)
_TOOL_PATHS: ToolPaths | None = None
_CONCURRENCY_SEM = anyio.Semaphore(EXEC_CFG.max_concurrent_operations) if EXEC_CFG.max_concurrent_operations else None


def _tool_paths() -> ToolPaths:
    """Return a memoized ToolPaths instance to avoid redundant resolution."""
    global _TOOL_PATHS
    if _TOOL_PATHS is None:
        _TOOL_PATHS = ToolPaths()
    return _TOOL_PATHS


def _log_event(level: int, message: str, **fields) -> None:
    request_id = _REQUEST_ID.get()
    if request_id:
        fields.setdefault("request_id", request_id)
    if _USE_JSON_LOG:
        logger.log(level, json.dumps({"event": message, **fields}, ensure_ascii=False))
    else:
        extras = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        prefix = f"[{request_id}] " if request_id else ""
        suffix = f" | {extras}" if extras else ""
        logger.log(level, "%s%s%s", prefix, message, suffix)


def _build_provenance(tool_name: str | None = None) -> dict[str, object]:
    provenance: dict[str, object] = {
        "threads_default": EXEC_CFG.default_threads,
        "per_tool_timeouts": EXEC_CFG.per_tool_timeouts,
        "request_id": _REQUEST_ID.get() or None,
        "concurrency_limit": EXEC_CFG.max_concurrent_operations,
    }
    start = _REQUEST_START.get()
    if start:
        provenance["duration_seconds"] = round(time.monotonic() - start, 3)
    if tool_name:
        provenance["effective_threads"] = EXEC_CFG.threads_for(tool_name)
        provenance["effective_timeout"] = EXEC_CFG.timeout_for(tool_name)
    if _INCLUDE_PROVENANCE_VERBOSE:
        pkg_version = None
        try:
            pkg_version = metadata.version("ont_qc_mcp")
        except metadata.PackageNotFoundError:
            pkg_version = None
        provenance.update(
            {
                "resolved_paths": _tool_paths().resolved(),
                "python_version": sys.version.split()[0],
                "package_version": pkg_version,
            }
        )
    return provenance


def _json_content(payload, tool_name: str | None = None) -> list[types.TextContent]:
    if isinstance(payload, dict):
        payload = {**payload, "provenance": _build_provenance(tool_name)}
    return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


async def env_status() -> list[types.TextContent]:
    """Check availability of required CLI tools."""
    status = env_check(_tool_paths())
    return _json_content(serialize_model(status), tool_name="env_status")


async def qc_alignment_tool(
    path: str,
    include_hist: bool = True,
    use_scaled: bool = False,
    flags: dict | None = None,
) -> list[types.TextContent]:
    """Run cramino stats on a BAM/CRAM alignment."""
    stats = await anyio.to_thread.run_sync(
        lambda: qc_alignment(
            path,
            tools=_tool_paths(),
            include_hist=include_hist,
            use_scaled=use_scaled,
            flags=flags,
        )
    )
    return _json_content(serialize_model(stats), tool_name="cramino")


async def qc_reads_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Run nanoq stats on a FASTQ file."""
    stats = await anyio.to_thread.run_sync(lambda: qc_reads(path, tools=_tool_paths(), flags=flags))
    return _json_content(serialize_model(stats), tool_name="nanoq")


async def filter_reads_tool(
    path: str,
    output_fastq: str | None = None,
    flags: dict | None = None,
) -> list[types.TextContent]:
    """Filter/trim reads with chopper."""
    report = await anyio.to_thread.run_sync(
        lambda: filter_reads(path, tools=_tool_paths(), output_fastq=output_fastq, flags=flags)
    )
    return _json_content(serialize_model(report), tool_name="chopper")


async def read_length_distribution_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Return length percentiles/histogram from nanoq."""
    report = await anyio.to_thread.run_sync(lambda: read_length_distribution(path, tools=_tool_paths(), flags=flags))
    return _json_content(serialize_model(report), tool_name="nanoq")


async def qscore_distribution_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Return q-score distribution/histogram from nanoq."""
    report = await anyio.to_thread.run_sync(lambda: qscore_distribution(path, tools=_tool_paths(), flags=flags))
    return _json_content(serialize_model(report), tool_name="nanoq")


async def read_length_distribution_bam_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Return length percentiles/histogram from BAM/CRAM via streaming nanoq."""
    report = await read_length_distribution_bam(path, tools=_tool_paths(), flags=flags)
    return _json_content(serialize_model(report), tool_name="nanoq")


async def qscore_distribution_bam_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Return q-score distribution/histogram from BAM/CRAM via streaming nanoq."""
    report = await qscore_distribution_bam(path, tools=_tool_paths(), flags=flags)
    return _json_content(serialize_model(report), tool_name="nanoq")


async def coverage_stats_tool(
    path: str,
    window: int | None = None,
    low_cov_threshold: float | None = None,
    flags: dict | None = None,
) -> list[types.TextContent]:
    """Compute coverage with mosdepth."""
    report = await anyio.to_thread.run_sync(
        lambda: coverage_stats(
            path,
            tools=_tool_paths(),
            window=window,
            low_cov_threshold=low_cov_threshold,
            flags=flags,
        )
    )
    return _json_content(serialize_model(report), tool_name="mosdepth")


async def alignment_error_profile_tool(path: str, flags: dict | None = None) -> list[types.TextContent]:
    """Parse error profile from samtools stats."""
    report = await anyio.to_thread.run_sync(lambda: alignment_error_profile(path, tools=_tool_paths(), flags=flags))
    return _json_content(serialize_model(report), tool_name="samtools")


async def alignment_summary_tool(
    path: str,
    include_coverage: bool = True,
    include_hist: bool = True,
    use_scaled: bool = False,
    include_error_profile: bool = False,
    coverage_window: int | None = None,
    coverage_low_cov_threshold: float | None = None,
    coverage_flags: dict | None = None,
    cramino_flags: dict | None = None,
    error_profile_flags: dict | None = None,
) -> list[types.TextContent]:
    """Aggregate cramino stats + mosdepth + samtools error profile."""
    report = await anyio.to_thread.run_sync(
        lambda: alignment_summary(
            path,
            include_coverage=include_coverage,
            include_hist=include_hist,
            use_scaled=use_scaled,
            include_error_profile=include_error_profile,
            coverage_window=coverage_window,
            coverage_low_cov_threshold=coverage_low_cov_threshold,
            coverage_flags=coverage_flags,
            cramino_flags=cramino_flags,
            error_profile_flags=error_profile_flags,
            tools=_tool_paths(),
        )
    )
    return _json_content(serialize_model(report), tool_name="alignment_summary_tool")


async def header_metadata_tool(
    path: str,
    file_type: str | None = None,
    flags: dict | None = None,
    max_lines: int | None = None,
) -> list[types.TextContent]:
    """Extract header metadata from BAM/CRAM/VCF and return JSON + summary."""
    meta = await anyio.to_thread.run_sync(
        lambda: header_metadata_lookup(path, file_type=file_type, flags=flags, tools=_tool_paths(), max_lines=max_lines)
    )
    payload = serialize_model(meta)
    payload["provenance"] = _build_provenance("header_metadata_tool")
    summary = meta.summary or ""
    return [
        types.TextContent(type="text", text=summary),
        types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2)),
    ]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable
    schema: dict
    metadata: dict[str, object]


# Common schema fragments
_PATH_PROP = {"type": "string", "description": "Path to the input file"}
_FLAGS_PROP = {"type": "object", "description": "Optional CLI flags to pass to the underlying tool"}


def _max_threads(*vals: int | None) -> int:
    candidates = [v for v in vals if v is not None]
    return max(candidates) if candidates else 0


_SUMMARY_TIMEOUT = max(
    EXEC_CFG.timeout_for("cramino"),
    EXEC_CFG.timeout_for("mosdepth"),
    EXEC_CFG.timeout_for("samtools"),
)
_SUMMARY_THREADS = _max_threads(
    EXEC_CFG.threads_for("cramino"),
    EXEC_CFG.threads_for("mosdepth"),
    EXEC_CFG.threads_for("samtools"),
)

_TOOL_SPECS = [
    ToolSpec(
        name="env_status",
        description="Check availability of required CLI tools",
        handler=env_status,
        schema={"type": "object", "properties": {}},
        metadata={
            "runtime_hint": "instant (<1s)",
            "io_hint": "No inputs; checks PATH for required CLI tools",
            "timeout_seconds": 30,
        },
    ),
    ToolSpec(
        name="qc_alignment_tool",
        description="Run cramino stats on a BAM/CRAM alignment",
        handler=qc_alignment_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "include_hist": {"type": "boolean", "description": "Include histogram data", "default": True},
                "use_scaled": {"type": "boolean", "description": "Use scaled histogram bins", "default": False},
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium (≈1-3 min for 1-5 GB BAM/CRAM)",
            "io_hint": "Reads BAM/CRAM; optional histograms",
            "default_threads": EXEC_CFG.threads_for("cramino"),
            "timeout_seconds": EXEC_CFG.timeout_for("cramino"),
            "when_to_use": "Alignment-level stats via cramino; use when you need read-length/identity summaries.",
        },
    ),
    ToolSpec(
        name="qc_reads_fastq_tool",
        description="Run nanoq stats on a FASTQ file",
        handler=qc_reads_tool,
        schema={
            "type": "object",
            "properties": {"path": {**_PATH_PROP, "description": "Path to FASTQ file"}, "flags": _FLAGS_PROP},
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "fast-medium (tens of seconds for 1-2 GB FASTQ)",
            "io_hint": "Reads FASTQ; returns read-level QC metrics",
            "default_threads": EXEC_CFG.threads_for("nanoq"),
            "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
            "when_to_use": "Quick QC for raw reads with nanoq; combine with recipes for strict/lenient QC.",
        },
    ),
    ToolSpec(
        name="filter_reads_fastq_tool",
        description="Filter/trim FASTQ reads with chopper",
        handler=filter_reads_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to input FASTQ file"},
                "output_fastq": {"type": "string", "description": "Path for filtered output FASTQ"},
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium (minutes for multi-GB FASTQ, depends on flags)",
            "io_hint": "Reads FASTQ, writes filtered FASTQ, emits JSON stats",
            "default_threads": EXEC_CFG.threads_for("chopper"),
            "timeout_seconds": EXEC_CFG.timeout_for("chopper"),
            "when_to_use": "Trim/filter ONT reads with chopper; specify output_fastq if persistence is needed.",
        },
    ),
    ToolSpec(
        name="read_length_distribution_fastq_tool",
        description="FASTQ: length percentiles/histogram via nanoq",
        handler=read_length_distribution_tool,
        schema={
            "type": "object",
            "properties": {"path": {**_PATH_PROP, "description": "Path to FASTQ file"}, "flags": _FLAGS_PROP},
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "fast-medium (reuses nanoq stats; tens of seconds)",
            "io_hint": "Reads FASTQ; returns percentiles + histogram",
            "default_threads": EXEC_CFG.threads_for("nanoq"),
            "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
            "when_to_use": "Get length percentiles/histogram without full QC payload.",
        },
    ),
    ToolSpec(
        name="read_length_distribution_bam_tool",
        description="BAM/CRAM: length percentiles/histogram via samtools fastq -> nanoq streaming",
        handler=read_length_distribution_bam_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium (streams BAM/CRAM through samtools fastq + nanoq)",
            "io_hint": "Streams BAM/CRAM to FASTQ; returns percentiles + histogram",
            "default_threads": EXEC_CFG.threads_for("nanoq"),
            "timeout_seconds": max(EXEC_CFG.timeout_for("samtools"), EXEC_CFG.timeout_for("nanoq")),
            "when_to_use": "Length percentiles/histogram from BAM/CRAM when FASTQ is not available.",
        },
    ),
    ToolSpec(
        name="qscore_distribution_fastq_tool",
        description="FASTQ: q-score histogram via nanoq",
        handler=qscore_distribution_tool,
        schema={
            "type": "object",
            "properties": {"path": {**_PATH_PROP, "description": "Path to FASTQ file"}, "flags": _FLAGS_PROP},
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "fast-medium (reuses nanoq stats; tens of seconds)",
            "io_hint": "Reads FASTQ; returns q-score histogram",
            "default_threads": EXEC_CFG.threads_for("nanoq"),
            "timeout_seconds": EXEC_CFG.timeout_for("nanoq"),
            "when_to_use": "Retrieve q-score distribution quickly; same cost as qc_reads.",
        },
    ),
    ToolSpec(
        name="qscore_distribution_bam_tool",
        description="BAM/CRAM: q-score histogram via samtools fastq -> nanoq streaming",
        handler=qscore_distribution_bam_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium (streams BAM/CRAM through samtools fastq + nanoq)",
            "io_hint": "Streams BAM/CRAM to FASTQ; returns q-score histogram",
            "default_threads": EXEC_CFG.threads_for("nanoq"),
            "timeout_seconds": max(EXEC_CFG.timeout_for("samtools"), EXEC_CFG.timeout_for("nanoq")),
            "when_to_use": "Q-score distribution from BAM/CRAM when FASTQ is not available.",
        },
    ),
    ToolSpec(
        name="coverage_stats_tool",
        description=(
            "Compute coverage with mosdepth (BAM/CRAM). Parses summary.txt only; ignores per-base/quantized outputs."
        ),
        handler=coverage_stats_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "window": {"type": "integer", "description": "Window size for coverage calculation"},
                "low_cov_threshold": {
                    "type": "number",
                    "description": "Mark contigs with mean depth below this threshold",
                },
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium-heavy (minutes; depends on BAM/CRAM size and window)",
            "io_hint": "Reads BAM/CRAM; writes temporary outputs only",
            "default_threads": EXEC_CFG.threads_for("mosdepth"),
            "timeout_seconds": EXEC_CFG.timeout_for("mosdepth"),
            "when_to_use": (
                "Depth-of-coverage summaries via mosdepth (summary.txt only); "
                "tune window/quantize/fast-mode to control cost."
            ),
        },
    ),
    ToolSpec(
        name="alignment_error_profile_tool",
        description="Parse error profile from samtools stats (BAM/CRAM)",
        handler=alignment_error_profile_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "flags": _FLAGS_PROP,
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "medium (1-3 min; scales with BAM/CRAM size)",
            "io_hint": "Reads BAM/CRAM; uses samtools stats",
            "default_threads": EXEC_CFG.threads_for("samtools"),
            "timeout_seconds": EXEC_CFG.timeout_for("samtools"),
            "when_to_use": (
                "Base error profile and indel/substitution rates from samtools stats; opt-in to avoid extra cost."
            ),
        },
    ),
    ToolSpec(
        name="alignment_summary_tool",
        description="Aggregate cramino + mosdepth + samtools stats (BAM/CRAM)",
        handler=alignment_summary_tool,
        schema={
            "type": "object",
            "properties": {
                "path": {**_PATH_PROP, "description": "Path to BAM/CRAM alignment file"},
                "include_coverage": {"type": "boolean", "description": "Include coverage stats", "default": True},
                "include_hist": {"type": "boolean", "description": "Include histogram data", "default": True},
                "use_scaled": {"type": "boolean", "description": "Use scaled histogram bins", "default": False},
                "include_error_profile": {
                    "type": "boolean",
                    "description": "Include samtools stats error profile",
                    "default": False,
                },
                "coverage_window": {"type": "integer", "description": "Window size for coverage calculation"},
                "coverage_low_cov_threshold": {
                    "type": "number",
                    "description": "Mark contigs with mean depth below this threshold",
                },
                "coverage_flags": {"type": "object", "description": "Flags for mosdepth coverage tool"},
                "cramino_flags": {"type": "object", "description": "Flags for cramino alignment tool"},
                "error_profile_flags": {"type": "object", "description": "Flags for samtools stats error profile"},
            },
            "required": ["path"],
        },
        metadata={
            "runtime_hint": "composite (bounded by cramino + mosdepth + samtools)",
            "io_hint": "Reads BAM/CRAM; aggregates multiple tools",
            "default_threads": _SUMMARY_THREADS,
            "timeout_seconds": _SUMMARY_TIMEOUT,
            "when_to_use": ("One-shot QC combining alignment, coverage, and optional error profile (opt-in)."),
        },
    ),
    ToolSpec(
        name="header_metadata_tool",
        description="Extract BAM/CRAM/VCF header metadata",
        handler=header_metadata_tool,
        schema={
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
        metadata={
            "runtime_hint": "fast (header-only; seconds)",
            "io_hint": "Reads header via samtools (BAM/CRAM) or text (VCF)",
            "default_threads": EXEC_CFG.threads_for("samtools"),
            "timeout_seconds": EXEC_CFG.timeout_for("samtools"),
            "when_to_use": "Summarize contigs/samples/programs without full QC run.",
        },
    ),
]

TOOL_SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOL_SPECS}


def _tool_description(spec: ToolSpec) -> str:
    meta = spec.metadata or {}
    base_desc = spec.description

    parts = []
    if runtime := meta.get("runtime_hint"):
        parts.append(f"runtime {runtime}")
    if threads := meta.get("default_threads"):
        parts.append(f"default threads={threads}")
    if timeout := meta.get("timeout_seconds"):
        parts.append(f"timeout≈{timeout}s")
    suffix = "; ".join(parts)
    return f"{base_desc} ({suffix})" if suffix else base_desc


def _error_result(
    kind: str, message: str, tool: str | None = None, details: dict | None = None
) -> types.CallToolResult:
    """Return a structured MCP error payload."""
    request_id = _REQUEST_ID.get()
    payload = {
        "kind": kind,
        "message": message,
        "tool": tool,
        "details": details or {},
    }
    if request_id:
        payload["request_id"] = request_id
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
        isError=True,
    )


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=spec.name,
            description=_tool_description(spec),
            inputSchema=spec.schema,
        )
        for spec in _TOOL_SPECS
    ]


@server.call_tool(validate_input=True)
async def dispatch_tool(name: str, arguments: dict | None) -> types.CallToolResult:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return _error_result("validation", f"Unknown tool: {name}", tool=name)

    request_id = str(uuid.uuid4())[:8]
    _REQUEST_ID.set(request_id)
    _REQUEST_START.set(time.monotonic())
    threads_hint = EXEC_CFG.threads_for(name)
    timeout_hint = EXEC_CFG.timeout_for(name)

    _log_event(
        logging.INFO,
        "tool_call_start",
        tool=name,
        args=arguments,
        threads=threads_hint,
        timeout=timeout_hint,
        concurrency_limit=EXEC_CFG.max_concurrent_operations,
    )

    try:
        if _CONCURRENCY_SEM:
            async with _CONCURRENCY_SEM:
                result = await spec.handler(**(arguments or {}))
        else:
            result = await spec.handler(**(arguments or {}))
    except FlagValidationError as exc:
        _log_event(logging.WARNING, "validation_error", tool=name, error=str(exc))
        return _error_result("validation", str(exc), tool=name)
    except FileNotFoundError as exc:
        _log_event(logging.WARNING, "not_found", tool=name, error=str(exc))
        return _error_result("not_found", str(exc), tool=name)
    except ValueError as exc:
        _log_event(logging.WARNING, "validation_error", tool=name, error=str(exc))
        return _error_result("validation", str(exc), tool=name)
    except TypeError as exc:
        _log_event(logging.WARNING, "type_error", tool=name, error=str(exc))
        return _error_result("validation", f"Invalid arguments for {name}: {exc}", tool=name)
    except Exception as exc:  # pragma: no cover
        _log_event(logging.ERROR, "runtime_error", tool=name, error=str(exc))
        return _error_result("runtime", str(exc), tool=name)
    finally:
        duration_ms = round((time.monotonic() - _REQUEST_START.get()) * 1000, 2)
        _log_event(logging.INFO, "tool_call_finished", tool=name, duration_ms=duration_ms)

    if isinstance(result, types.CallToolResult):
        return result

    return types.CallToolResult(content=result, isError=False)


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    templates = [
        types.ResourceTemplate(
            name="tool-flags",
            uriTemplate="tool://flags/{tool}",
            description="Flag schemas for supported CLI tools",
            mimeType="application/json",
        ),
        types.ResourceTemplate(
            name="tool-recipes",
            uriTemplate="tool://recipes/{tool}",
            description="Flag recipes/presets for supported CLI tools",
            mimeType="application/json",
        ),
        types.ResourceTemplate(
            name="tool-guidance",
            uriTemplate="tool://guidance/{tool}",
            description="Runtime guidance and defaults for supported tools",
            mimeType="application/json",
        ),
    ]
    if _ENABLE_CACHE_STATS:
        templates.append(
            types.ResourceTemplate(
                name="cache-stats",
                uriTemplate="tool://stats/cache",
                description="Cache hit/miss/eviction counters for streaming nanoq",
                mimeType="application/json",
            )
        )
    return templates


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    resources: list[types.Resource] = []
    for tool in TOOL_FLAGS.keys():
        resources.append(
            types.Resource(
                name=f"{tool} flags",
                uri=cast(AnyUrl, f"tool://flags/{tool}"),
                description=f"{tool} flag schema",
                mimeType="application/json",
            )
        )
        if get_tool_recipes(tool):
            resources.append(
                types.Resource(
                    name=f"{tool} recipes",
                    uri=cast(AnyUrl, f"tool://recipes/{tool}"),
                    description=f"{tool} flag recipes",
                    mimeType="application/json",
                )
            )
    for tool_name in TOOL_SPECS.keys():
        resources.append(
            types.Resource(
                name=f"{tool_name} guidance",
                uri=cast(AnyUrl, f"tool://guidance/{tool_name}"),
                description="Runtime guidance and defaults for tool selection",
                mimeType="application/json",
            )
        )
    if _ENABLE_CACHE_STATS:
        resources.append(
            types.Resource(
                name="nanoq cache stats",
                uri=cast(AnyUrl, "tool://stats/cache"),
                description="Cache hit/miss/eviction counters for nanoq",
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
        spec = TOOL_SPECS.get(tool)
        if spec is None:
            raise FileNotFoundError(f"Unknown resource URI: {uri_str}")
        meta = spec.metadata or {}
        payload = json.dumps(
            {
                "tool": tool,
                "description": spec.description,
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

    if uri_str == "tool://stats/cache":
        if not _ENABLE_CACHE_STATS:
            raise FileNotFoundError(f"Unknown resource URI: {uri_str}")
        stats = get_nanoq_cache_stats()
        payload = json.dumps({"nanoq_cache": stats}, indent=2)
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
