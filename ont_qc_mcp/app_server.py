import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable

import anyio
import jsonschema
import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from .cli_wrappers import FlagValidationError
from .config import ExecutionConfig, ToolPaths
from .stdio_compat import stdio_server_compat
from .threadpool import get_executor, run_sync
from .v2_alignment_qc import alignment_qc as execute_alignment_qc
from .v2_contracts import (
    AlignmentQCRequest,
    BedQCRequest,
    CoverageQCRequest,
    EnvironmentStatusRequest,
    ExecutionErrorResponse,
    FilterReadsRequest,
    HeaderInfoRequest,
    IgvSnapshotsRequest,
    ReadQCRequest,
    RunSummaryRequest,
    ToolContract,
    ValidationErrorResponse,
    ValidationIssue,
    VariantQCRequest,
    api_v2_contracts,
)
from .v2_coverage_qc import coverage_qc as execute_coverage_qc
from .v2_execution import PipelineOutputLimitError, PipelineStageError
from .v2_read_qc import read_qc as execute_read_qc
from .v2_supporting_tools import (
    bed_qc as execute_bed_qc,
    environment_status as execute_environment_status,
    filter_reads as execute_filter_reads,
    header_info as execute_header_info,
    igv_snapshots as execute_igv_snapshots,
    run_summary as execute_run_summary,
)
from .v2_variant_qc import variant_qc as execute_variant_qc

EXEC_CFG = ExecutionConfig()
logger = logging.getLogger(__name__)
_USE_JSON_LOG = os.getenv("MCP_LOG_FORMAT", "0").lower() in {"1", "true", "json", "structured"}
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_TOOL_PATHS: ToolPaths | None = None
_CONCURRENCY_SEM = anyio.Semaphore(EXEC_CFG.max_concurrent_operations) if EXEC_CFG.max_concurrent_operations else None
_CONFIG_SCOPE_NOTE = {
    "env_read_at_startup": True,
    "per_call_overrides": "Use typed request fields and namespaced extra_args; igv_snapshots supports output_dir.",
    "multi_client_note": "Different defaults require separate server instances.",
}


def _use_compat_stdio() -> bool:
    """Select the server stdio transport."""
    value = os.getenv("MCP_STDIO_TRANSPORT", "anyio").strip().lower()
    if value in {"compat", "asyncio", "pipe"}:
        return True
    if value in {"anyio", "default"}:
        return False
    logger.warning("Unknown MCP_STDIO_TRANSPORT=%r; falling back to 'anyio'", value)
    return False


def _tool_paths() -> ToolPaths:
    """Return a memoized ToolPaths instance to avoid redundant resolution."""
    global _TOOL_PATHS
    if _TOOL_PATHS is None:
        _TOOL_PATHS = ToolPaths()
    return _TOOL_PATHS


def _log_event(level: int, message: str, **fields: object) -> None:
    request_id = _REQUEST_ID.get()
    if request_id:
        fields.setdefault("request_id", request_id)
    if _USE_JSON_LOG:
        logger.log(level, json.dumps({"event": message, **fields}, ensure_ascii=False))
    else:
        extras = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        prefix = f"[{request_id}] " if request_id else ""
        suffix = f" | {extras}" if extras else ""
        logger.log(level, "%s%s%s", prefix, message, suffix)


def _json_content(payload: BaseModel | dict[str, object] | list[object]) -> list[types.ContentBlock]:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return [types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False, indent=2))]


async def read_qc_tool(request: ReadQCRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_read_qc, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def alignment_qc_tool(request: AlignmentQCRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_alignment_qc, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def coverage_qc_tool(request: CoverageQCRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_coverage_qc, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def variant_qc_tool(request: VariantQCRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_variant_qc, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def environment_status_tool(request: EnvironmentStatusRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_environment_status, request, tools=_tool_paths())
    return _json_content(response)


async def header_info_tool(request: HeaderInfoRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_header_info, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def bed_qc_tool(request: BedQCRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_bed_qc, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def run_summary_tool(request: RunSummaryRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_run_summary, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def filter_reads_tool(request: FilterReadsRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_filter_reads, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


async def igv_snapshots_tool(request: IgvSnapshotsRequest) -> list[types.ContentBlock]:
    response = await run_sync(execute_igv_snapshots, request, tools=_tool_paths(), exec_cfg=EXEC_CFG)
    return _json_content(response)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Awaitable[Sequence[types.ContentBlock]]]
    schema: dict[str, object]
    metadata: dict[str, object]


def _max_threads(*values: int | None) -> int:
    candidates = [value for value in values if value is not None]
    return max(candidates) if candidates else 0


_CONTRACTS = api_v2_contracts()
_REQUEST_MODELS: dict[str, type[BaseModel]] = {name: contract.request_model for name, contract in _CONTRACTS.items()}
_HANDLERS: dict[str, Callable[..., Awaitable[Sequence[types.ContentBlock]]]] = {
    "read_qc": read_qc_tool,
    "alignment_qc": alignment_qc_tool,
    "coverage_qc": coverage_qc_tool,
    "variant_qc": variant_qc_tool,
    "environment_status": environment_status_tool,
    "header_info": header_info_tool,
    "bed_qc": bed_qc_tool,
    "run_summary": run_summary_tool,
    "filter_reads": filter_reads_tool,
    "igv_snapshots": igv_snapshots_tool,
}
_DESCRIPTIONS = {
    "read_qc": "Read length and quality QC for FASTQ or selected complete sequences from BAM/CRAM",
    "alignment_qc": "Selected-record alignment counts, MAPQ, aligned-base quality, identity, and error metrics",
    "coverage_qc": "Indexed BAM/CRAM depth and breadth for contigs, requested intervals, or windows",
    "variant_qc": "BCF/VCF general, SNP, and indel QC with optional regional grouping",
    "environment_status": "Check availability and resolved paths for the required native tools",
    "header_info": "Extract BAM/CRAM/SAM/VCF header metadata",
    "bed_qc": "Validate BED structure and summarize accepted intervals",
    "run_summary": "Summarize an ONT sequencing-summary TSV",
    "filter_reads": "Filter or trim FASTQ reads with atomic output publication",
    "igv_snapshots": "Generate IGV snapshots from zero-based half-open regions or a caller batch file",
}
_METADATA: dict[str, dict[str, object]] = {
    "read_qc": {
        "runtime_hint": "fast to heavy; depends on input format, selected records, and requested distributions",
        "io_hint": "Reads FASTQ directly or streams selected BAM/CRAM records through samtools and nanoq",
        "default_threads": _max_threads(EXEC_CFG.threads_for("samtools"), EXEC_CFG.threads_for("nanoq")),
        "timeout_seconds": max(EXEC_CFG.timeout_for("samtools"), EXEC_CFG.timeout_for("nanoq")),
        "when_to_use": (
            "Use for whole stored-read length and quality evidence; regional BAM/CRAM selection keeps "
            "complete sequences."
        ),
    },
    "alignment_qc": {
        "runtime_hint": "medium to heavy; only requested metric backends run",
        "io_hint": "Reads BAM/CRAM; regional calls require an existing index",
        "default_threads": _max_threads(EXEC_CFG.threads_for("samtools"), EXEC_CFG.threads_for("cramino")),
        "timeout_seconds": max(EXEC_CFG.timeout_for("samtools"), EXEC_CFG.timeout_for("cramino")),
        "when_to_use": (
            "Use for alignment-record and aligned-base evidence. Call coverage_qc separately for reference depth."
        ),
    },
    "coverage_qc": {
        "runtime_hint": "medium to heavy; depends on alignment size and requested row count",
        "io_hint": "Reads indexed BAM/CRAM and creates bounded temporary mosdepth outputs",
        "default_threads": EXEC_CFG.threads_for("mosdepth"),
        "timeout_seconds": EXEC_CFG.timeout_for("mosdepth"),
        "when_to_use": "Use for reference-domain depth and threshold breadth across contigs, intervals, or windows.",
    },
    "variant_qc": {
        "runtime_hint": "medium; scales with VCF/BCF size and grouped regions",
        "io_hint": "Reads VCF/BCF with bcftools; regional calls require an existing index",
        "default_threads": EXEC_CFG.threads_for("bcftools"),
        "timeout_seconds": EXEC_CFG.timeout_for("bcftools"),
        "when_to_use": "Use for general, SNP, and indel record summaries with explicit overlap selection.",
    },
    "environment_status": {
        "runtime_hint": "instant (<1s)",
        "io_hint": "No input files; resolves native executables and the IGV runtime",
        "timeout_seconds": 30,
        "when_to_use": "Use before a workflow to identify missing native tools.",
    },
    "header_info": {
        "runtime_hint": "fast; header only",
        "io_hint": "Reads alignment headers through samtools or VCF headers as text/BGZF",
        "default_threads": EXEC_CFG.threads_for("samtools"),
        "timeout_seconds": EXEC_CFG.timeout_for("samtools"),
        "when_to_use": "Use to inspect contigs, samples, and program metadata without a full QC scan.",
    },
    "bed_qc": {
        "runtime_hint": "fast; pure Python",
        "io_hint": "Reads one BED file without changing it",
        "timeout_seconds": 30,
        "when_to_use": "Use before supplying a BED region source to numerical or IGV tools.",
    },
    "run_summary": {
        "runtime_hint": "fast to medium; pure Python",
        "io_hint": "Reads one ONT sequencing-summary TSV",
        "timeout_seconds": 120,
        "when_to_use": "Use for run yield, N50, Q-score, and anchored one-hour yield windows.",
    },
    "filter_reads": {
        "runtime_hint": "medium; depends on FASTQ size and selection",
        "io_hint": "Reads FASTQ and atomically publishes plain or gzip FASTQ output",
        "default_threads": EXEC_CFG.threads_for("chopper"),
        "timeout_seconds": EXEC_CFG.timeout_for("chopper"),
        "when_to_use": "Use for explicit FASTQ transformation; run read_qc on the resulting file when QC is needed.",
    },
    "igv_snapshots": {
        "runtime_hint": "slow (about 30s to 5min depending on region count)",
        "io_hint": "Reads reference/tracks and writes PNG or SVG snapshots",
        "timeout_seconds": EXEC_CFG.timeout_for("igv"),
        "when_to_use": "Use for visual inspection after numerical QC identifies loci of interest.",
    },
}
_PUBLIC_RECIPES: dict[str, dict[str, object]] = {
    "alignment_qc": {
        "alignment_and_coverage": {
            "description": "Replacement for the retired composite alignment summary",
            "calls": [
                {"tool": "alignment_qc", "arguments": {"path": "<alignment.bam>"}},
                {"tool": "coverage_qc", "arguments": {"path": "<alignment.bam>"}},
            ],
            "note": (
                "Pass the same regions, reference_path, and compatible selection intent explicitly when both reports "
                "must describe the same domain."
            ),
        }
    }
}


def _build_spec(contract: ToolContract) -> ToolSpec:
    return ToolSpec(
        name=contract.name,
        description=_DESCRIPTIONS[contract.name],
        handler=_HANDLERS[contract.name],
        schema=contract.request_model.model_json_schema(mode="validation"),
        metadata=_METADATA[contract.name],
    )


_TOOL_SPECS = [_build_spec(contract) for contract in _CONTRACTS.values()]
TOOL_SPECS: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOL_SPECS}


def _tool_description(spec: ToolSpec) -> str:
    parts = []
    if runtime := spec.metadata.get("runtime_hint"):
        parts.append(f"runtime {runtime}")
    if threads := spec.metadata.get("default_threads"):
        parts.append(f"default threads={threads}")
    if timeout := spec.metadata.get("timeout_seconds"):
        parts.append(f"timeout≈{timeout}s")
    suffix = "; ".join(parts)
    return f"{spec.description} ({suffix})" if suffix else spec.description


def _tool_meta(_spec: ToolSpec) -> dict[str, object]:
    return {"config_scope": _CONFIG_SCOPE_NOTE}


def _validation_result(tool: str, error: ValidationError | ValueError | FlagValidationError) -> types.CallToolResult:
    if isinstance(error, ValidationError):
        issues = []
        for entry in error.errors(include_url=False, include_context=False, include_input=False):
            message = str(entry["msg"])
            if message.startswith("Value error, "):
                message = message.removeprefix("Value error, ")
            issues.append(
                ValidationIssue(
                    location=[value if isinstance(value, (str, int)) else str(value) for value in entry["loc"]],
                    message=message,
                    code=str(entry["type"]),
                )
            )
    else:
        issues = [ValidationIssue(location=[], message=str(error), code="value_error")]
    payload = ValidationErrorResponse(tool=tool, message="Request validation failed", issues=issues)
    return types.CallToolResult(content=_json_content(payload), is_error=True)


def _execution_result(
    tool: str,
    error: BaseException,
    *,
    stage: str | None = None,
    backend: str | None = None,
    exit_code: int | None = None,
    timed_out: bool = False,
) -> types.CallToolResult:
    resolved_stage = stage or str(getattr(error, "stage", "execution"))
    payload = ExecutionErrorResponse(
        tool=tool,
        stage=resolved_stage,
        message=str(error),
        backend=backend or resolved_stage,
        exit_code=exit_code,
        timed_out=timed_out,
        cancelled=False,
    )
    return types.CallToolResult(content=_json_content(payload), is_error=True)


async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=spec.name,
            description=_tool_description(spec),
            input_schema=spec.schema,
            _meta=_tool_meta(spec),
        )
        for spec in _TOOL_SPECS
    ]


async def dispatch_tool(name: str, arguments: dict | None) -> types.CallToolResult:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return _validation_result(name, ValueError(f"Unknown tool: {name}"))

    request_model = _REQUEST_MODELS.get(name)
    try:
        validated = request_model.model_validate(arguments or {}) if request_model is not None else None
    except ValidationError as error:
        return _validation_result(name, error)

    request_id = str(uuid.uuid4())[:8]
    request_token = _REQUEST_ID.set(request_id)
    started = time.monotonic()
    _log_event(
        logging.INFO,
        "tool_call_start",
        tool=name,
        args=arguments,
        concurrency_limit=EXEC_CFG.max_concurrent_operations,
    )
    try:
        if _CONCURRENCY_SEM:
            async with _CONCURRENCY_SEM:
                if request_model is not None:
                    result = await spec.handler(request=validated)
                else:
                    result = await spec.handler(**(arguments or {}))
        else:
            if request_model is not None:
                result = await spec.handler(request=validated)
            else:
                result = await spec.handler(**(arguments or {}))
    except FlagValidationError as error:
        _log_event(logging.WARNING, "validation_error", tool=name, error=str(error))
        return _validation_result(name, error)
    except ValidationError as error:
        _log_event(logging.ERROR, "result_validation_error", tool=name, error=str(error))
        return _execution_result(name, error, stage="result_validation", backend="server")
    except ValueError as error:
        _log_event(logging.ERROR, "execution_validation_error", tool=name, error=str(error))
        return _execution_result(name, error, stage="execution_validation", backend="server")
    except PipelineStageError as error:
        _log_event(logging.ERROR, "execution_error", tool=name, stage=error.stage, error=str(error))
        return _execution_result(
            name,
            error,
            stage=error.stage,
            backend=error.stage,
            exit_code=error.result.returncode,
        )
    except PipelineOutputLimitError as error:
        _log_event(logging.ERROR, "execution_error", tool=name, stage=error.stage, error=str(error))
        return _execution_result(name, error, stage=error.stage, backend=error.stage)
    except TimeoutError as error:
        _log_event(logging.ERROR, "execution_timeout", tool=name, error=str(error))
        return _execution_result(name, error, timed_out=True)
    except FileNotFoundError as error:
        _log_event(logging.WARNING, "input_not_found", tool=name, error=str(error))
        return _execution_result(name, error, stage="input_validation", backend="server")
    except Exception as error:  # pragma: no cover - exercised through wire subprocesses
        _log_event(logging.ERROR, "execution_error", tool=name, error=str(error))
        return _execution_result(name, error)
    finally:
        duration_ms = round((time.monotonic() - started) * 1000, 2)
        _log_event(logging.INFO, "tool_call_finished", tool=name, duration_ms=duration_ms)
        _REQUEST_ID.reset(request_token)

    if isinstance(result, types.CallToolResult):
        return result
    return types.CallToolResult(content=list(result), is_error=False)


async def list_resource_templates() -> list[types.ResourceTemplate]:
    templates = [
        types.ResourceTemplate(
            name="tool-recipes",
            uri_template="tool://recipes/{tool}",
            description="Public API v2 workflow recipes",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            name="tool-guidance",
            uri_template="tool://guidance/{tool}",
            description="Runtime guidance and defaults for public API v2 tools",
            mime_type="application/json",
        ),
    ]
    return templates


async def list_resources() -> list[types.Resource]:
    resources = [
        types.Resource(
            name=f"{tool_name} guidance",
            uri=f"tool://guidance/{tool_name}",
            description="Runtime guidance, defaults, and request schema",
            mime_type="application/json",
        )
        for tool_name in TOOL_SPECS
    ]
    resources.extend(
        types.Resource(
            name=f"{tool_name} recipes",
            uri=f"tool://recipes/{tool_name}",
            description="Public API v2 workflow recipes",
            mime_type="application/json",
        )
        for tool_name in _PUBLIC_RECIPES
    )
    return resources


async def read_resource(uri: str) -> list[types.TextResourceContents]:
    uri_str = str(uri)
    if uri_str.startswith("tool://recipes/"):
        tool = uri_str.removeprefix("tool://recipes/")
        recipes = _PUBLIC_RECIPES.get(tool)
        if recipes is None:
            raise MCPError(types.INVALID_PARAMS, f"Unknown resource URI: {uri_str}")
        payload = json.dumps({"tool": tool, "recipes": recipes}, indent=2)
        return [types.TextResourceContents(uri=uri_str, text=payload, mime_type="application/json")]

    if uri_str.startswith("tool://guidance/"):
        tool = uri_str.removeprefix("tool://guidance/")
        spec = TOOL_SPECS.get(tool)
        if spec is None:
            raise MCPError(types.INVALID_PARAMS, f"Unknown resource URI: {uri_str}")
        payload = json.dumps(
            {
                "tool": tool,
                "description": spec.description,
                "runtime_hint": spec.metadata.get("runtime_hint"),
                "io_hint": spec.metadata.get("io_hint"),
                "defaults": {
                    "threads": spec.metadata.get("default_threads"),
                    "timeout_seconds": spec.metadata.get("timeout_seconds"),
                },
                "config_scope": _CONFIG_SCOPE_NOTE,
                "when_to_use": spec.metadata.get("when_to_use"),
                "request_schema": spec.schema,
                "recipes_uri": f"tool://recipes/{tool}" if tool in _PUBLIC_RECIPES else None,
            },
            indent=2,
        )
        return [types.TextResourceContents(uri=uri_str, text=payload, mime_type="application/json")]

    raise MCPError(types.INVALID_PARAMS, f"Unknown resource URI: {uri_str}")


async def _on_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(tools=await list_tools())


async def _on_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
    spec = TOOL_SPECS.get(params.name)
    if spec is not None and params.name not in _REQUEST_MODELS:
        try:
            jsonschema.validate(instance=params.arguments or {}, schema=spec.schema)
        except jsonschema.ValidationError as error:
            return _validation_result(params.name, ValueError(error.message))
    return await dispatch_tool(params.name, params.arguments)


async def _on_list_resources(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListResourcesResult:
    return types.ListResourcesResult(resources=await list_resources())


async def _on_list_resource_templates(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListResourceTemplatesResult:
    return types.ListResourceTemplatesResult(resource_templates=await list_resource_templates())


async def _on_read_resource(
    ctx: ServerRequestContext, params: types.ReadResourceRequestParams
) -> types.ReadResourceResult:
    return types.ReadResourceResult(contents=[*await read_resource(params.uri)])


server = Server(
    "ont-qc-mcp",
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
    on_list_resources=_on_list_resources,
    on_list_resource_templates=_on_list_resource_templates,
    on_read_resource=_on_read_resource,
)


async def _async_main() -> None:
    asyncio.get_running_loop().set_default_executor(get_executor())
    if _use_compat_stdio():
        async with stdio_server_compat() as (read, write):
            await server.run(read, write, server.create_initialization_options(), raise_exceptions=True)
    else:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options(), raise_exceptions=True)


def main() -> None:
    anyio.run(_async_main)


if __name__ == "__main__":
    main()
