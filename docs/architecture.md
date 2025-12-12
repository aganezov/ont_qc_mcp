# Architecture Overview

## Components
- **MCP server (`app_server.py`)**: Registers MCP tools/resources and dispatches to wrapper functions on worker threads. Optional concurrency guard via `MCP_MAX_CONCURRENCY`.
- **CLI wrappers (`ont_qc_mcp/cli_wrappers.py`)**: Build validated command lines, handle fallbacks, and parse outputs into models.
- **Tool orchestrator (`ont_qc_mcp/tools.py`)**: User-facing helpers (caching, validation, composition like `alignment_summary`).
- **Parsers/Schemas**: Defensive JSON/text parsing into Pydantic models with explicit `None` vs empty semantics.
- **Utils**: Subprocess execution (`run_command_with_retry`) with bounded capture and structured errors.

## Execution flow
1. MCP request hits `dispatch_tool` -> validated against schema -> dispatched.
2. Wrapper builds CLI args, applies timeouts/threads from `ExecutionConfig`, and runs through `run_command_with_retry`.
3. Outputs are parsed into Pydantic models and serialized; provenance is included (verbose when `MCP_INCLUDE_PROVENANCE=1`).
4. Caching: nanoq stats are cached with inflight deduplication; cache size is bounded.

## Error handling
- Failures raise `CommandError` with truncated stderr; `_error_result` returns structured JSON `{kind, message, tool, details}`.
- Streaming pipelines (samtools -> nanoq) drain stderr concurrently and tear down both processes on timeout.

## Resource limits
- `MCP_MAX_FILE_MB` enforces optional file-size limits.
- `MCP_MAX_CONCURRENCY` limits concurrent MCP calls when set.

