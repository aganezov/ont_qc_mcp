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

In executor mode, cancellation prevents queued work from starting. If a worker
has already started, the server-side handler keeps the concurrency slot until
the worker finishes, then propagates cancellation even if the worker failed.
The client can receive a cancellation response before that server-side cleanup
finishes. Repeated cancellation does not release the slot early.

A cancelled worker observes an operation-specific event at subprocess and retry
checkpoints. On POSIX systems, each command starts a new session; cleanup sends
TERM and then KILL if needed to its process group, reaps direct children, and
closes pipes. Streaming stderr readers can stop without waiting for inherited
pipes to reach EOF. Descendants that create a separate session escape this group.
Cancellation does not forcibly terminate Python threads, so parsing or other
code without checkpoints retains its slot until it returns.

Docker containers belong to the daemon rather than the CLI process group. Each
IGV invocation uses a unique container name, and failed or cancelled invocations
attempt a bounded `docker rm --force` for that name. Cleanup errors are logged;
removal cannot be guaranteed if the daemon is unreachable or creation completes
after the cleanup request. Automatic IGV output and batch directories are removed
on failure or cancellation. Successful returned paths and caller-supplied files
and output directories are preserved. Arbitrary IGV batch commands are not rolled
back. Chopper checks cancellation before publishing its staged output.

Direct/synchronous mode and the automatic synchronous fallback block the event
loop, so server-side cancellation remains deferred until synchronous work returns.
