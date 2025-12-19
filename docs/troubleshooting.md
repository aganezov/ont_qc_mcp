# Troubleshooting

## CLI not found
- Run `env_status` or `scripts/with-env.sh env` to verify PATH.
- Set tool overrides: `NANOQ`, `CHOPPER`, `CRAMINO`, `MOSDEPTH`, `SAMTOOLS`.

## Timeouts
- Increase `MCP_TIMEOUT_DEFAULT` or `MCP_TIMEOUT_<TOOL>`.
- For streaming BAM->nanoq, ensure disk/network throughput is adequate.

## Empty or missing histogram fields
- `null` means the upstream tool omitted the block (common on older nanoq versions).
- Empty list means the tool emitted an empty block.
- For nanoq v0.10.0, aux histograms are enabled by default; set `MCP_NANOQ_AUX_STATS=0` to disable the fallback.

## Large inputs
- Set `MCP_MAX_FILE_MB` to guard huge inputs.
- Use BAM/CRAM streaming tools to avoid temp FASTQ.

## Errors from MCP calls
- Errors are returned as JSON `{kind, message, tool, details}`.
- Check stderr snippets in the message; enable verbose provenance with `MCP_INCLUDE_PROVENANCE=1`.

## MCP subprocess hangs (stdio)
- If MCP handshake works but tool calls hang in a restricted environment, try:
  - `MCP_STDIO_TRANSPORT=compat`
  - `MCP_BLOCKING_MODE=sync` (or leave `auto` and let it fall back automatically)

## IGV snapshot / Apptainer issues
- To run the real IGV snapshot integration via Apptainer without implicit pulls, set `MCP_IGV_SIF_PATH=/path/to/igv_snapper.sif`.
- To preserve IGV snapshot test outputs, set `IGV_SNAPSHOT_DIR=/path/to/dir` and tests will copy snapshots there.
- If you see `Couldn't determine user account information: user: unknown userid ...`, your environment cannot resolve the current UID via NSS:
  - Check `whoami` and `getent passwd $(id -u)`.
  - Some restricted sandboxes lack `/etc/passwd` or NSS backends.
- If you see errors like `Could not write info to setgroups: Permission denied` or `cannot open /proc/self/uid_map: Permission denied`, unprivileged user namespaces are disabled on the host:
  - Use a system Apptainer install configured as setuid-root, or ask admins to enable unprivileged user namespaces.
  - Quick check: `unshare --user --map-root-user true` should succeed.
