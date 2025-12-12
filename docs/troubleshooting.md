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
- Upgrade the CLI or re-run with the latest flags.

## Large inputs
- Set `MCP_MAX_FILE_MB` to guard huge inputs.
- Use BAM/CRAM streaming tools to avoid temp FASTQ.

## Errors from MCP calls
- Errors are returned as JSON `{kind, message, tool, details}`.
- Check stderr snippets in the message; enable verbose provenance with `MCP_INCLUDE_PROVENANCE=1`.

