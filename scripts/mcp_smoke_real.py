#!/usr/bin/env python3
"""
Run a small set of MCP tool calls against real input files.

Examples:
  scripts/with-env.sh python scripts/mcp_smoke_real.py --dir /path/to/test_dir
  scripts/with-env.sh python scripts/mcp_smoke_real.py --fastq reads.fq.gz --bam aln.bam --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] | None = None


def _server_env() -> dict[str, str]:
    """
    Pass-through runtime configuration to the subprocess server.

    The upstream MCP stdio client intentionally inherits only a small set of
    "safe" environment variables by default. We explicitly forward `MCP_*`
    config and common tool overrides so local smoke runs match your shell env.
    """
    passthrough: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("MCP_"):
            passthrough[key] = value
        if key in {"NANOQ", "CHOPPER", "CRAMINO", "MOSDEPTH", "SAMTOOLS", "BCFTOOLS"}:
            passthrough[key] = value
    return passthrough


def _first_match(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _extract_text_payload(result) -> str:
    content = getattr(result, "content", None) or []
    texts: list[str] = []
    for entry in content:
        text = getattr(entry, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def _parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def _tool_error_payload(result) -> dict[str, Any]:
    return {"error": True, "message": _extract_text_payload(result)}


async def _call_tools(server_params: StdioServerParameters, calls: list[ToolCall]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for call in calls:
                res = await session.call_tool(call.name, call.arguments)
                if getattr(res, "isError", False):
                    outputs[call.name] = _tool_error_payload(res)
                    continue

                text = _extract_text_payload(res)
                outputs[call.name] = _parse_json_payload(text)
    return outputs


def _build_calls(
    fastq: Path | None,
    bam: Path | None,
    include_error_profile: bool,
    run_summary: bool,
) -> list[ToolCall]:
    calls: list[ToolCall] = [ToolCall("env_status")]

    if fastq:
        calls.extend(
            [
                ToolCall("qc_reads_fastq_tool", {"path": str(fastq)}),
                ToolCall("read_length_distribution_fastq_tool", {"path": str(fastq)}),
                ToolCall("qscore_distribution_fastq_tool", {"path": str(fastq)}),
            ]
        )

    if bam:
        calls.append(ToolCall("qc_alignment_tool", {"path": str(bam)}))
        calls.append(ToolCall("coverage_stats_tool", {"path": str(bam)}))
        if include_error_profile:
            calls.append(ToolCall("alignment_error_profile_tool", {"path": str(bam)}))
        if run_summary:
            calls.append(
                ToolCall(
                    "alignment_summary_tool",
                    {"path": str(bam), "include_error_profile": include_error_profile},
                )
            )

    return calls


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, help="Directory to scan for .fq/.fastq and .bam/.cram inputs")
    parser.add_argument("--fastq", type=Path, help="FASTQ(.gz/.bgz) path to use (overrides --dir scan)")
    parser.add_argument("--bam", type=Path, help="BAM/CRAM/SAM path to use (overrides --dir scan)")
    parser.add_argument("--out", type=Path, help="Write JSON output to this file (default: stdout)")
    parser.add_argument(
        "--include-error-profile",
        action="store_true",
        help="Also call alignment_error_profile_tool and include_error_profile in alignment_summary_tool",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip alignment_summary_tool (still runs qc_alignment_tool and coverage_stats_tool when --bam is present)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    directory: Path | None = args.dir
    fastq: Path | None = args.fastq
    bam: Path | None = args.bam

    if directory:
        directory = directory.resolve()
        if not directory.exists():
            raise SystemExit(f"--dir does not exist: {directory}")
        if not directory.is_dir():
            raise SystemExit(f"--dir is not a directory: {directory}")

    if fastq is None and directory is not None:
        fastq = _first_match(directory, ["*.fastq", "*.fq", "*.fastq.gz", "*.fq.gz", "*.fastq.bgz", "*.fq.bgz"])
    if bam is None and directory is not None:
        bam = _first_match(directory, ["*.bam", "*.cram", "*.sam"])

    if fastq is None and bam is None:
        raise SystemExit("Provide at least one of --fastq/--bam or a --dir containing matching files")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ont_qc_mcp.app_server"],
        cwd=str(REPO_ROOT),
        env=_server_env(),
    )
    calls = _build_calls(
        fastq=fastq,
        bam=bam,
        include_error_profile=bool(args.include_error_profile),
        run_summary=not bool(args.no_summary),
    )
    outputs = anyio.run(_call_tools, server_params, calls)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        return 0

    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
