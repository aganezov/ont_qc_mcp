"""Indexed regional evidence with explicit alignment-record denominators."""

from __future__ import annotations

import os
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .config import ExecutionConfig, ToolPaths
from .cli_wrappers import build_cli_args
from .line_stream import run_line_stream
from .process_control import check_cancelled
from .regional_metrics import RegionalAccumulator, RegionalInterval
from .tools import _validate_input_file
from .utils import CommandError, CommandResult, report_progress
from .v2_samtools import companion_index, file_identity, local_file

MAX_REGIONS = 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_SAM_LINE_BYTES = 16 * 1024 * 1024


def regional_alignment_stats(
    path: str,
    regions: list[dict[str, Any]],
    reference_path: str | None = None,
    exclude_flags: int = 1796,
    min_mapq: int = 0,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Summarize retained alignment records and aligned base qualities per interval.

    A successful response always represents the complete indexed scan. Missing
    qualities are excluded from means and counted separately. No result is returned
    after a command, parsing, timeout, cancellation or input-change failure.
    """
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError(f"regions must contain 1 to {MAX_REGIONS} explicit intervals")
    intervals = [RegionalInterval.model_validate(region) for region in regions]
    accumulator = RegionalAccumulator(intervals, exclude_flags=exclude_flags, min_mapq=min_mapq)
    if any(region.chrom.startswith("#") for region in intervals):
        raise ValueError("Contig names beginning with '#' are unsupported by indexed BED selection")
    cfg, tools = exec_cfg or ExecutionConfig(), tools or ToolPaths()
    timeout = cfg.timeout_for("samtools")
    if timeout <= 0:
        raise ValueError("samtools timeout must be positive")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        check_cancelled()
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise CommandError(
                CommandResult([tools.samtools], 124, "", ""),
                message_override=f"Regional analysis timed out after {timeout}s",
            )
        return seconds

    bam = local_file(path, cfg, (".bam", ".cram"), validator=_validate_input_file)
    index, _alignment_access_path = companion_index(
        Path(path), bam, cfg, (".crai",) if bam.suffix.lower() == ".cram" else (".csi", ".bai"), replace_suffix=True
    )
    reference = None
    reference_access_path = None
    fai = None
    if reference_path is not None:
        reference = local_file(reference_path, cfg, (".fa", ".fasta", ".fna"), validator=_validate_input_file)
        with reference.open("rb") as reference_stream:
            if reference_stream.read(1) != b">":
                raise ValueError("Reference must be an uncompressed FASTA beginning with '>'")
        fai, reference_access_path = companion_index(
            Path(reference_path), reference, cfg, (".fai",), replace_suffix=False
        )
    if bam.suffix.lower() == ".cram" and reference is None:
        raise ValueError("CRAM requires an explicit local uncompressed FASTA and existing .fai index")
    tracked = [bam, index] + ([reference, fai] if reference is not None and fai is not None else [])
    # Reference access must preserve its adjacent index. Track both lexical
    # paths as well as resolved files so retargeted symlinks invalidate results.
    if reference_access_path is not None:
        tracked.extend([reference_access_path, Path(str(reference_access_path) + ".fai")])
    identities = [file_identity(item) for item in tracked]
    env = dict(os.environ, REF_PATH=os.devnull, REF_CACHE=os.devnull)
    threads = cfg.threads_for("samtools")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 0):
        raise ValueError("samtools threads must be a nonnegative integer or unset")
    thread_args = build_cli_args("samtools", {"threads": threads})
    # Keep the validated reference/index pair together. samtools 1.24 can
    # create a default .fai even with explicit ##idx## routing; do not use it.
    reference_args = ["-T", str(reference_access_path)] if reference_access_path is not None else []
    warnings: list[str] = []

    def command(cmd: list[str], consume: Callable[[str], None]) -> None:
        result = run_line_stream(cmd, consume, timeout=remaining(), env=env, max_line_bytes=MAX_SAM_LINE_BYTES)
        if result.stderr:
            warnings.append(result.stderr)

    report_progress(f"regional alignment stats start: {bam}")
    version_lines: list[str] = []
    version_bytes = 0

    def collect_version(line: str) -> None:
        nonlocal version_bytes
        version_bytes += len(line.encode("utf-8")) + 1
        if version_bytes > 65536:
            raise ValueError("samtools version output exceeded 64 KiB")
        version_lines.append(line)

    command([tools.samtools, "--version"], collect_version)
    if not version_lines or not version_lines[0].startswith("samtools "):
        raise ValueError("Could not identify the samtools version")
    reference_lengths: dict[str, int] = {}
    reference_md5: dict[str, str] = {}
    read_groups: list[dict[str, str | None]] = []
    header_bytes = 0

    def collect_header(line: str) -> None:
        nonlocal header_bytes
        header_bytes += len(line.encode("utf-8")) + 1
        if header_bytes > MAX_HEADER_BYTES:
            raise ValueError("Alignment header exceeded the 16 MiB supported limit")
        if not line.startswith(("@SQ\t", "@RG\t")):
            return
        fields = dict(field.split(":", 1) for field in line.split("\t")[1:] if ":" in field)
        if line.startswith("@RG\t"):
            read_groups.append({"id": fields.get("ID"), "sample": fields.get("SM")})
            return
        name, raw_length = fields.get("SN", ""), fields.get("LN", "")
        if not name or not raw_length.isascii() or not raw_length.isdecimal() or int(raw_length) <= 0:
            raise ValueError("Alignment header has an invalid @SQ name or length")
        if name in reference_lengths:
            raise ValueError(f"Alignment header repeats reference {name!r}")
        reference_lengths[name] = int(raw_length)
        if "M5" in fields:
            reference_md5[name] = fields["M5"]

    command([tools.samtools, "view", "-H", "--no-PG", *thread_args, *reference_args, str(bam)], collect_header)
    for region in intervals:
        if region.chrom not in reference_lengths:
            raise ValueError(f"Contig {region.chrom!r} is absent from the alignment header")
        if region.end > reference_lengths[region.chrom]:
            raise ValueError(f"Interval end {region.end} exceeds reference length for {region.chrom!r}")
    if reference is not None and fai is not None:
        fasta_lengths: dict[str, int] = {}
        total = 0
        with fai.open(encoding="utf-8") as stream:
            while line := stream.readline(65537):
                remaining()
                total += len(line.encode("utf-8"))
                if len(line) > 65536 or total > MAX_HEADER_BYTES:
                    raise ValueError("FASTA index exceeds supported line/header limits")
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5 or not all(value.isascii() and value.isdecimal() for value in fields[1:]):
                    raise ValueError("Malformed FASTA index")
                length, offset, line_bases, line_width = map(int, fields[1:])
                if not fields[0] or length <= 0 or line_bases <= 0 or line_width < line_bases:
                    raise ValueError("Malformed FASTA index values")
                if offset >= reference.stat().st_size:
                    raise ValueError("FASTA index offset exceeds the reference file")
                if fields[0] in fasta_lengths:
                    raise ValueError("FASTA index repeats a reference")
                fasta_lengths[fields[0]] = int(fields[1])
        for name, length in reference_lengths.items():
            if fasta_lengths.get(name) != length:
                raise ValueError(f"Reference FASTA index length/name does not match alignment contig {name!r}")

    with tempfile.TemporaryDirectory(prefix="ont-qc-regions-") as directory:
        bed = Path(directory) / "regions.bed"
        bed.write_text("".join(f"{r.chrom}\t{r.start}\t{r.end}\n" for r in intervals), encoding="utf-8")
        # -X pins the exact reported index. Native filters reduce decoded SAM output;
        # the accumulator additionally treats MAPQ255 as unavailable.
        command(
            [
                tools.samtools,
                "view",
                "--no-PG",
                "-M",
                "-L",
                str(bed),
                "-X",
                "-x",
                "^",
                "-F",
                str(exclude_flags | 4),
                "-q",
                str(min_mapq),
                *thread_args,
                *reference_args,
                str(bam),
                str(index),
            ],
            accumulator.add_sam_line,
        )
    remaining()
    if identities != [file_identity(item) for item in tracked]:
        raise RuntimeError("An input, index or reference changed during regional analysis; retry with stable files")
    selected_refs = list(dict.fromkeys(region.chrom for region in intervals))
    try:
        server_version: str | None = version("ont-qc-mcp")
    except PackageNotFoundError:
        server_version = None
    report_progress(f"regional alignment stats done: {bam}")
    return {
        "schema_version": "1.0",
        "input": identities[0].as_dict(),
        "index": identities[1].as_dict(),
        "reference": identities[2].as_dict() if reference is not None else None,
        "reference_index": identities[3].as_dict() if reference is not None else None,
        "alignment_references": [
            {"name": name, "length": reference_lengths[name], "header_md5": reference_md5.get(name)}
            for name in selected_refs
        ],
        "read_groups": read_groups,
        "coordinate_system": "0-based_half-open",
        "counting_unit": "retained_alignment_record",
        "selection": {
            "mapped_only": True,
            "exclude_flags": exclude_flags,
            "min_mapq": min_mapq,
            "unavailable_mapq": "retain_at_minimum_zero_only",
            "base_quality_filter": "none",
            "read_groups": "all",
        },
        "definitions": {
            "span_overlapping_alignments": (
                "Retained records whose reference span overlaps the interval, including D/N-only overlap"
            ),
            "aligned_base_alignments": (
                "Retained records contributing at least one M, = or X query base inside the interval"
            ),
            "mean_mapq": "Arithmetic mean of available MAPQ over retained span-overlapping records; 255 is unavailable",
            "mean_base_quality": (
                "Arithmetic mean of stored Phred QUAL for M, = or X query bases inside "
                "the interval; unavailable QUAL is excluded"
            ),
        },
        "regions": accumulator.results(),
        "execution": {
            "access_pattern": "one_indexed_multi_region_pass",
            "samtools_version": version_lines[0],
            "server_version": server_version,
            "threads": threads,
            "samtools_filters": {"exclude_flags": exclude_flags | 4, "min_mapq": min_mapq},
            "reference_access_path": str(reference_access_path) if reference_access_path is not None else None,
            "timeout_seconds": timeout,
            "max_regions": MAX_REGIONS,
            "max_sam_line_bytes": MAX_SAM_LINE_BYTES,
        },
        "complete": True,
        "warnings": [
            (
                "Counts describe alignment records, not unique biological reads; "
                "overlapping region results must not be summed as unique totals."
            ),
            "Stored quality scores describe reported confidence, not observed sequencing or mapping accuracy.",
            "Input identity uses file metadata, not a content checksum; header MD5 values are declarations.",
            *warnings,
        ],
    }
