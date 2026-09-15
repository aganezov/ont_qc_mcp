"""Unregistered API v2 variant-QC backend built on bcftools stats."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import ExecutionConfig, ToolPaths
from .parsers import parse_vcf_header
from .regional_metrics import RegionalInterval
from .v2_contracts import (
    Provenance,
    VariantEffectiveRequest,
    VariantGeneralSection,
    VariantIndelSection,
    VariantQCGroup,
    VariantQCRequest,
    VariantQCResponse,
    VariantSnpSection,
)
from .v2_execution import PipelineStage, RequestDeadline, run_pipeline
from .v2_native_args import ValidatedNativeArgs, validate_native_args
from .v2_regions import NormalizedRegionSet, normalize_regions
from .v2_samtools import FileIdentity, companion_index, file_identity, local_file


MAX_VARIANT_HEADER_BYTES = 16 * 1024 * 1024
MAX_BCFTOOLS_STATS_BYTES = 16 * 1024 * 1024
VARIANT_SUFFIXES = (".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")


@dataclass(frozen=True)
class ResolvedVariantInput:
    """Resolved caller files and the lexical paths that retain adjacent indexes."""

    path: Path
    access_path: Path
    index: Path | None
    reference: Path | None
    reference_index: Path | None
    reference_access_path: Path | None
    tracked_paths: tuple[Path, ...]
    identities: tuple[FileIdentity, ...]

    def assert_unchanged(self) -> None:
        if self.identities != tuple(file_identity(path) for path in self.tracked_paths):
            raise RuntimeError("A variant input, index or reference changed during analysis; retry with stable files")


def resolve_variant_input(
    path: str,
    *,
    reference_path: str | None,
    require_index: bool,
    exec_cfg: ExecutionConfig | None = None,
) -> ResolvedVariantInput:
    """Resolve VCF/BCF, optional index, and optional indexed FASTA without creating files."""
    cfg = exec_cfg or ExecutionConfig()
    supplied = Path(path)
    variant = local_file(path, cfg, VARIANT_SUFFIXES)
    lowered = variant.name.lower()
    access_path = variant
    index = None
    if require_index:
        if lowered.endswith(".vcf"):
            raise ValueError("Regional variant selection requires indexed VCF.gz, VCF.bgz, or BCF input")
        suffixes = (".csi",) if lowered.endswith(".bcf") else (".csi", ".tbi")
        index, access_path = companion_index(supplied, variant, cfg, suffixes, replace_suffix=False)

    reference = reference_index = reference_access_path = None
    if reference_path is not None:
        reference = local_file(reference_path, cfg, (".fa", ".fasta", ".fna"))
        with reference.open("rb") as stream:
            if stream.read(1) != b">":
                raise ValueError("Reference must be an uncompressed FASTA beginning with '>'")
        reference_index, reference_access_path = companion_index(
            Path(reference_path), reference, cfg, (".fai",), replace_suffix=False
        )

    tracked = [variant]
    if access_path != variant:
        tracked.append(access_path)
    if index is not None:
        tracked.append(index)
    if reference is not None and reference_index is not None:
        tracked.extend((reference, reference_index))
    if reference_access_path is not None:
        tracked.extend((reference_access_path, Path(str(reference_access_path) + ".fai")))
    tracked_paths = tuple(dict.fromkeys(tracked))
    return ResolvedVariantInput(
        path=variant,
        access_path=access_path,
        index=index,
        reference=reference,
        reference_index=reference_index,
        reference_access_path=reference_access_path,
        tracked_paths=tracked_paths,
        identities=tuple(file_identity(item) for item in tracked_paths),
    )


def _threads(cfg: ExecutionConfig) -> tuple[str, ...]:
    threads = cfg.threads_for("bcftools")
    if threads is None:
        return ()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError("bcftools threads must be a nonnegative integer or unset")
    return ("--threads", str(threads))


def _fasta_lengths(path: Path, fasta: Path, deadline: RequestDeadline) -> dict[str, int]:
    lengths: dict[str, int] = {}
    total_bytes = 0
    fasta_size = fasta.stat().st_size
    with path.open(encoding="utf-8") as stream:
        while line := stream.readline(65537):
            deadline.checkpoint()
            total_bytes += len(line.encode("utf-8"))
            if len(line) > 65536 or total_bytes > MAX_VARIANT_HEADER_BYTES:
                raise ValueError("FASTA index exceeds supported line/header limits")
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5 or not all(value.isascii() and value.isdecimal() for value in fields[1:]):
                raise ValueError("Malformed FASTA index")
            length, offset, line_bases, line_width = map(int, fields[1:])
            if (
                not fields[0]
                or length <= 0
                or line_bases <= 0
                or line_width < line_bases
                or offset >= fasta_size
                or fields[0] in lengths
            ):
                raise ValueError("Malformed FASTA index values")
            RegionalInterval(chrom=fields[0], start=0, end=1)
            lengths[fields[0]] = length
    if not lengths:
        raise ValueError("FASTA index contains no usable reference lengths")
    return lengths


def read_variant_reference_lengths(
    variant: ResolvedVariantInput,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> dict[str, int]:
    """Read VCF/BCF contig lengths under the request deadline and verify an optional FASTA."""
    del exec_cfg  # The shared request deadline already owns this bounded header read.
    command = (
        tools.bcftools,
        "view",
        "--header-only",
        "--no-version",
        str(variant.access_path),
    )
    result = run_pipeline(
        [PipelineStage("bcftools_header", command)],
        deadline,
        max_output_bytes=MAX_VARIANT_HEADER_BYTES,
    )
    header = parse_vcf_header(result.final.stdout, file_path=str(variant.path))
    fasta_lengths = (
        _fasta_lengths(variant.reference_index, variant.reference, deadline)
        if variant.reference is not None and variant.reference_index is not None
        else {}
    )
    lengths: dict[str, int] = {}
    for reference in header.references:
        deadline.checkpoint()
        if reference.name in lengths:
            raise ValueError(f"Variant header repeats reference {reference.name!r}")
        RegionalInterval(chrom=reference.name, start=0, end=1)
        length = reference.length
        if length is None:
            length = fasta_lengths.get(reference.name)
        if length is None or length <= 0:
            raise ValueError(f"Variant header has no positive length for contig {reference.name!r}")
        if fasta_lengths and fasta_lengths.get(reference.name) != length:
            raise ValueError(f"Reference FASTA index length/name does not match variant contig {reference.name!r}")
        lengths[reference.name] = length
    if not lengths:
        raise ValueError("Variant header contains no usable contig reference lengths")
    variant.assert_unchanged()
    return lengths


def _nonnegative_integer(value: str, *, field: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"bcftools stats reported an invalid {field}: {value!r}")
    return int(value)


def _parse_bcftools_stats(stdout: str, metrics: set[str]) -> VariantQCGroup:
    """Parse the aggregate bcftools set while preserving unavailable TS/TV counts."""
    summary: dict[str, int] = {}
    transition_counts: tuple[int, int] | None = None
    keys = {
        "number of records": "records",
        "number of snps": "snps",
        "number of mnps": "mnps",
        "number of indels": "indels",
        "number of others": "others",
    }
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields[0] == "SN" and len(fields) >= 4:
            key = fields[2].strip().removesuffix(":").lower()
            if key in keys:
                summary[keys[key]] = _nonnegative_integer(fields[3].strip(), field=key)
        elif fields[0] == "TSTV":
            if len(fields) < 4:
                raise ValueError("bcftools stats emitted a malformed TSTV row")
            counts = (
                _nonnegative_integer(fields[2], field="transition count"),
                _nonnegative_integer(fields[3], field="transversion count"),
            )
            if transition_counts is not None and transition_counts != counts:
                raise ValueError("bcftools stats emitted multiple incompatible TSTV aggregate rows")
            transition_counts = counts

    required: set[str] = set()
    if "general" in metrics:
        required.update(("records", "mnps", "others"))
    if "snps" in metrics:
        required.add("snps")
    if "indels" in metrics:
        required.add("indels")
    missing = sorted(required - summary.keys())
    if missing:
        raise ValueError(f"bcftools stats output omitted requested summary fields: {', '.join(missing)}")

    general = (
        VariantGeneralSection(
            total_records=summary["records"],
            mnps=summary["mnps"],
            others=summary["others"],
        )
        if "general" in metrics
        else None
    )
    snps = None
    if "snps" in metrics:
        transitions = transversions = ratio = None
        if transition_counts is not None:
            transitions, transversions = transition_counts
            ratio = transitions / transversions if transversions else None
        snps = VariantSnpSection(
            count=summary["snps"],
            transitions=transitions,
            transversions=transversions,
            ts_tv_ratio=ratio,
        )
    indels = VariantIndelSection(count=summary["indels"]) if "indels" in metrics else None
    return VariantQCGroup(general=general, snps=snps, indels=indels)


def _native(request: VariantQCRequest) -> ValidatedNativeArgs:
    return validate_native_args(
        "bcftools_stats",
        request.extra_args.bcftools_stats,
        additionally_protected=("--split-by-ID", "-I"),
    )


def _selection_args(request: VariantQCRequest) -> list[str]:
    if request.selection.include_expression is not None:
        return ["--include", request.selection.include_expression]
    if request.selection.exclude_expression is not None:
        return ["--exclude", request.selection.exclude_expression]
    return []


def _write_regions(path: Path, regions: Iterable[RegionalInterval]) -> None:
    rows = list(regions)
    if any(region.chrom.startswith("#") for region in rows):
        raise ValueError("Contig names beginning with '#' are unsupported by bcftools BED selection")
    path.write_text("".join(f"{region.chrom}\t{region.start}\t{region.end}\n" for region in rows), encoding="utf-8")


def _run_stats(
    variant: ResolvedVariantInput,
    regions: Sequence[RegionalInterval],
    request: VariantQCRequest,
    tools: ToolPaths,
    cfg: ExecutionConfig,
    deadline: RequestDeadline,
    native: ValidatedNativeArgs,
    *,
    scope: str,
) -> tuple[VariantQCGroup, Provenance]:
    with tempfile.TemporaryDirectory(prefix="ont-qc-v2-variant-") as directory:
        region_args: list[str] = []
        if regions:
            if variant.index is None:
                raise FileNotFoundError("Regional bcftools selection requires an explicit existing variant index")
            bed = Path(directory) / "regions.bed"
            _write_regions(bed, regions)
            region_args = ["--regions-file", str(bed), "--regions-overlap", "1"]
        reference_args = (
            ["--fasta-ref", str(variant.reference_access_path)] if variant.reference_access_path is not None else []
        )
        command = (
            tools.bcftools,
            "stats",
            *_threads(cfg),
            *reference_args,
            *_selection_args(request),
            *region_args,
            *native.supplied_args,
            str(variant.access_path),
        )
        result = run_pipeline(
            [PipelineStage("bcftools_stats", command)],
            deadline,
            max_output_bytes=MAX_BCFTOOLS_STATS_BYTES,
        )
        group = _parse_bcftools_stats(result.final.stdout, set(request.metrics))
        variant.assert_unchanged()
        return group, Provenance(
            backend="bcftools_stats",
            effective_args=list(command[1:-1]),
            native_options_used=native.native_options_used,
            measurement_scope=scope,
        )


def variant_qc(
    request: VariantQCRequest | dict[str, object],
    *,
    tools: ToolPaths | None = None,
    exec_cfg: ExecutionConfig | None = None,
) -> VariantQCResponse:
    """Execute one validated variant-QC request without registering the v2 catalog."""
    validated = request if isinstance(request, VariantQCRequest) else VariantQCRequest.model_validate(request)
    cfg = exec_cfg or ExecutionConfig()
    tool_paths = tools or ToolPaths()
    deadline = RequestDeadline(validated.deadline_seconds or cfg.timeout_for("bcftools"))
    native = _native(validated)
    variant = resolve_variant_input(
        validated.path,
        reference_path=validated.reference_path,
        require_index=validated.regions is not None,
        exec_cfg=cfg,
    )

    if validated.regions is not None or variant.reference is not None:
        lengths = read_variant_reference_lengths(variant, tool_paths, cfg, deadline)
    if validated.regions is None:
        regions = NormalizedRegionSet((), ())
    else:
        regions = normalize_regions(validated.regions, lengths, exec_cfg=cfg, deadline=deadline)
    dependency_identities = tuple(file_identity(path) for path in regions.external_dependencies)
    variant.assert_unchanged()

    groups: list[VariantQCGroup] = []
    provenance: list[Provenance] = []
    if validated.group_by == "combined":
        group, source = _run_stats(
            variant,
            regions.union,
            validated,
            tool_paths,
            cfg,
            deadline,
            native,
            scope=(
                "unique records overlapping the normalized interval union"
                if regions.requested
                else "whole VCF/BCF record population"
            ),
        )
        groups.append(group)
        provenance.append(source)
    else:
        for region in regions.requested:
            group, source = _run_stats(
                variant,
                (RegionalInterval(chrom=region.chrom, start=region.start, end=region.end, name=region.name),),
                validated,
                tool_paths,
                cfg,
                deadline,
                native,
                scope="records overlapping one requested interval",
            )
            groups.append(group.model_copy(update={"region_id": region.region_id, "region_name": region.name}))
            provenance.append(source)

    if dependency_identities != tuple(file_identity(path) for path in regions.external_dependencies):
        raise RuntimeError("A BED or GFF3 region source changed during variant analysis; retry with stable files")

    effective = VariantEffectiveRequest(
        path=str(variant.path),
        reference_path=str(variant.reference) if variant.reference is not None else None,
        region_scope="normalized_intervals" if regions.requested else "whole_file",
        normalized_regions=list(regions.requested),
        resolved_group_by=validated.group_by,
        metrics=validated.metrics,
        selection=validated.selection,
        extra_args=validated.extra_args,
    )
    return VariantQCResponse(
        resolved_group_by=validated.group_by,
        effective_request=effective,
        results=groups,
        provenance=provenance,
    )


__all__ = [
    "ResolvedVariantInput",
    "read_variant_reference_lengths",
    "resolve_variant_input",
    "variant_qc",
]
