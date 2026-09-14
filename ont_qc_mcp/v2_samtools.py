"""Shared samtools input, selection, and FASTQ-conversion boundaries."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator, Sequence

from .config import ExecutionConfig, ToolPaths
from .line_stream import run_line_stream
from .regional_metrics import RegionalInterval
from .tools import _validate_input_file
from .v2_execution import PipelineStage, RequestDeadline
from .v2_native_args import ValidatedNativeArgs, validate_native_args
from .v2_regions import NormalizedRegionSet, normalize_regions


MAX_HEADER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    identity_method: str = "path_size_mtime_device_inode"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
            "identity_method": self.identity_method,
        }


def file_identity(path: Path) -> FileIdentity:
    stat = path.stat()
    return FileIdentity(
        path=str(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def local_file(
    value: str,
    cfg: ExecutionConfig,
    extensions: tuple[str, ...] | None = None,
    *,
    validator: Callable[..., None] = _validate_input_file,
) -> Path:
    if not isinstance(value, str) or not value or "##idx##" in value:
        raise ValueError("Expected a local file path without HTSlib index-routing syntax")
    path = Path(value).resolve()
    if "##idx##" in str(path):
        raise ValueError("Resolved local paths must not contain HTSlib index-routing syntax")
    validator(path, cfg, allowed_exts=extensions)
    return path


def companion_index(
    supplied: Path,
    resolved: Path,
    cfg: ExecutionConfig,
    suffixes: tuple[str, ...],
    *,
    replace_suffix: bool,
) -> tuple[Path, Path]:
    """Resolve an existing index and the lexical path that keeps it adjacent."""
    for base in dict.fromkeys((supplied.absolute(), resolved)):
        for suffix in suffixes:
            candidates = [Path(str(base) + suffix)]
            if replace_suffix:
                candidates.append(base.with_suffix(suffix))
            for candidate in candidates:
                if candidate.exists():
                    return local_file(str(candidate), cfg), base
    raise FileNotFoundError(f"An existing {suffixes} index is required for {supplied}; no index is created")


@dataclass(frozen=True)
class ResolvedAlignmentInput:
    alignment: Path
    alignment_access_path: Path
    index: Path | None
    reference: Path | None
    reference_index: Path | None
    reference_access_path: Path | None
    tracked_paths: tuple[Path, ...]
    identities: tuple[FileIdentity, ...]

    def assert_unchanged(self) -> None:
        if self.identities != tuple(file_identity(path) for path in self.tracked_paths):
            raise RuntimeError("An input, index or reference changed during analysis; retry with stable files")


def resolve_alignment_input(
    path: str,
    *,
    reference_path: str | None = None,
    require_index: bool,
    exec_cfg: ExecutionConfig | None = None,
) -> ResolvedAlignmentInput:
    """Resolve caller files without creating indexes or references."""
    cfg = exec_cfg or ExecutionConfig()
    supplied = Path(path)
    alignment = local_file(path, cfg, (".bam", ".cram"))
    alignment_access_path = alignment
    index = None
    if require_index:
        suffixes = (".crai",) if alignment.suffix.lower() == ".cram" else (".csi", ".bai")
        index, alignment_access_path = companion_index(supplied, alignment, cfg, suffixes, replace_suffix=True)

    reference = reference_index = reference_access_path = None
    if reference_path is not None:
        reference = local_file(reference_path, cfg, (".fa", ".fasta", ".fna"))
        with reference.open("rb") as stream:
            if stream.read(1) != b">":
                raise ValueError("Reference must be an uncompressed FASTA beginning with '>'")
        reference_index, reference_access_path = companion_index(
            Path(reference_path), reference, cfg, (".fai",), replace_suffix=False
        )
    if alignment.suffix.lower() == ".cram" and reference is None:
        raise ValueError("CRAM requires an explicit local uncompressed FASTA and existing .fai index")

    tracked = [alignment]
    if alignment_access_path != alignment:
        tracked.append(alignment_access_path)
    if index is not None:
        tracked.append(index)
    if reference is not None and reference_index is not None:
        tracked.extend((reference, reference_index))
    if reference_access_path is not None:
        tracked.extend((reference_access_path, Path(str(reference_access_path) + ".fai")))
    tracked_paths = tuple(tracked)
    return ResolvedAlignmentInput(
        alignment=alignment,
        alignment_access_path=alignment_access_path,
        index=index,
        reference=reference,
        reference_index=reference_index,
        reference_access_path=reference_access_path,
        tracked_paths=tracked_paths,
        identities=tuple(file_identity(item) for item in tracked_paths),
    )


def _threads(cfg: ExecutionConfig) -> tuple[str, ...]:
    threads = cfg.threads_for("samtools")
    if threads is None:
        return ()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError("samtools threads must be a nonnegative integer or unset")
    return ("-@", str(threads))


def _reference_args(alignment: ResolvedAlignmentInput) -> tuple[str, ...]:
    if alignment.reference_access_path is None:
        return ()
    return ("-T", str(alignment.reference_access_path))


def read_alignment_reference_lengths(
    alignment: ResolvedAlignmentInput,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> dict[str, int]:
    """Read strict @SQ names and lengths within the caller's request deadline."""
    reference_lengths: dict[str, int] = {}
    header_bytes = 0

    def consume(line: str) -> None:
        nonlocal header_bytes
        header_bytes += len(line.encode("utf-8")) + 1
        if header_bytes > MAX_HEADER_BYTES:
            raise ValueError("Alignment header exceeded the 16 MiB supported limit")
        if not line.startswith("@SQ\t"):
            return
        fields = dict(field.split(":", 1) for field in line.split("\t")[1:] if ":" in field)
        name, raw_length = fields.get("SN", ""), fields.get("LN", "")
        if not name or not raw_length.isascii() or not raw_length.isdecimal() or int(raw_length) <= 0:
            raise ValueError("Alignment header has an invalid @SQ name or length")
        if name in reference_lengths:
            raise ValueError(f"Alignment header repeats reference {name!r}")
        RegionalInterval(chrom=name, start=0, end=1)
        reference_lengths[name] = int(raw_length)

    command = (
        tools.samtools,
        "view",
        "-H",
        "--no-PG",
        *_threads(exec_cfg),
        *_reference_args(alignment),
        str(alignment.alignment),
    )
    environment = dict(os.environ, REF_PATH=os.devnull, REF_CACHE=os.devnull)
    run_line_stream(command, consume, timeout=deadline.remaining(), env=environment)
    deadline.checkpoint()
    if not reference_lengths:
        raise ValueError("Alignment header contains no usable @SQ reference lengths")
    if alignment.reference is not None and alignment.reference_index is not None:
        fasta_lengths: dict[str, int] = {}
        total_bytes = 0
        with alignment.reference_index.open(encoding="utf-8") as stream:
            while line := stream.readline(65537):
                deadline.checkpoint()
                total_bytes += len(line.encode("utf-8"))
                if len(line) > 65536 or total_bytes > MAX_HEADER_BYTES:
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
                    or offset >= alignment.reference.stat().st_size
                ):
                    raise ValueError("Malformed FASTA index values")
                if fields[0] in fasta_lengths:
                    raise ValueError("FASTA index repeats a reference")
                fasta_lengths[fields[0]] = length
        for name, length in reference_lengths.items():
            if fasta_lengths.get(name) != length:
                raise ValueError(f"Reference FASTA index length/name does not match alignment contig {name!r}")
    alignment.assert_unchanged()
    return reference_lengths


def resolve_and_normalize_regions(
    path: str,
    source: object | None,
    *,
    reference_path: str | None,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig,
    deadline: RequestDeadline,
) -> tuple[ResolvedAlignmentInput, NormalizedRegionSet]:
    """Resolve alignment prerequisites and normalize one request's regions."""
    alignment = resolve_alignment_input(
        path,
        reference_path=reference_path,
        require_index=source is not None,
        exec_cfg=exec_cfg,
    )
    if source is None:
        deadline.checkpoint()
        alignment.assert_unchanged()
        return alignment, normalize_regions(None, {}, exec_cfg=exec_cfg)
    lengths = read_alignment_reference_lengths(alignment, tools, exec_cfg, deadline)
    regions = normalize_regions(source, lengths, exec_cfg=exec_cfg, deadline=deadline)
    alignment.assert_unchanged()
    return alignment, regions


@dataclass(frozen=True)
class SamtoolsSelection:
    min_mapq: int = 0
    exclude_flags: int = 0
    include_flags: int | None = None
    include_unmapped: bool = False
    primary_only: bool = False

    def __post_init__(self) -> None:
        for name, value in (("min_mapq", self.min_mapq), ("exclude_flags", self.exclude_flags)):
            maximum = 254 if name == "min_mapq" else 65535
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [0, {maximum}]")
        if self.include_flags is not None and (
            isinstance(self.include_flags, bool)
            or not isinstance(self.include_flags, int)
            or not 0 <= self.include_flags <= 65535
        ):
            raise ValueError("include_flags must be an integer in [0, 65535] or unset")
        if type(self.include_unmapped) is not bool or type(self.primary_only) is not bool:
            raise ValueError("include_unmapped and primary_only must be booleans")
        if self.include_unmapped and self.exclude_flags & 0x4:
            raise ValueError("include_unmapped=True conflicts with exclude_flags bit 0x4")

    @property
    def effective_exclude_flags(self) -> int:
        flags = self.exclude_flags | (0x900 if self.primary_only else 0)
        if not self.include_unmapped:
            flags |= 0x4
        return flags


@dataclass(frozen=True)
class SamtoolsViewPlan:
    stage: PipelineStage
    native_args: ValidatedNativeArgs
    effective_exclude_flags: int
    regional: bool
    required_tags_preserved: bool
    output_format: str
    input: ResolvedAlignmentInput


@contextmanager
def samtools_view_plan(
    alignment: ResolvedAlignmentInput,
    regions: NormalizedRegionSet,
    selection: SamtoolsSelection,
    *,
    tools: ToolPaths,
    exec_cfg: ExecutionConfig | None = None,
    native_args: Sequence[str] = (),
) -> Iterator[SamtoolsViewPlan]:
    """Build one uncompressed-BAM selection stage and own any union BED file."""
    cfg = exec_cfg or ExecutionConfig()
    dynamic_protection = ("-e", "--expr") if selection.min_mapq > 0 else ()
    native = validate_native_args(
        "samtools_view",
        native_args,
        additionally_protected=dynamic_protection,
    )
    selection_args: list[str] = ["-F", str(selection.effective_exclude_flags), "-q", str(selection.min_mapq)]
    if selection.min_mapq > 0:
        selection_args.extend(("-e", "mapq != 255"))
    if selection.include_flags is not None:
        selection_args.extend(("-f", str(selection.include_flags)))

    with tempfile.TemporaryDirectory(prefix="ont-qc-v2-regions-") as directory:
        regional_args: list[str] = []
        if regions.requested:
            if alignment.index is None:
                raise FileNotFoundError("Regional samtools selection requires an explicit existing alignment index")
            if any(region.chrom.startswith("#") for region in regions.union):
                raise ValueError("Contig names beginning with '#' are unsupported by indexed BED selection")
            bed = Path(directory) / "union.bed"
            bed.write_text(regions.bed_text(), encoding="utf-8")
            regional_args = ["-M", "-L", str(bed), "-X"]

        command = [
            tools.samtools,
            "view",
            "-u",
            "-h",
            "--no-PG",
            *_threads(cfg),
            *_reference_args(alignment),
            *selection_args,
            *native.supplied_args,
            *regional_args,
            str(alignment.alignment),
        ]
        if regions.requested:
            command.append(str(alignment.index))
        yield SamtoolsViewPlan(
            stage=PipelineStage("samtools_view", tuple(command)),
            native_args=ValidatedNativeArgs(
                namespace=native.namespace,
                supplied_args=native.supplied_args,
                effective_args=tuple(command[2:]),
                native_options_used=native.native_options_used,
                reuse_safe=native.reuse_safe and not regions.external_dependencies,
            ),
            effective_exclude_flags=selection.effective_exclude_flags,
            regional=bool(regions.requested),
            required_tags_preserved=True,
            output_format="uncompressed_bam",
            input=alignment,
        )


def build_fastq_stage(
    tools: ToolPaths,
    *,
    exec_cfg: ExecutionConfig | None = None,
    native_args: Sequence[str] = (),
) -> PipelineStage:
    """Build conversion from selected BAM stdin without adding record filters."""
    return samtools_fastq_plan(tools, exec_cfg=exec_cfg, native_args=native_args).stage


@dataclass(frozen=True)
class SamtoolsFastqPlan:
    stage: PipelineStage
    native_args: ValidatedNativeArgs
    effective_exclude_flags: int = 0


def samtools_fastq_plan(
    tools: ToolPaths,
    *,
    exec_cfg: ExecutionConfig | None = None,
    native_args: Sequence[str] = (),
) -> SamtoolsFastqPlan:
    """Plan conversion and expose the exact effective arguments for provenance."""
    cfg = exec_cfg or ExecutionConfig()
    native = validate_native_args("samtools_fastq", native_args)
    command = (tools.samtools, "fastq", *_threads(cfg), "-F", "0", *native.supplied_args, "-")
    return SamtoolsFastqPlan(
        stage=PipelineStage("samtools_fastq", command),
        native_args=ValidatedNativeArgs(
            namespace=native.namespace,
            supplied_args=native.supplied_args,
            effective_args=command[2:],
            native_options_used=native.native_options_used,
            reuse_safe=native.reuse_safe,
        ),
    )


__all__ = [
    "FileIdentity",
    "MAX_HEADER_BYTES",
    "ResolvedAlignmentInput",
    "SamtoolsFastqPlan",
    "SamtoolsSelection",
    "SamtoolsViewPlan",
    "build_fastq_stage",
    "companion_index",
    "file_identity",
    "local_file",
    "read_alignment_reference_lengths",
    "resolve_alignment_input",
    "resolve_and_normalize_regions",
    "samtools_view_plan",
    "samtools_fastq_plan",
]
