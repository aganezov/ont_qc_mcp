"""Normalize API v2 region sources to ordered zero-based half-open intervals."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

from .config import ExecutionConfig
from .parsers import is_bed_coordinate_field, is_bed_metadata_line
from .process_control import check_cancelled
from .regional_metrics import RegionalInterval
from .tools import _validate_input_file
from .v2_contracts import BedFileRegions, BedTextRegions, Gff3Regions, NormalizedInterval, SamtoolsRegions
from .v2_execution import RequestDeadline


MAX_REGIONS = 1024
MAX_REGION_LINE_CHARS = 64 * 1024


def _region_source_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _assert_region_source_unchanged(path: Path, identity: tuple[int, int, int, int]) -> None:
    if identity != _region_source_identity(path):
        raise RuntimeError("A BED or GFF3 region source changed during normalization; retry with stable files")


def _check_region_count(count: int) -> None:
    if not 1 <= count <= MAX_REGIONS:
        raise ValueError(f"regions must contain 1 to {MAX_REGIONS} intervals")


@dataclass(frozen=True)
class NormalizedRegionSet:
    """Requested rows plus the de-duplicated genomic union used for selection."""

    requested: tuple[NormalizedInterval, ...]
    union: tuple[RegionalInterval, ...]
    external_dependencies: tuple[Path, ...] = ()

    @property
    def scope(self) -> str:
        return "normalized_intervals" if self.requested else "whole"

    def per_region(self) -> tuple["NormalizedRegionSet", ...]:
        return tuple(
            NormalizedRegionSet(
                requested=(region,),
                union=(RegionalInterval(chrom=region.chrom, start=region.start, end=region.end, name=region.name),),
                external_dependencies=self.external_dependencies,
            )
            for region in self.requested
        )

    def bed_text(self, *, union: bool = True) -> str:
        intervals: Sequence[RegionalInterval] = self.union if union else self.requested
        return "".join(f"{region.chrom}\t{region.start}\t{region.end}\n" for region in intervals)


def _reference_map(reference_lengths: Mapping[str, int], deadline: RequestDeadline | None = None) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for chrom, length in reference_lengths.items():
        _checkpoint(deadline)
        RegionalInterval(chrom=chrom, start=0, end=1)
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(f"Reference length for {chrom!r} must be a positive integer")
        lengths[chrom] = length
    return lengths


def _validate_bounds(region: RegionalInterval, reference_lengths: Mapping[str, int], context: str) -> None:
    if region.chrom not in reference_lengths:
        raise ValueError(f"{context}: contig {region.chrom!r} is absent from the reference")
    length = reference_lengths[region.chrom]
    if region.end > length:
        raise ValueError(
            f"{context}: interval end {region.end} exceeds reference length {length} for contig {region.chrom!r}"
        )


def _checkpoint(deadline: RequestDeadline | None) -> None:
    if deadline is None:
        check_cancelled()
    else:
        deadline.checkpoint()


def _bounded_lines(
    stream: TextIO,
    *,
    source: str,
    deadline: RequestDeadline | None,
) -> Iterator[str]:
    while True:
        _checkpoint(deadline)
        raw_line = stream.readline(MAX_REGION_LINE_CHARS + 2)
        _checkpoint(deadline)
        if not raw_line:
            return
        content = raw_line.removesuffix("\n").removesuffix("\r")
        if len(content) > MAX_REGION_LINE_CHARS:
            raise ValueError(f"{source} record exceeds {MAX_REGION_LINE_CHARS} characters")
        yield raw_line


def _bed_intervals(
    lines: Iterable[str], *, source: str, deadline: RequestDeadline | None = None
) -> list[RegionalInterval]:
    intervals: list[RegionalInterval] = []
    for line_number, raw_line in enumerate(lines, start=1):
        _checkpoint(deadline)
        line = raw_line.strip()
        if not line or line.startswith("#") or is_bed_metadata_line(line):
            continue
        fields = line.split("\t")
        context = f"{source} line {line_number}"
        if len(fields) < 3:
            raise ValueError(f"{context}: expected at least 3 BED columns")
        if not all(is_bed_coordinate_field(value) for value in fields[1:3]):
            raise ValueError(f"{context}: coordinates must be nonnegative ASCII decimal integers")
        name = fields[3] if len(fields) >= 4 else None
        try:
            intervals.append(RegionalInterval(chrom=fields[0], start=int(fields[1]), end=int(fields[2]), name=name))
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
        if len(intervals) > MAX_REGIONS:
            _check_region_count(len(intervals))
    if not intervals:
        raise ValueError(f"{source} must contain at least one BED interval")
    return intervals


def _samtools_interval(value: str, reference_lengths: Mapping[str, int]) -> RegionalInterval:
    if value in reference_lengths:
        return RegionalInterval(chrom=value, start=0, end=reference_lengths[value])

    for chrom in sorted(reference_lengths, key=len, reverse=True):
        prefix = f"{chrom}:"
        if not value.startswith(prefix):
            continue
        coordinate = value[len(prefix) :]
        if not coordinate or coordinate == "-":
            break
        raw_start: str | None
        raw_end: str | None
        if "-" in coordinate:
            if coordinate.count("-") != 1:
                break
            start_text, end_text = coordinate.split("-", 1)
            raw_start, raw_end = start_text or None, end_text or None
        else:
            raw_start, raw_end = coordinate, None
        for raw in (raw_start, raw_end):
            if raw is not None and (not raw.replace(",", "").isascii() or not raw.replace(",", "").isdecimal()):
                raise ValueError(f"Samtools region {value!r} does not contain valid ASCII decimal coordinates")
        start = int(raw_start.replace(",", "")) if raw_start is not None else 1
        end = int(raw_end.replace(",", "")) if raw_end is not None else reference_lengths[chrom]
        if start < 1 or end < 1:
            raise ValueError(f"Samtools region {value!r} uses one-based coordinates, which must be positive")
        if start > end:
            raise ValueError(f"Samtools region {value!r} start must not exceed end")
        return RegionalInterval(chrom=chrom, start=start - 1, end=end)
    raise ValueError(f"Samtools region {value!r} does not name a known contig and valid one-based range")


def _all_gff_genes(path: Path, deadline: RequestDeadline | None = None) -> list[tuple[str, int, int, str | None]]:
    genes: list[tuple[str, int, int, str | None]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(_bounded_lines(stream, source="GFF3", deadline=deadline), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2].lower() != "gene":
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError as error:
                raise ValueError(f"GFF3 line {line_number}: gene coordinates must be integers") from error
            attrs: dict[str, str] = {}
            for entry in fields[8].split(";"):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    attrs[key.lower()] = value
            genes.append((fields[0], start, end, attrs.get("name") or attrs.get("id")))
            if len(genes) > MAX_REGIONS:
                _check_region_count(len(genes))
    if not genes:
        raise ValueError(f"No gene features found in GFF3 file {path}")
    return genes


def _selected_gff_genes(
    path: Path,
    identifiers: Sequence[str],
    deadline: RequestDeadline | None,
) -> list[tuple[str, int, int, str]]:
    _check_region_count(len(identifiers))
    for identifier in identifiers:
        if not identifier.strip():
            raise ValueError("GFF3 gene identifiers must not be blank")

    wanted = {identifier.lower() for identifier in identifiers}
    matches: dict[str, tuple[str, int, int]] = {}
    with path.open(encoding="utf-8") as stream:
        for raw_line in _bounded_lines(stream, source="GFF3", deadline=deadline):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2].lower() != "gene":
                continue
            attrs: dict[str, str] = {}
            for entry in fields[8].split(";"):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    attrs[key.lower()] = value
            keys = {value.lower() for value in (attrs.get("id"), attrs.get("name")) if value}
            matching = (keys & wanted) - matches.keys()
            if not matching:
                continue
            try:
                start, end = int(fields[3]), int(fields[4])
            except ValueError:
                continue
            for key in matching:
                matches[key] = (fields[0], start, end)
            if matches.keys() >= wanted:
                break

    coordinates: list[tuple[str, int, int, str]] = []
    for identifier in identifiers:
        match = matches.get(identifier.lower())
        if match is None:
            raise ValueError(f"Gene {identifier!r} not found in GFF3 file {path}")
        coordinates.append((*match, identifier))
    return coordinates


def _gff_intervals(
    source: Gff3Regions,
    cfg: ExecutionConfig,
    deadline: RequestDeadline | None = None,
) -> tuple[list[RegionalInterval], Path]:
    path = Path(source.path)
    _validate_input_file(path, cfg, allowed_exts=(".gff", ".gff3"))
    resolved = path.resolve()
    identity = _region_source_identity(resolved)
    coordinates: Sequence[tuple[str, int, int, str | None]]
    if source.ids is None:
        coordinates = _all_gff_genes(resolved, deadline)
    else:
        _check_region_count(len(source.ids))
        coordinates = _selected_gff_genes(resolved, source.ids, deadline)
    _assert_region_source_unchanged(resolved, identity)

    intervals: list[RegionalInterval] = []
    for chrom, start, end, gene_name in coordinates:
        try:
            intervals.append(RegionalInterval(chrom=chrom, start=start - 1, end=end, name=gene_name))
        except ValueError as error:
            raise ValueError(f"GFF3 gene {gene_name or '<unnamed>'!r}: {error}") from error
    return intervals, resolved


def _union(intervals: Sequence[NormalizedInterval]) -> tuple[RegionalInterval, ...]:
    merged: list[RegionalInterval] = []
    for region in sorted(intervals, key=lambda item: (item.chrom, item.start, item.end)):
        if merged and merged[-1].chrom == region.chrom and region.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = RegionalInterval(
                chrom=previous.chrom,
                start=previous.start,
                end=max(previous.end, region.end),
            )
        else:
            merged.append(RegionalInterval(chrom=region.chrom, start=region.start, end=region.end))
    return tuple(merged)


def normalize_regions(
    source: object | None,
    reference_lengths: Mapping[str, int],
    *,
    exec_cfg: ExecutionConfig | None = None,
    deadline: RequestDeadline | None = None,
) -> NormalizedRegionSet:
    """Normalize one validated v2 source and enforce reference bounds.

    ``None`` represents whole-file/reference scope. Explicit empty lists remain
    invalid even when this helper is called below the Pydantic request layer.
    """
    lengths = _reference_map(reference_lengths, deadline)
    if source is None:
        return NormalizedRegionSet((), ())
    cfg = exec_cfg or ExecutionConfig()
    dependencies: tuple[Path, ...] = ()

    if isinstance(source, SamtoolsRegions):
        _check_region_count(len(source.values))
        intervals = [_samtools_interval(value, lengths) for value in source.values]
    elif isinstance(source, BedTextRegions):
        intervals = _bed_intervals(
            _bounded_lines(StringIO(source.text, newline=None), source="BED text", deadline=deadline),
            source="BED text",
        )
    elif isinstance(source, BedFileRegions):
        path = Path(source.path)
        _validate_input_file(path, cfg, allowed_exts=(".bed",))
        resolved = path.resolve()
        identity = _region_source_identity(resolved)
        with resolved.open(encoding="utf-8") as stream:
            intervals = _bed_intervals(
                _bounded_lines(stream, source=f"BED file {resolved}", deadline=deadline),
                source=f"BED file {resolved}",
            )
        _assert_region_source_unchanged(resolved, identity)
        dependencies = (resolved,)
    elif isinstance(source, Gff3Regions):
        intervals, dependency = _gff_intervals(source, cfg, deadline)
        dependencies = (dependency,)
    elif isinstance(source, list):
        if not source:
            raise ValueError("regions must be omitted for whole-file scope, not an empty list")
        _check_region_count(len(source))
        intervals = [
            item if isinstance(item, RegionalInterval) else RegionalInterval.model_validate(item) for item in source
        ]
    else:
        raise TypeError(f"Unsupported region source: {type(source).__name__}")

    _check_region_count(len(intervals))
    requested: list[NormalizedInterval] = []
    for index, interval in enumerate(intervals, start=1):
        _checkpoint(deadline)
        _validate_bounds(interval, lengths, f"Region {index}")
        requested.append(
            NormalizedInterval(
                chrom=interval.chrom,
                start=interval.start,
                end=interval.end,
                name=interval.name,
                region_id=f"region_{index}",
            )
        )
    frozen = tuple(requested)
    return NormalizedRegionSet(frozen, _union(frozen), dependencies)


__all__ = ["MAX_REGIONS", "NormalizedRegionSet", "normalize_regions"]
