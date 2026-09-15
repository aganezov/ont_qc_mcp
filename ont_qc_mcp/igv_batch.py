from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .schemas import IgvRegion


SNAPSHOT_EXTENSIONS = {"png", "svg"}

# The IGV batch format is line-oriented with no escaping mechanism: each line is a
# separate command. Any control character (newline/CR especially) in a caller-supplied
# field would inject additional IGV commands. Reject such values — fail closed — so the
# script's line structure is controlled by this code, never by untrusted field content.
# Covers C0 controls (incl. \n, \r, \t), DEL, and C1 controls (0x80-0x9f).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_OUTPUT_ROUTING_COMMANDS = {"snapshot", "snapshotdirectory"}


class IgvBatchValidationError(ValueError):
    """Raised when an IGV batch field contains characters that could inject commands."""


def _safe(value: str, field: str) -> str:
    """Return ``value`` unchanged, or raise if it contains a control character."""
    if _CONTROL_CHARS.search(value):
        raise IgvBatchValidationError(
            f"IGV batch field {field!r} contains a control character that could inject an "
            f"IGV command; refusing to build the batch script (value={value!r})"
        )
    return value


def _safe_extra_command(value: str, field: str) -> str:
    """Return a safe display command while reserving generated-batch output routing."""
    command = _safe(value, field)
    stripped = command.lstrip()
    verb = stripped.split(maxsplit=1)[0].casefold() if stripped else ""
    if verb in _OUTPUT_ROUTING_COMMANDS:
        raise IgvBatchValidationError(f"IGV batch field {field!r} contains a reserved output-routing command")
    return command


def _format_preference(key: str, value: str | int | float | bool) -> str:
    return f"preference {_safe(key, 'preference key')} {_safe(str(value), 'preference value')}"


def _snapshot_name(region: IgvRegion, snapshot_format: str) -> str:
    """
    Derive snapshot filename, appending extension when missing or mismatched.
    """
    name = region.name
    if not name:
        name = (
            f"{region.chrom}:{region.start}-{region.end}".replace(":", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
    elif "/" in name or "\\" in name:
        raise IgvBatchValidationError("IGV snapshot region names must not contain path separators")

    needs_extension = True
    for ext in SNAPSHOT_EXTENSIONS:
        if name.lower().endswith(f".{ext}"):
            needs_extension = ext != snapshot_format
            if needs_extension:
                name = name[: -(len(ext) + 1)]  # strip existing extension
            else:
                return name
            break

    return f"{name}.{snapshot_format}"


def _unique_snapshot_names(regions: list[IgvRegion], snapshot_format: str) -> list[str]:
    desired = [_snapshot_name(region, snapshot_format) for region in regions]
    reserved = set(desired)
    used: set[str] = set()
    unique: list[str] = []
    for name in desired:
        if name not in used:
            selected = name
        else:
            extension_start = name.rfind(".")
            stem, extension = name[:extension_start], name[extension_start:]
            suffix = 2
            selected = f"{stem}_{suffix}{extension}"
            while selected in reserved or selected in used:
                suffix += 1
                selected = f"{stem}_{suffix}{extension}"
        used.add(selected)
        unique.append(selected)
    return unique


def _expand_region(start: int, end: int, min_width: int, *, min_start: int | None = None) -> tuple[int, int]:
    width = end - start
    if min_width <= 0 or width >= min_width:
        return start, end
    diff = min_width - width
    pad_left = diff // 2
    pad_right = diff - pad_left
    expanded_start = start - pad_left
    expanded_end = end + pad_right
    if min_start is not None and expanded_start < min_start:
        expanded_end += min_start - expanded_start
        expanded_start = min_start
    return expanded_start, expanded_end


def _header_lines(
    genome: str,
    tracks: Iterable[str],
    snapshot_dir: Path,
    compact: str,
    color_by: str | None,
    group_by: str | None,
    small_indels_show: bool,
    small_indels_threshold: int,
    allele_threshold: float,
    extra_preferences: dict[str, str] | None,
    extra_commands: list[str] | None,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"genome {_safe(genome, 'genome')}")
    for track in tracks:
        lines.append(f"load {_safe(track, 'track')}")

    # Preferences
    hide_small_indels = not small_indels_show
    lines.append(_format_preference("SAM.HIDE_SMALL_INDEL", str(hide_small_indels).upper()))
    lines.append(_format_preference("SAM.SMALL_INDEL_BP_THRESHOLD", small_indels_threshold))
    lines.append(_format_preference("SAM.SHOW_INSERTION_MARKERS", "FALSE"))
    lines.append(_format_preference("SAM.ALLELE_THRESHOLD", f"{allele_threshold:.2f}"))

    # Display options
    lines.append(_safe(compact, "compact"))
    if color_by:
        lines.append(f"colorBy {_safe(color_by, 'color_by')}")
    if group_by:
        lines.append(f"group {_safe(group_by, 'group_by')}")
    lines.append(f"snapshotDirectory {_safe(str(snapshot_dir), 'snapshot_dir')}")

    # Extra preferences
    for key, value in (extra_preferences or {}).items():
        lines.append(_format_preference(key, value))

    # Global extra commands
    if extra_commands:
        lines.extend(_safe_extra_command(cmd, "extra_commands") for cmd in extra_commands)

    return lines


def _region_lines(
    region: IgvRegion,
    snapshot_format: str,
    min_snapshot_width: int,
    regions_are_zero_based_half_open: bool = False,
    snapshot_name: str | None = None,
) -> list[str]:
    start, end = _expand_region(
        region.start,
        region.end,
        min_snapshot_width,
        min_start=0 if regions_are_zero_based_half_open else None,
    )
    if regions_are_zero_based_half_open:
        start += 1
    region_str = f"{_safe(region.chrom, 'region.chrom')}:{start}-{end}"
    final_snapshot_name = _safe(snapshot_name or _snapshot_name(region, snapshot_format), "region.name")

    lines: list[str] = [f"goto {region_str}"]
    if region.extra_commands:
        lines.extend(_safe_extra_command(cmd, "region.extra_commands") for cmd in region.extra_commands)
    lines.append(f"snapshot {final_snapshot_name}")
    return lines


def generate_igv_batch(
    genome: str,
    tracks: list[str],
    regions: list[IgvRegion],
    output_path: Path,
    compact: str = "squish",
    color_by: str | None = None,
    group_by: str | None = None,
    snapshot_dir: str | Path = ".",
    snapshot_format: str = "png",
    min_snapshot_width: int = 0,
    # IGV preferences (common ones with typed parameters)
    small_indels_show: bool = False,
    small_indels_threshold: int = 100,
    allele_threshold: float = 0.2,
    # Extensibility
    extra_commands: list[str] | None = None,
    extra_preferences: dict[str, str] | None = None,
    regions_are_zero_based_half_open: bool = False,
) -> Path:
    """
    Generate an IGV batch file and return its path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snap_dir_path = Path(snapshot_dir).absolute()

    lines = _header_lines(
        genome=genome,
        tracks=tracks,
        snapshot_dir=snap_dir_path,
        compact=compact,
        color_by=color_by,
        group_by=group_by,
        small_indels_show=small_indels_show,
        small_indels_threshold=small_indels_threshold,
        allele_threshold=allele_threshold,
        extra_preferences=extra_preferences,
        extra_commands=extra_commands,
    )

    snapshot_names = _unique_snapshot_names(regions, snapshot_format)
    for region, snapshot_name in zip(regions, snapshot_names, strict=True):
        lines.extend(
            _region_lines(
                region,
                snapshot_format=snapshot_format,
                min_snapshot_width=min_snapshot_width,
                regions_are_zero_based_half_open=regions_are_zero_based_half_open,
                snapshot_name=snapshot_name,
            )
        )

    lines.append("exit")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = ["generate_igv_batch", "IgvBatchValidationError"]
