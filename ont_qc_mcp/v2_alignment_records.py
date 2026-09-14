"""Stream selected SAM records into API v2 alignment metric groups."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .process_control import check_cancelled
from .regional_metrics import RegionalAccumulator, RegionalInterval, _parse_alignment


def _integer_field(result: dict[str, object], name: str) -> int:
    value = result[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"regional metric {name!r} must be an integer")
    return value


def _optional_float_field(result: dict[str, object], name: str) -> float | None:
    value = result[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"regional metric {name!r} must be numeric or null")
    return float(value)


@dataclass
class _WholeRecordAccumulator:
    include_base_quality: bool
    eligible_records: int = 0
    mapped_records: int = 0
    unmapped_records: int = 0
    secondary_records: int = 0
    supplementary_records: int = 0
    known_mapq_records: int = 0
    missing_mapq_records: int = 0
    mapq_sum: int = 0
    aligned_query_bases: int = 0
    known_quality_bases: int = 0
    missing_quality_bases: int = 0
    quality_sum: int = 0

    def add_sam_line(self, line: str) -> None:
        check_cancelled()
        stripped = line.removesuffix("\n").removesuffix("\r")
        if stripped.startswith(("@HD\t", "@SQ\t", "@RG\t", "@PG\t", "@CO\t")):
            return
        fields = stripped.split("\t", 11)
        if len(fields) < 11:
            raise ValueError("invalid SAM record: expected at least 11 tab-separated fields")
        try:
            flag = int(fields[1])
            mapq = int(fields[4])
        except ValueError as error:
            raise ValueError("invalid SAM FLAG or MAPQ: expected decimal integers") from error
        if not 0 <= flag <= 65535 or not 0 <= mapq <= 255:
            raise ValueError("invalid SAM FLAG or MAPQ: out of range")

        alignment = _parse_alignment(line)
        self.eligible_records += 1
        self.secondary_records += bool(flag & 0x100)
        self.supplementary_records += bool(flag & 0x800)
        if flag & 0x4:
            self.unmapped_records += 1
        else:
            if alignment is None:
                raise ValueError("mapped SAM record was not parsed")
            self.mapped_records += 1
        if mapq == 255:
            self.missing_mapq_records += 1
        else:
            self.known_mapq_records += 1
            self.mapq_sum += mapq

        if not self.include_base_quality or alignment is None:
            return
        query_cursor = 0
        for operation_index, (length, operation) in enumerate(alignment.operations):
            if operation_index % 16384 == 0:
                check_cancelled()
            if operation in "M=X":
                self.aligned_query_bases += length
                if alignment.quality == "*":
                    if alignment.query_length == 1:
                        raise ValueError(
                            "SAM cannot distinguish one-base Q9 from missing quality (QUAL '*'); "
                            "alignment metrics cannot measure this contributing record unambiguously"
                        )
                    self.missing_quality_bases += length
                else:
                    self.known_quality_bases += length
                    for quality_start in range(query_cursor, query_cursor + length, 16384):
                        check_cancelled()
                        quality_end = min(quality_start + 16384, query_cursor + length)
                        self.quality_sum += sum(
                            ord(alignment.quality[index]) - 33 for index in range(quality_start, quality_end)
                        )
            if operation in "MIS=X":
                query_cursor += length

    def result(self) -> dict[str, object]:
        return {
            "counts": {
                "eligible_records": self.eligible_records,
                "mapped_records": self.mapped_records,
                "unmapped_records": self.unmapped_records,
                "secondary_records": self.secondary_records,
                "supplementary_records": self.supplementary_records,
            },
            "mapping_quality": {
                "known_records": self.known_mapq_records,
                "missing_records": self.missing_mapq_records,
                "mean_mapq": self.mapq_sum / self.known_mapq_records if self.known_mapq_records else None,
            },
            "aligned_base_quality": {
                "aligned_query_bases": self.aligned_query_bases,
                "known_quality_bases": self.known_quality_bases,
                "missing_quality_bases": self.missing_quality_bases,
                "mean_base_quality": (
                    self.quality_sum / self.known_quality_bases if self.known_quality_bases else None
                ),
            },
        }


def _regional_result(result: dict[str, object]) -> dict[str, object]:
    eligible = _integer_field(result, "span_overlapping_alignments")
    return {
        "region_id": result["region_id"],
        "region_name": result["name"],
        "counts": {
            "eligible_records": eligible,
            "mapped_records": eligible,
            "unmapped_records": 0,
            "secondary_records": result["secondary_alignments"],
            "supplementary_records": result["supplementary_alignments"],
        },
        "mapping_quality": {
            "known_records": result["mapq_known_alignments"],
            "missing_records": result["mapq_missing_alignments"],
            "mean_mapq": result["mean_mapq"],
        },
        "aligned_base_quality": {
            "aligned_query_bases": result["aligned_query_bases"],
            "known_quality_bases": result["quality_known_bases"],
            "missing_quality_bases": result["quality_missing_bases"],
            "mean_base_quality": result["mean_base_quality"],
        },
    }


def _combined_region_quality(results: list[dict[str, object]]) -> dict[str, object]:
    aligned = sum(_integer_field(result, "aligned_query_bases") for result in results)
    known = sum(_integer_field(result, "quality_known_bases") for result in results)
    missing = sum(_integer_field(result, "quality_missing_bases") for result in results)
    quality_sum = sum(
        (_optional_float_field(result, "mean_base_quality") or 0) * _integer_field(result, "quality_known_bases")
        for result in results
    )
    return {
        "aligned_query_bases": aligned,
        "known_quality_bases": known,
        "missing_quality_bases": missing,
        "mean_base_quality": quality_sum / known if known else None,
    }


def collect_alignment_records(
    stream: TextIO,
    *,
    regions: list[RegionalInterval],
    group_by: str,
    include_base_quality: bool,
) -> list[dict[str, object]]:
    """Collect one selected SAM population without returning partial evidence."""
    if group_by not in {"combined", "region"}:
        raise ValueError("group_by must be 'combined' or 'region'")
    if group_by == "region" and not regions:
        raise ValueError("region grouping requires regions")

    whole = _WholeRecordAccumulator(include_base_quality=include_base_quality and not regions)
    regional = (
        RegionalAccumulator(regions, exclude_flags=0, min_mapq=0, include_base_quality=include_base_quality)
        if regions
        else None
    )
    for line in stream:
        whole.add_sam_line(line)
        if regional is not None:
            regional.add_sam_line(line)

    if regional is None:
        return [whole.result()]
    regional_results = regional.results()
    if group_by == "region":
        return [_regional_result(result) for result in regional_results]
    combined = whole.result()
    combined["aligned_base_quality"] = _combined_region_quality(regional_results)
    return [combined]


def _load_regions(path: str | None) -> list[RegionalInterval]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("regions payload must be a list")
    return [RegionalInterval.model_validate(value) for value in payload]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions")
    parser.add_argument("--group-by", choices=("combined", "region"), required=True)
    parser.add_argument("--aligned-base-quality", action="store_true")
    args = parser.parse_args()
    try:
        result = collect_alignment_records(
            sys.stdin,
            regions=_load_regions(args.regions),
            group_by=args.group_by,
            include_base_quality=args.aligned_base_quality,
        )
        print(json.dumps({"groups": result}, separators=(",", ":")))
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        print(error, file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()


__all__ = ["collect_alignment_records"]
