"""Inspect selected SAM records before API v2 FASTQ conversion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


REPORT_PREFIX = "ONT_QC_READ_RECORDS\t"


@dataclass(frozen=True)
class ReadRecordReport:
    selected_records: int
    emitted_sequences: int
    conversion_exclusions: int


def inspect_sam(*, quality_required: bool, discard: bool) -> ReadRecordReport:
    """Drain SAM stdin, exclude missing sequences, and pass retained records on."""
    selected_records = 0
    emitted_sequences = 0
    conversion_exclusions = 0
    missing_quality = 0
    output = sys.stdout.buffer

    for line_number, line in enumerate(sys.stdin.buffer, start=1):
        if line.startswith(b"@"):
            if not discard:
                output.write(line)
            continue
        fields = line.rstrip(b"\r\n").split(b"\t")
        if len(fields) < 11:
            raise ValueError(f"Selected SAM record on line {line_number} has fewer than 11 fields")
        selected_records += 1
        missing_quality += fields[10] == b"*"
        if fields[9] == b"*":
            conversion_exclusions += 1
            continue
        emitted_sequences += 1
        if not discard:
            output.write(line)

    if quality_required and missing_quality:
        raise ValueError(
            f"Read quality was requested, but {missing_quality} selected primary record(s) have absent QUAL"
        )
    return ReadRecordReport(
        selected_records=selected_records,
        emitted_sequences=emitted_sequences,
        conversion_exclusions=conversion_exclusions,
    )


def parse_report(stderr: str) -> ReadRecordReport:
    matches = [line.removeprefix(REPORT_PREFIX) for line in stderr.splitlines() if line.startswith(REPORT_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError("Read-record inspection did not emit exactly one accounting report")
    data = json.loads(matches[0])
    return ReadRecordReport(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-required", action="store_true")
    parser.add_argument("--discard", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect_sam(quality_required=args.quality_required, discard=args.discard)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"{REPORT_PREFIX}{json.dumps(asdict(report), separators=(',', ':'))}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a pipeline stage
    raise SystemExit(main())


__all__ = ["REPORT_PREFIX", "ReadRecordReport", "inspect_sam", "parse_report"]
