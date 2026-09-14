"""Keep only samtools stats rows represented by the API v2 error profile."""

from __future__ import annotations

import sys
from typing import TextIO

from .process_control import check_cancelled


_SUMMARY_KEYS = (
    "SN\terror rate:",
    "SN\tmismatches per base:",
    "SN\tinsertions per base:",
    "SN\tdeletions per base:",
)
_ROW_PREFIXES = ("COV\t", "MPC\t", "IS\t")


def filter_error_profile_lines(source: TextIO, destination: TextIO) -> None:
    for line in source:
        check_cancelled()
        if line.startswith(_SUMMARY_KEYS) or line.startswith(_ROW_PREFIXES):
            destination.write(line)


def main() -> None:
    filter_error_profile_lines(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()


__all__ = ["filter_error_profile_lines"]
