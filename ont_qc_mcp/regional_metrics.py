"""Streaming alignment-record metrics for zero-based, half-open intervals.

Only M, =, and X operations contribute query bases or base qualities. D and N
contribute reference span, so a span overlap need not contain an aligned base.
Quality and MAPQ means are arithmetic Phred means, with missing values excluded.
Flag counts are nonexclusive and refer to retained span-overlapping records.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import asyncio
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .process_control import check_cancelled


# SAMv1 section 1.2.1: https://samtools.github.io/hts-specs/SAMv1.pdf
_REFERENCE_NAME = re.compile(r"[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*")
_QUERY_NAME = re.compile(r"[!-?A-~]{1,254}")
_CIGAR_OPERATION = re.compile(r"([0-9]+)([MIDNSHP=X])")
_UNSIGNED_INTEGER = re.compile(r"[0-9]+")
_SIGNED_INTEGER = re.compile(r"-?[0-9]+")
_SEQUENCE = re.compile(r"[A-Za-z=.]+")
_MAX_POSITION = 2**31 - 1


class RegionalInterval(BaseModel):
    """A SAM reference name and a nonempty zero-based, half-open interval."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    chrom: str = Field(max_length=1024)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    name: str | None = Field(default=None, max_length=256)

    @field_validator("chrom")
    @classmethod
    def validate_chrom(cls, value: str) -> str:
        if not _REFERENCE_NAME.fullmatch(value):
            raise ValueError("chrom must be a valid SAM reference sequence name")
        return value

    @model_validator(mode="after")
    def validate_coordinates(self) -> "RegionalInterval":
        if self.start >= self.end:
            raise ValueError("interval must satisfy 0 <= start < end")
        return self


@dataclass
class _Counts:
    span_overlapping_alignments: int = 0
    aligned_base_alignments: int = 0
    aligned_query_bases: int = 0
    mapq_known_alignments: int = 0
    mapq_missing_alignments: int = 0
    mapq_sum: int = 0
    quality_known_bases: int = 0
    quality_missing_bases: int = 0
    quality_sum: int = 0
    secondary_alignments: int = 0
    supplementary_alignments: int = 0
    duplicate_alignments: int = 0
    qcfail_alignments: int = 0
    reverse_alignments: int = 0


@dataclass(frozen=True)
class _Alignment:
    chrom: str
    start: int
    end: int
    flag: int
    mapq: int
    operations: list[tuple[int, str]]
    quality: str
    query_length: int


def _integer(value: str, name: str, minimum: int, maximum: int) -> int:
    pattern = _SIGNED_INTEGER if minimum < 0 else _UNSIGNED_INTEGER
    if not pattern.fullmatch(value):
        raise ValueError(f"invalid SAM {name}: expected a decimal integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"invalid SAM {name}: out of range")
    return result


def _parse_alignment(line: str) -> _Alignment | None:
    # At most one record is retained; optional tags are not split or interpreted.
    fields = line.removesuffix("\n").removesuffix("\r").split("\t", 11)
    if fields[0] in {"@HD", "@SQ", "@RG", "@PG", "@CO"}:
        return None
    if len(fields) < 11:
        raise ValueError("invalid SAM record: expected at least 11 tab-separated fields")
    qname, flag_text, chrom, pos_text, mapq_text, cigar, mate_chrom, mate_pos, tlen, sequence, quality = fields[:11]
    if not _QUERY_NAME.fullmatch(qname):
        raise ValueError("invalid SAM QNAME")
    flag = _integer(flag_text, "FLAG", 0, 65535)
    pos = _integer(pos_text, "POS", 0, _MAX_POSITION)
    mapq = _integer(mapq_text, "MAPQ", 0, 255)
    _integer(mate_pos, "PNEXT", 0, _MAX_POSITION)
    _integer(tlen, "TLEN", -_MAX_POSITION, _MAX_POSITION)
    if chrom != "*" and not _REFERENCE_NAME.fullmatch(chrom):
        raise ValueError("invalid SAM RNAME")
    if mate_chrom not in {"*", "="} and not _REFERENCE_NAME.fullmatch(mate_chrom):
        raise ValueError("invalid SAM RNEXT")
    if flag & 4:
        return None
    if chrom == "*" or pos == 0:
        raise ValueError("mapped SAM record requires RNAME and positive POS")
    if cigar == "*":
        raise ValueError("mapped SAM record requires CIGAR to measure regional metrics")

    operations: list[tuple[int, str]] = []
    cigar_cursor = query_length = reference_length = 0
    for operation_index, match in enumerate(_CIGAR_OPERATION.finditer(cigar)):
        if operation_index % 16384 == 0:
            check_cancelled()
        if match.start() != cigar_cursor:
            raise ValueError("invalid SAM CIGAR")
        length = int(match[1])
        if length == 0:
            raise ValueError("invalid SAM CIGAR: operation length must be positive")
        operation = match[2]
        operations.append((length, operation))
        if operation in "MIS=X":
            query_length += length
        if operation in "MDN=X":
            reference_length += length
        cigar_cursor = match.end()
    if not operations or cigar_cursor != len(cigar):
        raise ValueError("invalid SAM CIGAR")
    for index, (_, operation) in enumerate(operations):
        if index % 16384 == 0:
            check_cancelled()
        if operation == "H" and index not in {0, len(operations) - 1}:
            raise ValueError("invalid SAM CIGAR: hard clipping must be terminal")
        if operation == "S":
            left_terminal = index == 0 or (index == 1 and operations[0][1] == "H")
            right_terminal = index == len(operations) - 1 or (index == len(operations) - 2 and operations[-1][1] == "H")
            if not left_terminal and not right_terminal:
                raise ValueError("invalid SAM CIGAR: soft clipping must be terminal")
    if sequence != "*":
        if not _SEQUENCE.fullmatch(sequence) or len(sequence) != query_length:
            raise ValueError("invalid SAM SEQ or CIGAR query length mismatch")
    if quality != "*":
        if sequence == "*" or len(quality) != query_length:
            raise ValueError("invalid SAM QUAL or sequence length mismatch")
        for index, char in enumerate(quality):
            if index % 16384 == 0:
                check_cancelled()
            if not 33 <= ord(char) <= 126:
                raise ValueError("invalid SAM QUAL: expected printable ASCII Phred characters")
    return _Alignment(chrom, pos - 1, pos - 1 + reference_length, flag, mapq, operations, quality, query_length)


class RegionalAccumulator:
    """Accumulate independent interval metrics without retaining previous records.

    Repeated intervals remain distinct, and repeated QNAMEs remain separate
    records. FLAG 0x4 is always skipped. MAPQ 255 is missing and is retained only
    when min_mapq is zero. Any malformed mapped record invalidates this accumulator;
    results() then raises so callers cannot accidentally publish partial metrics.
    A contributing one-base query with QUAL '*' is rejected because SAM cannot
    distinguish a genuine Q9 character from its missing-quality sentinel.
    """

    def __init__(
        self,
        regions: list[RegionalInterval],
        exclude_flags: int = 1796,
        min_mapq: int = 0,
        *,
        include_base_quality: bool = True,
    ) -> None:
        if type(exclude_flags) is not int or not 0 <= exclude_flags <= 65535:
            raise ValueError("exclude_flags must be an integer in [0, 65535]")
        if type(min_mapq) is not int or not 0 <= min_mapq <= 254:
            raise ValueError("min_mapq must be an integer in [0, 254]")
        if not isinstance(regions, list) or any(not isinstance(region, RegionalInterval) for region in regions):
            raise ValueError("regions must be a list of RegionalInterval values")
        if type(include_base_quality) is not bool:
            raise ValueError("include_base_quality must be a boolean")
        self._regions = list(regions)
        self._exclude_flags = exclude_flags
        self._min_mapq = min_mapq
        self._include_base_quality = include_base_quality
        self._counts = [_Counts() for _ in regions]
        self._failed = False
        by_chrom: dict[str, list[int]] = {}
        for index, region in enumerate(regions):
            by_chrom.setdefault(region.chrom, []).append(index)
        self._index: dict[str, tuple[list[int], list[int], list[int]]] = {}
        for chrom, indices in by_chrom.items():
            indices.sort(key=lambda index: regions[index].start)
            starts = [regions[index].start for index in indices]
            prefix_ends: list[int] = []
            max_end = 0
            for index in indices:
                max_end = max(max_end, regions[index].end)
                prefix_ends.append(max_end)
            self._index[chrom] = (indices, starts, prefix_ends)

    def add_sam_line(self, line: str) -> None:
        if self._failed:
            raise ValueError("regional accumulator is invalid after malformed SAM input")
        try:
            check_cancelled()
            self._add_sam_line(line)
        except (ValueError, asyncio.CancelledError):
            self._failed = True
            raise

    def _add_sam_line(self, line: str) -> None:
        alignment = _parse_alignment(line)
        if alignment is None or alignment.flag & self._exclude_flags:
            return
        if self._min_mapq and (alignment.mapq == 255 or alignment.mapq < self._min_mapq):
            return
        indexed = self._index.get(alignment.chrom)
        if indexed is None or alignment.start == alignment.end:
            return
        indices, starts, prefix_ends = indexed
        lower = bisect_right(prefix_ends, alignment.start)
        upper = bisect_left(starts, alignment.end)
        for offset in range(lower, upper):
            check_cancelled()
            index = indices[offset]
            region = self._regions[index]
            if region.end > alignment.start:
                self._accumulate(self._counts[index], region, alignment, self._include_base_quality)

    @staticmethod
    def _accumulate(
        counts: _Counts,
        region: RegionalInterval,
        alignment: _Alignment,
        include_base_quality: bool,
    ) -> None:
        counts.span_overlapping_alignments += 1
        if alignment.mapq == 255:
            counts.mapq_missing_alignments += 1
        else:
            counts.mapq_known_alignments += 1
            counts.mapq_sum += alignment.mapq
        counts.secondary_alignments += bool(alignment.flag & 256)
        counts.supplementary_alignments += bool(alignment.flag & 2048)
        counts.duplicate_alignments += bool(alignment.flag & 1024)
        counts.qcfail_alignments += bool(alignment.flag & 512)
        counts.reverse_alignments += bool(alignment.flag & 16)
        reference_cursor = alignment.start
        query_cursor = 0
        aligned_bases = 0
        for operation_index, (length, operation) in enumerate(alignment.operations):
            if operation_index % 16384 == 0:
                check_cancelled()
            if operation in "M=X":
                start = max(reference_cursor, region.start)
                end = min(reference_cursor + length, region.end)
                if start < end:
                    bases = end - start
                    aligned_bases += bases
                    if not include_base_quality:
                        pass
                    elif alignment.quality == "*":
                        if alignment.query_length == 1:
                            raise ValueError(
                                "SAM cannot distinguish one-base Q9 from missing quality (QUAL '*'); "
                                "regional metrics cannot measure this contributing record unambiguously"
                            )
                        counts.quality_missing_bases += bases
                    else:
                        query_start = query_cursor + start - reference_cursor
                        counts.quality_known_bases += bases
                        # SAM already orients SEQ and QUAL to the alignment strand.
                        for quality_start in range(query_start, query_start + bases, 16384):
                            check_cancelled()
                            quality_end = min(quality_start + 16384, query_start + bases)
                            counts.quality_sum += sum(
                                ord(alignment.quality[i]) - 33 for i in range(quality_start, quality_end)
                            )
            if operation in "MIS=X":
                query_cursor += length
            if operation in "MDN=X":
                reference_cursor += length
        counts.aligned_query_bases += aligned_bases
        counts.aligned_base_alignments += bool(aligned_bases)

    def results(self) -> list[dict[str, object]]:
        """Return snapshots in input order; means with empty denominators are None."""
        if self._failed:
            raise ValueError("regional accumulator is invalid after malformed SAM input")
        results: list[dict[str, object]] = []
        for index, (region, counts) in enumerate(zip(self._regions, self._counts)):
            result: dict[str, object] = {"region_id": f"region_{index + 1}", **region.model_dump(), **vars(counts)}
            result.pop("mapq_sum")
            result.pop("quality_sum")
            result["mean_mapq"] = (
                counts.mapq_sum / counts.mapq_known_alignments if counts.mapq_known_alignments else None
            )
            result["mean_base_quality"] = (
                counts.quality_sum / counts.quality_known_bases if counts.quality_known_bases else None
            )
            results.append(result)
        return results
