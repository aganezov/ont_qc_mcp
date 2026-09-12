"""Hand-computed regional metrics and parser boundary controls."""

import asyncio
import threading

import pytest
from pydantic import ValidationError

from ont_qc_mcp.process_control import CANCEL_EVENT
from ont_qc_mcp.regional_metrics import RegionalAccumulator, RegionalInterval


def sam(
    cigar="3M",
    *,
    chrom="chr1",
    pos=101,
    flag=0,
    mapq=60,
    sequence="AAA",
    quality="III",
    qname="same_read",
):
    return f"{qname}\t{flag}\t{chrom}\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t{sequence}\t{quality}"


def interval(start=100, end=103, chrom="chr1", name=None):
    return RegionalInterval(chrom=chrom, start=start, end=end, name=name)


def test_mixed_cigar_boundaries_and_reverse_orientation():
    regions = [interval(102, 113), interval(105, 107), interval(109, 112), interval(114, 115), interval(99, 100)]
    accumulator = RegionalAccumulator(regions)
    accumulator.add_sam_line(
        sam("2H2S3M1I2=2D2X3N2M1S1H", sequence="A" * 13, quality="".join(chr(33 + i) for i in range(13)), flag=16)
    )
    mixed, deletion, skip, after, before = accumulator.results()
    assert mixed["span_overlapping_alignments"] == 1
    assert mixed["aligned_base_alignments"] == 1
    assert mixed["aligned_query_bases"] == 6
    assert mixed["quality_known_bases"] == 6
    assert mixed["quality_missing_bases"] == 0
    assert mixed["mean_base_quality"] == pytest.approx(44 / 6)
    assert mixed["reverse_alignments"] == 1
    for result in (deletion, skip):
        assert result["span_overlapping_alignments"] == 1
        assert result["aligned_base_alignments"] == 0
        assert result["aligned_query_bases"] == 0
        assert result["mean_base_quality"] is None
        assert result["mean_mapq"] == 60
    for result in (after, before):
        assert result["span_overlapping_alignments"] == 0
        assert result["mean_mapq"] is None


def test_whole_read_quality_negative_control():
    qualities = list(range(13))
    changed = qualities.copy()
    # Clipped, inserted, and aligned-outside-interval positions must not affect Q.
    for position in (0, 1, 2, 3, 5, 11, 12):
        changed[position] = 90
    results = []
    for scores in (qualities, changed):
        accumulator = RegionalAccumulator([interval(102, 113)])
        accumulator.add_sam_line(
            sam("2H2S3M1I2=2D2X3N2M1S1H", sequence="A" * 13, quality="".join(chr(33 + i) for i in scores), flag=16)
        )
        results.append(accumulator.results())
    assert results[0] == results[1]


def test_flags_are_nonexclusive_and_default_filters_retain_supplementary():
    default = RegionalAccumulator([interval()])
    permissive = RegionalAccumulator([interval()], exclude_flags=0)
    flags = [0, 4, 256, 512, 1024, 2048, 16 | 256 | 512 | 1024 | 2048]
    for flag in flags:
        line = sam(flag=flag)
        default.add_sam_line(line)
        permissive.add_sam_line(line)
    retained = default.results()[0]
    assert retained["span_overlapping_alignments"] == 2
    assert retained["supplementary_alignments"] == 1
    assert retained["secondary_alignments"] == 0
    all_mapped = permissive.results()[0]
    assert all_mapped["span_overlapping_alignments"] == 6
    for flag_count in ("secondary_alignments", "supplementary_alignments", "duplicate_alignments", "qcfail_alignments"):
        assert all_mapped[flag_count] == 2
    assert all_mapped["reverse_alignments"] == 1


@pytest.mark.parametrize("min_mapq,expected", [(0, 4), (1, 2), (20, 2), (21, 1), (254, 1)])
def test_missing_mapq_is_not_treated_as_high_quality(min_mapq, expected):
    accumulator = RegionalAccumulator([interval()], min_mapq=min_mapq)
    for mapq in (0, 20, 254, 255):
        accumulator.add_sam_line(sam(mapq=mapq))
    result = accumulator.results()[0]
    assert result["span_overlapping_alignments"] == expected
    assert result["mapq_missing_alignments"] == (1 if min_mapq == 0 else 0)
    assert result["mapq_known_alignments"] == (3 if min_mapq == 0 else expected)


def test_separate_mean_denominators_and_missing_sequence():
    accumulator = RegionalAccumulator([interval()])
    accumulator.add_sam_line(sam("1M", sequence="A", quality="+", mapq=20))
    accumulator.add_sam_line(sam(quality="555", mapq=40))
    accumulator.add_sam_line(sam("2M", sequence="*", quality="*", mapq=255))
    accumulator.add_sam_line(sam("2M", sequence="AA", quality="*", mapq=255))
    result = accumulator.results()[0]
    assert result["mean_mapq"] == 30
    assert result["mapq_known_alignments"] == 2
    assert result["mapq_missing_alignments"] == 2
    assert result["aligned_query_bases"] == 8
    assert result["quality_known_bases"] == 4
    assert result["quality_missing_bases"] == 4
    assert result["mean_base_quality"] == 17.5


@pytest.mark.parametrize("sequence", ["A", "*"])
def test_contributing_one_base_quality_sentinel_is_ambiguous(sequence):
    accumulator = RegionalAccumulator([interval()])
    accumulator.add_sam_line(sam())
    with pytest.raises(ValueError, match="cannot distinguish one-base Q9 from missing quality"):
        accumulator.add_sam_line(sam("1M", sequence=sequence, quality="*"))
    with pytest.raises(ValueError, match="invalid after"):
        accumulator.results()


@pytest.mark.parametrize("quality,expected", [("\u0029", 8), ("+", 10)])
def test_one_base_known_quality_neighbors_remain_measurable(quality, expected):
    accumulator = RegionalAccumulator([interval()])
    accumulator.add_sam_line(sam("1M", sequence="A", quality=quality))
    result = accumulator.results()[0]
    assert result["quality_known_bases"] == 1
    assert result["quality_missing_bases"] == 0
    assert result["mean_base_quality"] == expected


@pytest.mark.parametrize("updates", [{"flag": 256}, {"flag": 4}, {"mapq": 0}, {"mapq": 255}, {"pos": 501}])
def test_unselected_one_base_quality_ambiguity_does_not_fail(updates):
    accumulator = RegionalAccumulator([interval()], min_mapq=1)
    accumulator.add_sam_line(sam("1M", sequence="A", quality="*", **updates))
    assert accumulator.results()[0]["span_overlapping_alignments"] == 0


@pytest.mark.parametrize(
    "cigar,start,end,span_count", [("1I", 100, 103, 0), ("1I3D", 100, 103, 1), ("1M3D", 101, 104, 1)]
)
def test_one_base_quality_ambiguity_without_contributing_bases_does_not_fail(cigar, start, end, span_count):
    accumulator = RegionalAccumulator([interval(start, end)])
    accumulator.add_sam_line(sam(cigar, sequence="A", quality="*"))
    result = accumulator.results()[0]
    assert result["span_overlapping_alignments"] == span_count
    assert result["aligned_query_bases"] == 0
    assert result["mean_base_quality"] is None


def test_intervals_and_records_are_not_deduplicated_and_order_is_preserved():
    accumulator = RegionalAccumulator([interval(102, 104, name="last"), interval(name="first"), interval(name="first")])
    accumulator.add_sam_line(sam())
    accumulator.add_sam_line(sam())
    results = accumulator.results()
    assert [result["region_id"] for result in results] == ["region_1", "region_2", "region_3"]
    assert [result["name"] for result in results] == ["last", "first", "first"]
    assert [result["aligned_query_bases"] for result in results] == [2, 6, 6]
    assert [result["span_overlapping_alignments"] for result in results] == [2, 2, 2]
    results[0]["span_overlapping_alignments"] = 100
    assert accumulator.results()[0]["span_overlapping_alignments"] == 2


def test_empty_and_deletion_skip_only_records():
    accumulator = RegionalAccumulator([interval(), interval(chrom="chr2")])
    empty = accumulator.results()[0]
    assert empty["mean_mapq"] is None
    assert empty["mean_base_quality"] is None
    accumulator.add_sam_line(sam("2D1N", sequence="*", quality="*", mapq=255))
    result, other_chrom = accumulator.results()
    assert result["span_overlapping_alignments"] == 1
    assert result["mapq_missing_alignments"] == 1
    assert result["aligned_base_alignments"] == 0
    assert result["aligned_query_bases"] == 0
    assert result["quality_missing_bases"] == 0
    assert result["mean_base_quality"] is None
    assert other_chrom == {**empty, "region_id": "region_2", "chrom": "chr2"}


def test_zero_reference_span_does_not_overlap_and_padding_does_not_advance():
    accumulator = RegionalAccumulator([interval(100, 101)])
    accumulator.add_sam_line(sam("3I", sequence="AAA", quality="III"))
    assert accumulator.results()[0]["span_overlapping_alignments"] == 0
    accumulator.add_sam_line(sam("1M5P2M", quality="+55"))
    assert accumulator.results()[0]["aligned_query_bases"] == 1
    assert accumulator.results()[0]["mean_base_quality"] == 10


def test_nested_intervals_and_disjoint_index_boundaries():
    regions = [interval(i * 10, i * 10 + 5) for i in range(2000)]
    regions.insert(100, interval(0, 20000, name="outer"))
    accumulator = RegionalAccumulator(regions)
    accumulator.add_sam_line(sam("3M", pos=15003))
    results = accumulator.results()
    assert results[100]["aligned_query_bases"] == 3
    assert results[1501]["aligned_query_bases"] == 3
    assert sum(result["span_overlapping_alignments"] == 1 for result in results) == 2


def test_headers_line_endings_and_optional_tags():
    accumulator = RegionalAccumulator([interval()])
    for header in ("@HD\tVN:1.6", "@SQ\tSN:chr1\tLN:1000", "@CO\tcomment"):
        accumulator.add_sam_line(header)
    accumulator.add_sam_line(sam() + "\tNM:i:0\tZZ:Z:ignored\r\n")
    assert accumulator.results()[0]["span_overlapping_alignments"] == 1


@pytest.mark.parametrize(
    "chrom",
    [
        "",
        "*",
        "=chr1",
        "chr 1",
        "chr\t1",
        "chr\n1",
        "chr\x001",
        "chré",
        "chr(1)",
        "chr,1",
        "chr\\1",
        "chr<1>",
        "x" * 1025,
    ],
)
def test_invalid_reference_names(chrom):
    with pytest.raises(ValidationError):
        interval(chrom=chrom)


@pytest.mark.parametrize("chrom", ["chr1", "HLA:DRB1*01:01", "NC_000001.11", "a=b", "-chr1", "chr|1"])
def test_valid_sam_reference_names(chrom):
    assert interval(chrom=chrom).chrom == chrom


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (2, 1), (True, 2), (0, True), (0.0, 2), (0, 2.0), ("0", 2)])
def test_invalid_coordinates(start, end):
    with pytest.raises(ValidationError):
        interval(start, end)


def test_strict_extra_name_fields_and_model_immutability():
    for extra in ({"unknown": 1}, {"name": 7}, {"name": "x" * 257}):
        with pytest.raises(ValidationError):
            RegionalInterval(chrom="chr1", start=0, end=1, **extra)
    region = interval()
    with pytest.raises(ValidationError):
        region.start = 50


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("exclude_flags", -1),
        ("exclude_flags", 65536),
        ("exclude_flags", True),
        ("exclude_flags", 1.0),
        ("min_mapq", -1),
        ("min_mapq", 255),
        ("min_mapq", True),
        ("min_mapq", 1.0),
    ],
)
def test_strict_constructor_filters(parameter, value):
    with pytest.raises(ValueError):
        RegionalAccumulator([interval()], **{parameter: value})


@pytest.mark.parametrize(
    "updates",
    [
        {"flag": -1},
        {"flag": "1.0"},
        {"flag": "1x"},
        {"flag": 65536},
        {"pos": 0},
        {"pos": "1.5"},
        {"pos": " 101"},
        {"pos": 2**31},
        {"mapq": -1},
        {"mapq": 256},
        {"mapq": "NaN"},
        {"mapq": "６０"},
        {"cigar": "*"},
        {"cigar": "3Mgarbage"},
        {"cigar": "x3M"},
        {"cigar": "1Mbad2M"},
        {"cigar": "0M3M"},
        {"cigar": "3"},
        {"cigar": ""},
        {"cigar": "3m"},
        {"cigar": "3B"},
        {"cigar": "٣M"},
        {"cigar": "1M1S1M"},
        {"cigar": "1M1H2M"},
        {"quality": "II"},
        {"quality": "IIII"},
        {"quality": "I I"},
        {"quality": "I\x7fI"},
        {"quality": "IéI"},
        {"quality": "I\nI"},
        {"sequence": "AA"},
        {"sequence": "AéA"},
        {"sequence": "*", "quality": "III"},
        {"chrom": "*"},
        {"chrom": "chr 1"},
        {"qname": "read name"},
        {"qname": "@bad"},
    ],
)
def test_malformed_record_rejected_without_partial_results(updates):
    accumulator = RegionalAccumulator([interval()])
    accumulator.add_sam_line(sam())
    with pytest.raises(ValueError):
        accumulator.add_sam_line(sam(**updates))
    with pytest.raises(ValueError, match="invalid after"):
        accumulator.results()
    with pytest.raises(ValueError, match="invalid after"):
        accumulator.add_sam_line(sam())


@pytest.mark.parametrize("field,value", [(6, "bad name"), (7, "1.5"), (8, "NaN")])
def test_malformed_mate_fields_rejected(field, value):
    fields = sam().split("\t")
    fields[field] = value
    with pytest.raises(ValueError):
        RegionalAccumulator([interval()]).add_sam_line("\t".join(fields))


def test_unmapped_is_always_skipped_but_filtered_mapped_missing_cigar_fails():
    accumulator = RegionalAccumulator([interval()], exclude_flags=0)
    accumulator.add_sam_line(sam("*", flag=4, chrom="*", pos=0, sequence="*", quality="*"))
    assert accumulator.results()[0]["span_overlapping_alignments"] == 0
    with pytest.raises(ValueError, match="requires CIGAR"):
        RegionalAccumulator([interval()]).add_sam_line(sam("*", flag=256))


def test_cancellation_invalidates_accumulator():
    event = threading.Event()
    event.set()
    token = CANCEL_EVENT.set(event)
    accumulator = RegionalAccumulator([interval()])
    try:
        with pytest.raises(asyncio.CancelledError):
            accumulator.add_sam_line(sam())
    finally:
        CANCEL_EVENT.reset(token)
    with pytest.raises(ValueError, match="invalid after"):
        accumulator.results()
