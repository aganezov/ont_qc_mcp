"""Footer requirements and integer count invariants for mosdepth summaries."""

import pytest

from ont_qc_mcp.parsers import parse_mosdepth_summary

HEADER = "chrom\tlength\tbases\tmean\tmin\tmax\n"
INPUT = "sample-input.bam"
WHOLE_ROWS = ["chrA\t100\t1000\t10\t10\t10", "chrB\t300\t300\t1\t1\t1"]
WHOLE_TOTAL = "total\t400\t1300\t3.25\t1\t10"
REGION_ROWS = [WHOLE_ROWS[0], "chrA_region\t20\t300\t15\t15\t15", WHOLE_ROWS[1], "chrB_region\t0\t0\t0\t0\t0"]
REGION_TOTAL = "total_region\t20\t300\t15\t0\t15"


def summary(rows: list[str]) -> str:
    return HEADER + "\n".join(rows) + "\n"


def test_missing_final_region_total_fails_default_inference() -> None:
    # Before this fix the remaining total implied whole mode and region rows
    # silently appeared as contigs. Their counts cannot sum to this total.
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary([*REGION_ROWS, WHOLE_TOTAL]), INPUT)


@pytest.mark.parametrize("expected_region_mode", [False, True], ids=["whole", "region"])
@pytest.mark.parametrize("text", ["", HEADER, summary(WHOLE_ROWS)], ids=["empty", "header-only", "no-footer"])
def test_known_mode_requires_footer(text: str, expected_region_mode: bool) -> None:
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(text, INPUT, expected_region_mode=expected_region_mode)


@pytest.mark.parametrize(
    ("rows", "expected_region_mode"),
    [([*REGION_ROWS, WHOLE_TOTAL, REGION_TOTAL], False), ([*WHOLE_ROWS, WHOLE_TOTAL], True)],
    ids=["region-footer-in-whole-mode", "whole-footer-in-region-mode"],
)
def test_known_mode_rejects_wrong_footer(rows: list[str], expected_region_mode: bool) -> None:
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary(rows), INPUT, expected_region_mode=expected_region_mode)


@pytest.mark.parametrize("footer_index", [0, 1], ids=["whole-total", "region-total"])
@pytest.mark.parametrize("field_count", [1, 2, 3], ids=["name-only", "no-bases", "no-mean"])
def test_truncated_aggregate_rows_raise(footer_index: int, field_count: int) -> None:
    footers = [WHOLE_TOTAL, REGION_TOTAL]
    footers[footer_index] = "\t".join(footers[footer_index].split("\t")[:field_count])
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary([*REGION_ROWS, *footers]), INPUT)


@pytest.mark.parametrize("row_index", [0, 1, 4, 5], ids=["whole-row", "region-row", "whole-total", "region-total"])
@pytest.mark.parametrize("column", [1, 2], ids=["length", "bases"])
@pytest.mark.parametrize("count", ["-1", "1.5", "invalid"])
def test_counts_must_be_nonnegative_integers(row_index: int, column: int, count: str) -> None:
    rows = [*REGION_ROWS, WHOLE_TOTAL, REGION_TOTAL]
    fields = rows[row_index].split("\t")
    fields[column] = count
    rows[row_index] = "\t".join(fields)
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary(rows), INPUT)


@pytest.mark.parametrize("region_mode", [False, True], ids=["whole", "region"])
@pytest.mark.parametrize("column", [1, 2], ids=["length", "bases"])
def test_whole_total_must_equal_whole_rows(region_mode: bool, column: int) -> None:
    fields = WHOLE_TOTAL.split("\t")
    fields[column] = str(int(fields[column]) + 1)
    rows = [*(REGION_ROWS if region_mode else WHOLE_ROWS), "\t".join(fields)]
    if region_mode:
        rows.append(REGION_TOTAL)
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary(rows), INPUT)


@pytest.mark.parametrize("column", [1, 2], ids=["length", "bases"])
def test_region_total_must_equal_region_rows(column: int) -> None:
    fields = REGION_TOTAL.split("\t")
    fields[column] = str(int(fields[column]) + 1)
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary([*REGION_ROWS, WHOLE_TOTAL, "\t".join(fields)]), INPUT)


@pytest.mark.parametrize("region_mode", [False, True], ids=["whole", "region"])
def test_complete_empty_summaries_are_valid(region_mode: bool) -> None:
    rows = ["total\t0\t0\t0\t0\t0"]
    if region_mode:
        rows.append("total_region\t0\t0\t0\t0\t0")
    result = parse_mosdepth_summary(summary(rows), INPUT, expected_region_mode=region_mode)
    assert result.coverage_by_contig == []
    assert result.mean_depth == 0
    assert result.mean_depth_unweighted == 0


@pytest.mark.parametrize(
    ("region_rows", "region_total"),
    [
        (["chrA_region\t20\t300\t15\t15\t15", "chrB_region\t0\t0\t0\t0\t0"], REGION_TOTAL),
        (
            ["chrA_region\t200\t2000\t10\t10\t10", "chrB_region\t600\t600\t1\t1\t1"],
            "total_region\t800\t2600\t3.25\t1\t10",
        ),
    ],
    ids=["sparse-targets", "overlapping-targets"],
)
def test_region_totals_are_independent_of_whole_totals(region_rows: list[str], region_total: str) -> None:
    rows = [WHOLE_ROWS[0], region_rows[0], WHOLE_ROWS[1], region_rows[1], WHOLE_TOTAL, region_total]
    result = parse_mosdepth_summary(summary(rows), INPUT, expected_region_mode=True)
    assert [c.contig for c in result.coverage_by_contig] == ["chrA", "chrB"]
    assert result.mean_depth == pytest.approx(3.25)
    assert result.mean_depth_unweighted == pytest.approx(5.5)


def test_all_filtered_reads_allow_positive_length_and_zero_bases() -> None:
    rows = [
        "chrA\t100\t0\t0\t0\t0",
        "chrA_region\t100\t0\t0\t0\t0",
        "total\t100\t0\t0\t0\t0",
        "total_region\t100\t0\t0\t0\t0",
    ]
    result = parse_mosdepth_summary(summary(rows), INPUT, expected_region_mode=True)
    assert [(c.contig, c.length, c.mean_depth) for c in result.coverage_by_contig] == [("chrA", 100, 0)]


@pytest.mark.parametrize("region_mode", [False, True], ids=["whole", "region"])
def test_valid_real_contigs_named_like_aggregates(region_mode: bool) -> None:
    rows = ["total\t10\t20\t2\t2\t2"]
    if region_mode:
        rows.append("total_region\t10\t20\t2\t2\t2")
    rows.append("total_region\t30\t90\t3\t3\t3")
    if region_mode:
        rows.append("total_region_region\t30\t90\t3\t3\t3")
    rows.append("total\t40\t110\t2.75\t2\t3")
    if region_mode:
        rows.append("total_region\t40\t110\t2.75\t2\t3")
    result = parse_mosdepth_summary(summary(rows), INPUT, expected_region_mode=region_mode)
    assert [c.contig for c in result.coverage_by_contig] == ["total", "total_region"]
    assert result.mean_depth == pytest.approx(2.75)
    assert result.mean_depth_unweighted == pytest.approx(2.5)


def test_legacy_no_footer_and_reported_means_are_preserved() -> None:
    # Historical callers may omit the aggregate and have bases that do not
    # match length * mean. This change does not validate mean/rounding.
    rows = ["chrA\t100\t9000\t3", "chrB\t300\t5000\t12"]
    result = parse_mosdepth_summary(summary(rows), INPUT)
    assert [c.contig for c in result.coverage_by_contig] == ["chrA", "chrB"]
    assert result.mean_depth == pytest.approx(9.75)
    assert result.mean_depth_unweighted == pytest.approx(7.5)
    complete = parse_mosdepth_summary(summary([*rows, "total\t400\t14000\t35"]), INPUT)
    assert complete == result


def test_malformed_contig_numeric_error_identifies_input() -> None:
    with pytest.raises(ValueError, match=INPUT):
        parse_mosdepth_summary(summary(["chrA\t100\t1000\tinvalid", "total\t100\t1000\t10"]), INPUT)
