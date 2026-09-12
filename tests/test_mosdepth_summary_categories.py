"""Contig and aggregate row categories in mosdepth 0.3.14 summaries."""

import pytest

from ont_qc_mcp.parsers import parse_mosdepth_summary

HEADER = "chrom\tlength\tbases\tmean\tmin\tmax\n"
WHOLE = "chrA\t10000\t100000\t10.00\t10\t10\nchrB\t30000\t30000\t1.00\t1\t1\ntotal\t40000\t130000\t3.25\t1\t10\n"
WINDOWS = (
    "chrA\t10000\t100000\t10.00\t10\t10\n"
    "chrA_region\t10000\t100000\t10.00\t10\t10\n"
    "chrB\t30000\t30000\t1.00\t1\t1\n"
    "chrB_region\t30000\t30000\t1.00\t1\t1\n"
    "total\t40000\t130000\t3.25\t1\t10\n"
    "total_region\t40000\t130000\t3.25\t1\t10\n"
)
TARGETS = (
    "chrA\t10000\t100000\t10.00\t10\t10\n"
    "chrA_region\t100\t2000\t20.00\t20\t20\n"
    "chrB\t30000\t30000\t1.00\t1\t1\n"
    "chrB_region\t200\t0\t0.00\t0\t0\n"
    "total\t40000\t130000\t3.25\t1\t10\n"
    "total_region\t300\t2000\t6.67\t0\t20\n"
)


@pytest.mark.parametrize("rows", [WHOLE, WINDOWS, TARGETS], ids=["whole", "windows", "targets"])
def test_summary_metrics_use_only_real_contig_rows(rows: str) -> None:
    # WHOLE and WINDOWS reproduce a two-contig BAM probe; TARGETS makes region
    # means and lengths differ to expose accidental inclusion in weighted means.
    result = parse_mosdepth_summary(HEADER + rows, "reads.bam", threshold=4)
    assert [(c.contig, c.length, c.mean_depth) for c in result.coverage_by_contig] == [
        ("chrA", 10000, 10.0),
        ("chrB", 30000, 1.0),
    ]
    assert result.mean_depth == pytest.approx(3.25)
    assert result.mean_depth_unweighted == pytest.approx(5.5)
    assert [(r.contig, r.start, r.end, r.mean_depth) for r in result.low_coverage_regions] == [("chrB", 0, 30000, 1.0)]
    assert result.file == "reads.bam"
    assert result.coverage_distribution == []
    assert result.median_depth is None


@pytest.mark.parametrize("regions", [False, True], ids=["whole", "windows"])
def test_contig_names_can_match_summary_category_names(regions: bool) -> None:
    # Names and counts reproduce a BAM probe with collisions against both
    # aggregates, a generated region name, and the summary header prefix.
    names = ["total", "total_region", "chrA", "chrA_region", "chrom", "chromosome1"]
    rows = []
    for depth, name in enumerate(names, start=1):
        rows.append(f"{name}\t1000\t{depth * 1000}\t{depth}.00\t{depth}\t{depth}\n")
        if regions:
            rows.append(f"{name}_region\t1000\t{depth * 1000}\t{depth}.00\t{depth}\t{depth}\n")
    rows.append("total\t6000\t21000\t3.50\t1\t6\n")
    if regions:
        rows.append("total_region\t6000\t21000\t3.50\t1\t6\n")
    result = parse_mosdepth_summary(HEADER + "".join(rows), "reads.bam", threshold=3.5)
    assert [c.contig for c in result.coverage_by_contig] == names
    assert [c.mean_depth for c in result.coverage_by_contig] == [1, 2, 3, 4, 5, 6]
    assert result.mean_depth == pytest.approx(3.5)
    assert result.mean_depth_unweighted == pytest.approx(3.5)
    assert [r.contig for r in result.low_coverage_regions] == ["total", "total_region", "chrA"]


@pytest.mark.parametrize(
    "text",
    ["", HEADER, HEADER + "total\t0\t0\t0\t0\t0\n", HEADER + "total\t0\t0\t0\t0\t0\ntotal_region\t0\t0\t0\t0\t0\n"],
    ids=["empty", "header", "empty-whole", "empty-regions"],
)
def test_empty_summaries_have_no_contigs(text: str) -> None:
    result = parse_mosdepth_summary(text, "empty.bam", threshold=1)
    assert result.coverage_by_contig == []
    assert result.low_coverage_regions == []
    assert result.mean_depth == 0
    assert result.mean_depth_unweighted == 0


def test_legacy_four_column_summary_without_footer() -> None:
    text = "chrom\tlength\tbases\tmean\nchr1\t1000\t10000\t10\nchr2\t500\t2500\t5\n"
    result = parse_mosdepth_summary(text, "legacy.bam")
    assert [c.contig for c in result.coverage_by_contig] == ["chr1", "chr2"]
    assert result.mean_depth == pytest.approx(25 / 3)
    assert result.mean_depth_unweighted == pytest.approx(7.5)


@pytest.mark.parametrize(
    "rows",
    [
        "chrA\t100\t1000\t10\n",
        "chrA\t100\t1000\t10\nchrB_region\t100\t1000\t10\n",
        "chrA\t100\t1000\t10\nchrA_region\t100\n",
        "chrA_region\t100\t1000\t10\nchrA\t100\t1000\t10\n",
    ],
    ids=["missing-partner", "wrong-partner", "truncated-partner", "reversed-pair"],
)
def test_malformed_region_pairs_raise(rows: str) -> None:
    footer = "total\t100\t1000\t10\ntotal_region\t100\t1000\t10\n"
    with pytest.raises(ValueError, match="mosdepth.*region"):
        parse_mosdepth_summary(HEADER + rows + footer, "reads.bam")
