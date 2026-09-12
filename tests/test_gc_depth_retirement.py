"""Retired GCD output stays null without changing other samtools statistics."""

import pytest

from ont_qc_mcp.parsers import parse_error_profile


@pytest.mark.parametrize(
    "gcd",
    [
        "GCD\t40\t100\t2\t3\t4\t5\t6\n",  # integer-spelled quantiles previously became counts
        "GCD\t40.5\t100\t0.25\t0.5\t1.25\t2.5\t3.5\n",
        "GCD\t0\t0\t0\t0\t0\t0\t0\n",
        "GCD\tinvalid\t100\t2\n",
        "GCD\t40\n",
        "",
    ],
    ids=["integer", "fractional", "zero", "malformed", "truncated", "absent"],
)
def test_gc_coverage_is_retired_and_other_sections_preserved(gcd):
    text = (
        "SN\terror rate:\t0.023\t# NM-derived fallback\n"
        "COV\t[20-24]\t24\t50\n"
        "MPC\t2\t3\t10\t0\n"
        "IS\t100\t5\t0\t0\t5\n" + gcd
    )
    stats = parse_error_profile(text, "reads.bam")
    payload = stats.model_dump()
    assert "gc_coverage" in payload
    assert payload["gc_coverage"] is None
    assert stats.mismatch_rate == pytest.approx(0.023)
    assert payload["coverage_histogram"] == [{"start": 20, "end": 24, "count": 50}]
    assert payload["mismatch_counts_by_cycle"] == [{"cycle": 2, "n_count": 3, "mismatches_by_quality": [10, 0]}]
    assert payload["mismatch_by_cycle"] is None
    assert payload["insert_size_histogram"] == [{"start": 100.0, "end": 100.0, "count": 5}]
