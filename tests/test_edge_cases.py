import pytest

from ont_qc_mcp.parsers import parse_error_profile, parse_nanoq_json


def test_parse_nanoq_json_invalid_json():
    with pytest.raises(ValueError):
        parse_nanoq_json("not-json")


def test_parse_nanoq_json_negative_counts():
    payload = {"summary": {"reads": {"count": -1, "bases": 10}}}
    with pytest.raises(ValueError):
        parse_nanoq_json(payload)


def test_parse_error_profile_skips_malformed_gcd():
    sample = "\n".join(["GCD\t50\tbad\t", "GCD\t60\t10"])
    parsed = parse_error_profile(sample, file_path="align.bam")
    assert parsed.gc_coverage is None

