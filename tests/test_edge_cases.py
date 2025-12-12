import pytest

from ont_qc_mcp.parsers import parse_error_profile, parse_nanoq_json
from ont_qc_mcp.config import ExecutionConfig
from ont_qc_mcp.tools import _EXEC_CFG, _validate_input_file


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


def test_validate_input_accepts_bgz_fastq(tmp_path):
    fq_path = tmp_path / "reads.fq.bgz"
    fq_path.write_text("@r1\nACGT\n+\n!!!!\n")
    _validate_input_file(
        fq_path,
        _EXEC_CFG,
        allowed_exts=(".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.bgz", ".fq.bgz"),
    )


def test_validate_input_file_rejects_oversized(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_MAX_FILE_MB", "1")
    cfg = ExecutionConfig()
    large_file = tmp_path / "large.fastq"
    large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    with pytest.raises(ValueError, match="exceeds configured size limit"):
        _validate_input_file(large_file, cfg)
