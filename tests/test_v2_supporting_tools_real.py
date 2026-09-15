"""Pinned samtools evidence for the unregistered API v2 supporting adapters."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.v2_supporting_tools import header_info


pytestmark = pytest.mark.integration


def test_cram_header_info_uses_explicit_reference(tmp_path: Path) -> None:
    require_executable_tools(["samtools"])
    samtools = shutil.which("samtools")
    assert samtools is not None
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nACGTACGT\n")
    subprocess.run([samtools, "faidx", str(reference)], check=True, capture_output=True)
    sam = tmp_path / "reads.sam"
    sam.write_text("@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:8\nread1\t0\tchr1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n")
    cram = tmp_path / "reads.cram"
    subprocess.run(
        [samtools, "view", "-C", "-T", str(reference), "-o", str(cram), str(sam)],
        check=True,
        capture_output=True,
    )

    result = header_info(
        {"path": str(cram), "reference_path": str(reference)},
        tools=ToolPaths(samtools=samtools),
    )

    assert result.format == "cram"
    assert [(item.name, item.length) for item in result.references] == [("chr1", 8)]


def test_header_info_reads_bgzf_vcf_with_bgz_suffix(tmp_path: Path, synthetic_vcf: Path) -> None:
    bgz = tmp_path / "calls.vcf.bgz"
    shutil.copyfile(synthetic_vcf, bgz)

    result = header_info({"path": str(bgz)})

    assert result.format == "vcf"
    assert result.raw_header.startswith("##fileformat=VCF")
