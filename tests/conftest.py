import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_bam() -> Path:
    """
    Sample BAM for workflow tests.

    Defaults to the checked-in real fixture; can be overridden with MCP_SAMPLE_BAM.
    """
    env_path = os.environ.get("MCP_SAMPLE_BAM")
    default_path = Path(__file__).resolve().parent / "fixtures" / "real" / "haplotag.large.bam"
    bam_path = Path(env_path) if env_path else default_path

    if not bam_path.exists():
        pytest.fail(f"Sample BAM missing: {bam_path} (set MCP_SAMPLE_BAM to override)")
    return bam_path


@pytest.fixture
def sample_vcf() -> Path:
    """
    Sample VCF for header/metadata tests.

    Uses the checked-in real gzipped VCF; can be overridden with MCP_SAMPLE_VCF.
    """
    env_path = os.environ.get("MCP_SAMPLE_VCF")
    default_path = Path(__file__).resolve().parent / "fixtures" / "real" / "haplotag.large.vcf.gz"
    vcf_path = Path(env_path) if env_path else default_path

    if not vcf_path.exists():
        pytest.fail(f"Sample VCF missing: {vcf_path} (set MCP_SAMPLE_VCF to override)")
    return vcf_path


@pytest.fixture
def sample_fastq() -> Path:
    """
    Sample FASTQ for read-level tests.

    Uses the checked-in real gzipped FASTQ; can be overridden with MCP_SAMPLE_FASTQ.
    """
    env_path = os.environ.get("MCP_SAMPLE_FASTQ")
    default_path = Path(__file__).resolve().parent / "fixtures" / "real" / "haplotag.large.fq.gz"
    fastq_path = Path(env_path) if env_path else default_path

    if not fastq_path.exists():
        pytest.fail(f"Sample FASTQ missing: {fastq_path} (set MCP_SAMPLE_FASTQ to override)")
    return fastq_path


@pytest.fixture
def mcp_server_params():
    """Return StdioServerParameters for launching the MCP server."""
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ont_qc_mcp.app_server"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
