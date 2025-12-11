import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_bam() -> Path:
    """
    Optional sample BAM path for workflow tests.

    Provide MCP_SAMPLE_BAM env var to enable workflow smoke tests.
    """
    env_path = os.environ.get("MCP_SAMPLE_BAM")
    if not env_path:
        pytest.skip("Set MCP_SAMPLE_BAM to a BAM/CRAM path to run workflow tests.")
    bam_path = Path(env_path)
    if not bam_path.exists():
        pytest.skip(f"MCP_SAMPLE_BAM does not exist: {bam_path}")
    return bam_path


@pytest.fixture
def mcp_server_params():
    """Return StdioServerParameters for launching the MCP server."""
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp.app_server"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
