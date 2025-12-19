import os
import subprocess
import sys
from pathlib import Path
from shutil import which

import pytest

# Ensure the Python interpreter's bin directory is on PATH so integration tests
# can find bundled CLI tools (e.g., conda env bin/) even when the env is not activated.
try:
    _PYTHON_BIN: Path | None = Path(sys.executable).resolve().parent
except OSError:  # pragma: no cover
    _PYTHON_BIN = None
if _PYTHON_BIN and _PYTHON_BIN.exists():
    os.environ["PATH"] = f"{_PYTHON_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

# ---------------------------------------------------------------------------
# Synthetic fixture directory
# ---------------------------------------------------------------------------
SYNTHETIC_DIR = Path(__file__).resolve().parent / "fixtures" / "synthetic"

# ---------------------------------------------------------------------------
# Tool verification cache (to avoid repeated subprocess calls)
# ---------------------------------------------------------------------------
_TOOL_EXECUTABLE_CACHE: dict[str, bool] = {}

# Version flags for each tool (used to verify executability)
_TOOL_VERSION_FLAGS: dict[str, list[str]] = {
    "nanoq": ["--version"],
    "cramino": ["--version"],
    "mosdepth": ["--version"],
    "chopper": ["--version"],
    "samtools": ["--version"],
    "bcftools": ["--version"],
}


def is_tool_executable(tool_name: str) -> bool:
    """
    Check if a tool is both available on PATH and actually executable.

    This catches issues like:
    - Tool binary exists but is for wrong architecture (Exec format error)
    - Tool binary exists but has missing dependencies
    - Tool binary exists but has permission issues

    Results are cached to avoid repeated subprocess calls.
    """
    if tool_name in _TOOL_EXECUTABLE_CACHE:
        return _TOOL_EXECUTABLE_CACHE[tool_name]

    # First check if tool exists on PATH
    tool_path = which(tool_name)
    if tool_path is None:
        _TOOL_EXECUTABLE_CACHE[tool_name] = False
        return False

    # Try to actually run the tool to verify it's executable
    version_args = _TOOL_VERSION_FLAGS.get(tool_name, ["--version"])
    try:
        result = subprocess.run(
            [tool_path, *version_args],
            capture_output=True,
            timeout=10,
        )
        # Most tools return 0 for --version, but some may return non-zero
        # We consider it executable if it doesn't raise an exception
        version_in_output = b"version" in result.stdout.lower() or b"version" in result.stderr.lower()
        is_executable = result.returncode == 0 or version_in_output
        _TOOL_EXECUTABLE_CACHE[tool_name] = is_executable
        return is_executable
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        # OSError includes Exec format error (errno 8)
        _TOOL_EXECUTABLE_CACHE[tool_name] = False
        return False


def require_executable_tools(tools: list[str]) -> None:
    """
    Skip test if any of the specified tools are not executable.

    This is more robust than just checking PATH - it verifies the tool
    can actually run, catching architecture mismatches and other issues.
    """
    non_executable = [tool for tool in tools if not is_tool_executable(tool)]
    if non_executable:
        pytest.skip(f"Tools not executable (missing or wrong architecture): {', '.join(non_executable)}")


def _ensure_qc_fixtures_exist() -> None:
    """
    Ensure QC synthetic fixtures exist, generating them on-demand if missing.

    This allows tests to run even if the generated fixtures are not committed,
    as long as the required CLI tools (bgzip, tabix) are available.
    """
    import importlib.util

    # Load generator module dynamically to avoid mypy module resolution issues
    generator_path = SYNTHETIC_DIR / "generate_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_data", generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generator module from {generator_path}")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Sequencing summary
    seq_summary_path = SYNTHETIC_DIR / "sequencing_summary_mock.txt"
    if not seq_summary_path.exists():
        generator.generate_sequencing_summary(seq_summary_path)

    # VCF (plain, gzipped, and indexed)
    vcf_path = SYNTHETIC_DIR / "tiny.vcf"
    vcf_gz_path = SYNTHETIC_DIR / "tiny.vcf.gz"
    vcf_tbi_path = SYNTHETIC_DIR / "tiny.vcf.gz.tbi"
    if not vcf_path.exists():
        generator.generate_vcf(vcf_path)
    if not vcf_gz_path.exists() or not vcf_tbi_path.exists():
        generator.vcf_to_gzip_and_index(vcf_path)

    # GFF3
    gff3_path = SYNTHETIC_DIR / "genes_mock.gff3"
    if not gff3_path.exists():
        generator.generate_gff3(gff3_path)

    # BED files
    valid_bed_path = SYNTHETIC_DIR / "valid.bed"
    invalid_bed_path = SYNTHETIC_DIR / "invalid.bed"
    if not valid_bed_path.exists() or not invalid_bed_path.exists():
        generator.generate_bed_files(valid_bed_path, invalid_bed_path)


# Run fixture generation at module import time (before any tests run)
# This is wrapped in try/except to allow tests to fail gracefully if
# required tools (bgzip, tabix) are not available.
try:
    _ensure_qc_fixtures_exist()
except SystemExit as e:
    # Tool not available - tests that need these fixtures will fail
    import warnings

    warnings.warn(f"Could not generate QC fixtures: {e}")


# ---------------------------------------------------------------------------
# QC Synthetic Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_sequencing_summary() -> Path:
    """Synthetic ONT sequencing summary file."""
    path = SYNTHETIC_DIR / "sequencing_summary_mock.txt"
    if not path.exists():
        pytest.fail(f"Fixture missing: {path} (ensure bgzip/tabix are available)")
    return path


@pytest.fixture
def synthetic_vcf() -> Path:
    """Synthetic VCF file (gzipped with tabix index)."""
    path = SYNTHETIC_DIR / "tiny.vcf.gz"
    if not path.exists():
        pytest.fail(f"Fixture missing: {path} (ensure bgzip/tabix are available)")
    return path


@pytest.fixture
def synthetic_bed_valid() -> Path:
    """Synthetic valid BED file."""
    path = SYNTHETIC_DIR / "valid.bed"
    if not path.exists():
        pytest.fail(f"Fixture missing: {path}")
    return path


@pytest.fixture
def synthetic_bed_invalid() -> Path:
    """Synthetic invalid BED file (with deliberate format errors)."""
    path = SYNTHETIC_DIR / "invalid.bed"
    if not path.exists():
        pytest.fail(f"Fixture missing: {path}")
    return path


@pytest.fixture
def synthetic_gff3() -> Path:
    """Synthetic GFF3 gene annotation file."""
    path = SYNTHETIC_DIR / "genes_mock.gff3"
    if not path.exists():
        pytest.fail(f"Fixture missing: {path}")
    return path


# ---------------------------------------------------------------------------
# Real Data Fixtures
# ---------------------------------------------------------------------------
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
def sample_bam_highdepth() -> Path:
    """
    High-depth synthetic BAM for coverage-focused tests.
    """
    bam_path = Path(__file__).resolve().parent / "fixtures" / "synthetic" / "highdepth.bam"
    if not bam_path.exists():
        pytest.fail(f"High-depth BAM missing: {bam_path}")
    return bam_path


@pytest.fixture
def sample_reference() -> Path:
    """
    Synthetic reference FASTA for custom genome testing.
    """
    ref_path = Path(__file__).resolve().parent / "fixtures" / "synthetic" / "highdepth.fa"
    if not ref_path.exists():
        pytest.fail(f"Reference FASTA missing: {ref_path}")
    return ref_path


@pytest.fixture
def require_container_runtime():
    """
    Ensure a container runtime is available for IGV integration tests.
    Skips the test if no Docker/Apptainer is detected.
    """
    from ont_qc_mcp.cli_wrappers import detect_container_runtime
    from ont_qc_mcp.config import ToolPaths

    runtime = detect_container_runtime(ToolPaths())
    if runtime is None:
        pytest.skip("No container runtime (Docker/Apptainer) available")
    return runtime


@pytest.fixture
def mcp_server_params():
    """Return StdioServerParameters for launching the MCP server."""
    from mcp.client.stdio import StdioServerParameters

    env = {"MCP_STDIO_TRANSPORT": "compat"}
    if (sif := os.getenv("MCP_IGV_SIF_PATH")):
        env["MCP_IGV_SIF_PATH"] = sif
    for key in ("APPTAINER", "APPTAINER_CACHEDIR", "APPTAINER_TMPDIR", "APPTAINER_DISABLE_CACHE"):
        if (value := os.getenv(key)):
            env[key] = value

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ont_qc_mcp.app_server"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
