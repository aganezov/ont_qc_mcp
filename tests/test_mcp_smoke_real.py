"""Tests for the real-input MCP smoke script's public request wiring."""

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "mcp_smoke_real.py"
SPEC = importlib.util.spec_from_file_location("mcp_smoke_real_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_build_calls = MODULE._build_calls


def _calls_by_name(*, alignment: Path, reference: Path | None = None):
    return {
        call.name: call.arguments
        for call in _build_calls(
            fastq=None,
            bam=alignment,
            include_error_profile=False,
            reference=reference,
        )
    }


def test_bam_smoke_calls_do_not_invent_a_reference() -> None:
    calls = _calls_by_name(alignment=Path("reads.bam"))
    assert calls["alignment_qc"] == {"path": "reads.bam", "metrics": ["counts", "mapping_quality"]}
    assert calls["coverage_qc"] == {"path": "reads.bam"}


def test_cram_smoke_calls_forward_the_explicit_reference() -> None:
    calls = _calls_by_name(alignment=Path("reads.cram"), reference=Path("reference.fa"))
    assert calls["alignment_qc"] == {
        "path": "reads.cram",
        "reference_path": "reference.fa",
        "metrics": ["counts", "mapping_quality"],
    }
    assert calls["coverage_qc"] == {"path": "reads.cram", "reference_path": "reference.fa"}
