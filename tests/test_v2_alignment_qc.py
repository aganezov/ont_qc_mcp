"""API v2 alignment metric grouping and backend routing."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ont_qc_mcp.v2_alignment_qc import _error_section, _identity_section, _record_groups, alignment_qc
from ont_qc_mcp.v2_alignment_records import collect_alignment_records
from ont_qc_mcp.v2_contracts import AlignmentQCGroup, ErrorProfileSection, IdentitySection, Provenance
from ont_qc_mcp.v2_regions import NormalizedRegionSet
from ont_qc_mcp.v2_samtools_stats_filter import filter_error_profile_lines
from ont_qc_mcp.regional_metrics import RegionalInterval
from ont_qc_mcp.v2_contracts import AlignmentQCRequest, NormalizedInterval


SAM = (
    "@SQ\tSN:chr1\tLN:20\n"
    "known\t0\tchr1\t1\t30\t4M\t*\t0\t0\tAAAA\tIIII\tNM:i:1\n"
    "missing\t2048\tchr1\t3\t255\t4M\t*\t0\t0\tCCCC\t*\tNM:i:0\n"
    "unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tGGG\tIII\n"
)


def test_whole_record_counts_mapq_and_missing_quality_denominators() -> None:
    [group] = collect_alignment_records(
        StringIO(SAM),
        regions=[],
        group_by="combined",
        include_base_quality=True,
    )

    assert group["counts"] == {
        "eligible_records": 3,
        "mapped_records": 2,
        "unmapped_records": 1,
        "secondary_records": 0,
        "supplementary_records": 1,
    }
    assert group["mapping_quality"] == {"known_records": 2, "missing_records": 1, "mean_mapq": 15}
    assert group["aligned_base_quality"] == {
        "aligned_query_bases": 8,
        "known_quality_bases": 4,
        "missing_quality_bases": 4,
        "mean_base_quality": 40,
    }


def test_quality_sentinel_is_ignored_when_quality_metric_is_not_requested() -> None:
    ambiguous = "one\t0\tchr1\t1\t30\t1M\t*\t0\t0\tA\t*\n"
    [group] = collect_alignment_records(
        StringIO(ambiguous),
        regions=[],
        group_by="combined",
        include_base_quality=False,
    )
    counts = group["counts"]
    assert isinstance(counts, dict)
    assert counts["eligible_records"] == 1

    with pytest.raises(ValueError, match="cannot distinguish one-base Q9"):
        collect_alignment_records(
            StringIO(ambiguous),
            regions=[],
            group_by="combined",
            include_base_quality=True,
        )

    [regional] = collect_alignment_records(
        StringIO(ambiguous),
        regions=[RegionalInterval(chrom="chr1", start=0, end=2)],
        group_by="region",
        include_base_quality=False,
    )
    assert regional["counts"] == {
        "eligible_records": 1,
        "mapped_records": 1,
        "unmapped_records": 0,
        "secondary_records": 0,
        "supplementary_records": 0,
    }


def test_combined_disjoint_union_counts_a_bridging_record_once_and_bases_inside_union() -> None:
    sam = "bridge\t0\tchr1\t1\t20\t6M\t*\t0\t0\tAAAAAA\tIIIIII\n"
    [combined] = collect_alignment_records(
        StringIO(sam),
        regions=[
            RegionalInterval(chrom="chr1", start=0, end=2),
            RegionalInterval(chrom="chr1", start=4, end=6),
        ],
        group_by="combined",
        include_base_quality=True,
    )
    assert combined["counts"] == {
        "eligible_records": 1,
        "mapped_records": 1,
        "unmapped_records": 0,
        "secondary_records": 0,
        "supplementary_records": 0,
    }
    assert combined["aligned_base_quality"] == {
        "aligned_query_bases": 4,
        "known_quality_bases": 4,
        "missing_quality_bases": 0,
        "mean_base_quality": 40,
    }


def test_native_sections_preserve_distinct_identity_and_error_meanings() -> None:
    identity = _identity_section(
        '{"file_info":{"path":"-"},"alignment_stats":{"num_reads":2},'
        '"read_stats":{"mean_length":4,"median_length":4,"n50":4},'
        '"identity_stats":{"mean_identity":87.5}}'
    )
    assert identity.mean_identity == pytest.approx(0.875)

    error = _error_section(
        "SN\terror rate:\t0.125\t# NM-derived\n"
        "SN\tmismatches per base:\t0.05\t# substitutions\n"
        "SN\tinsertions per base:\t0.02\n"
        "SN\tdeletions per base:\t0.01\n"
    )
    assert error.nm_error_rate == pytest.approx(0.125)
    assert error.mismatch_rate == pytest.approx(0.05)
    assert error.insertion_rate == pytest.approx(0.02)
    assert error.deletion_rate == pytest.approx(0.01)


def test_stats_filter_keeps_only_rows_represented_by_the_error_contract() -> None:
    source = StringIO(
        "SN\terror rate:\t0.1\nSN\traw total sequences:\t10\nCOV\t[1-1]\t1\t5\nMPC\t1\t0\t1\nIS\t10\t2\nGCC\tignored\n"
    )
    destination = StringIO()
    filter_error_profile_lines(source, destination)
    assert destination.getvalue() == ("SN\terror rate:\t0.1\nCOV\t[1-1]\t1\t5\nMPC\t1\t0\t1\nIS\t10\t2\n")


def test_only_requested_native_backend_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ont_qc_mcp import v2_alignment_qc as module

    alignment = SimpleNamespace(alignment=tmp_path / "reads.bam", reference=None)
    regions = NormalizedRegionSet((), ())
    calls: list[str] = []
    monkeypatch.setattr(module, "resolve_and_normalize_regions", lambda *args, **kwargs: (alignment, regions))
    monkeypatch.setattr(module, "_record_groups", lambda *args, **kwargs: pytest.fail("record backend ran"))

    def identity(*args, **kwargs):
        calls.append("identity")
        return IdentitySection(mean_identity=0.9), [
            Provenance(backend="cramino", measurement_scope="whole selected alignment records")
        ]

    def errors(*args, **kwargs):
        calls.append("errors")
        return ErrorProfileSection(nm_error_rate=0.1), [
            Provenance(backend="samtools_stats", measurement_scope="whole selected alignment records")
        ]

    monkeypatch.setattr(module, "_run_identity", identity)
    monkeypatch.setattr(module, "_run_error_profile", errors)

    response = alignment_qc({"path": str(alignment.alignment), "metrics": ["identity"]})
    assert calls == ["identity"]
    assert response.results == [AlignmentQCGroup(identity=IdentitySection(mean_identity=0.9))]
    assert [value.backend for value in response.provenance] == ["cramino"]


def test_late_backend_failure_returns_no_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ont_qc_mcp import v2_alignment_qc as module

    alignment = SimpleNamespace(alignment=tmp_path / "reads.bam", reference=None)
    regions = NormalizedRegionSet((), ())
    group = AlignmentQCGroup(identity=IdentitySection(mean_identity=0.9))
    monkeypatch.setattr(module, "resolve_and_normalize_regions", lambda *args, **kwargs: (alignment, regions))
    monkeypatch.setattr(
        module,
        "_run_identity",
        lambda *args, **kwargs: (
            group.identity,
            [Provenance(backend="cramino", measurement_scope="whole selected alignment records")],
        ),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("samtools stats failed")

    monkeypatch.setattr(module, "_run_error_profile", fail)
    with pytest.raises(RuntimeError, match="samtools stats failed"):
        alignment_qc({"path": str(alignment.alignment), "metrics": ["identity", "error_profile"]})


@pytest.mark.parametrize("failure", [RuntimeError("selection failed"), asyncio.CancelledError()])
def test_record_metric_failure_and_cancellation_remove_region_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    from ont_qc_mcp import v2_alignment_qc as module

    regions = NormalizedRegionSet(
        requested=(NormalizedInterval(chrom="chr1", start=0, end=5, region_id="region_1"),),
        union=(),
    )
    request = AlignmentQCRequest.model_validate(
        {
            "path": str(tmp_path / "reads.bam"),
            "regions": [{"chrom": "chr1", "start": 0, "end": 5}],
            "group_by": "region",
        }
    )
    region_paths: list[Path] = []

    def fail(*args, **kwargs):
        stages = args[-1]
        command = stages[-1].command
        region_paths.append(Path(command[command.index("--regions") + 1]))
        raise failure

    monkeypatch.setattr(module, "_run_selected_pipeline", fail)
    with pytest.raises(type(failure)):
        _record_groups(
            cast(Any, SimpleNamespace()),
            regions,
            request,
            cast(Any, SimpleNamespace(samtools="samtools")),
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
        )

    assert len(region_paths) == 1
    assert not region_paths[0].parent.exists()
