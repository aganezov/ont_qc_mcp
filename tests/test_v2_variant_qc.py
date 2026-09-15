"""API v2 variant grouping, native selection, and bcftools parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.regional_metrics import RegionalInterval
from ont_qc_mcp.utils import CommandResult
from ont_qc_mcp.v2_contracts import NormalizedInterval
from ont_qc_mcp.v2_execution import PipelineResult, RequestDeadline
from ont_qc_mcp.v2_regions import NormalizedRegionSet
from ont_qc_mcp.v2_variant_qc import _parse_bcftools_stats, resolve_variant_input, variant_qc


STATS = """\
SN\t0\tnumber of records:\t5
SN\t0\tnumber of SNPs:\t3
SN\t0\tnumber of MNPs:\t1
SN\t0\tnumber of indels:\t2
SN\t0\tnumber of others:\t1
TSTV\t0\t4\t2\t2.00\t3\t2\t1.50
"""


def test_stats_parser_uses_allele_counts_and_recomputes_tstv() -> None:
    group = _parse_bcftools_stats(STATS, {"general", "snps", "indels"})

    assert group.general is not None
    assert group.general.model_dump() == {"total_records": 5, "mnps": 1, "others": 1}
    assert group.snps is not None
    assert group.snps.model_dump() == {
        "count": 3,
        "transitions": 4,
        "transversions": 2,
        "ts_tv_ratio": 2.0,
    }
    assert group.indels is not None and group.indels.count == 2


def test_stats_parser_preserves_missing_tstv_and_metric_subsets() -> None:
    snps = _parse_bcftools_stats("SN\t0\tnumber of SNPs:\t2\n", {"snps"})
    assert snps.general is None and snps.indels is None
    assert snps.snps is not None
    assert snps.snps.model_dump() == {
        "count": 2,
        "transitions": None,
        "transversions": None,
        "ts_tv_ratio": None,
    }

    general = _parse_bcftools_stats(
        "SN\t0\tnumber of records:\t0\nSN\t0\tnumber of MNPs:\t0\nSN\t0\tnumber of others:\t0\n",
        {"general"},
    )
    assert general.general is not None and general.general.total_records == 0
    assert general.snps is None and general.indels is None


def test_stats_parser_rejects_incomplete_requested_summary() -> None:
    with pytest.raises(ValueError, match="mnps, others"):
        _parse_bcftools_stats("SN\t0\tnumber of records:\t2\n", {"general"})


def _install_fake_variant_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ont_qc_mcp import v2_variant_qc as module

    vcf = tmp_path / "calls.vcf.gz"
    vcf.write_bytes(b"VCF")
    index = tmp_path / "calls.vcf.gz.tbi"
    index.write_bytes(b"index")
    variant = SimpleNamespace(
        path=vcf.resolve(),
        access_path=vcf.resolve(),
        index=index.resolve(),
        reference=None,
        reference_index=None,
        reference_access_path=None,
        assert_unchanged=lambda: None,
    )
    requested = (
        NormalizedInterval(chrom="chr1", start=0, end=5, name="first", region_id="region_1"),
        NormalizedInterval(chrom="chr1", start=3, end=8, name="overlap", region_id="region_2"),
    )
    regions = NormalizedRegionSet(
        requested=requested,
        union=(RegionalInterval(chrom="chr1", start=0, end=8),),
    )
    commands: list[tuple[str, ...]] = []
    region_files: list[Path] = []
    region_contents: list[str] = []
    deadlines: list[RequestDeadline] = []

    monkeypatch.setattr(module, "resolve_variant_input", lambda *args, **kwargs: variant)
    monkeypatch.setattr(module, "read_variant_reference_lengths", lambda *args, **kwargs: {"chr1": 10})
    monkeypatch.setattr(module, "normalize_regions", lambda *args, **kwargs: regions)

    def run(stages, deadline, **kwargs):
        command = stages[0].command
        commands.append(command)
        deadlines.append(deadline)
        if "--regions-file" in command:
            region_path = Path(command[command.index("--regions-file") + 1])
            region_contents.append(region_path.read_text())
            region_files.append(region_path)
        elif "--regions" in command:
            region_path = Path(command[command.index("--regions") + 1])
            assert region_path.suffix == ".bed"
            region_files.append(region_path)
        return PipelineResult((CommandResult(command, 0, STATS, ""),))

    monkeypatch.setattr(module, "run_pipeline", run)
    return vcf, commands, region_files, region_contents, deadlines


def test_combined_regions_use_one_overlap_safe_union_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vcf, commands, region_files, region_contents, deadlines = _install_fake_variant_backend(monkeypatch, tmp_path)

    result = variant_qc(
        {
            "path": str(vcf),
            "regions": [{"chrom": "chr1", "start": 0, "end": 5}, {"chrom": "chr1", "start": 3, "end": 8}],
            "selection": {"include_expression": "QUAL>=20"},
            "extra_args": {"bcftools_stats": ["--samples", "S1,S2", "--samples=S3"]},
        },
        tools=ToolPaths(bcftools="bcftools"),
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == ("bcftools", "stats")
    assert command.count("--include") == 1
    assert command[command.index("--include") + 1] == "QUAL>=20"
    assert "--regions-file" in command and command[command.index("--regions-overlap") + 1] == "1"
    assert region_contents == ["chr1\t0\t8\n"]
    assert command[-4:] == ("--samples", "S1,S2", "--samples=S3", str(vcf.resolve()))
    assert result.resolved_group_by == "combined"
    assert len(result.results) == 1
    assert result.effective_request.normalized_regions[1].name == "overlap"
    assert result.provenance[0].effective_args == list(command[1:-1])
    assert result.provenance[0].native_options_used is True
    assert len({id(deadline) for deadline in deadlines}) == 1
    assert all(not path.parent.exists() for path in region_files)


def test_region_grouping_preserves_requested_order_and_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vcf, commands, region_files, region_contents, _ = _install_fake_variant_backend(monkeypatch, tmp_path)

    result = variant_qc(
        {
            "path": str(vcf),
            "regions": [
                {"chrom": "chr1", "start": 0, "end": 5, "name": "first"},
                {"chrom": "chr1", "start": 3, "end": 8, "name": "overlap"},
            ],
            "group_by": "region",
            "metrics": ["general", "snps"],
        },
        tools=ToolPaths(bcftools="bcftools"),
    )

    assert len(commands) == 2
    assert [group.region_id for group in result.results] == ["region_1", "region_2"]
    assert [group.region_name for group in result.results] == ["first", "overlap"]
    assert region_contents == ["chr1\t0\t5\n", "chr1\t3\t8\n"]
    assert all(
        group.general is not None and group.snps is not None and group.indels is None for group in result.results
    )
    assert [path.read_text() if path.exists() else None for path in region_files] == [None, None]
    assert all(not path.parent.exists() for path in region_files)


def test_whole_file_does_not_require_header_or_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_variant_qc as module

    vcf = tmp_path / "calls.vcf"
    vcf.write_text("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    variant = SimpleNamespace(
        path=vcf.resolve(),
        access_path=vcf.resolve(),
        index=None,
        reference=None,
        reference_index=None,
        reference_access_path=None,
        assert_unchanged=lambda: None,
    )
    monkeypatch.setattr(module, "resolve_variant_input", lambda *args, **kwargs: variant)
    monkeypatch.setattr(
        module,
        "read_variant_reference_lengths",
        lambda *args, **kwargs: pytest.fail("whole-file request read the header"),
    )
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda stages, deadline, **kwargs: PipelineResult((CommandResult(stages[0].command, 0, STATS, ""),)),
    )

    result = variant_qc({"path": str(vcf), "metrics": ["indels"]}, tools=ToolPaths(bcftools="bcftools"))

    assert result.effective_request.region_scope == "whole_file"
    assert result.results[0].general is None and result.results[0].snps is None
    assert result.results[0].indels is not None and result.results[0].indels.count == 2


def test_whole_file_reference_is_validated_against_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_variant_qc as module

    vcf = tmp_path / "calls.vcf"
    vcf.write_text("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nA\n")
    variant = SimpleNamespace(
        path=vcf.resolve(),
        access_path=vcf.resolve(),
        index=None,
        reference=reference.resolve(),
        reference_index=Path(str(reference.resolve()) + ".fai"),
        reference_access_path=reference.resolve(),
        assert_unchanged=lambda: None,
    )
    header_reads: list[RequestDeadline] = []
    monkeypatch.setattr(module, "resolve_variant_input", lambda *args, **kwargs: variant)

    def read_lengths(*args, **kwargs):
        header_reads.append(args[-1])
        return {"chr1": 1}

    monkeypatch.setattr(module, "read_variant_reference_lengths", read_lengths)
    monkeypatch.setattr(
        module,
        "run_pipeline",
        lambda stages, deadline, **kwargs: PipelineResult((CommandResult(stages[0].command, 0, STATS, ""),)),
    )

    result = variant_qc(
        {"path": str(vcf), "reference_path": str(reference)},
        tools=ToolPaths(bcftools="bcftools"),
    )

    assert len(header_reads) == 1
    args = result.provenance[0].effective_args
    assert args[args.index("--fasta-ref") + 1] == str(reference.resolve())


def test_regional_input_requires_compressed_indexed_variant(tmp_path: Path) -> None:
    plain = tmp_path / "calls.vcf"
    plain.write_text("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    compressed = tmp_path / "calls.vcf.gz"
    compressed.write_bytes(b"compressed")

    with pytest.raises(ValueError, match="requires indexed VCF.gz"):
        resolve_variant_input(str(plain), reference_path=None, require_index=True)
    with pytest.raises(FileNotFoundError, match="existing .* index"):
        resolve_variant_input(str(compressed), reference_path=None, require_index=True)


def test_reference_requires_existing_fasta_index(tmp_path: Path) -> None:
    variant = tmp_path / "calls.vcf"
    variant.write_text("##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nACGT\n")

    with pytest.raises(FileNotFoundError, match="existing .*fai.* index"):
        resolve_variant_input(str(variant), reference_path=str(reference), require_index=False)


def test_external_region_source_must_remain_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_variant_qc as module

    vcf, _, _, _, _ = _install_fake_variant_backend(monkeypatch, tmp_path)
    bed = tmp_path / "targets.bed"
    bed.write_text("chr1\t0\t5\n")
    regions = NormalizedRegionSet(
        requested=(NormalizedInterval(chrom="chr1", start=0, end=5, region_id="region_1"),),
        union=(RegionalInterval(chrom="chr1", start=0, end=5),),
        external_dependencies=(bed.resolve(),),
    )
    monkeypatch.setattr(module, "normalize_regions", lambda *args, **kwargs: regions)

    def mutate_region_source(stages, deadline, **kwargs):
        bed.write_text("chr1\t1\t5\n")
        return PipelineResult((CommandResult(stages[0].command, 0, STATS, ""),))

    monkeypatch.setattr(module, "run_pipeline", mutate_region_source)

    with pytest.raises(RuntimeError, match="region source changed"):
        variant_qc(
            {"path": str(vcf), "regions": {"format": "bed", "path": str(bed)}},
            tools=ToolPaths(bcftools="bcftools"),
        )
