"""Shared API v2 region normalization."""

from pathlib import Path

import pytest

from ont_qc_mcp.config import ExecutionConfig
from ont_qc_mcp.v2_contracts import CoverageQCRequest
from ont_qc_mcp.v2_regions import MAX_REGIONS, normalize_regions


REFERENCE_LENGTHS = {"chr1": 100, "chr2": 50, "colon:contig": 25}


@pytest.mark.parametrize(
    "source",
    [
        [{"chrom": "chr1", "start": 9, "end": 20, "name": "target"}],
        {"format": "samtools", "values": ["chr1:10-20"]},
        {"format": "bed", "text": "chr1\t9\t20\ttarget\n"},
    ],
    ids=["explicit", "samtools", "bed-text"],
)
def test_equivalent_coordinate_forms_normalize_to_half_open(source: object) -> None:
    request = CoverageQCRequest.model_validate({"path": "reads.bam", "regions": source})
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS)
    assert [(r.chrom, r.start, r.end) for r in normalized.requested] == [("chr1", 9, 20)]
    assert normalized.requested[0].region_id == "region_1"


def test_bed_file_preserves_order_names_and_duplicates(tmp_path: Path) -> None:
    bed = tmp_path / "targets.bed"
    original = b"# note\ntrack name=targets\nchr1\t0\t10\tone\nchr1\t0\t10\ttwo\nchr2\t5\t8\n"
    bed.write_bytes(original)
    request = CoverageQCRequest.model_validate({"path": "reads.bam", "regions": {"format": "bed", "path": str(bed)}})
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS, exec_cfg=ExecutionConfig())
    assert [(r.region_id, r.name) for r in normalized.requested] == [
        ("region_1", "one"),
        ("region_2", "two"),
        ("region_3", None),
    ]
    assert [(r.chrom, r.start, r.end) for r in normalized.union] == [("chr1", 0, 10), ("chr2", 5, 8)]
    assert normalized.external_dependencies == (bed.resolve(),)
    assert bed.read_bytes() == original


@pytest.mark.parametrize("source_kind", ["bed", "gff3"])
def test_external_region_source_replacement_during_normalization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    from ont_qc_mcp import v2_regions

    if source_kind == "bed":
        source = tmp_path / "targets.bed"
        source.write_text("chr1\t0\t1\ttarget\n")
        request = CoverageQCRequest.model_validate(
            {"path": "reads.bam", "regions": {"format": "bed", "path": str(source)}}
        )
        original_bed_parser = v2_regions._bed_intervals

        def replace_after_read(*args, **kwargs):
            intervals = original_bed_parser(*args, **kwargs)
            replacement = tmp_path / "replacement.bed"
            replacement.write_text("chr1\t1\t2\treplacement\n")
            replacement.replace(source)
            return intervals

        monkeypatch.setattr(v2_regions, "_bed_intervals", replace_after_read)
    else:
        source = tmp_path / "genes.gff3"
        source.write_text("chr1\ttest\tgene\t1\t2\t.\t+\t.\tID=target\n")
        request = CoverageQCRequest.model_validate(
            {
                "path": "reads.bam",
                "regions": {"format": "gff3", "path": str(source), "feature_type": "gene"},
            }
        )
        original_gff_parser = v2_regions._all_gff_genes

        def replace_after_read(*args, **kwargs):
            genes = original_gff_parser(*args, **kwargs)
            replacement = tmp_path / "replacement.gff3"
            replacement.write_text("chr1\ttest\tgene\t2\t3\t.\t+\t.\tID=replacement\n")
            replacement.replace(source)
            return genes

        monkeypatch.setattr(v2_regions, "_all_gff_genes", replace_after_read)

    with pytest.raises(RuntimeError, match="changed during normalization"):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


def test_inline_bed_preserves_carriage_return_record_boundaries() -> None:
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {"format": "bed", "text": "chr1\t0\t1\tone\rchr2\t1\t2\ttwo\r"},
        }
    )
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS)
    assert [(region.chrom, region.start, region.end, region.name) for region in normalized.requested] == [
        ("chr1", 0, 1, "one"),
        ("chr2", 1, 2, "two"),
    ]


def test_union_merges_overlaps_and_adjacency_without_collapsing_requested_rows() -> None:
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": [
                {"chrom": "chr1", "start": 10, "end": 20, "name": "first"},
                {"chrom": "chr1", "start": 5, "end": 12, "name": "second"},
                {"chrom": "chr1", "start": 20, "end": 25, "name": "third"},
                {"chrom": "chr2", "start": 1, "end": 2},
            ],
        }
    )
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS)
    assert [r.region_id for r in normalized.requested] == ["region_1", "region_2", "region_3", "region_4"]
    assert [(r.chrom, r.start, r.end) for r in normalized.union] == [("chr1", 5, 25), ("chr2", 1, 2)]
    assert [[r.region_id for r in group.requested] for group in normalized.per_region()] == [
        ["region_1"],
        ["region_2"],
        ["region_3"],
        ["region_4"],
    ]


def test_samtools_contig_and_open_ranges_use_reference_lengths() -> None:
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {
                "format": "samtools",
                "values": ["chr1", "chr1:10", "chr1:-20", "chr1:90-", "colon:contig:2-3"],
            },
        }
    )
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS)
    assert [(r.chrom, r.start, r.end) for r in normalized.requested] == [
        ("chr1", 0, 100),
        ("chr1", 9, 100),
        ("chr1", 0, 20),
        ("chr1", 89, 100),
        ("colon:contig", 1, 3),
    ]


def test_gff3_gene_lookup_preserves_requested_id_order(tmp_path: Path) -> None:
    gff = tmp_path / "genes.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\ttest\tgene\t11\t20\t.\t+\t.\tID=g1;Name=FIRST\n"
        "chr2\ttest\tgene\t2\t4\t.\t-\t.\tID=g2;Name=SECOND\n"
    )
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {"format": "gff3", "path": str(gff), "feature_type": "gene", "ids": ["SECOND", "g1"]},
        }
    )
    normalized = normalize_regions(request.regions, REFERENCE_LENGTHS)
    assert [(r.chrom, r.start, r.end, r.name) for r in normalized.requested] == [
        ("chr2", 1, 4, "SECOND"),
        ("chr1", 10, 20, "g1"),
    ]
    assert normalized.external_dependencies == (gff.resolve(),)


def test_gff3_record_length_is_bounded_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_regions

    monkeypatch.setattr(v2_regions, "MAX_REGION_LINE_CHARS", 32)
    gff = tmp_path / "genes.gff3"
    gff.write_text("chr1\ttest\tgene\t1\t2\t.\t+\t.\tID=" + "x" * 33 + "\n")
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {"format": "gff3", "path": str(gff), "feature_type": "gene", "ids": ["target"]},
        }
    )
    with pytest.raises(ValueError, match="GFF3 record exceeds"):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


def test_reference_length_validation_observes_request_deadline() -> None:
    class ExpiringDeadline:
        def checkpoint(self) -> None:
            raise TimeoutError("request deadline exceeded")

    request = CoverageQCRequest.model_validate(
        {"path": "reads.bam", "regions": [{"chrom": "chr1", "start": 0, "end": 1}]}
    )
    with pytest.raises(TimeoutError, match="request deadline exceeded"):
        normalize_regions(
            request.regions,
            {"invalid\tchrom": 100},
            deadline=ExpiringDeadline(),  # type: ignore[arg-type]
        )


def test_blank_gff3_identifier_is_rejected_before_lookup(tmp_path: Path) -> None:
    gff = tmp_path / "genes.gff3"
    gff.write_text("chr1\ttest\tgene\t1\t2\t.\t+\t.\tNote=unnamed\n")
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {"format": "gff3", "path": str(gff), "feature_type": "gene", "ids": ["  "]},
        }
    )
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


def test_bed_file_region_limit_is_enforced_during_parsing(tmp_path: Path) -> None:
    bed = tmp_path / "too-many.bed"
    bed.write_text(
        "".join(f"chr1\t0\t1\tr{index}\n" for index in range(MAX_REGIONS + 1)) + "chr1\tnot-a-coordinate\t1\n"
    )
    request = CoverageQCRequest.model_validate({"path": "reads.bam", "regions": {"format": "bed", "path": str(bed)}})
    with pytest.raises(ValueError, match=f"1 to {MAX_REGIONS}"):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


def test_gff3_id_limit_is_checked_before_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ont_qc_mcp import v2_regions

    gff = tmp_path / "genes.gff3"
    gff.write_text("chr1\ttest\tgene\t1\t2\t.\t+\t.\tID=g1\n")
    request = CoverageQCRequest.model_validate(
        {
            "path": "reads.bam",
            "regions": {
                "format": "gff3",
                "path": str(gff),
                "feature_type": "gene",
                "ids": ["g1"] * (MAX_REGIONS + 1),
            },
        }
    )

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("lookup must not start before the ID count is validated")

    monkeypatch.setattr(v2_regions, "_selected_gff_genes", unexpected_lookup)
    with pytest.raises(ValueError, match=f"1 to {MAX_REGIONS}"):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"format": "bed", "text": "chr1\t0\t101\n"}, "exceeds reference length"),
        ({"format": "bed", "text": "missing\t0\t1\n"}, "absent from the reference"),
        ({"format": "bed", "text": "chr1\t０\t1\n"}, "ASCII decimal"),
        ({"format": "samtools", "values": ["chr1:0-1"]}, "one-based"),
        ({"format": "samtools", "values": ["chr1:20-10"]}, "start must not exceed end"),
    ],
)
def test_invalid_or_out_of_bounds_regions_fail(source: object, message: str) -> None:
    request = CoverageQCRequest.model_validate({"path": "reads.bam", "regions": source})
    with pytest.raises(ValueError, match=message):
        normalize_regions(request.regions, REFERENCE_LENGTHS)


def test_omitted_regions_mean_whole_scope() -> None:
    normalized = normalize_regions(None, REFERENCE_LENGTHS)
    assert normalized.requested == normalized.union == ()
    assert normalized.scope == "whole"
