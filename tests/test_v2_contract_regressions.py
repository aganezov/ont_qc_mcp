"""Negative controls for the remaining V2-01 review findings."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ont_qc_mcp import v2_contracts as v2

FIXTURES = Path(__file__).parent / "fixtures"
MODELS: dict[str, type[BaseModel]] = {
    "read_qc": v2.ReadQCResponse,
    "alignment_qc": v2.AlignmentQCResponse,
    "coverage_qc": v2.CoverageQCResponse,
    "variant_qc": v2.VariantQCResponse,
}


def response(tool, grouping):
    entries = []
    for name in ("responses.json", "response-examples.json"):
        entries.extend(json.loads((FIXTURES / "api_v2" / name).read_text()))
    return deepcopy(
        next(x["response"] for x in entries if x["tool"] == tool and x["response"]["resolved_group_by"] == grouping)
    )


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            v2.ReadLengthSection,
            dict(read_count=2, total_bases=30, min_length=10, max_length=20, mean_length=100, median_length=15, n50=20),
        ),
        (v2.ReadLengthSection, dict(read_count=0, total_bases=30)),
        (
            v2.ReadLengthSection,
            dict(read_count=2, total_bases=30, min_length=10, max_length=20, mean_length=15, median_length=25, n50=20),
        ),
        (
            v2.AlignmentCounts,
            dict(
                eligible_records=0, mapped_records=0, unmapped_records=0, secondary_records=0, supplementary_records=10
            ),
        ),
        (v2.V2LengthPercentiles, dict(p1=-10)),
        (v2.V2LengthPercentiles, dict(p1=10, p50=100, p99=1)),
        (v2.VariantGeneralSection, dict(total_records=0, mnps=5)),
        (v2.IgvSnapshotsRequest, dict(genome="hg38", tracks=["reads.bam"], regions="")),
    ],
)
def test_impossible_aggregate_or_request_is_rejected(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_read_mean_is_normalized_from_native_counts():
    native = json.loads((FIXTURES / "raw" / "nanoq_haplotag.large.json").read_text())
    payload = dict(
        read_count=native["reads"],
        total_bases=native["bases"],
        min_length=native["shortest"],
        max_length=native["longest"],
        mean_length=native["bases"] / native["reads"],
        median_length=native["median_length"],
        n50=native["n50"],
    )
    v2.ReadLengthSection.model_validate(payload)
    with pytest.raises(ValidationError, match="mean_length"):
        v2.ReadLengthSection.model_validate({**payload, "mean_length": native["mean_length"]})


@pytest.mark.parametrize("tool", MODELS)
def test_effective_metrics_are_unique(tool):
    payload = response(tool, "contig" if tool == "coverage_qc" else "combined")
    payload["effective_request"]["metrics"] *= 2
    with pytest.raises(ValidationError, match="duplicates"):
        MODELS[tool].model_validate(payload)


@pytest.mark.parametrize("tool", ["read_qc", "alignment_qc", "variant_qc"])
@pytest.mark.parametrize("mutation", ["name", "id"])
def test_regional_identity_matches_normalized_request(tool, mutation):
    payload = response(tool, "region")
    if mutation == "name":
        payload["results"][0]["region_name"] = "wrong-label"
    else:
        payload["results"][0]["region_id"] = "target-a"
    with pytest.raises(ValidationError, match="IDs and order"):
        MODELS[tool].model_validate(payload)


@pytest.mark.parametrize("tool", ["read_qc", "alignment_qc", "variant_qc"])
def test_matching_noncanonical_region_ids_are_rejected(tool):
    payload = response(tool, "region")
    payload["effective_request"]["normalized_regions"][0]["region_id"] = "target-a"
    payload["results"][0]["region_id"] = "target-a"
    with pytest.raises(ValidationError, match="normalized region IDs must follow request order"):
        MODELS[tool].model_validate(payload)


def test_coverage_grouping_and_zero_threshold():
    payload = response("coverage_qc", "contig")
    payload["resolved_group_by"] = payload["effective_request"]["resolved_group_by"] = "window"
    with pytest.raises(ValidationError, match="grouping"):
        v2.CoverageQCResponse.model_validate(payload)
    row = dict(
        row_id="x",
        chrom="chr1",
        start=0,
        end=10,
        reference_bases=10,
        breadth=[dict(threshold=0, bases_at_or_above=0, fraction_at_or_above=0)],
    )
    with pytest.raises(ValidationError, match="threshold zero"):
        v2.CoverageRow.model_validate(row)
    row["breadth"] = [dict(threshold=0, bases_at_or_above=10, fraction_at_or_above=1)]
    v2.CoverageRow.model_validate(row)


def test_union_length_uses_overlaps_per_contig():
    payload = response("coverage_qc", "region")
    regions: list[dict[str, Any]] = [
        dict(region_id=f"region_{i + 1}", chrom=chrom, start=start, end=end, name=None)
        for i, (chrom, start, end) in enumerate([("chr2", 0, 5), ("chr1", 0, 10), ("chr1", 5, 15), ("chr1", 15, 20)])
    ]
    payload["effective_request"].update(normalized_regions=regions, metrics=["depth"])
    payload["rows"] = [
        dict(
            row_id=r["region_id"],
            chrom=r["chrom"],
            start=r["start"],
            end=r["end"],
            name=None,
            reference_bases=r["end"] - r["start"],
            depth_sum=0,
            mean_depth=0,
            breadth=[],
        )
        for r in regions
    ]
    payload["union_summary"] = dict(reference_bases=25, depth_sum=0, mean_depth=0)
    v2.CoverageQCResponse.model_validate(payload)
    payload["union_summary"]["reference_bases"] = 30
    with pytest.raises(ValidationError, match="union"):
        v2.CoverageQCResponse.model_validate(payload)


def test_alignment_mapq_obeys_effective_selection():
    payload = response("alignment_qc", "combined")
    payload["effective_request"]["selection"]["min_mapq"] = 10
    with pytest.raises(ValidationError, match="selection"):
        v2.AlignmentQCResponse.model_validate(payload)


@pytest.mark.parametrize("exclude_flags", [0, 1792, 1796])
def test_unmapped_eligibility_requires_explicit_opt_in(exclude_flags):
    payload = response("alignment_qc", "combined")
    payload["effective_request"]["metrics"] = ["counts"]
    payload["effective_request"]["selection"] = dict(exclude_flags=exclude_flags, include_unmapped=False)
    payload["results"][0]["mapping_quality"] = None
    counts = payload["results"][0]["counts"]
    counts["unmapped_records"] = counts["eligible_records"]
    counts["mapped_records"] = counts["secondary_records"] = counts["supplementary_records"] = 0
    with pytest.raises(ValidationError, match="unmapped records violate effective selection"):
        v2.AlignmentQCResponse.model_validate(payload)

    payload["effective_request"]["selection"]["include_unmapped"] = True
    if exclude_flags & 0x4:
        with pytest.raises(ValidationError, match="unmapped bit"):
            v2.AlignmentQCResponse.model_validate(payload)
    else:
        v2.AlignmentQCResponse.model_validate(payload)


@pytest.mark.parametrize("grouping", ["combined", "region"])
@pytest.mark.parametrize("section", ["snps", "indels"])
def test_zero_variant_records_require_zero_counts_across_sections(grouping, section):
    payload = response("variant_qc", grouping)
    payload["effective_request"]["metrics"] = ["general", section]
    result = payload["results"][0]
    result.update(general=dict(total_records=0), snps=None, indels=None)
    result[section] = dict(count=1)
    with pytest.raises(ValidationError, match="zero variant records require zero subtype counts"):
        v2.VariantQCResponse.model_validate(payload)

    result[section]["count"] = 0
    v2.VariantQCResponse.model_validate(payload)

    # Subtypes need not partition records; do not infer a sum or equality.
    result["general"]["total_records"] = 1
    result[section]["count"] = 2
    v2.VariantQCResponse.model_validate(payload)

    # Omitted general metrics provide no evidence of an empty population.
    result["general"] = None
    payload["effective_request"]["metrics"] = [section]
    v2.VariantQCResponse.model_validate(payload)


@pytest.mark.parametrize("grouping", ["combined", "region"])
@pytest.mark.parametrize("transitions,transversions", [(1, 0), (0, 1), (1, 1)])
def test_zero_snp_records_require_zero_allele_counts(grouping, transitions, transversions):
    payload = response("variant_qc", grouping)
    payload["effective_request"]["metrics"] = ["general", "snps"]
    result = payload["results"][0]
    result.update(general=dict(total_records=0), indels=None)
    result["snps"] = dict(
        count=0,
        transitions=transitions,
        transversions=transversions,
        ts_tv_ratio=transitions / transversions if transversions else None,
    )
    with pytest.raises(ValidationError, match="zero SNP records require zero allele counts"):
        v2.VariantQCResponse.model_validate(payload)

    result["snps"].update(transitions=0, transversions=0, ts_tv_ratio=None)
    v2.VariantQCResponse.model_validate(payload)
    result["snps"].update(transitions=None, transversions=None)
    v2.VariantQCResponse.model_validate(payload)

    # Multiallelic records may contribute more than one SNP allele.
    v2.VariantSnpSection(count=1, transitions=2, transversions=4, ts_tv_ratio=0.5)
