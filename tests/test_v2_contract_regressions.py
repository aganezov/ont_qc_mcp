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
        payload["effective_request"]["normalized_regions"][0]["region_id"] = "target-a"
        payload["results"][0]["region_id"] = "target-a"
    with pytest.raises(ValidationError):
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


def test_alignment_result_obeys_effective_selection():
    payload = response("alignment_qc", "combined")
    payload["effective_request"]["selection"]["min_mapq"] = 10
    with pytest.raises(ValidationError, match="selection"):
        v2.AlignmentQCResponse.model_validate(payload)
    payload["effective_request"]["selection"]["min_mapq"] = 0
    counts = payload["results"][0]["counts"]
    counts["mapped_records"] = 0
    counts["unmapped_records"] = counts["eligible_records"]
    with pytest.raises(ValidationError, match="selection"):
        v2.AlignmentQCResponse.model_validate(payload)
