"""Contract tests for the unregistered API v2 skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from ont_qc_mcp.app_server import list_tools
from ont_qc_mcp.v2_contracts import (
    AlignmentQCRequest,
    AlignmentQCResponse,
    API_V2_CATALOG,
    CoverageQCRequest,
    CoverageQCResponse,
    ExecutionErrorResponse,
    FilterReadsRequest,
    IgvSnapshotsRequest,
    IdentitySection,
    LengthDistributionSection,
    ErrorProfileSection,
    ReadQCRequest,
    ReadQCResponse,
    ReadQualitySection,
    ReadSelection,
    ValidationErrorResponse,
    VariantSnpSection,
    VariantQCRequest,
    VariantQCResponse,
    api_v2_contracts,
)

FIXTURES = Path(__file__).parent / "fixtures" / "api_v2"
REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "read_qc": ReadQCRequest,
    "alignment_qc": AlignmentQCRequest,
    "coverage_qc": CoverageQCRequest,
    "variant_qc": VariantQCRequest,
}
RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "read_qc": ReadQCResponse,
    "alignment_qc": AlignmentQCResponse,
    "coverage_qc": CoverageQCResponse,
    "variant_qc": VariantQCResponse,
}


def load_fixture(name: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def test_catalog_has_the_accepted_names_and_real_server_is_unchanged() -> None:
    expected = {
        "read_qc",
        "alignment_qc",
        "coverage_qc",
        "variant_qc",
        "environment_status",
        "header_info",
        "bed_qc",
        "run_summary",
        "filter_reads",
        "igv_snapshots",
    }
    contracts = api_v2_contracts()
    assert set(contracts) == expected
    assert len(API_V2_CATALOG) == len(expected)
    assert all(contract.status == "unregistered" for contract in contracts.values())


@pytest.mark.asyncio
async def test_catalog_skeleton_does_not_advertise_v2_tools() -> None:
    advertised = {tool.name for tool in await list_tools()}
    assert len(advertised) == 18
    assert advertised.isdisjoint(api_v2_contracts())
    assert "qc_reads_fastq_tool" in advertised


@pytest.mark.parametrize("entry", load_fixture("requests.json"), ids=lambda entry: str(entry["case"]))
def test_numerical_request_fixtures_validate(entry: dict[str, object]) -> None:
    model = REQUEST_MODELS[str(entry["tool"])]
    request = model.model_validate(entry["request"])
    assert request.model_dump()["path"]


@pytest.mark.parametrize(
    "entry",
    load_fixture("responses.json") + load_fixture("response-examples.json"),
    ids=lambda entry: str(entry["case"]),
)
def test_numerical_response_fixtures_validate(entry: dict[str, object]) -> None:
    model = RESPONSE_MODELS[str(entry["tool"])]
    response = model.model_validate(entry["response"])
    payload = response.model_dump()
    assert payload["effective_request"]["metrics"]
    assert payload["provenance"]


@pytest.mark.parametrize("entry", load_fixture("errors.json"), ids=lambda entry: str(entry["case"]))
def test_error_fixtures_validate(entry: dict[str, object]) -> None:
    payload = entry["response"]
    assert isinstance(payload, dict)
    model = ValidationErrorResponse if payload["kind"] == "validation_error" else ExecutionErrorResponse
    error = model.model_validate(payload)
    if isinstance(error, ExecutionErrorResponse):
        assert error.partial_result_returned is False


@pytest.mark.parametrize(
    ("model", "payload", "message"),
    [
        (CoverageQCRequest, {"path": "x.bam", "regions": []}, "regions must be omitted"),
        (CoverageQCRequest, {"path": "x.bam", "group_by": "region"}, "group_by must be 'contig'"),
        (CoverageQCRequest, {"path": "x.bam", "window_size": 10, "group_by": "contig"}, "group_by must be 'window'"),
        (
            ReadQCRequest,
            {"path": "x.fastq", "input_format": "fastq", "regions": [{"chrom": "chr1", "start": 0, "end": 1}]},
            "FASTQ input",
        ),
        (
            ReadQCRequest,
            {"path": "x.fastq", "input_format": "fastq", "selection": {"min_mapq": 10}},
            "alignment selection",
        ),
        (
            ReadQCRequest,
            {"path": "x.fastq", "input_format": "fastq", "extra_args": {"samtools_view": ["-F", "4"]}},
            "samtools arguments",
        ),
        (AlignmentQCRequest, {"path": "x.bam", "selection": {"include_unmapped": True}}, "unmapped bit"),
        (
            AlignmentQCRequest,
            {
                "path": "x.bam",
                "regions": [{"chrom": "chr1", "start": 0, "end": 1}],
                "selection": {"include_unmapped": True, "exclude_flags": 1792},
            },
            "ineligible",
        ),
        (VariantQCRequest, {"path": "x.vcf", "group_by": "region"}, "requires regions"),
        (CoverageQCRequest, {"path": "x.bam", "extra_args": {"samtools_view": ["-e", "x"]}}, "Extra inputs"),
    ],
)
def test_contradictory_or_unsupported_requests_are_rejected(model, payload, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        model.model_validate(payload)


def test_defaults_and_native_argv_order_are_stable() -> None:
    read = ReadQCRequest(path="reads.bam")
    alignment = AlignmentQCRequest(path="reads.bam")
    coverage = CoverageQCRequest(path="reads.bam")
    assert read.metrics == ["length", "read_quality"]
    assert alignment.metrics == ["counts", "mapping_quality"]
    assert alignment.selection.exclude_flags == 1796
    assert coverage.thresholds == [1, 10, 20]
    assert coverage.depth_statistic == "mean"
    argv = ["--samples", "S1", "--samples", "S2"]
    request = VariantQCRequest.model_validate({"path": "calls.vcf.gz", "extra_args": {"bcftools_stats": argv}})
    assert request.extra_args.bcftools_stats == argv


def test_explicit_depth_statistic_requires_depth_metric() -> None:
    breadth_only = CoverageQCRequest.model_validate({"path": "reads.bam", "metrics": ["breadth"]})
    assert breadth_only.depth_statistic == "mean"
    with pytest.raises(ValidationError, match="depth_statistic requires the depth metric"):
        CoverageQCRequest.model_validate({"path": "reads.bam", "metrics": ["breadth"], "depth_statistic": "mean"})
    with pytest.raises(ValidationError, match="depth_statistic requires the depth metric"):
        CoverageQCRequest.model_validate({"path": "reads.bam", "metrics": ["breadth"], "depth_statistic": "median"})


def test_response_invariants_distinguish_zero_from_missing() -> None:
    with pytest.raises(ValidationError, match="null when read_count is zero"):
        ReadQCResponse.model_validate(
            {
                **cast(dict[str, Any], load_fixture("responses.json")[0]["response"]),
                "results": [{"length": {"read_count": 0, "total_bases": 0, "mean_length": 0}}],
            }
        )


def test_strict_literal_boolean_and_complete_quality_summary() -> None:
    with pytest.raises(ValidationError, match="must be a boolean"):
        ReadSelection.model_validate({"primary_only": 1})
    with pytest.raises(ValidationError, match="both mean and median"):
        ReadQualitySection(reads_with_quality=1, reads_missing_quality=0, mean_qscore=30, median_qscore=None)


def test_success_requires_every_requested_result_section() -> None:
    payload = load_fixture("responses.json")[0]["response"]
    assert isinstance(payload, dict)
    with pytest.raises(ValidationError, match="must match requested metrics"):
        ReadQCResponse.model_validate({**cast(dict[str, Any], payload), "results": [{}]})
    with pytest.raises(ValidationError, match="denominators must sum"):
        AlignmentQCResponse.model_validate(
            {
                **cast(dict[str, Any], load_fixture("responses.json")[1]["response"]),
                "results": [
                    {
                        "aligned_base_quality": {
                            "aligned_query_bases": 3,
                            "known_quality_bases": 1,
                            "missing_quality_bases": 1,
                            "mean_base_quality": 20,
                        }
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "case",
    ["read-bam-per-region-length-subset", "alignment-missing-data", "variant-by-region"],
)
def test_region_results_match_normalized_region_ids_in_order(case: str) -> None:
    entries = load_fixture("responses.json") + load_fixture("response-examples.json")
    entry = next(item for item in entries if item["case"] == case)
    payload = cast(dict[str, Any], entry["response"])
    effective = cast(dict[str, Any], payload["effective_request"])
    regions = cast(list[dict[str, Any]], effective["normalized_regions"])
    extra_region = {**regions[0], "start": 10, "end": 20, "region_id": "region_2"}
    with pytest.raises(ValidationError, match="IDs and order"):
        RESPONSE_MODELS[str(entry["tool"])].model_validate(
            {**payload, "effective_request": {**effective, "normalized_regions": [*regions, extra_region]}}
        )


def test_effective_region_scope_requires_nonempty_unique_normalized_ids() -> None:
    entry = next(
        item for item in load_fixture("response-examples.json") if item["case"] == "read-bam-per-region-length-subset"
    )
    payload = cast(dict[str, Any], entry["response"])
    effective = cast(dict[str, Any], payload["effective_request"])
    region = cast(list[dict[str, Any]], effective["normalized_regions"])[0]
    with pytest.raises(ValidationError, match="must be unique"):
        ReadQCResponse.model_validate(
            {
                **payload,
                "effective_request": {**effective, "normalized_regions": [region, {**region}]},
                "results": [*cast(list[dict[str, Any]], payload["results"]), {"region_id": region["region_id"]}],
            }
        )
    with pytest.raises(ValidationError, match="whole scope requires none"):
        ReadQCResponse.model_validate({**payload, "effective_request": {**effective, "region_scope": "whole_file"}})


def test_coverage_breadth_matches_requested_threshold_and_denominator() -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "coverage-whole-contigs")
    payload = cast(dict[str, Any], entry["response"])
    rows = cast(list[dict[str, Any]], payload["rows"])
    bad_row = {
        **rows[0],
        "breadth": [{"threshold": 999, "bases_at_or_above": 999, "fraction_at_or_above": 0.5}],
    }
    with pytest.raises(ValidationError):
        CoverageQCResponse.model_validate({**payload, "rows": [bad_row]})
    with pytest.raises(ValidationError, match="union mean_depth"):
        CoverageQCResponse.model_validate(
            {**payload, "union_summary": {"reference_bases": 10, "depth_sum": 10, "mean_depth": 0.5}}
        )


def test_breadth_only_response_omits_depth_fields() -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "coverage-whole-contigs")
    payload = cast(dict[str, Any], entry["response"])
    effective = cast(dict[str, Any], payload["effective_request"])
    row = cast(dict[str, Any], cast(list[dict[str, Any]], payload["rows"])[0])
    union = cast(dict[str, Any], payload["union_summary"])
    breadth_only = {
        **payload,
        "effective_request": {**effective, "metrics": ["breadth"]},
        "rows": [{key: value for key, value in row.items() if key not in {"depth_sum", "mean_depth"}}],
        "union_summary": {key: value for key, value in union.items() if key not in {"depth_sum", "mean_depth"}},
    }
    CoverageQCResponse.model_validate(breadth_only)
    with pytest.raises(ValidationError, match="depth fields must match requested metrics"):
        CoverageQCResponse.model_validate(
            {
                **breadth_only,
                "rows": [{**cast(list[dict[str, Any]], breadth_only["rows"])[0], "depth_sum": 0, "mean_depth": 0.0}],
            }
        )
    with pytest.raises(ValidationError, match="depth fields must match requested metrics"):
        CoverageQCResponse.model_validate(
            {**breadth_only, "union_summary": {"reference_bases": 10, "depth_sum": 0, "mean_depth": 0.0}}
        )
    breadth = cast(list[dict[str, Any]], cast(list[dict[str, Any]], breadth_only["rows"])[0]["breadth"])
    with pytest.raises(ValidationError):
        CoverageQCResponse.model_validate(
            {
                **breadth_only,
                "rows": [
                    {
                        **cast(list[dict[str, Any]], breadth_only["rows"])[0],
                        "breadth": [{key: value for key, value in breadth[0].items() if key != "fraction_at_or_above"}],
                    }
                ],
            }
        )


def test_depth_only_response_requires_depth_and_allows_empty_union() -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "coverage-requested-region")
    payload = cast(dict[str, Any], entry["response"])
    CoverageQCResponse.model_validate(payload)
    row = cast(dict[str, Any], cast(list[dict[str, Any]], payload["rows"])[0])
    with pytest.raises(ValidationError, match="depth fields must match requested metrics"):
        CoverageQCResponse.model_validate(
            {**payload, "rows": [{key: value for key, value in row.items() if key not in {"depth_sum", "mean_depth"}}]}
        )
    whole_entry = next(
        item for item in load_fixture("response-examples.json") if item["case"] == "coverage-whole-contigs"
    )
    whole = cast(dict[str, Any], whole_entry["response"])
    empty = {
        **whole,
        "rows": [],
        "union_summary": {"reference_bases": 0, "depth_sum": 0, "mean_depth": None},
    }
    CoverageQCResponse.model_validate(empty)


def test_median_depth_response_requires_only_row_medians() -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "coverage-requested-region")
    payload = cast(dict[str, Any], entry["response"])
    effective = cast(dict[str, Any], payload["effective_request"])
    row = cast(dict[str, Any], cast(list[dict[str, Any]], payload["rows"])[0])
    median_payload = {
        **payload,
        "effective_request": {**effective, "depth_statistic": "median"},
        "rows": [{**row, "median_depth": 1.0}],
    }
    CoverageQCResponse.model_validate(median_payload)
    with pytest.raises(ValidationError, match="median_depth must match"):
        CoverageQCResponse.model_validate(
            {**median_payload, "rows": [{key: value for key, value in row.items() if key != "median_depth"}]}
        )
    with pytest.raises(ValidationError, match="median_depth must match"):
        CoverageQCResponse.model_validate({**payload, "rows": [{**row, "median_depth": 1.0}]})


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicate", "shifted", "renamed"])
def test_coverage_region_rows_exactly_match_requested_intervals(mutation: str) -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "coverage-requested-region")
    payload = cast(dict[str, Any], entry["response"])
    effective = cast(dict[str, Any], payload["effective_request"])
    region = cast(list[dict[str, Any]], effective["normalized_regions"])[0]
    second = {**region, "start": 5, "end": 10, "name": "second", "region_id": "region_2"}
    row = cast(list[dict[str, Any]], payload["rows"])[0]
    second_row = {**row, "row_id": "region_2", "start": 5, "end": 10, "name": "second"}
    rows = [row, second_row]
    if mutation == "missing":
        rows = [row]
    elif mutation == "reordered":
        rows = [second_row, row]
    elif mutation == "duplicate":
        rows = [row, row]
    elif mutation == "shifted":
        rows = [row, {**second_row, "start": 6, "reference_bases": 4, "depth_sum": 4}]
    else:
        rows = [row, {**second_row, "name": "changed"}]
    with pytest.raises(ValidationError, match="coverage region rows must match"):
        CoverageQCResponse.model_validate(
            {
                **payload,
                "effective_request": {**effective, "normalized_regions": [region, second]},
                "rows": rows,
            }
        )


def test_successful_read_quality_rejects_missing_quality_records() -> None:
    with pytest.raises(ValidationError, match="missing quality"):
        ReadQualitySection(
            reads_with_quality=1,
            reads_missing_quality=1,
            mean_qscore=20,
            median_qscore=20,
        )


def test_combined_read_accounting_and_result_counts_are_consistent() -> None:
    entry = next(item for item in load_fixture("response-examples.json") if item["case"] == "read-whole-file-full")
    payload = cast(dict[str, Any], entry["response"])
    with pytest.raises(ValidationError, match="selected_records must equal"):
        ReadQCResponse.model_validate({**payload, "selected_records": 3})
    result = cast(list[dict[str, Any]], payload["results"])[0]
    length = cast(dict[str, Any], result["length"])
    with pytest.raises(ValidationError, match="emitted_sequences"):
        ReadQCResponse.model_validate(
            {
                **payload,
                "results": [
                    {
                        **result,
                        "length": {
                            **length,
                            "read_count": 1,
                            "total_bases": 15,
                            "min_length": 15,
                            "max_length": 15,
                            "mean_length": 15,
                            "median_length": 15,
                            "n50": 15,
                        },
                    }
                ],
            }
        )


def test_alignment_count_and_mapq_denominators_are_consistent() -> None:
    entry = next(
        item for item in load_fixture("response-examples.json") if item["case"] == "alignment-default-combined"
    )
    payload = cast(dict[str, Any], entry["response"])
    result = cast(list[dict[str, Any]], payload["results"])[0]
    counts = cast(dict[str, Any], result["counts"])
    with pytest.raises(ValidationError, match="mapped and unmapped"):
        AlignmentQCResponse.model_validate(
            {**payload, "results": [{**result, "counts": {**counts, "mapped_records": 1}}]}
        )
    mapq = cast(dict[str, Any], result["mapping_quality"])
    with pytest.raises(ValidationError, match="MAPQ denominators"):
        AlignmentQCResponse.model_validate(
            {**payload, "results": [{**result, "mapping_quality": {**mapq, "missing_records": 0}}]}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"count": 2, "transitions": None, "transversions": None, "ts_tv_ratio": 1.0},
        {"count": 2, "transitions": 1, "transversions": None, "ts_tv_ratio": None},
        {"count": 2, "transitions": 1, "transversions": 1, "ts_tv_ratio": 2.0},
        {"count": 2, "transitions": 1, "transversions": 0, "ts_tv_ratio": 1.0},
    ],
)
def test_variant_tstv_requires_available_consistent_allele_counts(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        VariantSnpSection.model_validate(payload)


@pytest.mark.parametrize(
    "histogram",
    [
        [{"start": 1.0, "end": 2.0, "count": -1}],
        [{"start": 2.0, "end": 1.0, "count": 1}],
        [{"start": float("inf"), "end": 2.0, "count": 1}],
        [{"start": 1.0, "end": 2.0, "count": 1.0}],
    ],
)
def test_v2_histograms_reject_invalid_legacy_shapes(histogram: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError):
        LengthDistributionSection.model_validate({"percentiles": {}, "histogram": histogram})


def test_igv_region_retains_strict_per_region_commands() -> None:
    request = IgvSnapshotsRequest.model_validate(
        {
            "genome": "reference.fa",
            "tracks": ["reads.bam"],
            "regions": [{"chrom": "chr1", "start": 0, "end": 10, "name": "target", "extra_commands": ["sort BASE"]}],
        }
    )
    assert isinstance(request.regions, list)
    assert request.regions[0].extra_commands == ["sort BASE"]
    with pytest.raises(ValidationError):
        IgvSnapshotsRequest.model_validate(
            {
                "genome": "reference.fa",
                "tracks": ["reads.bam"],
                "regions": [{"chrom": "chr1", "start": 0, "end": 10, "extra_commands": "sort BASE"}],
            }
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (IdentitySection, {"records_with_identity": 0, "records_missing_identity": 2, "mean_identity": 0.95}),
        (ErrorProfileSection, {"records_with_nm": 0, "records_missing_nm": 2, "nm_error_rate": 0.1}),
        (ErrorProfileSection, {"records_with_nm": 1, "records_missing_nm": 0, "nm_error_rate": float("inf")}),
    ],
)
def test_identity_and_nm_means_require_known_denominators_and_finite_values(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_supporting_contracts_retain_nontrivial_controls() -> None:
    filtered = FilterReadsRequest.model_validate(
        {
            "path": "reads.fastq.gz",
            "output_fastq": "filtered.fastq.gz",
            "selection": {"headcrop": 10, "tailcrop": 5, "trim_approach": "fixed-crop", "threads": 2},
        }
    )
    assert filtered.selection.headcrop == 10
    dynamic = IgvSnapshotsRequest.model_validate(
        {
            "genome": "reference.fa",
            "tracks": ["reads.bam"],
            "regions": [{"chrom": "chr1", "start": 0, "end": 10}],
            "compact": "collapse",
            "color_by": "READ_STRAND",
            "small_indels_show": True,
        }
    )
    assert dynamic.compact == "collapse"
    assert IgvSnapshotsRequest(batch_file="prepared.batch", output_dir="snapshots").batch_file == "prepared.batch"


def test_schema_contains_descriptions_and_forbids_unrestricted_objects() -> None:
    schema = CoverageQCRequest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["group_by"]["description"]
    assert schema["$defs"]["CoverageExtraArgs"]["additionalProperties"] is False
