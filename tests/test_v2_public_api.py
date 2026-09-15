"""Public catalog, dispatch, validation, and resource integration for API v2."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from ont_qc_mcp import app_server
from ont_qc_mcp.v2_contracts import (
    AlignmentQCRequest,
    BedQCRequest,
    CoverageQCRequest,
    EnvironmentStatusRequest,
    ExecutionErrorResponse,
    FilterReadsRequest,
    HeaderInfoRequest,
    IgvSnapshotsRequest,
    ReadQCRequest,
    ReadQCResponse,
    RunSummaryRequest,
    ValidationErrorResponse,
    VariantQCRequest,
    api_v2_contracts,
)


_PUBLIC_CALLS: list[tuple[str, dict[str, object], type, Callable[..., object]]] = [
    ("read_qc", {"path": "reads.fastq"}, ReadQCRequest, app_server.execute_read_qc),
    ("alignment_qc", {"path": "reads.bam"}, AlignmentQCRequest, app_server.execute_alignment_qc),
    ("coverage_qc", {"path": "reads.bam"}, CoverageQCRequest, app_server.execute_coverage_qc),
    ("variant_qc", {"path": "calls.vcf"}, VariantQCRequest, app_server.execute_variant_qc),
    ("environment_status", {}, EnvironmentStatusRequest, app_server.execute_environment_status),
    ("header_info", {"path": "reads.bam"}, HeaderInfoRequest, app_server.execute_header_info),
    ("bed_qc", {"path": "targets.bed"}, BedQCRequest, app_server.execute_bed_qc),
    ("run_summary", {"path": "summary.txt"}, RunSummaryRequest, app_server.execute_run_summary),
    ("filter_reads", {"path": "reads.fastq"}, FilterReadsRequest, app_server.execute_filter_reads),
    ("igv_snapshots", {"batch_file": "snapshots.batch"}, IgvSnapshotsRequest, app_server.execute_igv_snapshots),
]


def _payload(result) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(result.content[0].text))


@pytest.mark.asyncio
async def test_public_schemas_are_the_frozen_request_models() -> None:
    listed = {tool.name: tool for tool in await app_server.list_tools()}
    contracts = api_v2_contracts()
    assert set(listed) == set(contracts)
    for name, contract in contracts.items():
        assert listed[name].input_schema == contract.request_model.model_json_schema(mode="validation")


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "arguments", "request_type", "executor"), _PUBLIC_CALLS)
async def test_each_public_tool_dispatches_directly_to_its_v2_adapter(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, object],
    request_type: type,
    executor: Callable[..., object],
) -> None:
    observed: list[tuple[Callable[..., object], object]] = []

    async def fake_run_sync(function, request, **kwargs):
        observed.append((function, request))
        return {"ok": True}

    monkeypatch.setattr(app_server, "run_sync", fake_run_sync)
    result = await app_server.dispatch_tool(name, arguments)
    assert not result.is_error
    assert _payload(result) == {"ok": True}
    assert observed == [(executor, observed[0][1])]
    assert isinstance(observed[0][1], request_type)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_qc", {}),
        ("read_qc", {"path": 1}),
        ("coverage_qc", {"path": "reads.bam", "regions": []}),
        ("alignment_qc", {"path": "reads.bam", "unknown": True}),
        (
            "igv_snapshots",
            {"genome": "hg38", "tracks": [], "regions": [{"chrom": "chr1", "start": True, "end": 2}]},
        ),
    ],
)
async def test_strict_request_validation_happens_before_worker_entry(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, object],
) -> None:
    entered = False

    async def forbidden_run_sync(*args, **kwargs):
        nonlocal entered
        entered = True
        raise AssertionError("validation entered the worker")

    monkeypatch.setattr(app_server, "run_sync", forbidden_run_sync)
    result = await app_server.dispatch_tool(name, arguments)
    assert result.is_error
    ValidationErrorResponse.model_validate(_payload(result))
    assert not entered


@pytest.mark.asyncio
async def test_post_admission_value_error_is_an_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failed_run_sync(*args, **kwargs):
        raise ValueError("Mosdepth per-base output is inconsistent")

    monkeypatch.setattr(app_server, "run_sync", failed_run_sync)
    result = await app_server.dispatch_tool("coverage_qc", {"path": "reads.bam"})
    assert result.is_error
    payload = ExecutionErrorResponse.model_validate(_payload(result))
    assert payload.stage == "execution_validation"
    assert payload.backend == "server"
    assert "per-base output is inconsistent" in payload.message


@pytest.mark.asyncio
async def test_invalid_backend_response_model_is_an_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failed_run_sync(*args, **kwargs):
        ReadQCResponse.model_validate({})

    monkeypatch.setattr(app_server, "run_sync", failed_run_sync)
    result = await app_server.dispatch_tool("read_qc", {"path": "reads.fastq"})
    assert result.is_error
    payload = ExecutionErrorResponse.model_validate(_payload(result))
    assert payload.stage == "result_validation"
    assert payload.backend == "server"


@pytest.mark.asyncio
async def test_legacy_aliases_are_not_dispatchable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden_run_sync(*args, **kwargs):
        raise AssertionError("legacy alias entered the worker")

    monkeypatch.setattr(app_server, "run_sync", forbidden_run_sync)
    for name in ("env_status", "qc_reads_fastq_tool", "alignment_summary_tool", "targeted_coverage_tool"):
        result = await app_server.dispatch_tool(name, {})
        assert result.is_error
        payload = ValidationErrorResponse.model_validate(_payload(result))
        assert payload.tool == name
        assert "Unknown tool" in payload.issues[0].message


@pytest.mark.asyncio
async def test_resources_describe_only_public_tools_and_alignment_recipe() -> None:
    resources = await app_server.list_resources()
    uris = {str(resource.uri) for resource in resources}
    assert uris == {
        *(f"tool://guidance/{name}" for name in api_v2_contracts()),
        "tool://recipes/alignment_qc",
    }
    recipe = json.loads((await app_server.read_resource("tool://recipes/alignment_qc"))[0].text)
    calls = recipe["recipes"]["alignment_and_coverage"]["calls"]
    assert [call["tool"] for call in calls] == ["alignment_qc", "coverage_qc"]
    assert "alignment_summary_tool" not in json.dumps(recipe)
