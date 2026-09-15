"""Hand-counted indexed BAM/CRAM evidence, exercised through Python and MCP."""

import json
import subprocess
import struct
from pathlib import Path
from typing import Any, cast

import mcp_types as types
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from conftest import require_executable_tools
from ont_qc_mcp.regional import regional_alignment_stats
from ont_qc_mcp.utils import CommandError

pytestmark = pytest.mark.integration


@pytest.fixture
def regional_files(tmp_path):
    require_executable_tools(["samtools"])
    reference = tmp_path / "ref.fa"
    reference.write_text(">chr1\n" + "A" * 1000 + "\n>track\n" + "A" * 20 + "\n>#hash\n" + "A" * 20 + "\n")
    subprocess.run(["samtools", "faidx", str(reference)], check=True)
    sam = tmp_path / "reads.sam"
    q = "".join(chr(x + 33) for x in range(10, 23))
    records = [
        f"mixed\t0\tchr1\t101\t60\t2S3M2I2M2D2M3N1M1S\t*\t0\t0\t{'A' * 13}\t{q}",
        f"mixed\t16\tchr1\t101\t40\t2S3M2I2M2D2M3N1M1S\t*\t0\t0\t{'A' * 13}\t{q}",
        "unknown_mapq\t0\tchr1\t103\t255\t4M\t*\t0\t0\tAAAA\tIIII",
        "unknown_qual\t0\tchr1\t103\t20\t2M\t*\t0\t0\tAA\t*",
        "supp\t2048\tchr1\t103\t30\t1M\t*\t0\t0\tA\t?",
        "secondary\t256\tchr1\t103\t50\t1M\t*\t0\t0\tA\tS",
        "duplicate\t1024\tchr1\t103\t50\t1M\t*\t0\t0\tA\tS",
        "qcfail\t512\tchr1\t103\t50\t1M\t*\t0\t0\tA\tS",
        "keyword\t0\ttrack\t1\t20\t1M\t*\t0\t0\tA\tI",
        "hash\t0\t#hash\t1\t20\t1M\t*\t0\t0\tA\tI",
    ]
    sam.write_text(
        "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:track\tLN:20\n@SQ\tSN:#hash\tLN:20\n"
        + "\n".join(records)
        + "\n"
    )
    bam, cram = tmp_path / "reads.bam", tmp_path / "reads.cram"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(sam)], check=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    subprocess.run(["samtools", "view", "-C", "-T", str(reference), "-o", str(cram), str(bam)], check=True)
    subprocess.run(["samtools", "index", str(cram)], check=True)
    return bam, cram, reference


def selected_region():
    return {"chrom": "chr1", "start": 102, "end": 108, "name": "quality"}


def assert_known_result(item):
    assert item["span_overlapping_alignments"] == 5
    assert item["aligned_base_alignments"] == 5
    assert item["aligned_query_bases"] == 15
    assert item["quality_known_bases"] == 13
    assert item["quality_missing_bases"] == 2
    assert item["mean_base_quality"] == pytest.approx(326 / 13)
    assert item["mapq_known_alignments"] == 4 and item["mapq_missing_alignments"] == 1
    assert item["mean_mapq"] == 37.5
    assert item["supplementary_alignments"] == 1 and item["reverse_alignments"] == 1


def assert_known_v2_result(item):
    assert item["counts"] == {
        "eligible_records": 5,
        "mapped_records": 5,
        "unmapped_records": 0,
        "secondary_records": 0,
        "supplementary_records": 1,
    }
    assert item["aligned_base_quality"]["aligned_query_bases"] == 15
    assert item["aligned_base_quality"]["known_quality_bases"] == 13
    assert item["aligned_base_quality"]["missing_quality_bases"] == 2
    assert item["aligned_base_quality"]["mean_base_quality"] == pytest.approx(326 / 13)
    assert item["mapping_quality"]["known_records"] == 4
    assert item["mapping_quality"]["missing_records"] == 1
    assert item["mapping_quality"]["mean_mapq"] == 37.5


@pytest.mark.parametrize("file_type", ["bam", "cram"])
def test_hand_counted_batched_evidence(regional_files, file_type):
    bam, cram, reference = regional_files
    alignment = bam if file_type == "bam" else cram
    kwargs: dict[str, Any] = {"reference_path": str(reference)} if file_type == "cram" else {}
    regions = [
        selected_region(),
        {"chrom": "chr1", "start": 109, "end": 112},
        selected_region(),
        {"chrom": "chr1", "start": 113, "end": 114},
        {"chrom": "track", "start": 0, "end": 1},
    ]
    before = {p.name: p.read_bytes() for p in bam.parent.iterdir() if p.is_file()}
    report = regional_alignment_stats(str(alignment), regions, **kwargs)
    assert_known_result(report["regions"][0])
    assert report["regions"][1]["span_overlapping_alignments"] == 2
    assert report["regions"][1]["aligned_query_bases"] == 0
    assert report["regions"][1]["mean_base_quality"] is None
    assert report["regions"][3]["span_overlapping_alignments"] == 0
    assert report["regions"][4]["mean_base_quality"] == 40
    assert report["regions"][0]["region_id"] != report["regions"][2]["region_id"]
    for request, item in zip(regions, report["regions"], strict=True):
        single = regional_alignment_stats(str(alignment), [request], **kwargs)["regions"][0]
        assert {k: v for k, v in single.items() if k != "region_id"} == {
            k: v for k, v in item.items() if k != "region_id"
        }
    assert before == {p.name: p.read_bytes() for p in bam.parent.iterdir() if p.is_file()}


def test_bam_cram_metrics_agree(regional_files):
    bam, cram, reference = regional_files
    a = regional_alignment_stats(str(bam), [selected_region()])
    b = regional_alignment_stats(str(cram), [selected_region()], reference_path=str(reference))
    assert a["regions"] == b["regions"]


def test_filter_policy_and_unavailable_mapq(regional_files):
    bam, _cram, _reference = regional_files
    positive = regional_alignment_stats(str(bam), [selected_region()], min_mapq=1)["regions"][0]
    assert positive["span_overlapping_alignments"] == 4 and positive["mapq_missing_alignments"] == 0
    assert positive["mean_base_quality"] == pytest.approx(166 / 9)
    unfiltered = regional_alignment_stats(str(bam), [selected_region()], exclude_flags=0)["regions"][0]
    assert unfiltered["span_overlapping_alignments"] == 8
    assert (
        unfiltered["secondary_alignments"] == unfiltered["duplicate_alignments"] == unfiltered["qcfail_alignments"] == 1
    )


@pytest.mark.parametrize(
    "problem", ["missing_index", "corrupt_index", "missing_reference", "missing_fai", "wrong_reference"]
)
def test_cram_prerequisite_failure_preserves_inputs(regional_files, problem):
    _bam, cram, reference = regional_files
    if problem == "missing_index":
        Path(str(cram) + ".crai").unlink()
    elif problem == "corrupt_index":
        Path(str(cram) + ".crai").write_bytes(b"invalid index")
    elif problem == "missing_fai":
        Path(str(reference) + ".fai").unlink()
    elif problem == "wrong_reference":
        reference.write_text(reference.read_text().replace("A", "C"))
    before = {p.name: p.read_bytes() for p in cram.parent.iterdir() if p.is_file()}
    kwargs: dict[str, Any] = {} if problem == "missing_reference" else {"reference_path": str(reference)}
    with pytest.raises((ValueError, FileNotFoundError, CommandError)):
        regional_alignment_stats(str(cram), [selected_region()], **kwargs)
    assert before == {p.name: p.read_bytes() for p in cram.parent.iterdir() if p.is_file()}


def test_hash_contig_cannot_silently_drop_from_batch(regional_files):
    bam, _cram, _reference = regional_files
    with pytest.raises(ValueError, match="unsupported.*BED"):
        regional_alignment_stats(str(bam), [selected_region(), {"chrom": "#hash", "start": 0, "end": 1}])


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["compat", "anyio"])
async def test_mcp_regional_discovery_validation_and_result(regional_files, mcp_server_params, transport):
    bam, _cram, _reference = regional_files
    mcp_server_params.env = {**(mcp_server_params.env or {}), "MCP_STDIO_TRANSPORT": transport}
    async with stdio_client(mcp_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool = next(t for t in (await session.list_tools()).tools if t.name == "alignment_qc")
            assert tool.input_schema["title"] == "AlignmentQCRequest"
            assert tool.input_schema["additionalProperties"] is False
            invalid_arguments: list[dict[str, Any]] = [
                {"path": "absent.bam", "regions": [{"chrom": "chr1", "start": True, "end": 1}]},
                {"path": "absent.bam", "regions": [], "selection": {"min_mapq": 255}},
                {"path": "absent.bam", "regions": [selected_region()], "selection": {"exclude_flags": True}},
            ]
            for args in invalid_arguments:
                failed = await session.call_tool(tool.name, args)
                assert failed.is_error
                assert "File not found" not in cast(types.TextContent, failed.content[0]).text
            result = await session.call_tool(
                tool.name,
                {
                    "path": str(bam),
                    "regions": [selected_region()],
                    "group_by": "region",
                    "metrics": ["counts", "mapping_quality", "aligned_base_quality"],
                },
            )
            assert not result.is_error
            payload = json.loads(cast(types.TextContent, result.content[0]).text)
            assert_known_v2_result(payload["results"][0])
            assert payload["resolved_group_by"] == "region"
            assert payload["results"][0]["region_name"] == "quality"
            assert payload["provenance"]


@pytest.mark.parametrize("quality,expected", [(8, 8), (9, None), (10, 10), (255, None)])
def test_binary_bam_one_base_quality_ambiguity(tmp_path, quality, expected):
    require_executable_tools(["samtools"])
    # Construct the native BAM quality byte: starting with SAM would already
    # lose the distinction between Q9 and missing (both serialize as '*').
    header = b"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:100\n"
    data = (
        b"BAM\1"
        + struct.pack("<i", len(header))
        + header
        + struct.pack("<i", 1)
        + struct.pack("<i", 5)
        + b"chr1\0"
        + struct.pack("<i", 100)
    )
    core = struct.pack("<iiIIiiii", 0, 0, (4681 << 16) | (40 << 8) | 2, 1, 1, -1, -1, 0)
    record = core + b"r\0" + struct.pack("<I", 16) + b"\x10" + bytes([quality])
    raw = tmp_path / "raw.bam"
    raw.write_bytes(data + struct.pack("<i", len(record)) + record)
    bam = tmp_path / "quality.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(bam), str(raw)], check=True, capture_output=True)
    subprocess.run(["samtools", "index", str(bam)], check=True)
    subprocess.run(["samtools", "quickcheck", str(bam)], check=True)
    regions = [{"chrom": "chr1", "start": 0, "end": 1}]
    if expected is None:
        with pytest.raises(ValueError, match="one-base Q9"):
            regional_alignment_stats(str(bam), regions)
    else:
        item = regional_alignment_stats(str(bam), regions)["regions"][0]
        assert item["quality_known_bases"] == 1 and item["mean_base_quality"] == expected


@pytest.mark.parametrize("minimum,count,mean_mapq", [(40, 2, 50), (41, 1, 60)])
def test_real_mapq_threshold_boundary(regional_files, minimum, count, mean_mapq):
    bam, _cram, _reference = regional_files
    item = regional_alignment_stats(str(bam), [selected_region()], min_mapq=minimum)["regions"][0]
    assert item["span_overlapping_alignments"] == count
    assert item["mean_mapq"] == mean_mapq
    assert item["mean_base_quality"] == 17
    assert item["mapq_missing_alignments"] == 0


def test_real_supplementary_exclusion(regional_files):
    bam, _cram, _reference = regional_files
    item = regional_alignment_stats(str(bam), [selected_region()], exclude_flags=3844)["regions"][0]
    assert item["span_overlapping_alignments"] == 4
    assert item["supplementary_alignments"] == 0
    assert item["mean_base_quality"] == pytest.approx(296 / 12)


@pytest.mark.parametrize("file_type", ["bam", "cram"])
def test_index_beside_alignment_symlink(regional_files, file_type):
    bam, cram, reference = regional_files
    alignment = bam if file_type == "bam" else cram
    suffix = ".bai" if file_type == "bam" else ".crai"
    staging = alignment.parent / "staging"
    staging.mkdir()
    alias = staging / f"sample.{file_type}"
    alias.symlink_to(alignment)
    alias_index = Path(str(alias) + suffix)
    Path(str(alignment) + suffix).rename(alias_index)
    reference_args = ["-T", str(reference)] if file_type == "cram" else []
    # The native CLI accepts the layout; the wrapper must discover the same index.
    native = subprocess.check_output(["samtools", "view", "-c", *reference_args, str(alias), "chr1:103-108"], text=True)
    assert int(native) == 8
    result = regional_alignment_stats(
        str(alias), [selected_region()], reference_path=str(reference) if file_type == "cram" else None
    )
    assert_known_result(result["regions"][0])
    assert result["input"]["path"] == str(alignment)
    assert result["index"]["path"] == str(alias_index)
    assert not Path(str(alignment) + suffix).exists()


def test_index_beside_reference_symlink(regional_files):
    _bam, cram, reference = regional_files
    staging = reference.parent / "staging"
    staging.mkdir()
    alias = staging / "sample.fa"
    alias.symlink_to(reference)
    alias_index = Path(str(alias) + ".fai")
    Path(str(reference) + ".fai").rename(alias_index)
    native = subprocess.check_output(["samtools", "view", "-c", "-T", str(alias), str(cram), "chr1:103-108"], text=True)
    assert int(native) == 8
    result = regional_alignment_stats(str(cram), [selected_region()], reference_path=str(alias))
    assert_known_result(result["regions"][0])
    assert result["reference"]["path"] == str(reference)
    assert result["reference_index"]["path"] == str(alias_index)
    assert not Path(str(reference) + ".fai").exists()
