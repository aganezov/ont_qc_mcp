"""Protected native-option boundaries for API v2 backends."""

import pytest
from pydantic import ValidationError

from ont_qc_mcp.v2_contracts import AlignmentQCRequest, CoverageQCRequest, FilterReadsRequest, ReadQCRequest
from ont_qc_mcp.v2_native_args import NativeArgumentError, validate_native_args


def test_empty_and_ordered_native_args_are_reported_exactly() -> None:
    empty = validate_native_args("samtools_view", [])
    assert empty.supplied_args == ()
    assert empty.effective_args == ()
    assert empty.native_options_used is False
    assert empty.reuse_safe is True

    ordered = validate_native_args("bcftools_stats", ["--samples", "S1,S2", "--samples=S3"], owned_args=["stats"])
    assert ordered.supplied_args == ("--samples", "S1,S2", "--samples=S3")
    assert ordered.effective_args == ("stats", "--samples", "S1,S2", "--samples=S3")
    assert ordered.native_options_used is True
    assert ordered.reuse_safe is False


@pytest.mark.parametrize(
    "argv",
    [
        ["-q", "20"],
        ["-q20"],
        ["--min-MQ", "20"],
        ["--min-MQ=20"],
        ["-F1796"],
        ["--exclude-flags=0x704"],
        ["-L", "targets.bed"],
        ["--target-file=targets.bed"],
        ["-x", "NM"],
        ["--remove-tag=NM"],
        ["--keep-tag", "RG"],
        ["--targets-file=targets.bed"],
        ["--include-flags", "0x2"],
        ["--with-header"],
        ["-P"],
        ["-zall"],
        ["--"],
    ],
)
def test_samtools_view_rejects_typed_routing_parser_and_tag_conflicts(argv: list[str]) -> None:
    with pytest.raises(NativeArgumentError):
        validate_native_args("samtools_view", argv)


@pytest.mark.parametrize(
    ("model", "extra_args", "message"),
    [
        (ReadQCRequest, {"samtools_fastq": ["-o", "reads.fastq"]}, "samtools_fastq"),
        (AlignmentQCRequest, {"samtools_stats": ["--threads=8"]}, "samtools_stats"),
        (CoverageQCRequest, {"mosdepth": ["--mapq", "30"]}, "mosdepth"),
    ],
)
def test_contract_models_reject_protected_native_options(model, extra_args, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        model.model_validate({"path": "reads.bam", "extra_args": extra_args})


@pytest.mark.parametrize("native_args", [["--use-median"], ["-m"]])
def test_coverage_contract_owns_median_native_options(native_args: list[str]) -> None:
    with pytest.raises(ValidationError, match="wrapper-owned"):
        CoverageQCRequest.model_validate({"path": "reads.bam", "extra_args": {"mosdepth": native_args}})


@pytest.mark.parametrize("model", [ReadQCRequest, AlignmentQCRequest])
@pytest.mark.parametrize("native_args", [["-e", "mapq >= 0"], ["--expr=mapq >= 0"], ["--exp=mapq >= 0"]])
def test_positive_typed_mapq_rejects_native_expression_at_contract_boundary(model, native_args: list[str]) -> None:
    with pytest.raises(ValidationError, match="wrapper-owned"):
        model.model_validate(
            {
                "path": "reads.bam",
                "selection": {"min_mapq": 1},
                "extra_args": {"samtools_view": native_args},
            }
        )


@pytest.mark.parametrize("model", [ReadQCRequest, AlignmentQCRequest])
def test_zero_typed_mapq_allows_native_expression_at_contract_boundary(model) -> None:
    request = model.model_validate(
        {
            "path": "reads.bam",
            "selection": {"min_mapq": 0},
            "extra_args": {"samtools_view": ["-e", "mapq >= 0"]},
        }
    )
    assert request.extra_args.samtools_view == ["-e", "mapq >= 0"]


def test_filter_contract_rejects_native_conflict_with_typed_selection() -> None:
    with pytest.raises(ValidationError, match="chopper"):
        FilterReadsRequest.model_validate(
            {"path": "reads.fastq", "selection": {"quality": 10}, "extra_args": {"chopper": ["-q10"]}}
        )


@pytest.mark.parametrize(
    ("namespace", "argv"),
    [
        ("samtools_stats", ["--target-regions", "targets.bed"]),
        ("samtools_stats", ["--split-prefix=out"]),
        ("samtools_view", ["-tref.fai"]),
        ("samtools_view", ["-B"]),
        ("samtools_fastq", ["--exclude-flags=0x900"]),
        ("samtools_fastq", ["-D", "RG:read-groups.txt"]),
        ("samtools_fastq", ["--no-sc"]),
        ("nanoq", ["--read-lengths=lengths.txt"]),
        ("nanoq", ["-Qqualities.txt"]),
        ("cramino", ["--hist=hist.png"]),
        ("mosdepth", ["-n"]),
        ("bcftools_stats", ["--regions-overlap=1"]),
        ("bcftools_stats", ["-Fref.fa"]),
    ],
)
def test_backend_aliases_cannot_bypass_wrapper_owned_controls(namespace: str, argv: list[str]) -> None:
    with pytest.raises(NativeArgumentError):
        validate_native_args(namespace, argv)


def test_unknown_option_can_reach_native_cli_but_bare_positionals_cannot() -> None:
    result = validate_native_args("cramino", ["--future-option=value"])
    assert result.supplied_args == ("--future-option=value",)
    with pytest.raises(NativeArgumentError, match="positional"):
        validate_native_args("cramino", ["other-input.bam"])
    with pytest.raises(NativeArgumentError, match="positional"):
        validate_native_args("cramino", ["--future-option", "-"])
    with pytest.raises(NativeArgumentError, match="positional"):
        validate_native_args("cramino", ["--future-flag", "other-input.bam"])
    with pytest.raises(NativeArgumentError, match="positional"):
        validate_native_args("samtools_fastq", ["-n", "other-input.bam"])


def test_known_advanced_options_preserve_separate_values_and_order() -> None:
    result = validate_native_args(
        "samtools_view",
        ["-e", "[HP] == 1", "--subsample", "0.5", "--subsample-seed=7"],
    )
    assert result.supplied_args == ("-e", "[HP] == 1", "--subsample", "0.5", "--subsample-seed=7")

    clustered = validate_native_args("samtools_view", ["-Sn", "--subsample", "0.5"])
    assert clustered.supplied_args == ("-Sn", "--subsample", "0.5")


@pytest.mark.parametrize(
    ("namespace", "argv"),
    [
        ("samtools_view", ["--reference", "alternate.fa"]),
        ("samtools_view", ["--excl-fl=0"]),
        ("samtools_fastq", ["-nv20"]),
    ],
)
def test_backend_grammar_cannot_bypass_protected_options(namespace: str, argv: list[str]) -> None:
    with pytest.raises(NativeArgumentError):
        validate_native_args(namespace, argv)


@pytest.mark.parametrize("argv", [["-Sq0"], ["-Se1"]])
def test_recognized_short_flag_cluster_cannot_hide_protected_option(argv: list[str]) -> None:
    with pytest.raises(NativeArgumentError, match="wrapper-owned"):
        validate_native_args(
            "samtools_view",
            argv,
            additionally_protected=("-e", "--expr"),
        )


def test_unknown_short_option_is_rejected() -> None:
    with pytest.raises(NativeArgumentError, match="pinned short-option grammar"):
        validate_native_args("samtools_view", ["-Z"])


@pytest.mark.parametrize("bad", [[""], ["ok\x00bad"], [1]])
def test_native_argv_tokens_are_safe_strings(bad: list[object]) -> None:
    with pytest.raises(NativeArgumentError):
        validate_native_args("nanoq", bad)  # type: ignore[arg-type]
