"""Validate ordered native argument arrays at API v2 adapter boundaries.

The wrapper owns input/output routing, parser modes, typed selection controls,
temporary files, indexes, references, and configured resource limits. Native
extras may extend a command but cannot replace those controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Sequence


class NativeArgumentError(ValueError):
    """Raised when a native argument crosses a wrapper-owned boundary."""


@dataclass(frozen=True)
class NativeArgumentPolicy:
    protected_long: frozenset[str] = frozenset()
    protected_short: frozenset[str] = frozenset()
    value_long: frozenset[str] = frozenset()
    value_short: frozenset[str] = frozenset()
    flag_long: frozenset[str] = frozenset()
    flag_short: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ValidatedNativeArgs:
    namespace: str
    supplied_args: tuple[str, ...]
    effective_args: tuple[str, ...]
    native_options_used: bool
    reuse_safe: bool


def _policy(
    *,
    long: Iterable[str] = (),
    short: Iterable[str] = (),
    value_long: Iterable[str] = (),
    value_short: Iterable[str] = (),
    flag_long: Iterable[str] = (),
    flag_short: Iterable[str] = (),
) -> NativeArgumentPolicy:
    return NativeArgumentPolicy(
        frozenset(long),
        frozenset(short),
        frozenset(value_long),
        frozenset(value_short),
        frozenset(flag_long),
        frozenset(flag_short),
    )


NATIVE_ARGUMENT_POLICIES: Final[dict[str, NativeArgumentPolicy]] = {
    "samtools_view": _policy(
        long={
            "--add-flags",
            "--bam",
            "--count",
            "--cram",
            "--customized-index",
            "--exclude-flags",
            "--excl-flags",
            "--fai-reference",
            "--fast",
            "--fetch-pairs",
            "--header",
            "--header-only",
            "--include-flags",
            "--incl-flags",
            "--input-fmt",
            "--input-fmt-option",
            "--keep-tag",
            "--min-MQ",
            "--no-header",
            "--no-PG",
            "--only-header",
            "--output",
            "--output-fmt",
            "--output-fmt-option",
            "--output-unselected",
            "--region-file",
            "--regions-file",
            "--reference",
            "--remove-tag",
            "--remove-B",
            "--remove-flags",
            "--require-flags",
            "--rf",
            "--sanitize",
            "--save-counts",
            "--target-file",
            "--targets-file",
            "--threads",
            "--uncompressed",
            "--unmap",
            "--unoutput",
            "--use-index",
            "--with-header",
            "--write-index",
        },
        short={
            "-1",
            "-@",
            "-b",
            "-B",
            "-c",
            "-C",
            "-f",
            "-F",
            "-G",
            "-h",
            "-H",
            "-L",
            "-M",
            "-o",
            "-O",
            "-p",
            "-P",
            "-q",
            "-t",
            "-T",
            "-u",
            "-U",
            "-x",
            "-X",
            "-z",
        },
        value_long={
            "--expr",
            "--library",
            "--min-qlen",
            "--qname-file",
            "--read-group",
            "--read-group-file",
            "--subsample",
            "--subsample-seed",
            "--tag",
            "--tag-file",
        },
        value_short={"-d", "-D", "-e", "-l", "-m", "-N", "-r", "-R", "-s"},
        flag_long={"--exclude-no-read-group"},
        flag_short={"-S", "-n"},
    ),
    "samtools_fastq": _policy(
        long={
            "--exclude-flags",
            "--excl-flags",
            "--include-flags",
            "--incl-flags",
            "--input-fmt",
            "--input-fmt-option",
            "--i1",
            "--i2",
            "--no-sc",
            "--no-sc-bkp",
            "--output-fmt-option",
            "--reference",
            "--rf",
            "--sc-aux",
            "--threads",
        },
        short={"-0", "-1", "-2", "-@", "-c", "-d", "-D", "-f", "-F", "-G", "-o", "-O", "-s", "-v"},
        value_long={"--barcode-tag", "--index-format", "--quality-tag", "--UMI-tag"},
        value_short={"-T"},
        flag_long={"--UMI"},
        flag_short={"-i", "-n", "-N", "-t", "-U"},
    ),
    "samtools_stats": _policy(
        long={
            "--coverage",
            "--filtering-flag",
            "--input-fmt",
            "--input-fmt-option",
            "--reference",
            "--ref-seq",
            "--remove-dups",
            "--required-flag",
            "--split",
            "--split-prefix",
            "--target-regions",
            "--threads",
        },
        short={"-@", "-c", "-d", "-f", "-F", "-P", "-r", "-S", "-t", "-X"},
        value_long={
            "--cov-threshold",
            "--GC-depth",
            "--id",
            "--insert-size",
            "--most-inserts",
            "--read-length",
            "--ref-stats-chunk",
            "--trim-quality",
        },
        value_short={"-g", "-i", "-I", "-l", "-m", "-q"},
        flag_long={"--ref-stats", "--remove-overlaps", "--sparse"},
        flag_short={"-p", "-x"},
    ),
    "nanoq": _policy(
        long={
            "--compress-level",
            "--header",
            "--input",
            "--json",
            "--lengths",
            "--output",
            "--output-type",
            "--qualities",
            "--read-lengths",
            "--read-qualities",
            "--report",
            "--stats",
            "--threads",
        },
        short={"-c", "-H", "-i", "-j", "-L", "-o", "-O", "-Q", "-r", "-s"},
        value_long={"--max-len", "--max-qual", "--min-len", "--min-qual", "--top", "--trim-end", "--trim-start"},
        value_short={"-E", "-l", "-m", "-q", "-S", "-t", "-w"},
        flag_long={"--fast", "--verbose"},
        flag_short={"-f", "-v"},
    ),
    "cramino": _policy(
        long={"--arrow", "--format", "--hist", "--hist-count", "--reference", "--scaled", "--threads"},
        short={"-r", "-t"},
        value_long={"--min-read-len"},
        value_short={"-m"},
        flag_long={"--checksum", "--karyotype", "--phased", "--spliced", "--ubam"},
    ),
    "mosdepth": _policy(
        long={
            "--by",
            "--chrom",
            "--exclude",
            "--fasta",
            "--flag",
            "--include",
            "--include-flag",
            "--mapq",
            "--no-per-base",
            "--output",
            "--prefix",
            "--quantize",
            "--read-groups",
            "--thresholds",
            "--threads",
            "--use-median",
            "--d4",
        },
        short={"-b", "-c", "-f", "-F", "-i", "-n", "-Q", "-R", "-t"},
        value_long={"--max-frag-len", "--min-frag-len"},
        value_short={"-l", "-u"},
        flag_long={"--fast-mode", "--fragment-mode"},
        flag_short={"-a", "-x"},
    ),
    "bcftools_stats": _policy(
        long={
            "--exclude",
            "--fasta-ref",
            "--include",
            "--output",
            "--regions",
            "--regions-file",
            "--regions-overlap",
            "--threads",
            "--targets",
            "--targets-file",
            "--targets-overlap",
            "--write-index",
        },
        short={"-e", "-F", "-i", "-o", "-r", "-R", "-t", "-T", "-W"},
        value_long={
            "--af-bins",
            "--af-tag",
            "--apply-filters",
            "--collapse",
            "--depth",
            "--exons",
            "--samples",
            "--samples-file",
            "--user-tstv",
            "--verbosity",
        },
        value_short={"-c", "-d", "-E", "-f", "-s", "-S", "-u", "-v"},
        flag_long={"--1st-allele-only", "--debug", "--split-by-ID"},
        flag_short={"-1", "-I"},
    ),
    "chopper": _policy(
        long={
            "--cutoff",
            "--headcrop",
            "--input",
            "--inverse",
            "--maxlength",
            "--minlength",
            "--output",
            "--quality",
            "--tailcrop",
            "--threads",
            "--trim-approach",
        },
        short={"-i", "-o", "-q", "-t"},
    ),
}

# These options change which sequences nanoq retains or the sequence lengths it
# measures. The read-QC response has explicit samtools-selection and conversion
# denominators, so the family adapter cannot accept a second uncounted population
# change without weakening that accounting contract.
READ_QC_NANOQ_POPULATION_ARGS: Final[tuple[str, ...]] = (
    "--max-len",
    "--max-qual",
    "--min-len",
    "--min-qual",
    "--top",
    "--trim-end",
    "--trim-start",
    "-E",
    "-l",
    "-m",
    "-q",
    "-S",
    "-t",
    "-w",
)


def _long_option_kind(option: str, policy: NativeArgumentPolicy) -> tuple[str, str]:
    if option in policy.protected_long:
        return "protected", option
    if option in policy.value_long:
        return "value", option
    if option in policy.flag_long:
        return "flag", option
    protected_matches = sorted(candidate for candidate in policy.protected_long if candidate.startswith(option))
    if protected_matches:
        return "protected", protected_matches[0]
    return "unknown", option


def _short_option_kind(argument: str, policy: NativeArgumentPolicy) -> tuple[str, str | None]:
    body = argument[1:]
    offset = 0
    while offset < len(body):
        option = f"-{body[offset]}"
        if option in policy.protected_short:
            return "protected", option
        if option in policy.value_short:
            return ("self_contained", None) if offset + 1 < len(body) else ("value", option)
        if option not in policy.flag_short:
            return "unknown", option
        offset += 1
    return "self_contained", None


def validate_native_args(
    namespace: str,
    argv: Sequence[str],
    *,
    owned_args: Sequence[str] = (),
    additionally_protected: Sequence[str] = (),
) -> ValidatedNativeArgs:
    """Validate one backend/subcommand argument vector without reordering it.

    Known advanced options accept separate values according to the pinned CLI
    grammar. Unknown options remain available in self-contained or
    ``--option=value`` form for forward compatibility, but cannot authorize a
    following positional input or region.

    Any native extras conservatively disable cache and in-flight reuse. This
    avoids stale results when a native option names an external file whose
    content identity is not represented by the argument string.
    """
    if namespace not in NATIVE_ARGUMENT_POLICIES:
        raise NativeArgumentError(f"Unknown native argument namespace: {namespace}")
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise NativeArgumentError(f"{namespace} native arguments must be an ordered array of strings")

    policy = NATIVE_ARGUMENT_POLICIES[namespace]
    dynamic_long = frozenset(option for option in additionally_protected if option.startswith("--"))
    dynamic_short = frozenset(
        option for option in additionally_protected if option.startswith("-") and not option.startswith("--")
    )
    if any(option == "-" or not option.startswith("-") for option in additionally_protected):
        raise NativeArgumentError("additionally protected native options must use short or long option spelling")
    policy = NativeArgumentPolicy(
        protected_long=policy.protected_long | dynamic_long,
        protected_short=policy.protected_short | dynamic_short,
        value_long=policy.value_long - dynamic_long,
        value_short=policy.value_short - dynamic_short,
        flag_long=policy.flag_long - dynamic_long,
        flag_short=policy.flag_short - dynamic_short,
    )
    values: list[str] = []
    previous_accepts_value = False
    for index, argument in enumerate(argv):
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise NativeArgumentError(f"{namespace} native argument {index} must be a nonempty string without NUL")
        if previous_accepts_value:
            values.append(argument)
            previous_accepts_value = False
            continue
        if argument == "--":
            raise NativeArgumentError(f"{namespace} cannot use '--' because positional routing is wrapper-owned")
        if argument == "-":
            raise NativeArgumentError(
                f"{namespace} native argument '-' is positional; inputs and outputs are wrapper-owned"
            )
        if argument.startswith("--"):
            option, separator, _ = argument.partition("=")
            kind, canonical_long = _long_option_kind(option, policy)
            if kind == "protected":
                raise NativeArgumentError(
                    f"{namespace} option {canonical_long!r} conflicts with a typed or wrapper-owned control"
                )
            values.append(argument)
            previous_accepts_value = kind == "value" and not separator
            continue
        if argument.startswith("-"):
            kind, canonical_short = _short_option_kind(argument, policy)
            if kind == "protected":
                raise NativeArgumentError(
                    f"{namespace} option {canonical_short!r} conflicts with a typed or wrapper-owned control"
                )
            if kind == "unknown":
                raise NativeArgumentError(
                    f"{namespace} option {canonical_short!r} is not in the pinned short-option grammar"
                )
            values.append(argument)
            previous_accepts_value = kind == "value"
            continue
        if not previous_accepts_value:
            raise NativeArgumentError(
                f"{namespace} native argument {argument!r} is positional; "
                "inputs, outputs, and regions are wrapper-owned"
            )
        values.append(argument)
    if previous_accepts_value:
        raise NativeArgumentError(f"{namespace} native option {values[-1]!r} requires a value")

    supplied = tuple(values)
    effective = (*tuple(owned_args), *supplied)
    return ValidatedNativeArgs(
        namespace=namespace,
        supplied_args=supplied,
        effective_args=effective,
        native_options_used=bool(supplied),
        reuse_safe=not supplied,
    )


__all__ = [
    "NATIVE_ARGUMENT_POLICIES",
    "READ_QC_NANOQ_POPULATION_ARGS",
    "NativeArgumentError",
    "NativeArgumentPolicy",
    "ValidatedNativeArgs",
    "validate_native_args",
]
