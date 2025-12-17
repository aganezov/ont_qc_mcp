import json
import re
from pathlib import Path
from typing import Literal, Sequence, cast

from .schemas import (
    BedIssue,
    BedQCReport,
    CraminoStats,
    ErrorProfile,
    HeaderMetadata,
    HistogramBin,
    IndelStats,
    LengthPercentiles,
    LowCoverageRegion,
    MosdepthStats,
    NanoqStats,
    QScoreDistribution,
    ProgramRecord,
    ReadLengthDistribution,
    ReferenceRecord,
    RunYieldWindow,
    SampleRecord,
    SNPStats,
    SequencingSummaryStats,
    VCFFieldDef,
    VCFStats,
    VariantGeneralStats,
    CoverageByContig,
)


def _histogram_from_seq(bins: Sequence[Sequence[float]]) -> list[HistogramBin]:
    histogram: list[HistogramBin] = []
    for entry in bins:
        if len(entry) < 3:
            continue
        start, end, count = entry[0], entry[1], entry[2]
        histogram.append(HistogramBin(start=float(start), end=float(end), count=int(count)))
    return histogram


def _histogram_or_none(bins: Sequence[Sequence[float]] | None, present: bool) -> list[HistogramBin] | None:
    if not present:
        return None
    if not bins:
        return []
    return _histogram_from_seq(bins)


def _safe_percentiles(data: dict[str, float]) -> LengthPercentiles:
    return LengthPercentiles(
        p1=data.get("p1"),
        p5=data.get("p5"),
        p25=data.get("p25"),
        p50=data.get("p50") or data.get("median"),
        p75=data.get("p75"),
        p95=data.get("p95"),
        p99=data.get("p99"),
    )


def parse_nanoq_json(payload: str | dict) -> NanoqStats:
    """
    Parse nanoq --stats --json output.
    The JSON schema can vary by version; we defensively access keys.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid nanoq JSON: {exc}") from exc
    summary = data.get("summary", data)

    # nanoq >=0.10 emits flat keys; older versions nest under summary.reads.length/qscore.
    raw_reads = summary.get("reads", summary)
    if isinstance(raw_reads, dict):
        read_stats = raw_reads
        length_info = read_stats.get("length", {}) if isinstance(read_stats, dict) else {}
        qscore_info = read_stats.get("qscore", {}) if isinstance(read_stats, dict) else {}
    else:
        # Flat schema: pull directly from top-level keys.
        read_stats = {"count": raw_reads, "bases": summary.get("bases")}
        length_info = {
            "min": summary.get("shortest"),
            "max": summary.get("longest"),
            "mean": summary.get("mean_length"),
            "median": summary.get("median_length"),
            "n50": summary.get("n50"),
        }
        # Histogram/percentiles are not present in this schema; leave empty.
        qscore_info = {
            "mean": summary.get("mean_quality"),
            "median": summary.get("median_quality"),
        }

    file_name = summary.get("file") or summary.get("input") or "unknown"
    if not isinstance(length_info, dict):
        length_info = {}
    if not isinstance(qscore_info, dict):
        qscore_info = {}

    length_bins_present = isinstance(length_info, dict) and ("hist" in length_info or "histogram" in length_info)
    qscore_bins_present = isinstance(qscore_info, dict) and ("hist" in qscore_info or "histogram" in qscore_info)
    length_bins_raw = None
    qscore_bins_raw = None
    if isinstance(length_info, dict):
        length_bins_raw = length_info.get("hist")
        if length_bins_raw is None:
            length_bins_raw = length_info.get("histogram")
    if isinstance(qscore_info, dict):
        qscore_bins_raw = qscore_info.get("hist")
        if qscore_bins_raw is None:
            qscore_bins_raw = qscore_info.get("histogram")

    percentiles = (
        (
            _safe_percentiles(length_info.get("percentiles", {}))
            if isinstance(length_info, dict) and "percentiles" in length_info
            else None
        )
        if isinstance(length_info, dict)
        else None
    )

    # Fallbacks to keep previous defaults when values are missing.
    read_count = (
        read_stats.get("count", read_stats.get("reads", 0) or 0)
        if isinstance(read_stats, dict)
        else int(raw_reads or 0)
    )
    total_bases = (
        read_stats.get("bases", read_stats.get("total_bases", 0) or 0)
        if isinstance(read_stats, dict)
        else summary.get("bases", 0)
    )

    for name, val in (("read_count", read_count), ("total_bases", total_bases)):
        if val is not None and val < 0:
            raise ValueError(f"Invalid nanoq value: {name} must be non-negative, got {val}")

    return NanoqStats(
        file=file_name,
        read_count=int(read_count or 0),
        total_bases=int(total_bases or 0),
        min_len=int(length_info.get("min", 0) or 0),
        max_len=int(length_info.get("max", 0) or 0),
        mean_len=float(length_info.get("mean", 0.0) or 0.0),
        median_len=float(length_info.get("median", length_info.get("p50", 0.0) or 0.0)),
        n50=length_info.get("n50"),
        mean_qscore=float(qscore_info.get("mean", qscore_info.get("average", 0.0) or 0.0)) if qscore_info else None,
        median_qscore=float(qscore_info.get("median", qscore_info.get("p50", 0.0) or 0.0)) if qscore_info else None,
        gc_content=read_stats.get("gc") if isinstance(read_stats, dict) else summary.get("gc"),
        length_percentiles=percentiles,
        length_histogram=_histogram_or_none(length_bins_raw, length_bins_present),
        qscore_histogram=_histogram_or_none(qscore_bins_raw, qscore_bins_present),
    )


def parse_read_length_distribution(payload: str | dict) -> ReadLengthDistribution:
    data = json.loads(payload) if isinstance(payload, str) else payload
    stats = parse_nanoq_json(data)
    return ReadLengthDistribution(
        file=stats.file,
        percentiles=stats.length_percentiles or LengthPercentiles(),
        histogram=stats.length_histogram or [],
    )


def parse_qscore_distribution(payload: str | dict) -> QScoreDistribution:
    data = json.loads(payload) if isinstance(payload, str) else payload
    stats = parse_nanoq_json(data)
    return QScoreDistribution(
        file=stats.file,
        mean_qscore=stats.mean_qscore,
        median_qscore=stats.median_qscore,
        histogram=stats.qscore_histogram or [],
        per_position_mean=None,
    )


def parse_cramino_json(
    payload: str | dict,
    length_bins: list[HistogramBin] | None = None,
    length_bins_scaled: list[HistogramBin] | None = None,
) -> CraminoStats:
    """
    Parse cramino JSON output (e.g., --format json), supporting both count and scaled histograms.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid cramino JSON: {exc}") from exc
    summary = data.get("summary", data)
    # Support older summary.reads schema and newer cramino >=0.15 schema with alignment_stats/read_stats/identity_stats.
    alignment_stats = summary.get("alignment_stats", {}) if isinstance(summary, dict) else {}
    read_stats = summary.get("read_stats", {}) if isinstance(summary, dict) else {}
    identity_stats = summary.get("identity_stats", {}) if isinstance(summary, dict) else {}

    legacy_read_counts = summary.get("reads", summary)
    if alignment_stats:
        read_counts = {
            "total": alignment_stats.get("num_reads") or alignment_stats.get("num_alignments") or 0,
            # cramino JSON in this version does not emit mapped/unmapped breakdown.
        }
    elif isinstance(legacy_read_counts, dict):
        read_counts = legacy_read_counts
    else:
        read_counts = {"total": legacy_read_counts or 0}

    mapq_bins_counts = summary.get("mapq_hist", summary.get("mapq_histogram", []))
    mapq_bins_scaled = summary.get("mapq_hist_scaled") or summary.get("mapq_hist_scaled_bp") or []

    file_info = summary.get("file_info", {}) if isinstance(summary, dict) else {}
    file_path = file_info.get("path") or file_info.get("name") or summary.get("file", "unknown")

    mapped_val = read_counts.get("mapped") if isinstance(read_counts, dict) else None
    unmapped_val = read_counts.get("unmapped") if isinstance(read_counts, dict) else None

    return CraminoStats(
        file=file_path,
        total_reads=int(read_counts.get("total", read_counts.get("reads", 0) or 0)),
        mapped=int(mapped_val) if mapped_val is not None else None,
        unmapped=int(unmapped_val) if unmapped_val is not None else None,
        primary=read_counts.get("primary") if isinstance(read_counts, dict) else None,
        secondary=read_counts.get("secondary") if isinstance(read_counts, dict) else None,
        supplementary=read_counts.get("supplementary") if isinstance(read_counts, dict) else None,
        mean_length=summary.get("mean_length") or read_stats.get("mean_length"),
        median_length=summary.get("median_length") or read_stats.get("median_length"),
        n50=summary.get("n50") or read_stats.get("n50"),
        mean_identity=summary.get("mean_identity") or identity_stats.get("mean_identity"),
        median_identity=summary.get("median_identity") or identity_stats.get("median_identity"),
        length_histogram=length_bins or None,
        length_histogram_scaled=length_bins_scaled or None,
        mapq_histogram=_histogram_from_seq(mapq_bins_counts),
        mapq_histogram_scaled=_histogram_from_seq(mapq_bins_scaled) if mapq_bins_scaled else None,
    )


def parse_mosdepth_summary(text: str, file_path: str, threshold: float | int | None = None) -> MosdepthStats:
    """
    Parse mosdepth .summary.txt output.
    Expected columns: chrom, length, bases, mean
    """
    coverage_by_contig: list[CoverageByContig] = []
    coverage_distribution: list[HistogramBin] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("chrom"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        contig, length, _, mean = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
        coverage_by_contig.append(CoverageByContig(contig=contig, length=length, mean_depth=mean, median_depth=None))

    total_len = sum(c.length or 0 for c in coverage_by_contig)
    unweighted_mean = (
        sum(c.mean_depth for c in coverage_by_contig) / len(coverage_by_contig) if coverage_by_contig else 0.0
    )
    weighted_mean = (
        sum((c.mean_depth) * (c.length or 0) for c in coverage_by_contig) / total_len if total_len else unweighted_mean
    )

    low_cov: list[LowCoverageRegion] = []
    if threshold is not None:
        for c in coverage_by_contig:
            if c.mean_depth < float(threshold):
                low_cov.append(
                    LowCoverageRegion(
                        contig=c.contig,
                        start=0,
                        end=c.length or 0,
                        mean_depth=c.mean_depth,
                    )
                )

    return MosdepthStats(
        file=file_path,
        mean_depth=weighted_mean,
        mean_depth_unweighted=unweighted_mean,
        coverage_distribution=coverage_distribution,
        median_depth=None,
        coverage_by_contig=coverage_by_contig,
        low_coverage_regions=low_cov,
    )


def parse_error_profile(text: str, file_path: str) -> ErrorProfile:
    """
    Parse error-related metrics from samtools stats output.
    Falls back gracefully when metrics are absent.
    """
    metrics: dict[str, str] = {}
    coverage_hist: list[HistogramBin] = []
    gc_cov: list[HistogramBin] = []
    mismatch_by_cycle: dict[int, float] = {}
    insert_hist: list[HistogramBin] = []
    pattern = re.compile(r"^SN\t(.+?):\t(.+)")
    for line in text.splitlines():
        if line.startswith("COV\t"):
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[1].isdigit():
                depth = int(parts[1])
                try:
                    count = float(parts[2])
                except ValueError:
                    continue
                coverage_hist.append(HistogramBin(start=depth, end=depth, count=int(count)))
            continue
        if line.startswith("GCD\t"):
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3].isdigit():
                try:
                    gc_pct = float(parts[1])
                    count = int(parts[3])
                except ValueError:
                    continue
                gc_cov.append(HistogramBin(start=gc_pct, end=gc_pct, count=count))
            continue
        if line.startswith("MPC\t"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    cycle = int(parts[1])
                    rate = float(parts[2])
                    mismatch_by_cycle[cycle] = rate
                except ValueError:
                    pass
            continue
        if line.startswith("IS\t"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    insert_size = float(parts[1])
                    count = float(parts[2])
                except ValueError:
                    continue
                insert_hist.append(HistogramBin(start=insert_size, end=insert_size, count=int(count)))
            continue

        match = pattern.match(line)
        if match:
            key, value = match.group(1).strip().lower(), match.group(2).strip()
            metrics[key] = value

    def to_float(key: str) -> float | None:
        raw = metrics.get(key)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    mismatch_rate = to_float("mismatches per base")
    insertion_rate = to_float("insertions per base")
    deletion_rate = to_float("deletions per base")
    error_rate = to_float("error rate")

    # Prefer specific rates; fall back to overall error rate
    if mismatch_rate is None and error_rate is not None:
        mismatch_rate = error_rate

    mismatch_by_cycle_list = [rate for _, rate in sorted(mismatch_by_cycle.items())] if mismatch_by_cycle else None

    return ErrorProfile(
        file=file_path,
        mismatch_rate=mismatch_rate,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        error_by_position=None,
        coverage_histogram=coverage_hist or None,
        gc_coverage=gc_cov or None,
        mismatch_by_cycle=mismatch_by_cycle_list,
        insert_size_histogram=insert_hist or None,
    )


def _parse_tag_fields(parts: Sequence[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key] = value
    return fields


def parse_alignment_header(header_text: str, file_path: str, fmt: str) -> HeaderMetadata:
    """
    Parse SAM/BAM/CRAM header text into structured metadata.
    """
    references: list[ReferenceRecord] = []
    programs: list[ProgramRecord] = []
    samples: list[SampleRecord] = []
    comments: list[str] = []
    metadata: dict[str, str] = {}
    header_other: dict[str, str] = {}

    for line in header_text.splitlines():
        if not line.startswith("@"):
            continue
        parts = line.strip().split("\t")
        if not parts:
            continue
        tag = parts[0][1:]
        fields = _parse_tag_fields(parts[1:])

        if tag == "HD":
            if version := fields.get("VN"):
                metadata["version"] = version
            if sort_order := fields.get("SO"):
                metadata["sort_order"] = sort_order
            if group_order := fields.get("GO"):
                metadata["group_order"] = group_order
            for key, value in fields.items():
                if key not in {"VN", "SO", "GO"}:
                    header_other[key] = value
        elif tag == "SQ":
            name = fields.get("SN")
            if not name:
                continue
            length_raw = fields.get("LN")
            length = int(length_raw) if length_raw and length_raw.isdigit() else None
            known_keys = {"SN", "LN", "AS", "UR", "M5"}
            other_fields = {k: v for k, v in fields.items() if k not in known_keys}
            references.append(
                ReferenceRecord(
                    name=name,
                    length=length,
                    assembly=fields.get("AS"),
                    uri=fields.get("UR"),
                    md5=fields.get("M5"),
                    other=other_fields,
                )
            )
        elif tag == "RG":
            sample_name = fields.get("SM") or fields.get("ID") or "unknown"
            known_keys = {"ID", "SM", "LB", "PL", "PU", "CN", "DT", "PM", "FC", "DS"}
            other_fields = {k: v for k, v in fields.items() if k not in known_keys}
            samples.append(
                SampleRecord(
                    name=sample_name,
                    read_group_id=fields.get("ID"),
                    library=fields.get("LB"),
                    platform=fields.get("PL"),
                    platform_unit=fields.get("PU"),
                    sequencing_center=fields.get("CN"),
                    run_date=fields.get("DT"),
                    flowcell_id=fields.get("FC"),
                    platform_model=fields.get("PM"),
                    description=fields.get("DS"),
                    other=other_fields,
                )
            )
        elif tag == "PG":
            program_id = fields.get("ID") or fields.get("PN") or "unknown"
            known_keys = {"ID", "PN", "VN", "CL", "PP"}
            other_fields = {k: v for k, v in fields.items() if k not in known_keys}
            programs.append(
                ProgramRecord(
                    id=program_id,
                    name=fields.get("PN"),
                    version=fields.get("VN"),
                    command=fields.get("CL"),
                    previous_id=fields.get("PP"),
                    other=other_fields,
                )
            )
        elif tag == "CO":
            comments.append("\t".join(parts[1:]))

    return HeaderMetadata(
        file=file_path,
        format=fmt.lower(),
        references=references,
        programs=programs,
        samples=samples,
        metadata=metadata,
        header_other=header_other,
        comments=comments,
        raw_header=header_text,
    )


def _parse_angle_bracket_fields(value: str) -> dict[str, str]:
    """
    Parse VCF-style key/value lists inside angle brackets, respecting quoted commas.
    """
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]

    fields: dict[str, str] = {}
    current: list[str] = []
    in_quotes = False

    for char in value:
        if char == '"' and (not current or current[-1] != "\\"):
            in_quotes = not in_quotes
        if char == "," and not in_quotes:
            token = "".join(current).strip()
            if token:
                key, _, val = token.partition("=")
                fields[key] = val.strip('"')
            current = []
            continue
        current.append(char)

    token = "".join(current).strip()
    if token:
        key, _, val = token.partition("=")
        fields[key] = val.strip('"')

    return fields


def parse_vcf_header(header_text: str, file_path: str) -> HeaderMetadata:
    """
    Parse VCF header text into structured metadata.
    Only header lines (## / #CHROM) are required.
    """
    references: list[ReferenceRecord] = []
    info_fields: list[VCFFieldDef] = []
    format_fields: list[VCFFieldDef] = []
    filter_fields: list[VCFFieldDef] = []
    samples: list[SampleRecord] = []
    metadata: dict[str, str] = {}
    comments: list[str] = []

    for line in header_text.splitlines():
        if not line.startswith("#"):
            continue
        if line.startswith("##"):
            key_val = line[2:]
            if "=" not in key_val:
                continue
            key, value = key_val.split("=", 1)
            key_upper = key.upper()
            if key_upper == "CONTIG":
                attrs = _parse_angle_bracket_fields(value)
                contig_id = attrs.get("ID")
                if contig_id:
                    length_raw = attrs.get("length") or attrs.get("Length")
                    length = int(length_raw) if length_raw and str(length_raw).isdigit() else None
                    known_keys = {"ID", "length", "Length", "assembly", "Assembly", "URI", "url", "md5"}
                    other_fields = {k: v for k, v in attrs.items() if k not in known_keys}
                    references.append(
                        ReferenceRecord(
                            name=contig_id,
                            length=length,
                            assembly=attrs.get("assembly") or attrs.get("Assembly"),
                            uri=attrs.get("URI") or attrs.get("url"),
                            md5=attrs.get("md5"),
                            other=other_fields,
                        )
                    )
            elif key_upper in {"INFO", "FORMAT", "FILTER"}:
                attrs = _parse_angle_bracket_fields(value)
                ident = attrs.get("ID")
                if not ident:
                    continue
                known_keys = {"ID", "Number", "Type", "Description"}
                other_fields = {k: v for k, v in attrs.items() if k not in known_keys}
                field_def = VCFFieldDef(
                    id=ident,
                    number=attrs.get("Number"),
                    type=attrs.get("Type"),
                    description=attrs.get("Description"),
                    category=cast(Literal["INFO", "FORMAT", "FILTER"], key_upper),
                    other=other_fields,
                )
                if key_upper == "INFO":
                    info_fields.append(field_def)
                elif key_upper == "FORMAT":
                    format_fields.append(field_def)
                else:
                    filter_fields.append(field_def)
            else:
                metadata[key] = value.strip()
        elif line.startswith("#CHROM"):
            columns = line.strip().split("\t")
            for sample_name in columns[9:]:
                samples.append(SampleRecord(name=sample_name))
            break
        else:
            comments.append(line)

    return HeaderMetadata(
        file=file_path,
        format="vcf",
        references=references,
        programs=[],
        samples=samples,
        metadata=metadata,
        info_fields=info_fields,
        format_fields=format_fields,
        filter_fields=filter_fields,
        comments=comments,
        raw_header=header_text,
    )


def summarize_header(metadata: HeaderMetadata) -> str:
    """Generate a concise human-friendly summary for a header."""
    parts: list[str] = []
    parts.append(f"{metadata.format.upper()} header")
    if metadata.references:
        contigs = ", ".join(ref.name for ref in metadata.references[:3])
        if len(metadata.references) > 3:
            contigs += ", …"
        parts.append(f"{len(metadata.references)} contigs (e.g., {contigs})")
    if metadata.samples:
        sample_names = ", ".join(sample.name for sample in metadata.samples[:3])
        if len(metadata.samples) > 3:
            sample_names += ", …"
        parts.append(f"samples: {sample_names}")
    if metadata.programs:
        program_names = ", ".join(prog.name or prog.id for prog in metadata.programs[:2])
        parts.append(f"programs: {program_names}")
    if sort_order := metadata.metadata.get("sort_order"):
        parts.append(f"sort order={sort_order}")
    if metadata.info_fields:
        parts.append(f"{len(metadata.info_fields)} INFO fields")
    if metadata.format_fields:
        parts.append(f"{len(metadata.format_fields)} FORMAT fields")
    extra_header = len(metadata.header_other)
    extra_ref = sum(len(ref.other) for ref in metadata.references)
    extra_prog = sum(len(pg.other) for pg in metadata.programs)
    extra_rg = sum(len(rg.other) for rg in metadata.samples)
    extra_vcf = (
        sum(len(f.other) for f in metadata.info_fields)
        + sum(len(f.other) for f in metadata.format_fields)
        + sum(len(f.other) for f in metadata.filter_fields)
    )
    if any([extra_header, extra_ref, extra_prog, extra_rg, extra_vcf]):
        parts.append(f"extras h={extra_header},sq={extra_ref},pg={extra_prog},rg={extra_rg},vcf={extra_vcf}")
    summary = "; ".join(parts)
    metadata.summary = summary
    return summary


def parse_sequencing_summary(file_path: Path) -> SequencingSummaryStats:
    """
    Parse ONT sequencing summary file (tab-separated).

    Expected columns: filename, read_id, run_id, channel, start_time,
    sequence_length_template, mean_qscore_template (some columns optional).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return SequencingSummaryStats(
            file=str(file_path),
            total_yield=0,
            total_reads=0,
        )

    # Parse header
    header_line = lines[0].strip()
    if not header_line:
        return SequencingSummaryStats(
            file=str(file_path),
            total_yield=0,
            total_reads=0,
        )

    columns = header_line.split("\t")
    column_map = {col.lower(): idx for idx, col in enumerate(columns)}

    # Find required columns (with flexible naming)
    length_col_idx = None
    qscore_col_idx = None
    start_time_col_idx = None
    channel_col_idx = None

    for col_name, idx in column_map.items():
        if "length" in col_name or "sequence_length" in col_name:
            length_col_idx = idx
        if "qscore" in col_name or "quality" in col_name:
            qscore_col_idx = idx
        if "start_time" in col_name or "time" in col_name:
            start_time_col_idx = idx
        if "channel" in col_name:
            channel_col_idx = idx

    if length_col_idx is None:
        raise ValueError(f"Required column 'sequence_length_template' not found in {file_path}")

    # Parse data rows
    lengths: list[int] = []
    qscores: list[float] = []
    start_times: list[float] = []
    channels: set[int] = set()

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) <= length_col_idx:
            continue

        try:
            length = int(parts[length_col_idx])
            lengths.append(length)

            if qscore_col_idx is not None and len(parts) > qscore_col_idx:
                try:
                    qscore = float(parts[qscore_col_idx])
                    qscores.append(qscore)
                except (ValueError, IndexError):
                    pass

            if start_time_col_idx is not None and len(parts) > start_time_col_idx:
                try:
                    start_time = float(parts[start_time_col_idx])
                    start_times.append(start_time)
                except (ValueError, IndexError):
                    pass

            if channel_col_idx is not None and len(parts) > channel_col_idx:
                try:
                    channel = int(parts[channel_col_idx])
                    channels.add(channel)
                except (ValueError, IndexError):
                    pass
        except (ValueError, IndexError):
            continue

    total_reads = len(lengths)
    total_yield = sum(lengths)

    # Calculate statistics
    mean_length = float(total_yield) / total_reads if total_reads > 0 else None
    mean_qscore = float(sum(qscores)) / len(qscores) if qscores else None
    active_channels = len(channels) if channels else None

    # Calculate N50
    n50 = None
    if lengths:
        sorted_lengths = sorted(lengths, reverse=True)
        cumulative = 0
        half_total = total_yield / 2.0
        for length in sorted_lengths:
            cumulative += length
            if cumulative >= half_total:
                n50 = length
                break

    # Calculate run duration
    run_duration_hours = None
    if start_times:
        run_duration_hours = max(start_times) - min(start_times)

    # Calculate yield per hour windows (1-hour bins)
    yield_per_hour: list[RunYieldWindow] = []
    if start_times and lengths:
        min_time = min(start_times)
        max_time = max(start_times)
        if max_time > min_time:
            # Create 1-hour windows
            window_size = 1.0  # hours
            num_windows = int((max_time - min_time) / window_size) + 1

            for window_idx in range(num_windows):
                window_start = min_time + (window_idx * window_size)
                window_end = window_start + window_size

                window_yield = 0
                window_reads = 0

                for i, start_time in enumerate(start_times):
                    if window_start <= start_time < window_end:
                        window_yield += lengths[i]
                        window_reads += 1

                if window_reads > 0:
                    yield_per_hour.append(
                        RunYieldWindow(
                            window_start_hours=window_start,
                            yield_bp=window_yield,
                            read_count=window_reads,
                        )
                    )

    return SequencingSummaryStats(
        file=str(file_path),
        total_yield=total_yield,
        total_reads=total_reads,
        n50=n50,
        mean_length=mean_length,
        mean_qscore=mean_qscore,
        active_channels=active_channels,
        run_duration_hours=run_duration_hours,
        yield_per_hour=yield_per_hour,
    )


def parse_bcftools_stats(stdout: str, include_snps: bool, include_indels: bool) -> VCFStats:
    """
    Parse bcftools stats output text.

    Extracts summary numbers (SN lines) and transition/transversion ratio (TSTV section).
    """
    general_stats = VariantGeneralStats(
        total_records=0,
        mnps=0,
        others=0,
        singletons=None,
        sample_name=None,
    )

    snp_count = 0
    indel_count = 0
    ts_tv_ratio = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        section = parts[0]

        if section == "SN":
            # Summary Numbers section
            key = parts[2].lower() if len(parts) > 2 else ""
            value = parts[3] if len(parts) > 3 else ""

            if "number of records" in key:
                try:
                    general_stats.total_records = int(value)
                except ValueError:
                    pass
            elif "number of mnps" in key:
                try:
                    general_stats.mnps = int(value)
                except ValueError:
                    pass
            elif "number of others" in key:
                try:
                    general_stats.others = int(value)
                except ValueError:
                    pass
            elif "number of singletons" in key:
                try:
                    general_stats.singletons = int(value)
                except ValueError:
                    pass
            elif "sample name" in key:
                general_stats.sample_name = value
            elif "number of snps" in key:
                try:
                    snp_count = int(value)
                except ValueError:
                    pass
            elif "number of indels" in key:
                try:
                    indel_count = int(value)
                except ValueError:
                    pass

        elif section == "TSTV":
            # Transitions/Transversions section
            # Format: TSTV <sample_idx> <ts_count> <tv_count> <ratio>
            # Or sometimes: TSTV <ts_count> <tv_count> <ratio> (without sample_idx)
            if len(parts) >= 4:
                try:
                    # Try parsing with sample_idx (4 fields: TSTV, idx, ts, tv, ratio)
                    if len(parts) >= 5:
                        ts_count = int(parts[2])
                        tv_count = int(parts[3])
                        ratio_str = parts[4]
                    else:
                        # Format without sample_idx: TSTV <ts_count> <tv_count> <ratio>
                        ts_count = int(parts[1])
                        tv_count = int(parts[2])
                        ratio_str = parts[3]
                    
                    # Try to parse ratio directly first
                    try:
                        ts_tv_ratio = float(ratio_str)
                    except ValueError:
                        # If ratio not provided, calculate from counts
                        if tv_count > 0:
                            ts_tv_ratio = float(ts_count) / float(tv_count)
                        elif ts_count == 0 and tv_count == 0:
                            ts_tv_ratio = None
                        else:
                            ts_tv_ratio = float("inf") if ts_count > 0 else 0.0
                except (ValueError, IndexError):
                    pass

    # Build result
    snps = None
    if include_snps:
        snps = SNPStats(count=snp_count, ts_tv_ratio=ts_tv_ratio)

    indels = None
    if include_indels:
        indels = IndelStats(count=indel_count)

    return VCFStats(
        file="",  # File path not available in stats output
        general=general_stats,
        snps=snps,
        indels=indels,
    )


def parse_bed_qc(file_path: Path) -> BedQCReport:
    """
    Parse and validate a BED file.

    Validates that start < end and coordinates are integers.
    Returns a report with validation results and issues.
    """
    issues: list[BedIssue] = []
    valid_intervals = 0
    total_bases = 0

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return BedQCReport(
            file=str(file_path),
            total_intervals=0,
            valid_intervals=0,
            total_bases=0,
            is_valid=False,
            issues=[BedIssue(line_number=0, line_content="", issue="File not found")],
        )

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            issues.append(
                BedIssue(
                    line_number=line_num,
                    line_content=line,
                    issue="Line has fewer than 3 columns (chrom, start, end required)",
                )
            )
            continue

        start_str = parts[1]
        end_str = parts[2]

        # Validate coordinates are integers
        try:
            start = int(start_str)
        except ValueError:
            issues.append(
                BedIssue(
                    line_number=line_num,
                    line_content=line,
                    issue=f"Start coordinate '{start_str}' is not an integer",
                )
            )
            continue

        try:
            end = int(end_str)
        except ValueError:
            issues.append(
                BedIssue(
                    line_number=line_num,
                    line_content=line,
                    issue=f"End coordinate '{end_str}' is not an integer",
                )
            )
            continue

        # Validate start < end
        if start >= end:
            issues.append(
                BedIssue(
                    line_number=line_num,
                    line_content=line,
                    issue=f"Start coordinate ({start}) >= end coordinate ({end})",
                )
            )
            continue

        # Valid interval
        valid_intervals += 1
        total_bases += end - start

    total_intervals = valid_intervals + len(issues)
    is_valid = len(issues) == 0

    return BedQCReport(
        file=str(file_path),
        total_intervals=total_intervals,
        valid_intervals=valid_intervals,
        total_bases=total_bases,
        is_valid=is_valid,
        issues=issues,
    )


def find_gene_coordinates(gff_path: Path, gene_name: str) -> list[tuple[str, int, int]]:
    """
    Find gene coordinates from a GFF3 file.

    Searches for genes by Name or ID attribute (case-insensitive).
    Returns list of (chrom, start, end) tuples for exons or gene span.
    """
    try:
        with open(gff_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    gene_name_lower = gene_name.lower()
    gene_found = False
    gene_chrom = None
    gene_start = None
    gene_end = None
    gene_id = None

    # First pass: find the gene by Name or ID
    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            continue

        parts = line.split("\t")
        if len(parts) < 9:
            continue

        feature_type = parts[2].lower()
        if feature_type != "gene":
            continue

        attributes_str = parts[8]
        # Parse attributes (key=value;key=value format)
        attrs: dict[str, str] = {}
        for attr_pair in attributes_str.split(";"):
            if "=" in attr_pair:
                key, value = attr_pair.split("=", 1)
                attrs[key.lower()] = value

        # Check if this gene matches
        gene_id_attr = attrs.get("id", "").lower()
        gene_name_attr = attrs.get("name", "").lower()

        if gene_id_attr == gene_name_lower or gene_name_attr == gene_name_lower:
            gene_chrom = parts[0]
            try:
                gene_start = int(parts[3])
                gene_end = int(parts[4])
                gene_id = attrs.get("id", "")
                gene_found = True
                break
            except (ValueError, IndexError):
                continue

    if not gene_found:
        return []

    # Second pass: find all exons for this gene
    # We need to find mRNAs/transcripts that are children of this gene,
    # then find exons that are children of those mRNAs
    mrna_ids: set[str] = set()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            continue

        parts = line.split("\t")
        if len(parts) < 9:
            continue

        feature_type = parts[2].lower()
        attributes_str = parts[8]
        mrna_attrs: dict[str, str] = {}
        for attr_pair in attributes_str.split(";"):
            if "=" in attr_pair:
                key, value = attr_pair.split("=", 1)
                mrna_attrs[key.lower()] = value

        parent_id = mrna_attrs.get("parent", "").lower()

        if feature_type in ("mrna", "transcript"):
            if gene_id and parent_id == gene_id.lower():
                mrna_id = mrna_attrs.get("id", "")
                if mrna_id:
                    mrna_ids.add(mrna_id.lower())

    # Third pass: find exons
    exon_coords: list[tuple[str, int, int]] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("##"):
            continue

        parts = line.split("\t")
        if len(parts) < 9:
            continue

        feature_type = parts[2].lower()
        if feature_type != "exon":
            continue

        attributes_str = parts[8]
        exon_attrs: dict[str, str] = {}
        for attr_pair in attributes_str.split(";"):
            if "=" in attr_pair:
                key, value = attr_pair.split("=", 1)
                exon_attrs[key.lower()] = value

        parent_id = exon_attrs.get("parent", "").lower()

        # Check if this exon belongs to one of our mRNAs
        if parent_id in mrna_ids:
            chrom = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
                exon_coords.append((chrom, start, end))
            except (ValueError, IndexError):
                continue

    # Return gene span if available, otherwise return exons
    # The gene span represents the full gene region
    if gene_chrom and gene_start is not None and gene_end is not None:
        return [(gene_chrom, gene_start, gene_end)]
    elif exon_coords:
        return exon_coords
    else:
        return []


def parse_mosdepth_regions_bed(
    regions_bed_path: Path,
    bed_path: Path,
) -> list[dict[str, object]]:
    """
    Parse mosdepth regions.bed.gz output with region names from original BED.

    mosdepth regions.bed.gz format when --by is used with named BED:
        chrom, start, end, name, mean_depth  (5 columns)
    mosdepth regions.bed.gz format with 3-column BED:
        chrom, start, end, mean_depth  (4 columns)

    The mean_depth is always the LAST column.

    Returns list of dicts with: chrom, start, end, region_name, mean_depth
    """
    import gzip

    # Build map of (chrom, start, end) -> region_name from original BED
    region_names: dict[tuple[str, int, int], str] = {}
    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3] if len(parts) >= 4 else f"{chrom}:{start}-{end}"
                region_names[(chrom, start, end)] = name

    # Parse mosdepth regions.bed.gz
    results: list[dict[str, object]] = []
    with gzip.open(regions_bed_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            # mean_depth is always the LAST column (column 3 for 4-col output, column 4 for 5-col output)
            mean_depth = float(parts[-1])

            # Get region name from original BED or generate default
            region_name = region_names.get((chrom, start, end), f"{chrom}:{start}-{end}")

            results.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "region_name": region_name,
                "mean_depth": mean_depth,
            })

    return results


def parse_mosdepth_thresholds_bed(
    thresholds_bed_path: Path,
    thresholds: list[int],
) -> dict[tuple[str, int, int], dict[str, float]]:
    """
    Parse mosdepth thresholds.bed.gz output.

    Format when --by used with named BED:
        chrom, start, end, name, pct_at_threshold1, pct_at_threshold2, ...
    Format when --by used with 3-column BED:
        chrom, start, end, pct_at_threshold1, pct_at_threshold2, ...

    The percentages are cumulative (% of bases >= threshold).
    The threshold values are always the LAST N columns where N = len(thresholds).

    Returns dict mapping (chrom, start, end) to threshold percentages.
    """
    import gzip

    results: dict[tuple[str, int, int], dict[str, float]] = {}

    with gzip.open(thresholds_bed_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Minimum columns: chrom, start, end, plus one threshold value
            if len(parts) < 4:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            # Threshold values are at the END of the line
            # They start at index (len(parts) - len(thresholds))
            threshold_start_idx = len(parts) - len(thresholds)
            if threshold_start_idx < 3:
                # Not enough columns for all thresholds
                continue

            threshold_pcts: dict[str, float] = {}
            for i, threshold in enumerate(thresholds):
                col_idx = threshold_start_idx + i
                if col_idx < len(parts):
                    # mosdepth outputs fraction (0-1), convert to percentage
                    pct = float(parts[col_idx]) * 100.0
                    threshold_pcts[f"pct_coverage_{threshold}x"] = pct

            results[(chrom, start, end)] = threshold_pcts

    return results


__all__ = [
    "find_gene_coordinates",
    "parse_alignment_header",
    "parse_bcftools_stats",
    "parse_bed_qc",
    "parse_cramino_json",
    "parse_error_profile",
    "parse_mosdepth_regions_bed",
    "parse_mosdepth_summary",
    "parse_mosdepth_thresholds_bed",
    "parse_nanoq_json",
    "parse_qscore_distribution",
    "parse_read_length_distribution",
    "parse_sequencing_summary",
    "parse_vcf_header",
    "summarize_header",
]
