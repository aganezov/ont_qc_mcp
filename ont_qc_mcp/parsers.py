import json
import re
from typing import Dict, List, Optional, Sequence

from .schemas import (
    CraminoStats,
    ErrorProfile,
    HeaderMetadata,
    HistogramBin,
    LengthPercentiles,
    LowCoverageRegion,
    MosdepthStats,
    NanoqStats,
    QScoreDistribution,
    ProgramRecord,
    ReadLengthDistribution,
    ReferenceRecord,
    SampleRecord,
    VCFFieldDef,
    CoverageByContig,
)


def _histogram_from_seq(bins: Sequence[Sequence[float]]) -> List[HistogramBin]:
    histogram: List[HistogramBin] = []
    for entry in bins:
        if len(entry) < 3:
            continue
        start, end, count = entry[0], entry[1], entry[2]
        histogram.append(HistogramBin(start=float(start), end=float(end), count=int(count)))
    return histogram


def _safe_percentiles(data: Dict[str, float]) -> LengthPercentiles:
    return LengthPercentiles(
        p1=data.get("p1"),
        p5=data.get("p5"),
        p25=data.get("p25"),
        p50=data.get("p50") or data.get("median"),
        p75=data.get("p75"),
        p95=data.get("p95"),
        p99=data.get("p99"),
    )


def parse_nanoq_json(payload: str | Dict) -> NanoqStats:
    """
    Parse nanoq --stats --json output.
    The JSON schema can vary by version; we defensively access keys.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    summary = data.get("summary", data)
    read_stats = summary.get("reads", summary)
    if not isinstance(read_stats, dict):
        read_stats = summary

    file_name = summary.get("file") or summary.get("input") or "unknown"
    length_info = read_stats.get("length", {}) if isinstance(read_stats, dict) else {}
    if not isinstance(length_info, dict):
        length_info = {}
    qscore_info = read_stats.get("qscore", {}) if isinstance(read_stats, dict) else {}
    if not isinstance(qscore_info, dict):
        qscore_info = {}

    length_bins_raw = length_info.get("hist") or length_info.get("histogram") or []
    qscore_bins_raw = qscore_info.get("hist") or qscore_info.get("histogram") or []

    percentiles = _safe_percentiles(length_info.get("percentiles", {})) if length_info else None

    return NanoqStats(
        file=file_name,
        read_count=int(read_stats.get("count", read_stats.get("reads", 0) or 0)),
        total_bases=int(read_stats.get("bases", read_stats.get("total_bases", 0) or 0)),
        min_len=int(length_info.get("min", 0) or 0),
        max_len=int(length_info.get("max", 0) or 0),
        mean_len=float(length_info.get("mean", 0.0) or 0.0),
        median_len=float(length_info.get("median", length_info.get("p50", 0.0) or 0.0)),
        n50=length_info.get("n50"),
        mean_qscore=float(qscore_info.get("mean", qscore_info.get("average", 0.0) or 0.0))
        if qscore_info
        else None,
        median_qscore=float(qscore_info.get("median", qscore_info.get("p50", 0.0) or 0.0))
        if qscore_info
        else None,
        gc_content=read_stats.get("gc"),
        length_percentiles=percentiles,
        length_histogram=_histogram_from_seq(length_bins_raw),
        qscore_histogram=_histogram_from_seq(qscore_bins_raw),
    )


def parse_read_length_distribution(payload: str | Dict) -> ReadLengthDistribution:
    data = json.loads(payload) if isinstance(payload, str) else payload
    stats = parse_nanoq_json(data)
    return ReadLengthDistribution(
        file=stats.file,
        percentiles=stats.length_percentiles or LengthPercentiles(),
        histogram=stats.length_histogram or [],
    )


def parse_qscore_distribution(payload: str | Dict) -> QScoreDistribution:
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
    payload: str | Dict,
    length_bins: Optional[List[HistogramBin]] = None,
    length_bins_scaled: Optional[List[HistogramBin]] = None,
) -> CraminoStats:
    """
    Parse cramino JSON output (e.g., --format json), supporting both count and scaled histograms.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    summary = data.get("summary", data)
    read_counts = summary.get("reads", summary)

    mapq_bins_counts = summary.get("mapq_hist", summary.get("mapq_histogram", []))
    mapq_bins_scaled = summary.get("mapq_hist_scaled") or summary.get("mapq_hist_scaled_bp") or []

    return CraminoStats(
        file=summary.get("file", "unknown"),
        total_reads=int(read_counts.get("total", read_counts.get("reads", 0) or 0)),
        mapped=int(read_counts.get("mapped", 0) or 0),
        unmapped=int(read_counts.get("unmapped", 0) or 0),
        primary=read_counts.get("primary"),
        secondary=read_counts.get("secondary"),
        supplementary=read_counts.get("supplementary"),
        mean_length=summary.get("mean_length"),
        median_length=summary.get("median_length"),
        n50=summary.get("n50"),
        mean_identity=summary.get("mean_identity"),
        median_identity=summary.get("median_identity"),
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
    coverage_by_contig = []
    coverage_distribution = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("chrom"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        contig, length, bases, mean = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
        coverage_by_contig.append(
            CoverageByContig(contig=contig, length=length, mean_depth=mean, median_depth=None)
        )

    total_len = sum(c.length or 0 for c in coverage_by_contig)
    unweighted_mean = (
        sum(c.mean_depth for c in coverage_by_contig) / len(coverage_by_contig) if coverage_by_contig else 0.0
    )
    weighted_mean = (
        sum((c.mean_depth) * (c.length or 0) for c in coverage_by_contig) / total_len if total_len else unweighted_mean
    )

    low_cov: List[LowCoverageRegion] = []
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
    metrics: Dict[str, str] = {}
    coverage_hist: List[HistogramBin] = []
    gc_cov: List[HistogramBin] = []
    mismatch_by_cycle: Dict[int, float] = {}
    insert_hist: List[HistogramBin] = []
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
            if len(parts) >= 3:
                try:
                    gc_pct = float(parts[1])
                    depth = float(parts[2])
                    count = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
                except ValueError:
                    continue
                gc_cov.append(HistogramBin(start=gc_pct, end=gc_pct, count=count or int(depth)))
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


def _parse_tag_fields(parts: Sequence[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
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
    references: List[ReferenceRecord] = []
    programs: List[ProgramRecord] = []
    samples: List[SampleRecord] = []
    comments: List[str] = []
    metadata: Dict[str, str] = {}
    header_other: Dict[str, str] = {}

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


def _parse_angle_bracket_fields(value: str) -> Dict[str, str]:
    """
    Parse VCF-style key/value lists inside angle brackets, respecting quoted commas.
    """
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]

    fields: Dict[str, str] = {}
    current = []
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
    references: List[ReferenceRecord] = []
    info_fields: List[VCFFieldDef] = []
    format_fields: List[VCFFieldDef] = []
    filter_fields: List[VCFFieldDef] = []
    samples: List[SampleRecord] = []
    metadata: Dict[str, str] = {}
    comments: List[str] = []

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
                    category=key_upper,
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
    parts: List[str] = []
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
        parts.append(
            f"extras h={extra_header},sq={extra_ref},pg={extra_prog},rg={extra_rg},vcf={extra_vcf}"
        )
    summary = "; ".join(parts)
    metadata.summary = summary
    return summary

