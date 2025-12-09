import csv
import io
import re
from typing import Dict

from .schemas import SamtoolsStats, SeqkitStats


def parse_seqkit_stats(text: str) -> SeqkitStats:
    """
    Parse the first data line of `seqkit stats` (tsv).
    Expected header contains at least: file, num_seqs, sum_len, min_len, avg_len, max_len, N50, GC.
    """
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("seqkit stats output missing data rows")

    header = [h.strip() for h in rows[0]]
    data = rows[1]
    mapping = dict(zip(header, data))

    def to_int(key: str, default: int = 0) -> int:
        value = mapping.get(key)
        return int(float(value)) if value not in (None, "") else default

    def to_float(key: str) -> float:
        value = mapping.get(key)
        if value in (None, ""):
            return 0.0
        return float(value)

    gc_raw = mapping.get("GC", None)
    gc = float(gc_raw) if gc_raw not in (None, "") else None
    if gc and gc > 1.0:
        gc = gc / 100.0

    n50_raw = mapping.get("N50", None)
    n50 = to_int("N50") if n50_raw not in (None, "") else None

    return SeqkitStats(
        file=mapping.get("file", "unknown"),
        num_seqs=to_int("num_seqs"),
        total_bases=to_int("sum_len"),
        min_len=to_int("min_len"),
        avg_len=to_float("avg_len"),
        max_len=to_int("max_len"),
        n50=n50,
        gc=gc,
    )


def parse_samtools_stats(text: str) -> SamtoolsStats:
    """
    Parse `samtools stats` output.
    Only `SN` lines are parsed; other keys are stored in `other_metrics`.
    """
    metrics: Dict[str, str] = {}
    pattern = re.compile(r"^SN\t(.+?):\t(.+)")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        metrics[key] = value

    def get_int(label: str):
        raw = metrics.get(label)
        return int(raw) if raw and raw.isdigit() else None

    def get_float(label: str):
        raw = metrics.get(label)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    known = {
        "raw total sequences": "raw_total_sequences",
        "filtered sequences": "filtered_sequences",
        "reads mapped": "reads_mapped",
        "reads properly paired": "reads_properly_paired",
        "bases mapped (cigar)": "bases_mapped",
        "error rate": "error_rate",
        "average length": "average_length",
        "insert size average": "insert_size_average",
    }

    parsed_known: Dict[str, str] = {}
    for k, v in list(metrics.items()):
        if k in known:
            parsed_known[k] = v
            metrics.pop(k)

    return SamtoolsStats(
        raw_total_sequences=get_int("raw total sequences"),
        filtered_sequences=get_int("filtered sequences"),
        reads_mapped=get_int("reads mapped"),
        reads_properly_paired=get_int("reads properly paired"),
        bases_mapped=get_int("bases mapped (cigar)"),
        error_rate=get_float("error rate"),
        average_length=get_float("average length"),
        insert_size_average=get_float("insert size average"),
        other_metrics={known.get(k, k): v for k, v in metrics.items()},
    )

