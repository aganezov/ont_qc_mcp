from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .schemas import HistogramBin, LengthPercentiles


def _quantile_sorted(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return float(values[0])
    if q >= 1:
        return float(values[-1])
    n = len(values)
    if n == 1:
        return float(values[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def _build_histogram(counts: dict[int, int], bin_width: float, max_index: int, start: float = 0.0) -> list[HistogramBin]:
    if max_index < 0:
        return []
    bins: list[HistogramBin] = []
    for idx in range(max_index + 1):
        bins.append(
            HistogramBin(
                start=float(start + idx * bin_width),
                end=float(start + (idx + 1) * bin_width),
                count=int(counts.get(idx, 0)),
            )
        )
    return bins


def _histogram_and_values_from_file(
    path: Path,
    *,
    bin_width: float,
    cast: type[int] | type[float],
    start: float = 0.0,
    exact_max: int | None = None,
) -> tuple[list[HistogramBin], list[float] | None, int]:
    if bin_width <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    counts: dict[int, int] = defaultdict(int)
    max_index = -1
    total = 0
    values: list[float] | None = [] if exact_max and exact_max > 0 else None

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                val = cast(raw)  # type: ignore[call-arg]
            except ValueError:
                continue

            total += 1
            fval = float(val)
            idx = int((fval - start) // bin_width) if fval >= start else 0
            counts[idx] += 1
            if idx > max_index:
                max_index = idx

            if values is not None:
                values.append(fval)
                if exact_max and len(values) > exact_max:
                    values = None

    return _build_histogram(counts, bin_width=bin_width, max_index=max_index, start=start), values, total


def length_histogram_and_percentiles(
    lengths_path: Path,
    *,
    bin_width: int = 2000,
    percentiles_exact_max_reads: int = 200_000,
) -> tuple[list[HistogramBin], LengthPercentiles | None]:
    histogram, values, total = _histogram_and_values_from_file(
        lengths_path,
        bin_width=float(bin_width),
        cast=int,
        start=0.0,
        exact_max=percentiles_exact_max_reads,
    )
    if values is None or not values:
        return histogram, None
    if total > percentiles_exact_max_reads:
        return histogram, None

    values.sort()
    return (
        histogram,
        LengthPercentiles(
            p1=_quantile_sorted(values, 0.01),
            p5=_quantile_sorted(values, 0.05),
            p25=_quantile_sorted(values, 0.25),
            p50=_quantile_sorted(values, 0.50),
            p75=_quantile_sorted(values, 0.75),
            p95=_quantile_sorted(values, 0.95),
            p99=_quantile_sorted(values, 0.99),
        ),
    )


def qscore_histogram(
    qualities_path: Path,
    *,
    bin_width: float = 1.0,
) -> list[HistogramBin]:
    histogram, _values, _total = _histogram_and_values_from_file(
        qualities_path,
        bin_width=float(bin_width),
        cast=float,
        start=0.0,
        exact_max=None,
    )
    return histogram


__all__ = ["length_histogram_and_percentiles", "qscore_histogram"]

