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


def _build_histogram(
    counts: dict[int, int],
    bin_width: float,
    max_index: int,
    start: float = 0.0,
) -> list[HistogramBin]:
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


class _HistogramAccumulator:
    def __init__(
        self, bin_width: float, cast: type[int] | type[float], exact_max: int | None = None, start: float = 0.0
    ):
        if bin_width <= 0:
            raise ValueError(f"bin_width must be > 0, got {bin_width}")
        self.bin_width = bin_width
        self.cast = cast
        self.start = start
        self.exact_max = exact_max
        self.counts: dict[int, int] = defaultdict(int)
        self.max_index = -1
        self.total = 0
        self.values: list[float] | None = [] if exact_max and exact_max > 0 else None

    def add_line(self, line: str) -> None:
        raw = line.strip()
        if not raw:
            return
        try:
            val = self.cast(raw)
        except ValueError:
            return
        self.total += 1
        fval = float(val)
        idx = int((fval - self.start) // self.bin_width) if fval >= self.start else 0
        self.counts[idx] += 1
        self.max_index = max(self.max_index, idx)
        if self.values is not None:
            self.values.append(fval)
            if self.exact_max and len(self.values) > self.exact_max:
                self.values = None

    def histogram(self) -> list[HistogramBin]:
        return _build_histogram(self.counts, self.bin_width, self.max_index, self.start)


def _histogram_and_values_from_file(
    path: Path,
    *,
    bin_width: float,
    cast: type[int] | type[float],
    start: float = 0.0,
    exact_max: int | None = None,
) -> tuple[list[HistogramBin], list[float] | None, int]:
    accumulator = _HistogramAccumulator(bin_width, cast, exact_max, start)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            accumulator.add_line(line)
    return accumulator.histogram(), accumulator.values, accumulator.total


def _length_percentiles(values: list[float] | None) -> LengthPercentiles | None:
    if not values:
        return None
    values.sort()
    return LengthPercentiles(
        p1=_quantile_sorted(values, 0.01),
        p5=_quantile_sorted(values, 0.05),
        p25=_quantile_sorted(values, 0.25),
        p50=_quantile_sorted(values, 0.50),
        p75=_quantile_sorted(values, 0.75),
        p95=_quantile_sorted(values, 0.95),
        p99=_quantile_sorted(values, 0.99),
    )


def length_histogram_and_percentiles(
    lengths_path: Path,
    *,
    bin_width: int = 2000,
    percentiles_exact_max_reads: int = 200_000,
) -> tuple[list[HistogramBin], LengthPercentiles | None]:
    histogram, values, _total = _histogram_and_values_from_file(
        lengths_path,
        bin_width=float(bin_width),
        cast=int,
        start=0.0,
        exact_max=percentiles_exact_max_reads,
    )
    return histogram, _length_percentiles(values)


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
