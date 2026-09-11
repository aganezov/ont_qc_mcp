from collections.abc import Sequence
from pathlib import Path

from .schemas import CraminoHistogramBin, HistogramBin


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'ont-qc-mcp[plots]' or pip install \"matplotlib>=3.8\"."
        ) from exc


def _plot_histogram(
    bins: Sequence[HistogramBin | CraminoHistogramBin],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str | None,
) -> str:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = f"{title.replace(' ', '_').lower()}.png"
    else:
        output_path = str(output_path)

    finite_widths = [b.end - b.start for b in bins if b.end is not None]
    # Overflow bars need a display width, not an inferred upper bound.
    open_width = finite_widths[-1] if finite_widths else 1.0
    widths = [b.end - b.start if b.end is not None else open_width for b in bins]
    centers = [b.start + width / 2.0 for b, width in zip(bins, widths)]
    counts = [b.count for b in bins]

    plt.figure(figsize=(6, 4), dpi=150)
    bars = plt.bar(centers, counts, width=widths, align="center", edgecolor="black")
    for b, bar, center in zip(bins, bars, centers):
        if b.end is None:
            bar.set_hatch("//")
            plt.annotate(
                f"≥ {b.start:g}",
                (center, b.count),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    if len(finite_widths) < len(bins):
        plt.margins(y=0.15)
        if not finite_widths:
            plt.xticks(centers, [f"≥ {b.start:g}" for b in bins])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_length_histogram(bins: Sequence[HistogramBin | CraminoHistogramBin], output_path: str | None = None) -> str:
    """Save a length histogram PNG and return its path."""
    return _plot_histogram(
        bins,
        xlabel="Read length (bp)",
        ylabel="Count",
        title="read_length_histogram",
        output_path=output_path,
    )


def plot_qscore_histogram(bins: Sequence[HistogramBin | CraminoHistogramBin], output_path: str | None = None) -> str:
    """Save a q-score histogram PNG and return its path."""
    return _plot_histogram(bins, xlabel="Q-score", ylabel="Count", title="qscore_histogram", output_path=output_path)


__all__ = ["plot_length_histogram", "plot_qscore_histogram"]
