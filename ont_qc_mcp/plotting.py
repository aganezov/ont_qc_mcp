from pathlib import Path

from .schemas import HistogramBin


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
    bins: list[HistogramBin],
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

    centers = [(b.start + b.end) / 2.0 for b in bins]
    widths = [b.end - b.start for b in bins]
    counts = [b.count for b in bins]

    plt.figure(figsize=(6, 4), dpi=150)
    plt.bar(centers, counts, width=widths, align="center", edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_length_histogram(bins: list[HistogramBin], output_path: str | None = None) -> str:
    """Save a length histogram PNG and return its path."""
    return _plot_histogram(
        bins,
        xlabel="Read length (bp)",
        ylabel="Count",
        title="read_length_histogram",
        output_path=output_path,
    )


def plot_qscore_histogram(bins: list[HistogramBin], output_path: str | None = None) -> str:
    """Save a q-score histogram PNG and return its path."""
    return _plot_histogram(bins, xlabel="Q-score", ylabel="Count", title="qscore_histogram", output_path=output_path)


__all__ = ["plot_length_histogram", "plot_qscore_histogram"]
