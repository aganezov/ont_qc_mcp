from pathlib import Path
from typing import List, Optional

from .schemas import HistogramBin


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting; install with [plots] extra") from exc


def _plot_histogram(bins: List[HistogramBin], xlabel: str, ylabel: str, title: str, output_path: Optional[str]) -> str:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt  # type: ignore

    if output_path is None:
        output_path = str(Path.cwd() / f"{title.replace(' ', '_').lower()}.png")

    lefts = [b.start for b in bins]
    widths = [b.end - b.start for b in bins]
    counts = [b.count for b in bins]

    plt.figure(figsize=(8, 4))
    plt.bar(lefts, counts, width=widths, align="edge", color="#3b82f6", edgecolor="#1e3a8a")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_length_histogram(bins: List[HistogramBin], output_path: Optional[str] = None) -> str:
    """Save a length histogram PNG and return its path."""
    return _plot_histogram(bins, xlabel="Read length (bp)", ylabel="Count", title="read_length_histogram", output_path=output_path)


def plot_qscore_histogram(bins: List[HistogramBin], output_path: Optional[str] = None) -> str:
    """Save a q-score histogram PNG and return its path."""
    return _plot_histogram(bins, xlabel="Q-score", ylabel="Count", title="qscore_histogram", output_path=output_path)
