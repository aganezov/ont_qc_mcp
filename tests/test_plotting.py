import pytest

from ont_qc_mcp.plotting import plot_length_histogram, plot_qscore_histogram
from ont_qc_mcp.schemas import CraminoHistogramBin, HistogramBin


def test_plot_length_histogram(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    bins = [HistogramBin(start=0, end=10, count=5), HistogramBin(start=10, end=20, count=3)]
    output = tmp_path / "len.png"
    path = plot_length_histogram(bins, output_path=output)
    assert output.exists()
    assert str(output) == path


def test_plot_qscore_histogram(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    bins = [HistogramBin(start=0, end=5, count=2), HistogramBin(start=5, end=10, count=4)]
    output = tmp_path / "qscore.png"
    path = plot_qscore_histogram(bins, output_path=output)
    assert output.exists()
    assert str(output) == path


@pytest.mark.parametrize("plotter,start", [(plot_length_histogram, 2000), (plot_qscore_histogram, 40)])
@pytest.mark.parametrize("only_open_bin", [False, True])
def test_plot_cramino_open_bin_preserves_counts_and_labels_bound(tmp_path, monkeypatch, plotter, start, only_open_bin):
    plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
    bins = [] if only_open_bin else [CraminoHistogramBin(start=0, end=start, count=1, bases=100)]
    bins.append(CraminoHistogramBin(start=start, end=None, count=3, bases=6000))
    figures = []
    savefig = plt.savefig

    def capture_figure(*args, **kwargs):
        figures.append(plt.gcf())
        return savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", capture_figure)
    output = tmp_path / "histogram.png"
    plotter(bins, output_path=output)
    axes = figures[0].axes[0]
    assert output.is_file()
    assert [bar.get_height() for bar in axes.patches] == [b.count for b in bins]
    assert any(f"≥ {start}" in text.get_text() for text in axes.texts)
    if only_open_bin:
        assert [tick.get_text() for tick in axes.get_xticklabels()] == [f"≥ {start}"]
