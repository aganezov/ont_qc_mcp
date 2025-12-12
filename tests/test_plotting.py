import pytest

from ont_qc_mcp.plotting import plot_length_histogram, plot_qscore_histogram
from ont_qc_mcp.schemas import HistogramBin


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
