from pathlib import Path

import pytest

from ont_qc_mcp.nanoq_aux import length_histogram_and_percentiles, qscore_histogram


def test_length_histogram_and_percentiles(tmp_path: Path) -> None:
    lengths = [1, 2, 3, 4, 5]
    path = tmp_path / "lengths.txt"
    path.write_text("\n".join(str(v) for v in lengths) + "\n", encoding="utf-8")

    hist, percentiles = length_histogram_and_percentiles(path, bin_width=2, percentiles_exact_max_reads=100)

    assert percentiles is not None
    assert percentiles.p50 == 3.0
    assert [b.count for b in hist] == [1, 2, 2]


def test_qscore_histogram(tmp_path: Path) -> None:
    quals = [0.0, 0.1, 1.9, 2.0]
    path = tmp_path / "quals.txt"
    path.write_text("\n".join(str(v) for v in quals) + "\n", encoding="utf-8")

    hist = qscore_histogram(path, bin_width=1.0)
    assert [b.count for b in hist] == [2, 1, 1]


def test_invalid_bin_width_raises(tmp_path: Path) -> None:
    path = tmp_path / "lengths.txt"
    path.write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        length_histogram_and_percentiles(path, bin_width=0)

