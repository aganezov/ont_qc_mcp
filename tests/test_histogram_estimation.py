import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp.cli_wrappers import cramino_stats
from ont_qc_mcp.config import ToolPaths
from ont_qc_mcp.parsers import estimate_scaled_histogram
from ont_qc_mcp.schemas import HistogramBin


def _parse_ascii_read_length_histogram(text: str) -> dict[tuple[int, int], int]:
    bins: dict[tuple[int, int], int] = {}
    in_lengths = False
    for line in text.splitlines():
        if "Histogram for read lengths" in line:
            in_lengths = True
            continue
        if "Histogram for Phred" in line:
            break
        if not in_lengths or "-" not in line:
            continue
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(\S+)?$", line.rstrip())
        if not match:
            continue
        start = int(float(match.group(1)))
        end = int(float(match.group(2)))
        bars = match.group(3) or ""
        bins[(start, end)] = len(bars)
    return bins


def _normalize(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    if not values:
        return {}
    max_val = max(values.values())
    if max_val <= 0:
        return {key: 0.0 for key in values}
    return {key: val / max_val for key, val in values.items()}


def _parse_json_read_length_bases(payload: str) -> dict[tuple[int, int], int]:
    data = json.loads(payload)
    hist = data.get("histograms", {}).get("read_length", {}).get("bins", [])
    bases: dict[tuple[int, int], int] = {}
    for entry in hist:
        start = entry.get("start")
        end = entry.get("end")
        if start is None or end is None:
            continue
        bases[(int(start), int(end))] = int(entry.get("bases", 0))
    return bases


def test_estimate_scaled_histogram_midpoint():
    bins = [
        HistogramBin(start=0.0, end=100.0, count=2),
        HistogramBin(start=100.0, end=200.0, count=1),
    ]
    scaled = estimate_scaled_histogram(bins)
    assert [b.count for b in scaled] == [100, 150]


@pytest.mark.integration
def test_scaled_histogram_disabled_has_no_estimate(sample_bam: Path):
    require_executable_tools(["cramino"])

    tools = ToolPaths()
    stats = cramino_stats(sample_bam, tools, include_hist=True, use_scaled=False)
    assert stats.length_histogram_scaled is None
    assert stats.length_histogram_scaled_is_estimated is None


@pytest.mark.integration
def test_scaled_histogram_estimate_matches_ascii(sample_bam: Path):
    require_executable_tools(["cramino"])

    tools = ToolPaths()
    stats = cramino_stats(sample_bam, tools, include_hist=True, use_scaled=True)
    assert stats.length_histogram_scaled, "Expected estimated scaled histogram bins"
    assert stats.length_histogram_scaled_is_estimated in {True, False}

    if stats.length_histogram_scaled_is_estimated:
        ascii_result = subprocess.run(
            [tools.cramino, "--format", "text", "--hist", "--scaled", str(sample_bam)],
            check=True,
            capture_output=True,
            text=True,
        )
        bars = _parse_ascii_read_length_histogram(ascii_result.stdout)
        estimated = {(int(bin.start), int(bin.end)): bin.count for bin in stats.length_histogram_scaled or []}

        bars_norm = _normalize({key: float(val) for key, val in bars.items()})
        est_norm = _normalize({key: float(val) for key, val in estimated.items()})

        assert bars_norm, "Expected ASCII histogram bins from cramino"
        assert est_norm, "Expected estimated scaled histogram bins"

        for key, bar_val in bars_norm.items():
            est_val = est_norm.get(key, 0.0)
            if bar_val == 0.0 and est_val == 0.0:
                continue
            diff = abs(bar_val - est_val)
            assert diff <= 0.2, f"Estimated bin {key} diverges too much from ASCII bars (diff={diff:.2f})"
    else:
        with tempfile.NamedTemporaryFile(suffix=".cramino.hist.txt", delete=False) as tmp:
            hist_path = Path(tmp.name)
        json_result = subprocess.run(
            [tools.cramino, "--format", "json", f"--hist={hist_path}", str(sample_bam)],
            check=True,
            capture_output=True,
            text=True,
        )
        bases = _parse_json_read_length_bases(json_result.stdout)
        actual = {(int(bin.start), int(bin.end)): bin.count for bin in stats.length_histogram_scaled or []}
        assert bases, "Expected JSON histogram bins with bases"
        assert actual, "Expected scaled bins from cramino JSON"
        for key, value in bases.items():
            assert actual.get(key) == value
