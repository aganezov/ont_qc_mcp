from concurrent.futures import ThreadPoolExecutor

from ont_qc_mcp import tools as m_tools
from ont_qc_mcp.schemas import LengthPercentiles, NanoqStats


def test_qscore_distribution_concurrent(monkeypatch, tmp_path):
    m_tools._NANOQ_CACHE.clear()
    fastq_path = tmp_path / "reads.fastq"
    fastq_path.write_text("@r1\nACGT\n+\n!!!!\n")

    call_count = {"n": 0}

    def fake_nanoq(path, tool_paths, flags=None, exec_cfg=None):
        call_count["n"] += 1
        return NanoqStats(
            file=str(path),
            read_count=1,
            total_bases=4,
            min_len=4,
            max_len=4,
            mean_len=4.0,
            median_len=4.0,
            n50=None,
            mean_qscore=12.0,
            median_qscore=12.0,
            gc_content=None,
            length_percentiles=LengthPercentiles(p50=4),
            length_histogram=[],
            qscore_histogram=[],
        )

    monkeypatch.setattr(m_tools, "nanoq_stats", fake_nanoq)

    def run_cached():
        return m_tools.qscore_distribution(str(fastq_path))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: run_cached(), range(6)))

    assert call_count["n"] == 1
    m_tools._NANOQ_CACHE.clear()
