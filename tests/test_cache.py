from ont_qc_mcp import tools as m_tools
from ont_qc_mcp.schemas import LengthPercentiles, NanoqStats


def test_nanoq_cache_eviction(monkeypatch, tmp_path):
    # Reduce cache size for faster eviction and reset state.
    original_max = m_tools._NANOQ_CACHE_MAX
    monkeypatch.setattr(m_tools, "_NANOQ_CACHE_MAX", 2)
    m_tools._NANOQ_CACHE.clear()
    m_tools._NANOQ_INFLIGHT.clear()
    m_tools._NANOQ_CACHE_STATS.update({"hits": 0, "misses": 0, "evictions": 0})

    def fake_nanoq(path, tool_paths, flags=None, exec_cfg=None):
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

    # Create three unique FASTQ files to exceed cache capacity (set to 2).
    paths = []
    for i in range(3):
        fq = tmp_path / f"reads_{i}.fastq"
        fq.write_text("@r\nACGT\n+\n!!!!\n")
        paths.append(fq)

    # First two fill the cache; third should trigger one eviction.
    for fq in paths:
        m_tools.qc_reads(str(fq))

    stats = m_tools.get_nanoq_cache_stats()
    assert stats["size"] == 2
    assert stats["evictions"] == 1

    # Accessing first file again should be a miss (evicted), increasing misses count.
    prev_misses = stats["misses"]
    m_tools.qc_reads(str(paths[0]))
    stats_after = m_tools.get_nanoq_cache_stats()
    assert stats_after["misses"] == prev_misses + 1

    # Restore original max
    monkeypatch.setattr(m_tools, "_NANOQ_CACHE_MAX", original_max)
