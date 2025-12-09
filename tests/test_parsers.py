from mcp.parsers import parse_samtools_stats, parse_seqkit_stats


def test_parse_seqkit_stats_basic():
    sample = "file\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\tN50\tGC\nreads.fastq\t10\t1000\t50\t100.0\t200\t150\t45.0\n"
    parsed = parse_seqkit_stats(sample)
    assert parsed.file == "reads.fastq"
    assert parsed.num_seqs == 10
    assert parsed.total_bases == 1000
    assert parsed.n50 == 150
    assert parsed.gc == 0.45


def test_parse_samtools_stats_sn_lines():
    sample = "\n".join(
        [
            "SN\traw total sequences:\t123",
            "SN\tfiltered sequences:\t3",
            "SN\treads mapped:\t100",
            "SN\taverage length:\t500.5",
            "SN\tinsert size average:\t300.1",
        ]
    )
    parsed = parse_samtools_stats(sample)
    assert parsed.raw_total_sequences == 123
    assert parsed.filtered_sequences == 3
    assert parsed.reads_mapped == 100
    assert parsed.average_length == 500.5
    assert parsed.insert_size_average == 300.1

