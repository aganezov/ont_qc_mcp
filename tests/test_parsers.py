from ont_qc_mcp.parsers import (
    parse_alignment_header,
    parse_cramino_json,
    parse_error_profile,
    parse_mosdepth_summary,
    parse_nanoq_json,
    parse_vcf_header,
    summarize_header,
)
from ont_qc_mcp.tools import header_metadata_lookup


def test_parse_nanoq_json():
    payload = {
        "summary": {
            "file": "reads.fastq",
            "reads": {
                "count": 5,
                "bases": 5000,
                "gc": 0.42,
                "length": {
                    "min": 100,
                    "max": 2000,
                    "mean": 1000.0,
                    "median": 900.0,
                    "n50": 1200,
                    "percentiles": {"p50": 900, "p95": 1800},
                    "hist": [[0, 500, 1], [500, 1000, 2], [1000, 1500, 2]],
                },
                "qscore": {"mean": 13.2, "median": 13.0, "hist": [[10, 12, 2], [12, 14, 3]]},
            },
        }
    }
    parsed = parse_nanoq_json(payload)
    assert parsed.file == "reads.fastq"
    assert parsed.read_count == 5
    assert parsed.total_bases == 5000
    assert parsed.length_percentiles.p50 == 900
    assert parsed.qscore_histogram[0].count == 2


def test_parse_cramino_json():
    payload = {
        "summary": {
            "file": "align.bam",
            "reads": {"total": 10, "mapped": 9, "unmapped": 1, "primary": 8, "secondary": 1},
            "mean_length": 1200.5,
            "median_length": 1100,
            "n50": 1500,
            "mean_identity": 0.96,
            "median_identity": 0.95,
            "mapq_hist": [[0, 10, 2], [10, 20, 5], [20, 30, 3]],
        }
    }
    parsed = parse_cramino_json(payload)
    assert parsed.file == "align.bam"
    assert parsed.total_reads == 10
    assert parsed.mapped == 9
    assert parsed.mapq_histogram[-1].count == 3


def test_parse_cramino_json_scaled():
    payload = {
        "summary": {
            "file": "align.bam",
            "reads": {"total": 10, "mapped": 9, "unmapped": 1},
            "mapq_hist": [[0, 10, 2], [10, 20, 5]],
            "mapq_hist_scaled": [[0, 10, 500], [10, 20, 1500]],
        }
    }
    parsed = parse_cramino_json(payload)
    assert parsed.mapq_histogram_scaled is not None
    assert parsed.mapq_histogram_scaled[1].count == 1500


def test_parse_mosdepth_summary():
    summary = "chrom\tlength\tbases\tmean\nchr1\t1000\t10000\t10\nchr2\t500\t2500\t5\n"
    parsed = parse_mosdepth_summary(summary, file_path="align.bam")
    assert parsed.file == "align.bam"
    assert len(parsed.coverage_by_contig) == 2
    assert abs(parsed.mean_depth - 8.3333333333) < 1e-6
    assert abs(parsed.mean_depth_unweighted - 7.5) < 1e-6


def test_parse_mosdepth_summary_low_cov_threshold():
    summary = "chrom\tlength\tbases\tmean\nchr1\t1000\t10000\t3\nchr2\t500\t2500\t12\n"
    parsed = parse_mosdepth_summary(summary, file_path="align.bam", threshold=5)
    assert parsed.low_coverage_regions
    assert parsed.low_coverage_regions[0].contig == "chr1"
    assert parsed.low_coverage_regions[0].mean_depth == 3


def test_parse_error_profile():
    sample = "\n".join(
        [
            "SN\tmismatches per base:\t0.01",
            "SN\tinsertions per base:\t0.002",
            "SN\tdeletions per base:\t0.003",
            "COV\t10\t1000",
            "MPC\t1\t0.01",
            "MPC\t2\t0.02",
            "IS\t100\t5",
        ]
    )
    parsed = parse_error_profile(sample, file_path="align.bam")
    assert parsed.mismatch_rate == 0.01
    assert parsed.insertion_rate == 0.002
    assert parsed.deletion_rate == 0.003
    assert parsed.coverage_histogram and parsed.coverage_histogram[0].count == 1000
    assert parsed.mismatch_by_cycle == [0.01, 0.02]
    assert parsed.insert_size_histogram and parsed.insert_size_histogram[0].start == 100


def test_parse_alignment_header_and_summary():
    header_text = "\n".join(
        [
            "@HD\tVN:1.6\tSO:coordinate",
            "@SQ\tSN:chr1\tLN:1000\tAS:GRCh38\tSP:human",
            "@SQ\tSN:chr2\tLN:2000\tUR:http://example",
            "@RG\tID:rg1\tSM:SAMPLE1\tLB:lib1\tPL:ONT\tPU:PU123\tCN:ONT\tDT:2024-01-01\tPM:GridION\tFC:FLOW123\tXX:extra",
            "@PG\tID:pg1\tPN:samtools\tVN:1.17\tCL:samtools view -H test.bam\tZZ:pg_extra",
        ]
    )
    meta = parse_alignment_header(header_text, file_path="align.bam", fmt="bam")
    assert meta.references[0].name == "chr1"
    assert meta.references[0].length == 1000
    assert meta.references[0].other["SP"] == "human"
    assert meta.samples[0].name == "SAMPLE1"
    assert meta.samples[0].platform_unit == "PU123"
    assert meta.samples[0].sequencing_center == "ONT"
    assert meta.samples[0].run_date == "2024-01-01"
    assert meta.samples[0].flowcell_id == "FLOW123"
    assert meta.samples[0].other["XX"] == "extra"
    assert meta.programs[0].name == "samtools"
    assert meta.programs[0].other["ZZ"] == "pg_extra"
    assert meta.header_other == {}
    summary = summarize_header(meta)
    assert "contigs" in summary
    assert "SAMPLE1" in summary


def test_parse_vcf_header_and_summary():
    header_text = "\n".join(
        [
            "##fileformat=VCFv4.3",
            "##source=test-generator",
            '##contig=<ID=chr1,length=5000,IDX=1>',
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth",Source="caller">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype",Version="1.0">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        ]
    )
    meta = parse_vcf_header(header_text, file_path="variants.vcf")
    assert meta.format == "vcf"
    assert meta.references[0].name == "chr1"
    assert meta.references[0].other["IDX"] == "1"
    assert meta.info_fields[0].id == "DP"
    assert meta.info_fields[0].other["Source"] == "caller"
    assert len(meta.samples) == 2
    summary = summarize_header(meta)
    assert "VCF" in summary
    assert "samples" in summary


def test_header_metadata_lookup_vcf_max_lines(tmp_path):
    vcf_header = "\n".join(
        [
            "##fileformat=VCFv4.3",
            "##source=test-suite",
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        ]
    )
    vcf_path = tmp_path / "trunc.vcf"
    vcf_path.write_text(vcf_header)

    meta = header_metadata_lookup(path=str(vcf_path), file_type="vcf", max_lines=2)
    # Only the first two lines should be retained due to max_lines guard.
    assert meta.raw_header.count("\n") == 1
    assert meta.info_fields == []
    assert meta.samples == []


def test_alignment_header_other_fields():
    header_text = "\n".join(
        [
            "@HD\tVN:1.6\tSO:coordinate\tXX:hd_extra",
            "@SQ\tSN:chr1\tLN:1000\tCX:contig_extra",
            "@RG\tID:rg1\tSM:SAMPLE1\tPL:ONT\tPU:UNIT1\tZZ:rg_extra",
            "@PG\tID:pg1\tPN:samtools\tVN:1.17\tPP:prev\tCL:cmd\tXY:pg_extra",
        ]
    )
    meta = parse_alignment_header(header_text, file_path="align.bam", fmt="bam")
    assert meta.header_other["XX"] == "hd_extra"
    assert meta.references[0].other["CX"] == "contig_extra"
    assert meta.samples[0].other["ZZ"] == "rg_extra"
    assert meta.programs[0].other["XY"] == "pg_extra"
    assert meta.programs[0].previous_id == "prev"

