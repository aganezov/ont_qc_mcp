"""
Generate tiny synthetic FASTQ and BAM fixtures for local testing.

- FASTQ: 100 reads of length 100 bp with simple qualities.
- SAM -> BAM: requires `samtools` on PATH to convert and sort.
- Sequencing summary: ONT summary with 20 rows
- VCF: VCF v4.3 with 3 SNPs + 2 indels (requires bgzip/tabix)
- GFF3: Gene annotation with 2-3 mock genes on chr1
- BED: Valid and invalid BED files for testing
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path


def generate_fastq(path: Path, reads: int = 100, length: int = 100, seed: int = 42) -> None:
    random.seed(seed)
    bases = ["A", "C", "G", "T"]
    with path.open("w", encoding="utf-8") as fh:
        for i in range(reads):
            seq = "".join(random.choices(bases, k=length))
            qual = "I" * length
            fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")


def generate_sam(path: Path, reads: int = 50, length: int = 100, seed: int = 7) -> None:
    random.seed(seed)
    bases = ["A", "C", "G", "T"]
    header = "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:10000\n"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for i in range(reads):
            seq = "".join(random.choices(bases, k=length))
            qual = "I" * length
            pos = 100 + i * 10
            # Simple perfect match cigar with random MAPQ
            mapq = 20 + (i % 30)
            fh.write(f"read{i}\t0\tchr1\t{pos}\t{mapq}\t{length}M\t*\t0\t0\t{seq}\t{qual}\n")


def sam_to_bam(sam_path: Path, bam_path: Path) -> None:
    if shutil.which("samtools") is None:
        raise SystemExit("samtools is required to build BAM fixtures")
    tmp_bam = bam_path.with_suffix(".unsorted.bam")
    subprocess.run(["samtools", "view", "-bS", "-o", str(tmp_bam), str(sam_path)], check=True)
    subprocess.run(["samtools", "sort", "-o", str(bam_path), str(tmp_bam)], check=True)
    tmp_bam.unlink(missing_ok=True)


def generate_sequencing_summary(path: Path, rows: int = 20, seed: int = 42) -> None:
    """Generate ONT sequencing summary with specified number of rows."""
    random.seed(seed)
    columns = [
        "filename",
        "read_id",
        "run_id",
        "channel",
        "start_time",
        "sequence_length_template",
        "mean_qscore_template",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for i in range(rows):
            filename = f"read_{i:04d}.fastq"
            read_id = f"read{i:04d}"
            run_id = "run_001"
            channel = random.randint(1, 512)
            start_time = random.uniform(0.0, 72.0)  # Hours
            seq_length = random.randint(1000, 50000)
            mean_qscore = round(random.uniform(8.0, 15.0), 2)
            fh.write(
                f"{filename}\t{read_id}\t{run_id}\t{channel}\t{start_time:.2f}\t{seq_length}\t{mean_qscore:.2f}\n"
            )


def generate_vcf(vcf_path: Path, seed: int = 42) -> None:
    """Generate VCF v4.3 with 3 SNPs + 2 indels."""
    random.seed(seed)
    header_lines = [
        "##fileformat=VCFv4.3",
        "##contig=<ID=chr1,length=248956422>",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1",
    ]
    variants = [
        # 3 SNPs
        ("chr1", "1000000", ".", "A", "G", "30", "PASS", ".", "GT", "0/1"),
        ("chr1", "2000000", ".", "C", "T", "40", "PASS", ".", "GT", "1/1"),
        ("chr1", "3000000", ".", "G", "A", "35", "PASS", ".", "GT", "0/1"),
        # 2 indels
        ("chr1", "4000000", ".", "AT", "A", "25", "PASS", ".", "GT", "0/1"),  # deletion
        ("chr1", "5000000", ".", "C", "CG", "20", "PASS", ".", "GT", "0/1"),  # insertion
    ]
    with vcf_path.open("w", encoding="utf-8") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        for variant in variants:
            fh.write("\t".join(variant) + "\n")


def vcf_to_gzip_and_index(vcf_path: Path) -> None:
    """Compress VCF with bgzip and create tabix index."""
    if shutil.which("bgzip") is None:
        raise SystemExit("bgzip is required to build VCF fixtures")
    if shutil.which("tabix") is None:
        raise SystemExit("tabix is required to build VCF fixtures")
    vcf_gz_path = vcf_path.with_suffix(".vcf.gz")
    # Compress with bgzip
    with vcf_gz_path.open("wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf_path)], stdout=fh, check=True)
    # Create tabix index
    subprocess.run(["tabix", "-p", "vcf", str(vcf_gz_path)], check=True)


def generate_gff3(path: Path, seed: int = 42) -> None:
    """Generate GFF3 with 2-3 mock genes on chr1."""
    random.seed(seed)
    gff3_lines = [
        "##gff-version 3",
        "##sequence-region chr1 1 248956422",
        # Gene 1: MOCK_GENE1
        "chr1\t.\tgene\t100000\t200000\t.\t+\t.\tID=gene1;Name=MOCK_GENE1",
        "chr1\t.\tmRNA\t100000\t200000\t.\t+\t.\tID=mrna1;Parent=gene1",
        "chr1\t.\texon\t100000\t120000\t.\t+\t.\tID=exon1;Parent=mrna1",
        "chr1\t.\texon\t180000\t200000\t.\t+\t.\tID=exon2;Parent=mrna1",
        # Gene 2: MOCK_GENE2
        "chr1\t.\tgene\t500000\t600000\t.\t-\t.\tID=gene2;Name=MOCK_GENE2",
        "chr1\t.\tmRNA\t500000\t600000\t.\t-\t.\tID=mrna2;Parent=gene2",
        "chr1\t.\texon\t500000\t520000\t.\t-\t.\tID=exon3;Parent=mrna2",
        "chr1\t.\texon\t580000\t600000\t.\t-\t.\tID=exon4;Parent=mrna2",
        # Gene 3: MOCK_GENE3
        "chr1\t.\tgene\t1000000\t1100000\t.\t+\t.\tID=gene3;Name=MOCK_GENE3",
        "chr1\t.\tmRNA\t1000000\t1100000\t.\t+\t.\tID=mrna3;Parent=gene3",
        "chr1\t.\texon\t1000000\t1050000\t.\t+\t.\tID=exon5;Parent=mrna3",
    ]
    with path.open("w", encoding="utf-8") as fh:
        for line in gff3_lines:
            fh.write(line + "\n")


def generate_bed_files(valid_path: Path, invalid_path: Path) -> None:
    """Generate valid and invalid BED files."""
    # Valid BED file with 5 intervals
    valid_lines = [
        "chr1\t1000\t2000\tgene1_exon1\t.\t+",
        "chr1\t5000\t6000\tgene1_exon2\t.\t+",
        "chr1\t10000\t12000\tgene2_exon1\t.\t-",
        "chr1\t15000\t16000\tgene2_exon2\t.\t-",
        "chr1\t20000\t25000\tgene3_exon1\t.\t+",
    ]
    with valid_path.open("w", encoding="utf-8") as fh:
        for line in valid_lines:
            fh.write(line + "\n")

    # Invalid BED file with deliberate errors
    invalid_lines = [
        "chr1\t1000\t2000\tvalid_interval\t.\t+",  # Valid line
        "chr1\t5000\t3000\tstart_gt_end\t.\t+",  # Error: start > end
        "chr1\tabc\t6000\tnon_integer_start\t.\t+",  # Error: non-integer start
        "chr1\t10000\txyz\tnon_integer_end\t.\t-",  # Error: non-integer end
        "chr1\t20000\t25000\tvalid_interval2\t.\t+",  # Valid line
    ]
    with invalid_path.open("w", encoding="utf-8") as fh:
        for line in invalid_lines:
            fh.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Output directory for fixtures")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Existing fixtures
    fastq_path = out_dir / "synthetic_reads.fastq"
    sam_path = out_dir / "synthetic_alignment.sam"
    bam_path = out_dir / "synthetic_alignment.bam"

    generate_fastq(fastq_path)
    generate_sam(sam_path)
    sam_to_bam(sam_path, bam_path)

    print(f"Wrote {fastq_path}")
    print(f"Wrote {sam_path}")
    print(f"Wrote {bam_path}")

    # New fixtures
    seq_summary_path = out_dir / "sequencing_summary_mock.txt"
    vcf_path = out_dir / "tiny.vcf"
    gff3_path = out_dir / "genes_mock.gff3"
    valid_bed_path = out_dir / "valid.bed"
    invalid_bed_path = out_dir / "invalid.bed"

    generate_sequencing_summary(seq_summary_path)
    print(f"Wrote {seq_summary_path}")

    generate_vcf(vcf_path)
    vcf_to_gzip_and_index(vcf_path)
    print(f"Wrote {vcf_path}")
    print(f"Wrote {vcf_path}.gz")
    print(f"Wrote {vcf_path}.gz.tbi")

    generate_gff3(gff3_path)
    print(f"Wrote {gff3_path}")

    generate_bed_files(valid_bed_path, invalid_bed_path)
    print(f"Wrote {valid_bed_path}")
    print(f"Wrote {invalid_bed_path}")


if __name__ == "__main__":
    main()
