"""
Generate tiny synthetic FASTQ and BAM fixtures for local testing.

- FASTQ: 100 reads of length 100 bp with simple qualities.
- SAM -> BAM: requires `samtools` on PATH to convert and sort.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Output directory for fixtures")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fastq_path = out_dir / "synthetic_reads.fastq"
    sam_path = out_dir / "synthetic_alignment.sam"
    bam_path = out_dir / "synthetic_alignment.bam"

    generate_fastq(fastq_path)
    generate_sam(sam_path)
    sam_to_bam(sam_path, bam_path)

    print(f"Wrote {fastq_path}")
    print(f"Wrote {sam_path}")
    print(f"Wrote {bam_path}")


if __name__ == "__main__":
    main()
