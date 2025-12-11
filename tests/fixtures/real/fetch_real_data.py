"""
Fetch small real-world datasets (<5MB) for integration testing.

Defaults pull:
- ONT FASTQ: minimap2 test reads (ont.fq)
- BAM: pysam ex1.bam (generic small BAM for tooling smoke tests)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DEFAULT_FASTQ_URL = "https://raw.githubusercontent.com/lh3/minimap2/master/test/ont.fq"
DEFAULT_BAM_URL = "https://github.com/pysam-developers/pysam/raw/master/tests/pysam_data/ex1.bam"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastq-url", default=DEFAULT_FASTQ_URL, help="URL of small ONT FASTQ file")
    parser.add_argument("--bam-url", default=DEFAULT_BAM_URL, help="URL of small BAM file")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent, help="Output directory")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fastq_path = out_dir / "real_reads.fastq"
    bam_path = out_dir / "real_alignment.bam"

    download(args.fastq_url, fastq_path)
    download(args.bam_url, bam_path)

    print(f"Downloaded FASTQ to {fastq_path}")
    print(f"Downloaded BAM to {bam_path}")


if __name__ == "__main__":
    main()
