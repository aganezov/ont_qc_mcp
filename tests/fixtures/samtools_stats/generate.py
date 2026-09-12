"""Reproduce literal samtools 1.24 SN/COV/MPC fixture rows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


CASES = {
    "default": [],
    "reference": ["-r", "reference.fa"],
    "binned": ["-c", "5,15,5"],
    "multiwidth": ["-c", "15,25,5"],
    "underflow": ["-c", "25,35,5"],
}


def generate(output_dir: Path, samtools: str) -> None:
    version = subprocess.run([samtools, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    if version != "samtools 1.24":
        raise ValueError(f"Expected samtools 1.24, got {version}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference.fa").write_text(">chr1\n" + "A" * 100 + "\n")
    rows = ["@HD\tVN:1.6\tSO:coordinate", "@SQ\tSN:chr1\tLN:100"]
    for i in range(20):
        sequence = ("N" if i < 3 else "A") + "C" + "A" * 48
        qualities = "I" + ("!" if i < 10 else "I") + "I" * 48
        rows.append(f"r{i}\t0\tchr1\t1\t60\t50M\t*\t0\t0\t{sequence}\t{qualities}\tNM:i:{2 if i < 3 else 1}")
    (output_dir / "reads.sam").write_text("\n".join(rows) + "\n")
    subprocess.run([samtools, "faidx", "reference.fa"], cwd=output_dir, check=True)
    commands: dict[str, list[str]] = {}
    for name, flags in CASES.items():
        command = [samtools, "stats", *flags, "reads.sam"]
        result = subprocess.run(command, cwd=output_dir, check=True, text=True, capture_output=True)
        # Preserve the CLI's literal fields; omit unrelated sections and path-bearing headers.
        selected = [row for row in result.stdout.splitlines() if row.startswith(("SN\t", "COV\t", "MPC\t"))]
        (output_dir / f"{name}.stats").write_text("\n".join(selected) + "\n")
        commands[name] = ["samtools", *command[1:]]
    (output_dir / "reference.fa.fai").unlink()
    files = ["reads.sam", "reference.fa", *(f"{name}.stats" for name in CASES)]
    metadata = {
        "version": version,
        "upstream_commit": "dc71c7274044d1050ccb64901731373ec7e915b6",
        "retained_sections": ["SN", "COV", "MPC"],
        "commands": commands,
        "sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in files},
    }
    (output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samtools", default=os.environ.get("SAMTOOLS", "samtools"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    generate(args.output_dir.resolve(), args.samtools)
