"""Keep the bundled real alignment on the platform this server targets."""

import gzip
import hashlib
import json
import struct
from pathlib import Path


def test_bundled_alignment_is_ont():
    path = Path(__file__).parent / "fixtures" / "real" / "hg002_ont_chr1.bam"
    with gzip.open(path, "rb") as stream:
        assert stream.read(4) == b"BAM\x01"
        header_length = struct.unpack("<i", stream.read(4))[0]
        header = stream.read(header_length).decode()
    groups = [
        dict(field.split(":", 1) for field in line.split("\t")[1:])
        for line in header.splitlines()
        if line.startswith("@RG\t")
    ]
    if groups:
        assert {group.get("PL", "").upper() for group in groups} == {"ONT"}
    assert any("PN:minimap2" in line and "-ax map-ont " in line for line in header.splitlines())


def test_bundled_files_match_provenance():
    fixtures = Path(__file__).parent / "fixtures"
    provenance = json.loads((fixtures / "real" / "provenance.json").read_text())
    assert provenance["sample"].startswith("HG002 ")
    for name, expected in provenance["files"].items():
        data = (fixtures / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name

    with gzip.open(fixtures / "real" / "hg002_ont_chr1.fq.gz", "rt") as stream:
        lines = stream.read().splitlines()
    assert len(lines) // 4 == provenance["fastq_reads"] == 60
    assert sum(len(sequence) for sequence in lines[1::4]) == provenance["fastq_bases"] == 1701648

    with gzip.open(fixtures / "real" / "hg002_ont_chr1.vcf.gz", "rt") as stream:
        lines = stream.read().splitlines()
    assert "##source=Clair3" in lines
    assert next(line for line in lines if line.startswith("#CHROM")).split("\t")[9:] == ["hg002"]
    variants = [line.split("\t") for line in lines if not line.startswith("#")]
    assert len(variants) == provenance["variant_records"] == 27
    assert all(row[0] == "chr1" and 4130001 <= int(row[1]) <= 4140000 for row in variants)
