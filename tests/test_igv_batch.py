from pathlib import Path

from ont_qc_mcp.igv_batch import _region_lines, generate_igv_batch
from ont_qc_mcp.schemas import IgvRegion
from ont_qc_mcp.tools import _parse_bed_regions


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_generate_batch_basic(tmp_path: Path) -> None:
    batch_path = tmp_path / "igv.batch"
    snapshot_dir = tmp_path / "out"
    region = IgvRegion(chrom="chr1", start=100, end=200, name="region1")

    generate_igv_batch(
        genome="hg38",
        tracks=["/data/sample.bam", "/data/sample.vcf.gz"],
        regions=[region],
        output_path=batch_path,
        snapshot_dir=snapshot_dir,
    )

    lines = _read_lines(batch_path)
    assert lines[0] == "genome hg38"
    assert "load /data/sample.bam" in lines
    assert "load /data/sample.vcf.gz" in lines
    assert any(line.startswith("preference SAM.SMALL_INDEL_BP_THRESHOLD") for line in lines)
    assert f"snapshotDirectory {snapshot_dir}" in lines
    assert "goto chr1:100-200" in lines
    assert "snapshot region1.png" in lines
    assert lines[-1] == "exit"


def test_generate_batch_with_extra_commands(tmp_path: Path) -> None:
    batch_path = tmp_path / "igv.batch"
    snapshot_dir = tmp_path / "out"
    region = IgvRegion(chrom="chr2", start=10, end=20)

    generate_igv_batch(
        genome="hg38",
        tracks=["/data/sample.bam"],
        regions=[region],
        output_path=batch_path,
        snapshot_dir=snapshot_dir,
        extra_commands=["maxPanelHeight 500", "sort STRAND"],
    )

    lines = _read_lines(batch_path)
    goto_idx = lines.index("goto chr2:10-20")
    # Global extra commands should appear before regions are processed.
    assert "maxPanelHeight 500" in lines[:goto_idx]
    assert "sort STRAND" in lines[:goto_idx]


def test_generate_batch_with_extra_preferences(tmp_path: Path) -> None:
    batch_path = tmp_path / "igv.batch"
    region = IgvRegion(chrom="chr3", start=5, end=15)

    generate_igv_batch(
        genome="hg19",
        tracks=["/data/sample.bam"],
        regions=[region],
        output_path=batch_path,
        snapshot_dir=tmp_path,
        extra_preferences={"SAM.SHOW_SOFT_CLIPPED": "TRUE", "SAM.FILTER_DUPLICATES": "FALSE"},
    )

    lines = _read_lines(batch_path)
    assert "preference SAM.SHOW_SOFT_CLIPPED TRUE" in lines
    assert "preference SAM.FILTER_DUPLICATES FALSE" in lines


def test_generate_batch_per_region_commands(tmp_path: Path) -> None:
    batch_path = tmp_path / "igv.batch"
    region = IgvRegion(chrom="chr4", start=50, end=80, extra_commands=["viewaspairs", "sort BASE"])

    generate_igv_batch(
        genome="hg38",
        tracks=["/data/sample.bam"],
        regions=[region],
        output_path=batch_path,
        snapshot_dir=tmp_path,
    )

    lines = _read_lines(batch_path)
    goto_idx = lines.index("goto chr4:50-80")
    assert lines[goto_idx + 1 : goto_idx + 3] == ["viewaspairs", "sort BASE"]
    assert lines[goto_idx + 3] == "snapshot chr4_50_80.png"


def test_parse_bed_file(tmp_path: Path) -> None:
    bed_path = tmp_path / "regions.bed"
    bed_path.write_text("chr1\t10\t20\tregionA\tviewaspairs;sort BASE\nchr2\t30\t40\n", encoding="utf-8")

    regions = _parse_bed_regions(bed_path, snapshot_format="png", min_snapshot_width=0)
    assert len(regions) == 2
    assert regions[0].name == "regionA"
    assert regions[0].extra_commands == ["viewaspairs", "sort BASE"]
    assert regions[1].chrom == "chr2"


def test_region_min_width_expansion(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr5", start=100, end=120)
    lines = _region_lines(region, snapshot_format="png", min_snapshot_width=100)
    assert lines[0] == "goto chr5:60-160"
    assert lines[-1] == "snapshot chr5_100_120.png"
