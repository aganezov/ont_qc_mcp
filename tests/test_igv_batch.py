from pathlib import Path

import pytest

from ont_qc_mcp.igv_batch import IgvBatchValidationError, _region_lines, generate_igv_batch
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


# --- C1: IGV batch injection safety (GHSA-634p-vpv6-fxg8) --------------------
# The batch format is line-oriented; a control char (esp. newline) in any caller
# field would inject an extra IGV command. Every interpolated field must be rejected.

_INJECTION = ["\n", "\r", "\nexecute evil", "x\nload /etc/passwd", "\x00", "\x1b"]


@pytest.mark.parametrize("payload", _INJECTION)
def test_injection_rejected_in_chrom(tmp_path: Path, payload: str) -> None:
    region = IgvRegion(chrom=f"chr1{payload}", start=1, end=100)
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region], output_path=tmp_path / "b")


@pytest.mark.parametrize("payload", _INJECTION)
def test_injection_rejected_in_genome(tmp_path: Path, payload: str) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100)
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(genome=f"hg38{payload}", tracks=["a.bam"], regions=[region], output_path=tmp_path / "b")


def test_injection_rejected_in_track(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100)
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(genome="hg38", tracks=["a.bam\nexecute evil"], regions=[region], output_path=tmp_path / "b")


def test_injection_rejected_in_region_name(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100, name="r\nload /etc/passwd")
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region], output_path=tmp_path / "b")


def test_snapshot_name_path_traversal_is_rejected(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100, name="../outside")
    with pytest.raises(IgvBatchValidationError, match="path separators"):
        generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region], output_path=tmp_path / "b")


def test_generated_snapshot_name_sanitizes_reference_path_separators(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1/alternate", start=1, end=100)
    output = tmp_path / "b"
    generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region], output_path=output)
    assert "snapshot chr1_alternate_1_100.png" in output.read_text()


def test_snapshot_names_are_unique_after_extension_normalization(tmp_path: Path) -> None:
    output = tmp_path / "b"
    generate_igv_batch(
        genome="hg38",
        tracks=["a.bam"],
        regions=[
            IgvRegion(chrom="chr1", start=1, end=2, name="foo"),
            IgvRegion(chrom="chr1", start=2, end=3, name="foo.png"),
        ],
        output_path=output,
    )
    snapshots = [line for line in _read_lines(output) if line.startswith("snapshot ")]
    assert snapshots == ["snapshot foo.png", "snapshot foo_2.png"]


def test_repeated_derived_snapshot_names_are_disambiguated(tmp_path: Path) -> None:
    output = tmp_path / "b"
    region = IgvRegion(chrom="chr1", start=1, end=2)
    generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region, region], output_path=output)
    snapshots = [line for line in _read_lines(output) if line.startswith("snapshot ")]
    assert snapshots == ["snapshot chr1_1_2.png", "snapshot chr1_1_2_2.png"]


def test_generated_suffixes_do_not_steal_requested_snapshot_names(tmp_path: Path) -> None:
    output = tmp_path / "b"
    generate_igv_batch(
        genome="hg38",
        tracks=["a.bam"],
        regions=[
            IgvRegion(chrom="chr1", start=1, end=2, name="foo"),
            IgvRegion(chrom="chr1", start=2, end=3, name="foo"),
            IgvRegion(chrom="chr1", start=3, end=4, name="foo_2"),
        ],
        output_path=output,
    )
    snapshots = [line for line in _read_lines(output) if line.startswith("snapshot ")]
    assert snapshots == ["snapshot foo.png", "snapshot foo_3.png", "snapshot foo_2.png"]


def test_injection_rejected_in_extra_commands(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100)
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(
            genome="hg38",
            tracks=["a.bam"],
            regions=[region],
            output_path=tmp_path / "b",
            extra_commands=["setSleepInterval 50\nexecute evil"],
        )


def test_legit_input_still_builds(tmp_path: Path) -> None:
    """The guard must not reject valid input (no false positives)."""
    region = IgvRegion(chrom="chr1", start=1000, end=2000, name="my_region")
    out = generate_igv_batch(
        genome="hg38",
        tracks=["sample.bam"],
        regions=[region],
        output_path=tmp_path / "b",
        extra_commands=["maxPanelHeight 500"],
    )
    text = out.read_text(encoding="utf-8")
    assert "goto chr1:1000-2000" in text
    assert "load sample.bam" in text
    assert "snapshot my_region.png" in text
    assert "execute" not in text


def test_injection_rejected_in_region_extra_commands(tmp_path: Path) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100, extra_commands=["maxPanelHeight 500\nexecute evil"])
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(genome="hg38", tracks=["a.bam"], regions=[region], output_path=tmp_path / "b")


@pytest.mark.parametrize("command", ["SNAPSHOT ../outside.png", "  snapshotDirectory ../outside"])
@pytest.mark.parametrize("scope", ["global", "region"])
def test_output_routing_commands_are_reserved(tmp_path: Path, command: str, scope: str) -> None:
    region_commands = [command] if scope == "region" else []
    global_commands = [command] if scope == "global" else []
    region = IgvRegion(chrom="chr1", start=1, end=100, extra_commands=region_commands)

    with pytest.raises(IgvBatchValidationError, match="output-routing command"):
        generate_igv_batch(
            genome="hg38",
            tracks=["a.bam"],
            regions=[region],
            output_path=tmp_path / "b",
            extra_commands=global_commands,
        )


@pytest.mark.parametrize("bad", [{"BAD\nKEY": "v"}, {"pref": "val\nexecute evil"}])
def test_injection_rejected_in_extra_preferences(tmp_path: Path, bad: dict[str, str]) -> None:
    region = IgvRegion(chrom="chr1", start=1, end=100)
    with pytest.raises(IgvBatchValidationError):
        generate_igv_batch(
            genome="hg38",
            tracks=["a.bam"],
            regions=[region],
            output_path=tmp_path / "b",
            extra_preferences=bad,
        )
