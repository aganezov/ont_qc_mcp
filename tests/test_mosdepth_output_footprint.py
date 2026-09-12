"""Coverage reports do not need mosdepth's per-base BED or its index."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import require_executable_tools
from ont_qc_mcp import cli_wrappers as cli
from ont_qc_mcp import tools
from ont_qc_mcp.config import ToolPaths


@pytest.mark.parametrize("mode", ["whole", "window", "targeted"])
def test_coverage_commands_suppress_per_base(tmp_path, monkeypatch, mode):
    def run(cmd, **kwargs):
        assert cmd.count("--no-per-base") == 1
        prefix = Path(cmd[-2])
        if mode == "targeted":
            assert cmd[cmd.index("--by") + 1] == str(tmp_path / "targets.bed")
            assert cmd[cmd.index("--thresholds") + 1] == "1,10,20"
            prefix.with_suffix(".regions.bed.gz").touch()
        else:
            if mode == "window":
                assert cmd[cmd.index("--by") + 1] == "100"
            Path(f"{prefix}.mosdepth.summary.txt").write_text("summary")

    monkeypatch.setattr(cli, "run_command", run)
    monkeypatch.setattr(cli, "parse_mosdepth_summary", lambda *args, **kwargs: None)
    if mode == "targeted":
        _, _, output = cli.run_mosdepth_targeted(
            tmp_path / "reads.bam", tmp_path / "targets.bed", ToolPaths(), thresholds=[1, 10, 20]
        )
        shutil.rmtree(output)
    else:
        cli.mosdepth_coverage(tmp_path / "reads.bam", ToolPaths(), window=100 if mode == "window" else None)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["whole", "window", "targeted"])
def test_real_mosdepth_preserves_reports_without_per_base(tmp_path, monkeypatch, mode):
    require_executable_tools(["samtools", "mosdepth"])
    samtools = shutil.which("samtools")
    mosdepth = shutil.which("mosdepth")
    assert samtools is not None and mosdepth is not None
    paths = ToolPaths(samtools=samtools, mosdepth=mosdepth)
    # Two contigs, one uncovered; the other has depths 30, 10, 1, and 0.
    rows = ["@HD\tVN:1.6\tSO:coordinate", "@SQ\tSN:chr1\tLN:1000", "@SQ\tSN:chr2\tLN:500"]
    for start, depth in [(1, 30), (201, 10), (501, 1)]:
        for i in range(depth):
            rows.append(f"r{start}_{i}\t0\tchr1\t{start}\t60\t100M\t*\t0\t0\t{'A' * 100}\t{'I' * 100}")
    sam = tmp_path / "reads.sam"
    bam = tmp_path / "reads.bam"
    bed = tmp_path / "targets.bed"
    sam.write_text("\n".join(rows) + "\n")
    bed.write_text("chr1\t0\t400\tmixed\nchr1\t500\t600\tone\nchr2\t0\t500\tzero\n")
    subprocess.run([samtools, "view", "-b", "-o", str(bam), str(sam)], check=True)
    subprocess.run([samtools, "index", str(bam)], check=True)

    original_run = cli.run_command
    manifests: dict[str, dict[str, int]] = {}
    baseline = True

    def run(cmd, **kwargs):
        assert "--no-per-base" in cmd
        effective_cmd = [arg for arg in cmd if arg != "--no-per-base"] if baseline else cmd
        result = original_run(effective_cmd, **kwargs)
        prefix = Path(cmd[-2])
        manifests["baseline" if baseline else "suppressed"] = {
            path.name.removeprefix(prefix.name): path.stat().st_size for path in prefix.parent.iterdir()
        }
        return result

    monkeypatch.setattr(cli, "run_command", run)

    def report():
        if mode == "targeted":
            return [row.model_dump() for row in tools.targeted_coverage(str(bam), bed_path=str(bed), tools=paths)]
        return cli.mosdepth_coverage(
            bam, paths, window=100 if mode == "window" else None, low_cov_threshold=5
        ).model_dump()

    expected = report()
    baseline = False
    actual = report()
    assert actual == expected
    if mode == "targeted":
        assert [row["mean_depth"] for row in actual] == [10.0, 1.0, 0.0]
        assert actual[0]["pct_coverage_20x"] == 25.0
    else:
        chr1 = next(row for row in actual["coverage_by_contig"] if row["contig"] == "chr1")
        assert chr1["mean_depth"] == pytest.approx(4.1)
    assert ".per-base.bed.gz" in manifests["baseline"]
    assert ".per-base.bed.gz.csi" in manifests["baseline"]
    assert not any(".per-base." in name for name in manifests["suppressed"])
    assert sum(manifests["suppressed"].values()) < sum(manifests["baseline"].values())
    print(json.dumps({"mode": mode, "output_bytes": manifests}, sort_keys=True))
