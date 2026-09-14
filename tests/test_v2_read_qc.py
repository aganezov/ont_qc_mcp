import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.parsers import parse_nanoq_json
from ont_qc_mcp.process_control import CANCEL_EVENT
from ont_qc_mcp.v2_contracts import ReadQCRequest
from ont_qc_mcp.v2_execution import PipelineStage, PipelineStageError, RequestDeadline, run_pipeline
from ont_qc_mcp.v2_read_qc import read_qc
from ont_qc_mcp.v2_read_records import parse_report


def _python_stage(name: str, code: str) -> PipelineStage:
    return PipelineStage(name, (sys.executable, "-c", code))


def _record_stage(*args: str) -> PipelineStage:
    return PipelineStage(
        "read_record_filter",
        (sys.executable, "-m", "ont_qc_mcp.v2_read_records", *args),
    )


def test_record_filter_counts_selected_and_excludes_missing_sequence() -> None:
    sam = (
        "@SQ\tSN:chr1\tLN:20\n"
        "kept\t0\tchr1\t1\t30\t4M\t*\t0\t0\tACGT\tIIII\n"
        "missing\t0\tchr1\t5\t30\t4M\t*\t0\t0\t*\t*\n"
    )
    result = run_pipeline(
        [
            _python_stage("sam", f"print({sam!r}, end='')"),
            _record_stage(),
            _python_stage(
                "sink",
                "import json,sys; records=[x for x in sys.stdin if not x.startswith('@')]; "
                "print(json.dumps({'records': len(records)}))",
            ),
        ],
        RequestDeadline(5),
    )

    assert json.loads(result.final.stdout) == {"records": 1}
    assert parse_report(result.stages[1].stderr).selected_records == 2
    assert parse_report(result.stages[1].stderr).emitted_sequences == 1
    assert parse_report(result.stages[1].stderr).conversion_exclusions == 1


def test_missing_quality_failure_is_not_masked_by_downstream_success() -> None:
    sam = "r1\t0\tchr1\t1\t30\t4M\t*\t0\t0\tACGT\t*\n"
    with pytest.raises(PipelineStageError) as caught:
        run_pipeline(
            [
                _python_stage("sam", f"print({sam!r}, end='')"),
                _record_stage("--quality-required"),
                _python_stage("nanoq", "import sys; sys.stdin.read(); print('{\"reads\":0}')"),
            ],
            RequestDeadline(5),
        )
    assert caught.value.stage == "read_record_filter"
    assert "absent QUAL" in caught.value.result.stderr


def test_empty_alignment_stream_yields_native_zero_report_without_starting_nanoq() -> None:
    result = run_pipeline(
        [
            _python_stage("empty", "pass"),
            PipelineStage(
                "nanoq",
                (
                    sys.executable,
                    "-m",
                    "ont_qc_mcp.v2_nanoq_stream",
                    "--",
                    "/definitely/missing/nanoq",
                    "--stats",
                    "--json",
                ),
            ),
        ],
        RequestDeadline(5),
    )
    stats = parse_nanoq_json(result.final.stdout)
    assert stats.read_count == 0
    assert stats.total_bases == 0
    assert stats.mean_qscore is None
    assert stats.median_qscore is None


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
def test_stream_adapter_child_is_cancelled_with_the_owned_pipeline_group(tmp_path: Path) -> None:
    nanoq = tmp_path / "nanoq"
    pid_file = nanoq.with_suffix(".pid")
    nanoq.write_text(
        f"#!{sys.executable}\n"
        "import os, signal\n"
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        "signal.pause()\n"
    )
    nanoq.chmod(0o755)
    event = threading.Event()
    token = CANCEL_EVENT.set(event)
    timer = threading.Timer(0.5, event.set)
    timer.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            run_pipeline(
                [
                    _python_stage("fastq", "print('@r1\\nA\\n+\\nI')"),
                    PipelineStage(
                        "nanoq",
                        (
                            sys.executable,
                            "-m",
                            "ont_qc_mcp.v2_nanoq_stream",
                            "--",
                            str(nanoq),
                            "--stats",
                            "--json",
                        ),
                    ),
                ],
                RequestDeadline(5),
            )
    finally:
        timer.cancel()
        timer.join()
        CANCEL_EVENT.reset(token)

    assert pid_file.exists()
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while True:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            pytest.fail("nested nanoq child survived pipeline cancellation")
        time.sleep(0.02)


@pytest.fixture
def fake_nanoq(tmp_path: Path) -> Path:
    script = tmp_path / "nanoq"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "pathlib.Path(__file__).with_suffix('.calls').open('a').write('call\\n')\n"
        "if '--read-lengths' in args:\n"
        "    pathlib.Path(args[args.index('--read-lengths') + 1]).write_text('10\\n20\\n')\n"
        "if '--read-qualities' in args:\n"
        "    pathlib.Path(args[args.index('--read-qualities') + 1]).write_text('10\\n20\\n')\n"
        "print(json.dumps({'reads': 2, 'bases': 30, 'n50': 20, 'longest': 20, "
        "'shortest': 10, 'mean_length': 15, 'median_length': 15, "
        "'mean_quality': 15, 'median_quality': 15}))\n"
    )
    script.chmod(0o755)
    return script


def test_fastq_default_and_requested_sections_use_one_nanoq_execution(fake_nanoq: Path, tmp_path: Path) -> None:
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@r1\nAAAAAAAAAA\n+\n++++++++++\n@r2\nCCCCCCCCCCCCCCCCCCCC\n+\n55555555555555555555\n")
    tools = ToolPaths(nanoq=str(fake_nanoq))
    cfg = ExecutionConfig(nanoq_aux_stats=True)

    default = read_qc({"path": str(fastq)}, tools=tools, exec_cfg=cfg)
    assert default.selected_records == default.emitted_sequences == 2
    assert default.results[0].length is not None
    assert default.results[0].length.mean_length == 15
    assert default.results[0].read_quality is not None
    assert default.results[0].length_distribution is None
    assert default.results[0].quality_distribution is None
    assert fake_nanoq.with_suffix(".calls").read_text().splitlines() == ["call"]

    detailed = read_qc(
        {
            "path": str(fastq),
            "metrics": ["length_distribution", "quality_distribution"],
        },
        tools=tools,
        exec_cfg=cfg,
    )
    assert detailed.results[0].length is None
    assert detailed.results[0].read_quality is None
    assert detailed.results[0].length_distribution is not None
    assert sum(bin.count for bin in detailed.results[0].length_distribution.histogram) == 2
    assert detailed.results[0].quality_distribution is not None
    assert sum(bin.count for bin in detailed.results[0].quality_distribution.histogram) == 2
    assert fake_nanoq.with_suffix(".calls").read_text().splitlines() == ["call", "call"]


def test_fastq_inferred_format_rejects_alignment_controls(fake_nanoq: Path, tmp_path: Path) -> None:
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@r1\nA\n+\nI\n")
    with pytest.raises(ValueError, match="FASTQ input does not support"):
        read_qc(
            {"path": str(fastq), "selection": {"min_mapq": 10}},
            tools=ToolPaths(nanoq=str(fake_nanoq)),
        )


@pytest.mark.parametrize("args", [["--min-len", "10"], ["--trim-start=5"], ["-q", "7"]])
def test_population_changing_nanoq_options_are_rejected(args: list[str]) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        ReadQCRequest.model_validate({"path": "reads.fastq", "extra_args": {"nanoq": args}})
