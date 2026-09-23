from pathlib import Path

import pytest

from ont_qc_mcp import cli_wrappers, tools
from ont_qc_mcp.config import ExecutionConfig, ToolPaths
from ont_qc_mcp.utils import CommandResult


@pytest.fixture
def captured_commands(monkeypatch):
    calls = []
    fixtures = Path(__file__).parent / "fixtures"

    def run(cmd, timeout):
        calls.append((cmd, timeout))
        if cmd[0] == "cramino":
            stdout = (fixtures / "raw/cramino_hg002_ont_chr1.json").read_text()
        elif cmd[0] == "mosdepth":
            Path(f"{cmd[-2]}.mosdepth.summary.txt").write_text(
                "chrom\tlength\tbases\tmean\tmin\tmax\nchr1\t10\t20\t2\t2\t2\ntotal\t10\t20\t2\t0\t2\n"
            )
            stdout = ""
        elif cmd[0] == "samtools":
            stdout = (fixtures / "samtools_stats/default.stats").read_text()
        else:
            raise AssertionError(cmd)
        return CommandResult(cmd, 0, stdout, "")

    monkeypatch.setattr(cli_wrappers, "run_command", run)
    monkeypatch.setattr(tools, "run_command", run)
    return calls


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("include_coverage,include_errors", [(True, True), (False, False)])
def test_summary_configuration_reaches_commands(
    tmp_path, monkeypatch, captured_commands, explicit, include_coverage, include_errors
):
    path = tmp_path / "reads.bam"
    path.write_bytes(b"fixture")
    cfg = ExecutionConfig(
        per_tool_threads={"cramino": 7, "mosdepth": 9, "samtools": 11},
        per_tool_timeouts={"cramino": 17, "mosdepth": 19, "samtools": 23},
        max_file_size_bytes=20,
    )
    monkeypatch.setattr(tools, "_EXEC_CFG", ExecutionConfig() if explicit else cfg)
    report = tools.alignment_summary(
        str(path),
        tools=ToolPaths(cramino="cramino", mosdepth="mosdepth", samtools="samtools"),
        exec_cfg=cfg if explicit else None,
        include_coverage=include_coverage,
        include_error_profile=include_errors,
    )
    expected = [("cramino", "--threads", "7", 17)]
    if include_coverage:
        expected.append(("mosdepth", "--threads", "9", 19))
    if include_errors:
        expected.append(("samtools", "-@", "11", 23))
    assert len(captured_commands) == len(expected)
    for (cmd, timeout), (name, flag, value, expected_timeout) in zip(captured_commands, expected):
        assert cmd[0] == name
        assert cmd[cmd.index(flag) + 1] == value
        assert timeout == expected_timeout
    assert report.alignment is not None
    assert (report.coverage is not None) == include_coverage
    assert (report.errors is not None) == include_errors


@pytest.mark.parametrize("function", [tools.alignment_summary, tools.qc_alignment, tools.coverage_stats])
@pytest.mark.parametrize("caller_limit,global_limit,allowed", [(2, 20, False), (20, 2, True)])
def test_configuration_controls_input_limit(
    tmp_path, monkeypatch, captured_commands, function, caller_limit, global_limit, allowed
):
    path = tmp_path / "reads.bam"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(tools, "_EXEC_CFG", ExecutionConfig(max_file_size_bytes=global_limit))
    cfg = ExecutionConfig(max_file_size_bytes=caller_limit)
    paths = ToolPaths(cramino="cramino", mosdepth="mosdepth")
    if allowed:
        function(str(path), tools=paths, exec_cfg=cfg)
        assert captured_commands
    else:
        with pytest.raises(ValueError, match="configured size limit"):
            function(str(path), tools=paths, exec_cfg=cfg)
        assert not captured_commands
