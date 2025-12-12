import pytest

from ont_qc_mcp.utils import CommandError, format_cmd, run_command_with_retry


def test_format_cmd_quotes_spaces():
    cmd = ["echo", "hello world", "$HOME"]
    formatted = format_cmd(cmd)
    assert "hello world" in formatted
    assert formatted.startswith("echo")


def test_run_command_with_retry_success():
    result = run_command_with_retry(["echo", "ok"], max_attempts=2)
    assert result.stdout.strip() == "ok"
    assert result.returncode == 0


def test_run_command_with_retry_failure():
    with pytest.raises(CommandError):
        run_command_with_retry(["python", "-c", "import sys; sys.exit(2)"], max_attempts=2, backoff_seconds=0.01)

