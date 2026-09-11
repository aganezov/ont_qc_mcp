import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("format_exit", [0, 1])
def test_local_ci_checks_formatting(tmp_path, format_exit):
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'echo "$*" >> "$CI_TEST_COMMANDS"\n'
        'if [ "$*" = "run --no-sync ruff format --check ." ]; then\n'
        '    exit "$CI_TEST_FORMAT_EXIT"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    commands = tmp_path / "commands.txt"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "CI_TEST_COMMANDS": str(commands),
        "CI_TEST_FORMAT_EXIT": str(format_exit),
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "ci-local.sh"
    result = subprocess.run(["bash", str(script), "--fast"], env=env, capture_output=True, text=True, timeout=10)

    assert "run --no-sync ruff format --check ." in commands.read_text().splitlines()
    assert result.returncode == format_exit, result.stdout + result.stderr
