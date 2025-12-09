import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CommandResult:
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "cmd": list(self.cmd),
                "returncode": self.returncode,
                "stdout": self.stdout,
                "stderr": self.stderr,
            },
            ensure_ascii=False,
        )


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, result: CommandResult):
        super().__init__(result.stderr or f"Command failed: {' '.join(result.cmd)}")
        self.result = result


def run_command(cmd: Sequence[str], timeout: int = 120) -> CommandResult:
    """Run a command and capture stdio."""
    process = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    result = CommandResult(cmd=cmd, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    if process.returncode != 0:
        raise CommandError(result)
    return result


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)

