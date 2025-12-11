import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import IO, List, Optional, Sequence

import anyio


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


def _truncate_stderr(stderr: str, max_lines: int = 20) -> str:
    """Keep first/last chunks of stderr to avoid flooding clients."""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    if len(lines) <= max_lines * 2:
        return stderr
    return "\n".join(lines[:max_lines] + ["... (truncated) ..."] + lines[-max_lines:])


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, result: CommandResult, stderr_max_lines: int = 20):
        message = _truncate_stderr(result.stderr, stderr_max_lines) or f"Command failed: {' '.join(result.cmd)}"
        super().__init__(message)
        self.result = result


from typing import IO, Optional, Sequence


def run_command(cmd: Sequence[str], timeout: int = 120, stdin: Optional[IO[str]] = None) -> CommandResult:
    """Run a command and capture stdio."""
    process = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
        stdin=stdin,
    )
    result = CommandResult(cmd=cmd, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    if process.returncode != 0:
        raise CommandError(result)
    return result


async def run_command_async(cmd: Sequence[str], timeout: int = 120, stdin: Optional[IO[str]] = None) -> CommandResult:
    """
    Async wrapper that runs the command in a worker thread to avoid blocking.
    """
    return await anyio.to_thread.run_sync(run_command, cmd, timeout, stdin)


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)

