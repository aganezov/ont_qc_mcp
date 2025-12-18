import json
import logging
import os
import shlex
import subprocess  # nosec B404: intentional use for CLI execution
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Sequence

from .threadpool import run_sync


logger = logging.getLogger(__name__)
_PROGRESS_ENABLED = os.getenv("MCP_PROGRESS", "0") not in {"", "0", "false", "False"}


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

    def __init__(self, result: CommandResult, stderr_max_lines: int = 20, message_override: str | None = None):
        message = (
            message_override
            or _truncate_stderr(result.stderr, stderr_max_lines)
            or f"Command failed: {' '.join(result.cmd)}"
        )
        super().__init__(message)
        self.result = result


def _decode_bytes(data) -> str:
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")
    return str(data)


def report_progress(message: str) -> None:
    """Emit opt-in progress updates for long-running operations."""
    if _PROGRESS_ENABLED:
        logger.info("progress: %s", message)


def run_command(
    cmd: Sequence[str],
    timeout: int = 120,
    stdin: IO[str] | None = None,
    stdout_path: str | Path | None = None,
    max_stdout_chars: int | None = None,
    max_stderr_chars: int | None = None,
) -> CommandResult:
    """Run a command and capture stdio. When stdout_path is provided, stream stdout to that file."""
    out_file: IO[str] | None = None

    def _bound(text: str, limit: int | None) -> str:
        if limit is None or limit <= 0 or len(text) <= limit:
            return text
        truncated = len(text) - limit
        return f"{text[:limit]}... (truncated {truncated} chars)"

    try:
        logger.debug("Running command: %s", format_cmd(cmd))
        if stdout_path is not None:
            out_file = Path(stdout_path).open("w", encoding="utf-8")
            process = subprocess.run(  # nosec B603: commands built from trusted args, shell=False
                cmd,
                check=False,
                text=True,
                stdout=out_file,
                stderr=subprocess.PIPE,
                timeout=timeout,
                stdin=stdin,
            )
        else:
            process = subprocess.run(  # nosec B603: commands built from trusted args, shell=False
                cmd,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout,
                stdin=stdin,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_bytes(exc.stdout) if stdout_path is None else f"<streamed to {stdout_path}>"
        stderr = _decode_bytes(exc.stderr)
        result = CommandResult(
            cmd=cmd,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )
        if max_stdout_chars:
            result.stdout = _bound(result.stdout, max_stdout_chars)
        if max_stderr_chars:
            result.stderr = _bound(result.stderr, max_stderr_chars)
        message = f"Command timed out after {timeout}s: {format_cmd(cmd)}"
        if stderr:
            message += f"\n{_truncate_stderr(stderr)}"
        raise CommandError(result, message_override=message) from exc
    finally:
        if out_file is not None:
            out_file.close()

    stdout_val = process.stdout if stdout_path is None else f"<streamed to {stdout_path}>"
    bounded_stdout = _bound(stdout_val or "", max_stdout_chars)
    bounded_stderr = _bound(process.stderr or "", max_stderr_chars)
    result = CommandResult(cmd=cmd, returncode=process.returncode, stdout=bounded_stdout, stderr=bounded_stderr)
    if process.returncode != 0:
        logger.warning("Command failed rc=%s: %s", process.returncode, format_cmd(cmd))
        raise CommandError(result)
    logger.debug("Command succeeded rc=%s: %s", process.returncode, format_cmd(cmd))
    return result


async def run_command_async(
    cmd: Sequence[str], timeout: int = 120, stdin: IO[str] | None = None, stdout_path: str | Path | None = None
) -> CommandResult:
    """
    Async wrapper that runs the command in a worker thread to avoid blocking.
    """
    return await run_sync(run_command, cmd, timeout, stdin, stdout_path)


def run_command_with_retry(
    cmd: Sequence[str],
    *,
    timeout: int = 120,
    stdin: IO[str] | None = None,
    stdout_path: str | Path | None = None,
    max_attempts: int = 1,
    backoff_seconds: float = 0.5,
    max_stdout_chars: int | None = None,
    max_stderr_chars: int | None = None,
) -> CommandResult:
    """
    Run a command with simple exponential backoff retry for transient failures.
    Retries on CommandError; caller decides max_attempts.
    """
    attempt = 0
    last_error: CommandError | None = None
    while attempt < max_attempts:
        try:
            return run_command(
                cmd,
                timeout=timeout,
                stdin=stdin,
                stdout_path=stdout_path,
                max_stdout_chars=max_stdout_chars,
                max_stderr_chars=max_stderr_chars,
            )
        except CommandError as exc:  # pragma: no cover - exercised via callers
            last_error = exc
            attempt += 1
            if attempt >= max_attempts:
                break
            logger.warning("Retrying command (attempt %s/%s): %s", attempt + 1, max_attempts, format_cmd(cmd))
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    if last_error is None:
        raise RuntimeError("Command retry exhausted without captured error")
    raise last_error


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


__all__ = [
    "CommandError",
    "CommandResult",
    "format_cmd",
    "report_progress",
    "run_command",
    "run_command_async",
    "run_command_with_retry",
    "_truncate_stderr",
]
