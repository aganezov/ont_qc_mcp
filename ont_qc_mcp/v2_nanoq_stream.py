"""Run nanoq for a nonempty stream and synthesize its native empty report."""

from __future__ import annotations

import json
import subprocess  # nosec B404: shell-free execution inside the owned pipeline group
import sys


EMPTY_NANOQ_REPORT = {
    "reads": 0,
    "bases": 0,
    "n50": 0,
    "longest": 0,
    "shortest": 0,
    "mean_length": 0,
    "median_length": 0,
    "mean_quality": None,
    "median_quality": None,
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] != "--" or len(arguments) == 1:
        print("Expected '--' followed by the nanoq command", file=sys.stderr)
        return 2
    command = arguments[1:]
    first = sys.stdin.buffer.read(64 * 1024)
    if not first:
        print(json.dumps(EMPTY_NANOQ_REPORT, separators=(",", ":")))
        return 0

    process = subprocess.Popen(  # nosec B603: validated shell-free command vector
        command,
        stdin=subprocess.PIPE,
        stdout=sys.stdout.buffer,
        stderr=sys.stderr.buffer,
    )
    broken_pipe = False
    try:
        if process.stdin is None:
            raise RuntimeError("nanoq stream adapter has no stdin pipe")
        try:
            process.stdin.write(first)
            for chunk in iter(lambda: sys.stdin.buffer.read(64 * 1024), b""):
                process.stdin.write(chunk)
        except BrokenPipeError:
            broken_pipe = True
            for _ in iter(lambda: sys.stdin.buffer.read(64 * 1024), b""):
                pass
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                broken_pipe = True
        return_code = process.wait()
        return return_code or int(broken_pipe)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":  # pragma: no cover - exercised as a pipeline stage
    raise SystemExit(main())
