"""Run the dependency audit used by CI and local validation."""

import re
import subprocess
import sys
import time

# Match the known PyPI service traceback; unknown failures remain fail-closed.
TEMPORARY_HTTP_ERROR = re.compile(
    r"^requests\.exceptions\.HTTPError: (?:429|500|502|503|504) (?:Client|Server) Error: "
    r"[^\n]*for url: https://pypi\.org/[^\s]+$",
    re.MULTILINE,
)


def main() -> int:
    delays = iter((10, 30))
    while True:
        result = subprocess.run([sys.executable, "-m", "pip_audit"], capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        retryable = TEMPORARY_HTTP_ERROR.search(result.stderr) and (
            "pip_audit._service.interface.ServiceError" in result.stderr
        )
        if result.returncode != 1 or not retryable:
            return result.returncode
        delay = next(delays, None)
        if delay is None:
            return result.returncode
        print(f"Temporary PyPI audit failure; retrying in {delay}s.", file=sys.stderr, flush=True)
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
