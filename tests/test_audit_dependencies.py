"""The audit can recover from service outages without suppressing findings."""

import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, cast

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit-dependencies.py"


def service_error(status: int) -> str:
    return (
        f"requests.exceptions.HTTPError: {status} Server Error: Backend is unhealthy "
        "for url: https://pypi.org/pypi/ont-qc-mcp/0.1.0/json\n"
        "pip_audit._service.interface.ServiceError\n"
    )


@pytest.fixture
def audit(monkeypatch: pytest.MonkeyPatch):
    results: list[subprocess.CompletedProcess[str]] = []
    calls: list[list[str]] = []
    delays: list[float] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "text": True}
        calls.append(command)
        return results.pop(0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", delays.append)
    main = cast(Callable[[], int], runpy.run_path(str(SCRIPT))["main"])
    return main, results, calls, delays


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_temporary_service_failure_recovers(audit, capsys, status: int) -> None:
    main, results, calls, delays = audit
    results.extend(
        [
            subprocess.CompletedProcess([], 1, "", service_error(status)),
            subprocess.CompletedProcess([], 0, "No known vulnerabilities\n", ""),
        ]
    )
    assert main() == 0
    assert calls == [[sys.executable, "-m", "pip_audit"]] * 2
    assert delays == [10]
    output = capsys.readouterr()
    assert service_error(status) in output.err
    assert "No known vulnerabilities" in output.out


def test_persistent_service_failure_still_fails(audit) -> None:
    main, results, calls, delays = audit
    results.extend([subprocess.CompletedProcess([], 1, "", service_error(503))] * 3)
    assert main() == 1
    assert len(calls) == 3
    assert delays == [10, 30]


@pytest.mark.parametrize(
    "stderr",
    [
        "Found 1 known vulnerability in 1 package\n",
        "usage: pip-audit: invalid option\n",
        service_error(401),
        service_error(403),
        "pip_audit._service.interface.ServiceError: malformed version\n",
        "Unrelated error mentioning 503\n",
    ],
)
def test_other_failures_are_not_retried(audit, capsys, stderr: str) -> None:
    main, results, calls, delays = audit
    results.append(subprocess.CompletedProcess([], 1, "audit details\n", stderr))
    assert main() == 1
    assert len(calls) == 1
    assert delays == []
    output = capsys.readouterr()
    assert output.out == "audit details\n"
    assert output.err == stderr


def test_recovery_followed_by_vulnerability_still_fails(audit) -> None:
    main, results, calls, delays = audit
    results.extend(
        [
            subprocess.CompletedProcess([], 1, "", service_error(503)),
            subprocess.CompletedProcess([], 1, "vulnerability details\n", "Found 1 known vulnerability\n"),
        ]
    )
    assert main() == 1
    assert len(calls) == 2
    assert delays == [10]


def test_success_needs_no_retry(audit) -> None:
    main, results, calls, delays = audit
    results.append(subprocess.CompletedProcess([], 0, "", "No known vulnerabilities\n"))
    assert main() == 0
    assert len(calls) == 1
    assert delays == []
