"""Exercise public JSON-RPC frames without an MCP client or SDK model imports."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = json.loads((_ROOT / "tests/fixtures/mcp_contract_v1.json").read_text())
_MODERN = "2026-07-28"

# Observe real entry/cleanup boundaries; do not replace validation, dispatch,
# the semaphore, worker execution, or subprocess cleanup.
_SERVER = r"""
import json, os, time
from pathlib import Path
from ont_qc_mcp import app_server as app, tools, utils

def record(event, **fields):
    data = json.dumps(dict(event=event, time=time.monotonic(), **fields)) + '\n'
    fd = os.open(os.environ['WIRE_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try: os.write(fd, data.encode())
    finally: os.close(fd)

original_run_sync = app.run_sync
async def observed_run_sync(func, *args, **kwargs):
    path = str(args[0]) if args else None
    record('worker_enter', function=func.__name__, path=path)
    try: return await original_run_sync(func, *args, **kwargs)
    finally: record('worker_finished', function=func.__name__, path=path)
app.run_sync = observed_run_sync

original_validate = tools._validate_input_file
def observed_validate(*args, **kwargs):
    record('validate_file', path=str(args[0]))
    return original_validate(*args, **kwargs)
tools._validate_input_file = observed_validate

original_cleanup = utils.cleanup_processes
def observed_cleanup(processes):
    owned = [p for p in processes if p is not None]
    record('cleanup_start', pids=[p.pid for p in owned])
    try: return original_cleanup(processes)
    finally:
        record('cleanup_finished', pids=[p.pid for p in owned],
               returncodes=[p.returncode for p in owned],
               pipes_closed=all(f is None or f.closed for p in owned for f in (p.stdin,p.stdout,p.stderr)))
utils.cleanup_processes = observed_cleanup
app.main()
"""

_NANOQ = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

def record(event, **fields):
    data = json.dumps(dict(event=event, time=time.monotonic(), pid=os.getpid(), **fields)) + '\n'
    fd = os.open(os.environ['WIRE_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try: os.write(fd, data.encode())
    finally: os.close(fd)

def resistant_term(signum, frame):
    record('term_received')

if '--child' in sys.argv:
    signal.signal(signal.SIGTERM, resistant_term)
    record('child_started')
    time.sleep(30)
    sys.exit(0)

path = Path(sys.argv[sys.argv.index('--input') + 1])
aux = [sys.argv[sys.argv.index(flag)+1] for flag in ('--read-lengths','--read-qualities') if flag in sys.argv]
if path.name.startswith('blocked'):
    signal.signal(signal.SIGTERM, resistant_term)
    child = subprocess.Popen([sys.executable, __file__, '--child'])
    record('producer_started', path=str(path), child=child.pid, aux=aux)
    time.sleep(30)
else:
    record('producer_started', path=str(path), aux=aux)
    if path.name.startswith('failure'):
        print('controlled nanoq failure', file=sys.stderr)
        sys.exit(2)
print(json.dumps(dict(reads=1, bases=4, shortest=4, longest=4, mean_length=4,
                     median_length=4, n50=4, mean_quality=40, median_quality=40)))
"""


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # A concurrently written final line may be incomplete; retry on the next poll.
    return [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]


def _wait(predicate, description: str, timeout: float = 6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for {description}")


def _pid_running(pid: int) -> bool:
    # Linux can retain a killed orphan as a zombie until its new parent reaps it.
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _WireClient:
    def __init__(self, process, protocol: str, stderr_path: Path):
        self.process = process
        self.protocol = protocol
        self.stderr_path = stderr_path
        self.transcript_path = stderr_path.with_suffix(".wire.jsonl")
        self.next_id = 0
        self.messages: dict[int, dict] = {}
        self.incoming: queue.Queue = queue.Queue()

        def read_stdout():
            try:
                for line in process.stdout:
                    message = json.loads(line)
                    self._record("received", message)
                    self.incoming.put(message)
            except Exception as exc:
                self.incoming.put(exc)
            finally:
                self.incoming.put(None)

        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()
        if protocol != _MODERN:
            reply = self.call(
                "initialize",
                {"protocolVersion": protocol, "capabilities": {}, "clientInfo": {"name": "raw-wire", "version": "1"}},
            )
            assert reply["result"]["protocolVersion"] == protocol, reply
            self.notify("notifications/initialized", {})

    def _record(self, direction: str, message: dict):
        line = json.dumps({"direction": direction, "message": message}) + "\n"
        fd = os.open(self.transcript_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def _write(self, message: dict):
        message = {"jsonrpc": "2.0", **message}
        self._record("sent", message)
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> int:
        self.next_id += 1
        params = dict(params or {})
        if self.protocol == _MODERN:
            params["_meta"] = {
                "io.modelcontextprotocol/protocolVersion": self.protocol,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "raw-wire", "version": "1"},
            }
        self._write({"id": self.next_id, "method": method, "params": params})
        return self.next_id

    def notify(self, method: str, params: dict):
        self._write({"method": method, "params": params})

    def response(self, request_id: int, timeout: float = 8) -> dict:
        deadline = time.monotonic() + timeout
        while request_id not in self.messages:
            try:
                message = self.incoming.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                pytest.fail(f"No response for {request_id}; stderr:\n{self.stderr_path.read_text()}")
            assert message is not None, f"Server disconnected; stderr:\n{self.stderr_path.read_text()}"
            assert isinstance(message, dict), message
            assert message.get("jsonrpc") == "2.0", message
            if "id" in message:
                self.messages[message["id"]] = message
        reply = self.messages.pop(request_id)
        if self.protocol == _MODERN and "result" in reply:
            assert reply["result"]["resultType"] == "complete", reply
            assert reply["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "ont-qc-mcp", reply
        return reply

    def call(self, method: str, params: dict | None = None) -> dict:
        return self.response(self.request(method, params))

    def tool(self, name: str, arguments: dict) -> dict:
        return self.call("tools/call", {"name": name, "arguments": arguments})


@pytest.fixture(params=["2025-06-18", _MODERN])
def wire_protocol(request):
    if request.param == _MODERN and int(version("mcp").split(".")[0]) < 2:
        pytest.skip("SDK1 baseline does not implement the 2026 per-request protocol")
    return request.param


@pytest.fixture(params=["compat", "anyio"])
def wire_server(tmp_path, request, wire_protocol):
    events = tmp_path / "events.jsonl"
    executable = tmp_path / "nanoq"
    executable.write_text(f"#!{sys.executable}\n{_NANOQ}")
    executable.chmod(0o755)
    stderr_path = tmp_path / "server.stderr"
    env = {
        **os.environ,
        "PYTHONPATH": str(_ROOT),
        "MCP_STDIO_TRANSPORT": request.param,
        "MCP_BLOCKING_MODE": "executor",
        "MCP_MAX_CONCURRENCY": "1",
        "MCP_THREADPOOL_WORKERS": "2",
        "MCP_NANOQ_AUX_STATS": "1",
        "MCP_CACHE_STATS": "0",
        "MCP_INCLUDE_PROVENANCE": "0",
        "MCP_PROGRESS": "0",
        "WIRE_EVENTS": str(events),
        "NANOQ": str(executable),
    }
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-c", _SERVER],
            cwd=_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            yield _WireClient(process, wire_protocol, stderr_path), events
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            # Bounded teardown also handles deliberately broken cancellation paths.
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            for event in _events(events):
                if event["event"] in {"producer_started", "child_started"} and _pid_running(event["pid"]):
                    try:
                        os.kill(event["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if process.stdout:
                process.stdout.close()


def _tool_payload(reply: dict, *, error: bool = False):
    result = reply["result"]
    assert result["isError"] is error, reply
    assert "is_error" not in result
    assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
    return json.loads(result["content"][0]["text"])


def test_wire_catalog_results_and_resources(wire_server, tmp_path):
    client, events = wire_server
    tools = client.call("tools/list")["result"]["tools"]
    schemas = {tool["name"]: tool["inputSchema"] for tool in tools}
    # The v1 fixture remains frozen. The new regional primitive is an explicit
    # additive surface with its own discovery and numerical contract tests.
    assert set(schemas) == set(_CONTRACT["tools"]) | {"regional_alignment_stats_tool"}
    assert {name: schemas[name] for name in _CONTRACT["tools"]} == _CONTRACT["tools"]
    assert all("input_schema" not in tool for tool in tools)
    for method, field, expected in [
        ("resources/list", "resources", _CONTRACT["resources"]),
        ("resources/templates/list", "resourceTemplates", _CONTRACT["templates"]),
    ]:
        actual = client.call(method)["result"][field]
        if field == "resources":
            added = [item for item in actual if item["uri"] == "tool://guidance/regional_alignment_stats_tool"]
            assert len(added) == 1
            actual = [item for item in actual if item not in added]
        assert [{key: item[key] for key in expected[0]} for item in actual] == expected
    resource = client.call("resources/read", {"uri": "tool://flags/nanoq"})["result"]["contents"][0]
    assert resource["uri"] == "tool://flags/nanoq" and resource["mimeType"] == "application/json"
    assert json.loads(resource["text"])["tool"] == "nanoq"

    summary = tmp_path / "summary.txt"
    summary.write_text("sequence_length_template\tstart_time\n4\t3600\n8\t3600\n")
    payload = _tool_payload(client.tool("sequencing_summary_tool", {"path": str(summary)}))
    assert payload["total_reads"] == 2 and payload["total_yield"] == 12
    assert payload["run_duration_hours"] == 0
    assert payload["yield_per_hour"] == [{"window_start_hours": 1.0, "yield_bp": 12, "read_count": 2}]
    assert isinstance(payload["provenance"]["request_id"], str)

    valid_fastq = tmp_path / "valid.fastq"
    valid_fastq.write_text("@r\nACGT\n+\nIIII\n")
    valid = _tool_payload(client.tool("qc_reads_fastq_tool", {"path": str(valid_fastq), "flags": {"min_len": None}}))
    assert valid["read_count"] == 1 and valid["total_bases"] == 4

    missing = _tool_payload(client.tool("sequencing_summary_tool", {"path": str(tmp_path / "missing.txt")}), error=True)
    assert missing["kind"] == "not_found" and missing["tool"] == "sequencing_summary_tool"
    assert set(missing) == {"kind", "message", "tool", "details", "request_id"}
    fastq = tmp_path / "failure.fastq"
    fastq.write_text("@r\nACGT\n+\nIIII\n")
    invalid_flag = _tool_payload(
        client.tool("qc_reads_fastq_tool", {"path": str(fastq), "flags": {"min_len": True}}), error=True
    )
    assert invalid_flag["kind"] == "validation" and "got bool" in invalid_flag["message"]
    failed = _tool_payload(client.tool("qc_reads_fastq_tool", {"path": str(fastq)}), error=True)
    assert failed["kind"] == "runtime" and "controlled nanoq failure" in failed["message"]
    assert client.call("tools/list")["result"]["tools"]


def test_wire_schema_rejects_before_worker_or_filesystem(wire_server, tmp_path):
    client, events = wire_server
    output = tmp_path / "must-not-exist"
    cases = [
        ("qc_reads_fastq_tool", {}),
        ("qc_reads_fastq_tool", {"path": 123}),
        ("qc_reads_fastq_tool", {"path": "absent.fastq", "flags": None}),
        ("qc_alignment_tool", {"path": "absent.bam", "include_hist": "false"}),
        ("coverage_stats_tool", {"path": "absent.bam", "window": True}),
        ("coverage_stats_tool", {"path": "absent.bam", "window": None}),
        ("coverage_stats_tool", {"path": "absent.bam", "low_cov_threshold": False}),
        ("igv_snapshot_tool", {"batch_file": "absent.batch", "output_dir": str(output), "snapshot_format": "jpeg"}),
        (
            "igv_snapshot_tool",
            {
                "genome": "hg38",
                "tracks": [],
                "regions": [{"chrom": "chr1", "start": True, "end": 20}],
                "output_dir": str(output),
            },
        ),
        (
            "igv_snapshot_tool",
            {"genome": "hg38", "tracks": [], "regions": [{"chrom": "chr1", "start": 0}], "output_dir": str(output)},
        ),
        ("targeted_coverage_tool", {"bam_path": "absent.bam", "location": "chr1:0-10", "bed_path": "absent.bed"}),
    ]
    for name, arguments in cases:
        reply = client.tool(name, arguments)
        assert reply["result"]["isError"] is True, reply
        assert reply["result"]["content"][0]["text"].startswith("Input validation error:"), reply
        assert not _events(events), f"Invalid {name} entered a worker or filesystem validation"
        assert not output.exists()
    # A valid call after the rejection exercises this same connection and observer.
    missing = _tool_payload(client.tool("qc_reads_fastq_tool", {"path": str(tmp_path / "missing.fastq")}), error=True)
    assert missing["kind"] == "not_found"
    assert {e["event"] for e in _events(events)} >= {"worker_enter", "validate_file"}


@pytest.mark.skipif(os.name != "posix", reason="Owned process-group cleanup requires POSIX")
@pytest.mark.parametrize("disconnect", [False, True], ids=["cancel-and-reuse", "disconnect"])
def test_wire_cancellation_keeps_capacity_until_process_cleanup(wire_server, tmp_path, disconnect):
    client, events = wire_server
    blocked = tmp_path / "blocked.fastq"
    blocked.write_text("@r\nACGT\n+\nIIII\n")
    first_id = client.request("tools/call", {"name": "qc_reads_fastq_tool", "arguments": {"path": str(blocked)}})
    first = _wait(
        lambda: next((e for e in _events(events) if e["event"] == "producer_started"), None), "producer start"
    )
    child = _wait(
        lambda: next((e for e in _events(events) if e["event"] == "child_started"), None), "owned child start"
    )
    assert first["aux"] and all(Path(path).exists() for path in first["aux"])
    if disconnect:
        client.process.stdin.close()
        try:
            client.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Disconnect did not finish server cleanup; stderr:\n{client.stderr_path.read_text()}")
    else:
        client.notify("notifications/cancelled", {"requestId": first_id, "reason": "wire regression test"})
        followup = tmp_path / "followup.fastq"
        followup.write_text("@next\nACGT\n+\nIIII\n")
        second_id = client.request("tools/call", {"name": "qc_reads_fastq_tool", "arguments": {"path": str(followup)}})
        payload = _tool_payload(client.response(second_id))
        assert payload["read_count"] == 1
        cleaned = _wait(
            lambda: next(
                (e for e in _events(events) if e["event"] == "cleanup_finished" and first["pid"] in e["pids"]), None
            ),
            "cancelled producer cleanup",
        )
        records = _events(events)
        admitted = next(e for e in records if e["event"] == "worker_enter" and e["path"] == str(followup))
        assert cleaned["time"] <= admitted["time"], "Capacity was released before the cancelled subprocess was cleaned"
        assert client.call("tools/list")["result"]["tools"]
        if first_id in client.messages:
            assert "error" in client.messages[first_id], "Cancelled work returned a success result"
    _wait(lambda: not _pid_running(first["pid"]) and not _pid_running(child["pid"]), "owned process exit")
    records = _events(events)
    cleaned = next(e for e in records if e["event"] == "cleanup_finished" and first["pid"] in e["pids"])
    assert cleaned["pipes_closed"] and cleaned["returncodes"] == [-signal.SIGKILL]
    assert any(e["event"] == "term_received" for e in records)
    assert all(not Path(path).exists() for path in first["aux"])
    assert blocked.read_text() == "@r\nACGT\n+\nIIII\n"
