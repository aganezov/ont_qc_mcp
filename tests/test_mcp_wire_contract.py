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
_MODERN = "2026-07-28"

# Observe real entry/cleanup boundaries; do not replace validation, dispatch,
# the semaphore, worker execution, or subprocess cleanup.
_SERVER = r"""
import json, os, time
from pathlib import Path
from ont_qc_mcp import app_server as app, v2_execution, v2_read_qc

def record(event, **fields):
    data = json.dumps(dict(event=event, time=time.monotonic(), **fields)) + '\n'
    fd = os.open(os.environ['WIRE_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try: os.write(fd, data.encode())
    finally: os.close(fd)

original_run_sync = app.run_sync
async def observed_run_sync(func, *args, **kwargs):
    request = args[0] if args else None
    path = str(getattr(request, 'path', request)) if request is not None else None
    record('worker_enter', function=func.__name__, path=path)
    try: return await original_run_sync(func, *args, **kwargs)
    finally: record('worker_finished', function=func.__name__, path=path)
app.run_sync = observed_run_sync

original_validate = v2_read_qc._validate_input_file
def observed_validate(*args, **kwargs):
    record('validate_file', path=str(args[0]))
    return original_validate(*args, **kwargs)
v2_read_qc._validate_input_file = observed_validate

original_cleanup = v2_execution.cleanup_processes
def observed_cleanup(processes):
    owned = [p for p in processes if p is not None]
    record('cleanup_start', pids=[p.pid for p in owned])
    try: return original_cleanup(processes)
    finally:
        record('cleanup_finished', pids=[p.pid for p in owned],
               returncodes=[p.returncode for p in owned],
               pipes_closed=all(f is None or f.closed for p in owned for f in (p.stdin,p.stdout,p.stderr)))
v2_execution.cleanup_processes = observed_cleanup
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
    from ont_qc_mcp.v2_contracts import api_v2_contracts

    contracts = api_v2_contracts()
    assert set(schemas) == set(contracts)
    assert schemas == {
        name: contract.request_model.model_json_schema(mode="validation") for name, contract in contracts.items()
    }
    assert all("input_schema" not in tool for tool in tools)
    resources = client.call("resources/list")["result"]["resources"]
    assert {item["uri"] for item in resources} == {
        *(f"tool://guidance/{name}" for name in contracts),
        "tool://recipes/alignment_qc",
    }
    templates = client.call("resources/templates/list")["result"]["resourceTemplates"]
    assert {item["uriTemplate"] for item in templates} == {
        "tool://guidance/{tool}",
        "tool://recipes/{tool}",
    }
    recipe = client.call("resources/read", {"uri": "tool://recipes/alignment_qc"})["result"]["contents"][0]
    assert recipe["uri"] == "tool://recipes/alignment_qc" and recipe["mimeType"] == "application/json"
    calls = json.loads(recipe["text"])["recipes"]["alignment_and_coverage"]["calls"]
    assert [call["tool"] for call in calls] == ["alignment_qc", "coverage_qc"]

    summary = tmp_path / "summary.txt"
    summary.write_text("sequence_length_template\tstart_time\n4\t3600\n8\t3600\n")
    payload = _tool_payload(client.tool("run_summary", {"path": str(summary)}))
    assert payload["total_reads"] == 2 and payload["total_yield"] == 12
    assert payload["run_duration_hours"] == 0
    assert payload["yield_per_hour"] == [{"window_start_hours": 1.0, "yield_bp": 12, "read_count": 2}]

    valid_fastq = tmp_path / "valid.fastq"
    valid_fastq.write_text("@r\nACGT\n+\nIIII\n")
    valid = _tool_payload(client.tool("read_qc", {"path": str(valid_fastq)}))
    assert valid["results"][0]["length"]["read_count"] == 1
    assert valid["results"][0]["length"]["total_bases"] == 4

    missing = _tool_payload(client.tool("run_summary", {"path": str(tmp_path / "missing.txt")}), error=True)
    assert missing["kind"] == "execution_error" and missing["tool"] == "run_summary"
    assert missing["stage"] == "input_validation" and missing["partial_result_returned"] is False
    fastq = tmp_path / "failure.fastq"
    fastq.write_text("@r\nACGT\n+\nIIII\n")
    invalid_flag = _tool_payload(
        client.tool("read_qc", {"path": str(fastq), "metrics": ["length", "length"]}), error=True
    )
    assert invalid_flag["kind"] == "validation_error"
    assert "duplicates" in invalid_flag["issues"][0]["message"]
    failed = _tool_payload(client.tool("read_qc", {"path": str(fastq)}), error=True)
    assert failed["kind"] == "execution_error" and failed["backend"] == "nanoq"
    assert failed["exit_code"] == 2 and "controlled nanoq failure" in failed["message"]
    assert client.call("tools/list")["result"]["tools"]


def test_wire_schema_rejects_before_worker_or_filesystem(wire_server, tmp_path):
    client, events = wire_server
    output = tmp_path / "must-not-exist"
    cases = [
        ("read_qc", {}),
        ("read_qc", {"path": 123}),
        ("read_qc", {"path": "absent.fastq", "flags": None}),
        ("alignment_qc", {"path": "absent.bam", "include_hist": "false"}),
        ("coverage_qc", {"path": "absent.bam", "window_size": True}),
        ("coverage_qc", {"path": "absent.bam", "window_size": 0}),
        ("coverage_qc", {"path": "absent.bam", "group_by": "region"}),
        ("igv_snapshots", {"batch_file": "absent.batch", "output_dir": str(output), "snapshot_format": "jpeg"}),
        (
            "igv_snapshots",
            {
                "genome": "hg38",
                "tracks": ["reads.bam"],
                "regions": [{"chrom": "chr1", "start": True, "end": 20}],
                "output_dir": str(output),
            },
        ),
        (
            "igv_snapshots",
            {
                "genome": "hg38",
                "tracks": ["reads.bam"],
                "regions": [{"chrom": "chr1", "start": 0}],
                "output_dir": str(output),
            },
        ),
        ("variant_qc", {"path": "absent.vcf.gz", "group_by": "region"}),
    ]
    for name, arguments in cases:
        reply = client.tool(name, arguments)
        assert reply["result"]["isError"] is True, reply
        payload = json.loads(reply["result"]["content"][0]["text"])
        assert payload["kind"] == "validation_error" and payload["issues"], reply
        assert not _events(events), f"Invalid {name} entered a worker or filesystem validation"
        assert not output.exists()
    # A valid call after the rejection exercises this same connection and observer.
    missing = _tool_payload(client.tool("read_qc", {"path": str(tmp_path / "missing.fastq")}), error=True)
    assert missing["kind"] == "execution_error" and missing["stage"] == "input_validation"
    assert {e["event"] for e in _events(events)} >= {"worker_enter", "validate_file"}


@pytest.mark.skipif(os.name != "posix", reason="Owned process-group cleanup requires POSIX")
@pytest.mark.parametrize("disconnect", [False, True], ids=["cancel-and-reuse", "disconnect"])
def test_wire_cancellation_keeps_capacity_until_process_cleanup(wire_server, tmp_path, disconnect):
    client, events = wire_server
    blocked = tmp_path / "blocked.fastq"
    blocked.write_text("@r\nACGT\n+\nIIII\n")
    first_id = client.request(
        "tools/call",
        {
            "name": "read_qc",
            "arguments": {
                "path": str(blocked),
                "metrics": ["length", "read_quality", "length_distribution", "quality_distribution"],
            },
        },
    )
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
        second_id = client.request("tools/call", {"name": "read_qc", "arguments": {"path": str(followup)}})
        payload = _tool_payload(client.response(second_id))
        assert payload["results"][0]["length"]["read_count"] == 1
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
