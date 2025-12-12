import anyio
import json
import importlib
import logging

from ont_qc_mcp import app_server


def test_list_tools_and_resources_smoke():
    tools = anyio.run(app_server.list_tools)
    resources = anyio.run(app_server.list_resources)
    assert tools
    assert resources
    assert any(str(r.uri).startswith("tool://") for r in resources)


def test_provenance_includes_request_and_effective_settings():
    result = anyio.run(app_server.dispatch_tool, "env_status", {})
    assert not result.isError
    payload = json.loads(result.content[0].text)
    provenance = payload.get("provenance", {})
    assert provenance.get("request_id")
    assert provenance.get("concurrency_limit") == app_server.EXEC_CFG.max_concurrent_operations
    assert provenance.get("effective_threads") == app_server.EXEC_CFG.threads_for("env_status")
    assert provenance.get("effective_timeout") == app_server.EXEC_CFG.timeout_for("env_status")


def test_provenance_verbose_mode(monkeypatch):
    monkeypatch.setenv("MCP_INCLUDE_PROVENANCE", "1")
    # Reload to pick up env flag
    srv = importlib.reload(app_server)

    result = anyio.run(srv.dispatch_tool, "env_status", {})
    assert not result.isError
    payload = json.loads(result.content[0].text)
    provenance = payload.get("provenance", {})
    assert provenance.get("resolved_paths")
    assert provenance.get("python_version")
    assert "package_version" in provenance


def test_request_id_logged(caplog):
    caplog.set_level(logging.INFO, logger=app_server.__name__)
    result = anyio.run(app_server.dispatch_tool, "env_status", {})
    assert not result.isError

    # Expect a tool_call_start log with a request_id tag/prefix.
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("tool_call_start" in msg for msg in messages)
    assert any("request_id" in msg or msg.startswith("[") for msg in messages)
