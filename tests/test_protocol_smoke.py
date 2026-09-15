import anyio
import json
import logging
import mcp_types as types

from ont_qc_mcp import app_server


def test_list_tools_and_resources_smoke():
    tools = anyio.run(app_server.list_tools)
    resources = anyio.run(app_server.list_resources)
    assert tools
    assert resources
    assert any(str(r.uri).startswith("tool://") for r in resources)


def test_environment_status_dispatch():
    result = anyio.run(app_server.dispatch_tool, "environment_status", {})
    assert not result.is_error
    assert isinstance(result.content[0], types.TextContent)
    payload = json.loads(result.content[0].text)
    assert set(payload) == {"available", "resolved_paths", "missing", "igv_runtime"}


def test_request_id_logged(caplog):
    caplog.set_level(logging.INFO, logger=app_server.__name__)
    result = anyio.run(app_server.dispatch_tool, "environment_status", {})
    assert not result.is_error

    # Expect a tool_call_start log with a request_id tag/prefix.
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("tool_call_start" in msg for msg in messages)
    assert any("request_id" in msg or msg.startswith("[") for msg in messages)
