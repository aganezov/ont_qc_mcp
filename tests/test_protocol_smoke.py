import anyio

from ont_qc_mcp import app_server


def test_list_tools_and_resources_smoke():
    tools = anyio.run(app_server.list_tools)
    resources = anyio.run(app_server.list_resources)
    assert tools
    assert resources
    assert any(str(r.uri).startswith("tool://") for r in resources)

