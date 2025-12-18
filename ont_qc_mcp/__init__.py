"""ONT QC MCP package."""

import importlib
from types import ModuleType

__all__ = [
    "config",
    "cli_wrappers",
    "parsers",
    "schemas",
    "tools",
    "app_server",
]

def __getattr__(name: str) -> ModuleType:
    # Keep `ont_qc_mcp.app_server` available without importing it eagerly.
    if name == "app_server":
        mod = importlib.import_module(".app_server", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
