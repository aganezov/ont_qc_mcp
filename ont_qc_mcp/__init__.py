"""ONT QC MCP package."""

__all__ = [
    "config",
    "cli_wrappers",
    "parsers",
    "schemas",
    "tools",
    "app_server",
]

# Expose app_server as a package attribute for runtime access and type checking.
from . import app_server  # noqa: F401
