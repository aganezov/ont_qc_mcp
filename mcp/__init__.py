"""ONT QC MCP package."""

from pathlib import Path
from pkgutil import extend_path

_paths = list(extend_path(__path__, __name__))
_here = str(Path(__file__).resolve().parent)

# Put this repository first, keep site-packages next, then any other duplicate paths.
def _path_priority(p: str) -> int:
    if p == _here:
        return 0
    if "site-packages" in p:
        return 1
    return 2

__path__ = sorted(_paths, key=_path_priority)

__all__ = [
    "config",
    "cli_wrappers",
    "parsers",
    "schemas",
    "tools",
    "app_server",
]

