import os
from dataclasses import dataclass, field
from shutil import which
from typing import Dict, List, Optional


def _env_or_default(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)


def _env_int(env_var: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class ToolPaths:
    """Paths to external CLI tools used by the MCP."""

    nanoq: str = _env_or_default("NANOQ", "nanoq")
    chopper: str = _env_or_default("CHOPPER", "chopper")
    cramino: str = _env_or_default("CRAMINO", "cramino")
    mosdepth: str = _env_or_default("MOSDEPTH", "mosdepth")
    samtools: str = _env_or_default("SAMTOOLS", "samtools")

    def as_dict(self) -> Dict[str, str]:
        return {
            "nanoq": self.nanoq,
            "chopper": self.chopper,
            "cramino": self.cramino,
            "mosdepth": self.mosdepth,
            "samtools": self.samtools,
        }

    def missing(self) -> List[str]:
        """Return missing tools based on PATH resolution."""
        missing: List[str] = []
        for name, path in self.as_dict().items():
            if which(path) is None:
                missing.append(name)
        return missing

    def with_overrides(self, **overrides: str) -> "ToolPaths":
        data = self.as_dict()
        data.update({k: v for k, v in overrides.items() if v})
        return ToolPaths(**data)


# Conservative execution defaults; can be overridden via environment variables.
DEFAULT_TOOL_TIMEOUTS: Dict[str, int] = {
    "nanoq": 300,
    "chopper": 300,
    "cramino": 300,
    "mosdepth": 600,
    "samtools": 300,
}


@dataclass
class ExecutionConfig:
    """
    Control runtime defaults for MCP tool invocations.

    Environment overrides:
    - MCP_THREADS_DEFAULT / MCP_THREADS_<TOOL>
    - MCP_TIMEOUT_DEFAULT / MCP_TIMEOUT_<TOOL>
    """

    default_threads: int = field(default_factory=lambda: _env_int("MCP_THREADS_DEFAULT", 2))
    default_timeout: int = field(default_factory=lambda: _env_int("MCP_TIMEOUT_DEFAULT", 600))
    per_tool_threads: Dict[str, int] = field(
        default_factory=lambda: {
            tool: _env_int(f"MCP_THREADS_{tool.upper()}", None)  # type: ignore[arg-type]
            for tool in DEFAULT_TOOL_TIMEOUTS.keys()
            if os.getenv(f"MCP_THREADS_{tool.upper()}") is not None
        }
    )
    per_tool_timeouts: Dict[str, int] = field(
        default_factory=lambda: {
            tool: _env_int(f"MCP_TIMEOUT_{tool.upper()}", DEFAULT_TOOL_TIMEOUTS[tool])
            for tool in DEFAULT_TOOL_TIMEOUTS.keys()
        }
    )

    def threads_for(self, tool: str) -> int:
        """Return threads default for a tool, applying env overrides."""
        value = self.per_tool_threads.get(tool)
        return value if value is not None else self.default_threads

    def timeout_for(self, tool: str) -> int:
        """Return timeout default for a tool, applying env overrides."""
        base = self.per_tool_timeouts.get(tool)
        return base if base is not None else self.default_timeout

