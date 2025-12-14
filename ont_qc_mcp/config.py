import os
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which


def _env_or_default(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)


def _env_int(env_var: str, default: int | None) -> int | None:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bytes(env_var: str, default: int | None) -> int | None:
    """
    Read an integer MB value from env and return bytes. Returns None when unset/invalid.
    """
    raw_val = _env_int(env_var, None)
    if raw_val is None:
        return default
    return raw_val * 1024 * 1024


def _conda_env_bin() -> Path | None:
    """
    Resolve the active conda env bin directory when CONDA_PREFIX is set.
    Returns None when no conda environment is active.
    """
    prefix = os.getenv("CONDA_PREFIX")
    if not prefix:
        return None
    return Path(prefix) / "bin"


_CONDA_ENV_BIN = _conda_env_bin()
_CARGO_BIN = Path.home() / ".cargo" / "bin"


def _preferred_tool_path(env_var: str, candidates: list[str | Path | None], fallback: str) -> str:
    override = os.getenv(env_var)
    if override:
        return override
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return fallback


@dataclass
class ToolPaths:
    """Paths to external CLI tools used by the MCP."""

    nanoq: str = field(
        default_factory=lambda: _preferred_tool_path(
            "NANOQ",
            [(_CARGO_BIN / "nanoq"), (_CONDA_ENV_BIN / "nanoq") if _CONDA_ENV_BIN else None],
            "nanoq",
        )
    )
    chopper: str = field(
        default_factory=lambda: _preferred_tool_path(
            "CHOPPER",
            [(_CONDA_ENV_BIN / "chopper") if _CONDA_ENV_BIN else None],
            "chopper",
        )
    )
    cramino: str = field(
        default_factory=lambda: _preferred_tool_path(
            "CRAMINO",
            [(_CONDA_ENV_BIN / "cramino") if _CONDA_ENV_BIN else None],
            "cramino",
        )
    )
    mosdepth: str = field(
        default_factory=lambda: _preferred_tool_path(
            "MOSDEPTH",
            [(_CONDA_ENV_BIN / "mosdepth") if _CONDA_ENV_BIN else None],
            "mosdepth",
        )
    )
    samtools: str = field(
        default_factory=lambda: _preferred_tool_path(
            "SAMTOOLS",
            [(_CONDA_ENV_BIN / "samtools") if _CONDA_ENV_BIN else None],
            "samtools",
        )
    )
    docker: str = field(default_factory=lambda: _preferred_tool_path("DOCKER", [], "docker"))
    apptainer: str = field(default_factory=lambda: _preferred_tool_path("APPTAINER", [], "apptainer"))
    singularity: str = field(default_factory=lambda: _preferred_tool_path("SINGULARITY", [], "singularity"))
    igv: str = field(default_factory=lambda: _preferred_tool_path("IGV", [], "igv.sh"))
    xvfb_run: str = field(default_factory=lambda: _preferred_tool_path("XVFB_RUN", [], "xvfb-run"))

    def as_dict(self) -> dict[str, str]:
        return {
            "nanoq": self.nanoq,
            "chopper": self.chopper,
            "cramino": self.cramino,
            "mosdepth": self.mosdepth,
            "samtools": self.samtools,
            "docker": self.docker,
            "apptainer": self.apptainer,
            "singularity": self.singularity,
            "igv": self.igv,
            "xvfb_run": self.xvfb_run,
        }

    def resolved(self) -> dict[str, str]:
        """Return best-effort resolved paths (absolute when found)."""
        resolved: dict[str, str] = {}
        for name, path in self.as_dict().items():
            resolved_path = which(path)
            resolved[name] = resolved_path or path
        return resolved

    def missing(self) -> list[str]:
        """Return missing tools based on PATH resolution."""
        missing: list[str] = []
        for name, path in self.as_dict().items():
            if which(path) is None:
                missing.append(name)
        return missing

    def with_overrides(self, **overrides: str) -> "ToolPaths":
        data = self.as_dict()
        data.update({k: v for k, v in overrides.items() if v})
        return ToolPaths(**data)


# Conservative execution defaults; can be overridden via environment variables.
DEFAULT_TOOL_TIMEOUTS: dict[str, int] = {
    "nanoq": 300,
    "chopper": 300,
    "cramino": 300,
    "mosdepth": 600,
    "samtools": 300,
    "igv": 600,
}

# Tools for which we do NOT set threads by default (leave unset unless explicitly overridden).
DISABLE_THREADS_DEFAULT: set[str] = {"nanoq", "samtools"}


@dataclass
class ExecutionConfig:
    """
    Control runtime defaults for MCP tool invocations.

    Environment overrides:
    - MCP_THREADS_DEFAULT / MCP_THREADS_<TOOL>
    - MCP_TIMEOUT_DEFAULT / MCP_TIMEOUT_<TOOL>

    Per-tool timeouts are seeded from DEFAULT_TOOL_TIMEOUTS and can be
    overridden via MCP_TIMEOUT_<TOOL>. `default_timeout` applies only when
    a tool name is not present in the per_tool_timeouts mapping.
    """

    default_threads: int | None = field(default_factory=lambda: _env_int("MCP_THREADS_DEFAULT", 4))
    default_timeout: int | None = field(default_factory=lambda: _env_int("MCP_TIMEOUT_DEFAULT", 600))
    max_file_size_bytes: int | None = field(default_factory=lambda: _env_bytes("MCP_MAX_FILE_MB", None))
    max_concurrent_operations: int | None = field(default_factory=lambda: _env_int("MCP_MAX_CONCURRENCY", 4))
    igv_container_image: str = field(
        default_factory=lambda: _env_or_default("MCP_IGV_CONTAINER_IMAGE", "aganezov/igv_snapper:0.2")
    )
    igv_sif_path: str | None = field(default_factory=lambda: os.getenv("MCP_IGV_SIF_PATH"))
    per_tool_threads: dict[str, int] = field(
        default_factory=lambda: {
            tool: value
            for tool in DEFAULT_TOOL_TIMEOUTS.keys()
            if (value := _env_int(f"MCP_THREADS_{tool.upper()}", None)) is not None
        }
    )
    per_tool_timeouts: dict[str, int] = field(
        default_factory=lambda: {
            tool: value
            for tool in DEFAULT_TOOL_TIMEOUTS.keys()
            if (value := _env_int(f"MCP_TIMEOUT_{tool.upper()}", DEFAULT_TOOL_TIMEOUTS[tool])) is not None
        }
    )

    def threads_for(self, tool: str) -> int | None:
        """Return threads default for a tool, applying env overrides. None means do not set."""
        value = self.per_tool_threads.get(tool)
        if value is not None:
            return value
        if tool in DISABLE_THREADS_DEFAULT:
            return None
        return self.default_threads

    def timeout_for(self, tool: str) -> int:
        """
        Return timeout for a tool, applying env overrides.

        Per-tool defaults always exist for known tools; `default_timeout`
        is used only if the tool name is absent from `per_tool_timeouts`.
        """
        base = self.per_tool_timeouts.get(tool)
        return base if base is not None else (self.default_timeout or DEFAULT_TOOL_TIMEOUTS.get(tool, 600))


__all__ = ["ToolPaths", "ExecutionConfig", "DEFAULT_TOOL_TIMEOUTS", "DISABLE_THREADS_DEFAULT"]
