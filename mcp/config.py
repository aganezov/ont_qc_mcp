import os
from dataclasses import dataclass
from shutil import which
from typing import Dict, List


def _env_or_default(env_var: str, default: str) -> str:
    return os.getenv(env_var, default)


@dataclass
class ToolPaths:
    """Paths to external CLI tools used by the MCP."""

    seqkit: str = _env_or_default("SEQKIT", "seqkit")
    samtools: str = _env_or_default("SAMTOOLS", "samtools")
    fastp: str = _env_or_default("FASTP", "fastp")
    nanoplot: str = _env_or_default("NANOPLOT", "NanoPlot")

    def as_dict(self) -> Dict[str, str]:
        return {
            "seqkit": self.seqkit,
            "samtools": self.samtools,
            "fastp": self.fastp,
            "nanoplot": self.nanoplot,
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

