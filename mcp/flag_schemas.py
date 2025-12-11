from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

FlagType = Literal["bool", "int", "float", "str", "path"]


class FlagDef(BaseModel):
    """Definition for a single CLI flag exposed to MCP."""

    param: str = Field(..., description="Logical parameter name used in MCP requests")
    name: str = Field(..., description="Long CLI flag, e.g. --min-len")
    short: Optional[str] = Field(default=None, description="Short CLI flag, e.g. -l")
    type: FlagType = Field(..., description="Expected value type")
    description: str
    default: Optional[Any] = None
    aliases: List[str] = Field(default_factory=list, description="Additional accepted MCP flag keys")

    def all_keys(self) -> List[str]:
        """Return all accepted keys for this flag (param + aliases)."""
        return [self.param, *self.aliases]


# Conservative flag sets that do not change output schema/shape
TOOL_FLAGS: Dict[str, List[FlagDef]] = {
    "nanoq": [
        FlagDef(param="min_len", name="--min-len", short="-l", type="int", description="Minimum read length (bp)"),
        FlagDef(param="max_len", name="--max-len", short="-L", type="int", description="Maximum read length (bp)"),
        FlagDef(param="min_qual", name="--min-qual", short="-q", type="float", description="Minimum mean Q-score"),
        FlagDef(param="max_qual", name="--max-qual", type="float", description="Maximum mean Q-score"),
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
    ],
    "chopper": [
        FlagDef(param="headcrop", name="--headcrop", type="int", description="Trim bases from start"),
        FlagDef(param="tailcrop", name="--tailcrop", type="int", description="Trim bases from end"),
        FlagDef(param="minlength", name="--minlength", type="int", description="Minimum read length"),
        FlagDef(param="maxlength", name="--maxlength", type="int", description="Maximum read length"),
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
    ],
    "cramino": [
        FlagDef(param="hist", name="--hist", type="bool", description="Emit histograms"),
        FlagDef(param="scaled", name="--scaled", type="bool", description="Weight histograms by bases"),
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
    ],
    "mosdepth": [
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
        FlagDef(param="window", name="--by", type="int", description="Window size", aliases=["by"]),
        FlagDef(param="mapq", name="--mapq", type="int", description="Minimum MAPQ to include"),
    ],
    "samtools": [
        FlagDef(param="threads", name="-@", type="int", description="Worker threads"),
    ],
}

# Recipes provide guided presets for common workflows
RECIPES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "nanoq": {
        "strict_qc": {"min_len": 1000, "min_qual": 10},
        "lenient_qc": {"min_len": 200, "min_qual": 7},
    },
    "chopper": {
        "aggressive_trim": {"headcrop": 50, "tailcrop": 50, "minlength": 500},
    },
    "cramino": {
        "with_hist_scaled": {"hist": True, "scaled": True},
    },
    "mosdepth": {
        "window_1kb": {"window": 1000},
    },
    "samtools": {},
}


def get_tool_flags(tool: str) -> List[FlagDef]:
    """Return flag definitions for a tool (empty list if unknown)."""
    return TOOL_FLAGS.get(tool, [])


def get_tool_recipes(tool: str) -> Dict[str, Dict[str, Any]]:
    """Return recipes for a tool (empty dict if unknown)."""
    return RECIPES.get(tool, {})
