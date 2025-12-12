from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FlagType = Literal["bool", "int", "float", "str", "path"]


class FlagDef(BaseModel):
    """Definition for a single CLI flag exposed to MCP."""

    param: str = Field(..., description="Logical parameter name used in MCP requests")
    name: str = Field(..., description="Long CLI flag, e.g. --min-len")
    short: str | None = Field(default=None, description="Short CLI flag, e.g. -l")
    type: FlagType = Field(..., description="Expected value type")
    description: str
    default: Any | None = None
    aliases: list[str] = Field(default_factory=list, description="Additional accepted MCP flag keys")

    def all_keys(self) -> list[str]:
        """Return all accepted keys for this flag (param + aliases)."""
        return [self.param, *self.aliases]


# Conservative flag sets that do not change output schema/shape
TOOL_FLAGS: dict[str, list[FlagDef]] = {
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
        FlagDef(param="quality", name="--quality", short="-q", type="int", description="Minimum mean Q-score"),
        FlagDef(param="cutoff", name="--cutoff", type="int", description="Quality cutoff for adaptive trimming"),
        FlagDef(param="trim_approach", name="--trim-approach", type="str", description="Trimming strategy (e.g., fixed-crop, trim-by-quality)"),
        FlagDef(param="inverse", name="--inverse", type="bool", description="Inverse selection (emit reads that would be filtered)"),
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
    ],
    "cramino": [
        FlagDef(param="hist", name="--hist", type="bool", description="Emit histograms"),
        FlagDef(param="scaled", name="--scaled", type="bool", description="Weight histograms by bases"),
        FlagDef(param="mapq", name="--mapq", type="bool", description="Emit MAPQ histogram when supported"),
        FlagDef(param="flags", name="--flags", type="bool", description="Emit SAM flag histogram when supported"),
        FlagDef(param="format", name="--format", type="str", description="Output format (text, json, tsv)"),
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
    ],
    "mosdepth": [
        FlagDef(param="threads", name="--threads", short="-t", type="int", description="Worker threads"),
        FlagDef(param="window", name="--by", type="int", description="Window size", aliases=["by"]),
        FlagDef(param="mapq", name="--mapq", type="int", description="Minimum MAPQ to include"),
        FlagDef(param="flag", name="--flag", type="int", description="Include reads with SAM flag filter"),
        FlagDef(param="fast_mode", name="--fast-mode", short="-x", type="bool", description="Skip overlap correction (faster)"),
        FlagDef(param="quantize", name="--quantize", type="str", description="Coverage quantization bins (colon-separated)"),
        FlagDef(param="thresholds", name="--thresholds", type="str", description="Emit threshold BED for depths (comma-separated)"),
    ],
    "samtools": [
        FlagDef(param="threads", name="-@", type="int", description="Worker threads"),
    ],
}

# Recipes provide guided presets for common workflows
RECIPES: dict[str, dict[str, dict[str, Any]]] = {
    "nanoq": {
        "strict_qc": {"min_len": 1000, "min_qual": 10},
        "lenient_qc": {"min_len": 200, "min_qual": 7},
    },
    "chopper": {
        "aggressive_trim": {"headcrop": 50, "tailcrop": 50, "minlength": 500},
        "qual_trim": {"quality": 12, "cutoff": 10, "trim_approach": "trim-by-quality"},
        "inverse_short_reads": {"minlength": 1000, "inverse": True},
    },
    "cramino": {
        "with_hist_scaled": {"hist": True, "scaled": True},
        "with_flags_and_mapq": {"hist": True, "mapq": True, "flags": True},
    },
    "mosdepth": {
        "window_1kb": {"window": 1000},
        "fast_quantized": {"fast_mode": True, "quantize": "0:1:10:30:100:"},
    },
    "samtools": {},
}


def get_tool_flags(tool: str) -> list[FlagDef]:
    """Return flag definitions for a tool (empty list if unknown)."""
    return TOOL_FLAGS.get(tool, [])


def get_tool_recipes(tool: str) -> dict[str, dict[str, Any]]:
    """Return recipes for a tool (empty dict if unknown)."""
    return RECIPES.get(tool, {})


__all__ = ["FlagDef", "TOOL_FLAGS", "RECIPES", "get_tool_flags", "get_tool_recipes"]
