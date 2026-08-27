"""Stage implementations, and the registry the driver builds from.

The registry is deliberately explicit rather than discovered by import scanning.
A stage that silently fails to register would show up as "nothing to do", which
is indistinguishable from a finished campaign -- and that is the one report a
driver must never get wrong.
"""

from __future__ import annotations

from pathlib import Path

from ..config.loader import ResolvedConfig
from .base import Stage, StageReport, WorkItem
from .screen_stage import ScreenStage
from .source_stage import SourceStage

__all__ = ["Stage", "StageReport", "WorkItem", "ScreenStage", "SourceStage",
           "build_registry", "IMPLEMENTED", "PLANNED"]

# What exists today, in funnel order.
IMPLEMENTED = ["source", "screen"]

# What does not, and which milestone brings it. Named here so `csp run` can say
# "screen arrives in M1" rather than "unknown stage".
PLANNED = {
    "generate": "M3",
    "reference": "M1",
    "calibrate": "M1",
    "filter": "M1",
    "dft": "M2",
    "analyze": "M4",
}


def build_registry(cfg: ResolvedConfig, base_dir: Path | None = None) -> list[Stage]:
    """Every stage that has an implementation, given this configuration."""
    return [SourceStage(cfg, base_dir), ScreenStage(cfg)]
