"""MLIP engines."""

from __future__ import annotations

from .base import MLIP, BatchStats, RelaxResult, validate_structure
from .mattersim_engine import MatterSimEngine, apply_ase_compat_shim

__all__ = ["MLIP", "BatchStats", "MatterSimEngine", "RelaxResult",
           "apply_ase_compat_shim", "for_config"]


def for_config(screen) -> MLIP:
    """The engine a campaign's `screen:` block asks for."""
    if screen.mlip == "mattersim":
        return MatterSimEngine(
            model=screen.mattersim.model,
            fmax=screen.mattersim.fmax,
            max_steps=screen.mattersim.max_steps,
        )
    raise NotImplementedError(
        f"screen.mlip={screen.mlip!r} is accepted by the schema but has no engine yet. "
        f"Implemented: mattersim."
    )
