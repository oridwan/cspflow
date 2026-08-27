"""Structure generators.

One registry function, so that adding SymmCD or LEGO-xtal later is a new module
plus one line here -- and so that an unknown engine name fails at configuration
load with the list of what does exist, rather than at 3 a.m. in an array task.
"""

from __future__ import annotations

from .base import (GenerationOutcome, GenerationRequest, Generator, GeneratorError,
                   check_composition, legacy_batch_total, read_generated, split_batches)
from .mattergen_engine import MatterGenEngine, estimate_gpu_minutes

__all__ = ["GenerationOutcome", "GenerationRequest", "Generator", "GeneratorError",
           "check_composition", "legacy_batch_total", "read_generated", "split_batches",
           "MatterGenEngine", "estimate_gpu_minutes", "for_config", "ENGINES"]

ENGINES = {"mattergen": MatterGenEngine}


def for_config(generate_cfg) -> Generator:
    """Build the generator a resolved `generate:` block asks for."""
    engine = generate_cfg.engine
    if engine not in ENGINES:
        raise GeneratorError(
            f"unknown generator engine {engine!r}; available: {sorted(ENGINES)}")
    if engine == "mattergen":
        block = generate_cfg.mattergen
        return MatterGenEngine(
            model=block.model, mode=block.mode,
            max_batch_size=block.max_batch_size,
            timeout_per_batch=block.timeout_per_batch,
        )
    raise GeneratorError(f"engine {engine!r} is registered but has no constructor")
