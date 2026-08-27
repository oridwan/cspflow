"""Scheduler adapters.  Only `generate`, `screen` and `dft` ever reach here."""

from __future__ import annotations

from ..config.schema import Machine
from .base import (
    JobSpec,
    JobState,
    JobStatus,
    Limits,
    Remedy,
    Scheduler,
    Throttle,
    chunk_array,
    compute_throttle,
)
from .local import LocalScheduler
from .slurm import SchedulerError, SlurmScheduler, normalise_state

__all__ = [
    "JobSpec", "JobState", "JobStatus", "Limits", "LocalScheduler", "Remedy",
    "Scheduler", "SchedulerError", "SlurmScheduler", "Throttle", "chunk_array",
    "compute_throttle", "for_machine", "normalise_state",
]


def for_machine(machine: Machine, *, dry_run: bool = False) -> Scheduler:
    """The adapter this machine profile asks for."""
    if machine.scheduler == "local":
        return LocalScheduler(machine, dry_run=dry_run)
    return SlurmScheduler(machine, dry_run=dry_run)
