"""Stage 0 as a driver stage.

Stage 0 already has a complete implementation in `cspflow.source`; this wraps it
in the `Stage` protocol so that `csp run` and `csp source` do the same thing by
construction rather than by discipline. That is the property the plan asks for
in sec.4.4 -- "manual and automatic are the same code" -- and the cheapest way
to keep it true is to give the manual command nowhere else to go.

It is an in-process stage: pure CPU bookkeeping that runs in seconds, before a
single GPU-minute is spent.
"""

from __future__ import annotations

from pathlib import Path

from ..config.loader import ResolvedConfig
from ..db.store import Store
from ..source import expand_all, write_plan
from .base import StageReport, WorkItem


class SourceStage:
    name = "source"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, base_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.base_dir = base_dir
        self._plan = None

    def pending(self, store: Store) -> int:
        """One unit of work, and only if the campaign has no rows yet.

        Stage 0 is idempotent -- `add_composition` upserts, so re-running writes
        the same rows -- but reporting it as pending forever would make the
        driver loop never conclude the campaign was finished.
        """
        return 0 if store.compositions() else 1

    def claim(self, store: Store, budget: int) -> list[WorkItem]:  # pragma: no cover
        raise AssertionError("source is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                               # pragma: no cover
        raise AssertionError("source is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items) -> None:    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        plan = expand_all(self.cfg.campaign, self.base_dir)
        stats = write_plan(plan, store)
        note = f"{len(plan.chemsystems())} chemical systems"
        if plan.collisions:
            note += f", {len(plan.collisions)} cross-source collisions reported"
        warnings = [w for r in plan.results for w in r.warnings]
        if warnings:
            note += f", {len(warnings)} warning(s)"
        self._plan = plan
        return StageReport(
            stage=self.name,
            claimed=stats.compositions + stats.structures,
            reconciled=0,
            note=note,
        )
