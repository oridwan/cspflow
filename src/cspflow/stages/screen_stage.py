"""Stage 2 -- MLIP relaxation.

A submitted stage: chunks of structures go to a GPU array. The shape of it is
worth stating because it is a deliberate departure from how the legacy scripts
work.

**Array workers never write to the database.** They read the structures they
were given (SQLite in WAL mode allows any number of concurrent readers), relax
them, and write their results to a per-task JSON file. The driver reads those
files and does all the writing, one process at a time.

The alternative -- every array task opening the campaign database for writing --
is what the ~48 concurrent DFT jobs of this cluster's CPU cap would produce, and
SQLite serialises writers with a lock. At best that is 48 processes taking turns;
at worst it is `database is locked` after the busy timeout, in a job that has
already spent its GPU minutes. Writing to a file the worker owns outright cannot
contend with anything, and the handoff is a file rename.

It also makes the failure mode benign: a worker that dies leaves no results file,
which reconciliation reports as missing rather than as silent partial data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..scheduler.base import JobSpec, JobStatus
from .base import StageReport, WorkItem

# How many structures one array task relaxes. The plan's figure (pipeline.md
# sec.4.4) is 500-2,000; the default is the low end because a task that dies
# loses everything it had not yet written, and at ~1 s per structure on GPU a
# 500-structure task is under ten minutes.
DEFAULT_CHUNK = 500


class ScreenStage:
    name = "screen"
    role = "gpu"
    in_process = False

    def __init__(self, cfg: ResolvedConfig, chunk: int = DEFAULT_CHUNK) -> None:
        self.cfg = cfg
        self.chunk = chunk

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return store.count_structures(state=StructureState.new.value)

    def estimate_tasks(self, store: Store, budget: int) -> int:
        """`budget` is in array tasks; `pending` is in structures."""
        import math

        return min(budget, math.ceil(self.pending(store) / self.chunk))

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        """Take up to `budget` structures and mark them `screening`.

        Marking on claim is what makes the driver safe to run twice: a second
        cycle, or a second driver, sees `screening` rather than `new` and does
        not take the same structures again. A worker that then dies leaves them
        in `screening`, which `csp status` reports as stuck -- visible, rather
        than quietly re-run forever.
        """
        ids = store.structure_ids(state=StructureState.new.value)[: budget * self.chunk]
        if not ids:
            return []
        items = []
        for start in range(0, len(ids), self.chunk):
            batch = ids[start: start + self.chunk]
            for sid in batch:
                store.set_structure_state(sid, StructureState.screening)
            items.append(WorkItem(key=f"screen-{batch[0]}-{batch[-1]}",
                                  structure_ids=batch))
        return items

    # -- the job -----------------------------------------------------------

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """One array, one task per chunk, driven by a manifest on disk.

        The manifest is written rather than passed on the command line because a
        2,000-id argument list is both unreadable and, at scale, longer than the
        shell will accept.
        """
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        tag = items[0].key
        manifest = workdir / f"{tag}.manifest.json"
        manifest.write_text(json.dumps({
            "key": tag,
            "db": str(self.cfg.campaign_db.resolve()),
            "model": self.cfg.campaign.screen.mattersim.model,
            "fmax": self.cfg.campaign.screen.mattersim.fmax,
            "max_steps": self.cfg.campaign.screen.mattersim.max_steps,
            "chunks": [item.structure_ids for item in items],
        }, indent=2))

        resources = self.cfg.campaign.screen.resources
        return JobSpec(
            name=tag, stage=self.name, workdir=workdir,
            command=f"csp screen-worker --manifest {manifest}",
            role=self.role, ntasks=resources.ntasks or 1,
            cpus_per_task=resources.cpus_per_task or 1,
            gpus=resources.gpus or 1, mem=resources.mem or "32G",
            time=resources.time, array_size=len(items),
            env={"CSPFLOW_MANIFEST": str(manifest)},
        )

    # -- folding the answer back -------------------------------------------

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        """Read what the workers wrote and record it.  Only the driver writes."""
        workdir = Path(job_row["workdir"])
        for index, item in enumerate(items):
            results_file = workdir / f"{item.key}.task{index}.json"
            if not results_file.is_file():
                # The task produced nothing. Its structures are still marked
                # `screening`, which is the correct record: work was claimed and
                # did not come back. They are not silently returned to `new`,
                # because an unbounded retry of a structure that crashes the MLIP
                # is a loop, not a recovery.
                for sid in item.structure_ids:
                    store.set_structure_state(
                        sid, StructureState.failed,
                        fail_reason=f"screen worker produced no results ({status.raw_state})",
                    )
                continue
            self._absorb(store, json.loads(results_file.read_text()))

    def _absorb(self, store: Store, payload: dict) -> None:
        for row in payload.get("results", []):
            sid = int(row["structure_id"])
            if row.get("error"):
                store.set_structure_state(sid, StructureState.failed,
                                          fail_reason=row["error"][:200])
                store.add_filter_event(structure_id=sid, gate="screen:validate",
                                       passed=False, detail=row["error"][:200])
                continue

            store.add_relaxation(
                structure_id=sid, engine=row.get("engine", "mattersim"),
                energy=row.get("energy"), e_per_atom=row.get("e_per_atom"),
                converged=bool(row.get("converged")), n_steps=int(row.get("n_steps", 0)),
                volume_before=row.get("volume_before"), volume_after=row.get("volume_after"),
            )
            kv = {"mlip_e_per_atom": row["e_per_atom"],
                  "mlip_converged": bool(row["converged"]),
                  "mlip_steps": int(row.get("n_steps", 0))}
            if row.get("volume_drift") is not None:
                kv["mlip_volume_drift"] = float(row["volume_drift"])
            store.set_structure_state(sid, StructureState.screened, **kv)
            store.add_filter_event(
                structure_id=sid, gate="screen:converged",
                passed=bool(row["converged"]),
                value=float(row.get("n_steps", 0)),
                threshold=float(payload.get("max_steps", 0)),
                detail="" if row["converged"] else "stopped at the step limit",
            )

    def run(self, store: Store) -> StageReport:            # pragma: no cover
        raise AssertionError("screen is a submitted stage; the driver calls claim/build")
