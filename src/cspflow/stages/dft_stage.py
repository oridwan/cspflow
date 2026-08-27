"""Stage 6 -- DFT.

The unit of work is **one structure at one recipe step**, submitted as one task
of a job array. That granularity is deliberate: one structure's failure is one
array task, recorded with a reason, rather than a batch that has to be
disentangled afterwards.

A structure walks the recipe's steps in order (`relax` then `static` in the
shipped recipe), and its position is a number on the row rather than a
directory-naming convention. That is what makes the walk resumable: a driver
that restarts reads the number, not the filesystem.

**The retry ladder is matched to causes, not applied uniformly.** From the
scheduler layer (D046) a job carries both a process outcome and a remedy, and
from the VASP parser it carries a physics outcome. The two are kept apart the
whole way through:

    TIMEOUT              -> resume from CONTCAR, more walltime
    ionic_step_limit     -> resume from CONTCAR, raise NSW
    scf_not_converged    -> ALGO = Normal, raise NELM
    exit code 127        -> do NOT retry; it is a missing binary, and retrying
                            burns three submission slots to reproduce it

Every attempt records what it changed, so `csp status --why` can replay not just
whether a structure succeeded but what was done to make it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..dft.recipe import Recipe, load_recipe
from ..dft.vasp.inputs import InputError, resolve_inputs, write_inputs
from ..dft.vasp.parse import read_job_directory
from ..scheduler.base import JobSpec, JobStatus, Remedy
from .base import StageReport, WorkItem

# Where a structure is in the recipe, stored on the row rather than inferred
# from the filesystem so that a restarted driver reads it rather than guessing.
STEP_KEY = "dft_step"
DIR_KEY = "dft_dir"
# Set by calibrate:pilot on the structures it chose. Mirrored here rather than
# imported to keep the two stages from importing each other.
PILOT_KEY = "pilot"
ATTEMPT_KEY = "dft_attempt"


class DftStage:
    name = "dft"
    role = "cpu"
    in_process = False

    def __init__(self, cfg: ResolvedConfig, recipe: Recipe | None = None) -> None:
        self.cfg = cfg
        self.recipe = recipe or load_recipe(cfg.campaign.dft.recipe)

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._ready(store))

    def _ready(self, store: Store) -> list[Any]:
        """Structures selected for DFT that are not finished and not in flight.

        Held back by the pilot gate unless they *are* the pilot. Stage 4b is the
        barrier between the cheap tier and the expensive one, and it cannot
        answer without some DFT of its own -- so its own members always pass,
        and everything else waits for its verdict.
        """
        gate = self._pilot_gate(store)
        out = []
        for state in (StructureState.selected.value, StructureState.dft_done.value):
            for row in store.structures(state=state):
                if gate and not row.key_value_pairs.get(PILOT_KEY):
                    continue
                step = int(row.key_value_pairs.get(STEP_KEY, 0))
                if step < len(self.recipe.stages):
                    out.append(row)
        return out

    def _pilot_gate(self, store: Store) -> str:
        """Why the expensive tier is held, or "" if it is open.

        `on_fail: block` (the default) means: no non-pilot DFT until 4b has
        returned a verdict, and none at all if that verdict is FAIL. `warn`
        reports and proceeds; `off` disables the gate entirely.

        A campaign with no `calibrate:` block, or one whose pilot found nothing
        to select from, is not gated -- the barrier exists to stop spending on a
        model that has not been checked, not to stop a campaign that has nothing
        to check it with.
        """
        calibrate = getattr(self.cfg.campaign, "calibrate", None)
        pilot = getattr(calibrate, "pilot", None) if calibrate else None
        policy = getattr(getattr(pilot, "on_fail", None), "value",
                         getattr(pilot, "on_fail", "off"))
        if policy != "block":
            return ""

        latest = store.latest_calibration("pilot")
        if latest is None:
            has_pilot = any(r.key_value_pairs.get(PILOT_KEY) for r in store.structures())
            return ("waiting for the pilot calibration (4b)" if has_pilot
                    else "")
        if latest["verdict"] == "fail":
            return f"pilot calibration FAILED: {latest['detail'][:120]}"
        return ""

    def claim(self, store: Store, budget: int) -> list[WorkItem]:
        items = []
        for row in self._ready(store)[:budget]:
            step = int(row.key_value_pairs.get(STEP_KEY, 0))
            stage = self.recipe.stages[step]
            store.set_structure_state(int(row.id), StructureState.dft_queued,
                                      **{STEP_KEY: step})
            items.append(WorkItem(
                key=f"dft-{row.id}-{stage.name}",
                structure_ids=[int(row.id)],
                payload={"step": step, "step_name": stage.name,
                         "attempt": int(row.key_value_pairs.get(ATTEMPT_KEY, 0))},
            ))
        return items

    # -- the job -----------------------------------------------------------

    def build(self, items: list[WorkItem], workdir: Path) -> JobSpec:
        """Write every task's input directory, then one array over them.

        Inputs are written now rather than inside the job so that a failure to
        assemble them -- an unresolvable POTCAR, a mixed 4f convention -- is
        caught here, on the login node, in milliseconds, instead of on a compute
        node after the queue wait.
        """
        workdir = workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        directories = []
        for item in items:
            directory = workdir / item.key
            self._write_inputs(item, directory)
            directories.append(str(directory))

        tag = items[0].key
        manifest = workdir / f"{tag}.tasks.json"
        manifest.write_text(json.dumps({"dirs": directories}, indent=2))

        stage = self.recipe.stages[items[0].payload["step"]]
        resources = {**self.recipe.stages[0].resources, **stage.resources}
        launcher = self.cfg.machine.codes.mpi_launcher
        binary = self.cfg.machine.codes.vasp_std or "vasp_std"

        command = "\n".join([
            f'DIRS=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))'
            f"['dirs'][int(sys.argv[2])])\" {manifest} ${{SLURM_ARRAY_TASK_ID:-0}})",
            'cd "$DIRS"',
            f"{launcher} {binary} > vasp.out 2>&1",
            "echo done > VASP_DONE",   # a marker that the process exited, nothing more
        ])

        return JobSpec(
            name=tag, stage=self.name, workdir=workdir, command=command,
            role=self.role,
            ntasks=int(resources.get("ntasks", self.cfg.machine.defaults.ntasks)),
            cpus_per_task=int(resources.get("cpus_per_task", 1)),
            mem=str(resources.get("mem", self.cfg.machine.defaults.mem)),
            time=str(resources.get("time", "24:00:00")),
            array_size=len(items),
        )

    def _write_inputs(self, item: WorkItem, directory: Path) -> None:
        from ..db.store import Store as _Store

        store = _Store.open(self.cfg.campaign_db)
        try:
            atoms = store.get_structure(item.structure_ids[0]).toatoms()
        finally:
            store.close()

        stage = self.recipe.stages[item.payload["step"]]
        overrides = item.payload.get("incar_overrides") or {}
        if overrides:
            stage = stage.with_overrides(overrides)

        try:
            resolved = resolve_inputs(atoms, stage, self.cfg.campaign.dft,
                                      self.cfg.machine)
        except InputError as exc:
            raise InputError(
                f"structure {item.structure_ids[0]} at step '{stage.name}': {exc}"
            ) from exc
        write_inputs(resolved, atoms, directory)
        item.payload["settings_hash"] = resolved.settings_hash

    # -- folding the answer back -------------------------------------------

    def reconcile(self, store: Store, job_row: Any, status: JobStatus,
                  items: list[WorkItem]) -> None:
        workdir = Path(job_row["workdir"])
        for item in items:
            sid = item.structure_ids[0]
            directory = workdir / item.key
            outcome = read_job_directory(directory)
            step = int(item.payload.get("step", 0))
            step_name = item.payload.get("step_name", "")

            store.add_filter_event(
                structure_id=sid, gate=f"dft:{step_name}:converged",
                passed=bool(outcome.converged),
                value=float(outcome.n_ionic_steps),
                threshold=float(outcome.step_limit) if outcome.step_limit else None,
                detail=outcome.exit_reason,
            )
            if outcome.energy is not None:
                store.add_relaxation(
                    structure_id=sid, engine=f"vasp:{step_name}",
                    energy=outcome.energy, e_per_atom=outcome.e_per_atom,
                    converged=outcome.converged, n_steps=outcome.n_ionic_steps,
                )

            if outcome.converged:
                self._advance(store, sid, step, outcome, directory)
            else:
                self._retry_or_fail(store, sid, step, step_name, outcome, status, item)

    def _advance(self, store: Store, sid: int, step: int, outcome,
                 directory: Path) -> None:
        """This step succeeded.  Move to the next, or finish."""
        # Where the outputs are, recorded on the row rather than left to be
        # rebuilt from a filename convention later. `analyze` reads it: the
        # alternative is a second place that knows how job directories are
        # named, and two such places drift.
        kv: dict[str, Any] = {STEP_KEY: step + 1, ATTEMPT_KEY: 0,
                              DIR_KEY: str(directory.resolve())}
        if outcome.energy is not None:
            kv["vasp_energy"] = outcome.energy
        if outcome.e_per_atom is not None:
            kv["e_per_atom"] = outcome.e_per_atom
        if outcome.magnetisation is not None:
            kv["magnetisation"] = outcome.magnetisation

        finished = step + 1 >= len(self.recipe.stages)
        store.set_structure_state(
            sid, StructureState.dft_done if finished else StructureState.selected, **kv
        )

    def _retry_or_fail(self, store: Store, sid: int, step: int, step_name: str,
                       outcome, status: JobStatus, item: WorkItem) -> None:
        """Apply the ladder, or stop and say why."""
        attempt = int(item.payload.get("attempt", 0)) + 1
        max_attempts = _max_attempts(self.recipe.stages[step])

        rule = self._rule_for(step, outcome, status)
        if rule is None or attempt >= max_attempts:
            reason = outcome.exit_reason or status.raw_state or "unknown"
            store.set_structure_state(
                sid, StructureState.failed,
                dft_fail_reason=f"{step_name}: {reason}"[:200],
                **{ATTEMPT_KEY: attempt},
            )
            return

        store.set_structure_state(
            sid, StructureState.selected,
            **{STEP_KEY: step, ATTEMPT_KEY: attempt,
               "dft_last_remedy": json.dumps(rule.get("set", {}))[:200]},
        )

    def _rule_for(self, step: int, outcome, status: JobStatus) -> dict | None:
        """Match a failure to a remedy.  A remedy has to fit its cause.

        Exit 127 is never retried: it is `command not found`, this account hit it
        287 times in 60 days, and retrying it unchanged burns a submission slot
        to get the identical failure.
        """
        if status.remedy() is Remedy.do_not_retry:
            return None

        triggers = {outcome.exit_reason}
        if status.raw_state.startswith("TIMEOUT"):
            triggers.add("timeout")
        if status.raw_state.startswith("OUT_OF_MEMORY"):
            triggers.add("out_of_memory")
        if outcome.unconverged_but_finished:
            # VASP exited cleanly without reaching the criterion. Which criterion
            # it missed decides the remedy, and `exit_reason` already carries it
            # -- `ionic_step_limit` wants a resume with a higher NSW, an SCF
            # failure wants a different algorithm. They are not interchangeable.
            triggers.add(outcome.exit_reason or "scf_not_converged")
            if outcome.step_limit and outcome.n_ionic_steps >= outcome.step_limit:
                triggers.add("ionic_step_limit")
            elif outcome.exit_reason.startswith("finished without"):
                # The parser's phrasing for "exited cleanly, never reached the
                # force criterion, and did not hit NSW either" -- which is an SCF
                # problem rather than an ionic one.
                triggers.add("scf_not_converged")

        for rule in self.recipe.stages[step].retry:
            if rule.get("when") in triggers:
                return rule
        return None

    def run(self, store: Store) -> StageReport:            # pragma: no cover
        raise AssertionError("dft is a submitted stage; the driver calls claim/build")


def _max_attempts(stage) -> int:
    """One attempt per ladder rung, plus the original."""
    return len(stage.retry) + 1
