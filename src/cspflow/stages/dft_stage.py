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
LAST_REMEDY_KEY = "dft_last_remedy"
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
        self.recipe = recipe or load_recipe(cfg.campaign.dft.recipe, cfg.base_dir)

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._ready(store))

    def _ready(self, store: Store) -> list[Any]:
        """Structures selected for DFT that are not finished and not in flight.

        Held back by the pilot gate unless they *are* the pilot. Stage 4b is the
        barrier between the cheap tier and the expensive one, and it cannot
        answer without some DFT of its own -- so its own members always pass,
        and everything else waits for its verdict.

        Ordered by `select.rank_by` and capped by `select.max_total`. Both were
        in the schema and the shipped template and read by nothing, so a
        campaign asking for at most 1,500 DFT jobs ranked by hull distance got
        every candidate in database order.
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
        return self._rank_and_cap(store, out)

    def _rank_and_cap(self, store: Store, rows: list[Any]) -> list[Any]:
        """Order by `select.rank_by`, then apply `select.max_total`.

        A structure already part-way through the recipe keeps its place at the
        front: abandoning a half-finished relaxation to start a better-ranked
        one from scratch spends more and finishes less.

        `max_total` is a ceiling on the campaign, not a per-cycle throttle, so
        the count includes every structure that has already entered DFT --
        finished, failed or in progress. Capping the ready list alone would let
        a finished structure make room for a new one and the campaign would
        never stop.
        """
        select = getattr(self.cfg.campaign.dft, "select", None)
        if select is None:
            return rows

        key = getattr(select, "rank_by", "") or ""
        started = [r for r in rows if int(r.key_value_pairs.get(STEP_KEY, 0)) > 0]
        fresh = [r for r in rows if int(r.key_value_pairs.get(STEP_KEY, 0)) == 0]

        if key:
            def rank(row):
                value = row.key_value_pairs.get(key)
                # A structure with no value for the ranking key sorts last, not
                # first: an absent hull distance is not a good one.
                return (value is None, float(value) if value is not None else 0.0)

            fresh.sort(key=rank)

        cap = getattr(select, "max_total", None)
        if not cap:
            return started + fresh

        spent = _entered_dft(store)
        room = max(int(cap) - spent, 0)
        return (started + fresh)[:max(room, len(started))]

    def resource_hint(self) -> tuple[int, str]:
        """(ntasks, walltime) for one task, from the recipe's own resources.

        The driver's budget estimate has to come from here: the DFT resources
        live per recipe step, not in the campaign's `dft:` block, so nothing
        outside this stage can find them.  The longest step is used, since the
        estimate is a ceiling.
        """
        ntasks, hours, walltime = 0, 0.0, "24:00:00"
        for stage in self.recipe.stages:
            resources = {**self.recipe.stages[0].resources, **stage.resources}
            ntasks = max(ntasks, int(resources.get("ntasks", 0) or 0))
            time = str(resources.get("time", "24:00:00"))
            parsed = _hours(time)
            if parsed > hours:
                hours, walltime = parsed, time
        return ntasks or self.cfg.machine.defaults.ntasks, walltime

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
            kv = row.key_value_pairs
            step = int(kv.get(STEP_KEY, 0))
            stage = self.recipe.stages[step]
            store.set_structure_state(int(row.id), StructureState.dft_queued,
                                      **{STEP_KEY: step})
            # The ladder's decision, carried forward. Without this the remedy was
            # recorded on the row and then never read: `set: {NSW: 200}` was
            # stored as `dft_last_remedy` and the rerun was written with the
            # original NSW. Every retry repeated the identical calculation and
            # failed the identical way, which is worse than not retrying at all
            # -- it costs the same again and looks like diligence.
            remedy = _decode_remedy(kv.get(LAST_REMEDY_KEY))
            items.append(WorkItem(
                key=f"dft-{row.id}-{stage.name}",
                structure_ids=[int(row.id)],
                payload={"step": step, "step_name": stage.name,
                         "attempt": int(kv.get(ATTEMPT_KEY, 0)),
                         "incar_overrides": remedy.get("set", {}),
                         "remedy": remedy.get("remedy", "")},
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

        # Whatever a previous attempt left here is moved aside before anything
        # is written, so a retry does not erase the evidence of what it is
        # retrying. `csp status --why` names the remedy; the archive is where
        # you look to see whether it was the right one.
        archived = _archive_previous(directory, int(item.payload.get("attempt", 0)))

        store = _Store.open(self.cfg.campaign_db)
        try:
            row = store.get_structure(item.structure_ids[0])
            atoms = row.toatoms()
            previous_dir = row.key_value_pairs.get(DIR_KEY)
        finally:
            store.close()

        # Where this step starts from, in order of preference.
        #
        # 1. A retry that asked to resume picks up its own previous attempt.
        # 2. Otherwise a step after the first starts from the *previous step's*
        #    CONTCAR. This is the one that matters most: `static` exists to give
        #    a high-accuracy energy AT THE RELAXED GEOMETRY, and its energy is
        #    what goes onto the DFT hull. Started from the structure in the
        #    database it runs on the generated cell instead, and reports a
        #    number that looks entirely plausible and is wrong by whatever the
        #    relaxation was worth. Measured live before the fix: 176.15 vs
        #    179.03 A^3, 172.92 vs 179.71, 260.70 vs 260.84.
        # 3. Otherwise the structure as generated.
        source = None
        if item.payload.get("remedy") == "resume_from_contcar" and archived:
            source = archived / "CONTCAR"
        elif int(item.payload.get("step", 0)) > 0 and previous_dir:
            source = Path(previous_dir) / "CONTCAR"

        if source is not None:
            carried = _read_contcar(source)
            if carried is not None:
                atoms = carried
                item.payload["started_from"] = str(source)
            elif int(item.payload.get("step", 0)) > 0:
                raise InputError(
                    f"structure {item.structure_ids[0]} is at recipe step "
                    f"{item.payload['step']} but {source} is missing or empty. "
                    f"Running this step on the unrelaxed geometry would produce "
                    f"a plausible energy at the wrong structure.")

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
        # The remedy belongs to the step that failed. Carried into the next
        # step it silently rewrites that step's INCAR -- live, a `relax` retry's
        # `NSW: 200` landed in the `static` INCAR, so a fixed-position
        # calculation ran two hundred identical ionic steps.
        kv: dict[str, Any] = {STEP_KEY: step + 1, ATTEMPT_KEY: 0,
                              LAST_REMEDY_KEY: "",
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
               LAST_REMEDY_KEY: json.dumps({"set": rule.get("set", {}),
                                            "remedy": rule.get("remedy", "")})[:200]},
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


def _decode_remedy(raw) -> dict:
    """The ladder rung a previous cycle chose, as `{set: {...}, remedy: str}`.

    Tolerates the older shape, which stored the INCAR overrides alone.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    if "set" in value or "remedy" in value:
        return {"set": value.get("set") or {}, "remedy": value.get("remedy") or ""}
    return {"set": value, "remedy": ""}


def _archive_previous(directory: Path, attempt: int) -> Path | None:
    """Move a finished run's files into `attempt-N/`, and say where they went.

    Returns the most recent archive whether or not this call created it: a
    directory whose outputs were archived by an earlier cycle still has a
    previous attempt to resume from, and returning None there is how the resume
    quietly turned back into a restart.
    """
    import shutil

    directory = Path(directory)
    if (directory / "OUTCAR").is_file():
        target = directory / f"attempt-{max(attempt - 1, 0)}"
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            for entry in sorted(directory.iterdir()):
                if entry.is_dir() or entry.name.startswith("attempt-"):
                    continue
                shutil.move(str(entry), str(target / entry.name))
        return target
    return _latest_archive(directory)


def _latest_archive(directory: Path) -> Path | None:
    """The highest-numbered `attempt-N/` in `directory`, or None."""
    archives = []
    for entry in Path(directory).glob("attempt-*"):
        if not entry.is_dir():
            continue
        suffix = entry.name.split("-", 1)[1]
        if suffix.isdigit():
            archives.append((int(suffix), entry))
    return max(archives)[1] if archives else None


def _read_contcar(path: Path):
    """The relaxed geometry a previous attempt reached, or None.

    A CONTCAR that is absent or empty is the normal outcome of a job that died
    before its first ionic step, and resuming from nothing is not a resume --
    the caller falls back to the structure in the database.
    """
    import ase.io

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return ase.io.read(str(path), format="vasp")
    except Exception:
        return None


def _entered_dft(store: Store) -> int:
    """How many structures the campaign has already committed to DFT.

    Everything that reached `dft_done`, everything past step 0, and everything
    that failed with a DFT reason. A structure that failed still spent its
    core-hours, so it counts against the ceiling.
    """
    seen = set()
    for state in (StructureState.dft_done.value, StructureState.selected.value,
                  StructureState.dft_queued.value, StructureState.dft_running.value,
                  StructureState.failed.value):
        for row in store.structures(state=state):
            kv = row.key_value_pairs
            if int(kv.get(STEP_KEY, 0)) > 0 or state == StructureState.dft_done.value \
                    or kv.get("dft_fail_reason"):
                seen.add(int(row.id))
    return len(seen)


def _hours(walltime: str) -> float:
    """`[D-]HH:MM:SS` as hours."""
    days, _, rest = walltime.partition("-")
    if not rest:
        days, rest = "0", walltime
    parts = [float(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    return float(days) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600
