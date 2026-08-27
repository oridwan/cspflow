"""Stage 6 -- the DFT stage and its retry ladder.

Run against synthetic VASP output directories, so the whole state machine --
walking a two-step recipe, retrying with the right remedy, giving up for the
right reason -- is exercised with no cluster and no VASP.
"""

import json
from pathlib import Path

import pytest
import yaml
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.scheduler.base import JobState, JobStatus
from cspflow.stages.dft_stage import ATTEMPT_KEY, STEP_KEY, DftStage

MACHINES = Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"
POTCARS = Path("/projects/mmi/Ridwan/potcarFiles/pmg")
has_potcars = pytest.mark.skipif(not POTCARS.is_dir(), reason="POTCAR tree not present")

CAMPAIGN = """\
name: t
machine: orion
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
dft:
  recipe: magnets
  magnetism: {{mode: ferrimagnetic_retm}}
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "campaign.db", campaign="t") as s:
        yield s


def add_selected(store, n=1):
    ids = []
    for _ in range(n):
        ids.append(store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                                       origin=Origin.generated,
                                       state=StructureState.selected))
    return ids


OUTCAR_CONVERGED = """\
 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                  Total CPU time used (sec):       100.0
"""

OUTCAR_STEP_LIMIT = """\
 General timing and accounting informations for this job:
                  Total CPU time used (sec):       100.0
"""


def fake_job_output(directory: Path, *, converged: bool, energy=-16.9,
                    steps=None, nsw=99):
    # An unconverged run that reached NSW is the ionic-step-limit case; a
    # converged one stops earlier.
    steps = steps if steps is not None else (12 if converged else nsw)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"   1 F= {energy:.6E} E0= {energy:.6E}  d E =0.0  mag=     4.2"]
    for i in range(2, steps + 1):
        lines.append(f"   {i} F= {energy:.6E} E0= {energy:.6E}  d E =0.0  mag=     4.2")
    (directory / "OSZICAR").write_text("\n".join(lines) + "\n")
    body = OUTCAR_CONVERGED if converged else OUTCAR_STEP_LIMIT
    (directory / "OUTCAR").write_text(f"   NIONS = 2\n" + body)
    # NSW is read from the INCAR, not the OUTCAR -- which is where the parser
    # looks and where a real job directory has it.
    (directory / "INCAR").write_text(f"NSW = {nsw}\nNCORE = 4\n")
    (directory / "VASP_DONE").write_text("done\n")
    return directory


class TestClaim:
    def test_pending_counts_selected_structures(self, cfg, store):
        add_selected(store, 3)
        assert DftStage(cfg).pending(store) == 3

    def test_claiming_marks_them_queued(self, cfg, store):
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        assert len(items) == 2
        assert store.count_structures(state="dft_queued") == 2
        assert stage.pending(store) == 0

    def test_a_claim_carries_its_recipe_step(self, cfg, store):
        add_selected(store, 1)
        item = DftStage(cfg).claim(store, budget=1)[0]
        assert item.payload["step"] == 0
        assert item.payload["step_name"] == "relax"

    def test_the_budget_caps_the_claim(self, cfg, store):
        add_selected(store, 5)
        assert len(DftStage(cfg).claim(store, budget=2)) == 2

    def test_a_finished_structure_is_not_reclaimed(self, cfg, store):
        sid = add_selected(store, 1)[0]
        store.set_structure_state(sid, StructureState.dft_done, **{STEP_KEY: 2})
        assert DftStage(cfg).pending(store) == 0


class TestRecipeWalk:
    def _reconcile(self, cfg, store, sid, workdir, *, converged, step=0,
                   step_name="relax", attempt=0, status=None):
        from cspflow.stages.base import WorkItem

        item = WorkItem(key=f"dft-{sid}-{step_name}", structure_ids=[sid],
                        payload={"step": step, "step_name": step_name,
                                 "attempt": attempt})
        fake_job_output(Path(workdir) / item.key, converged=converged)
        DftStage(cfg).reconcile(
            store, {"workdir": str(workdir), "id": 1},
            status or JobStatus(job_id="1", state=JobState.done, raw_state="COMPLETED"),
            [item],
        )
        return item

    def test_a_converged_relax_advances_to_the_next_step(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(cfg, store, sid, tmp_path, converged=True)
        row = store.get_structure(sid)
        assert row.state == "selected" and row.dft_step == 1

    def test_the_last_step_finishes_the_structure(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(cfg, store, sid, tmp_path, converged=True,
                        step=1, step_name="static")
        assert store.get_structure(sid).state == "dft_done"

    def test_the_energy_is_recorded(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(cfg, store, sid, tmp_path, converged=True)
        assert store.get_structure(sid).vasp_energy is not None
        assert store.relaxation_outcomes() == {"vasp:relax:converged": 1}

    def test_a_gate_event_is_recorded_either_way(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._reconcile(cfg, store, sid, tmp_path, converged=False)
        gates = [e["gate"] for e in store.filter_events(sid)]
        assert "dft:relax:converged" in gates


class TestRetryLadder:
    def _fail(self, cfg, store, sid, workdir, *, status, attempt=0):
        from cspflow.stages.base import WorkItem

        item = WorkItem(key=f"dft-{sid}-relax-{attempt}", structure_ids=[sid],
                        payload={"step": 0, "step_name": "relax", "attempt": attempt})
        fake_job_output(Path(workdir) / item.key, converged=False)
        DftStage(cfg).reconcile(store, {"workdir": str(workdir), "id": 1},
                                status, [item])

    def test_an_ionic_step_limit_is_retried(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        row = store.get_structure(sid)
        assert row.state == "selected"          # queued again
        assert row.dft_attempt == 1

    def test_the_remedy_that_was_applied_is_recorded(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        assert "NSW" in store.get_structure(sid).dft_last_remedy

    def test_a_timeout_is_retried(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.timeout,
                                    raw_state="TIMEOUT"))
        assert store.get_structure(sid).state == "selected"

    def test_command_not_found_is_never_retried(self, cfg, store, tmp_path):
        """Exit 127 hit this account 287 times in 60 days; retrying cannot fix it."""
        sid = add_selected(store, 1)[0]
        self._fail(cfg, store, sid, tmp_path,
                   status=JobStatus(job_id="1", state=JobState.failed,
                                    exit_code=127, raw_state="FAILED"))
        row = store.get_structure(sid)
        assert row.state == "failed" and row.dft_attempt == 1

    def test_the_ladder_runs_out_and_says_why(self, cfg, store, tmp_path):
        sid = add_selected(store, 1)[0]
        self._fail(cfg, store, sid, tmp_path, attempt=99,
                   status=JobStatus(job_id="1", state=JobState.done,
                                    raw_state="COMPLETED"))
        row = store.get_structure(sid)
        assert row.state == "failed"
        assert "relax" in row.dft_fail_reason

    def test_a_success_resets_the_attempt_counter(self, cfg, store, tmp_path):
        from cspflow.stages.base import WorkItem

        sid = add_selected(store, 1)[0]
        store.set_structure_state(sid, StructureState.selected, **{ATTEMPT_KEY: 2})
        item = WorkItem(key=f"dft-{sid}-relax", structure_ids=[sid],
                        payload={"step": 0, "step_name": "relax", "attempt": 2})
        fake_job_output(tmp_path / item.key, converged=True)
        DftStage(cfg).reconcile(store, {"workdir": str(tmp_path), "id": 1},
                                JobStatus(job_id="1", state=JobState.done,
                                          raw_state="COMPLETED"), [item])
        assert store.get_structure(sid).dft_attempt == 0


@has_potcars
class TestBuild:
    def test_it_writes_a_complete_input_directory_per_task(self, cfg, store, tmp_path):
        """Inputs are assembled on the login node, so an unresolvable POTCAR
        fails in milliseconds rather than after the queue wait."""
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        spec = stage.build(items, tmp_path / "dft")
        assert spec.array_size == 2
        for item in items:
            directory = (tmp_path / "dft" / item.key)
            for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "inputs.json"):
                assert (directory / name).is_file(), f"{item.key}/{name}"

    def test_the_task_manifest_lists_every_directory(self, cfg, store, tmp_path):
        add_selected(store, 2)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=10)
        stage.build(items, tmp_path / "dft")
        manifest = json.loads(
            next((tmp_path / "dft").glob("*.tasks.json")).read_text())
        assert len(manifest["dirs"]) == 2

    def test_the_settings_hash_is_carried_on_the_item(self, cfg, store, tmp_path):
        add_selected(store, 1)
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        stage.build(items, tmp_path / "dft")
        assert len(items[0].payload["settings_hash"]) == 64

    def test_the_command_uses_the_machine_profile_binary(self, cfg, store, tmp_path):
        add_selected(store, 1)
        stage = DftStage(cfg)
        spec = stage.build(stage.claim(store, budget=1), tmp_path / "dft")
        assert "vasp_std" in spec.command
        assert "srun" in spec.command

    def test_vasp_done_is_written_as_a_process_marker_only(self, cfg, store, tmp_path):
        """It means VASP exited. 61% of the legacy campaign wrote it unconverged."""
        add_selected(store, 1)
        stage = DftStage(cfg)
        spec = stage.build(stage.claim(store, budget=1), tmp_path / "dft")
        assert "VASP_DONE" in spec.command


class TestPlumbing:
    def test_it_is_a_submitted_stage(self, cfg):
        assert DftStage(cfg).in_process is False

    def test_it_sits_last_before_analyze(self):
        from cspflow.driver import STAGE_ORDER

        assert STAGE_ORDER.index("filter") < STAGE_ORDER.index("dft")
        assert STAGE_ORDER.index("dft") < STAGE_ORDER.index("analyze")

    def test_the_recipe_is_the_campaigns(self, cfg):
        assert DftStage(cfg).recipe.stage_names == ["relax", "static"]
