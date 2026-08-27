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
from cspflow.dft.vasp.inputs import InputError
from cspflow.stages.dft_stage import (ATTEMPT_KEY, DIR_KEY, STEP_KEY,
                                      DftStage)

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


# -- the ladder has to change something ------------------------------------

class TestTheLadderActuallyApplies:
    """A retry that reruns the identical calculation is worse than no retry.

    It costs the same again, fails the same way, and looks like diligence. The
    remedy was recorded on the structure row as `dft_last_remedy` and read by
    nothing: `claim` never put it in the payload, so `_write_inputs` always saw
    an empty override dict.

    Found live: three VASP relaxations hit the ionic step limit, the ladder
    recorded `{"NSW": 200}`, and the INCAR written for the retry said `NSW = 99`.
    """

    def test_the_incar_override_reaches_the_written_incar(self, cfg, store, tmp_path):
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{ATTEMPT_KEY: 1,
                                     "dft_last_remedy": json.dumps(
                                         {"set": {"NSW": 200}, "remedy": ""})})
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        assert items[0].payload["incar_overrides"] == {"NSW": 200}

    def test_the_older_remedy_shape_still_reads(self, cfg, store):
        """It was stored as the bare override dict before the remedy was added."""
        from cspflow.stages.dft_stage import _decode_remedy

        assert _decode_remedy(json.dumps({"NSW": 200})) == {"set": {"NSW": 200},
                                                            "remedy": ""}
        assert _decode_remedy(None) == {}
        assert _decode_remedy("not json") == {}

    def test_the_remedy_name_is_carried_too(self, cfg, store):
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{"dft_last_remedy": json.dumps(
                                      {"set": {}, "remedy": "resume_from_contcar"})})
        items = DftStage(cfg).claim(store, budget=1)
        assert items[0].payload["remedy"] == "resume_from_contcar"


class TestArchivingAndResuming:
    """A retry must not erase the evidence of what it is retrying."""

    def test_a_previous_attempt_is_moved_aside(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        (d / "OUTCAR").write_text("old outcar")
        (d / "OSZICAR").write_text("old oszicar")
        (d / "INCAR").write_text("NSW = 99")

        archived = _archive_previous(d, attempt=1)
        assert archived == d / "attempt-0"
        assert (archived / "OUTCAR").read_text() == "old outcar"
        assert not (d / "OUTCAR").exists()

    def test_a_first_attempt_archives_nothing(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        assert _archive_previous(d, attempt=0) is None

    def test_archiving_twice_does_not_lose_the_first_archive(self, tmp_path):
        from cspflow.stages.dft_stage import _archive_previous

        d = tmp_path / "job"
        d.mkdir()
        (d / "OUTCAR").write_text("first")
        _archive_previous(d, attempt=1)
        (d / "OUTCAR").write_text("second")
        second = _archive_previous(d, attempt=1)
        assert (second / "OUTCAR").read_text() == "first"

    def test_resuming_reads_the_previous_contcar(self, tmp_path):
        import ase.io
        from ase.build import bulk

        from cspflow.stages.dft_stage import _read_contcar

        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        path = tmp_path / "CONTCAR"
        ase.io.write(str(path), atoms, format="vasp")
        read = _read_contcar(path)
        assert read is not None and len(read) == len(atoms)

    def test_an_empty_contcar_is_not_a_resume(self, tmp_path):
        from cspflow.stages.dft_stage import _read_contcar

        path = tmp_path / "CONTCAR"
        path.write_text("")
        assert _read_contcar(path) is None
        assert _read_contcar(tmp_path / "absent") is None

    @has_potcars
    def test_a_resume_starts_from_the_relaxed_geometry(self, cfg, store, tmp_path):
        import ase.io

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"

        # First attempt: write inputs, then pretend VASP ran and moved the cell.
        first = stage.claim(store, budget=1)
        stage.build(first, workdir)
        directory = workdir / first[0].key
        (directory / "OUTCAR").write_text("pretend")
        moved = ase.io.read(str(directory / "POSCAR"), format="vasp")
        moved.set_cell(moved.get_cell() * 1.05, scale_atoms=True)
        ase.io.write(str(directory / "CONTCAR"), moved, format="vasp")

        # Second attempt, with the resume remedy recorded.
        store.set_structure_state(
            sid, StructureState.selected,
            **{ATTEMPT_KEY: 1, "dft_last_remedy": json.dumps(
                {"set": {"NSW": 200}, "remedy": "resume_from_contcar"})})
        second = stage.claim(store, budget=1)
        stage.build(second, workdir)

        assert (directory / "attempt-0" / "OUTCAR").is_file()
        assert second[0].payload["started_from"].endswith("attempt-0/CONTCAR")
        written = ase.io.read(str(directory / "POSCAR"), format="vasp")
        assert written.get_volume() == pytest.approx(moved.get_volume(), rel=1e-6)
        assert "NSW = 200" in (directory / "INCAR").read_text()


def test_an_already_archived_directory_still_offers_its_previous_attempt(tmp_path):
    """Returning None for a directory whose outputs a previous cycle already
    archived is how a resume quietly turns back into a restart."""
    from cspflow.stages.dft_stage import _archive_previous, _latest_archive

    d = tmp_path / "job"
    (d / "attempt-0").mkdir(parents=True)
    (d / "attempt-0" / "CONTCAR").write_text("relaxed")
    assert _archive_previous(d, attempt=1) == d / "attempt-0"
    assert _latest_archive(d) == d / "attempt-0"


def test_the_latest_archive_wins(tmp_path):
    from cspflow.stages.dft_stage import _latest_archive

    d = tmp_path / "job"
    for n in (0, 1, 2):
        (d / f"attempt-{n}").mkdir(parents=True)
    (d / "attempt-notanumber").mkdir()
    assert _latest_archive(d) == d / "attempt-2"


def test_no_archive_at_all_is_none(tmp_path):
    from cspflow.stages.dft_stage import _latest_archive

    d = tmp_path / "job"
    d.mkdir()
    assert _latest_archive(d) is None


class TestOptionsThatDidNothing:
    """Four documented options were in the schema and the shipped template and
    read by no code at all. Same family as B29: a value recorded and never
    applied is worse than an absent one, because the config file says it works.
    """

    def test_a_static_step_is_not_judged_by_the_force_criterion(self, tmp_path):
        """VASP never prints "reached required accuracy" for IBRION=-1, NSW=0,
        so a static run was always "not converged" and the ladder retried it.

        Found live: a static step reported 200 ionic steps, having inherited
        NSW=200 from the relax step's remedy and been retried once.
        """
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "static"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = -1\nNSW = 0\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n"
            "                         Elapsed time (sec):     10.0\n")
        (d / "OSZICAR").write_text("   1 F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n")
        outcome = read_job_directory(d)
        assert outcome.converged
        assert outcome.state == "done"
        assert outcome.exit_reason == ""

    def test_a_relax_step_still_needs_the_force_criterion(self, tmp_path):
        from cspflow.dft.vasp.parse import read_job_directory

        d = tmp_path / "relax"
        d.mkdir()
        (d / "INCAR").write_text("IBRION = 1\nNSW = 99\n")
        (d / "OUTCAR").write_text(
            "   NIONS =      2\n"
            " General timing and accounting informations for this job:\n")
        (d / "OSZICAR").write_text(
            "".join(f"  {i} F= -.20E+02 E0= -.20E+02  d E =0.0  mag=  1.0\n"
                    for i in range(1, 100)))
        outcome = read_job_directory(d)
        assert not outcome.converged
        assert outcome.exit_reason == "ionic_step_limit"

    def test_advancing_a_step_clears_the_previous_remedy(self, cfg, store, tmp_path):
        """A remedy belongs to the step that failed. Carried forward it rewrites
        the next step's INCAR -- live, a relax retry's NSW=200 landed in the
        static INCAR and a fixed-position calculation ran 200 ionic steps."""
        [sid] = add_selected(store, 1)
        store.set_structure_state(
            sid, StructureState.selected,
            **{"dft_last_remedy": json.dumps({"set": {"NSW": 200}, "remedy": ""})})
        stage = DftStage(cfg)
        outcome = type("O", (), {"energy": -1.0, "e_per_atom": -0.5,
                                 "magnetisation": None})()
        stage._advance(store, sid, step=0, outcome=outcome, directory=tmp_path)
        assert store.get_structure(sid).key_value_pairs["dft_last_remedy"] == ""

    def test_rank_by_orders_the_fresh_candidates(self, cfg, store, tmp_path):
        ids = add_selected(store, 3)
        for sid, hull in zip(ids, (0.30, 0.05, 0.20)):
            store.update_structure(sid, e_above_hull_mlip=hull)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[1], ids[2], ids[0]]

    def test_a_candidate_with_no_ranking_value_sorts_last(self, cfg, store):
        ids = add_selected(store, 2)
        store.update_structure(ids[1], e_above_hull_mlip=0.9)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[1], ids[0]]

    def test_max_total_caps_the_campaign(self, cfg, store):
        add_selected(store, 5)
        cfg.campaign.dft.select.max_total = 2
        assert len(DftStage(cfg)._ready(store)) == 2

    def test_a_structure_already_started_keeps_its_place(self, cfg, store):
        """Abandoning a half-finished relaxation to start a better-ranked one
        from scratch spends more and finishes less."""
        ids = add_selected(store, 3)
        store.update_structure(ids[2], dft_step=1, e_above_hull_mlip=0.9)
        for sid, hull in zip(ids[:2], (0.01, 0.02)):
            store.update_structure(sid, e_above_hull_mlip=hull)
        cfg.campaign.dft.select.rank_by = "e_above_hull_mlip"
        cfg.campaign.dft.select.max_total = 1
        ready = DftStage(cfg)._ready(store)
        assert [int(r.id) for r in ready] == [ids[2]]


class TestTheNextStepStartsFromTheRelaxedGeometry:
    """`static` exists to give a high-accuracy energy AT THE RELAXED GEOMETRY,
    and its energy is what goes onto the DFT hull.

    Started from the structure in the database it runs on the generated cell
    instead and reports a number that looks entirely plausible and is wrong by
    whatever the relaxation was worth. Measured live before the fix: static
    POSCARs at 176.15, 172.92 and 260.70 A^3 against relax CONTCARs at 179.03,
    179.71 and 260.84.
    """

    @has_potcars
    def test_the_static_step_uses_the_relax_contcar(self, cfg, store, tmp_path):
        import ase.io

        [sid] = add_selected(store, 1)
        stage = DftStage(cfg)
        workdir = tmp_path / "dft"

        relax_items = stage.claim(store, budget=1)
        stage.build(relax_items, workdir)
        relax_dir = workdir / relax_items[0].key
        relaxed = ase.io.read(str(relax_dir / "POSCAR"), format="vasp")
        relaxed.set_cell(relaxed.get_cell() * 1.04, scale_atoms=True)
        ase.io.write(str(relax_dir / "CONTCAR"), relaxed, format="vasp")

        outcome = type("O", (), {"energy": -1.0, "e_per_atom": -0.5,
                                 "magnetisation": None})()
        stage._advance(store, sid, step=0, outcome=outcome, directory=relax_dir)

        static_items = stage.claim(store, budget=1)
        assert static_items[0].payload["step_name"] == "static"
        stage.build(static_items, workdir)
        written = ase.io.read(str(workdir / static_items[0].key / "POSCAR"),
                              format="vasp")
        assert written.get_volume() == pytest.approx(relaxed.get_volume(), rel=1e-6)
        assert static_items[0].payload["started_from"].endswith("CONTCAR")

    @has_potcars
    def test_a_missing_contcar_refuses_rather_than_using_the_generated_cell(
            self, cfg, store, tmp_path):
        """Silently falling back is exactly the failure this guards."""
        [sid] = add_selected(store, 1)
        store.set_structure_state(sid, StructureState.selected,
                                  **{STEP_KEY: 1, DIR_KEY: str(tmp_path / "gone")})
        stage = DftStage(cfg)
        items = stage.claim(store, budget=1)
        with pytest.raises(InputError, match="unrelaxed geometry"):
            stage.build(items, tmp_path / "dft")

    def test_the_first_step_needs_no_previous_geometry(self, cfg, store, tmp_path):
        [sid] = add_selected(store, 1)
        items = DftStage(cfg).claim(store, budget=1)
        assert items[0].payload["step"] == 0
        assert "started_from" not in items[0].payload


def test_the_resource_hint_comes_from_the_recipe(cfg):
    """DFT resources live per recipe step, not in the campaign's `dft:` block,
    so nothing outside this stage can find them."""
    ntasks, walltime = DftStage(cfg).resource_hint()
    assert ntasks > 0
    assert walltime.count(":") == 2


@pytest.mark.parametrize("text,hours", [
    ("02:00:00", 2.0), ("12:30:00", 12.5), ("1-00:00:00", 24.0),
    ("2-06:00:00", 54.0), ("00:45:00", 0.75),
])
def test_walltime_parsing(text, hours):
    from cspflow.stages.dft_stage import _hours

    assert _hours(text) == pytest.approx(hours)
