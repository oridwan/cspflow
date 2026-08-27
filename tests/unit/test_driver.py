"""The driver loop.

Tested against a fake stage and the local scheduler, which is the reason
`LocalScheduler` exists: the driver's interesting behaviour -- resuming,
throttling, holding at a budget, stopping cleanly -- is exactly what is painful
to exercise through a real queue.
"""

from pathlib import Path

import pytest
import yaml

from cspflow.config.loader import load_campaign
from cspflow.db.store import Store
from cspflow.driver import STAGE_ORDER, Driver, DriverError, DriverOptions
from cspflow.scheduler.base import JobSpec, JobState, JobStatus, Limits
from cspflow.stages.base import StageReport, WorkItem

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: composition_list
    name: list
    composition_list: {{items: [{{formula: FeCo5}}]}}
generate:
  engine: mattergen
  mattergen: {{model: /tmp/model}}
dft:
  max_in_flight: 4
  max_concurrent_tasks: 2
  select: {{budget_core_hours: 100}}
"""


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


class FakeScheduler:
    """Records what it was asked to do; answers what it is told to answer."""

    def __init__(self, limits: Limits | None = None):
        self.submitted: list[JobSpec] = []
        self.statuses: dict[str, JobStatus] = {}
        self._limits = limits or Limits()
        self._n = 0

    def submit(self, spec: JobSpec) -> str:
        self._n += 1
        self.submitted.append(spec)
        return str(1000 + self._n)

    def poll(self, job_ids):
        return {
            j: self.statuses.get(j, JobStatus(job_id=j, state=JobState.running))
            for j in job_ids
        }

    def cancel(self, job_ids):
        pass

    def limits(self, role):
        return self._limits

    def in_flight(self):
        return 0


class FakeStage:
    """A submitted stage over a fixed pool of structure ids."""

    name = "dft"
    role = "cpu"
    in_process = False

    def __init__(self, work: int = 6):
        self.pool = list(range(1, work + 1))
        self.reconciled: list[tuple] = []

    def pending(self, store):
        return len(self.pool)

    def claim(self, store, budget):
        taken, self.pool = self.pool[:budget], self.pool[budget:]
        return [WorkItem(key=f"s{i}", structure_ids=[i], est_core_hours=10.0) for i in taken]

    def build(self, items, workdir):
        return JobSpec(name=f"dft-{items[0].key}", stage="dft", workdir=workdir,
                       command="true", array_size=len(items))

    def reconcile(self, store, job_row, status, items):
        self.reconciled.append((job_row["id"], status.state, len(items)))

    def run(self, store):                                # pragma: no cover
        raise AssertionError("submitted stage must not be run in process")


class FakeInProcessStage:
    name = "filter"
    role = "cpu"
    in_process = True

    def __init__(self, work: int = 3):
        self.remaining = work
        self.ran = 0

    def pending(self, store):
        return self.remaining

    def claim(self, store, budget):                      # pragma: no cover
        raise AssertionError("in-process stage must not be claimed against")

    def build(self, items, workdir):                     # pragma: no cover
        raise AssertionError("in-process stage must not be built")

    def reconcile(self, store, job_row, status, items):  # pragma: no cover
        pass

    def run(self, store):
        self.ran += 1
        done, self.remaining = self.remaining, 0
        return StageReport(stage=self.name, claimed=done, reconciled=done)


def driver(cfg, store, scheduler, impls, **opts):
    options = DriverOptions(interval=0, **opts)
    return Driver(cfg, store, scheduler, impls, options, sleep=lambda _s: None)


# --------------------------------------------------------------------------


class TestOrdering:
    def test_stages_run_in_funnel_order(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(),
                   [FakeStage(), FakeInProcessStage()], stages=["dft", "filter"])
        assert [s.name for s in d.stages] == ["filter", "dft"]

    def test_a_stage_outside_the_funnel_is_refused(self, cfg, store):
        class Weird(FakeStage):
            name = "teleport"

        with pytest.raises(DriverError, match="funnel order"):
            driver(cfg, store, FakeScheduler(), [Weird()], stages=["teleport"])

    def test_asking_for_an_unregistered_stage_is_refused(self, cfg, store):
        with pytest.raises(DriverError, match="no implementation"):
            driver(cfg, store, FakeScheduler(), [FakeStage()], stages=["dft", "screen"])

    def test_the_order_is_the_documented_funnel(self):
        assert STAGE_ORDER[0] == "source" and STAGE_ORDER[-1] == "analyze"
        assert STAGE_ORDER.index("screen") < STAGE_ORDER.index("dft")


class TestSubmission:
    def test_claims_and_submits_up_to_the_throttle(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        report = d.cycle()
        # max_in_flight is 4 in the fixture
        assert report.stages[0].submitted == 4
        assert len(sched.submitted) == 1
        assert len(stage.pool) == 6

    def test_job_rows_are_written_for_every_item(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=10)], stages=["dft"])
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"queued": 4}

    def test_the_array_throttle_comes_from_the_qos(self, cfg, store):
        sched = FakeScheduler(Limits(max_submit=2048, max_cpus=32))
        d = driver(cfg, store, sched, [FakeStage(work=10)], stages=["dft"])
        d.cycle()
        # cpu=32 at the machine default ntasks=16 permits 2 concurrent
        assert sched.submitted[0].array_throttle == 2

    def test_nothing_pending_means_nothing_submitted(self, cfg, store):
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=0)], stages=["dft"])
        assert d.cycle().stages[0].submitted == 0
        assert not sched.submitted

    def test_dry_run_claims_nothing_and_submits_nothing(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"], dry_run=True)
        report = d.cycle()
        assert not sched.submitted and len(stage.pool) == 10
        assert "dry-run" in report.stages[0].note

    def test_in_process_stage_runs_here_and_now(self, cfg, store):
        stage = FakeInProcessStage(work=3)
        d = driver(cfg, store, FakeScheduler(), [stage], stages=["filter"])
        report = d.cycle()
        assert stage.ran == 1 and report.stages[0].claimed == 3

    def test_in_flight_work_is_subtracted_from_the_budget(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=10)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        d.cycle()                       # submits 4, all stay 'queued'
        second = d.cycle()
        # max_in_flight 4 minus 4 already queued leaves nothing
        assert second.stages[0].submitted == 0
        assert "held" in second.stages[0].note


class TestReconciliation:
    def _submitted(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=4)
        d = driver(cfg, store, sched, [stage], stages=["dft"])
        d.cycle()
        return d, sched, stage

    def test_a_finished_job_updates_its_row(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        elapsed_seconds=3600, alloc_cpus=16,
                                        raw_state="COMPLETED")
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"done": 4}

    def test_the_stage_is_handed_the_result(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        raw_state="COMPLETED")
        d.cycle()
        assert stage.reconciled and stage.reconciled[0][1] is JobState.done

    def test_an_unknown_job_is_left_alone(self, cfg, store):
        """Neither squeue nor sacct remembers it -- that is not success."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.unknown)
        d.cycle()
        assert store.count_jobs_by_state("dft") == {"queued": 4}
        assert not stage.reconciled

    def test_a_timeout_records_its_remedy(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.timeout,
                                        raw_state="TIMEOUT")
        d.cycle()
        assert store.jobs()[0]["remedy"] == "more_walltime"

    def test_command_not_found_is_marked_do_not_retry(self, cfg, store):
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.failed,
                                        exit_code=127, raw_state="FAILED")
        d.cycle()
        assert store.jobs()[0]["remedy"] == "do_not_retry"

    def test_core_hours_are_split_across_an_arrays_rows(self, cfg, store):
        """One submission, four rows: the campaign total must still be right."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        elapsed_seconds=7200, alloc_cpus=8,
                                        raw_state="COMPLETED")
        d.cycle()
        rows = store.jobs()
        assert len(rows) == 4
        assert sum(r["core_hours"] for r in rows) == pytest.approx(16.0)

    def test_every_row_of_an_array_is_updated_not_just_one(self, cfg, store):
        """Keyed as a dict on slurm_id, three of four rows stayed queued forever."""
        d, sched, stage = self._submitted(cfg, store)
        jid = store.jobs()[0]["slurm_id"]
        sched.statuses[jid] = JobStatus(job_id=jid, state=JobState.done,
                                        raw_state="COMPLETED")
        d.cycle()
        assert {r["state"] for r in store.jobs()} == {"done"}


class TestBudget:
    def test_projection_counts_queued_work_not_only_finished(self, cfg, store):
        """A budget check that only counts finished jobs approves an overspend."""
        sched = FakeScheduler()
        d = driver(cfg, store, sched, [FakeStage(work=10)], stages=["dft"])
        first = d.cycle()
        assert first.core_hours_spent == 0.0
        assert first.core_hours_projected > 0.0

    def test_submission_is_held_once_the_budget_is_reached(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=100)
        d = driver(cfg, store, sched, [stage], stages=["dft"],
                   budget_core_hours=1.0)
        report = d.cycle()
        assert report.stages[0].note == "held: budget"
        assert not sched.submitted
        assert report.budget_exhausted        # cannot afford even one more unit

    def test_the_budget_default_comes_from_the_campaign(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1)], stages=["dft"])
        assert d.cfg.campaign.dft.select.budget_core_hours == 100


class TestLoop:
    def test_stops_when_no_work_remains(self, cfg, store):
        """One cycle: the stage runs, reports 0 pending, and the loop concludes."""
        stage = FakeInProcessStage(work=2)
        d = driver(cfg, store, FakeScheduler(), [stage], stages=["filter"])
        reports = d.run(watch=False)
        assert len(reports) == 1 and stage.ran == 1

    def test_max_cycles_caps_the_loop(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1000)],
                   stages=["dft"], max_cycles=3)
        assert len(d.run(watch=True)) == 3

    def test_a_stop_file_ends_the_loop(self, cfg, store, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("")
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=1000)],
                   stages=["dft"], max_cycles=99, stop_file=stop)
        assert len(d.run(watch=True)) == 1

    def test_stop_ends_the_loop_at_a_cycle_boundary(self, cfg, store):
        sched = FakeScheduler()
        stage = FakeStage(work=1000)
        d = driver(cfg, store, sched, [stage], stages=["dft"], max_cycles=99)

        original = stage.claim

        def claim_then_stop(store_, budget):
            d.stop()
            return original(store_, budget)

        stage.claim = claim_then_stop
        reports = d.run(watch=True)
        assert len(reports) == 1
        assert len(sched.submitted) == 1     # the cycle it was in still completed

    def test_reports_render_without_error(self, cfg, store):
        d = driver(cfg, store, FakeScheduler(), [FakeStage(work=5)], stages=["dft"])
        text = d.cycle().render()
        assert "cycle 1" in text and "dft" in text


# --------------------------------------------------------------------------
# The source stage, and the registry
# --------------------------------------------------------------------------


class TestSourceStage:
    def test_is_in_process(self, cfg, store):
        from cspflow.stages import SourceStage

        assert SourceStage(cfg).in_process is True

    def test_writes_composition_rows(self, cfg, store):
        from cspflow.stages import SourceStage

        stage = SourceStage(cfg)
        assert stage.pending(store) == 1
        report = stage.run(store)
        assert report.claimed == 1                # FeCo5 at Z=1
        assert store.chemsystems() == ["Co-Fe"]

    def test_stops_being_pending_once_it_has_run(self, cfg, store):
        """Otherwise the driver never concludes the campaign is finished."""
        from cspflow.stages import SourceStage

        stage = SourceStage(cfg)
        stage.run(store)
        assert stage.pending(store) == 0

    def test_the_driver_runs_it_end_to_end(self, cfg, store):
        from cspflow.stages import SourceStage

        d = driver(cfg, store, FakeScheduler(), [SourceStage(cfg)], stages=["source"])
        reports = d.run(watch=False)
        assert store.compositions()
        assert reports[0].stages[0].claimed == 1

    def test_the_loop_does_not_sleep_after_finishing(self, cfg, store):
        """`pending` was measured before the stage ran, so a finished stage
        still looked pending and the driver slept a full interval."""
        from cspflow.stages import SourceStage

        d = driver(cfg, store, FakeScheduler(), [SourceStage(cfg)], stages=["source"])
        report = d.cycle()
        assert report.stages[0].pending == 0

    def test_registry_lists_only_what_is_implemented(self, cfg):
        from cspflow.stages import IMPLEMENTED, PLANNED, build_registry

        names = [s.name for s in build_registry(cfg)]
        assert names == IMPLEMENTED
        assert set(names).isdisjoint(PLANNED)
        assert set(IMPLEMENTED) | set(PLANNED) == set(STAGE_ORDER)
