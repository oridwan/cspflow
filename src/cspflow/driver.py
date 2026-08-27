"""The driver loop.

`csp run` is not a science job. It is a tiny long-lived process that wakes,
queries the database, submits whatever is ready and under the limits, and sleeps
again (pipeline.md sec.4.4). `csp gen | screen | dft` perform *exactly the same
submission* for one stage -- manual and automatic are the same code, which is
the property that stops the two drifting apart.

Four things live here and nowhere else, so that policy exists in one place:

*   **Reconciliation before submission.** Every cycle polls what is in flight
    and folds it back into the database *first*. Submitting before reconciling
    would let a driver dispatch work whose predecessor had already failed.

*   **Throttling.** Stages describe work; the driver decides how much of it goes
    out, against the live QOS limits (D045).

*   **The budget.** Phase B is a triage engine, not a throughput engine: with
    thousands of compositions the honest framing is that DFT time runs out
    before the candidate list does. The driver stops submitting when projected
    spend exceeds `budget_core_hours` and says so, rather than quietly emptying
    the allocation.

*   **Stopping.** A stop file, a cycle cap, and SIGTERM all end the loop
    cleanly, at a cycle boundary, with nothing half-submitted.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .config.loader import ResolvedConfig
from .db.store import Store
from .scheduler.base import JobSpec, JobState, JobStatus, Scheduler, compute_throttle
from .stages.base import Stage, StageReport, WorkItem

# Stage order is the funnel order.  `--through` and `--from` slice this list,
# which is what makes Phase A a barrier and Phase B a stream (pipeline.md 4.3).
STAGE_ORDER = [
    "source", "generate", "screen", "reference", "calibrate",
    "filter", "dft", "analyze",
]


class DriverError(Exception):
    pass


@dataclass
class CycleReport:
    """One pass of the loop."""

    cycle: int
    stages: list[StageReport] = field(default_factory=list)
    reconciled: int = 0
    in_flight: int = 0
    core_hours_spent: float = 0.0
    core_hours_projected: float = 0.0
    budget_exhausted: bool = False
    note: str = ""

    @property
    def did_something(self) -> bool:
        return self.reconciled > 0 or any(s.did_something for s in self.stages)

    def render(self) -> str:
        lines = [f"cycle {self.cycle}  in flight {self.in_flight}  "
                 f"spent {self.core_hours_spent:,.0f} core-h"]
        if self.core_hours_projected:
            lines[0] += f"  projected {self.core_hours_projected:,.0f}"
        for s in self.stages:
            if s.did_something or s.pending:
                lines.append("  " + s.render())
        if self.budget_exhausted:
            lines.append("  BUDGET REACHED -- submitting nothing further")
        if self.note:
            lines.append("  " + self.note)
        return "\n".join(lines)


@dataclass
class DriverOptions:
    interval: int = 300                     # seconds between cycles
    max_cycles: int | None = None           # None = until nothing is left
    stop_file: Path | None = None
    stages: Sequence[str] | None = None     # None = every registered stage
    dry_run: bool = False
    budget_core_hours: float | None = None


class Driver:
    """Reconcile, submit, sleep, repeat."""

    def __init__(
        self,
        cfg: ResolvedConfig,
        store: Store,
        scheduler: Scheduler,
        stages: Sequence[Stage],
        options: DriverOptions | None = None,
        *,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.scheduler = scheduler
        self.options = options or DriverOptions()
        self.emit = emit or (lambda _msg: None)
        self._sleep = sleep
        self._stop = False
        self._claims: dict[str, list[WorkItem]] = {}

        self.stages = self._order(stages)

    # -- setup -------------------------------------------------------------

    def _order(self, stages: Sequence[Stage]) -> list[Stage]:
        by_name = {s.name: s for s in stages}
        unknown = sorted(set(by_name) - set(STAGE_ORDER))
        if unknown:
            raise DriverError(
                f"stage(s) {unknown} are not in the funnel order {STAGE_ORDER}. "
                f"A stage the driver cannot place is a stage it cannot decide when "
                f"to run."
            )
        wanted = list(self.options.stages) if self.options.stages else STAGE_ORDER
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise DriverError(
                f"no implementation registered for stage(s) {missing}. "
                f"Registered: {sorted(by_name)}"
            )
        return [by_name[name] for name in STAGE_ORDER if name in wanted]

    # -- the loop ----------------------------------------------------------

    def run(self, *, watch: bool = False) -> list[CycleReport]:
        """Run cycles until there is nothing left, or forever if `watch`.

        SIGTERM and SIGINT set a flag rather than raising, so the loop finishes
        the cycle it is in. A driver killed mid-submission would leave jobs in
        the queue with no rows recording them, which is the one state the
        database cannot recover from by itself.
        """
        reports: list[CycleReport] = []
        with self._graceful_stop():
            n = 0
            while True:
                n += 1
                report = self.cycle(n)
                reports.append(report)
                self.emit(report.render())

                if self._should_stop(report, n, watch):
                    break
                self._sleep(self.options.interval)
        return reports

    def _should_stop(self, report: CycleReport, n: int, watch: bool) -> bool:
        if self._stop:
            self.emit("stopping: asked to")
            return True
        if self.options.stop_file and self.options.stop_file.exists():
            self.emit(f"stopping: {self.options.stop_file} exists")
            return True
        if self.options.max_cycles is not None and n >= self.options.max_cycles:
            return True
        if report.budget_exhausted and report.in_flight == 0:
            self.emit("stopping: budget reached and nothing in flight")
            return True
        if not watch and not self._work_remains(report):
            return True
        return False

    def _work_remains(self, report: CycleReport) -> bool:
        return report.in_flight > 0 or any(s.pending for s in report.stages)

    def cycle(self, n: int = 1) -> CycleReport:
        """Reconcile what is in flight, then submit what fits."""
        report = CycleReport(cycle=n)

        report.reconciled = self._reconcile()
        spent, projected, in_flight = self._accounting()
        report.core_hours_spent = spent
        report.core_hours_projected = projected
        report.in_flight = in_flight

        budget = self.options.budget_core_hours
        if budget is None:
            budget = float(self.cfg.campaign.dft.select.budget_core_hours)
        report.budget_exhausted = projected >= budget

        for stage in self.stages:
            report.stages.append(self._advance(stage, report))

        # Re-read after acting, so the report describes the end of the cycle
        # rather than its beginning. The pre-cycle numbers above are what the
        # decisions were made on; these are what actually happened.
        spent, projected, in_flight = self._accounting()
        report.core_hours_spent = spent
        report.core_hours_projected = projected
        report.in_flight = in_flight
        return report

    # -- reconciliation ----------------------------------------------------

    def _reconcile(self) -> int:
        """Poll every non-terminal job and fold the answer back into the DB."""
        live = [j for j in self.store.jobs() if j["state"] in {"queued", "running", "held"}]
        if not live:
            return 0

        # One array submission produces MANY job rows sharing one slurm_id, so
        # this is a list per id, not a row per id. Keyed as a dict it silently
        # kept only the last row and left the rest queued forever.
        by_id: dict[str, list] = {}
        for job in live:
            if job["slurm_id"]:
                by_id.setdefault(job["slurm_id"], []).append(job)
        if not by_id:
            return 0

        statuses = self.scheduler.poll(sorted(by_id))
        n = 0
        for slurm_id, status in statuses.items():
            rows = by_id.get(slurm_id)
            if not rows:
                continue
            if status.state is JobState.unknown:
                # Deliberately not touched. A job neither squeue nor sacct
                # remembers has NOT been shown to have succeeded, and writing
                # 'done' here is how an outage becomes a campaign reported
                # complete (D043).
                continue
            # Core-hours are for the submission as a whole; splitting them
            # across its rows keeps the campaign total honest whether the work
            # went out as one array or as individual jobs.
            share = status.core_hours / len(rows) if rows else 0.0
            for job in rows:
                if status.state.terminal or status.state.value != job["state"]:
                    self.store.update_job(
                        job["id"], state=status.state.value,
                        core_hours=share,
                        exit_reason=status.reason or status.raw_state,
                        remedy=status.remedy().value,
                    )
                    n += 1
            if status.state.terminal:
                self._hand_back(rows[0], status)
        return n

    def _hand_back(self, job, status: JobStatus) -> None:
        stage = next((s for s in self.stages if s.name == job["stage"]), None)
        if stage is None:
            return
        items = self._claims.pop(str(job["slurm_id"]), [])
        stage.reconcile(self.store, job, status, items)

    # -- accounting --------------------------------------------------------

    def _accounting(self) -> tuple[float, float, int]:
        """Core-hours spent, core-hours projected, and jobs in flight.

        Projection matters more than the spend: a budget check that only counts
        finished jobs approves a submission that the already-queued work will
        overspend before it ever runs.
        """
        spent = 0.0
        projected = 0.0
        in_flight = 0
        for job in self.store.jobs():
            if job["state"] in {"queued", "running", "held"}:
                in_flight += 1
                projected += self._estimate(job)
            else:
                spent += float(job["core_hours"] or 0.0)
        return spent, spent + projected, in_flight

    def _estimate(self, job) -> float:
        """What an in-flight job will probably cost.

        Uses the mean of what finished jobs of the same stage actually cost --
        this campaign's own history, not a guess. Falls back to the requested
        walltime x ntasks, which over-estimates, and over-estimating a budget is
        the safe direction.
        """
        recorded = float(job["core_hours"] or 0.0)
        if recorded:
            return recorded
        row = self.store.sql.execute(
            "SELECT AVG(core_hours) AS mean FROM job "
            "WHERE stage=? AND state='done' AND core_hours > 0",
            (job["stage"],),
        ).fetchone()
        if row and row["mean"]:
            return float(row["mean"])
        return self._walltime_estimate(job["stage"])

    def _walltime_estimate(self, stage: str) -> float:
        resources = {
            "generate": self.cfg.campaign.generate.resources if self.cfg.campaign.generate else None,
            "screen": self.cfg.campaign.screen.resources,
        }.get(stage)
        ntasks = (resources.ntasks if resources and resources.ntasks
                  else self.cfg.machine.defaults.ntasks)
        hours = _walltime_hours(resources.time if resources else "24:00:00")
        return hours * ntasks

    # -- submission --------------------------------------------------------

    def _advance(self, stage: Stage, report: CycleReport) -> StageReport:
        pending = stage.pending(self.store)
        out = StageReport(stage=stage.name, pending=pending)
        if not pending:
            return out

        if stage.in_process:
            if self.options.dry_run:
                out.note = "dry-run: not executed"
                return out
            done = stage.run(self.store)
            out.claimed = done.claimed
            out.reconciled = done.reconciled
            out.note = done.note
            # Re-read: `pending` was measured before the stage ran, and the loop
            # decides whether to sleep by asking whether work remains. Left
            # stale, an in-process stage that finished its work in this very
            # cycle still reported it as pending, and the driver slept a full
            # interval before noticing it was done.
            out.pending = stage.pending(self.store)
            return out

        if report.budget_exhausted:
            out.note = "held: budget"
            return out

        throttle = self._throttle(stage, report.in_flight)
        budget = self._budget_limited(stage, throttle.in_flight, report)
        if budget <= 0:
            if throttle.in_flight > 0:
                out.note = "held: budget"
                # Not enough left to afford even one more unit of work. That is
                # the budget being exhausted just as much as having overspent
                # it, and saying so is what lets the loop terminate rather than
                # waking every five minutes to hold again.
                report.budget_exhausted = True
            else:
                out.note = f"held: {throttle.binding}"
            return out

        if self.options.dry_run:
            # Claiming marks rows in the database, so a dry run must not do it.
            # The count is derivable without mutating anything.
            out.note = (f"dry-run: would submit {min(pending, budget)} "
                        f"({throttle.render()})")
            return out

        items = stage.claim(self.store, budget)
        out.claimed = len(items)
        out.pending = stage.pending(self.store)
        if not items:
            return out

        workdir = Path(self.cfg.campaign.workdir) / stage.name
        spec = stage.build(items, workdir)
        spec.array_throttle = spec.array_throttle or throttle.concurrent_tasks

        job_id = self.scheduler.submit(spec)
        self._claims[str(job_id)] = items
        for item in items:
            row = self.store.add_job(
                stage=stage.name,
                structure_id=item.structure_ids[0] if item.structure_ids else None,
                workdir=str(workdir),
            )
            self.store.update_job(row, state="queued", slurm_id=str(job_id))
        out.submitted = len(items)
        out.note = throttle.render()
        return out

    def _budget_limited(self, stage: Stage, want: int, report: CycleReport) -> int:
        """Cap this cycle's claim by what is left of the budget.

        Without this the budget is checked only against work already sent, so a
        single cycle can approve a thousand jobs and overspend the allocation
        before the next cycle ever runs. The per-item estimate is this
        campaign's own mean for the stage where one exists, and the requested
        walltime x ntasks otherwise -- which over-estimates, and over-estimating
        a budget errs in the safe direction.
        """
        if report.budget_exhausted:
            return 0
        budget = self.options.budget_core_hours
        if budget is None:
            budget = float(self.cfg.campaign.dft.select.budget_core_hours)
        remaining = budget - report.core_hours_projected
        if remaining <= 0:
            return 0
        per_item = self._per_item_estimate(stage.name)
        if per_item <= 0:
            return want
        return max(0, min(want, int(remaining // per_item)))

    def _per_item_estimate(self, stage: str) -> float:
        row = self.store.sql.execute(
            "SELECT AVG(core_hours) AS mean FROM job "
            "WHERE stage=? AND state='done' AND core_hours > 0",
            (stage,),
        ).fetchone()
        if row and row["mean"]:
            return float(row["mean"])
        return self._walltime_estimate(stage)

    def _throttle(self, stage: Stage, already_in_flight: int):
        dft = self.cfg.campaign.dft
        resources = {
            "generate": self.cfg.campaign.generate.resources if self.cfg.campaign.generate else None,
            "screen": self.cfg.campaign.screen.resources,
        }.get(stage.name)
        ntasks = (resources.ntasks if resources and resources.ntasks
                  else self.cfg.machine.defaults.ntasks)
        gpus = resources.gpus if resources and resources.gpus else 0
        try:
            limits = self.scheduler.limits(stage.role)
        except Exception:                               # pragma: no cover - live only
            limits = None
        return compute_throttle(
            requested_in_flight=dft.max_in_flight,
            requested_concurrent=dft.max_concurrent_tasks,
            ntasks=ntasks, gpus_per_job=gpus, limits=limits,
            already_in_flight=already_in_flight,
        )

    # -- stopping ----------------------------------------------------------

    def stop(self) -> None:
        self._stop = True

    class _GracefulStop:
        def __init__(self, driver: "Driver") -> None:
            self.driver = driver
            self.previous: dict[int, object] = {}

        def __enter__(self):
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self.previous[sig] = signal.signal(sig, self._handler)
                except ValueError:      # not the main thread; tests, mostly
                    pass
            return self

        def _handler(self, *_args) -> None:
            self.driver.stop()

        def __exit__(self, *exc):
            for sig, handler in self.previous.items():
                try:
                    signal.signal(sig, handler)          # type: ignore[arg-type]
                except ValueError:                       # pragma: no cover
                    pass
            return False

    def _graceful_stop(self) -> "Driver._GracefulStop":
        return Driver._GracefulStop(self)


def _walltime_hours(walltime: str) -> float:
    """`'2-00:00:00'` -> 48.0."""
    days, _, clock = walltime.partition("-")
    if not clock:
        days, clock = "0", days
    parts = [float(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return int(days) * 24 + parts[-3] + parts[-2] / 60 + parts[-1] / 3600
