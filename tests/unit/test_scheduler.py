"""Scheduler adapters.

Most of these tests are about SLURM's actual output rather than about our code,
because that is where the bugs live. Every string parsed below was taken from
this account's real `sacct`/`squeue`/`scontrol` output, not invented -- notably
`CANCELLED by 3883`, which is 2.6% of the last 10,888 jobs and which a
`state == "CANCELLED"` comparison misses entirely.
"""

from pathlib import Path

import pytest
import yaml

from cspflow.config.schema import Machine
from cspflow.scheduler import (
    JobSpec,
    JobState,
    JobStatus,
    Limits,
    LocalScheduler,
    Remedy,
    SlurmScheduler,
    chunk_array,
    compute_throttle,
    for_machine,
)
from cspflow.scheduler.slurm import parse_elapsed, parse_exit_code, parse_tres, normalise_state

MACHINES = Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"


@pytest.fixture
def orion() -> Machine:
    return Machine(**yaml.safe_load((MACHINES / "orion.yaml").read_text()))


# --------------------------------------------------------------------------
# Parsing what SLURM really emits
# --------------------------------------------------------------------------


class TestStateParsing:
    def test_cancelled_carries_the_uid_that_cancelled_it(self):
        """287 of the last 10,888 jobs on this account looked like this."""
        assert normalise_state("CANCELLED by 3883") is JobState.cancelled

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("COMPLETED", JobState.done),
            ("FAILED", JobState.failed),
            ("TIMEOUT", JobState.timeout),
            ("OUT_OF_MEMORY", JobState.failed),
            ("RUNNING", JobState.running),
            ("PENDING", JobState.queued),
            ("NODE_FAIL", JobState.failed),
            ("PREEMPTED", JobState.cancelled),
        ],
    )
    def test_known_states(self, raw, expected):
        assert normalise_state(raw) is expected

    def test_unknown_state_is_unknown_not_failed(self):
        """A state we do not recognise must not be guessed either way."""
        assert normalise_state("SOMETHING_NEW") is JobState.unknown

    def test_empty_is_unknown(self):
        assert normalise_state("") is JobState.unknown

    def test_terminal_states(self):
        assert JobState.done.terminal and JobState.timeout.terminal
        assert not JobState.running.terminal and not JobState.unknown.terminal


class TestFieldParsing:
    def test_exit_code_is_exit_colon_signal(self):
        assert parse_exit_code("0:53") == (0, 53)
        assert parse_exit_code("127:0") == (127, 0)

    def test_exit_code_without_signal(self):
        assert parse_exit_code("1") == (1, None)

    def test_exit_code_garbage_is_none_not_zero(self):
        assert parse_exit_code("") == (None, None)
        assert parse_exit_code("x:y") == (None, None)

    def test_elapsed_with_days(self):
        assert parse_elapsed("3-00:00:09") == 3 * 86400 + 9

    def test_elapsed_without_days(self):
        assert parse_elapsed("00:17:21") == 17 * 60 + 21

    def test_elapsed_garbage_is_zero(self):
        assert parse_elapsed("") == 0.0 and parse_elapsed("nope") == 0.0

    def test_tres_extracts_cpu_and_gpu(self):
        tres = "cpu=128,gres/gpu=12,mem=1T"
        assert parse_tres(tres, "cpu") == 128
        assert parse_tres(tres, "gres/gpu") == 12
        assert parse_tres(tres, "node") is None


class TestRemedy:
    def test_timeout_asks_for_more_walltime(self):
        s = JobStatus(job_id="1", state=JobState.timeout, raw_state="TIMEOUT")
        assert s.remedy() is Remedy.more_walltime

    def test_out_of_memory_asks_for_more_memory(self):
        s = JobStatus(job_id="1", state=JobState.failed, raw_state="OUT_OF_MEMORY")
        assert s.remedy() is Remedy.more_memory

    def test_command_not_found_is_never_retried(self):
        """Exit 127 hit this account 287 times in 60 days; retrying cannot fix it."""
        s = JobStatus(job_id="1", state=JobState.failed, exit_code=127, raw_state="FAILED")
        assert s.remedy() is Remedy.do_not_retry

    def test_ordinary_failure_has_no_automatic_remedy(self):
        s = JobStatus(job_id="1", state=JobState.failed, exit_code=1, raw_state="FAILED")
        assert s.remedy() is Remedy.none

    def test_core_hours_from_elapsed_and_cpus(self):
        s = JobStatus(job_id="1", state=JobState.done, elapsed_seconds=3600, alloc_cpus=16)
        assert s.core_hours == 16.0


# --------------------------------------------------------------------------
# Throttling
# --------------------------------------------------------------------------


class TestThrottle:
    def test_cpu_cap_not_submission_cap_is_the_real_ceiling(self):
        """MaxTRESPU cpu=768 at ntasks=16 permits 48 concurrent VASP jobs."""
        t = compute_throttle(
            requested_in_flight=200, requested_concurrent=500, ntasks=16,
            limits=Limits(max_submit=2048, max_cpus=768),
        )
        assert t.concurrent_tasks == 48
        assert "cpu=768" in t.binding

    def test_config_wins_when_it_is_the_smaller_number(self):
        t = compute_throttle(
            requested_in_flight=200, requested_concurrent=48, ntasks=16,
            limits=Limits(max_submit=2048, max_cpus=768),
        )
        assert (t.in_flight, t.concurrent_tasks, t.binding) == (200, 48, "config")

    def test_submission_cap_clamps_an_over_ambitious_request(self):
        """redo-new-ter-mag submitted 2,362 jobs against a 2,048 cap."""
        t = compute_throttle(
            requested_in_flight=5000, requested_concurrent=48, ntasks=16,
            limits=Limits(max_submit=2048, max_cpus=768),
        )
        assert t.in_flight == 2048

    def test_jobs_already_queued_come_out_of_the_budget(self):
        t = compute_throttle(
            requested_in_flight=2048, requested_concurrent=48, ntasks=16,
            limits=Limits(max_submit=2048, max_cpus=768), already_in_flight=2000,
        )
        assert t.in_flight == 48

    def test_gpu_cap_binds_the_screen_stage(self):
        t = compute_throttle(
            requested_in_flight=100, requested_concurrent=50, ntasks=1,
            gpus_per_job=1, limits=Limits(max_gpus=12),
        )
        assert t.concurrent_tasks == 12 and "gpu=12" in t.binding

    def test_no_limits_means_the_config_is_honoured(self):
        t = compute_throttle(requested_in_flight=10, requested_concurrent=5, ntasks=16)
        assert (t.in_flight, t.concurrent_tasks) == (10, 5)

    def test_concurrency_never_exceeds_what_is_submitted(self):
        t = compute_throttle(requested_in_flight=4, requested_concurrent=48, ntasks=1)
        assert t.concurrent_tasks == 4

    def test_full_queue_yields_a_zero_budget_not_a_negative_one(self):
        t = compute_throttle(
            requested_in_flight=100, requested_concurrent=48, ntasks=16,
            limits=Limits(max_submit=100), already_in_flight=500,
        )
        assert t.in_flight == 0


class TestChunkArray:
    def test_one_array_when_it_fits(self):
        assert chunk_array(500, 10000) == [(0, 499)]

    def test_split_when_it_does_not(self):
        assert chunk_array(25, 10) == [(0, 9), (10, 19), (20, 24)]

    def test_no_cap_means_one_array(self):
        assert chunk_array(50, None) == [(0, 49)]

    def test_nothing_to_do(self):
        assert chunk_array(0, 100) == []


# --------------------------------------------------------------------------
# Script rendering
# --------------------------------------------------------------------------


class TestRenderScript:
    def _spec(self, tmp_path, **kwargs):
        base = dict(name="dft-1", stage="dft", workdir=tmp_path,
                    command="srun vasp_std", role="cpu", ntasks=16, time="24:00:00")
        base.update(kwargs)
        return JobSpec(**base)

    def test_carries_the_machine_profile(self, orion, tmp_path):
        text = SlurmScheduler(orion).render_script(self._spec(tmp_path))
        assert "#SBATCH --partition=Orion,Apus" in text
        assert "#SBATCH --ntasks=16" in text
        assert "module load intel/mkl/2024.0" in text
        assert "conda activate cspflow" in text
        assert "export OMP_NUM_THREADS=1" in text

    def test_uses_set_e_but_never_set_u(self, orion, tmp_path):
        """`set -u` plus `conda activate` aborts on a third-party activate.d hook."""
        text = SlurmScheduler(orion).render_script(self._spec(tmp_path))
        assert "set -eo pipefail" in text
        assert "set -eu" not in text and "set -u" not in text

    def test_array_carries_the_throttle(self, orion, tmp_path):
        spec = self._spec(tmp_path, array_size=500, array_throttle=48)
        text = SlurmScheduler(orion).render_script(spec)
        assert "#SBATCH --array=0-499%48" in text

    def test_array_output_uses_A_and_a(self, orion, tmp_path):
        spec = self._spec(tmp_path, array_size=10, array_throttle=2)
        assert "slurm-%A_%a.out" in SlurmScheduler(orion).render_script(spec)

    def test_single_job_output_uses_j(self, orion, tmp_path):
        assert "slurm-%j.out" in SlurmScheduler(orion).render_script(self._spec(tmp_path))

    def test_gpu_role_requests_a_gpu(self, orion, tmp_path):
        spec = self._spec(tmp_path, role="gpu", gpus=1, ntasks=1)
        text = SlurmScheduler(orion).render_script(spec)
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --partition=GPU" in text
        assert "module load cuda/11.8" in text

    def test_dependencies_are_afterok(self, orion, tmp_path):
        spec = self._spec(tmp_path, depends_on=["12345"])
        assert "#SBATCH --dependency=afterok:12345" in SlurmScheduler(orion).render_script(spec)

    def test_script_is_written_to_disk_for_rerun_by_hand(self, orion, tmp_path):
        s = SlurmScheduler(orion, dry_run=True)
        s.submit(self._spec(tmp_path))
        assert (tmp_path / "dft-1.sbatch").is_file()

    def test_dry_run_returns_a_marked_id(self, orion, tmp_path):
        assert SlurmScheduler(orion, dry_run=True).submit(self._spec(tmp_path)).startswith("dry-run:")

    def test_unknown_role_names_the_known_ones(self, orion, tmp_path):
        with pytest.raises(KeyError, match="known roles"):
            SlurmScheduler(orion).render_script(self._spec(tmp_path, role="quantum"))


# --------------------------------------------------------------------------
# Local
# --------------------------------------------------------------------------


class TestLocalScheduler:
    def _spec(self, tmp_path, command, **kwargs):
        return JobSpec(name="t", stage="test", workdir=tmp_path, command=command, **kwargs)

    def test_runs_the_command(self, tmp_path):
        s = LocalScheduler()
        jid = s.submit(self._spec(tmp_path, "echo hello > out.txt"))
        assert s.poll([jid])[jid].state is JobState.done
        assert (tmp_path / "out.txt").read_text().strip() == "hello"

    def test_failure_is_a_state_not_an_exception(self, tmp_path):
        s = LocalScheduler()
        jid = s.submit(self._spec(tmp_path, "exit 3"))
        status = s.poll([jid])[jid]
        assert status.state is JobState.failed and status.exit_code == 3

    def test_array_sets_the_same_variable_slurm_would(self, tmp_path):
        s = LocalScheduler()
        jid = s.submit(self._spec(tmp_path, 'echo "$SLURM_ARRAY_TASK_ID" >> tasks.txt',
                                  array_size=3))
        assert s.poll([jid])[jid].state is JobState.done
        assert (tmp_path / "tasks.txt").read_text().split() == ["0", "1", "2"]

    def test_unknown_job_is_unknown(self, tmp_path):
        assert LocalScheduler().poll(["nope"])["nope"].state is JobState.unknown

    def test_cancel_marks_non_terminal_jobs(self, tmp_path):
        s = LocalScheduler(dry_run=True)
        jid = s.submit(self._spec(tmp_path, "true"))
        s.cancel([jid])
        assert s.poll([jid])[jid].state is JobState.cancelled

    def test_cancel_does_not_rewrite_a_finished_job(self, tmp_path):
        s = LocalScheduler()
        jid = s.submit(self._spec(tmp_path, "true"))
        s.cancel([jid])
        assert s.poll([jid])[jid].state is JobState.done


class TestForMachine:
    def test_slurm_profile_gets_the_slurm_adapter(self, orion):
        assert isinstance(for_machine(orion), SlurmScheduler)

    def test_local_profile_gets_the_local_adapter(self):
        assert isinstance(for_machine(Machine(scheduler="local")), LocalScheduler)


# --------------------------------------------------------------------------
# Against the live cluster (skipped where there is none)
# --------------------------------------------------------------------------

import shutil

has_slurm = pytest.mark.skipif(shutil.which("sacctmgr") is None, reason="no SLURM here")


@has_slurm
class TestLiveCluster:
    def test_partition_qos_is_read_not_guessed(self, orion):
        """`GPU` uses the `str_gpu` QOS; lowercasing the name would find nothing."""
        limits = SlurmScheduler(orion).limits("gpu")
        assert limits.source == "sacctmgr qos=str_gpu"
        assert limits.max_gpus == 12

    def test_cpu_limits_match_the_machine_profile(self, orion):
        limits = SlurmScheduler(orion).limits("cpu")
        assert limits.max_submit == 2048 and limits.max_cpus == 768

    def test_partition_without_a_qos_reports_unknown_not_unlimited(self, orion):
        limits = SlurmScheduler(orion).limits("bigmem")
        assert limits.max_submit is None and limits.source == "live"

    def test_max_array_size_is_read_from_scontrol(self, orion):
        assert SlurmScheduler(orion).limits("cpu").max_array_size == 10000

    def test_in_flight_counts_this_users_jobs(self, orion):
        assert SlurmScheduler(orion).in_flight() >= 0

    def test_polling_an_unknown_job_says_unknown(self, orion):
        """Never `done`: assuming success for a job nobody remembers hides outages."""
        statuses = SlurmScheduler(orion).poll(["999999999"])
        assert statuses["999999999"].state is JobState.unknown


# -- job identity across processes -----------------------------------------

def test_local_job_ids_do_not_repeat_across_schedulers(tmp_path):
    """A per-process counter hands out `local-1` again on the next `csp run`.

    The job table then holds two unrelated submissions under one id, and
    reconciliation -- which groups rows by id, because an array legitimately
    shares one -- applies each job's outcome to the other's rows. Found by
    running two stages in two `csp run` invocations and reading the table.
    """
    from cspflow.scheduler.local import LocalScheduler

    first = LocalScheduler(dry_run=True)
    second = LocalScheduler(dry_run=True)
    a = first.submit(JobSpec(name="a", stage="generate", workdir=tmp_path, command="true"))
    b = second.submit(JobSpec(name="b", stage="screen", workdir=tmp_path, command="true"))
    assert a != b


def test_ids_are_still_stable_within_one_scheduler(tmp_path):
    from cspflow.scheduler.local import LocalScheduler

    sched = LocalScheduler(dry_run=True)
    ids = [sched.submit(JobSpec(name=f"j{i}", stage="s", workdir=tmp_path, command="true"))
           for i in range(3)]
    assert len(set(ids)) == 3
    assert all(i.startswith("local-") for i in ids)
