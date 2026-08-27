"""SLURM.

Everything site-specific comes from the machine profile; nothing in this file
names a partition, an account or a module.  What it does encode is how SLURM
actually behaves, which is where the sharp edges are:

*   **`squeue` and `sacct` answer different questions.**  `squeue` knows about
    live jobs and forgets a job the moment it finishes.  `sacct` knows about
    finished jobs but only within the accounting retention window.  A job in
    neither is **unknown**, not done -- and treating unknown as done is how a
    driver loop marks a whole campaign complete after an outage.

*   **SLURM's state strings are not the enum you expect.**  Measured on this
    account: 287 of the last 10,888 jobs came back as `CANCELLED by 3883`, with
    the cancelling UID inside the state field.  A `state == "CANCELLED"`
    comparison misses every one of them.

*   **Array tasks have compound IDs** (`26493163_336`), and `sacct` will report
    both the array job and its tasks unless asked not to (`-X`).

*   **`sbatch` failing must not create a job record.**  A submission that did
    not happen, recorded as queued, is a row the driver waits on forever.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from ..config.schema import Machine
from .base import JobSpec, JobState, JobStatus, Limits, chunk_array

# `sbatch` prints exactly this on success.
_SUBMITTED = re.compile(r"Submitted batch job (\d+)")

# SLURM state -> our vocabulary.  Matching is on the FIRST WORD because several
# real states carry a trailing clause: `CANCELLED by 3883` is the common one and
# accounts for 2.6% of this account's job history.
_STATES = {
    "PENDING": JobState.queued,
    "CONFIGURING": JobState.queued,
    "RUNNING": JobState.running,
    "COMPLETING": JobState.running,
    "SUSPENDED": JobState.running,
    "COMPLETED": JobState.done,
    "FAILED": JobState.failed,
    "NODE_FAIL": JobState.failed,
    "BOOT_FAIL": JobState.failed,
    "OUT_OF_MEMORY": JobState.failed,
    "DEADLINE": JobState.timeout,
    "TIMEOUT": JobState.timeout,
    "CANCELLED": JobState.cancelled,
    "PREEMPTED": JobState.cancelled,
    "REVOKED": JobState.cancelled,
    "SPECIAL_EXIT": JobState.failed,
    "REQUEUED": JobState.queued,
    "RESIZING": JobState.running,
    "STOPPED": JobState.held,
}


class SchedulerError(Exception):
    """A scheduler command that failed, with what was run and what came back."""


def normalise_state(raw: str) -> JobState:
    """`'CANCELLED by 3883'` -> `JobState.cancelled`.

    The trailing clause is why this is a function and not a dict lookup.
    """
    if not raw:
        return JobState.unknown
    return _STATES.get(raw.strip().split()[0].upper(), JobState.unknown)


def parse_exit_code(field: str) -> tuple[int | None, int | None]:
    """`'0:53'` -> `(0, 53)`.  SLURM writes `exit:signal`, not a bare integer."""
    if not field or ":" not in field:
        try:
            return int(field), None
        except (TypeError, ValueError):
            return None, None
    exit_s, _, signal_s = field.partition(":")
    try:
        return int(exit_s), int(signal_s)
    except ValueError:
        return None, None


def parse_elapsed(field: str) -> float:
    """`'3-00:00:09'` or `'00:17:21'` -> seconds."""
    if not field:
        return 0.0
    days, _, clock = field.partition("-")
    if not clock:
        days, clock = "0", days
    parts = clock.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return 0.0
    while len(values) < 3:
        values.insert(0, 0.0)
    hours, minutes, seconds = values[-3:]
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_tres(field: str, key: str) -> int | None:
    """Pull `cpu=768` or `gres/gpu=12` out of a `MaxTRESPU` string."""
    for item in (field or "").split(","):
        name, _, value = item.partition("=")
        if name.strip() == key:
            try:
                return int(value)
            except ValueError:
                return None
    return None


class SlurmScheduler:
    """Submit, poll and cancel through `sbatch` / `squeue` / `sacct`."""

    def __init__(self, machine: Machine, *, dry_run: bool = False,
                 user: str | None = None) -> None:
        self.machine = machine
        self.dry_run = dry_run
        self.user = user or os.environ.get("USER", "")
        self._limits_cache: dict[str, Limits] = {}

    # -- submission --------------------------------------------------------

    def render_script(self, spec: JobSpec) -> str:
        """The sbatch script, in full.

        Written to disk next to the job rather than piped to `sbatch` so that a
        failed job can be rerun by hand exactly as the driver ran it.  That is
        the difference between "I can reproduce this failure" and "the driver
        did something to it".
        """
        m = self.machine
        partition = spec.partition or m.partition_for(spec.role).name
        profile = m.partition_for(spec.role) if spec.role in m.partitions else None

        lines = ["#!/bin/bash", f"#SBATCH --job-name={spec.name}",
                 f"#SBATCH --partition={partition}",
                 f"#SBATCH --nodes={m.defaults.nodes}",
                 f"#SBATCH --ntasks={spec.ntasks}",
                 f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
                 f"#SBATCH --mem={spec.mem}",
                 f"#SBATCH --time={spec.time}",
                 f"#SBATCH --chdir={spec.workdir}",
                 f"#SBATCH --output={spec.workdir}/slurm-%A_%a.out"
                 if spec.is_array else f"#SBATCH --output={spec.workdir}/slurm-%j.out"]

        if spec.gpus:
            lines.append(f"#SBATCH --gres=gpu:{spec.gpus}")
        if spec.is_array:
            throttle = f"%{spec.array_throttle}" if spec.array_throttle else ""
            lines.append(f"#SBATCH --array=0-{spec.array_size - 1}{throttle}")
        for key, value in (("account", spec.account or (profile.account if profile else None)),
                           ("qos", spec.qos or (profile.qos if profile else None)),
                           ("constraint", spec.constraint or (profile.constraint if profile else None)),
                           ("exclude", spec.exclude or (profile.exclude if profile else None))):
            if value:
                lines.append(f"#SBATCH --{key}={value}")
        for dep in spec.depends_on:
            lines.append(f"#SBATCH --dependency=afterok:{dep}")

        lines.append("")
        # `set -e` but NOT `set -u`: `conda activate` sources third-party
        # activate.d hooks, and one on this machine (julia_activate.sh) reads an
        # unset variable and aborts the whole job under `set -u`.  See D026.
        lines.append("set -eo pipefail")
        lines.append("")

        for module in (spec.modules or m.modules.get(spec.role, [])):
            lines.append(f"module load {module}")
        for key, value in {**m.env, **spec.env}.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
        env_name = spec.conda_env or m.conda.get(spec.role, "")
        if env_name:
            lines.append('eval "$(conda shell.bash hook)"')
            lines.append(f"conda activate {env_name}")

        lines.append("")
        lines.append(spec.command)
        lines.append("")
        return "\n".join(lines)

    def submit(self, spec: JobSpec) -> str:
        """Write the script, run `sbatch`, return the job id.

        A non-zero `sbatch` raises rather than returning a fake id.  A
        submission that did not happen, recorded as queued, is a row the driver
        waits on forever.
        """
        spec.workdir.mkdir(parents=True, exist_ok=True)
        script = spec.workdir / f"{spec.name}.sbatch"
        script.write_text(self.render_script(spec))
        script.chmod(0o755)

        if self.dry_run:
            return f"dry-run:{spec.name}"

        result = self._run(["sbatch", "--parsable", str(script)])
        text = result.stdout.strip()
        # `--parsable` prints `jobid[;cluster]`; without it, a sentence.
        job_id = text.split(";")[0].strip()
        if not job_id.isdigit():
            match = _SUBMITTED.search(text)
            if not match:
                raise SchedulerError(
                    f"sbatch did not return a job id for {script}.\n"
                    f"stdout: {text!r}\nstderr: {result.stderr.strip()!r}"
                )
            job_id = match.group(1)
        return job_id

    def submit_array(self, spec: JobSpec, total_tasks: int) -> list[str]:
        """Submit `total_tasks` as one array, or as several if the site caps it."""
        cap = self.limits(spec.role).max_array_size
        ids = []
        for offset, (start, end) in enumerate(chunk_array(total_tasks, cap)):
            chunk = JobSpec(**{**spec.__dict__})
            chunk.array_size = end - start + 1
            chunk.name = spec.name if offset == 0 else f"{spec.name}-{offset}"
            chunk.env = {**spec.env, "CSPFLOW_ARRAY_OFFSET": start}
            ids.append(self.submit(chunk))
        return ids

    # -- polling -----------------------------------------------------------

    def poll(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Live state from `squeue`, finished state from `sacct`, merged.

        Neither alone is sufficient: `squeue` forgets a job the moment it ends,
        `sacct` only reaches back through the accounting retention window.  A
        job neither knows about stays `unknown` -- never `done` -- because
        assuming success for a job nobody remembers is how an outage turns into
        a campaign reported complete.
        """
        if not job_ids:
            return {}
        statuses: dict[str, JobStatus] = {
            jid: JobStatus(job_id=jid, state=JobState.unknown) for jid in job_ids
        }
        statuses.update(self._squeue(job_ids))
        for jid, status in self._sacct(job_ids).items():
            if statuses.get(jid) is None or statuses[jid].state is JobState.unknown:
                statuses[jid] = status
            elif not statuses[jid].state.terminal and status.state.terminal:
                statuses[jid] = status
        return statuses

    def _squeue(self, job_ids: list[str]) -> dict[str, JobStatus]:
        result = self._run(
            ["squeue", "-h", "-o", "%i|%T|%r|%M|%C", "--jobs", ",".join(job_ids)],
            check=False,
        )
        out: dict[str, JobStatus] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 5:
                continue
            jid, raw, reason, elapsed, cpus = (p.strip() for p in parts[:5])
            out[jid] = JobStatus(
                job_id=jid, state=normalise_state(raw), reason=reason,
                elapsed_seconds=parse_elapsed(elapsed),
                alloc_cpus=int(cpus) if cpus.isdigit() else 0, raw_state=raw,
            )
        return out

    def _sacct(self, job_ids: list[str]) -> dict[str, JobStatus]:
        # -X: allocation rows only.  Without it every job also reports its
        # `.batch` and `.extern` steps, which have their own states.
        result = self._run(
            ["sacct", "-X", "-n", "-P", "-o",
             "JobID,State,ExitCode,Elapsed,AllocCPUS,Reason",
             "--jobs", ",".join(job_ids)],
            check=False,
        )
        out: dict[str, JobStatus] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            jid, raw, code, elapsed, cpus, reason = (p.strip() for p in parts[:6])
            exit_code, signal = parse_exit_code(code)
            out[jid] = JobStatus(
                job_id=jid, state=normalise_state(raw), exit_code=exit_code,
                signal=signal, reason=reason, elapsed_seconds=parse_elapsed(elapsed),
                alloc_cpus=int(cpus) if cpus.isdigit() else 0, raw_state=raw,
            )
        return out

    def cancel(self, job_ids: list[str]) -> None:
        if job_ids and not self.dry_run:
            self._run(["scancel", *job_ids], check=False)

    def in_flight(self) -> int:
        """How many of this user's jobs are queued or running right now."""
        result = self._run(["squeue", "-h", "-u", self.user, "-o", "%i"], check=False)
        return len([line for line in result.stdout.splitlines() if line.strip()])

    # -- limits ------------------------------------------------------------

    def limits(self, role: str) -> Limits:
        """Read the live QOS and cluster limits, not the machine file.

        The machine profile records them as a starting point, but they are the
        site's to change and a stale copy silently produces submissions that get
        rejected -- or worse, accepted right up to a cap that then blocks
        everything else the account is doing.
        """
        if role in self._limits_cache:
            return self._limits_cache[role]

        qos = self._qos_for(role)
        limits = Limits(max_array_size=self._max_array_size(), source="live")
        if qos:
            row = self._run(
                ["sacctmgr", "-n", "-P", "show", "qos", f"name={qos}",
                 "format=MaxSubmitJobsPU,MaxJobsPU,MaxTRESPU"],
                check=False,
            ).stdout.strip()
            if row:
                submit, jobs, tres = (row.split("|") + ["", "", ""])[:3]
                limits = Limits(
                    max_submit=int(submit) if submit.isdigit() else None,
                    max_jobs=int(jobs) if jobs.isdigit() else None,
                    max_cpus=parse_tres(tres, "cpu"),
                    max_gpus=parse_tres(tres, "gres/gpu"),
                    max_array_size=limits.max_array_size,
                    source=f"sacctmgr qos={qos}",
                )
        self._limits_cache[role] = limits
        return limits

    def _qos_for(self, role: str) -> str | None:
        """Ask SLURM which QOS the partition uses; never guess from its name.

        Guessing looks like it works: on this cluster the `Orion` partition does
        use the `orion` QOS, so a lowercase-the-name heuristic passes its first
        test. It then silently fails on `GPU`, whose QOS is `str_gpu` -- and the
        failure is invisible, because `sacctmgr` simply returns nothing for a
        QOS that does not exist and the limits come back all-`None`, which reads
        as "no limits" rather than "we did not find them".

        Measured here: Orion->orion, GPU->str_gpu, Nebula->nebula,
        Nebula_GPU->nebula_gpu, Apus and Hydrus->N/A (no partition QOS).

        Where a role names several partitions, the first is used. That is the
        conservative choice: it means a comma-list of a limited and an unlimited
        partition throttles to the limited one.
        """
        profile = self.machine.partitions.get(role)
        if profile is None:
            return None
        if profile.qos:
            return profile.qos

        first = profile.name.split(",")[0].strip()
        if not first:
            return None
        out = self._run(["scontrol", "show", "partition", first], check=False).stdout
        match = re.search(r"\bQoS=(\S+)", out)
        if not match or match.group(1) in {"N/A", "(null)"}:
            return None
        return match.group(1)

    def _max_array_size(self) -> int | None:
        out = self._run(["scontrol", "show", "config"], check=False).stdout
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", out)
        return int(match.group(1)) if match else None

    # -- process plumbing --------------------------------------------------

    def _run(self, argv: list[str], *, check: bool = True, timeout: int = 60):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise SchedulerError(
                f"{argv[0]} is not on PATH. This machine profile says "
                f"scheduler: slurm -- run `csp doctor` to check."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SchedulerError(f"{' '.join(argv)} timed out after {timeout}s") from exc
        if check and result.returncode != 0:
            raise SchedulerError(
                f"{' '.join(argv)} exited {result.returncode}\n"
                f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
            )
        return result
