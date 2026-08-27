"""MatterGen, driven as a subprocess.

MatterGen is invoked through its `mattergen-generate` console script rather than
imported, which is the same choice the legacy campaign made and for the same
reason: it builds a Hydra config, a PyTorch Lightning module and a CUDA context
at import time, none of which unwind cleanly inside a long-lived process that
also has to talk to SQLite.  A subprocess that exits takes its GPU memory with
it.

What is different here is what happens around the call.  Three things the legacy
path left implicit are made explicit, because each of them fails silently:

*   **The device.**  MatterGen's `get_device()` returns CPU when CUDA is absent
    and says nothing.  A generation job that lands without a GPU therefore runs
    -- just some three orders of magnitude slower -- until the walltime kills
    it, and the queue records a TIMEOUT with no cause.  `preflight` refuses.
*   **The exit code.**  The legacy wrapper decided success by looking for output
    files and never inspected `returncode`; a crash that had already written one
    supercell's structures counted as a complete success.  Both are checked.
*   **The count.**  Requested and produced are recorded separately, so a
    shortfall is a number in the database rather than an absence nobody reads.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from .base import (GenerationOutcome, GenerationRequest, GeneratorError,
                   check_composition, read_generated, split_batches)

# How long to allow for model load and CUDA init, on top of the per-batch
# sampling budget. Loading a checkpoint is tens of seconds and is paid once per
# subprocess, so a short request must not be timed out by its own startup.
STARTUP_SECONDS = 600


class MatterGenEngine:
    """One configured MatterGen checkpoint."""

    name = "mattergen"

    def __init__(self, model: str, *, mode: str = "csp", max_batch_size: int = 100,
                 timeout_per_batch: int = 1800, record_trajectories: bool = False,
                 allow_cpu: bool | None = None) -> None:
        self.model = str(model)
        # An escape hatch for smoke tests and for a stand-in binary, never for a
        # campaign: sampling on CPU is ~1000x slower, so a real run that took it
        # would be killed by its walltime having produced nothing. It is opt-in,
        # it is recorded in the results file, and `device` on the outcome says
        # which one was actually used.
        self.allow_cpu = (
            os.environ.get("CSPFLOW_ALLOW_CPU_GENERATION", "") not in ("", "0", "false")
            if allow_cpu is None else bool(allow_cpu))
        self.mode = mode
        self.max_batch_size = int(max_batch_size)
        self.timeout_per_batch = int(timeout_per_batch)
        self.record_trajectories = bool(record_trajectories)

    # -- is this thing going to work ---------------------------------------

    @property
    def is_checkpoint(self) -> bool:
        """A path on disk, versus a name to pull from the HuggingFace hub.

        The legacy test was `'/' in model_path or Path(model_path).exists()`,
        which calls a bare relative directory name a pretrained model and sends
        it to the hub.  Existence on disk is the only thing that actually
        decides which branch works, so it is the only thing asked.
        """
        return Path(self.model).expanduser().is_dir()

    def preflight(self) -> list[str]:
        """Everything that must be true before this is worth queue time."""
        problems: list[str] = []

        if shutil.which("mattergen-generate") is None:
            problems.append(
                "mattergen-generate is not on PATH -- the generate stage runs it as "
                "a subprocess, so the job's environment must be the one MatterGen "
                "is installed in, not merely one that can import it")

        if self.is_checkpoint:
            root = Path(self.model).expanduser()
            if not (root / "checkpoints").is_dir():
                problems.append(
                    f"{root} has no checkpoints/ subdirectory -- MatterGen expects the "
                    f"run directory (which holds config.yaml and checkpoints/), not the "
                    f".ckpt file itself")
            if not (root / "config.yaml").is_file():
                problems.append(f"{root} has no config.yaml")
        elif "/" in self.model:
            problems.append(
                f"model {self.model!r} is not a directory on disk but looks like a path. "
                f"A pretrained name has no slash; a checkpoint must exist.")

        if not self.allow_cpu:
            problems.extend(self._gpu_problems())

        if self.mode == "csp" and self.is_checkpoint:
            problems.extend(self._csp_problems())
        return problems

    @staticmethod
    def device() -> str:
        """What MatterGen's `get_device()` would pick, resolved the same way."""
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            return "unknown"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"                                     # pragma: no cover
        return "cpu"

    @staticmethod
    def _gpu_problems() -> list[str]:
        try:
            import torch
        except ImportError:
            return ["torch is not importable in this environment"]
        if not torch.cuda.is_available():
            return ["no CUDA device visible. MatterGen's get_device() falls back to "
                    "CPU silently, so this would run to the walltime instead of "
                    "failing. Check --gres=gpu and the CUDA module."]
        return []

    def _csp_problems(self) -> list[str]:
        """CSP mode needs a model trained for it.  Hydra records which.

        The tell is not in `config.yaml` -- a CSP model's `property_embeddings`
        is `{}`, exactly like an unconditional one's, because composition
        conditioning enters through the *sampling* config, not the model.  What
        distinguishes them is the training config Hydra stamps into
        `.hydra/hydra.yaml` as `config_name`, which reads `csp` for the
        campaign's checkpoint.

        Getting this wrong is not loud: an unconditional model accepts
        `--target_compositions`, ignores it, and returns well-formed structures
        of whatever chemistry it likes.  Nothing downstream would object -- the
        composition check in `check_composition` would reject every one of them,
        after the GPU time was spent.
        """
        root = Path(self.model).expanduser()
        hydra = root / ".hydra" / "hydra.yaml"
        if not hydra.is_file():
            return [f"{root} has no .hydra/hydra.yaml, so whether this checkpoint was "
                    f"trained for CSP cannot be confirmed. Generation will run; if the "
                    f"model is unconditional every structure will fail the composition "
                    f"check after the GPU time is spent."]
        name = _hydra_config_name(hydra.read_text(errors="replace"))
        if name is None:
            return [f"{hydra} records no config_name; cannot confirm CSP training"]
        if name != "csp":
            return [f"{hydra} says this checkpoint was trained with config_name={name!r}, "
                    f"not 'csp'. --target_compositions is silently ignored by a model "
                    f"that was not trained for composition conditioning."]
        return []

    # -- the call ----------------------------------------------------------

    def build_command(self, counts: list[dict[str, int]], out: Path,
                      batch_size: int, num_batches: int) -> list[str]:
        """The exact argv.  Kept separate so a test can read it without a GPU.

        `--target_compositions` is JSON with no spaces: MatterGen's CLI is
        `fire`, which splits on whitespace before it ever sees the value, so a
        pretty-printed dictionary arrives as several unparseable arguments.
        """
        cmd = ["mattergen-generate", str(out)]
        if self.is_checkpoint:
            cmd.append(f"--model_path={Path(self.model).expanduser().resolve()}")
        else:
            cmd.append(f"--pretrained_name={self.model}")
        if self.mode == "csp":
            cmd.append("--sampling_config_name=csp")
            payload = json.dumps([dict(c) for c in counts], separators=(",", ":"))
            cmd.append(f"--target_compositions={payload}")
        cmd.append(f"--batch_size={batch_size}")
        cmd.append(f"--num_batches={num_batches}")
        cmd.append(f"--record_trajectories={self.record_trajectories}")
        return cmd

    def plan(self, requests: list[GenerationRequest]) -> list[tuple[int, int]]:
        """`[(batch_size, num_batches), ...]` covering every request in one group.

        MatterGen splits a multi-composition request by integer division::

            per_composition = num_batches * batch_size // len(target_compositions)

        which drops the remainder in silence, and returns **zero structures with
        exit code 0** whenever the product is smaller than the number of
        compositions.  Rather than work around that after the fact, the plan is
        built so the division is always exact.

        For `k` compositions each wanting `n`, `split_batches(n, cap)` gives
        parts summing to `n`; a part of size `m` occurring `c` times becomes one
        call with `batch_size=m, num_batches=k*c`, so each composition receives
        exactly `c*m` and every batch holds one composition's worth.  Sizes
        differ by at most one, so this is at most two calls regardless of `n` --
        and the model is loaded once per call, not once per composition.
        """
        if not requests:
            return []
        targets = {r.n_requested for r in requests}
        if len(targets) != 1:
            raise GeneratorError(
                f"one call must ask the same count for every composition; got "
                f"{sorted(targets)}. Group requests by n_requested first.")
        k = len(requests)
        n = requests[0].n_requested
        plan = []
        for size, count in sorted(Counter(split_batches(n, self.max_batch_size)).items(),
                                  reverse=True):
            num_batches = k * count
            assert (num_batches * size) % k == 0          # exact by construction
            plan.append((size, num_batches))
        return plan

    def generate(self, request: GenerationRequest, workdir: Path) -> GenerationOutcome:
        """One composition.  Convenience wrapper over `generate_many`."""
        return self.generate_many([request], workdir)[0]

    def generate_many(self, requests: list[GenerationRequest],
                      workdir: Path) -> list[GenerationOutcome]:
        """Run a group of equal-sized requests and read back what came out.

        Structures are assigned to requests **by their composition**, never by
        their position in the output.  MatterGen's condition loader is built in
        composition-major order and its batches line up with that, so position
        would work -- but it would work silently until the day it did not, and
        an off-by-one there attributes one compound's structures to another.
        Every structure is checked against a composition anyway (see
        `check_composition`), so keying on it is free.
        """
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        by_key = {_key(r.counts): r for r in requests}
        if len(by_key) != len(requests):
            raise GeneratorError("duplicate composition in one generation group")

        outcomes = {
            r.composition_id: GenerationOutcome(
                composition_id=r.composition_id, formula=r.formula,
                n_requested=r.n_requested, output=str(workdir))
            for r in requests
        }
        collected: dict[int, list] = {r.composition_id: [] for r in requests}
        rejected: Counter = Counter()
        started = time.monotonic()
        error = ""
        batches: list[int] = []

        for index, (batch_size, num_batches) in enumerate(self.plan(requests)):
            batches.extend([batch_size] * (num_batches // len(requests)))
            group_dir = workdir / f"call{index}_b{batch_size}x{num_batches}"
            group_dir.mkdir(parents=True, exist_ok=True)
            cmd = self.build_command([r.counts for r in requests], group_dir,
                                     batch_size, num_batches)
            (group_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            timeout = STARTUP_SECONDS + self.timeout_per_batch * num_batches

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout, check=False, env=self._env())
            except subprocess.TimeoutExpired:
                error = (f"timed out after {timeout}s on {num_batches} batch(es) "
                         f"of {batch_size}")
                break
            except OSError as exc:
                error = f"could not run mattergen-generate: {exc}"
                break

            (group_dir / "stdout.log").write_text(proc.stdout or "")
            (group_dir / "stderr.log").write_text(proc.stderr or "")
            if proc.returncode != 0:
                error = (f"mattergen-generate exited {proc.returncode}: "
                         f"{_last_line(proc.stderr)}")
                break

            try:
                produced = read_generated(group_dir)
            except GeneratorError as exc:
                error = str(exc)
                break

            for atoms in produced:
                key = _key(_counts_of(atoms))
                target = by_key.get(key)
                if target is None:
                    rejected[f"unrequested composition {key}"] += 1
                    continue
                collected[target.composition_id].append(atoms)

        elapsed = time.monotonic() - started
        for request in requests:
            outcome = outcomes[request.composition_id]
            structures = collected[request.composition_id]
            outcome.n_produced = len(structures)
            outcome.seconds = elapsed / len(requests)
            outcome.batches = list(batches)
            outcome.device = self.device()
            outcome.rejected = dict(rejected)
            if error:
                outcome.error = error
            elif not structures:
                outcome.error = ("the call succeeded but produced nothing for this "
                                 "composition -- check the conditioning")
            else:
                outcome.output = str(self._write_combined(
                    workdir / _slug(request.formula), structures))
        return [outcomes[r.composition_id] for r in requests]

    @staticmethod
    def _write_combined(directory: Path, structures: list) -> Path:
        """One extxyz per request, written atomically.

        The driver reads this file; a half-written one would be indistinguishable
        from a short run, so it is renamed into place rather than streamed.
        """
        import ase.io

        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "generated_crystals.extxyz"
        tmp = directory / "generated_crystals.extxyz.partial"
        ase.io.write(str(tmp), structures, format="extxyz")
        tmp.replace(final)
        return final

    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        # Fragmentation is the usual cause of an OOM part-way through a long
        # sampling run; the legacy scripts set this in the batch file, which
        # meant it was absent whenever anything else invoked the script.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env


def _hydra_config_name(text: str) -> str | None:
    """The `config_name:` Hydra recorded, read without importing a YAML parser.

    `hydra.yaml` is large and nested; the key appears once, under `job.config`.
    A line scan is enough and keeps this usable from `csp doctor`, which must
    work before anything heavy is importable.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("config_name:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            return value or None
    return None


def _key(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """A composition as a hashable, order-independent key."""
    return tuple(sorted((str(k), int(v)) for k, v in counts.items()))


def _counts_of(atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _slug(formula: str) -> str:
    """A directory name per composition.  Formulas are already `[A-Za-z0-9]+`."""
    return "".join(ch for ch in formula if ch.isalnum()) or "unnamed"


def _last_line(text: str | None) -> str:
    if not text:
        return "(no stderr)"
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:300] if lines else "(no stderr)"


def estimate_gpu_minutes(total_structures: int, per_structure_seconds: float = 1.5) -> float:
    """Rough queue-time estimate.  Deliberately crude and openly so.

    There is no measurement of MatterGen's sampling rate on this cluster in any
    log under /projects/mmi/shuo -- the generation jobs ran with a seven-day
    walltime and no per-composition timing -- so this is a placeholder the
    driver uses only for ordering, never for a budget decision.
    """
    return math.ceil(total_structures * per_structure_seconds) / 60.0
