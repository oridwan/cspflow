"""What an array task runs.

The worker is deliberately the dumbest process in the system. It reads a
manifest, relaxes the structures named in its chunk, writes a JSON file, and
exits. It does not open the campaign database for writing, does not decide what
work to do, and does not retry.

Everything about that is on purpose (see `stages/screen_stage.py`): the driver
is the only writer, so at 48 concurrent tasks nothing contends for SQLite's
write lock; and a worker that dies leaves no results file, which reconciliation
reports as work that did not come back rather than as silent partial data.

The results file is written to a temporary name and renamed, so a task killed
mid-write cannot leave a half-parsed JSON file behind that looks like a result.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


class WorkerError(Exception):
    pass


def run_screen_task(manifest_path: str | Path, task_id: int | None = None) -> Path:
    """Relax one chunk of a screen manifest and write its results file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise WorkerError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    chunks = manifest["chunks"]
    if not 0 <= task_id < len(chunks):
        raise WorkerError(
            f"array task {task_id} has no chunk in {manifest_path} "
            f"({len(chunks)} chunk(s)). Was --array sized to match the manifest?"
        )
    ids = chunks[task_id]

    from .db.store import Store
    from .mlip import MatterSimEngine

    # Read-only use of the database. WAL mode permits any number of concurrent
    # readers, so every task in the array can do this at once.
    store = Store.open(manifest["db"])
    try:
        structures = [(sid, store.get_structure(sid).toatoms()) for sid in ids]
    finally:
        store.close()

    engine = MatterSimEngine(
        model=manifest.get("model", "MatterSim-v1.0.0-5M.pth"),
        fmax=float(manifest.get("fmax", 0.01)),
        max_steps=int(manifest.get("max_steps", 500)),
    )

    results: list[dict[str, Any]] = []
    for sid, atoms in structures:
        result = engine.relax(atoms)
        results.append({
            "structure_id": sid,
            "energy": result.energy,
            "e_per_atom": result.e_per_atom,
            "converged": result.converged,
            "n_steps": result.n_steps,
            "fmax": result.fmax,
            "volume_before": result.volume_before,
            "volume_after": result.volume_after,
            "volume_drift": result.volume_drift,
            "error": result.error,
            "engine": result.engine,
        })

    key = manifest.get("key") or manifest_path.name.split(".manifest")[0]
    out = manifest_path.parent / f"{key}.task{task_id}.json"
    payload = {
        "task_id": task_id,
        "max_steps": manifest.get("max_steps"),
        "device": engine.device,
        "results": results,
    }
    _atomic_write_json(out, payload)
    return out


def run_generate_task(manifest_path: str | Path, task_id: int | None = None) -> Path:
    """Generate structures for one chunk of compositions and write its results.

    The chunk is a group of compositions that all want the same number of
    structures, so the whole chunk is one MatterGen call (or two, when the count
    exceeds the batch cap).  The checkpoint is therefore loaded once for the
    task rather than once per composition.

    Preflight runs *before* the model is touched.  The check that matters most
    is CUDA: MatterGen's `get_device()` returns CPU when no GPU is visible and
    logs nothing, so a task that landed on the wrong partition would sample at
    roughly a thousandth of the intended rate and be killed by its walltime,
    leaving a TIMEOUT with no cause anywhere in the logs.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise WorkerError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    chunks = manifest["chunks"]
    if not 0 <= task_id < len(chunks):
        raise WorkerError(
            f"array task {task_id} has no chunk in {manifest_path} "
            f"({len(chunks)} chunk(s)). Was --array sized to match the manifest?")
    chunk = chunks[task_id]

    from .chem import parse_formula
    from .generators import GenerationRequest, MatterGenEngine

    engine = MatterGenEngine(
        model=manifest["model"], mode=manifest.get("mode", "csp"),
        max_batch_size=int(manifest.get("max_batch_size", 100)),
        timeout_per_batch=int(manifest.get("timeout_per_batch", 1800)),
    )
    problems = engine.preflight()
    if problems:
        raise WorkerError("generation preflight failed:\n  - " + "\n  - ".join(problems))

    requests = []
    for entry in chunk:
        per_fu = parse_formula(entry["formula"])
        counts = {element: count * int(entry["z"]) for element, count in per_fu.items()}
        if sum(counts.values()) != int(entry["n_atoms"]):
            raise WorkerError(
                f"{entry['formula']} Z={entry['z']} is {sum(counts.values())} atoms "
                f"but the composition row says {entry['n_atoms']}")
        requests.append(GenerationRequest(
            composition_id=int(entry["id"]), formula=entry["formula"],
            counts=counts, n_requested=int(entry["n_requested"])))

    key = manifest.get("key") or manifest_path.name.split(".manifest")[0]
    task_dir = manifest_path.parent / f"{key}.task{task_id}.out"
    outcomes = engine.generate_many(requests, task_dir)

    out = manifest_path.parent / f"{key}.task{task_id}.json"
    _atomic_write_json(out, {
        "task_id": task_id,
        "engine": engine.name,
        "model": engine.model,
        "results": [o.as_dict() for o in outcomes],
    })
    return out


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write to a temporary name, then rename.

    A task killed part-way through writing would otherwise leave a truncated
    file that parses as nothing and reads, to the driver, as a corrupt result
    rather than an absent one. `rename` within a directory is atomic.
    """
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:      # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(prog="csp-worker")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args(argv)
    out = run_screen_task(args.manifest, args.task_id)
    print(out)
    return 0


if __name__ == "__main__":                            # pragma: no cover
    sys.exit(main())
