"""Import an existing campaign directory into a cspflow database.

This exists for two reasons. It is the M0 acceptance test -- a legacy campaign
whose counts are already known is the cheapest possible check that the state
model can represent real work. And it is how the results already on disk stop
being unreachable: once ingested, `csp status` answers questions about them that
previously took a directory walk.

Layout understood (as produced by the scripts in /projects/mmi/shuo):

    <root>/VASP_JOBS/<Formula>/<Formula>_s<NNN>/<Step>/
        INCAR KPOINTS POSCAR POTCAR CONTCAR OSZICAR OUTCAR vasprun.xml
        VASP_DONE            <- written when VASP exits, NOT when it converges
        vasp_<slurmid>.out
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import chem
from .db.store import Origin, Store, StructureState
from .dft.vasp.parse import JobOutcome, read_job_directory

_SNAPSHOT = re.compile(r"^(?P<formula>.+?)_s(?P<index>\d+)$")


class IngestError(Exception):
    pass


@dataclass
class IngestStats:
    formulas: int = 0
    structures: int = 0
    jobs_by_state: dict[str, int] = field(default_factory=dict)
    converged: int = 0
    hit_step_limit: int = 0
    no_outcar: int = 0
    skipped: list[str] = field(default_factory=list)
    core_hours: float = 0.0

    def note_job(self, outcome: JobOutcome) -> None:
        self.jobs_by_state[outcome.state] = self.jobs_by_state.get(outcome.state, 0) + 1
        self.core_hours += outcome.core_hours
        if outcome.converged:
            self.converged += 1
        elif outcome.exit_reason == "ionic_step_limit":
            self.hit_step_limit += 1
        if outcome.exit_reason == "no OUTCAR":
            self.no_outcar += 1

    @property
    def jobs(self) -> int:
        return sum(self.jobs_by_state.values())

    def render(self) -> str:
        lines = [
            f"formulas            {self.formulas}",
            f"structures          {self.structures}",
            f"jobs                {self.jobs}",
        ]
        for state, n in sorted(self.jobs_by_state.items()):
            lines.append(f"    {state:<15} {n}")
        lines += [
            f"converged           {self.converged}",
            f"hit ionic limit     {self.hit_step_limit}"
            + ("   <- finished cleanly but NOT relaxed" if self.hit_step_limit else ""),
            f"core-hours          {self.core_hours:,.0f}",
        ]
        if self.skipped:
            lines.append(f"skipped             {len(self.skipped)}")
            for s in self.skipped[:5]:
                lines.append(f"    {s}")
            if len(self.skipped) > 5:
                lines.append(f"    ... and {len(self.skipped) - 5} more")
        return "\n".join(lines)


def parse_formula(formula: str) -> tuple[str, dict[str, int]]:
    """`Gd1Co10Cr2` -> ('Co-Cr-Gd', {'Gd':1,'Co':10,'Cr':2}).

    A thin wrapper over `cspflow.chem` that converts a parse failure into
    `IngestError`.  Ingest walks a directory tree nobody curated, so a name it
    cannot understand must become a skipped-with-reason rather than an exception
    that ends the walk -- the opposite of Stage 0's composition list, where an
    unparseable formula is a mistake the user wants raised immediately.
    """
    try:
        counts = chem.parse_formula(formula)
    except chem.ChemError as exc:
        raise IngestError(str(exc)) from exc
    return chem.chemsys(counts), counts


def find_job_dirs(structure_dir: Path) -> list[Path]:
    """Recipe steps under one structure, e.g. `Relax`, `Static`."""
    return sorted(p for p in structure_dir.iterdir() if p.is_dir())


def iter_formula_dirs(root: Path, limit: int | None = None) -> Iterator[Path]:
    vasp_jobs = root / "VASP_JOBS" if (root / "VASP_JOBS").is_dir() else root
    if not vasp_jobs.is_dir():
        raise IngestError(f"{root} contains no VASP_JOBS directory")
    seen = 0
    for entry in sorted(vasp_jobs.iterdir()):
        if not entry.is_dir():
            continue
        yield entry
        seen += 1
        if limit is not None and seen >= limit:
            return


def ingest_campaign(
    root: str | Path,
    store: Store,
    *,
    limit: int | None = None,
    source_name: str = "ingested",
    progress: Callable[[str], None] | None = None,
    read_structures: bool = True,
) -> IngestStats:
    """Walk a legacy campaign directory and record it.

    `limit` caps the number of formula directories, which is what makes this
    usable as a fast test: a handful of formulas exercises every code path that
    all of them would.
    """
    root = Path(root)
    stats = IngestStats()

    for formula_dir in iter_formula_dirs(root, limit):
        formula = formula_dir.name
        try:
            chemsys, counts = parse_formula(formula)
        except IngestError as exc:
            stats.skipped.append(f"{formula}: {exc}")
            continue
        reduced, _ = chem.reduce_counts(counts)
        canonical = chem.canonical_formula(reduced)
        # Z counts copies of the REDUCED formula, matching Stage 0, so a legacy
        # `Fe2Co10` directory and a generated `Co5Fe1` land on the same row.
        formula_atoms = sum(reduced.values())
        stats.formulas += 1
        if progress:
            progress(formula)

        structure_dirs = [p for p in sorted(formula_dir.iterdir()) if p.is_dir()]
        for structure_dir in structure_dirs:
            job_dirs = find_job_dirs(structure_dir)
            if not job_dirs:
                stats.skipped.append(f"{structure_dir.name}: no recipe-step directory")
                continue

            outcomes = [(d.name, read_job_directory(d)) for d in job_dirs]
            last_step, last = outcomes[-1]

            n_atoms = last.n_atoms or formula_atoms
            z = max(1, round(n_atoms / formula_atoms)) if formula_atoms else 1
            # The directory name is provenance, not an identity: `Gd1Co10Cr2`
            # and `Co10Cr2Gd1` are one composition, and `composition` is unique
            # on (formula, z, source_name).  Storing the name as written would
            # split one material across two rows and make every per-composition
            # count downstream wrong, so the canonical form is the key and the
            # name as written is kept on the structure row as `formula_dir`.
            comp_id = store.add_composition(
                formula=canonical, chemsys=chemsys, z=z, n_atoms=n_atoms,
                n_target=len(structure_dirs), source_mode="ingest",
                source_name=source_name, state="generated",
            )

            atoms = _read_structure(structure_dir, job_dirs) if read_structures else None
            if atoms is None and read_structures:
                # No POSCAR or CONTCAR anywhere under this structure. We cannot
                # create an ASE row without geometry, but the jobs still
                # happened and their core-hours are still spent, so they are
                # recorded unattached rather than dropped.
                stats.skipped.append(f"{structure_dir.name}: no readable geometry")
                for step_name, outcome in outcomes:
                    stats.note_job(outcome)
                    job_id = store.add_job(stage="dft", structure_id=None,
                                           recipe_step=step_name, workdir=str(outcome.path))
                    store.update_job(job_id, state=outcome.state, slurm_id=outcome.slurm_id,
                                     core_hours=outcome.core_hours,
                                     exit_reason=outcome.exit_reason, attempt=1)
                continue

            state = (
                StructureState.dft_done if last.converged
                else StructureState.failed if last.state != "done"
                else StructureState.dft_done
            )
            kv = {
                "composition_id": comp_id,
                "formula_dir": formula,
                "source_path": str(structure_dir),
                # The directory holding the outputs `analyze` will read. Recorded
                # under the same key the DFT stage uses, so an ingested campaign
                # and a native one are the same thing to every later stage --
                # which is what makes replaying an old campaign a real test.
                "dft_dir": str(job_dirs[-1].resolve()),
                "converged": last.converged,
            }
            if last.energy is not None:
                kv["vasp_energy"] = last.energy
            if last.e_per_atom is not None:
                kv["e_per_atom"] = last.e_per_atom
            if last.magnetisation is not None:
                kv["magnetisation"] = last.magnetisation

            sid = store.add_structure(atoms, origin=Origin.generated, state=state, **kv)
            stats.structures += 1

            for step_name, outcome in outcomes:
                stats.note_job(outcome)
                job_id = store.add_job(
                    stage="dft", structure_id=sid, recipe_step=step_name,
                    workdir=str(outcome.path),
                )
                store.update_job(
                    job_id, state=outcome.state, slurm_id=outcome.slurm_id,
                    core_hours=outcome.core_hours, exit_reason=outcome.exit_reason,
                    attempt=1,
                )
                if outcome.energy is not None:
                    store.add_relaxation(
                        structure_id=sid, engine=f"vasp:{step_name.lower()}",
                        energy=outcome.energy, e_per_atom=outcome.e_per_atom,
                        converged=outcome.converged, n_steps=outcome.n_ionic_steps,
                    )
                store.add_filter_event(
                    structure_id=sid, gate=f"{step_name.lower()}:converged",
                    passed=outcome.converged,
                    value=float(outcome.n_ionic_steps),
                    threshold=float(outcome.step_limit) if outcome.step_limit else None,
                    detail=outcome.exit_reason,
                )
    return stats


def _read_structure(structure_dir: Path, job_dirs: list[Path]):
    """Prefer the last step's CONTCAR (the relaxed geometry), fall back to POSCAR."""
    from ase.io import read

    for directory in reversed(job_dirs):
        for name in ("CONTCAR", "POSCAR"):
            path = directory / name
            if path.is_file() and path.stat().st_size > 0:
                try:
                    return read(str(path), format="vasp")
                except Exception:
                    continue
    return None
