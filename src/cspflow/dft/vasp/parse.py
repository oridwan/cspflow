"""Readers for VASP output.

Two constraints shape this module:

* **OUTCARs are large.** The ones in `/projects/mmi/shuo/redo-new-ter-mag` run
  to 20-26 MB each, and there are thousands. Every status question is answered
  from a tail read, never by loading the file.

* **"Finished" and "converged" are different questions, and conflating them is
  the bug this exists to avoid.** Measured over 106 jobs in that campaign: 100 %
  wrote a `VASP_DONE` marker and 100 % reached VASP's own "General timing"
  epilogue -- so every one of them exited cleanly -- but only 39 % reached the
  force criterion. The other 61 % hit `NSW` and stopped. A marker file cannot
  tell you which, so we read the OUTCAR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Enough to cover VASP's epilogue plus the last few ionic steps.
_TAIL_BYTES = 200_000

_OSZICAR_IONIC = re.compile(
    r"^\s*(\d+)\s+F=\s*([-.\dE+]+)\s+E0=\s*([-.\dE+]+)(?:.*?mag=\s*([-.\dE+]+))?",
    re.MULTILINE,
)
_INCAR_TAG = re.compile(r"^\s*([A-Z_]+)\s*=\s*(.+?)\s*(?:[#!].*)?$", re.MULTILINE)


def head_text(path: Path, nbytes: int = 40_000) -> str:
    """Read the first `nbytes` of a file as text.

    Needed because VASP prints the run's dimensions (NIONS, NBANDS, the POTCAR
    list) in the header and its outcome in the epilogue, so answering "how many
    atoms, and did it converge?" takes a read at each end -- still far cheaper
    than loading a 26 MB OUTCAR.
    """
    with open(path, "rb") as fh:
        return fh.read(nbytes).decode("utf8", "ignore")


def tail_text(path: Path, nbytes: int = _TAIL_BYTES) -> str:
    """Read the last `nbytes` of a file as text, tolerating binary noise."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf8", "ignore")


@dataclass(frozen=True)
class OszicarResult:
    n_ionic_steps: int
    energy: float | None       # F=  (free energy)
    e0: float | None           # E0= (energy with sigma->0; what we store)
    magnetisation: float | None


def read_oszicar(path: Path) -> OszicarResult:
    """Last ionic step of an OSZICAR.

    `E0` is the sigma->0 energy and is the one that belongs in a hull; `F` is
    the free energy at the smearing width actually used. They differ by a few
    meV for the ISMEAR=1/SIGMA=0.05 settings in use here, which is the same
    order as the hull thresholds, so the distinction is kept explicit rather
    than left to whichever the caller happens to grab.
    """
    if not path.is_file():
        return OszicarResult(0, None, None, None)
    text = tail_text(path)
    matches = _OSZICAR_IONIC.findall(text)
    if not matches:
        return OszicarResult(0, None, None, None)
    step, f_energy, e0, mag = matches[-1]
    return OszicarResult(
        n_ionic_steps=int(step),
        energy=float(f_energy),
        e0=float(e0),
        magnetisation=float(mag) if mag else None,
    )


@dataclass(frozen=True)
class OutcarStatus:
    exists: bool
    finished: bool          # VASP wrote its epilogue -- the process ended normally
    converged: bool         # ionic relaxation reached the force criterion
    n_atoms: int | None
    elapsed_seconds: float | None

    @property
    def hit_step_limit(self) -> bool:
        """Finished cleanly but never converged: stopped at NSW."""
        return self.finished and not self.converged


def read_outcar_status(path: Path) -> OutcarStatus:
    if not path.is_file():
        return OutcarStatus(False, False, False, None, None)
    text = tail_text(path)
    finished = "General timing and accounting" in text
    converged = "reached required accuracy" in text
    elapsed = None
    m = re.search(r"Elapsed time \(sec\):\s*([\d.]+)", text)
    if m:
        elapsed = float(m.group(1))
    # NIONS is in the header, not the epilogue.
    n_atoms = None
    m = re.search(r"NIONS\s*=\s*(\d+)", head_text(path))
    if m:
        n_atoms = int(m.group(1))
    return OutcarStatus(True, finished, converged, n_atoms, elapsed)


def read_incar(path: Path) -> dict[str, str]:
    """INCAR as a plain tag -> string mapping. No interpretation."""
    if not path.is_file():
        return {}
    return {m.group(1): m.group(2) for m in _INCAR_TAG.finditer(path.read_text(errors="ignore"))}


def incar_int(incar: dict[str, str], tag: str) -> int | None:
    raw = incar.get(tag)
    if raw is None:
        return None
    m = re.search(r"-?\d+", raw)
    return int(m.group()) if m else None


def read_potcar_symbols(path: Path) -> list[str]:
    """The ordered POTCAR symbols actually used, from the TITEL lines.

    Read from the run's own POTCAR rather than inferred from the structure, so
    what is recorded is what VASP was given.
    """
    if not path.is_file():
        return []
    symbols: list[str] = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "TITEL" in line:
                parts = line.split("=", 1)[1].split()
                if len(parts) >= 2:
                    symbols.append(parts[1])
    return symbols


@dataclass(frozen=True)
class JobOutcome:
    """Everything one relaxation directory can tell us."""

    path: Path
    state: str                  # done | failed | running | timeout
    converged: bool
    n_ionic_steps: int
    step_limit: int | None
    energy: float | None        # E0, eV
    e_per_atom: float | None
    magnetisation: float | None
    n_atoms: int | None
    slurm_id: str
    core_hours: float
    exit_reason: str
    potcar_symbols: list[str]

    @property
    def unconverged_but_finished(self) -> bool:
        return self.state == "done" and not self.converged


def _slurm_id_from_dir(directory: Path) -> str:
    """Slurm ids survive only in the stdout/stderr filenames (`vasp_<id>.out`)."""
    for pattern in ("vasp_*.out", "vasp_*.err", "*.out"):
        for candidate in sorted(directory.glob(pattern)):
            m = re.search(r"(\d{5,})", candidate.name)
            if m:
                return m.group(1)
    return ""


def read_job_directory(directory: Path) -> JobOutcome:
    """Classify one VASP run directory.

    The state machine deliberately separates process outcome from physics:
    `state` says what happened to the job, `converged` says whether the answer
    is usable. A run that finished cleanly at the ionic step limit is
    `state='done', converged=False` with `exit_reason='ionic_step_limit'` --
    not a failure, and emphatically not a success.
    """
    outcar = read_outcar_status(directory / "OUTCAR")
    oszicar = read_oszicar(directory / "OSZICAR")
    incar = read_incar(directory / "INCAR")
    nsw = incar_int(incar, "NSW")
    n_atoms = outcar.n_atoms

    if not outcar.exists:
        state, reason = "failed", "no OUTCAR"
    elif not outcar.finished:
        # No epilogue: the process was killed. Walltime is the usual cause.
        state, reason = "timeout", "OUTCAR has no epilogue (killed mid-run)"
    elif outcar.converged:
        state, reason = "done", ""
    elif nsw is not None and oszicar.n_ionic_steps >= nsw:
        state, reason = "done", "ionic_step_limit"
    else:
        state, reason = "done", "finished without reaching the force criterion"

    e_per_atom = None
    if oszicar.e0 is not None and n_atoms:
        e_per_atom = oszicar.e0 / n_atoms

    core_hours = 0.0
    if outcar.elapsed_seconds:
        ncore = incar_int(incar, "NCORE") or 1
        core_hours = outcar.elapsed_seconds / 3600.0 * ncore

    return JobOutcome(
        path=directory,
        state=state,
        converged=outcar.converged,
        n_ionic_steps=oszicar.n_ionic_steps,
        step_limit=nsw,
        energy=oszicar.e0,
        e_per_atom=e_per_atom,
        magnetisation=oszicar.magnetisation,
        n_atoms=n_atoms,
        slurm_id=_slurm_id_from_dir(directory),
        core_hours=core_hours,
        exit_reason=reason,
        potcar_symbols=read_potcar_symbols(directory / "POTCAR"),
    )
