"""Assemble a complete VASP job directory.

`csp dft --dry-run` calls exactly this and prints what it produced, which is the
property that makes the dry run worth having: there is no second code path that
"would" write the files.

Four files, and one manifest:

    INCAR     the recipe's literal tags plus the computed ones (incar.py)
    POSCAR    the structure, species-sorted
    KPOINTS   from the recipe's scheme and the cell (kpoints.py)
    POTCAR    concatenated in POSCAR species order
    inputs.json  what was resolved, and the hashes that pin it

The manifest is the part that is easy to skip and expensive to have skipped. It
records the resolved INCAR, the POTCAR `md5_header_hash` per element, and the
recipe stage -- so `settings_hash` describes what was actually written rather
than what the config said, and two jobs can be shown to be comparable without
re-reading their inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ...config.schema import Dft, Machine
from ..recipe import RecipeStage
from . import potcar as pc
from .incar import IncarContext, build_incar, render_incar
from .kpoints import KpointGrid, grid_for


class InputError(Exception):
    pass


@dataclass
class ResolvedInputs:
    """Everything that will be written, before anything is."""

    stage: str
    incar: dict[str, Any]
    grid: KpointGrid
    symbols: list[str]                 # POSCAR species order
    potcars: list[pc.PotcarInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def settings_hash(self) -> str:
        """A hash over what will actually be computed.

        Deliberately includes the POTCAR hashes: two jobs with identical INCARs
        and different pseudopotentials are not comparable, and nothing else in
        the record would show it.
        """
        payload = {
            "stage": self.stage,
            "incar": {k: _hashable(v) for k, v in sorted(self.incar.items())},
            "kpoints": [self.grid.a, self.grid.b, self.grid.c, self.grid.scheme],
            "potcars": sorted((p.element, p.symbol, p.md5_header_hash)
                              for p in self.potcars),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.settings_hash[:12]

    def render(self) -> str:
        lines = [f"stage {self.stage}   settings_hash {self.short_hash}", "",
                 "INCAR", *(f"  {line}" for line in
                            render_incar(self.incar).rstrip().splitlines()),
                 "", "KPOINTS",
                 *(f"  {line}" for line in self.grid.render().rstrip().splitlines()),
                 "", "POTCAR"]
        for p in self.potcars:
            lines.append(f"  {p.element:<4} {p.symbol:<6} {p.titel:<28} "
                         f"ZVAL {p.zval:6.1f}  {p.short_hash}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def resolve_inputs(
    atoms,
    stage: RecipeStage,
    dft: Dft,
    machine: Machine,
    *,
    ntasks: int | None = None,
) -> ResolvedInputs:
    """Work out every input file's contents.  Writes nothing."""
    symbols = _species_order(atoms)
    infos, errors = pc.resolve_all(
        sorted(set(symbols)), machine,
        tree=dft.potcar.tree, f_treatment=dft.rare_earth.f_treatment,
        overrides=dft.potcar.overrides if hasattr(dft.potcar, "overrides") else None,
    )
    if errors:
        raise InputError(
            "cannot resolve every POTCAR for this structure:\n  " + "\n  ".join(errors)
        )
    pc.assert_one_f_convention(infos)

    by_element = {p.element: p for p in infos}
    zvals = {e: p.zval for e, p in by_element.items()}
    f_in_valence = any(p.n_f_valence for p in infos)

    context = IncarContext(
        symbols=list(atoms.get_chemical_symbols()),
        formula=atoms.get_chemical_formula(),
        zvals=zvals,
        f_in_valence=f_in_valence,
    )
    incar = build_incar(
        stage.incar, context,
        magnetism=dft.magnetism, rare_earth=dft.rare_earth, ldau=dft.ldau,
        nbands=dft.nbands, overrides=dft.incar_overrides,
    )

    warnings = []
    encut = incar.get("ENCUT")
    max_enmax = pc.max_enmax(infos)
    if encut is not None and float(encut) < max_enmax:
        warnings.append(
            f"ENCUT {encut} eV is below the largest ENMAX in this structure's POTCARs "
            f"({max_enmax:.1f} eV). VASP will run, and the basis is under-converged "
            f"for that species."
        )

    grid = grid_for(list(atoms.cell.lengths()), stage.kpoints, n_atoms=len(atoms))

    ordered = [by_element[s] for s in symbols]
    return ResolvedInputs(stage=stage.name, incar=incar, grid=grid,
                          symbols=symbols, potcars=ordered, warnings=warnings)


def write_inputs(resolved: ResolvedInputs, atoms, directory: Path) -> Path:
    """Write the four files and the manifest.  Returns the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "INCAR").write_text(
        render_incar(resolved.incar,
                     comment=f"cspflow {resolved.stage}  {resolved.short_hash}")
    )
    (directory / "KPOINTS").write_text(resolved.grid.render())
    _write_poscar(atoms, resolved.symbols, directory / "POSCAR")
    _concat_potcars(resolved.potcars, directory / "POTCAR")

    (directory / "inputs.json").write_text(json.dumps({
        "stage": resolved.stage,
        "settings_hash": resolved.settings_hash,
        "incar": {k: _hashable(v) for k, v in sorted(resolved.incar.items())},
        "kpoints": {"a": resolved.grid.a, "b": resolved.grid.b, "c": resolved.grid.c,
                    "scheme": resolved.grid.scheme, "gamma": resolved.grid.gamma},
        "potcars": [{"element": p.element, "symbol": p.symbol, "titel": p.titel,
                     "zval": p.zval, "enmax": p.enmax,
                     "md5_header_hash": p.md5_header_hash} for p in resolved.potcars],
        "warnings": resolved.warnings,
    }, indent=2, sort_keys=True))
    return directory


# --------------------------------------------------------------------------


def _species_order(atoms) -> list[str]:
    """First appearance, not alphabetical.

    POTCAR concatenation order and the positional LDAU arrays both key on this,
    so getting it wrong applies the wrong pseudopotential to the wrong element --
    and VASP will not complain, because the file is well-formed.
    """
    seen: list[str] = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.append(symbol)
    return seen


def _write_poscar(atoms, symbols: Sequence[str], path: Path) -> None:
    """Species-sorted POSCAR with an explicit element line.

    The element line is not optional here for the same reason Stage 0.3 refuses
    a POSCAR without one: a VASP-4 file has no species names and every reader
    that accepts it has to invent them.
    """
    from ase.io import write

    ordered = atoms.copy()
    order = sorted(range(len(ordered)),
                   key=lambda i: symbols.index(ordered.get_chemical_symbols()[i]))
    ordered = ordered[order]
    write(str(path), ordered, format="vasp", direct=True, sort=False)

    lines = path.read_text().splitlines()
    if len(lines) > 5 and all(tok.lstrip("+-").isdigit() for tok in lines[5].split()):
        # Older ASE omits the species line; put it back rather than shipping a
        # file we would ourselves refuse to read.
        counts: list[str] = []
        current, n = None, 0
        for s in ordered.get_chemical_symbols():
            if s != current:
                if current is not None:
                    counts.append(str(n))
                current, n = s, 1
            else:
                n += 1
        counts.append(str(n))
        lines.insert(5, " ".join(symbols))
        path.write_text("\n".join(lines) + "\n")


def _concat_potcars(infos: Sequence[pc.PotcarInfo], path: Path) -> None:
    with path.open("wb") as out:
        for info in infos:
            out.write(Path(info.path).read_bytes())


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    return value
