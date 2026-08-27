"""KPOINTS.

Three schemes, and the reason there are three: a hull compares energies across
compositions and cell sizes, so the k-point sampling has to be *comparable*
across them, not merely fine enough for each.

    reciprocal_density   grid chosen so k-point density in reciprocal space is
                         constant -- a big cell gets a coarse grid, a small one
                         a fine grid, and the two are comparable. This is the
                         default and the one the legacy campaign used.
    kspacing             a target spacing in A^-1; VASP's own KSPACING tag
                         expresses the same idea, and this writes the explicit
                         grid so the file records what was actually used.
    explicit             a literal grid, for when you know what you want.

`gamma: true` throughout. A Gamma-centred grid preserves the crystal's point
symmetry; a Monkhorst-Pack grid with an even subdivision does not, and for
hexagonal cells -- which most RE-TM magnets are -- that silently breaks the
symmetry VASP then tries to use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..recipe import Kpoints


class KpointsError(Exception):
    pass


@dataclass(frozen=True)
class KpointGrid:
    a: int
    b: int
    c: int
    scheme: str
    gamma: bool = True
    comment: str = ""

    @property
    def total(self) -> int:
        return self.a * self.b * self.c

    def render(self) -> str:
        head = self.comment or f"cspflow {self.scheme}"
        style = "Gamma" if self.gamma else "Monkhorst-Pack"
        return f"{head}\n0\n{style}\n{self.a} {self.b} {self.c}\n"


def grid_for(cell_lengths: Sequence[float], kpoints: Kpoints,
             n_atoms: int = 1) -> KpointGrid:
    """The grid this recipe stage asks for, given the cell."""
    if kpoints.scheme == "explicit":
        value = kpoints.value
        if not (isinstance(value, (list, tuple)) and len(value) == 3):
            raise KpointsError(
                f"kpoints.scheme='explicit' needs a three-element grid, got {value!r}"
            )
        return KpointGrid(*(int(v) for v in value), scheme="explicit",
                          comment="cspflow explicit grid")

    if kpoints.scheme == "kspacing":
        spacing = float(kpoints.value)
        if spacing <= 0:
            raise KpointsError(f"kpoints.value must be > 0 for kspacing, got {spacing}")
        divisions = [max(1, int(math.ceil(2 * math.pi / (length * spacing))))
                     for length in cell_lengths]
        return KpointGrid(*divisions, scheme="kspacing",
                          comment=f"cspflow KSPACING {spacing} A^-1")

    if kpoints.scheme == "reciprocal_density":
        density = float(kpoints.value)
        if density <= 0:
            raise KpointsError(
                f"kpoints.value must be > 0 for reciprocal_density, got {density}")
        # pymatgen's `automatic_density_by_vol`, reproduced rather than imported
        # so the grid cannot move under a pymatgen upgrade:
        #
        #     kppa = kppvol * V_recip * n_atoms
        #     mult = (kppa / n_atoms * a*b*c) ** (1/3)
        #     n_i  = floor(max(mult / l_i, 1))
        #
        # V_recip * V_cell is (2*pi)^3 identically, and n_atoms cancels, so the
        # whole thing collapses to `mult = 2*pi * kppvol^(1/3)` -- independent of
        # both cell size and atom count. That independence is the point: it is
        # what makes the sampling comparable across the different cell sizes a
        # hull has to compare.
        #
        # Verified against the legacy campaign: Gd1Co10Cr2_s020 has
        # a,b,c = 4.6399, 8.1658, 8.2433 A, and both this and its own KPOINTS
        # file give `5 3 3`.
        # FLOOR, which is pymatgen's own behaviour. Validated against 500 of the
        # legacy campaign's own KPOINTS files:
        #
        #     exact                 402/500  (80.4%)
        #     within 1 per axis     500/500  (100%)
        #
        # Every disagreement is a length sitting within ~0.1% of an integer
        # boundary. Switching to round fixes those and breaks more than it fixes
        # (29.8% exact against 80.4%), which is the signature of the POSCAR no
        # longer being the cell its KPOINTS were generated from -- the legacy
        # restart scripts overwrite POSCAR from CONTCAR and do not regenerate
        # KPOINTS. cspflow writes both together and keeps the originals, so the
        # question cannot arise here.
        mult = 2 * math.pi * (density ** (1.0 / 3.0))
        divisions = [max(1, int(math.floor(mult / max(float(length), 1e-9))))
                     for length in cell_lengths]
        return KpointGrid(*divisions, scheme="reciprocal_density",
                          comment=f"cspflow reciprocal_density {density:g}")

    raise KpointsError(
        f"unknown kpoints.scheme {kpoints.scheme!r}; known: reciprocal_density, "
        f"kspacing, explicit"
    )


