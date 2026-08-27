"""Magnetic moments, read from an OUTCAR and put back together.

Two numbers here are both called "the magnetisation" and they are not equal.

*   **The cell magnetisation** is the spin density integrated over the whole
    cell -- OSZICAR's `mag=`, and OUTCAR's `number of electron ... magnetization`.
*   **The sphere sum** is the site-projected table `LORBIT=11` writes, summed
    over ions.  PAW spheres do not tile the cell, so this misses whatever moment
    lives in the interstitial.

On the campaign's own `Gd2Cr4Co20` they are 23.2240 and 23.768 -- 2.3% apart.
Either is a defensible number to quote; quoting one under the other's name is
not, so `MomentReport` carries both and says which is which.  `m_dft_raw` is
the cell value, because that is the one that does not depend on where a PAW
sphere happens to end.

The sublattice split can only come from the projected table, so a sublattice
moment is a sphere quantity and is labelled that way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..chem import RARE_EARTHS
from ..dft.vasp.parse import tail_text

# The site-projected table, as LORBIT=11 writes it. There is one per ionic step
# and one more in the final block, so the parser takes the last.
_TABLE_HEAD = re.compile(
    r"^ magnetization \(x\)\s*\n\s*\n"
    r"# of ion\s+((?:[spdf]\s+)+)tot\s*\n"
    r"-+\s*\n",
    re.M)
_ION_ROW = re.compile(r"^\s*(\d+)((?:\s+-?\d+\.\d+)+)\s*$", re.M)
_CELL_MAG = re.compile(r"number of electron\s+[-\d.]+\s+magnetization\s+(-?[\d.]+)")

# How much of the OUTCAR to read from the end. The final magnetisation table
# and the epilogue are within a few tens of kB even for a large cell; the file
# itself is often ~10 MB, and analysis runs over thousands of them.
_TAIL = 400_000


class MomentError(Exception):
    pass


@dataclass(frozen=True)
class SiteMoment:
    """One ion's projected moment, in Bohr magnetons."""

    index: int              # 1-based, as VASP numbers ions
    element: str
    channels: dict[str, float]
    total: float

    @property
    def d(self) -> float:
        return self.channels.get("d", 0.0)

    @property
    def f(self) -> float:
        return self.channels.get("f", 0.0)


@dataclass
class MomentReport:
    """Everything one relaxation says about its magnetism."""

    m_cell: float | None = None            # integrated over the cell
    m_spheres: float | None = None         # summed over PAW spheres
    sites: list[SiteMoment] = field(default_factory=list)
    by_element: dict[str, float] = field(default_factory=dict)
    rare_earth: float = 0.0                # sphere sum over RE ions
    transition_metal: float = 0.0          # sphere sum over everything else
    n_atoms: int = 0
    note: str = ""

    @property
    def sphere_deficit(self) -> float | None:
        """Cell minus spheres: the moment the projection does not see."""
        if self.m_cell is None or self.m_spheres is None:
            return None
        return self.m_cell - self.m_spheres

    def per_formula_unit(self, z: int) -> float | None:
        if self.m_cell is None or z <= 0:
            return None
        return self.m_cell / z

    def per_volume(self, volume: float) -> float | None:
        """Bohr magnetons per cubic angstrom -- the figure of merit for a magnet."""
        if self.m_cell is None or volume <= 0:
            return None
        return self.m_cell / volume


def read_cell_magnetisation(outcar: Path) -> float | None:
    """The last `number of electron ... magnetization` in the file."""
    if not Path(outcar).is_file():
        return None
    matches = _CELL_MAG.findall(tail_text(Path(outcar), _TAIL))
    return float(matches[-1]) if matches else None


def read_site_moments(outcar: Path, symbols: list[str]) -> MomentReport:
    """The last site-projected table, mapped onto `symbols`.

    `symbols` is the per-ion element list in POSCAR order -- the same order VASP
    numbers ions in, so ion *i* is `symbols[i-1]`.  It is required rather than
    parsed out of the OUTCAR because the OUTCAR names only the POTCAR titles and
    their counts, and a mismatch between those and the structure is exactly the
    error this would otherwise hide.
    """
    path = Path(outcar)
    report = MomentReport(n_atoms=len(symbols))
    if not path.is_file():
        report.note = f"{path} not found"
        return report

    text = tail_text(path, _TAIL)
    report.m_cell = float(_CELL_MAG.findall(text)[-1]) if _CELL_MAG.search(text) else None

    heads = list(_TABLE_HEAD.finditer(text))
    if not heads:
        report.note = ("no site-projected magnetisation table. LORBIT=11 writes "
                       "one; without it only the cell magnetisation is available "
                       "and no sublattice split is possible.")
        return report

    head = heads[-1]
    channels = head.group(1).split()
    block = text[head.end():]
    end = block.find("---")
    rows = _ION_ROW.findall(block if end < 0 else block[:end])

    if len(rows) != len(symbols):
        raise MomentError(
            f"{path}: the projected table has {len(rows)} ions but the structure "
            f"has {len(symbols)}. The OUTCAR and the structure are not the same "
            f"calculation.")

    total = 0.0
    for (index, values), symbol in zip(rows, symbols):
        # `index  <one number per channel>  tot` -- the last column is the sum
        # over channels, so it is taken rather than recomputed: VASP rounds each
        # column to three decimals and the printed total is not always the sum
        # of the printed parts.
        numbers = [float(v) for v in values.split()]
        site_total = numbers[-1]
        moment = SiteMoment(index=int(index), element=symbol,
                            channels=dict(zip(channels, numbers[:-1])),
                            total=site_total)
        report.sites.append(moment)
        report.by_element[symbol] = report.by_element.get(symbol, 0.0) + site_total
        if symbol in RARE_EARTHS:
            report.rare_earth += site_total
        else:
            report.transition_metal += site_total
        total += site_total

    report.m_spheres = total
    return report


def frozen_4f_is_visible(report: MomentReport, threshold: float = 0.5) -> bool:
    """True when the rare-earth ions carry essentially no moment.

    Which is the *expected* result for a frozen-4f POTCAR (`Gd_3`, `Sm_3`): the
    f electrons are in the core, so there is no 4f moment to find. On the
    campaign's `Gd2Cr4Co20` the two Gd ions come back at -0.248 each having been
    started at -7.0.

    It is worth asserting rather than assuming, because the same near-zero
    number is also what a *valence* calculation produces when the f states
    collapsed onto the Fermi level -- and that one is wrong.
    """
    re_sites = [s for s in report.sites if s.element in RARE_EARTHS]
    if not re_sites:
        return False
    return all(abs(s.total) < threshold for s in re_sites)
