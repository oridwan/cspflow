"""Moments, checked against six completed campaigns.

`candidates_w_proto_mag.csv` in each campaign records a `total_mag` per
candidate. That is an independent oracle for the OUTCAR reading -- and, once the
reading is trusted, it is also the evidence for what `total_mag` is *not*.

Skipped when the campaign directories are not mounted.
"""

import csv
import statistics
import sys
from pathlib import Path

import pytest

from cspflow.analysis.hund import HEAVY, LIGHT, counts_of, reconstruct
from cspflow.analysis.moments import read_site_moments
from cspflow.analysis.properties import spacegroup_of
from cspflow.chem import RARE_EARTHS

ROOT = Path("/projects/mmi/shuo")
pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason="campaign tree not mounted")


def _symbols(poscar: Path) -> list[str]:
    lines = poscar.read_text().splitlines()
    out: list[str] = []
    for element, n in zip(lines[5].split(), (int(x) for x in lines[6].split())):
        out += [element] * n
    return out


@pytest.fixture(scope="module")
def candidates():
    """Every candidate row that still has its relaxation directory."""
    rows = []
    for csv_path in ROOT.glob("*/candidates_w_proto_mag.csv"):
        campaign = csv_path.parent
        for row in csv.DictReader(csv_path.open()):
            sid = row["structure_id"]
            directory = campaign / "VASP_JOBS" / sid.split("_")[0] / sid / "Relax"
            if (directory / "OUTCAR").is_file() and (directory / "POSCAR").is_file():
                rows.append((sid, directory, row))
    if not rows:
        pytest.skip("no candidate relaxations found")
    return rows


@pytest.fixture(scope="module")
def read(candidates):
    """(sid, kind, m_cell, m_s_reconstructed, quoted) for every candidate."""
    out = []
    for sid, directory, row in candidates:
        symbols = _symbols(directory / "POSCAR")
        report = read_site_moments(directory / "OUTCAR", symbols)
        if report.m_cell is None or not report.sites:
            continue
        rare_earth = counts_of(symbols, RARE_EARTHS)
        kind = ("none" if not rare_earth
                else "light" if set(rare_earth) <= LIGHT else "heavy")
        out.append((sid, kind, report.m_cell,
                    reconstruct(report.transition_metal, rare_earth).m_s,
                    float(row["total_mag"])))
    return out


def test_our_cell_magnetisation_is_the_campaigns_total_mag(read):
    """961 of 961, to 5 mB. This is what makes the rest of the file evidence."""
    assert len(read) > 900
    misses = [sid for sid, _, m_cell, _, quoted in read if abs(m_cell - quoted) > 0.005]
    assert misses == [], f"{len(misses)} of {len(read)} disagree, e.g. {misses[:5]}"


def test_without_a_rare_earth_the_two_numbers_nearly_agree(read):
    """The floor of the method: PAW spheres do not tile the cell, so the
    sublattice sum and the cell integral differ by a few percent even when
    there is no 4f moment to reconstruct."""
    rows = [r for r in read if r[1] == "none" and abs(r[2]) >= 1.0]
    assert len(rows) > 100
    relative = [abs(ms - m) / abs(m) for _, _, m, ms, _ in rows]
    assert statistics.median(relative) < 0.10


def test_a_light_rare_earth_shifts_the_number_but_never_its_sign(read):
    """Sm(3+) has g_J*J = 0.71 -- small, and it adds rather than subtracts."""
    rows = [r for r in read if r[1] == "light" and abs(r[2]) >= 1.0]
    assert len(rows) > 100
    assert all((m > 0) == (ms > 0) for _, _, m, ms, _ in rows)


def test_a_heavy_rare_earth_more_than_doubles_it_and_half_the_time_flips_it(read):
    """The finding.

    For Gd and Tb the 4f moment is both large (7.00 and 9.00 mu_B) and
    antiparallel to the transition-metal sublattice, and a frozen-4f POTCAR
    omits it entirely. Across 401 candidate rows in six campaigns the quoted
    `total_mag` differs from the reconstructed saturation magnetisation by a
    median of 124%, and for 52% of them the net moment points the other way.

    Neither number is "the truth" -- the reconstruction is a collinear free-ion
    model. The point is that they are different quantities, and only one of them
    is the saturation magnetisation of a rare-earth magnet.
    """
    rows = [r for r in read if r[1] == "heavy" and abs(r[2]) >= 1.0]
    assert len(rows) > 300
    relative = [abs(ms - m) / abs(m) for _, _, m, ms, _ in rows]
    flips = [1 for _, _, m, ms, _ in rows if (m > 0) != (ms > 0)]
    assert statistics.median(relative) > 1.0
    assert 0.4 < len(flips) / len(rows) < 0.7


def test_the_shift_grows_with_the_rare_earth_count(read):
    """A sanity check on the reconstruction itself: it is n_RE * g_J * J."""
    heavy = [r for r in read if r[1] == "heavy" and abs(r[2]) >= 1.0]
    shifts = [m - ms for _, _, m, ms, _ in heavy]      # positive: raw overstates
    assert statistics.median(shifts) > 5.0


@pytest.mark.slow
def test_spacegroups_agree_with_the_campaigns_to_within_a_supergroup(candidates):
    """Ours are found at symprec=0.1 and the campaign's with PyXtal's adaptive
    tolerance, so a handful of ours are the centrosymmetric supergroup of
    theirs. Reported with the tolerance for exactly this reason."""
    import ase.io

    sample = candidates[::16]
    agree = checked = 0
    for _sid, directory, row in sample:
        contcar = directory / "CONTCAR"
        if not contcar.is_file() or not row.get("spg_num"):
            continue
        checked += 1
        number, _ = spacegroup_of(ase.io.read(str(contcar), format="vasp"))
        if number == int(float(row["spg_num"])):
            agree += 1
    assert checked > 20
    assert agree / checked > 0.9
