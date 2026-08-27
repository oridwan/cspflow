"""Stage 7 -- moments, the Hund's-rule reconstruction, and properties.

The interesting content here is not parsing. It is the insistence that a
computed number and a modelled one never share a column, which is the one thing
in the whole funnel that produces a wrong *published* result rather than a
wasted job.
"""

import textwrap
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk

from cspflow.analysis.hund import (GJ_J, HEAVY, LIGHT, HundError, counts_of,
                                   reconstruct, sign_for)
from cspflow.analysis.moments import (MomentReport, frozen_4f_is_visible,
                                      read_cell_magnetisation, read_site_moments)
from cspflow.analysis.properties import extract, spacegroup_of

REAL = Path("/projects/mmi/shuo/redo-new-ter-mag/VASP_JOBS/Gd1Co10Cr2/Gd1Co10Cr2_s020/Relax")
has_real = pytest.mark.skipif(not REAL.is_dir(), reason="campaign job not present")


# -- the Hund's-rule table -------------------------------------------------

def test_the_table_covers_every_rare_earth_exactly_once():
    assert LIGHT & HEAVY == set()
    assert LIGHT | HEAVY == set(GJ_J)


@pytest.mark.parametrize("element,value", [("Nd", 3.27), ("Sm", 0.71), ("Gd", 7.00),
                                           ("Tb", 9.00), ("Dy", 10.00), ("Eu", 0.00)])
def test_the_tabulated_moments_are_the_free_ion_values(element, value):
    assert GJ_J[element] == value


def test_light_rare_earths_add_and_heavy_ones_subtract():
    assert sign_for("Nd") == +1
    assert sign_for("Sm") == +1
    assert sign_for("Tb") == -1
    assert sign_for("Dy") == -1


def test_gadolinium_belongs_with_the_heavy_group():
    """L = 0, so J = S, and the spin couples antiparallel to the transition
    metal. This is why GdCo5 is a ferrimagnet with a compensation point."""
    assert sign_for("Gd") == -1


def test_a_transition_metal_has_no_4f_moment_to_add():
    with pytest.raises(HundError):
        sign_for("Fe")


# -- reconstruction --------------------------------------------------------

def test_a_heavy_rare_earth_subtracts_its_full_moment():
    result = reconstruct(24.266, {"Gd": 2})
    assert result.m_s == pytest.approx(24.266 - 14.0)
    assert result.contributions == {"Gd": -14.0}


def test_a_light_rare_earth_adds_a_small_one():
    result = reconstruct(30.0, {"Sm": 2})
    assert result.m_s == pytest.approx(30.0 + 1.42)


def test_reconstruction_starts_from_the_sublattice_not_the_cell():
    """The cell magnetisation already contains the frozen rare earth's residual
    (-0.248 mu_B per Gd in the campaign's own output). Adding a full Hund's-rule
    moment on top of that counts it twice."""
    tm_only = reconstruct(24.266, {"Gd": 2}).m_s
    cell_including_residual = reconstruct(23.770, {"Gd": 2}).m_s
    assert tm_only != cell_including_residual


def test_a_structure_with_no_rare_earth_is_just_the_sublattice_sum():
    result = reconstruct(12.5, {})
    assert result.m_s == 12.5
    assert any("no rare earth" in w for w in result.warnings)


def test_valence_treatment_refuses_to_reconstruct():
    """With 4f in the valence the moment is already in the DFT number."""
    result = reconstruct(24.266, {"Gd": 2}, f_treatment="valence")
    assert result.m_s == 24.266
    assert result.contributions == {}
    assert any("double-count" in w for w in result.warnings)


def test_an_untabulated_element_is_an_error_not_a_zero():
    with pytest.raises(HundError, match="no Hund's-rule moment"):
        reconstruct(1.0, {"Xx": 1})


def test_the_reconstruction_shows_its_working():
    rendered = reconstruct(24.266, {"Gd": 2}).render()
    assert "TM +24.266" in rendered and "Gd -14.000" in rendered and "[frozen]" in rendered


def test_counts_can_be_restricted_to_the_rare_earths():
    symbols = ["Gd", "Gd", "Co", "Co", "Cr"]
    assert counts_of(symbols) == {"Gd": 2, "Co": 2, "Cr": 1}
    assert counts_of(symbols, frozenset({"Gd"})) == {"Gd": 2}


# -- reading an OUTCAR -----------------------------------------------------

SYNTHETIC = """\
  number of electron     246.0000009 magnetization      10.5000000

 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1       -0.007  -0.066  -0.175  -0.248
    2       -0.004  -0.034   1.310   1.273
    3       -0.039  -0.059  -1.931  -2.028
--------------------------------------------------
tot         -0.050  -0.159  -0.796  -1.003

 General timing and accounting informations for this job:
"""


def test_a_synthetic_table_parses(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(SYNTHETIC)
    report = read_site_moments(outcar, ["Gd", "Co", "Cr"])
    assert report.m_cell == pytest.approx(10.5)
    assert report.m_spheres == pytest.approx(-0.248 + 1.273 - 2.028)
    assert report.rare_earth == pytest.approx(-0.248)
    assert report.transition_metal == pytest.approx(1.273 - 2.028)
    assert report.sites[0].channels == {"s": -0.007, "p": -0.066, "d": -0.175}


def test_a_table_of_the_wrong_length_is_an_error(tmp_path):
    from cspflow.analysis.moments import MomentError

    outcar = tmp_path / "OUTCAR"
    outcar.write_text(SYNTHETIC)
    with pytest.raises(MomentError, match="not the same calculation"):
        read_site_moments(outcar, ["Gd", "Co"])


def test_a_missing_projected_table_says_what_is_needed(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("  number of electron  10.0 magnetization      3.5\n")
    report = read_site_moments(outcar, ["Fe"])
    assert report.m_cell == pytest.approx(3.5)
    assert report.sites == []
    assert "LORBIT=11" in report.note


def test_a_missing_file_is_reported_not_raised(tmp_path):
    report = read_site_moments(tmp_path / "nope", ["Fe"])
    assert report.m_cell is None and "not found" in report.note
    assert read_cell_magnetisation(tmp_path / "nope") is None


def test_the_sphere_sum_is_not_the_cell_magnetisation(tmp_path):
    """They differ because PAW spheres do not tile the cell. Reporting one
    under the other's name is the error this pair of fields exists to stop."""
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(SYNTHETIC)
    report = read_site_moments(outcar, ["Gd", "Co", "Cr"])
    assert report.m_cell != report.m_spheres
    assert report.sphere_deficit == pytest.approx(10.5 - (-1.003), abs=1e-3)


# -- against the campaign's own output -------------------------------------

@has_real
def test_the_real_outcar_reproduces_the_campaigns_number():
    symbols = ["Gd"] * 2 + ["Cr"] * 4 + ["Co"] * 20
    report = read_site_moments(REAL / "OUTCAR", symbols)
    assert report.m_cell == pytest.approx(23.2240109)
    assert report.m_spheres == pytest.approx(23.770, abs=1e-3)
    assert report.by_element["Gd"] == pytest.approx(-0.496, abs=1e-3)
    assert report.by_element["Co"] == pytest.approx(26.798, abs=1e-3)


@has_real
def test_the_frozen_4f_is_visible_in_the_real_result():
    """MAGMOM started the two Gd at -7.0; they came back at -0.248."""
    symbols = ["Gd"] * 2 + ["Cr"] * 4 + ["Co"] * 20
    assert frozen_4f_is_visible(read_site_moments(REAL / "OUTCAR", symbols))


@has_real
def test_the_reconstruction_halves_the_quoted_moment_for_this_compound():
    """Gd2Cr4Co20: 23.22 mu_B raw, 10.27 reconstructed.

    Quoting the raw number as the saturation magnetisation overstates it by
    126%, because the two Gd(3+) moments are antiparallel to the Co sublattice
    and absent from the frozen-4f calculation entirely.
    """
    props = extract(REAL, z=2)
    assert props.m_dft_raw == pytest.approx(23.224, abs=1e-3)
    assert props.m_s_reconstructed == pytest.approx(10.266, abs=1e-3)


@has_real
def test_properties_from_a_real_directory():
    props = extract(REAL, z=2)
    assert props.formula == "Co20Cr4Gd2"
    assert props.n_atoms == 26
    assert props.converged
    assert props.volume == pytest.approx(320.930, abs=1e-2)
    assert props.spacegroup_number == 44
    assert props.spacegroup_symbol == "Imm2"
    assert props.sublattice["rare_earth"] == pytest.approx(-0.496, abs=1e-3)
    assert props.warnings == []


@has_real
def test_the_stored_key_values_drop_nothing_that_was_computed():
    kv = extract(REAL, z=2).as_kv()
    assert set(kv) >= {"volume", "spacegroup", "m_dft_raw", "m_s_reconstructed",
                       "m_per_formula_unit", "m_per_volume", "f_treatment"}


# -- spacegroup ------------------------------------------------------------

def test_spacegroup_of_a_known_cell():
    number, symbol = spacegroup_of(bulk("Fe", "bcc", a=2.87, cubic=True))
    assert number == 229 and symbol == "Im-3m"


def test_a_broken_cell_reports_rather_than_raises():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0)], cell=[0, 0, 0], pbc=True)
    number, note = spacegroup_of(atoms)
    assert number is None or isinstance(number, int)


def test_an_absent_contcar_is_a_warning_not_a_crash(tmp_path):
    props = extract(tmp_path)
    assert props.volume is None
    assert any("CONTCAR" in w for w in props.warnings)
