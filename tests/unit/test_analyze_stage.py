"""Stage 7 as a driver stage: extraction, the DFT hull, and idempotence."""

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.stages.analyze_stage import DIR_KEY, DONE_KEY, AnalyzeStage

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
dft:
  recipe: magnets
  magnetism: {{mode: ferrimagnetic_retm}}
"""

OUTCAR = """\
   NIONS =      3
  free  energy   TOTEN  =       -20.000000 eV
  energy  without entropy=      -20.100000  energy(sigma->0) =      -20.000000
  number of electron     30.0000000 magnetization       4.0000000

 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1       -0.007  -0.066  -0.175  -0.100
    2       -0.004  -0.034   1.310   1.500
    3       -0.004  -0.034   1.310   1.500
--------------------------------------------------
tot         -0.015  -0.134   2.445   2.900

 reached required accuracy - stopping structural energy minimisation
 General timing and accounting informations for this job:
                         Elapsed time (sec):     100.0
"""

OSZICAR = "   1 F= -.20000000E+02 E0= -.20000000E+02  d E =0.0  mag=     4.0000\n"

CONTCAR = """\
Gd1 Co2
1.0
   4.0 0.0 0.0
   0.0 4.0 0.0
   0.0 0.0 4.0
Gd Co
1 2
Direct
0.00 0.00 0.00
0.50 0.50 0.00
0.00 0.50 0.50
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def make_job(tmp_path, name="dft-1-relax", nsw=100):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "OUTCAR").write_text(OUTCAR)
    (d / "OSZICAR").write_text(OSZICAR)
    (d / "CONTCAR").write_text(CONTCAR)
    (d / "INCAR").write_text(f"NSW = {nsw}\nLORBIT = 11\n")
    return d


def add_done(store, tmp_path, energy=-20.0, symbols="GdCo2", **kv):
    atoms = Atoms(symbols, positions=[(0, 0, 0), (2, 2, 0), (0, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    return store.add_structure(atoms, origin=Origin.generated,
                               state=StructureState.dft_done,
                               vasp_energy=energy, **kv)


# -- extraction ------------------------------------------------------------

def test_properties_are_extracted_and_stored(cfg, store, tmp_path):
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    report = AnalyzeStage(cfg).run(store)

    assert report.claimed == 1
    row = store.get_structure(sid)
    kv = row.key_value_pairs
    assert kv["m_dft_raw"] == pytest.approx(4.0)
    assert kv["volume"] == pytest.approx(64.0)
    assert kv["spacegroup"] > 0
    assert kv[DONE_KEY] is True


def test_the_two_moments_are_stored_as_separate_properties(cfg, store, tmp_path):
    """`m_dft_raw` is computed; `m_s_reconstructed` is a model on top of it."""
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)

    props = {p["key"]: p["value"] for p in store.properties(sid)}
    assert props["m_dft_raw"] == pytest.approx(4.0)
    # TM sublattice 3.0, one Gd at -7.0: 3.0 - 7.0 = -4.0
    assert props["m_s_reconstructed"] == pytest.approx(-4.0)
    assert props["m_dft_raw"] != props["m_s_reconstructed"]


def test_the_sublattice_split_is_stored(cfg, store, tmp_path):
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)
    props = {p["key"]: p["value"] for p in store.properties(sid)}
    assert props["m_rare_earth"] == pytest.approx(-0.1)
    assert props["m_transition_metal"] == pytest.approx(3.0)


def test_a_second_run_does_not_re_read_the_outcar(cfg, store, tmp_path):
    job = make_job(tmp_path)
    add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    stage = AnalyzeStage(cfg)
    assert stage.run(store).claimed == 1
    assert stage.pending(store) == 0
    assert stage.run(store).claimed == 0


def test_a_structure_with_no_recorded_directory_is_marked_and_noted(cfg, store,
                                                                    tmp_path):
    sid = add_done(store, tmp_path)
    AnalyzeStage(cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    assert kv[DONE_KEY] is True
    assert "no DFT directory" in kv["analyze_note"]


def test_an_unconverged_job_is_analysed_but_flagged(cfg, store, tmp_path):
    job = make_job(tmp_path)
    (job / "OUTCAR").write_text(OUTCAR.replace(
        "reached required accuracy - stopping structural energy minimisation", ""))
    sid = add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    AnalyzeStage(cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    assert kv["volume"] == pytest.approx(64.0)
    assert "not converged" in kv["analyze_note"]


# -- the DFT hull ----------------------------------------------------------

def add_reference(store, formula, e_per_atom, chemsys):
    """`e_dft_raw` is per atom, as the reference stage stores it."""
    from cspflow.chem import parse_formula

    counts = parse_formula(formula)
    return store.add_reference_entry(
        mp_id=f"mp-{formula}", formula=formula, chemsys=chemsys,
        e_dft_raw=e_per_atom, e_dft_corrected=e_per_atom, run_type="GGA",
        thermo_type="GGA_GGA+U", n_atoms=sum(counts.values()), state="fetched")


def test_the_hull_is_built_on_our_energies_against_mp(cfg, store, tmp_path):
    job = make_job(tmp_path)
    sid = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    add_reference(store, "Gd1", -3.0, "Gd")
    add_reference(store, "Co1", -5.0, "Co")

    AnalyzeStage(cfg).run(store)
    kv = store.get_structure(sid).key_value_pairs
    # GdCo2: -20.0 total; references -3 and -5 per atom -> formation energy
    # (-20 - (-3) - 2*(-5)) / 3 = -2.333 eV/atom, and it is the only ternary
    # point, so it is on the hull.
    assert kv["dft_e_formation"] == pytest.approx(-2.3333, abs=1e-3)
    assert kv["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)


def test_a_system_with_no_elemental_reference_is_reported_not_guessed(cfg, store,
                                                                      tmp_path):
    job = make_job(tmp_path)
    add_done(store, tmp_path, **{DIR_KEY: str(job), "z": 1})
    report = AnalyzeStage(cfg).run(store)
    assert "Co-Gd" in report.note


def test_a_better_candidate_pushes_the_other_off_the_hull(cfg, store, tmp_path):
    job = make_job(tmp_path)
    add_reference(store, "Gd1", -3.0, "Gd")
    add_reference(store, "Co1", -5.0, "Co")
    high = add_done(store, tmp_path, energy=-15.0, **{DIR_KEY: str(job), "z": 1})
    low = add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})

    AnalyzeStage(cfg).run(store)
    assert store.get_structure(low).key_value_pairs["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)
    assert store.get_structure(high).key_value_pairs["dft_e_above_hull"] > 1.0


def test_the_hull_is_recomputed_when_a_new_result_lands(cfg, store, tmp_path):
    """`e_above_hull` is not a property of one structure; it moves when a
    competing phase appears. Re-placing every cycle is what keeps it honest."""
    job = make_job(tmp_path)
    add_reference(store, "Gd1", -3.0, "Gd")
    add_reference(store, "Co1", -5.0, "Co")
    first = add_done(store, tmp_path, energy=-15.0, **{DIR_KEY: str(job), "z": 1})
    stage = AnalyzeStage(cfg)
    stage.run(store)
    assert store.get_structure(first).key_value_pairs["dft_e_above_hull"] == pytest.approx(0.0, abs=1e-9)

    add_done(store, tmp_path, energy=-20.0, **{DIR_KEY: str(job), "z": 1})
    stage.run(store)
    assert store.get_structure(first).key_value_pairs["dft_e_above_hull"] > 1.0


# -- registry --------------------------------------------------------------

def test_every_funnel_stage_now_has_an_implementation():
    from cspflow.driver import STAGE_ORDER
    from cspflow.stages import IMPLEMENTED, PLANNED

    assert PLANNED == {}
    assert set(IMPLEMENTED) == set(STAGE_ORDER)
    assert IMPLEMENTED == STAGE_ORDER
