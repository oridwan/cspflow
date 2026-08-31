"""Adoption of a finished legacy campaign.

Built on a synthetic flow rather than the real 225,000-structure one: the whole
point of adoption is that six artefacts have to agree with each other, and a
fixture is the only way to assert what happens when they do not.  Two opt-in
tests read the real campaigns to confirm the fixture is not a fiction.
"""

import json
from pathlib import Path

import pytest

from ase import Atoms
from ase.db import connect

from cspflow.db.store import Store
from cspflow.legacy import (
    REGISTRY,
    AdoptStats,
    LegacyError,
    LegacyFlow,
    adopt,
    render_config,
)

from .test_ingest import _oszicar, make_job

SHUO = Path("/projects/mmi/shuo")
has_real = pytest.mark.skipif(not SHUO.is_dir(), reason="legacy campaigns not present")


# --------------------------------------------------------------------------
# a synthetic legacy flow
# --------------------------------------------------------------------------

def _cell(formula: str, z: int) -> Atoms:
    """One formula unit repeated Z times, so the adopter sees the Z it must infer."""
    symbols = formula.replace("1", "").replace("2", "2")
    atoms = Atoms(formula, cell=[5.0, 5.0, 5.0], pbc=True,
                  scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    return atoms * (z, 1, 1)


def _prescreen_db(path: Path) -> None:
    """Four survivors of Shuo's prescreening, two of which passed.

    `Gd1Co2_s003` is deliberately a two-formula-unit cell: MatterGen returned a
    mix of cell sizes for every formula it was asked for, and the adopter has to
    infer Z from the geometry rather than from the name.
    """
    db = connect(str(path))
    rows = [
        # (structure_id, composition, Z, mlip e_above_hull, passed prescreening)
        ("Gd1Co2_s001", "Gd1Co2", 1, 0.02, 1),
        ("Gd1Co2_s002", "Gd1Co2", 1, 0.30, 0),
        ("Gd1Co2_s003", "Gd1Co2", 2, 0.05, 1),
        # A second chemical system, so adoption has a split to get right.
        ("Gd1Fe2_s001", "Gd1Fe2", 1, 0.44, 0),
        # The column says it passed; the JSON below says it did not, because
        # this flow tightened its gate after the column was written.  Both
        # ternary campaigns are exactly this, ~20,000 structures each.
        ("Gd1Co2_s005", "Gd1Co2", 1, 0.08, 1),
        # Passed the gate, ran, and came back with an energy that cannot be one.
        ("Gd1Co2_s007", "Gd1Co2", 1, 0.03, 1),
    ]
    for sid, comp, z, e_hull, passed in rows:
        elements = "-".join(sorted({e for e in ("Gd", "Co", "Fe") if e in comp}))
        db.write(_cell(comp.replace("1", ""), z),
                 structure_id=sid, composition=comp, chemsys=elements,
                 status="valid", pearson_symbol="cF4", wps="4a",
                 e_above_hull=e_hull, e_mattersim=-6.5, passed_prescreening=passed,
                 space_group_number=225, symmetrized=1, density=8.9, dof=3)


def _flow(tmp_path: Path) -> LegacyFlow:
    root = tmp_path / "legacy_flow"
    jobs = root / "VASP_JOBS"
    jobs.mkdir(parents=True)

    _prescreen_db(jobs / "prescreening_structures.db")

    (jobs / "prescreening_stability.json").write_text(json.dumps({
        "summary": {"total_structures": 10, "unique_structures_processed": 5,
                    "passed_prescreening": 2, "hull_threshold": 0.06},
        "results": [
            {"structure_id": "Gd1Co2_s001", "energy_above_hull": 0.02,
             "passed_prescreening": True},
            {"structure_id": "Gd1Co2_s002", "energy_above_hull": 0.30,
             "passed_prescreening": False},
            {"structure_id": "Gd1Co2_s003", "energy_above_hull": 0.05,
             "passed_prescreening": True},
            {"structure_id": "Gd1Fe2_s001", "energy_above_hull": 0.44,
             "passed_prescreening": False},
            # 0.08 passed the old 0.10 gate and fails the 0.06 one.
            {"structure_id": "Gd1Co2_s005", "energy_above_hull": 0.08,
             "passed_prescreening": False},
        ],
    }))
    (jobs / "mp_vaspdft.json").write_text(json.dumps([
        {"chemsys": "Co", "composition": {"Co": 4.0}, "energy": -28.0,
         "entry_id": "mp-1-GGA", "mp_id": "mp-1"},
        {"chemsys": "Gd", "composition": {"Gd": 2.0}, "energy": -14.0,
         "entry_id": "mp-2-GGA", "mp_id": "mp-2"},
    ]))
    (jobs / "mp_mattersim.json").write_text(json.dumps([
        {"chemsys": "Co", "composition": {"Co": 4.0}, "energy": -27.5,
         "entry_id": "ms_mp-1", "mp_id": "mp-1"},
    ]))
    (jobs / "hull_comparison.json").write_text(json.dumps({
        "summary": {"total_analyzed": 2, "mae": 0.015, "correlation": 0.65,
                    "rmse": 0.024, "precision": 0.64, "recall": 1.0,
                    "n_outliers_filtered": 0},
    }))

    # Two structures reached DFT: one converged, one hit the ionic step limit.
    make_job(jobs / "Gd1Co2" / "Gd1Co2_s001" / "Relax", steps=58, converged=True,
             slurm="111")
    make_job(jobs / "Gd1Co2" / "Gd1Co2_s003" / "Relax", steps=100, nsw=100,
             converged=False, slurm="222")
    # VASP exited cleanly and reported +3489.05 eV/atom -- the electronic loop
    # diverged.  Modelled on Gd1Ni19_s013, the worst of the 43 real ones.
    diverged = make_job(jobs / "Gd1Co2" / "Gd1Co2_s007" / "Relax", steps=12,
                        converged=True, slurm="333")
    (diverged / "OSZICAR").write_text(_oszicar(12, e0=3489.0454 * 3))

    (jobs / "workflow.json").write_text(json.dumps({
        "config": {"output_dir": "VASP_JOBS"},
        "structures": {
            "Gd1Co2_s001": {"composition": "Gd1Co2", "chemsys": "Co-Gd",
                            "state": "RELAX_DONE", "relax_job_id": "111",
                            "relax_dir": "VASP_JOBS/Gd1Co2/Gd1Co2_s001/Relax"},
            "Gd1Co2_s003": {"composition": "Gd1Co2", "chemsys": "Co-Gd",
                            "state": "RELAX_TMOUT", "relax_job_id": "222",
                            "relax_dir": "VASP_JOBS/Gd1Co2/Gd1Co2_s003/Relax"},
            "Gd1Co2_s007": {"composition": "Gd1Co2", "chemsys": "Co-Gd",
                            "state": "RELAX_DONE", "relax_job_id": "333",
                            "relax_dir": "VASP_JOBS/Gd1Co2/Gd1Co2_s007/Relax"},
        },
    }))
    (jobs / "dft_stability_results.json").write_text(json.dumps({
        "summary": {"total_structures": 2},
        "results": [
            {"structure_id": "Gd1Co2_s001", "composition": "Gd1Co2",
             "chemsys": "Co-Gd", "vasp_energy_per_atom": -6.8,
             "energy_above_hull": 0.011, "is_stable": False,
             "decomposition": [{"formula": "Co", "fraction": 0.5,
                                "energy_per_atom": -7.0}]},
            {"structure_id": "Gd1Co2_s003", "composition": "Gd1Co2",
             "chemsys": "Co-Gd", "vasp_energy_per_atom": -6.7,
             "energy_above_hull": 0.090, "is_stable": False, "decomposition": []},
            # The published hull placement for the diverged run: arithmetic on
            # a number that was never a measurement.
            {"structure_id": "Gd1Co2_s007", "composition": "Gd1Co2",
             "chemsys": "Co-Gd", "vasp_energy_per_atom": 3489.0454,
             "energy_above_hull": 3495.22, "is_stable": False, "decomposition": []},
        ],
    }))

    gen = root / "mattergen_results" / "binary_csp_magnets"
    gen.mkdir(parents=True)
    (gen / "generation_summary.json").write_text(json.dumps({
        "total_compositions": 2, "total_structures_generated": 10,
        "compositions": [{"formula": "Gd1Co2", "structures_generated": 8},
                         {"formula": "Gd1Fe2", "structures_generated": 2}],
    }))

    (root / "candidates_w_proto_mag.csv").write_text(
        "structure_id,mattersim_e_hull,dft_e_hull,mattersim_energy_per_atom,"
        "vasp_energy_per_atom,spg_num,aflow_proto,aflow_anrl,pearson_symbol,"
        "match_type,total_mag,cell_volume,mag_per_vol\n"
        "Gd1Co2_s001,0.02,0.011,-6.5,-6.8,225,AB2,A2B_cF24_227_d_a-001,cF24,"
        "binary,12.5,180.0,0.0694\n"
    )

    flow = LegacyFlow(name="fixture", source_dir="legacy_flow",
                      groups={"RE": {"elements": ["Gd"], "pick": 1},
                              "TM": {"elements": ["Co", "Fe"], "pick": 1,
                                     "min_fraction": 0.75}},
                      description="synthetic")
    assert root.is_dir()
    return flow


@pytest.fixture()
def flow(tmp_path, monkeypatch):
    import cspflow.legacy as legacy

    f = _flow(tmp_path)
    monkeypatch.setattr(legacy, "SHUO", tmp_path)
    return f


SYSTEM = "Co-Gd"          # the fixture's main system; Fe-Gd holds one structure


@pytest.fixture()
def adopted(flow, tmp_path):
    stats = adopt(flow, tmp_path / "out")
    return stats, Store.open(tmp_path / "out" / "fixture" / SYSTEM / "campaign.db")


# --------------------------------------------------------------------------
# the reconstructed config
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_flow_renders_a_valid_campaign_file(name, tmp_path):
    """The adopted campaign.yaml has to load through the real loader.

    An adopted campaign that `csp status --campaign` cannot open is an archive,
    not a campaign, and the whole point of writing the file is that the settings
    travel with the numbers.
    """
    from cspflow.config.loader import load_campaign

    path = tmp_path / f"{name}.yaml"
    path.write_text(render_config(REGISTRY[name], tmp_path / name))
    cfg = load_campaign(path)
    assert cfg.campaign.name == name
    assert Path(cfg.campaign.workdir) == tmp_path / name


def test_registry_names_the_rare_earths_the_flow_actually_ran():
    """`new_bin_mag` is Gd AND Y, and `bin_mag_flow` is Sm AND Tb.

    Named from the directory they came from, two of the four campaigns would be
    labelled with half their chemistry -- which is how a collaborator ends up
    looking for Tb results in a campaign called `sm-binary`.
    """
    assert set(REGISTRY["SmTb_FeCoNi_binary"].groups["RE"]["elements"]) == {"Sm", "Tb"}
    assert set(REGISTRY["GdY_FeCoNi_binary"].groups["RE"]["elements"]) == {"Gd", "Y"}
    assert set(REGISTRY["GdY_FeCoNi_X_ternary"].groups["X"]["elements"]) == {
        "Ti", "V", "Cr", "Mn", "Cu", "Zn"}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_a_campaign_name_never_reads_as_a_chemical_system(name):
    """A hyphen means "these elements in one system"; a campaign is not one.

    `sm-tb-binary` names the Sm-Tb binary, of which that campaign contains
    exactly zero structures -- `max_rare_earth: 1`, so Sm and Tb never
    co-occur.  The name asserted the one system the campaign had none of.

    Groups are separated by `_`, elements of a system by `-`, and the two must
    never be confusable: the website's URLs, its directory names and the hull
    plotter all key on the hyphen form.
    """
    assert "-" not in name, name
    # The group count is the arity, and both are stated.
    flow = REGISTRY[name]
    arity = "ternary" if len(flow.groups) == 3 else "binary"
    assert name.endswith(arity), f"{name} has {len(flow.groups)} groups"


# --------------------------------------------------------------------------
# the shape of an adopted campaign
# --------------------------------------------------------------------------

def test_adopt_writes_a_family_of_one_campaign_per_chemical_system(flow, tmp_path):
    """The family is the flow; each campaign inside it is one system.

    The MLIP-vs-DFT agreement that decides whether a screen was worth running
    is a property of the system -- 0.03 to 0.97 inside a single flow -- and
    cspflow stores one calibration per campaign, so this is the granularity at
    which that verdict can be recorded at all.
    """
    adopt(flow, tmp_path / "out")
    family = tmp_path / "out" / "fixture"
    assert (family / "family.json").is_file()
    assert (family / "legacy" / "workflow.json").is_file()

    systems = sorted(p.name for p in family.iterdir() if (p / "campaign.db").is_file())
    assert systems == ["Co-Gd", "Fe-Gd"]

    campaign = family / SYSTEM
    assert (campaign / "campaign.yaml").is_file()
    assert (campaign / "campaign.db").is_file()
    assert (campaign / "dft").is_dir()
    assert (campaign / "report" / "candidates.csv").is_file()


def test_the_family_index_answers_which_systems_are_good_without_opening_them(flow, tmp_path):
    """"Which are good and which are bad" should not need 36 database opens."""
    adopt(flow, tmp_path / "out")
    index = json.loads((tmp_path / "out" / "fixture" / "family.json").read_text())
    assert index["family"] == "fixture"
    assert index["totals"]["n_systems"] == 2
    by_system = {s["chemsys"]: s for s in index["systems"]}
    assert set(by_system) == {"Co-Gd", "Fe-Gd"}
    assert by_system["Co-Gd"]["n_candidates"] == 1
    assert by_system["Fe-Gd"]["n_candidates"] == 0


def test_each_system_only_carries_its_own_candidates(flow, tmp_path):
    """The candidate CSV covers the whole flow; a campaign is one system of it.

    Copied wholesale, every system would report the flow's entire candidate
    table as its own.
    """
    adopt(flow, tmp_path / "out")
    family = tmp_path / "out" / "fixture"
    rows = (family / SYSTEM / "report" / "candidates.csv").read_text().splitlines()
    assert len(rows) == 2                       # header + the one Co-Gd candidate
    assert "Gd1Co2_s001" in rows[1]
    assert not (family / "Fe-Gd" / "report").exists()


def test_a_campaign_only_holds_reference_phases_its_own_hull_can_use(flow, tmp_path):
    """Fe-Gd's hull has no Co axis, so mp-1 (elemental Co) does not belong in it.

    Carried into every campaign, the reference sets would be identical while
    `ref_set_hash` claimed they described different hulls.
    """
    adopt(flow, tmp_path / "out")
    family = tmp_path / "out" / "fixture"
    for system, expected in (("Co-Gd", {"mp-1", "mp-2"}), ("Fe-Gd", {"mp-2"})):
        store = Store.open(family / system / "campaign.db")
        assert {r["mp_id"] for r in store.reference_entries()} == expected, system
        store.close()


def test_adopt_leaves_no_write_ahead_log_beside_the_database(flow, tmp_path):
    """A `-wal` that survives the move is how the adopted campaign fails to open.

    `Store._check_sidecars` refuses an orphaned WAL by design, and SQLite's own
    error underneath it is a bare 'disk I/O error'.  The sealing step exists
    only to make this assertion true.
    """
    adopt(flow, tmp_path / "out")
    campaign = tmp_path / "out" / "fixture" / SYSTEM
    assert not (campaign / "campaign.db-wal").exists()
    assert not (campaign / "campaign.db-shm").exists()
    Store.open(campaign / "campaign.db").close()          # must not raise


def test_adopt_refuses_to_write_over_an_existing_campaign(flow, tmp_path):
    adopt(flow, tmp_path / "out")
    with pytest.raises(LegacyError, match="already exists"):
        adopt(flow, tmp_path / "out")


def test_staging_produces_the_same_campaign_as_building_in_place(flow, tmp_path):
    direct = adopt(flow, tmp_path / "direct")
    staged = adopt(flow, tmp_path / "staged", staging=tmp_path / "stage")
    assert staged.structures == direct.structures
    assert staged.by_state == direct.by_state
    assert not (tmp_path / "stage" / "fixture").exists()   # moved, not copied


# --------------------------------------------------------------------------
# what the tables end up holding
# --------------------------------------------------------------------------

def test_the_gate_comes_from_the_file_that_decided_it_not_the_stale_column(adopted):
    """`prescreening_structures.db` records a gate the campaign later tightened.

    Both ternary flows cut at 0.06 eV/atom and left the database's
    `passed_prescreening` column describing the earlier 0.10 gate -- 26,074
    against 3,305, and 20,724 against 2,362.  The JSON agrees with
    `workflow.json`, i.e. with what was actually submitted; the column does not.

    Read from the column, those campaigns each gain ~20,000 structures marked
    as having passed a shortlist they never passed and then never run, which
    reads as a campaign that abandoned most of its own candidates.
    """
    _, store = adopted
    rows = {r.legacy_id: r for r in store.structures()}
    assert rows["Gd1Co2_s005"].state == "filtered_out"
    assert rows["Gd1Co2_s005"].filter_reason == "mlip e_above_hull > 0.06"
    events = store.filter_events(rows["Gd1Co2_s005"].id)
    gate = next(e for e in events if e["gate"] == "filter:e_above_hull")
    assert gate["passed"] == 0
    assert gate["threshold"] == pytest.approx(0.06)
    store.close()


def test_the_reconstructed_config_states_the_gate_the_campaign_used(tmp_path):
    """0.10 for the binaries, 0.06 for the ternaries -- not a shared constant."""
    text = render_config(REGISTRY["SmTb_FeCoNi_X_ternary"], tmp_path)
    assert "e_above_hull_max: 0.06" in text
    assert "e_above_hull_max: 0.1\n" not in text
    binary = render_config(REGISTRY["SmTb_FeCoNi_binary"], tmp_path)
    assert "e_above_hull_max: 0.1" in binary


def test_structures_carry_their_mlip_result_and_screening_verdict(adopted):
    _, store = adopted
    rows = {r.legacy_id: r for r in store.structures()}
    # Gd1Fe2_s001 is Fe-Gd and lives in its own campaign.
    assert set(rows) == {"Gd1Co2_s001", "Gd1Co2_s002", "Gd1Co2_s003",
                         "Gd1Co2_s005", "Gd1Co2_s007"}
    assert rows["Gd1Co2_s002"].state == "filtered_out"
    assert rows["Gd1Co2_s002"].mlip_e_above_hull == pytest.approx(0.30)
    assert rows["Gd1Co2_s001"].mlip_e_per_atom == pytest.approx(-6.5)
    store.close()


def test_compositions_are_split_by_z_because_the_generator_varied_the_cell(adopted):
    """`Gd1Co2` came back at one and at two formula units; those are two rows.

    Collapsed onto one, `n_produced` becomes the sum of two different things and
    every per-composition count downstream is wrong.  The schema already says
    so -- `composition` is UNIQUE(formula, z, source_name) -- and this is the
    assertion that the adopter respects it.
    """
    _, store = adopted
    rows = {(c.formula, c.z): c for c in store.compositions()}
    gdco2 = sorted(z for (f, z) in rows if f == "Co2Gd1")
    assert gdco2 == [1, 2], sorted(rows)
    assert rows[("Co2Gd1", 1)].n_produced == 4      # s001, s002, s005, s007
    assert rows[("Co2Gd1", 2)].n_produced == 1      # s003
    store.close()


def test_n_target_is_apportioned_so_the_campaign_totals_reconcile(adopted):
    """sum(n_target) is what the generator was asked for, not a multiple of it.

    The legacy flow asked once per formula and chose Z itself, so repeating the
    formula's ask on each Z row -- the obvious mapping -- reports a threefold
    shortfall that never happened.  Apportioned by production, `csp status`
    reports the dedup loss instead, which is the real number.
    """
    _, store = adopted
    total = sum(c.n_target for c in store.compositions())
    assert total == 8            # Gd1Co2's ask; Gd1Fe2's 2 belong to Fe-Gd
    store.close()


def test_deduped_away_structures_are_counted_but_never_invented(adopted):
    """10 generated, 4 with geometry.  The other 6 get no ASE row.

    Their cells were discarded before the database Shuo kept was written, and a
    fabricated one would be believed by every later stage.
    """
    stats, store = adopted
    # Dedup is a property of the flow, not of any one system, so the counts
    # live on the family and only the survivors live in a campaign.
    assert stats.generated_pre_dedup == 10
    assert stats.structures == 6                          # across both systems
    assert stats.dedup_removed == 4
    assert store.meta("legacy.unique") == "5"             # this system's share
    assert store.count_structures() == 5
    store.close()


def test_reference_entries_carry_both_energy_scales(adopted):
    """Per atom, both of them, and only where the MLIP actually ran.

    mp-2 has no MatterSim energy in the fixture, and must come back with
    `e_mlip_static` unset rather than zero -- 0.0 eV/atom is a plausible-looking
    number that would put a fictitious phase on the MLIP hull.
    """
    _, store = adopted
    entries = {r["mp_id"]: r for r in store.reference_entries()}
    assert entries["mp-1"]["e_dft_raw"] == pytest.approx(-7.0)
    assert entries["mp-1"]["e_mlip_static"] == pytest.approx(-6.875)
    assert entries["mp-2"]["e_mlip_static"] is None
    assert entries["mp-2"]["state"] == "fetched"
    store.close()


def test_a_run_that_hit_the_ionic_limit_is_done_but_not_converged(adopted):
    """`RELAX_TMOUT` in the manager, `state='done', converged=False` on disk.

    Both are recorded.  Reading only the manager's word loses the fact that the
    geometry is not relaxed; reading only the directory loses the fact that the
    scheduler killed it.
    """
    _, store = adopted
    row = next(r for r in store.structures() if r.legacy_id == "Gd1Co2_s003")
    assert row.state == "dft_done"
    assert row.converged is False
    assert row.legacy_job_state == "RELAX_TMOUT"
    store.close()


def test_dft_directories_point_at_the_legacy_tree_and_are_not_copied(adopted, flow):
    """549 GB of VASP output stays where it is; the row records where.

    `campaign_meta['legacy.root']` is the prefix, so a move needs one UPDATE
    rather than a re-adoption.
    """
    _, store = adopted
    row = next(r for r in store.structures() if r.legacy_id == "Gd1Co2_s001")
    assert Path(row.dft_dir).is_dir()
    assert str(flow.root) in row.dft_dir
    assert store.meta("legacy.root") == str(flow.root)
    store.close()


def test_both_hulls_are_recorded_against_different_reference_sets(adopted):
    """The MLIP hull and the DFT hull are separate placements, not one number.

    They were built against different reference energies, and `ref_set_hash` is
    what stops a later analysis from averaging them.
    """
    _, store = adopted
    rows = store.sql.execute(
        "SELECT hull_type, ref_set_hash, COUNT(*) n FROM hull GROUP BY hull_type"
    ).fetchall()
    by_type = {r["hull_type"]: r for r in rows}
    assert by_type["mlip"]["n"] == 5          # Co-Gd only; Fe-Gd is its own
    assert by_type["dft"]["n"] == 2
    assert by_type["mlip"]["ref_set_hash"] != by_type["dft"]["ref_set_hash"]
    store.close()


def test_the_decomposition_lands_in_the_data_blob_not_a_column(adopted):
    _, store = adopted
    row = next(r for r in store.structures() if r.legacy_id == "Gd1Co2_s001")
    assert row.data["decomposition"][0]["formula"] == "Co"
    store.close()


def test_candidate_properties_say_which_are_computed_and_which_are_modelled(adopted):
    """The moment is DFT; the AFLOW prototype is a library match.

    Stored with the same `source` they would have if the pipeline had produced
    them, so nothing downstream has to remember which is which.
    """
    stats, store = adopted
    assert stats.candidates == 1
    sid = next(r.id for r in store.structures() if r.legacy_id == "Gd1Co2_s001")
    props = {(r["key"], r["source"]): r for r in store.properties(sid)}
    assert props[("m_dft_raw", "dft")]["value"] == pytest.approx(12.5)
    assert props[("aflow_prototype", "model")]["text_value"] == "AB2"
    assert props[("dft_e_above_hull", "dft")]["value"] == pytest.approx(0.011)
    store.close()


def test_calibration_is_measured_on_this_system_not_copied_from_the_flow(adopted):
    """Each system scores itself, from its own MLIP/DFT pairs.

    `hull_comparison.json` reports one number for a whole flow, and the reason
    campaigns are split per system is that it is not one number: within a
    single real flow the rank correlation runs from 0.027 (`Gd-Ni`) to 0.965
    (`Cu-Fe-Sm`). Copying the flow-wide figure onto every system would put the
    same verdict on the screen that worked and the screen that did not.

    Here: MLIP 0.02/0.05 against DFT 0.011/0.090, so MAE is
    mean(0.009, 0.040) = 0.0245, and rho is undefined on two points.
    """
    _, store = adopted
    row = store.latest_calibration("mp")
    assert row["kind"] == "mp"
    assert row["n_points"] == 2
    assert row["mae_e_hull"] == pytest.approx(0.0245)
    assert row["spearman"] is None                        # two points is not a rank
    assert row["verdict"] == "warn"
    assert "this system" in row["detail"]
    store.close()


# --------------------------------------------------------------------------
# against the real thing
# --------------------------------------------------------------------------

@has_real
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_flow_has_the_six_artefacts_it_claims(name):
    flow = REGISTRY[name]
    assert flow.root.is_dir(), flow.root
    for rel in ("VASP_JOBS/prescreening_structures.db",
                "VASP_JOBS/prescreening_stability.json",
                "VASP_JOBS/dft_stability_results.json",
                "VASP_JOBS/mp_vaspdft.json",
                "VASP_JOBS/workflow.json",
                "candidates_w_proto_mag.csv"):
        assert (flow.root / rel).is_file(), f"{name}: {rel}"


@has_real
def test_a_slice_of_the_real_campaign_adopts(tmp_path):
    """200 structures of the real flow, to catch fixture drift.

    The fixture asserts behaviour; this asserts that the behaviour is being
    asserted about the right file formats.
    """
    stats = adopt(REGISTRY["SmTb_FeCoNi_binary"], tmp_path / "out", limit=50)
    family = tmp_path / "out" / "SmTb_FeCoNi_binary"
    systems = sorted(p.name for p in family.iterdir() if (p / "campaign.db").is_file())
    assert systems == ["Co-Sm", "Co-Tb", "Fe-Sm", "Fe-Tb", "Ni-Sm", "Ni-Tb"]
    # `limit` caps each system, so the family holds at most 50 per system.
    assert stats.structures == 50 * len(systems)
    store = Store.open(family / "Co-Sm" / "campaign.db")
    assert store.count_structures() == 50
    store.close()


def test_adoption_is_deterministic(flow, tmp_path):
    """Two runs over the same read-only sources produce the same database.

    Which is what makes the campaign on disk checkable: it can be rebuilt at
    any time and diffed, so a repair, a schema change or a corrupted copy is
    something you can detect rather than something you have to trust.
    Established for real against `SmTb_FeCoNi_binary` (all eleven tables identical,
    ASE geometry included); this is the cheap version that runs every time.
    """
    import hashlib
    import json
    import sqlite3

    def fingerprint(db: Path) -> dict[str, str]:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            queries = {
                "composition": "SELECT formula,z,n_target,n_produced,state FROM composition ORDER BY formula,z",
                "relaxation": "SELECT engine,energy,e_per_atom,converged,n_steps FROM relaxation ORDER BY structure_id,engine",
                "reference": "SELECT mp_id,formula,n_atoms,e_dft_raw,e_mlip_static,state FROM reference_entry ORDER BY mp_id",
                "hull": "SELECT hull_type,e_above_hull,ref_set_hash FROM hull ORDER BY structure_id,hull_type",
                "job": "SELECT stage,recipe_step,slurm_id,state,exit_reason FROM job ORDER BY workdir",
                "filter_event": "SELECT gate,passed,value,threshold FROM filter_event ORDER BY structure_id,gate",
                "property": "SELECT key,source,value,text_value FROM property ORDER BY structure_id,key,source",
                "kv": "SELECT key_value_pairs FROM systems ORDER BY id",
                "geometry": "SELECT numbers,positions,cell,pbc FROM systems ORDER BY id",
            }
            return {
                name: hashlib.sha256(
                    json.dumps([list(map(str, r)) for r in con.execute(q)]).encode()
                ).hexdigest()
                for name, q in queries.items()
            }
        finally:
            con.close()

    adopt(flow, tmp_path / "first")
    adopt(flow, tmp_path / "second")
    assert (fingerprint(tmp_path / "first" / "fixture" / SYSTEM / "campaign.db")
            == fingerprint(tmp_path / "second" / "fixture" / SYSTEM / "campaign.db"))


def test_a_diverged_scf_is_a_failure_and_never_an_energy(adopted):
    """+3489.05 eV/atom is not a result, and must not reach any numeric column.

    43 of the four campaigns' 12,193 DFT runs came back with a POSITIVE
    per-atom energy -- an electronic loop that diverged and was written down
    anyway. Every real value in this data runs between -6 and -14 eV/atom, so
    the sign alone settles it.

    Left in `vasp_energy` it enters the hull, and against reference phases near
    -8 eV/atom it produces a hull distance of 3495 that sorts, plots and
    averages exactly like a real one -- which is how it reached the website.
    """
    _, store = adopted
    row = next(r for r in store.structures() if r.legacy_id == "Gd1Co2_s007")

    assert row.state == "failed"
    assert "SCF diverged" in row.dft_fail_reason
    assert "+3489.05" in row.dft_fail_reason          # the value is kept, as text
    # ...and nowhere a number can be read from.
    assert "vasp_energy" not in row.key_value_pairs
    assert "e_per_atom" not in row.key_value_pairs
    assert "dft_e_above_hull" not in row.key_value_pairs

    placements = store.sql.execute(
        "SELECT COUNT(*) n FROM hull WHERE structure_id=? AND hull_type='dft'",
        (row.id,)).fetchone()["n"]
    assert placements == 0, "a diverged run has no hull placement to record"

    gate = next(e for e in store.filter_events(row.id)
                if e["gate"] == "dft:relax:converged")
    assert gate["passed"] == 0
    assert "diverged" in gate["detail"]
    store.close()


def test_a_diverged_run_is_kept_as_evidence_that_it_ran(adopted):
    """The core-hours were spent. The job row is where that stays recorded."""
    stats, store = adopted
    assert stats.diverged == 1
    row = next(r for r in store.structures() if r.legacy_id == "Gd1Co2_s007")
    job = store.sql.execute(
        "SELECT * FROM job WHERE structure_id=?", (row.id,)).fetchone()
    assert job["state"] == "failed"
    assert job["exit_reason"] == "scf_diverged"
    assert job["core_hours"] > 0
    store.close()


def test_a_diverged_run_is_excluded_from_the_calibration_it_would_dominate(adopted):
    """One pair at 3495 against two near 0.05 would set the MAE by itself."""
    _, store = adopted
    row = store.latest_calibration("mp")
    assert row["n_points"] == 2                   # s001 and s003, not s007
    assert row["mae_e_hull"] < 1.0
    store.close()
