"""Campaign database: ASE and our tables in one file, with write-time refusals."""

import sqlite3

import pytest
from ase.build import bulk

from cspflow.db.store import Origin, Store, StoreError, StructureState


@pytest.fixture()
def store(tmp_path):
    with Store.create(tmp_path / "campaign.db", campaign="t", config_hash="abc") as s:
        yield s


def test_both_halves_live_in_one_file(store):
    """The whole point of the ASE decision: one file, two schemas."""
    tables = {r[0] for r in store.sql.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"systems", "keys", "species"} <= tables          # ASE's
    assert {"composition", "job", "hull", "provenance"} <= tables  # ours


def test_ase_still_works_after_our_tables_exist(store):
    sid = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    assert store.count_structures() == 1
    assert store.get_structure(sid).origin == "generated"


def test_wal_mode(store):
    mode = store.sql.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# --- the ASE constraints we designed around --------------------------------


def test_none_is_refused_with_an_actionable_message(store):
    with pytest.raises(StoreError, match="cannot hold null"):
        store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated, e_mlip=None)


def test_containers_are_refused_and_point_at_the_data_blob(store):
    with pytest.raises(StoreError, match="data=. blob"):
        store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.seed, wyckoff=["4a"])


def test_structured_values_round_trip_through_the_data_blob(store):
    sid = store.add_structure(
        bulk("Fe", "bcc", a=2.87), origin=Origin.seed,
        data={"incar": {"ENCUT": 520}, "wyckoff": ["4a", "2b"]},
    )
    assert store.get_structure(sid).data["incar"]["ENCUT"] == 520


def test_pending_work_is_found_by_state_not_by_key_absence(store):
    """ASE cannot query for a missing key, so state is explicit everywhere.

    Regression guard for the failure this prevents: a driver loop that finds no
    work and merely looks idle.
    """
    a = bulk("Fe", "bcc", a=2.87)
    s1 = store.add_structure(a, origin=Origin.generated)                    # new
    s2 = store.add_structure(a, origin=Origin.generated, e_mlip=-8.2)
    store.set_structure_state(s2, StructureState.screened)

    assert store.structure_ids(state="new") == [s1]
    assert store.structure_ids(state="screened") == [s2]
    # and the thing that does NOT work, documented so nobody reintroduces it:
    assert store.structure_ids("~e_mlip") == []


def test_enum_values_are_accepted(store):
    sid = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.mp,
                              state=StructureState.screened)
    assert store.get_structure(sid).state == "screened"


# --- lifecycle -------------------------------------------------------------


def test_open_rejects_a_foreign_database(tmp_path):
    p = tmp_path / "other.db"
    sqlite3.connect(str(p)).execute("CREATE TABLE x (i INTEGER)")
    with pytest.raises(StoreError, match="not a cspflow database"):
        Store.open(p)


def test_open_rejects_a_schema_version_mismatch(tmp_path):
    p = tmp_path / "c.db"
    Store.create(p, campaign="t").close()
    Store(p).set_meta("schema_version", "999")
    with pytest.raises(StoreError, match="schema version 999"):
        Store.open(p)


def test_open_missing_file(tmp_path):
    with pytest.raises(StoreError, match="csp init"):
        Store.open(tmp_path / "nope.db")


# --- compositions ----------------------------------------------------------


def _comp(store, formula="SmFe11Ti", z=1, name="sweep"):
    return store.add_composition(
        formula=formula, chemsys="Fe-Sm-Ti", z=z, n_atoms=13, n_target=26,
        source_mode="chemical_space", source_name=name,
    )


def test_composition_upsert_is_idempotent(store):
    a = _comp(store)
    b = _comp(store)
    assert a == b
    assert len(store.compositions()) == 1


def test_same_formula_from_two_sources_is_two_rows(store):
    """Sources are pooled but never silently merged -- a seed reappearing as a
    generated candidate is a result, not a redundancy."""
    _comp(store, name="sweep")
    _comp(store, name="prototypes")
    assert len(store.compositions()) == 2


def test_composition_state_transitions(store):
    cid = _comp(store)
    store.set_composition_state(cid, "generated")
    assert store.compositions(state="generated")[0].id == cid


def test_bad_composition_state_is_rejected_by_the_schema(store):
    cid = _comp(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.set_composition_state(cid, "nonsense")


# --- hull integrity: the write-time refusals -------------------------------


def test_hull_refuses_mixed_energy_scales(store):
    a = bulk("Fe", "bcc", a=2.87)
    s1 = store.add_structure(a, origin=Origin.generated)
    s2 = store.add_structure(a, origin=Origin.generated)
    store.add_hull(structure_id=s1, hull_type="dft", energy_scale="raw",
                   e_above_hull=0.01, ref_set_hash="R1")
    store.add_hull(structure_id=s2, hull_type="dft", energy_scale="mp_corrected",
                   e_above_hull=0.02, ref_set_hash="R1")
    with pytest.raises(StoreError, match="mixes energy scales"):
        store.assert_hull_consistent("R1")


def test_hull_refuses_mixed_settings(store):
    a = bulk("Fe", "bcc", a=2.87)
    s1 = store.add_structure(a, origin=Origin.generated)
    s2 = store.add_structure(a, origin=Origin.generated)
    store.add_hull(structure_id=s1, hull_type="dft", energy_scale="raw",
                   e_above_hull=0.01, ref_set_hash="R1", settings_hash="aaa")
    store.add_hull(structure_id=s2, hull_type="dft", energy_scale="raw",
                   e_above_hull=0.02, ref_set_hash="R1", settings_hash="bbb")
    with pytest.raises(StoreError, match="mixes settings_hash"):
        store.assert_hull_consistent("R1")


def test_consistent_hull_passes(store):
    s = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    store.add_hull(structure_id=s, hull_type="mlip", energy_scale="raw",
                   e_above_hull=0.0, ref_set_hash="R1", settings_hash="aaa")
    store.assert_hull_consistent("R1")


def test_hull_upsert_updates_in_place(store):
    s = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    for e in (0.10, 0.05):
        store.add_hull(structure_id=s, hull_type="mlip", energy_scale="raw",
                       e_above_hull=e, ref_set_hash="R1")
    rows = list(store.sql.execute("SELECT e_above_hull FROM hull"))
    assert len(rows) == 1 and rows[0][0] == 0.05


def test_bad_hull_type_rejected(store):
    s = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_hull(structure_id=s, hull_type="guess", energy_scale="raw",
                       e_above_hull=0.0, ref_set_hash="R1")


# --- reference: MP's mixed-functional trap ---------------------------------


def test_reference_refuses_mixed_functionals(store):
    store.add_reference_entry(mp_id="mp-1", chemsys="Fe-Sm", thermo_type="GGA_GGA+U",
                              e_dft_raw=-7.1966)
    store.add_reference_entry(mp_id="mp-2", chemsys="Fe-Sm", thermo_type="R2SCAN",
                              e_dft_raw=-19.4095)
    with pytest.raises(StoreError, match="mixes functionals"):
        store.assert_single_thermo_type()


def test_single_functional_passes(store):
    store.add_reference_entry(mp_id="mp-1", chemsys="Fe-Sm", thermo_type="GGA_GGA+U")
    assert store.assert_single_thermo_type() == "GGA_GGA+U"


def test_reference_upsert_by_mp_id_and_snapshot(store):
    store.add_reference_entry(mp_id="mp-1", chemsys="Fe-Sm", thermo_type="GGA_GGA+U",
                              snapshot_id="2026-08-26", e_dft_raw=-7.0)
    store.add_reference_entry(mp_id="mp-1", chemsys="Fe-Sm", thermo_type="GGA_GGA+U",
                              snapshot_id="2026-08-26", e_mlip_static=-7.05)
    rows = store.reference_entries()
    assert len(rows) == 1 and rows[0]["e_mlip_static"] == -7.05


def test_two_snapshots_coexist(store):
    """A frozen snapshot and a refreshed one must not overwrite each other."""
    for snap in ("2026-08-26", "2026-11-01"):
        store.add_reference_entry(mp_id="mp-1", chemsys="Fe-Sm",
                                  thermo_type="GGA_GGA+U", snapshot_id=snap)
    assert len(store.reference_entries()) == 2


def test_unknown_reference_field_rejected(store):
    with pytest.raises(StoreError, match="unknown reference_entry field"):
        store.add_reference_entry(mp_id="mp-1", chemsys="X", thermo_type="GGA_GGA+U",
                                  e_dft_ray=-7.0)


# --- jobs ------------------------------------------------------------------


def test_job_lifecycle(store):
    jid = store.add_job(stage="dft", structure_id=1, recipe_step="relax")
    store.update_job(jid, state="queued", slurm_id="12345")
    store.update_job(jid, state="running")
    store.update_job(jid, state="done", core_hours=12.5)
    row = store.jobs(state="done")[0]
    assert row["slurm_id"] == "12345" and row["core_hours"] == 12.5


def test_unknown_job_field_rejected(store):
    jid = store.add_job(stage="dft")
    with pytest.raises(StoreError, match="unknown job field"):
        store.update_job(jid, statee="done")


def test_bad_job_state_rejected(store):
    jid = store.add_job(stage="dft")
    with pytest.raises(sqlite3.IntegrityError):
        store.update_job(jid, state="probably-fine")


def test_job_counts_by_state(store):
    for st in ("done", "done", "failed"):
        store.update_job(store.add_job(stage="dft"), state=st)
    assert store.count_jobs_by_state("dft") == {"done": 2, "failed": 1}


# --- provenance, filters, properties, calibration --------------------------


def test_provenance_is_deduplicated(store):
    a = store.add_provenance(config_hash="h1", settings_hash="s1")
    b = store.add_provenance(config_hash="h1", settings_hash="s1")
    assert a == b


def test_filter_events_replay_a_decision(store):
    s = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    store.add_filter_event(structure_id=s, gate="e_above_hull", passed=False,
                           value=0.31, threshold=0.10)
    ev = store.filter_events(s)[0]
    assert ev["passed"] == 0 and ev["value"] == 0.31


def test_both_moments_stored_side_by_side(store):
    """m_dft_raw and m_s_reconstructed never collapse into one column."""
    s = store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    store.add_property(structure_id=s, key="m_dft_raw", value=24.1, source="dft")
    store.add_property(structure_id=s, key="m_s_reconstructed", value=31.5, source="model")
    props = {r["key"]: (r["value"], r["source"]) for r in store.properties(s)}
    assert props["m_dft_raw"] == (24.1, "dft")
    assert props["m_s_reconstructed"] == (31.5, "model")


def test_calibration_records_alpha_beta(store):
    store.add_calibration(kind="mp", n_points=41, verdict="pass",
                          mae_e_per_atom=0.03, spearman=0.97, alpha=1.4, beta=0.01)
    row = store.latest_calibration("mp")
    assert row["alpha"] == 1.4 and row["verdict"] == "pass"


def test_bad_calibration_verdict_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_calibration(kind="mp", n_points=1, verdict="probably")


def test_summary(store):
    _comp(store)
    store.add_structure(bulk("Fe", "bcc", a=2.87), origin=Origin.generated)
    s = store.summary()
    assert s["compositions"] == 1 and s["structures"] == 1
    assert s["structures_by_state"] == {"new": 1}
