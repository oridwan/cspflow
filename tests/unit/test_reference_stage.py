"""Stage 3 as a driver stage.

Offline throughout: the MP fetch is served from a cache file written by the
test, so the stage's own logic -- what it stores, what it places, and what it
keeps separate -- is exercised without a key or a network.
"""

import json

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.reference.mp import THERMO_GGA, snapshot_id
from cspflow.stages import ReferenceStage

pytest.importorskip("pymatgen.analysis.phase_diagram", reason="hulls need pymatgen")

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
reference:
  thermo_type: "GGA_GGA+U"
  energy_scale: raw
"""


def ref_row(mp_id, counts, raw):
    return {
        "mp_id": mp_id,
        "formula": "".join(f"{e}{counts[e]}" for e in sorted(counts)),
        "chemsys": "-".join(sorted(counts)),
        "counts": counts,
        "n_atoms": sum(counts.values()),
        "thermo_type": THERMO_GGA,
        "e_raw_per_atom": raw,
        "e_corrected_per_atom": raw,
        "e_above_hull_mp": None,
        "run_type": THERMO_GGA,
    }


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A campaign whose reference cache is pre-populated for Fe-Sm."""
    (tmp_path / "seeds").mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("CSPFLOW_CACHE", str(cache))

    rows = [ref_row("mp-13", {"Fe": 1}, 0.0),
            ref_row("mp-69", {"Sm": 1}, 0.0),
            ref_row("mp-1729", {"Fe": 2, "Sm": 1}, -0.5)]
    from cspflow.reference.mp import ReferenceEntry

    entries = [ReferenceEntry(**r) for r in rows]
    (cache / f"Fe-Sm__{THERMO_GGA.replace('+', 'p')}.json").write_text(json.dumps({
        "chemsys": "Fe-Sm", "thermo_type": THERMO_GGA,
        "snapshot_id": snapshot_id(entries), "fetched_at": "2026-08-27T00:00:00",
        "warnings": [], "entries": rows,
    }))

    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def screened(tmp_path):
    """Two screened Fe-Sm candidates with MLIP energies, one good and one not."""
    store = Store.create(tmp_path / "c.db", campaign="t")
    store.add_composition(formula="Fe2Sm1", chemsys="Fe-Sm", z=1, n_atoms=3,
                          n_target=0, source_mode="structure_list", source_name="s",
                          state="generated")
    good = bulk("Fe", "bcc", a=2.87, cubic=True) * (1, 1, 3)   # 6 atoms
    good.symbols = ["Sm", "Sm", "Fe", "Fe", "Fe", "Fe"]        # Fe2Sm1
    bad = good.copy()
    store.add_structure(good, origin=Origin.seed, state=StructureState.screened,
                        mlip_e_per_atom=-0.6, mlip_converged=True)
    store.add_structure(bad, origin=Origin.seed, state=StructureState.screened,
                        mlip_e_per_atom=-0.2, mlip_converged=True)
    yield store
    store.close()


class TestReferenceStage:
    def test_it_is_in_process(self, cfg):
        assert ReferenceStage(cfg).in_process is True

    def test_it_stores_the_reference_entries(self, cfg, screened):
        ReferenceStage(cfg).run(screened)
        rows = screened.reference_entries(chemsys="Fe-Sm", include_subsystems=True)
        assert {r["mp_id"] for r in rows} == {"mp-13", "mp-69", "mp-1729"}

    def test_each_entry_is_stored_under_its_own_chemical_system(self, cfg, screened):
        """Elemental Fe is 'Fe', not 'Fe-Sm' -- which is why hulls need subsystems."""
        ReferenceStage(cfg).run(screened)
        exact = screened.reference_entries(chemsys="Fe-Sm")
        assert {r["mp_id"] for r in exact} == {"mp-1729"}
        assert not [r for r in exact if r["mp_id"] in {"mp-13", "mp-69"}]

    def test_both_energy_scales_are_kept(self, cfg, screened):
        """Neither is preferred at storage time; the choice is made at the hull."""
        ReferenceStage(cfg).run(screened)
        row = next(r for r in screened.reference_entries() if r["mp_id"] == "mp-1729")
        assert row["e_dft_raw"] == pytest.approx(-0.5)
        assert row["e_dft_corrected"] == pytest.approx(-0.5)
        assert row["correction"] == pytest.approx(0.0)

    def test_candidates_are_placed_on_the_hull(self, cfg, screened):
        report = ReferenceStage(cfg).run(screened)
        assert report.reconciled == 2
        rows = list(screened.sql.execute("SELECT * FROM hull ORDER BY e_above_hull"))
        assert len(rows) == 2

    def test_the_better_candidate_sits_lower(self, cfg, screened):
        ReferenceStage(cfg).run(screened)
        rows = list(screened.sql.execute(
            "SELECT structure_id, e_above_hull FROM hull ORDER BY e_above_hull"))
        assert rows[0]["structure_id"] == 1 and rows[1]["structure_id"] == 2
        assert rows[0]["e_above_hull"] < rows[1]["e_above_hull"]

    def test_the_placement_is_labelled_mlip_not_dft(self, cfg, screened):
        """An MLIP hull must never be mistaken for a DFT one."""
        ReferenceStage(cfg).run(screened)
        assert {r["hull_type"] for r in screened.sql.execute("SELECT hull_type FROM hull")} == {"mlip"}

    def test_the_snapshot_id_is_recorded_as_the_reference_hash(self, cfg, screened):
        stage = ReferenceStage(cfg)
        stage.run(screened)
        hashes = {r["ref_set_hash"] for r in screened.sql.execute("SELECT ref_set_hash FROM hull")}
        assert hashes == {stage.last_snapshots["Fe-Sm"]}

    def test_e_above_hull_is_written_back_to_the_structure(self, cfg, screened):
        ReferenceStage(cfg).run(screened)
        assert screened.get_structure(1).e_above_hull_mlip is not None

    def test_running_twice_does_not_duplicate_rows(self, cfg, screened):
        stage = ReferenceStage(cfg)
        stage.run(screened)
        first = len(screened.reference_entries())
        stage.run(screened)
        assert len(screened.reference_entries()) == first
        assert screened.sql.execute("SELECT COUNT(*) n FROM hull").fetchone()["n"] == 2

    def test_nothing_screened_means_nothing_pending(self, cfg, tmp_path):
        with Store.create(tmp_path / "empty.db", campaign="t") as store:
            assert ReferenceStage(cfg).pending(store) == 0

    def test_pending_falls_to_zero_once_everything_is_placed(self, cfg, screened):
        stage = ReferenceStage(cfg)
        assert stage.pending(screened) == 2
        stage.run(screened)
        assert stage.pending(screened) == 0

    def test_the_correction_audit_is_reported(self, cfg, screened):
        """Fe-Sm is correction-free, and the stage says so rather than staying silent."""
        assert "correction-free" in ReferenceStage(cfg).run(screened).note
