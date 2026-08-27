"""Deduplication of screened structures.

The two behaviours worth pinning are the asymmetry between generated structures
and seeds, and the fact that the survivor of a group is chosen by energy rather
than by database order.
"""

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.stages import DedupStage

pytest.importorskip("pymatgen.analysis.structure_matcher", reason="dedup needs pymatgen")

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
screen:
  dedup:
    matcher: {{ltol: 0.2, stol: 0.3, angle_tol: 5.0}}
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "seeds").mkdir()
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path))
    return load_campaign(path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def add(store, atoms, energy, origin=Origin.generated):
    return store.add_structure(atoms, origin=origin, state=StructureState.screened,
                               mlip_e_per_atom=energy)


class TestDedup:
    def test_identical_structures_collapse_to_one(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        keep = add(store, fe, -8.5)
        drop = add(store, fe.copy(), -8.4)
        report = DedupStage(cfg).run(store)
        assert report.claimed == 1
        assert store.get_structure(keep).state == "screened"
        assert store.get_structure(drop).state == "deduped"

    def test_the_survivor_is_the_lowest_energy_not_the_first_row(self, cfg, store):
        """Otherwise which structure survives depends on row order."""
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        first = add(store, fe, -8.0)
        second = add(store, fe.copy(), -8.9)
        DedupStage(cfg).run(store)
        assert store.get_structure(second).state == "screened"
        assert store.get_structure(first).state == "deduped"

    def test_the_duplicate_points_at_its_survivor(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        keep = add(store, fe, -8.9)
        drop = add(store, fe.copy(), -8.0)
        DedupStage(cfg).run(store)
        assert store.get_structure(drop).duplicate_of == keep

    def test_different_structures_both_survive(self, cfg, store):
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), -8.5)
        add(store, bulk("Ni", "fcc", a=3.52, cubic=True), -5.7)
        assert DedupStage(cfg).run(store).claimed == 0

    def test_different_polymorphs_of_one_element_both_survive(self, cfg, store):
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 1, 1), -8.5)
        add(store, bulk("Fe", "fcc", a=3.6, cubic=True), -8.2)
        assert DedupStage(cfg).run(store).claimed == 0

    def test_a_dedup_event_is_recorded_for_replay(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        add(store, fe, -8.9)
        drop = add(store, fe.copy(), -8.0)
        DedupStage(cfg).run(store)
        gates = [e["gate"] for e in store.filter_events(drop)]
        assert "dedup" in gates

    def test_seeds_are_reported_and_kept_by_default(self, cfg, store):
        """A relaxed and an unrelaxed copy of one prototype is a result, not noise."""
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        keep = add(store, fe, -8.9, origin=Origin.seed)
        other = add(store, fe.copy(), -8.0, origin=Origin.seed)
        report = DedupStage(cfg).run(store)
        assert report.claimed == 0
        assert store.get_structure(other).state == "screened"
        assert "seed collision" in report.note
        assert any(e["gate"] == "dedup:seed_collision" for e in store.filter_events(other))

    def test_seeds_can_be_deduplicated_when_asked(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        add(store, fe, -8.9, origin=Origin.seed)
        other = add(store, fe.copy(), -8.0, origin=Origin.seed)
        DedupStage(cfg, include_seeds=True).run(store)
        assert store.get_structure(other).state == "deduped"

    def test_running_twice_does_not_re_drop(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        add(store, fe, -8.9)
        add(store, fe.copy(), -8.0)
        stage = DedupStage(cfg)
        assert stage.run(store).claimed == 1
        assert stage.run(store).claimed == 0

    def test_pending_falls_to_zero(self, cfg, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        add(store, fe, -8.9)
        add(store, fe.copy(), -8.0)
        stage = DedupStage(cfg)
        assert stage.pending(store) == 2
        stage.run(store)
        assert stage.pending(store) == 1        # the survivor stays comparable

    def test_nothing_to_compare(self, cfg, store):
        assert DedupStage(cfg).run(store).note == "nothing to compare"

    def test_it_sits_between_screen_and_reference_in_the_funnel(self):
        """It must run before the hull: duplicates change every count downstream."""
        from cspflow.driver import STAGE_ORDER

        assert STAGE_ORDER.index("screen") < STAGE_ORDER.index("dedup")
        assert STAGE_ORDER.index("dedup") < STAGE_ORDER.index("reference")
