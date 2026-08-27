"""Stage 5 -- the barrier between GPU-minutes and core-hours.

The two behaviours worth pinning are the order of the two cuts, and the fact
that a calibrated threshold that cannot be calibrated says so rather than
quietly becoming a literal one.
"""

import pytest
from ase.build import bulk

from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.stages import FilterStage

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
filter:
  e_above_hull_max: 0.10
  e_above_hull_max_source: {source}
  max_per_composition: 2
"""


def make_cfg(tmp_path, source="literal"):
    (tmp_path / "seeds").mkdir(exist_ok=True)
    path = tmp_path / f"campaign-{source}.yaml"
    path.write_text(CAMPAIGN.format(workdir=tmp_path, source=source))
    return load_campaign(path)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def add(store, atoms, e_hull):
    sid = store.add_structure(atoms, origin=Origin.generated,
                              state=StructureState.screened)
    store.add_hull(structure_id=sid, hull_type="mlip", energy_scale="raw",
                   e_above_hull=e_hull, ref_set_hash="ref")
    return sid


class TestHullCut:
    def test_candidates_below_the_threshold_are_selected(self, cfg, store):
        sid = add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        FilterStage(cfg).run(store)
        assert store.get_structure(sid).state == "selected"

    def test_candidates_above_it_are_filtered_out_with_a_reason(self, cfg, store):
        sid = add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.5)
        FilterStage(cfg).run(store)
        row = store.get_structure(sid)
        assert row.state == "filtered_out" and "hull threshold" in row.filter_reason

    def test_the_boundary_is_inclusive(self, cfg, store):
        sid = add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.10)
        FilterStage(cfg).run(store)
        assert store.get_structure(sid).state == "selected"

    def test_the_number_that_decided_it_is_recorded(self, cfg, store):
        """`csp status --why` has to be able to replay the decision."""
        sid = add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.5)
        FilterStage(cfg).run(store)
        event = next(e for e in store.filter_events(sid)
                     if e["gate"] == "filter:e_above_hull")
        assert event["value"] == pytest.approx(0.5)
        assert event["threshold"] == pytest.approx(0.10)
        assert not event["passed"]


class TestPerCompositionCap:
    def _three_of_one_formula(self, store):
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        return [add(store, fe.copy(), e) for e in (0.01, 0.02, 0.03)]

    def test_only_the_best_n_survive(self, cfg, store):
        ids = self._three_of_one_formula(store)
        FilterStage(cfg).run(store)         # max_per_composition is 2
        states = [store.get_structure(i).state for i in ids]
        assert states == ["selected", "selected", "filtered_out"]

    def test_the_cap_is_applied_after_the_hull_cut_not_before(self, cfg, store):
        """A composition whose candidates are all bad contributes none."""
        fe = bulk("Fe", "bcc", a=2.87, cubic=True)
        ids = [add(store, fe.copy(), e) for e in (0.5, 0.6, 0.7)]
        FilterStage(cfg).run(store)
        assert all(store.get_structure(i).state == "filtered_out" for i in ids)

    def test_different_compositions_get_their_own_allowance(self, cfg, store):
        fe = [add(store, bulk("Fe", "bcc", a=2.87, cubic=True), e) for e in (0.01, 0.02)]
        ni = [add(store, bulk("Ni", "fcc", a=3.52, cubic=True), e) for e in (0.01, 0.02)]
        FilterStage(cfg).run(store)
        assert all(store.get_structure(i).state == "selected" for i in fe + ni)

    def test_the_rank_that_capped_it_is_recorded(self, cfg, store):
        ids = self._three_of_one_formula(store)
        FilterStage(cfg).run(store)
        event = next(e for e in store.filter_events(ids[2])
                     if e["gate"] == "filter:per_composition")
        assert event["value"] == 3.0 and not event["passed"]


class TestThreshold:
    def test_a_literal_threshold_is_used_as_written(self, cfg, store):
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        stage = FilterStage(cfg)
        stage.run(store)
        assert stage.effective_threshold == pytest.approx(0.10)

    def test_calibrated_falls_back_loudly_when_there_is_no_fit(self, tmp_path, store):
        """A threshold that quietly stopped being calibrated changes results
        with no visible cause."""
        cfg = make_cfg(tmp_path, source="calibrated")
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        report = FilterStage(cfg).run(store)
        assert "no calibration fit available" in report.note

    def test_a_fit_converts_the_threshold_not_the_energies(self, tmp_path, store):
        cfg = make_cfg(tmp_path, source="calibrated")
        store.add_calibration(kind="mp", n_points=10, verdict="pass",
                              alpha=1.4, beta=0.0)
        sid = add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        stage = FilterStage(cfg)
        stage.run(store)
        assert stage.effective_threshold == pytest.approx(0.10 / 1.4, abs=1e-6)
        # The stored hull energy is untouched.
        row = store.sql.execute("SELECT e_above_hull FROM hull WHERE structure_id=?",
                                (sid,)).fetchone()
        assert row["e_above_hull"] == pytest.approx(0.05)

    def test_the_conversion_is_explained_in_the_report(self, tmp_path, store):
        cfg = make_cfg(tmp_path, source="calibrated")
        store.add_calibration(kind="mp", n_points=10, verdict="pass",
                              alpha=1.4, beta=0.0)
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        assert "calibrated:" in FilterStage(cfg).run(store).note

    def test_a_zero_alpha_falls_back_rather_than_dividing(self, tmp_path, store):
        cfg = make_cfg(tmp_path, source="calibrated")
        store.add_calibration(kind="mp", n_points=10, verdict="pass",
                              alpha=0.0, beta=0.0)
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        assert "alpha=0" in FilterStage(cfg).run(store).note


class TestPlumbing:
    def test_it_is_in_process(self, cfg):
        assert FilterStage(cfg).in_process is True

    def test_nothing_screened_means_nothing_to_do(self, cfg, store):
        assert FilterStage(cfg).run(store).note == "nothing screened to filter"

    def test_a_structure_with_no_hull_placement_is_not_a_candidate(self, cfg, store):
        store.add_structure(bulk("Fe", "bcc", a=2.87, cubic=True),
                            origin=Origin.generated, state=StructureState.screened)
        assert FilterStage(cfg).pending(store) == 0

    def test_pending_falls_to_zero(self, cfg, store):
        add(store, bulk("Fe", "bcc", a=2.87, cubic=True), 0.05)
        stage = FilterStage(cfg)
        assert stage.pending(store) == 1
        stage.run(store)
        assert stage.pending(store) == 0

    def test_it_sits_after_calibrate_in_the_funnel(self):
        from cspflow.driver import STAGE_ORDER

        assert STAGE_ORDER.index("calibrate") < STAGE_ORDER.index("filter")
        assert STAGE_ORDER.index("filter") < STAGE_ORDER.index("dft")
