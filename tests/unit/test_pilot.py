"""Stage 4b -- the calibration that costs DFT, and the barrier it guards.

Two things are being tested and only one of them is arithmetic. The other is
that the gate is closed by default and opens for the right reason: 4b exists to
stop a campaign spending its allocation on a model nobody checked, and a gate
that quietly stays open is worse than no gate, because it looks like one.
"""

import pytest
from ase import Atoms

from cspflow.calibrate.parity import ParityPoint
from cspflow.calibrate.pilot import (PilotError, build_pilot_report,
                                     stratified_sample, top_n_agreement)
from cspflow.config.loader import load_campaign
from cspflow.db.store import Origin, Store, StructureState
from cspflow.stages.calibrate_stage import PILOT_KEY, CalibrateStage
from cspflow.stages.dft_stage import DftStage

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/seeds"]}}
calibrate:
  pilot: {{on_fail: {policy}, pilot_n: {n}}}
dft:
  recipe: magnets
  magnetism: {{mode: ferrimagnetic_retm}}
"""


@pytest.fixture
def make_cfg(tmp_path):
    def build(policy="block", n=4):
        (tmp_path / "seeds").mkdir(exist_ok=True)
        path = tmp_path / f"c-{policy}-{n}.yaml"
        path.write_text(CAMPAIGN.format(workdir=tmp_path, policy=policy, n=n))
        return load_campaign(path)
    return build


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def add(store, hull, state=StructureState.deduped, **kv):
    atoms = Atoms("Fe2Sm", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    return store.add_structure(atoms, origin=Origin.generated, state=state,
                               mlip_e_per_atom=-7.0 - hull,
                               e_above_hull_mlip=hull, **kv)


# -- choosing the pilot ----------------------------------------------------

def test_the_pilot_spans_the_range_rather_than_taking_the_top():
    """A pilot drawn from the best-ranked candidates measures the model only
    where it already ranks things highly. The failure that matters is a
    structure the MLIP puts low and DFT puts high, and such a pilot cannot see
    one by construction."""
    labels = [f"s{i}" for i in range(20)]
    values = [i * 0.05 for i in range(20)]
    chosen = stratified_sample(labels, values, 5)
    assert chosen[0] == "s0" and chosen[-1] == "s19"
    assert chosen != labels[:5]


def test_the_pilot_is_deterministic():
    labels = [f"s{i}" for i in range(30)]
    values = [(i * 7) % 30 * 0.01 for i in range(30)]
    assert stratified_sample(labels, values, 6) == stratified_sample(labels, values, 6)


def test_asking_for_more_than_exists_takes_everything():
    labels, values = ["a", "b", "c"], [0.1, 0.2, 0.3]
    assert sorted(stratified_sample(labels, values, 50)) == ["a", "b", "c"]


def test_asking_for_n_returns_n():
    labels = [f"s{i}" for i in range(50)]
    values = [i * 0.01 for i in range(50)]
    for n in (1, 2, 7, 13, 49, 50):
        assert len(stratified_sample(labels, values, n)) == n


@pytest.mark.parametrize("bad", [0, -3])
def test_a_nonsense_pilot_size_is_refused(bad):
    with pytest.raises(PilotError):
        stratified_sample(["a"], [1.0], bad)


# -- ranking ---------------------------------------------------------------

def points(pairs):
    return [ParityPoint(label=label, counts={"Fe": 2, "Sm": 1},
                        e_mlip_per_atom=mlip, e_dft_per_atom=dft)
            for label, mlip, dft in pairs]


def test_perfect_ranking_recovers_everything():
    overlap, missed = top_n_agreement(points([("a", 1, 1), ("b", 2, 2), ("c", 3, 3)]), 2)
    assert overlap == 2 and missed == []


def test_a_reversed_ranking_recovers_nothing():
    overlap, missed = top_n_agreement(points([("a", 3, 1), ("b", 2, 2), ("c", 1, 3)]), 1)
    assert overlap == 0 and missed == ["a"]


def test_ranking_uses_the_hull_when_both_sides_have_one():
    pts = [ParityPoint(label="a", counts={"Fe": 1}, e_mlip_per_atom=-1, e_dft_per_atom=-1,
                       e_hull_mlip=0.5, e_hull_dft=0.5),
           ParityPoint(label="b", counts={"Fe": 1}, e_mlip_per_atom=-9, e_dft_per_atom=-9,
                       e_hull_mlip=0.0, e_hull_dft=0.0)]
    overlap, _ = top_n_agreement(pts, 1)
    assert overlap == 1


# -- the verdict -----------------------------------------------------------

def test_good_energies_with_bad_selection_still_fail():
    """A model can have an excellent MAE and select the wrong structures, and
    selection is the only thing the MLIP is being asked to do."""
    pairs = [(f"s{i}", -7.0 + 0.001 * i, -7.0 - 0.001 * i) for i in range(12)]
    report = build_pilot_report(points(pairs), mae_max=0.05, spearman_min=-1.0,
                                top_n=3, top_n_min_fraction=0.6)
    assert report.parity.mae_e_per_atom < 0.05
    assert report.verdict == "fail"
    assert any("selection" in r for r in report.reasons)


def test_agreement_on_both_passes():
    pairs = [(f"s{i}", -7.0 + 0.01 * i, -7.0 + 0.01 * i) for i in range(12)]
    report = build_pilot_report(points(pairs), top_n=3)
    assert report.verdict == "pass"
    assert report.top_n_overlap == 3


def test_no_points_is_a_failure_not_a_pass():
    assert build_pilot_report([]).verdict == "fail"


def test_the_report_shows_what_was_missed():
    pairs = [(f"s{i}", -7.0 + 0.001 * i, -7.0 - 0.001 * i) for i in range(8)]
    rendered = build_pilot_report(points(pairs), top_n=2, top_n_min_fraction=0.9).render()
    assert "recovered by the MLIP" in rendered and "missed:" in rendered


# -- the stage -------------------------------------------------------------

def test_the_stage_selects_a_pilot_set_and_sends_it_to_dft(make_cfg, store):
    for i in range(20):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=5)
    note = CalibrateStage(cfg).run(store).note
    assert "pilot set of 5" in note
    members = [r for r in store.structures() if r.key_value_pairs.get(PILOT_KEY)]
    assert len(members) == 5
    assert all(r.key_value_pairs["state"] == StructureState.selected.value
               for r in members)


def test_the_stage_waits_rather_than_judging_a_half_finished_pilot(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=4)
    stage = CalibrateStage(cfg)
    stage.run(store)
    members = [r for r in store.structures() if r.key_value_pairs.get(PILOT_KEY)]
    store.set_structure_state(int(members[0].id), StructureState.dft_done,
                              e_per_atom=-7.1)
    assert "waiting on 3 of 4" in stage.run(store).note


def test_the_stage_judges_a_finished_pilot(make_cfg, store):
    for i in range(12):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=4)
    stage = CalibrateStage(cfg)
    stage.run(store)
    for row in [r for r in store.structures() if r.key_value_pairs.get(PILOT_KEY)]:
        mlip = row.key_value_pairs["mlip_e_per_atom"]
        store.set_structure_state(int(row.id), StructureState.dft_done,
                                  e_per_atom=mlip + 0.01)
    note = stage.run(store).note
    assert "PASS" in note or "WARN" in note
    assert store.latest_calibration("pilot") is not None


def test_a_disabled_pilot_selects_nothing(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    CalibrateStage(make_cfg(policy="off")).run(store)
    assert not [r for r in store.structures() if r.key_value_pairs.get(PILOT_KEY)]


# -- the barrier -----------------------------------------------------------

def test_the_expensive_tier_is_held_until_the_pilot_reports(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=3)
    CalibrateStage(cfg).run(store)
    # Everything else is now selected for DFT too.
    for row in store.structures(state=StructureState.deduped.value):
        store.set_structure_state(int(row.id), StructureState.selected)

    dft = DftStage(cfg)
    ready = dft._ready(store)
    assert len(ready) == 3
    assert all(r.key_value_pairs.get(PILOT_KEY) for r in ready)


def test_the_barrier_opens_when_the_pilot_passes(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=3)
    stage = CalibrateStage(cfg)
    stage.run(store)
    for row in [r for r in store.structures() if r.key_value_pairs.get(PILOT_KEY)]:
        store.set_structure_state(int(row.id), StructureState.dft_done,
                                  e_per_atom=row.key_value_pairs["mlip_e_per_atom"])
    stage.run(store)
    for row in store.structures(state=StructureState.deduped.value):
        store.set_structure_state(int(row.id), StructureState.selected)
    assert len(DftStage(cfg)._ready(store)) > 3


def test_a_failed_pilot_keeps_the_barrier_closed(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    cfg = make_cfg(n=3)
    store.add_calibration(kind="pilot", n_points=3, verdict="fail",
                          detail="the MLIP recovers none of DFT's best")
    for row in store.structures(state=StructureState.deduped.value):
        store.set_structure_state(int(row.id), StructureState.selected)
    assert DftStage(cfg)._ready(store) == []


def test_a_campaign_with_nothing_to_calibrate_is_not_gated(make_cfg, store):
    """The barrier stops spending on an unchecked model, not a campaign that
    has nothing to check it with."""
    cfg = make_cfg()
    for i in range(3):
        sid = add(store, hull=0.1, state=StructureState.selected)
    assert len(DftStage(cfg)._ready(store)) == 3


def test_warn_reports_without_blocking(make_cfg, store):
    for i in range(10):
        add(store, hull=i * 0.02)
    cfg = make_cfg(policy="warn", n=3)
    CalibrateStage(cfg).run(store)
    for row in store.structures(state=StructureState.deduped.value):
        store.set_structure_state(int(row.id), StructureState.selected)
    assert len(DftStage(cfg)._ready(store)) > 3
