"""The shipped examples must load, expand, and mean what they claim.

An example that stops matching its own README is worse than no example: it is
read as documentation and believed. These tests run the same code path
`csp source --dry-run` runs, so a schema change that invalidates an example
fails here rather than in a user's first campaign.
"""

import os
from pathlib import Path

import pytest

from cspflow.config.loader import load_campaign
from cspflow.source import expand_all

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str):
    """Load an example the way the CLI does, with $USER guaranteed."""
    path = EXAMPLES / name / "campaign.yaml"
    env = dict(os.environ)
    env.setdefault("USER", "tester")
    cfg = load_campaign(path, env=env)
    return cfg, expand_all(cfg.campaign, cfg.base_dir)


def test_all_three_modes_have_an_example():
    assert {p.name for p in EXAMPLES.iterdir() if p.is_dir()} == {
        "1-chemical-space", "2-composition-list", "3-structure-list"}


@pytest.mark.parametrize("name", ["1-chemical-space", "2-composition-list",
                                  "3-structure-list"])
def test_example_loads_and_expands(name):
    cfg, plan = _load(name)
    assert cfg.campaign.name.startswith("example-")
    assert plan.render()          # the same summary --dry-run prints


def test_chemical_space_example_sweeps_what_it_says():
    """The header claims 36 systems and ~165k structures. Hold it to that."""
    _, plan = _load("1-chemical-space")
    assert len(plan.compositions) == 3384
    assert len(plan.chemsystems()) == 36
    assert plan.n_target_total == 165_456


def test_composition_list_example_reads_both_the_items_and_the_csv():
    """34 = 4 inline + 13 CSV rows, expanded over Z, with one collision."""
    _, plan = _load("2-composition-list")
    assert len(plan.compositions) == 34
    formulas = {c.formula for c in plan.compositions}   # canonical, reduced
    assert "Fe29Sm3Ti2" in formulas          # Sm3Fe29Ti2, inline only
    assert "Fe17Sm2" in formulas             # Sm2Fe17, CSV only
    assert {"Fe1", "Sm1", "Ti1"} <= formulas, "the CSV's hull anchors"
    warnings = [w for r in plan.results for w in r.warnings]
    assert any("SmFe11Ti" in w and "appears twice" in w for w in warnings), \
        "the example deliberately overrides one CSV row from the items block"


def test_structure_list_example_reads_the_real_seeds():
    """Five POSCARs, parsed by both readers, entering the funnel at screen."""
    cfg, plan = _load("3-structure-list")
    assert cfg.campaign.generate is None, "a seeds-only campaign has no generate block"
    assert len(plan.structures) == 5
    assert plan.compositions == []
    assert all(r.entry_stage == "screen" for r in plan.results)
    seeds = EXAMPLES / "3-structure-list" / "inputs" / "seeds"
    assert len(list(seeds.glob("*.vasp"))) == 5


def test_the_biggest_seed_clears_the_max_atoms_gate():
    """Sm2Fe17 is 57 atoms; an example whose own seed is refused is a trap."""
    cfg, plan = _load("3-structure-list")
    cap = cfg.campaign.source[0].structure_list.max_atoms
    assert cap >= 57
    assert max(s.n_atoms for s in plan.structures) == 57
    assert len(plan.structures) == 5
