"""Stage 1 -- the generator interface, and the arithmetic around MatterGen.

None of this needs a GPU or a checkpoint. The parts that would -- sampling --
are behind a subprocess boundary, and everything on this side of it is decisions
about how much to ask for, how to split it, and what to believe about what came
back. Those are the parts that were wrong in the legacy campaign, so those are
the parts under test.
"""

import json
import math
from pathlib import Path

import pytest
from ase import Atoms

from cspflow.generators import (GenerationRequest, GeneratorError, MatterGenEngine,
                                check_composition, legacy_batch_total, read_generated,
                                split_batches)
from cspflow.generators.mattergen_engine import _hydra_config_name


# -- how much to ask for ---------------------------------------------------

@pytest.mark.parametrize("target,cap", [(1, 100), (54, 100), (100, 100), (101, 100),
                                        (108, 100), (162, 100), (200, 100), (254, 100),
                                        (999, 100), (7, 3), (12, 5)])
def test_a_split_sums_to_exactly_the_target(target, cap):
    parts = split_batches(target, cap)
    assert sum(parts) == target
    assert all(0 < p <= cap for p in parts)
    assert len(parts) == math.ceil(target / cap)


def test_split_parts_differ_by_at_most_one():
    """Which is what keeps a request to two subprocess calls, not many."""
    for target in range(1, 400):
        parts = split_batches(target, 100)
        assert max(parts) - min(parts) <= 1


def test_the_legacy_split_inflates_and_this_one_does_not():
    """The measurement that motivated `split_batches`.

    The legacy wrapper computed `num_batches = ceil(target/cap)` and then
    generated `cap * num_batches`. For the campaign's own distribution of
    requests that is 49.9% more structures than were asked for.
    """
    assert legacy_batch_total(108, 100) == 200
    assert sum(split_batches(108, 100)) == 108
    assert legacy_batch_total(100, 100) == 100      # unchanged below the cap
    assert legacy_batch_total(1, 100) == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsense_target_is_refused(bad):
    with pytest.raises(GeneratorError):
        split_batches(bad, 100)


def test_a_request_must_be_positive_and_whole():
    with pytest.raises(GeneratorError):
        GenerationRequest(1, "Fe1", {"Fe": 1}, 0)
    with pytest.raises(GeneratorError):
        GenerationRequest(1, "Fe1", {"Fe": 0}, 5)


# -- planning a call -------------------------------------------------------

def engine(**kw):
    return MatterGenEngine(model="mattergen_base", **kw)


def reqs(n, count=1, base=100):
    return [GenerationRequest(base + i, f"X{i}", {"Fe": 2, "Sm": 1}, n)
            for i in range(count)]


def test_one_composition_below_the_cap_is_one_batch():
    assert engine(max_batch_size=100).plan(reqs(72)) == [(72, 1)]


def test_one_composition_above_the_cap_splits_without_inflating():
    plan = engine(max_batch_size=100).plan(reqs(254))
    assert plan == [(85, 2), (84, 1)]
    assert sum(size * num for size, num in plan) == 254


def test_a_group_shares_one_call_and_divides_exactly():
    """MatterGen gives each composition `num_batches*batch_size // k`.

    The plan is built so that division is exact, because the remainder is
    dropped in silence and a product below `k` returns nothing at all.
    """
    k = 20
    plan = engine(max_batch_size=100).plan(reqs(72, count=k))
    assert plan == [(72, 20)]
    for size, num in plan:
        total = size * num
        assert total % k == 0
        assert total // k == 72


def test_a_group_above_the_cap_still_divides_exactly():
    k = 7
    n = 254
    plan = engine(max_batch_size=100).plan(reqs(n, count=k))
    assert sum(size * num for size, num in plan) == k * n
    for size, num in plan:
        assert (size * num) % k == 0
    assert sum((size * num) // k for size, num in plan) == n


def test_a_group_must_want_the_same_count():
    mixed = reqs(72) + reqs(96, base=200)
    with pytest.raises(GeneratorError, match="same count"):
        engine().plan(mixed)


def test_an_empty_group_plans_nothing():
    assert engine().plan([]) == []


# -- the command line ------------------------------------------------------

def test_the_command_names_a_checkpoint_by_path(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "config.yaml").write_text("x: 1\n")
    cmd = MatterGenEngine(model=str(tmp_path)).build_command(
        [{"Fe": 2}], tmp_path / "out", 10, 2)
    assert f"--model_path={tmp_path.resolve()}" in cmd
    assert not any(a.startswith("--pretrained_name") for a in cmd)


def test_the_command_names_a_pretrained_model_by_name(tmp_path):
    cmd = engine().build_command([{"Fe": 2}], tmp_path, 10, 1)
    assert "--pretrained_name=mattergen_base" in cmd
    assert not any(a.startswith("--model_path") for a in cmd)


def test_target_compositions_carry_no_whitespace(tmp_path):
    """MatterGen's CLI is `fire`, which splits the argv on spaces first.

    A pretty-printed dictionary arrives as several arguments none of which
    parse, and the failure is a fire usage error rather than anything about
    compositions.
    """
    cmd = engine().build_command([{"Sm": 1, "Co": 10}, {"Sm": 1, "Cu": 10}],
                                 tmp_path, 10, 2)
    arg = next(a for a in cmd if a.startswith("--target_compositions="))
    payload = arg.split("=", 1)[1]
    assert " " not in payload
    assert json.loads(payload) == [{"Sm": 1, "Co": 10}, {"Sm": 1, "Cu": 10}]


def test_csp_mode_is_requested_explicitly(tmp_path):
    cmd = engine(max_batch_size=8).build_command([{"Fe": 2}], tmp_path, 4, 1)
    assert "--sampling_config_name=csp" in cmd
    assert "--record_trajectories=False" in cmd


def test_unconditional_mode_sends_no_composition(tmp_path):
    cmd = MatterGenEngine(model="mattergen_base", mode="unconditional").build_command(
        [{"Fe": 2}], tmp_path, 4, 1)
    assert not any(a.startswith("--target_compositions") for a in cmd)
    assert not any(a.startswith("--sampling_config_name") for a in cmd)


# -- preflight -------------------------------------------------------------

def test_a_checkpoint_without_checkpoints_is_reported(tmp_path):
    (tmp_path / "config.yaml").write_text("x: 1\n")
    problems = MatterGenEngine(model=str(tmp_path)).preflight()
    assert any("checkpoints/" in p for p in problems)


def test_a_path_that_does_not_exist_is_not_treated_as_a_model_name(tmp_path):
    """The legacy test was `'/' in path or path.exists()`.

    A relative directory name with no slash and no existence went to the
    HuggingFace hub as a pretrained model name, which fails much later and in
    someone else's vocabulary.
    """
    problems = MatterGenEngine(model=str(tmp_path / "nope")).preflight()
    assert any("not a directory on disk" in p for p in problems)


def test_csp_training_is_confirmed_from_hydra_not_from_the_model_config(tmp_path):
    """A CSP checkpoint and an unconditional one have the same
    `property_embeddings: {}` -- composition conditioning is a *sampling*
    setting. What tells them apart is the training config Hydra recorded."""
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "config.yaml").write_text("lightning_module:\n  property_embeddings: {}\n")
    hydra = tmp_path / ".hydra"
    hydra.mkdir()
    hydra.joinpath("hydra.yaml").write_text("hydra:\n  job:\n    config:\n"
                                            "      config_name: default\n")
    problems = MatterGenEngine(model=str(tmp_path)).preflight()
    assert any("not 'csp'" in p for p in problems)

    hydra.joinpath("hydra.yaml").write_text("hydra:\n  job:\n    config:\n"
                                            "      config_name: csp\n")
    problems = MatterGenEngine(model=str(tmp_path)).preflight()
    assert not any("csp" in p for p in problems)


def test_a_checkpoint_with_no_hydra_record_says_so_rather_than_assuming(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "config.yaml").write_text("x: 1\n")
    problems = MatterGenEngine(model=str(tmp_path)).preflight()
    assert any("cannot be confirmed" in p for p in problems)


@pytest.mark.parametrize("text,expected", [
    ("hydra:\n  job:\n    config:\n      config_name: csp\n", "csp"),
    ("      config_name: 'csp'\n", "csp"),
    ('      config_name: "default"\n', "default"),
    ("nothing here\n", None),
])
def test_reading_the_hydra_config_name(text, expected):
    assert _hydra_config_name(text) == expected


def test_preflight_refuses_a_missing_gpu(monkeypatch):
    """MatterGen falls back to CPU in silence; a job that lands without a GPU
    would otherwise sample until its walltime and record a bare TIMEOUT."""
    problems = MatterGenEngine._gpu_problems()
    import torch
    if torch.cuda.is_available():                       # pragma: no cover
        assert problems == []
    else:
        assert any("CUDA" in p for p in problems)


# -- reading what came back ------------------------------------------------

def test_a_missing_output_file_names_the_reason_it_can_be_missing(tmp_path):
    with pytest.raises(GeneratorError, match="swallows IOError"):
        read_generated(tmp_path)


def test_an_empty_output_file_is_not_a_result(tmp_path):
    (tmp_path / "generated_crystals.extxyz").write_text("")
    with pytest.raises(GeneratorError, match="empty"):
        read_generated(tmp_path)


def test_reading_a_real_extxyz(tmp_path):
    import ase.io

    atoms = Atoms("Fe2Sm", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    ase.io.write(str(tmp_path / "generated_crystals.extxyz"), [atoms, atoms])
    assert len(read_generated(tmp_path)) == 2


def test_the_composition_check_accepts_what_was_asked_for():
    atoms = Atoms("Fe2Sm", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    assert check_composition(atoms, {"Fe": 2, "Sm": 1}) == ""


def test_the_composition_check_names_both_sides_when_it_fails():
    atoms = Atoms("Fe3", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    problem = check_composition(atoms, {"Fe": 2, "Sm": 1})
    assert "Fe2Sm1" in problem and "Fe3" in problem
