"""Schema validation: the rules the plan says must be enforced, enforced."""

import pytest
from pydantic import ValidationError

from cspflow.config.schema import (
    Campaign,
    ChemicalSpace,
    CompositionItem,
    ElementGroup,
    NStructures,
    Source,
    SourceMode,
    ZRange,
)


def _min_campaign(**over):
    base = dict(
        name="t",
        machine="local",
        workdir="/tmp/t",
        source=[
            dict(
                mode="structure_list",
                name="seeds",
                structure_list=dict(paths=["/tmp/*.vasp"]),
            )
        ],
    )
    base.update(over)
    return base


# --- typos are errors, not silent no-ops -----------------------------------


def test_unknown_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        Campaign.model_validate(_min_campaign(hull_threshold=0.1))


def test_unknown_nested_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        Campaign.model_validate(_min_campaign(filter={"e_above_hull_maximum": 0.1}))


# --- pick is required and bounded ------------------------------------------


def test_pick_is_required():
    with pytest.raises(ValidationError, match="pick"):
        ElementGroup.model_validate({"elements": ["Fe", "Co"]})


def test_pick_cannot_exceed_group_size():
    with pytest.raises(ValidationError, match="exceeds"):
        ElementGroup.model_validate({"elements": ["Fe", "Co"], "pick": 3})


def test_pick_accepts_list_of_arities():
    g = ElementGroup.model_validate({"elements": ["Fe", "Co", "Ni"], "pick": [1, 2]})
    assert g.arities() == [1, 2]


def test_pick_must_be_positive():
    with pytest.raises(ValidationError, match=">= 1"):
        ElementGroup.model_validate({"elements": ["Fe"], "pick": 0})


def test_duplicate_elements_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        ElementGroup.model_validate({"elements": ["Fe", "Fe"], "pick": 1})


def test_fraction_bounds_ordered():
    with pytest.raises(ValidationError, match="min_fraction"):
        ElementGroup.model_validate(
            {"elements": ["Fe"], "pick": 1, "min_fraction": 0.9, "max_fraction": 0.5}
        )


# --- max_rare_earth is separate from pick ----------------------------------


def test_max_rare_earth_defaults_to_one():
    cs = ChemicalSpace.model_validate(
        {"groups": {"A": {"elements": ["Sm", "Fe"], "pick": 1}}}
    )
    assert cs.max_rare_earth == 1


def test_max_rare_earth_can_be_disabled():
    cs = ChemicalSpace.model_validate(
        {"groups": {"A": {"elements": ["Sm"], "pick": 1}}, "max_rare_earth": None}
    )
    assert cs.max_rare_earth is None


# --- source mode must match its block --------------------------------------


def test_mode_requires_matching_block():
    with pytest.raises(ValidationError, match="requires a 'chemical_space:' block"):
        Source.model_validate({"mode": "chemical_space"})


def test_mode_rejects_foreign_blocks():
    with pytest.raises(ValidationError, match="also set"):
        Source.model_validate(
            {
                "mode": "structure_list",
                "structure_list": {"paths": ["a"]},
                "composition_list": {"items": [{"formula": "Fe"}]},
            }
        )


def test_entry_stage():
    seeds = Source.model_validate(
        {"mode": "structure_list", "structure_list": {"paths": ["a"]}}
    )
    assert seeds.entry_stage == "screen"
    comps = Source.model_validate(
        {"mode": "composition_list", "composition_list": {"items": [{"formula": "Fe"}]}}
    )
    assert comps.entry_stage == "generate"


# --- multi-source rules ----------------------------------------------------


def test_single_source_may_be_a_bare_mapping():
    c = Campaign.model_validate(
        _min_campaign(
            source=dict(mode="structure_list", structure_list=dict(paths=["/tmp/a"]))
        )
    )
    assert len(c.source) == 1


def test_duplicate_source_names_rejected():
    s = dict(mode="structure_list", name="x", structure_list=dict(paths=["/tmp/a"]))
    with pytest.raises(ValidationError, match="duplicate source names"):
        Campaign.model_validate(_min_campaign(source=[s, dict(s)]))


def test_multi_source_requires_explicit_names():
    a = dict(mode="structure_list", structure_list=dict(paths=["/tmp/a"]))
    b = dict(mode="structure_list", name="b", structure_list=dict(paths=["/tmp/b"]))
    with pytest.raises(ValidationError, match="explicit 'name:'"):
        Campaign.model_validate(_min_campaign(source=[a, b]))


def test_generating_source_requires_generate_block():
    with pytest.raises(ValidationError, match="'generate:' block is required"):
        Campaign.model_validate(
            _min_campaign(
                source=[
                    dict(
                        mode="composition_list",
                        composition_list=dict(items=[dict(formula="SmFe11Ti")]),
                    )
                ]
            )
        )


def test_seed_only_campaign_needs_no_generate_block():
    c = Campaign.model_validate(_min_campaign())
    assert c.needs_generation is False
    assert c.generate is None


# --- small conveniences that must not silently misbehave -------------------


def test_z_list_shorthand():
    item = CompositionItem.model_validate({"formula": "SmFe11Ti", "z": [1, 3]})
    assert item.z.values() == [1, 2, 3]


def test_z_shorthand_rejects_wrong_arity():
    with pytest.raises(ValidationError, match=r"\[min, max\]"):
        CompositionItem.model_validate({"formula": "Fe", "z": [1, 2, 3]})


def test_z_range_must_be_ordered():
    with pytest.raises(ValidationError, match="z.max"):
        ZRange.model_validate({"min": 4, "max": 2})


def test_n_structures_requires_the_field_for_its_mode():
    with pytest.raises(ValidationError, match="requires 'count'"):
        NStructures.model_validate({"mode": "fixed"})
    with pytest.raises(ValidationError, match="structures_per_atom"):
        NStructures.model_validate({"mode": "per_atom", "structures_per_atom": None})


def test_default_n_structures_is_itself_valid():
    """The default_factory must produce a model that validates.

    Regression: `structures_per_atom` originally defaulted to None while `mode`
    defaulted to per_atom, so every Campaign built without an explicit
    n_structures block failed validation on a default it never chose.
    """
    assert NStructures().target_for(10) == 20


def test_n_structures_target():
    assert NStructures(mode="fixed", count=300).target_for(40) == 300
    assert NStructures(mode="per_atom", structures_per_atom=2.0).target_for(13) == 26


# --- the two moments are never merged --------------------------------------


def test_ambiguous_moment_property_rejected():
    with pytest.raises(ValidationError, match="ambiguous"):
        Campaign.model_validate(_min_campaign(analyze={"properties": ["m_s"]}))


def test_both_moments_by_default():
    c = Campaign.model_validate(_min_campaign())
    assert "m_dft_raw" in c.analyze.properties
    assert "m_s_reconstructed" in c.analyze.properties


# --- walltime format -------------------------------------------------------


def test_bad_walltime_rejected():
    with pytest.raises(ValidationError, match="HH:MM:SS"):
        Campaign.model_validate(_min_campaign(screen={"resources": {"time": "24h"}}))


def test_rare_earth_detection():
    c = Campaign.model_validate(
        _min_campaign(
            source=[
                dict(
                    mode="chemical_space",
                    chemical_space=dict(
                        groups={
                            "A": dict(elements=["Sm", "Tb"], pick=1),
                            "B": dict(elements=["Fe", "Y"], pick=1),
                        }
                    ),
                )
            ],
            generate=dict(mattergen=dict(model="/tmp/ckpt")),
        )
    )
    # Y is group 3, not a rare earth for 4f purposes.
    assert c.rare_earth_elements == {"Sm", "Tb"}
