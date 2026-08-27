"""Layered loading: merge order, variable expansion, provenance."""

import pytest

from cspflow.config.loader import (
    ConfigError,
    apply_set,
    deep_merge,
    expand_vars,
    load_campaign,
    parse_set,
    resolve_machine_path,
)

CAMPAIGN = """
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list:
      paths: ["/tmp/a.vasp"]
filter:
  e_above_hull_max: 0.15
"""


def _write(tmp_path, text, name="campaign.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# --- variable expansion ----------------------------------------------------


def test_expand_simple_and_braced():
    env = {"USER": "ridwan", "SCRATCH": "/scratch/ridwan"}
    assert expand_vars("$SCRATCH/x", env) == "/scratch/ridwan/x"
    assert expand_vars("${USER}-run", env) == "ridwan-run"


def test_expand_is_recursive():
    env = {"A": "1"}
    out = expand_vars({"k": ["$A", {"j": "$A"}]}, env)
    assert out == {"k": ["1", {"j": "1"}]}


def test_undefined_variable_is_an_error_not_empty_string():
    """Silently expanding to '' turns $SCRATCH/x into /x -- an absolute path at
    the filesystem root.  That must never happen quietly."""
    with pytest.raises(ConfigError, match="undefined variable"):
        expand_vars("$NOPE_NOT_SET/x", {})


def test_undefined_variable_names_its_location():
    with pytest.raises(ConfigError, match="workdir"):
        expand_vars({"workdir": "$NOPE_NOT_SET"}, {})


def test_non_strings_pass_through():
    assert expand_vars({"n": 3, "b": True, "f": 1.5}, {}) == {"n": 3, "b": True, "f": 1.5}


# --- merging ---------------------------------------------------------------


def test_deep_merge_recurses_into_dicts():
    origins = {}
    out = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}, "L2", origins)
    assert out == {"a": {"x": 1, "y": 3}}
    assert origins["a.y"] == "L2"
    assert "a.x" not in origins


def test_lists_replace_rather_than_append():
    origins = {}
    out = deep_merge({"p": [1, 2, 3]}, {"p": [9]}, "L2", origins)
    assert out["p"] == [9]


def test_merge_marks_whole_new_subtree():
    origins = {}
    deep_merge({}, {"a": {"b": {"c": 1}}}, "L2", origins)
    assert origins["a.b.c"] == "L2"


# --- --set -----------------------------------------------------------------


def test_parse_set_uses_yaml_scalars():
    assert parse_set("filter.e_above_hull_max=0.2") == ("filter.e_above_hull_max", 0.2)
    assert parse_set("dft.ldau.enabled=true") == ("dft.ldau.enabled", True)
    assert parse_set("name=abc") == ("name", "abc")


def test_parse_set_requires_equals():
    with pytest.raises(ConfigError, match="key=value"):
        parse_set("nope")


def test_apply_set_creates_intermediate_maps():
    cfg, origins = {}, {}
    apply_set(cfg, "a.b.c", 1, origins)
    assert cfg == {"a": {"b": {"c": 1}}}
    assert origins["a.b.c"] == "--set"


def test_apply_set_refuses_to_descend_into_a_scalar():
    cfg, origins = {"a": 5}, {}
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_set(cfg, "a.b", 1, origins)


# --- machine resolution ----------------------------------------------------


def test_shipped_machines_resolve():
    for name in ("orion", "generic_slurm", "local"):
        assert resolve_machine_path(name).is_file()


def test_unknown_machine_lists_what_exists():
    with pytest.raises(ConfigError, match="shipped profiles"):
        resolve_machine_path("nosuchcluster")


# --- end to end ------------------------------------------------------------


def test_load_campaign(tmp_path):
    p = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path))
    rc = load_campaign(p)
    assert rc.campaign.name == "t"
    assert rc.campaign.filter.e_above_hull_max == 0.15
    assert rc.machine.scheduler == "local"


def test_cli_set_wins_over_file(tmp_path):
    p = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path))
    rc = load_campaign(p, sets=["filter.e_above_hull_max=0.42"])
    assert rc.campaign.filter.e_above_hull_max == 0.42
    assert rc.origins["filter.e_above_hull_max"] == "--set"


def test_origin_is_recorded_for_file_values(tmp_path):
    p = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path))
    rc = load_campaign(p)
    assert rc.origins["filter.e_above_hull_max"] == str(p)


def test_config_hash_is_stable_and_sensitive(tmp_path):
    p = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path))
    a = load_campaign(p).config_hash
    b = load_campaign(p).config_hash
    assert a == b and len(a) == 64
    c = load_campaign(p, sets=["filter.e_above_hull_max=0.99"]).config_hash
    assert c != a


def test_config_hash_ignores_formatting(tmp_path):
    """The hash tracks the physics, not the file."""
    a = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path), "a.yaml")
    reordered = (
        "source:\n"
        "  - mode: structure_list\n"
        "    name: seeds\n"
        "    structure_list: {paths: ['/tmp/a.vasp']}\n"
        "filter: {e_above_hull_max: 0.15}\n"
        f"workdir: {tmp_path}\n"
        "machine: local\n"
        "name: t\n"
    )
    b = _write(tmp_path, reordered, "b.yaml")
    assert load_campaign(a).config_hash == load_campaign(b).config_hash


def test_missing_campaign_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_campaign(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    p = _write(tmp_path, "name: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_campaign(p)


def test_validation_error_names_the_file(tmp_path):
    p = _write(tmp_path, CAMPAIGN.format(workdir=tmp_path) + "\nbogus_key: 1\n")
    with pytest.raises(ConfigError, match=str(p)):
        load_campaign(p)


def test_expansion_happens_after_all_layers(tmp_path):
    """A later layer may override a value that would need an undefined var."""
    env = {"USER": "tester"}  # enough for the machine profile's own $USER
    p = _write(tmp_path, CAMPAIGN.format(workdir="$UNSET_ON_PURPOSE/x"))
    with pytest.raises(ConfigError, match="undefined variable"):
        load_campaign(p, env=env)
    rc = load_campaign(p, sets=[f"workdir={tmp_path}"], env=env)
    assert rc.campaign.workdir == str(tmp_path)
