"""The campaign folder: what `csp init` writes, and how it resolves.

A campaign is a folder -- campaign.yaml plus editable copies of the machine
profile and the DFT recipe -- so the two things worth testing are that the
folder is complete and that a relative path inside it means "beside the
campaign file" from anywhere you might run the command.
"""

import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cspflow.cli import _find_campaign, app
from cspflow.config.loader import ConfigError, load_campaign, resolve_machine_path
from cspflow.dft.recipe import load_recipe
from cspflow.templates import campaign_yaml

runner = CliRunner()

WORKSPACE = {"campaign.yaml", "machine.yaml", "recipe.yaml",
             "README.md", "inputs/README.md"}


def _init(tmp_path: Path, *args: str):
    result = runner.invoke(app, ["init", "demo", "-d", str(tmp_path / "demo"),
                                 "-m", "local", *args])
    assert result.exit_code == 0, result.output
    return tmp_path / "demo"


# --- what init writes ------------------------------------------------------

def test_init_writes_a_folder_with_every_knob_in_it(tmp_path):
    root = _init(tmp_path)
    written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert written == WORKSPACE


def test_the_copies_are_real_copies_not_stubs(tmp_path):
    """The point of copying is that the knobs are *there* to be edited."""
    root = _init(tmp_path)
    machine = yaml.safe_load((root / "machine.yaml").read_text())
    recipe = yaml.safe_load((root / "recipe.yaml").read_text())
    assert machine["scheduler"] == "local"
    assert recipe["stages"], "a recipe with no stages is not editable"


def test_minimal_writes_only_the_campaign_file(tmp_path):
    root = _init(tmp_path, "--minimal")
    written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert written == {"campaign.yaml"}
    assert yaml.safe_load((root / "campaign.yaml").read_text())["machine"] == "local"


def test_existing_files_are_not_clobbered(tmp_path):
    root = _init(tmp_path)
    (root / "campaign.yaml").write_text("name: mine\n")
    result = runner.invoke(app, ["init", "demo", "-d", str(root), "-m", "local"])
    assert result.exit_code == 1
    assert (root / "campaign.yaml").read_text() == "name: mine\n"


# --- the annotated and terse files cannot drift ----------------------------

def test_minimal_is_the_annotated_file_with_the_comments_removed(tmp_path):
    """Same keys, both parse. `--minimal` is a view, not a second template."""
    full = yaml.safe_load(campaign_yaml(name="demo"))
    terse = yaml.safe_load(campaign_yaml(name="demo", minimal=True))
    assert full == terse


def _uncomment_knobs(text: str) -> str:
    """Delete the leading "# " from every commented-out YAML line.

    This is the property the template promises: a knob is enabled by deleting
    two characters, and what is left is valid YAML at the right depth. Prose
    comments are left alone -- an indented comment is legal YAML wherever it
    sits, so leaving them costs nothing.
    """
    knob = re.compile(r"\s*(-\s+)?[A-Za-z_]\w*:|\s*-\s+\{")
    return "\n".join(
        line[2:] if line.startswith("# ") and knob.match(line[2:]) else line
        for line in text.splitlines()
    )


def test_uncommenting_every_knob_leaves_a_valid_campaign():
    """A knob the schema rejects, or one indented under the wrong parent, is a
    trap: the user deletes two characters and gets an error they did not write.

    `extra="forbid"` makes this catch a renamed key, a key at the wrong depth,
    and a key that never existed -- the three ways a hand-written template goes
    stale against the schema it documents.
    """
    from cspflow.config.schema import Campaign

    doc = yaml.safe_load(_uncomment_knobs(campaign_yaml(name="demo")))
    campaign = Campaign.model_validate(doc)

    assert campaign.screen.mattersim.max_steps == 500      # nested, not top level
    assert campaign.dft.select.max_total == 1500
    assert {s.mode.value for s in campaign.source} == {
        "chemical_space", "composition_list", "structure_list"}


# --- resolution ------------------------------------------------------------

def test_machine_and_recipe_resolve_beside_the_campaign_file(tmp_path, monkeypatch):
    root = _init(tmp_path)
    monkeypatch.chdir(tmp_path)             # deliberately *not* in the folder
    cfg = load_campaign(root / "campaign.yaml")
    assert cfg.machine_path == root / "machine.yaml"
    assert cfg.base_dir == root
    assert load_recipe(cfg.campaign.dft.recipe, cfg.base_dir).stages


def test_a_relative_machine_path_reports_where_it_looked(tmp_path):
    with pytest.raises(ConfigError) as exc:
        resolve_machine_path("nowhere.yaml", tmp_path)
    assert str(tmp_path) in str(exc.value)


def test_a_shipped_name_still_wins_over_the_folder(tmp_path):
    """`machine: local` must keep meaning the shipped profile."""
    assert resolve_machine_path("local", tmp_path).name == "local.yaml"


# --- finding the campaign from anywhere ------------------------------------

def test_commands_walk_up_to_the_campaign(tmp_path, monkeypatch):
    root = _init(tmp_path)
    deep = root / "inputs" / "seeds"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert _find_campaign(Path("campaign.yaml")) == root / "campaign.yaml"


def test_a_named_file_is_never_hunted_for(tmp_path, monkeypatch):
    """If the user names a file, a missing one is an error, not a search."""
    _init(tmp_path)
    monkeypatch.chdir(tmp_path / "demo" / "inputs")
    assert _find_campaign(Path("other.yaml")) == Path("other.yaml")
