"""Preflight checks and the command line."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cspflow import doctor as D
from cspflow.cli import app
from cspflow.config.loader import load_campaign
from cspflow.config.schema import Machine

runner = CliRunner()


def output_of(result) -> str:
    """stdout + stderr, whichever the installed click keeps them in.

    click < 8.2 merged stderr into `result.output`; 8.2+ keeps them separate.
    The suite has to pass in both the base env (click 8.1) and the cspflow env
    (click 8.5), so tests never assume which stream an error landed in.
    """
    parts = [result.stdout or ""]
    try:
        parts.append(result.stderr or "")
    except (ValueError, AttributeError):
        pass
    return "".join(parts)

CAMPAIGN = """\
name: t
machine: local
workdir: {workdir}
source:
  - mode: structure_list
    name: seeds
    structure_list: {{paths: ["{workdir}/a.vasp"]}}
"""


@pytest.fixture()
def campaign_file(tmp_path):
    p = tmp_path / "campaign.yaml"
    p.write_text(CAMPAIGN.format(workdir=tmp_path))
    return p


# --- report rendering ------------------------------------------------------


def test_report_fails_if_any_check_fails():
    r = D.Report([D.Check("a", "ok"), D.Check("b", "fail", "boom")])
    assert r.failed
    assert "FAIL" in r.render() and "boom" in r.render()


def test_report_passes_when_clean():
    r = D.Report([D.Check("a", "ok"), D.Check("b", "warn")])
    assert not r.failed
    assert "All checks passed" not in r.render()  # a warning is still reported


def test_all_clear_says_so():
    assert "All checks passed" in D.Report([D.Check("a", "ok")]).render()


# --- individual checks -----------------------------------------------------


def test_vasp_binary_missing_is_a_failure():
    m = Machine.model_validate({"codes": {"vasp_std": "/nope/vasp_std"}})
    assert D.check_vasp(m).status == "fail"


def test_no_vasp_configured_is_skipped():
    assert D.check_vasp(Machine()).status == "skip"


def test_vasp_binary_present(tmp_path):
    exe = tmp_path / "vasp_std"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    m = Machine.model_validate({"codes": {"vasp_std": str(exe)}})
    assert D.check_vasp(m).status == "ok"


def test_binary_without_exec_bit_is_a_failure(tmp_path):
    exe = tmp_path / "vasp_std"
    exe.write_text("x")
    exe.chmod(0o644)
    m = Machine.model_validate({"codes": {"vasp_std": str(exe)}})
    assert D.check_vasp(m).status == "fail"


def test_local_scheduler_needs_no_queue():
    assert D.check_scheduler(Machine(scheduler="local")).status == "ok"


def test_unwritable_workdir_is_a_failure(campaign_file, tmp_path):
    cfg = load_campaign(campaign_file, sets=["workdir=/nonexistent-root/x/y"])
    assert D.check_paths(cfg).status == "fail"


def test_writable_workdir_passes(campaign_file):
    assert D.check_paths(load_campaign(campaign_file)).status == "ok"


def test_potcar_layout_skipped_when_profile_defines_none():
    assert D.check_potcar_layout(Machine()).status == "skip"


TREE = Path("/projects/mmi/Ridwan/potcarFiles/pmg")
has_tree = pytest.mark.skipif(not TREE.is_dir(), reason="POTCAR symlink tree not created")


@has_tree
def test_4f_check_is_not_a_vacuous_pass(campaign_file):
    """One rare earth cannot disagree with itself, and zero certainly cannot.
    Reporting 'ok' there would be worse than reporting nothing."""
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name.split(" (")[0]: c for c in D.check_potcars(cfg, ["Fe"])}
    assert by_name["4f convention"].status == "skip"


@has_tree
def test_4f_check_passes_on_a_consistent_series(campaign_file):
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name.split(" (")[0]: c for c in D.check_potcars(cfg, ["Sm", "Gd", "Fe"])}
    assert by_name["4f convention"].status == "ok"
    assert by_name["POTCAR resolution"].status == "ok"


@has_tree
def test_encut_warning_when_absent_and_ok_when_explicit(campaign_file):
    """Removing MPRelaxSet removed the thing that was silently supplying
    ENCUT=520, so doctor has to surface it."""
    cfg = load_campaign(campaign_file, machine="orion")
    by_name = {c.name: c for c in D.check_potcars(cfg, ["Sm", "Fe", "Ti"])}
    assert by_name["ENCUT"].status == "warn"

    cfg2 = load_campaign(campaign_file, machine="orion",
                         sets=["dft.incar_overrides.ENCUT=520"])
    by_name2 = {c.name: c for c in D.check_potcars(cfg2, ["Sm", "Fe", "Ti"])}
    assert by_name2["ENCUT"].status == "ok"


@has_tree
def test_missing_potcar_root_is_a_clean_failure(campaign_file):
    cfg = load_campaign(campaign_file, machine="local")
    checks = D.check_potcars(cfg, ["Fe"])
    assert checks[0].status == "fail" and "potcar_root" in checks[0].detail


# --- CLI -------------------------------------------------------------------


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0 and "cspflow" in output_of(res)


def test_init_writes_a_valid_campaign(tmp_path):
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["init", "demo", "-o", str(out), "-m", "local"])
    assert res.exit_code == 0 and out.is_file()
    cfg = load_campaign(out, sets=[f"workdir={tmp_path}"])
    assert cfg.campaign.name == "demo"


def test_init_full_is_also_valid(tmp_path):
    """The Tier-2 block is commented out, so it must not break parsing."""
    out = tmp_path / "c.yaml"
    res = runner.invoke(app, ["init", "demo", "-o", str(out), "-m", "local", "--full"])
    assert res.exit_code == 0
    assert "Tier 2" in out.read_text()
    load_campaign(out, sets=[f"workdir={tmp_path}"])


def test_init_refuses_to_clobber(tmp_path):
    out = tmp_path / "c.yaml"
    out.write_text("existing")
    res = runner.invoke(app, ["init", "demo", "-o", str(out), "-m", "local"])
    assert res.exit_code == 1 and out.read_text() == "existing"
    ok = runner.invoke(app, ["init", "demo", "-o", str(out), "-m", "local", "--force"])
    assert ok.exit_code == 0


def test_config_show(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file)])
    assert res.exit_code == 0
    out = output_of(res)
    assert "config_hash" in out and "name: t" in out


def test_config_show_origins(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file), "--origins"])
    assert res.exit_code == 0 and str(campaign_file) in output_of(res)


def test_config_show_json(campaign_file):
    import json

    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file), "--json"])
    assert res.exit_code == 0
    out = output_of(res)
    payload = json.loads(out[: out.rindex("}") + 1])
    assert payload["name"] == "t"


def test_config_defaults_lists_schema_defaults():
    res = runner.invoke(app, ["config", "defaults"])
    assert res.exit_code == 0 and "e_above_hull_max" in output_of(res)


def test_set_flows_through_the_cli(campaign_file):
    res = runner.invoke(app, ["config", "show", "-c", str(campaign_file),
                              "-s", "filter.e_above_hull_max=0.33"])
    assert res.exit_code == 0 and "0.33" in output_of(res)


def test_bad_config_exits_nonzero_with_a_message(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: t\nmachine: local\nworkdir: /tmp\nsource: []\n")
    res = runner.invoke(app, ["config", "show", "-c", str(bad)])
    assert res.exit_code == 1


def test_status_without_a_database_is_a_clean_error(campaign_file):
    res = runner.invoke(app, ["status", "-c", str(campaign_file)])
    assert res.exit_code == 1
    assert "no campaign database" in output_of(res)


def test_doctor_exit_code_reflects_failure(campaign_file):
    """doctor must be usable as a gate in a submission script."""
    res = runner.invoke(app, ["doctor", "-c", str(campaign_file),
                              "-s", "workdir=/nonexistent-root/x"])
    assert res.exit_code == 1
