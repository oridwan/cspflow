"""The module check.

A missing modulefile is the failure this exists for, and it is a nasty one:
`module load` writes its error to stderr and still exits 0, so under the `set -e`
every generated script uses, the job dies at whatever command needed the module
and reports that command's failure instead.

The first live GPU submission from this repository failed in zero seconds with
`Unable to locate a modulefile for 'cuda/11.8'` and nothing else -- the profile
listed a module the cluster had removed.
"""

import pytest

from cspflow.config.schema import Machine
from cspflow.doctor import available_modules, check_modules

MODULES = available_modules()
has_modules = pytest.mark.skipif(not MODULES, reason="no module system here")


def machine(**modules):
    return Machine(scheduler="slurm", modules=modules)


def test_a_profile_with_no_modules_is_fine():
    check = check_modules(machine())
    assert check.status == "ok"


@has_modules
def test_the_listing_is_read_from_stderr_too():
    """Environment Modules writes `-t avail` to stderr and exits 0.

    Reading stdout alone gets an empty string and a clean exit code -- which is
    indistinguishable from "this cluster has no modules", and would skip the
    check that exists to catch the failure that motivated it.
    """
    assert MODULES and len(MODULES) > 10
    assert all(not name.endswith(":") for name in MODULES)
    assert all("(default)" not in name for name in MODULES)


@has_modules
def test_a_module_that_exists_passes():
    name = next(iter(sorted(MODULES)))
    check = check_modules(machine(cpu=[name]))
    assert check.status == "ok"
    assert any("present" in row for row in check.rows)


@has_modules
def test_a_module_that_does_not_exist_fails():
    check = check_modules(machine(gpu=["definitely-not-a-module/9.9"]))
    assert check.status == "fail"
    assert any("no definitely-not-a-module module at all" in row for row in check.rows)


@has_modules
def test_a_wrong_version_names_the_versions_that_do_exist():
    """The useful message. `cuda/11.8` is gone from this cluster; 12.4, 12.8 and
    13.2 are not, and saying so is the difference between a fix and a hunt."""
    base = next((name.split("/")[0] for name in sorted(MODULES) if "/" in name), None)
    if base is None:                                          # pragma: no cover
        pytest.skip("no versioned modules to test against")
    check = check_modules(machine(gpu=[f"{base}/0.0-nope"]))
    assert check.status == "fail"
    assert any("this cluster has" in row and base in row for row in check.rows)


def test_the_orion_profile_no_longer_loads_a_cuda_module():
    """Verified on a GPU node: driver 570.195.03, torch 2.2.1+cu118 ships its
    own runtime, and `torch.cuda.is_available()` is True with nothing loaded."""
    from pathlib import Path

    import yaml

    profile = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"
         / "orion.yaml").read_text())
    assert profile["modules"]["gpu"] == []
