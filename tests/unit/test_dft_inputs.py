"""M2 -- recipes, INCAR, KPOINTS and job-directory assembly.

Several of these check cspflow against the legacy campaign's own input files,
which is the only independent evidence that the port is faithful. Where they
disagree, the test says which is which and why.
"""

import json
import math
from pathlib import Path

import pytest
import yaml
from ase.build import bulk

from cspflow.config.schema import Dft, Ldau, Machine, Magnetism, RareEarth
from cspflow.dft.recipe import (
    Kpoints,
    Recipe,
    RecipeError,
    RecipeStage,
    build_recipe,
    load_recipe,
    validate_recipe,
)
from cspflow.dft.vasp.incar import (
    FERRI_RETM,
    IncarContext,
    IncarError,
    build_incar,
    ldau_block,
    lmaxmix_for,
    magmom_for,
    nbands_auto,
    render_incar,
)
from cspflow.dft.vasp.kpoints import KpointsError, grid_for

MACHINES = Path(__file__).resolve().parents[2] / "src" / "cspflow" / "machines"
LEGACY = Path("/projects/mmi/shuo/redo-new-ter-mag/VASP_JOBS/Gd1Co10Cr2/Gd1Co10Cr2_s020/Relax")
has_legacy = pytest.mark.skipif(not LEGACY.is_dir(), reason="legacy campaign not present")


@pytest.fixture
def orion() -> Machine:
    return Machine(**yaml.safe_load((MACHINES / "orion.yaml").read_text()))


def minimal_incar(**extra):
    base = {"ENCUT": 520, "ISPIN": 2, "LASPH": ".TRUE.", "NELM": 200, "LORBIT": 11}
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------


class TestRecipe:
    def test_inherit_copies_it_does_not_defer(self):
        """A resolved stage holds every tag literally, so --dry-run can print it."""
        recipe = build_recipe({"stages": [
            {"name": "relax", "incar": minimal_incar(NSW=99, ISMEAR=1)},
            {"name": "static", "inherit": "relax", "incar": {"NSW": 0, "ISMEAR": -5}},
        ]})
        static = recipe.stage("static")
        assert static.incar["ENCUT"] == 520          # inherited
        assert static.incar["NSW"] == 0              # overridden
        assert static.incar["ISMEAR"] == -5

    def test_inheriting_forwards_is_refused(self):
        """Single-pass resolution, so no cycle is possible."""
        with pytest.raises(RecipeError, match="not defined above it"):
            build_recipe({"stages": [
                {"name": "a", "inherit": "b", "incar": {}},
                {"name": "b", "incar": {}},
            ]})

    def test_resources_and_retry_are_inherited_too(self):
        recipe = build_recipe({"stages": [
            {"name": "relax", "incar": minimal_incar(),
             "resources": {"ntasks": 64}, "retry": [{"when": "timeout"}]},
            {"name": "static", "inherit": "relax", "incar": {}},
        ]})
        static = recipe.stage("static")
        assert static.resources["ntasks"] == 64
        assert [r["when"] for r in static.retry] == ["timeout"]

    def test_a_missing_stage_names_the_ones_that_exist(self):
        recipe = build_recipe({"stages": [{"name": "relax", "incar": minimal_incar()}]})
        with pytest.raises(RecipeError, match=r"\['relax'\]"):
            recipe.stage("soc")

    def test_no_stages_is_an_error(self):
        with pytest.raises(RecipeError, match="no stages"):
            build_recipe({"stages": []})

    def test_an_unknown_recipe_lists_the_shipped_ones(self):
        with pytest.raises(RecipeError, match="Shipped recipes"):
            load_recipe("no-such-recipe")


class TestRecipeValidation:
    @pytest.mark.parametrize(
        "tag, fragment",
        [
            ("ENCUT", "CHANGES WITH COMPOSITION"),
            ("ISPIN", "every moment would be zero"),
            ("LASPH", "aspherical"),
            ("NELM", "gives up early"),
            ("LORBIT", "nothing to analyse"),
        ],
    )
    def test_a_missing_tag_is_refused_with_what_vasp_would_do(self, tag, fragment):
        incar = minimal_incar()
        del incar[tag]
        with pytest.raises(RecipeError, match=fragment):
            validate_recipe(build_recipe({"stages": [{"name": "x", "incar": incar}]}))

    def test_lmaxmix_may_be_omitted_because_it_is_computed(self):
        """From the POTCARs -- more reliable than asking the user to keep it in step."""
        validate_recipe(build_recipe({"stages": [{"name": "x", "incar": minimal_incar()}]}))

    def test_an_unknown_tag_warns_but_is_not_dropped(self):
        """"Add any tag you like" has to remain true, so there is no whitelist."""
        warnings = validate_recipe(build_recipe({"stages": [
            {"name": "x", "incar": minimal_incar(MYTAG=1)}]}))
        assert any("MYTAG" in w for w in warnings)

    def test_the_yaml_on_trap_is_caught(self):
        """YAML 1.1 parses a bare `on` as True, so `- on: timeout` loses its key."""
        warnings = validate_recipe(build_recipe({"stages": [
            {"name": "x", "incar": minimal_incar(), "retry": [{True: "timeout"}]}]}))
        assert any("YAML parses a bare `on`" in w for w in warnings)

    def test_the_shipped_magnets_recipe_validates_cleanly(self):
        assert validate_recipe(load_recipe("magnets")) == []

    def test_the_shipped_recipe_has_relax_then_static(self):
        assert load_recipe("magnets").stage_names == ["relax", "static"]


# --------------------------------------------------------------------------
# MAGMOM
# --------------------------------------------------------------------------


class TestMagmom:
    def test_the_retm_convention_is_re_negative_tm_positive(self):
        moments = magmom_for(["Gd", "Co"], Magnetism(), RareEarth())
        assert moments[0] < 0 < moments[1]

    def test_early_3d_metals_are_antiparallel_to_late_3d(self):
        """Ported from the campaign's own table: 'early 3d TM = small negative'."""
        for early in ("Ti", "V", "Cr"):
            assert FERRI_RETM[early] < 0, early
        for late in ("Fe", "Co", "Ni"):
            assert FERRI_RETM[late] > 0, late

    def test_it_reproduces_the_campaigns_own_magmom(self):
        """Gd1Co10Cr2 was run with `MAGMOM = 2*-7.0 4*-2.0 20*2.0`."""
        symbols = ["Gd"] * 2 + ["Cr"] * 4 + ["Co"] * 20
        moments = magmom_for(symbols, Magnetism(), RareEarth())
        assert moments == [-7.0] * 2 + [-2.0] * 4 + [2.0] * 20

    def test_ferro_flips_the_rare_earth_parallel(self):
        """The one-line version of the old-vs-redo campaign difference."""
        ferri = magmom_for(["Gd", "Co"], Magnetism(), RareEarth(magnetic_order="ferri"))
        ferro = magmom_for(["Gd", "Co"], Magnetism(), RareEarth(magnetic_order="ferro"))
        assert ferri[0] < 0 and ferro[0] > 0
        assert abs(ferri[0]) == abs(ferro[0])

    def test_mode_none_writes_nothing(self):
        assert magmom_for(["Fe"], Magnetism(mode="none")) is None

    def test_strict_refuses_an_element_with_no_entry(self):
        """A silent default moment converges to the wrong magnetic state quietly."""
        with pytest.raises(IncarError, match="no initial moment"):
            magmom_for(["Xe"], Magnetism(mode="table", table={"Fe": 2.0}))

    def test_non_strict_falls_back_and_says_what_it_used(self):
        moments = magmom_for(["Xe"], Magnetism(mode="table", table={"Fe": 2.0},
                                               strict=False))
        assert moments == [0.6]

    def test_site_overrides_win(self):
        moments = magmom_for(["Gd", "Co"], Magnetism(site_overrides={"Gd": -3.5}),
                             RareEarth())
        assert moments[0] == -3.5

    def test_a_user_table_replaces_the_preset(self):
        moments = magmom_for(["Fe"], Magnetism(mode="table", table={"Fe": 5.0}))
        assert moments == [5.0]


# --------------------------------------------------------------------------
# NBANDS, LMAXMIX, LDAU
# --------------------------------------------------------------------------


class TestComputedTags:
    def test_nbands_uses_the_ported_formula(self):
        context = IncarContext(symbols=["Fe"] * 4, zvals={"Fe": 8.0})
        # nelect 32; max(16 + max(2,10), 19) = 26 -> ceil to 28 at NCORE 4
        assert nbands_auto(context, ncore=4) == 28

    def test_nbands_rounds_up_to_a_multiple_of_ncore(self):
        context = IncarContext(symbols=["Fe"] * 10, zvals={"Fe": 8.0})
        assert nbands_auto(context, ncore=8) % 8 == 0

    def test_nbands_is_none_without_zvals(self):
        assert nbands_auto(IncarContext(symbols=["Fe"])) is None

    def test_lmaxmix_is_6_with_f_in_valence_and_4_otherwise(self):
        assert lmaxmix_for(True) == 6
        assert lmaxmix_for(False) == 4

    def test_ldau_is_empty_when_disabled(self):
        assert ldau_block(Ldau(), ["Fe"]) == {}

    def test_ldau_refuses_a_missing_u(self):
        """A missing entry would silently become 0 and mix U with non-U results."""
        with pytest.raises(IncarError, match="no U is given"):
            ldau_block(Ldau(enabled=True, u={"Fe": 4.0}), ["Fe", "Co"])

    def test_ldau_arrays_follow_poscar_element_order(self):
        block = ldau_block(Ldau(enabled=True, u={"Fe": 4.0, "Co": 0.0}), ["Co", "Fe"])
        assert block["LDAUU"] == [0.0, 4.0]      # Co first, as given
        assert block["LDAUL"] == [-1, 2]


class TestBuildIncar:
    def _context(self):
        return IncarContext(symbols=["Gd", "Co", "Co"], formula="Co2Gd1",
                            zvals={"Gd": 9.0, "Co": 9.0}, f_in_valence=False)

    def test_system_defaults_to_the_formula(self):
        incar = build_incar(minimal_incar(), self._context())
        assert incar["SYSTEM"] == "Co2Gd1"

    def test_lmaxmix_is_added_from_the_potcars(self):
        assert build_incar(minimal_incar(), self._context())["LMAXMIX"] == 4

    def test_an_explicit_lmaxmix_in_the_recipe_wins(self):
        assert build_incar(minimal_incar(LMAXMIX=6), self._context())["LMAXMIX"] == 6

    def test_magmom_is_skipped_when_ispin_is_1(self):
        incar = build_incar(minimal_incar(ISPIN=1), self._context(),
                            magnetism=Magnetism(), rare_earth=RareEarth())
        assert "MAGMOM" not in incar

    def test_campaign_overrides_win_over_everything(self):
        incar = build_incar(minimal_incar(), self._context(),
                            overrides={"ENCUT": 700, "NEW_TAG": 1})
        assert incar["ENCUT"] == 700 and incar["NEW_TAG"] == 1


class TestRenderIncar:
    def test_python_booleans_become_vasp_booleans(self):
        """A user writing `LASPH: true` in YAML gets a Python bool here."""
        assert "LASPH = .TRUE." in render_incar({"LASPH": True})
        assert "LWAVE = .FALSE." in render_incar({"LWAVE": False})

    def test_magmom_is_run_length_encoded(self):
        text = render_incar({"MAGMOM": [-7.0, -7.0, 2.0, 2.0, 2.0]})
        assert "MAGMOM = 2*-7 3*2" in text

    def test_a_single_run_still_encodes(self):
        assert "MAGMOM = 3*2" in render_incar({"MAGMOM": [2.0, 2.0, 2.0]})

    def test_lists_other_than_magmom_are_space_separated(self):
        assert "LDAUU = 4 0" in render_incar({"LDAUU": [4.0, 0.0]})

    def test_a_comment_is_prefixed(self):
        assert render_incar({"ENCUT": 520}, comment="hello").startswith("! hello")


# --------------------------------------------------------------------------
# KPOINTS
# --------------------------------------------------------------------------


class TestKpoints:
    def test_reciprocal_density_reproduces_the_campaigns_grid(self):
        """Gd1Co10Cr2_s020: a,b,c = 4.6399, 8.1658, 8.2433 A -> `5 3 3`."""
        grid = grid_for([4.6399342629, 8.1658250943, 8.2432836376],
                        Kpoints("reciprocal_density", 64))
        assert (grid.a, grid.b, grid.c) == (5, 3, 3)

    def test_the_grid_is_independent_of_the_atom_count(self):
        """V_recip * V_cell is (2*pi)^3 identically, so n_atoms cancels -- which
        is what makes the sampling comparable across the cell sizes a hull spans."""
        lengths = [4.64, 8.17, 8.24]
        small = grid_for(lengths, Kpoints("reciprocal_density", 64), n_atoms=2)
        large = grid_for(lengths, Kpoints("reciprocal_density", 64), n_atoms=200)
        assert (small.a, small.b, small.c) == (large.a, large.b, large.c)

    def test_a_denser_setting_gives_a_denser_grid(self):
        lengths = [5.0, 5.0, 5.0]
        coarse = grid_for(lengths, Kpoints("reciprocal_density", 64))
        fine = grid_for(lengths, Kpoints("reciprocal_density", 512))
        assert fine.total > coarse.total

    def test_kspacing(self):
        grid = grid_for([4.6399, 8.1658, 8.2433], Kpoints("kspacing", 0.3))
        assert (grid.a, grid.b, grid.c) == (5, 3, 3)

    def test_explicit(self):
        grid = grid_for([5.0, 5.0, 5.0], Kpoints("explicit", [3, 3, 2]))
        assert (grid.a, grid.b, grid.c) == (3, 3, 2)

    def test_explicit_needs_three_numbers(self):
        with pytest.raises(KpointsError, match="three-element"):
            grid_for([5.0] * 3, Kpoints("explicit", [3, 3]))

    def test_an_unknown_scheme_names_the_known_ones(self):
        with pytest.raises(KpointsError, match="known:"):
            grid_for([5.0] * 3, Kpoints("magic", 1))

    def test_the_grid_is_never_zero(self):
        grid = grid_for([500.0, 500.0, 500.0], Kpoints("reciprocal_density", 1))
        assert grid.a >= 1 and grid.b >= 1 and grid.c >= 1

    def test_gamma_centred_by_default(self):
        """Preserves the point symmetry; an even Monkhorst-Pack grid does not."""
        assert "Gamma" in grid_for([5.0] * 3, Kpoints("reciprocal_density", 64)).render()


# --------------------------------------------------------------------------
# Whole job directories
# --------------------------------------------------------------------------


class TestJobDirectory:
    def _resolve(self, orion, atoms, **kw):
        from cspflow.dft.vasp.inputs import resolve_inputs

        return resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                              Dft(recipe="magnets", **kw), orion)

    @pytest.mark.skipif(not (MACHINES / "orion.yaml").is_file(), reason="no profile")
    def test_species_order_is_first_appearance_not_alphabetical(self, orion):
        """POTCAR concatenation and the LDAU arrays both key on this order."""
        from cspflow.dft.vasp.inputs import _species_order

        atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
        atoms.symbols = ["Ni", "Co"]
        assert _species_order(atoms) == ["Ni", "Co"]

    def test_settings_hash_covers_the_potcars(self):
        """Identical INCARs with different pseudopotentials are not comparable."""
        from cspflow.dft.vasp.inputs import ResolvedInputs
        from cspflow.dft.vasp.kpoints import KpointGrid
        from cspflow.dft.vasp.potcar import PotcarInfo

        def make(hash_):
            return ResolvedInputs(
                stage="relax", incar={"ENCUT": 520},
                grid=KpointGrid(3, 3, 3, "explicit"), symbols=["Fe"],
                potcars=[PotcarInfo("Fe", "Fe_pv", Path("/x"), "t", 8.0, 268.0,
                                    hash_, False)],
            )

        assert make("aaa").settings_hash != make("bbb").settings_hash

    def test_settings_hash_is_stable_for_identical_inputs(self):
        from cspflow.dft.vasp.inputs import ResolvedInputs
        from cspflow.dft.vasp.kpoints import KpointGrid

        def make():
            return ResolvedInputs(stage="relax", incar={"ENCUT": 520, "ISPIN": 2},
                                  grid=KpointGrid(3, 3, 3, "explicit"), symbols=["Fe"])

        assert make().settings_hash == make().settings_hash


@has_legacy
class TestAgainstTheLegacyCampaign:
    """The port is faithful where it should be, and different where it means to be."""

    def _legacy_atoms(self):
        from ase.io import read

        return read(str(LEGACY / "POSCAR"))

    def test_kpoints_match_exactly(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        legacy = [int(x) for x in (LEGACY / "KPOINTS").read_text().splitlines()[3].split()]
        assert [resolved.grid.a, resolved.grid.b, resolved.grid.c] == legacy

    def test_magmom_matches_exactly(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        rendered = render_incar({"MAGMOM": resolved.incar["MAGMOM"]}).strip()
        assert rendered == "MAGMOM = 2*-7 4*-2 20*2"

    def test_potcar_order_matches_the_poscar_species_line(self, orion):
        from cspflow.dft.vasp.inputs import resolve_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        legacy_species = (LEGACY / "POSCAR").read_text().splitlines()[5].split()
        assert [p.element for p in resolved.potcars] == legacy_species

    def test_what_we_write_passes_our_own_stage_0_gate(self, orion, tmp_path):
        """The POSCAR we emit must be one we would accept as input."""
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs
        from cspflow.source.structure_list import read_seed

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        read_atoms, counts = read_seed(out / "POSCAR")
        assert counts == {"Gd": 2, "Cr": 4, "Co": 20}

    def test_the_manifest_records_what_was_written(self, orion, tmp_path):
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        manifest = json.loads((out / "inputs.json").read_text())
        assert manifest["settings_hash"] == resolved.settings_hash
        assert {p["element"] for p in manifest["potcars"]} == {"Gd", "Cr", "Co"}
        assert manifest["incar"]["ENCUT"] == 520

    def test_all_four_files_are_written(self, orion, tmp_path):
        from cspflow.dft.vasp.inputs import resolve_inputs, write_inputs

        atoms = self._legacy_atoms()
        resolved = resolve_inputs(atoms, load_recipe("magnets").stage("relax"),
                                  Dft(recipe="magnets"), orion)
        out = write_inputs(resolved, atoms, tmp_path / "job")
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "inputs.json"):
            assert (out / name).is_file(), name
