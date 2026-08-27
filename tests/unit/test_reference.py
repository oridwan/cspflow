"""Stage 3 -- MP corrections and convex hulls.

Entirely offline: no MP key, no network, no DFT. The hull tests use synthetic
entries whose geometry is worked out by hand, which is the only way to know the
guards fire on the cases they are meant to and not on the cases they are not.
"""

import pytest

from cspflow.reference.corrections import (
    ANION_CORRECTED,
    U_CORRECTED,
    audit_chemsystems,
    correction_risk,
    load_mp2020_tables,
)
from cspflow.reference.hull import (
    Entry,
    HullError,
    assert_elemental_references,
    assert_one_functional,
    assert_one_scale,
    build_hull,
    find_duplicates,
    place_on_hull,
)

pytest.importorskip("pymatgen.analysis.phase_diagram", reason="hulls need pymatgen")


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------


class TestCorrectionRisk:
    def test_intermetallics_are_correction_free(self):
        """The chemistry being run today: every MP2020 correction is exactly zero."""
        for chemsys in ["Co-Fe-Sm", "Co-Gd-Ti", "Fe-Sm-Ti", "Co-Cr-Gd"]:
            assert not correction_risk(chemsys).affected

    def test_a_nitride_triggers_the_composition_correction(self):
        """Sm-Fe-N is an obvious next target, and it is where this fires."""
        risk = correction_risk("Fe-N-Sm")
        assert risk.affected and risk.anion_elements == ("N",)

    def test_an_oxide_triggers_both_corrections(self):
        risk = correction_risk("Fe-O")
        assert risk.anion_elements == ("O",) and risk.u_elements == ("Fe",)

    def test_hubbard_u_needs_oxygen_or_fluorine_present(self):
        """Fe alone gets no U correction; Fe with O does."""
        assert correction_risk("Co-Fe").u_elements == ()
        assert "Fe" in correction_risk("Fe-O").u_elements
        assert "Fe" in correction_risk("F-Fe").u_elements

    def test_the_message_says_which_way_it_matters(self):
        assert "DIFFER" in correction_risk("Fe-N-Sm").render()
        assert "identical" in correction_risk("Co-Fe").render()

    def test_audit_reports_only_the_affected_systems(self):
        risks, message = audit_chemsystems(["Co-Fe-Sm", "Co-Gd-Ti", "Fe-N-Sm"])
        assert len(risks) == 3
        assert "1 of 3" in message and "Fe-N-Sm" in message

    def test_audit_says_so_plainly_when_nothing_is_affected(self):
        _, message = audit_chemsystems(["Co-Fe-Sm", "Co-Gd-Ti"])
        assert "correction-free" in message

    def test_our_element_lists_match_the_installed_pymatgen(self):
        """A pymatgen upgrade that moves these is a test failure, not a silent shift."""
        tables = load_mp2020_tables()
        anion_keys = set(tables["anion"])
        # pymatgen spells the oxygen corrections as oxide/peroxide/superoxide/ozonide.
        elemental = {k for k in anion_keys if k[0].isupper() and "oxide" not in k}
        assert elemental <= ANION_CORRECTED
        assert set(tables["u"]) == {"O", "F"}          # the TRIGGER elements
        assert "Fe" in U_CORRECTED and "Co" in U_CORRECTED


# --------------------------------------------------------------------------
# Hull guards
# --------------------------------------------------------------------------


def fe_sm_reference() -> list[Entry]:
    """A hand-worked Fe-Sm reference set.

    Per atom: Fe 0, Sm 0, Fe2Sm -0.500, Fe17Sm2 -0.3158. The tie-line from Fe to
    Fe17Sm2 at x_Sm = 1/13 sits at -0.2314 eV/atom, so a Fe12Sm entry below
    -3.008 eV total is stable and one above it is not.
    """
    return [
        Entry("Fe", {"Fe": 1}, 0.0, source="mp", run_type="GGA"),
        Entry("Sm", {"Sm": 1}, 0.0, source="mp", run_type="GGA"),
        Entry("Fe2Sm1", {"Fe": 2, "Sm": 1}, -1.5, source="mp", run_type="GGA"),
        Entry("Fe17Sm2", {"Fe": 17, "Sm": 2}, -6.0, source="mp", run_type="GGA"),
    ]


class TestEntry:
    def test_energy_is_total_not_per_atom(self):
        e = Entry("x", {"Fe": 2, "Sm": 1}, -1.5)
        assert e.n_atoms == 3 and e.e_per_atom == -0.5

    def test_elemental_is_one_species(self):
        assert Entry("Fe", {"Fe": 4}, 0.0).is_elemental
        assert not Entry("FeSm", {"Fe": 1, "Sm": 1}, 0.0).is_elemental

    def test_formula_is_canonical(self):
        assert Entry("x", {"Sm": 1, "Fe": 2}, 0.0).formula == "Fe2Sm1"


class TestScaleGuard:
    def test_one_scale_passes(self):
        assert assert_one_scale(fe_sm_reference()) == "raw"

    def test_mixing_scales_is_a_hard_error(self):
        entries = [*fe_sm_reference(),
                   Entry("mp", {"Fe": 1, "Sm": 1}, -1.0, scale="mp_corrected")]
        with pytest.raises(HullError, match="energy scales"):
            build_hull(entries)

    def test_the_error_says_why_a_warning_would_not_do(self):
        entries = [*fe_sm_reference(),
                   Entry("mp", {"Fe": 1, "Sm": 1}, -1.0, scale="mp_corrected")]
        with pytest.raises(HullError, match="same size as the filter threshold"):
            build_hull(entries)

    def test_no_entries_at_all(self):
        with pytest.raises(HullError, match="no entries"):
            assert_one_scale([])


class TestElementalReferenceGuard:
    def test_missing_elemental_reference_is_refused(self):
        """pymatgen builds a diagram regardless; every e_above_hull would be wrong."""
        with pytest.raises(HullError, match="no elemental reference"):
            build_hull([Entry("Fe", {"Fe": 1}, 0.0),
                        Entry("Fe2Sm1", {"Fe": 2, "Sm": 1}, -1.5)])

    def test_the_error_names_the_missing_element(self):
        with pytest.raises(HullError, match=r"\['Sm'\]"):
            assert_elemental_references([Entry("Fe", {"Fe": 1}, 0.0),
                                         Entry("FeSm", {"Fe": 1, "Sm": 1}, -1.0)])

    def test_a_complete_set_passes(self):
        assert_elemental_references(fe_sm_reference())


class TestFunctionalGuard:
    def test_mixing_gga_and_gga_plus_u_is_refused(self):
        entries = [*fe_sm_reference(),
                   Entry("u", {"Fe": 1, "Sm": 1}, -1.0, run_type="GGA+U")]
        with pytest.raises(HullError, match="mix functionals"):
            build_hull(entries)

    def test_unrecorded_run_type_is_not_a_conflict(self):
        """"Not recorded" differs from "conflicts"; refusing would ban hand-built entries."""
        assert assert_one_functional([Entry("a", {"Fe": 1}, 0.0),
                                      Entry("b", {"Sm": 1}, 0.0)]) == ""

    def test_unrecorded_alongside_recorded_warns(self):
        result = build_hull([*fe_sm_reference(),
                             Entry("nofunc", {"Fe": 1, "Sm": 1}, -0.4)])
        assert any("no run_type" in w for w in result.warnings)

    def test_the_guard_can_be_turned_off_deliberately(self):
        entries = [*fe_sm_reference(),
                   Entry("u", {"Fe": 1, "Sm": 1}, -1.0, run_type="GGA+U")]
        assert build_hull(entries, strict_functional=False).n_entries == 5


class TestDuplicates:
    def test_same_composition_from_two_sources_is_reported(self):
        entries = [*fe_sm_reference(),
                   Entry("mine", {"Fe": 2, "Sm": 1}, -1.6, source="ours", run_type="GGA")]
        notes = find_duplicates(entries)
        assert notes and "Fe2Sm1" in notes[0] and "spread" in notes[0]

    def test_polymorphs_from_one_source_are_not_flagged(self):
        entries = [*fe_sm_reference(),
                   Entry("alt", {"Fe": 2, "Sm": 1}, -1.4, source="mp", run_type="GGA")]
        assert not find_duplicates(entries)


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


class TestPlacement:
    def test_a_stable_candidate_sits_on_the_hull(self):
        candidate = Entry("good", {"Fe": 12, "Sm": 1}, -3.5, source="ours", run_type="GGA")
        assert place_on_hull(candidate, fe_sm_reference()) == pytest.approx(0.0, abs=1e-9)

    def test_an_unstable_candidate_sits_above_it(self):
        candidate = Entry("bad", {"Fe": 12, "Sm": 1}, -1.0, source="ours", run_type="GGA")
        assert place_on_hull(candidate, fe_sm_reference()) == pytest.approx(0.1538, abs=1e-3)

    def test_the_reference_phases_are_all_stable(self):
        result = build_hull(fe_sm_reference())
        assert result.stable == {"Fe", "Sm", "Fe2Sm1", "Fe17Sm2"}

    def test_a_better_candidate_moves_a_worse_ones_number(self):
        """Why the hull is rebuilt when results arrive, rather than cached per candidate."""
        reference = fe_sm_reference()
        worse = Entry("bad", {"Fe": 12, "Sm": 1}, -1.0, source="ours", run_type="GGA")
        better = Entry("good", {"Fe": 12, "Sm": 1}, -3.5, source="ours", run_type="GGA")

        alone = place_on_hull(worse, reference)
        together = build_hull([*reference, better, worse]).e_above_hull["bad"]
        assert together > alone

    def test_formation_energies_are_reported(self):
        result = build_hull(fe_sm_reference())
        assert result.formation_energy["Fe2Sm1"] == pytest.approx(-0.5, abs=1e-9)
        assert result.formation_energy["Fe"] == pytest.approx(0.0, abs=1e-9)

    def test_render_marks_the_stable_ones(self):
        text = build_hull(fe_sm_reference()).render()
        assert "raw scale" in text and text.count("*") >= 4
