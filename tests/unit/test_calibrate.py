"""Stage 4 -- parity statistics, and the two claims that shape what may be fitted.

The two network tests at the end reproduce pipeline.md §4a's central numbers
against live MP data. They are the reason the design says what it says, so they
are checked rather than quoted:

* a formula-only fit over Sm-Fe-Ti MP phases reaches R^2 = 0.971;
* a per-element energy correction moves `e_above_hull` by ~1.8e-15 eV/atom.

Measured 2026-08-27: **R^2 = 0.9714** and **max shift 1.776e-15 eV/atom**.
"""

import math
import os

import pytest

from cspflow.calibrate import (
    ParityPoint,
    apply_per_element_correction,
    build_report,
    composition_only_r2,
    correction_shifts_hull_by,
    fit_threshold,
    per_element_mae,
    spearman,
)


def point(label, counts, mlip, dft, **kw):
    return ParityPoint(label=label, counts=counts, e_mlip_per_atom=mlip,
                       e_dft_per_atom=dft, **kw)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_agreement_is_one(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_inversion_is_minus_one(self):
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_it_measures_rank_not_value(self):
        """Screening needs the ranking right, not the absolute energies."""
        assert spearman([1, 2, 3], [1, 100, 10000]) == pytest.approx(1.0)

    def test_ties_get_average_ranks(self):
        assert spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)

    def test_too_few_points_is_none_not_zero(self):
        assert spearman([1], [1]) is None

    def test_a_constant_series_is_none(self):
        assert spearman([1, 1, 1], [1, 2, 3]) is None


class TestPerElementMae:
    def test_it_attributes_error_to_the_elements_present(self):
        points = [point("a", {"Sm": 1, "Fe": 1}, -1.0, -1.5),
                  point("b", {"Fe": 2}, -2.0, -2.0)]
        mae = per_element_mae(points)
        assert mae["Sm"] > 0 and mae["Fe"] < mae["Sm"]

    def test_a_perfect_element_has_zero_error(self):
        assert per_element_mae([point("a", {"Fe": 1}, -1.0, -1.0)])["Fe"] == 0.0


# --------------------------------------------------------------------------
# The composition-only baseline
# --------------------------------------------------------------------------


class TestCompositionOnlyBaseline:
    def test_a_purely_compositional_series_is_fitted_exactly(self):
        """The point of the baseline: composition alone explains a great deal."""
        points = [
            point("a", {"Fe": 1}, 0.0, -8.0),
            point("b", {"Sm": 1}, 0.0, -4.0),
            point("c", {"Fe": 1, "Sm": 1}, 0.0, -6.0),
            point("d", {"Fe": 3, "Sm": 1}, 0.0, -7.0),
        ]
        assert composition_only_r2(points) == pytest.approx(1.0, abs=1e-9)

    def test_too_few_points_is_none(self):
        assert composition_only_r2([point("a", {"Fe": 1}, 0.0, -8.0)]) is None

    def test_it_appears_in_the_report_so_r2_is_read_against_it(self):
        points = [point(f"p{i}", {"Fe": i, "Sm": 1}, -i * 0.1, -i * 0.1)
                  for i in range(1, 6)]
        assert "composition-only R^2 baseline" in build_report(points).render()


# --------------------------------------------------------------------------
# The correction that cannot move anything
# --------------------------------------------------------------------------


class TestPerElementCorrection:
    def test_it_really_does_change_the_energies(self):
        points = [point("a", {"Sm": 1, "Fe": 2}, -5.0, -5.0)]
        shifted = apply_per_element_correction(points, {"Sm": 0.85, "Fe": -0.40})
        assert shifted[0] != pytest.approx(-5.0)

    def test_and_yet_it_cannot_move_a_hull_distance(self):
        """The compound's shift and its terminals' shifts are the same linear
        function of composition, so they cancel along the tie-line."""
        shift = correction_shifts_hull_by(
            {"Sm": 1, "Fe": 12},
            terminals=[{"Sm": 1}, {"Fe": 1}],
            deltas={"Sm": 0.85, "Fe": -0.40},
        )
        assert abs(shift) < 1e-12

    def test_it_cancels_for_a_ternary_too(self):
        shift = correction_shifts_hull_by(
            {"Sm": 1, "Fe": 11, "Ti": 1},
            terminals=[{"Sm": 1}, {"Fe": 1}, {"Ti": 1}],
            deltas={"Sm": 0.85, "Fe": -0.40, "Ti": 1.25},
        )
        assert abs(shift) < 1e-12

    def test_it_cancels_whatever_the_deltas_are(self):
        for deltas in ({"Sm": 100.0, "Fe": -50.0}, {"Sm": -3.3, "Fe": 0.0},
                       {"Sm": 0.0, "Fe": 7.7}):
            shift = correction_shifts_hull_by({"Sm": 2, "Fe": 17},
                                              [{"Sm": 1}, {"Fe": 1}], deltas)
            assert abs(shift) < 1e-12, deltas


# --------------------------------------------------------------------------
# The fit that IS useful
# --------------------------------------------------------------------------


class TestThresholdFit:
    def _compressed(self, factor=1.4):
        """An MLIP that compresses hull distances by `factor`."""
        return [ParityPoint(f"p{i}", {"Fe": 1}, 0.0, 0.0,
                            e_hull_mlip=x / factor, e_hull_dft=x)
                for i, x in enumerate([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])]

    def test_it_recovers_the_compression(self):
        fit = fit_threshold(self._compressed(1.4))
        assert fit.alpha == pytest.approx(1.4, abs=1e-6)
        assert fit.beta == pytest.approx(0.0, abs=1e-9)

    def test_it_converts_the_threshold_you_want_into_the_one_to_use(self):
        """Screening at 0.10 in MLIP units would silently discard candidates
        sitting at 0.10 in DFT units."""
        fit = fit_threshold(self._compressed(1.4))
        assert fit.mlip_cutoff_for(0.10) == pytest.approx(0.10 / 1.4, abs=1e-6)

    def test_a_faithful_mlip_needs_no_conversion(self):
        fit = fit_threshold(self._compressed(1.0))
        assert fit.mlip_cutoff_for(0.10) == pytest.approx(0.10, abs=1e-9)

    def test_it_reports_r2(self):
        assert fit_threshold(self._compressed()).r2 == pytest.approx(1.0, abs=1e-9)

    def test_too_few_points_is_none(self):
        assert fit_threshold(self._compressed()[:2]) is None

    def test_points_without_hull_distances_are_not_usable(self):
        assert fit_threshold([point(f"p{i}", {"Fe": 1}, -i, -i) for i in range(5)]) is None

    def test_a_degenerate_fit_refuses_to_convert(self):
        from cspflow.calibrate import ThresholdFit

        with pytest.raises(ValueError, match="alpha is zero"):
            ThresholdFit(alpha=0.0, beta=0.0, n=5).mlip_cutoff_for(0.1)


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


class TestReport:
    def _good(self):
        return [ParityPoint(f"p{i}", {"Fe": 1, "Sm": i}, -8.0 + i * 0.1, -8.0 + i * 0.1,
                            e_hull_mlip=i * 0.01, e_hull_dft=i * 0.01,
                            volume_mlip=100.0, volume_dft=100.0)
                for i in range(1, 8)]

    def test_a_faithful_model_passes(self):
        assert build_report(self._good()).verdict == "pass"

    def test_no_points_at_all_is_a_failure(self):
        report = build_report([])
        assert report.verdict == "fail" and "no calibration points" in report.reasons[0]

    def test_a_large_hull_error_fails(self):
        """e_above_hull is the quantity candidates are filtered on."""
        points = [ParityPoint(f"p{i}", {"Fe": 1}, -8.0, -8.0,
                              e_hull_mlip=i * 0.01, e_hull_dft=i * 0.01 + 0.3)
                  for i in range(1, 8)]
        report = build_report(points)
        assert report.verdict == "fail"
        assert any("filtered on" in r for r in report.reasons)

    def test_bad_ranking_fails_even_with_small_errors(self):
        points = [ParityPoint(f"p{i}", {"Fe": 1}, -8.0, -8.0,
                              e_hull_mlip=x, e_hull_dft=y)
                  for i, (x, y) in enumerate(
                      [(0.00, 0.03), (0.01, 0.00), (0.02, 0.04),
                       (0.03, 0.01), (0.04, 0.02), (0.05, 0.005)])]
        report = build_report(points)
        assert report.verdict == "fail"
        assert any("RANKING" in r for r in report.reasons)

    def test_volume_drift_warns_and_is_never_folded_into_the_energy(self):
        points = [ParityPoint(f"p{i}", {"Fe": 1}, -8.0, -8.0,
                              e_hull_mlip=i * 0.01, e_hull_dft=i * 0.01,
                              volume_mlip=120.0, volume_dft=100.0)
                  for i in range(1, 8)]
        report = build_report(points)
        assert report.verdict == "warn"
        assert any("not fixed by an energy correction" in r for r in report.reasons)

    def test_one_bad_element_is_named(self):
        points = [point("a", {"Sm": 4}, -1.0, -1.5),
                  point("b", {"Fe": 4}, -2.0, -2.001),
                  point("c", {"Ti": 4}, -3.0, -3.001)]
        report = build_report(points, mae_max=10.0)
        assert any("Sm" in r and "fine-tuning" in r for r in report.reasons)

    def test_the_render_separates_energy_from_geometry(self):
        text = build_report(self._good()).render()
        assert "single point at the DFT geometry" in text
        assert "never folded into the energy" in text


# --------------------------------------------------------------------------
# Against live MP data
# --------------------------------------------------------------------------


def mp_usable() -> bool:
    if not os.environ.get("MP_API_KEY"):
        return False
    try:
        import warnings

        from mp_api.client import MPRester

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with MPRester(os.environ["MP_API_KEY"]) as mpr:
                mpr.materials.thermo.search(chemsys=["Fe"], fields=["material_id"],
                                            num_chunks=1, chunk_size=1)
        return True
    except Exception:
        return False


needs_mp = pytest.mark.skipif(not mp_usable(), reason="MP unusable in this env")


@needs_mp
@pytest.mark.network
class TestAgainstRealMPData:
    def test_a_formula_only_fit_reaches_r2_of_0_971(self, tmp_path):
        """pipeline.md §4a's first claim. Measured 2026-08-27: 0.9714 over 41 phases.

        This is why a parity R^2 on total energies is not evidence: the baseline
        it has to beat is 0.97, not 0.
        """
        from cspflow.reference.mp import fetch_chemsys

        entries = fetch_chemsys("Fe-Sm-Ti", cache=tmp_path).entries
        points = [point(e.mp_id, e.counts, 0.0, e.e_raw_per_atom) for e in entries]
        assert len(points) >= 40
        assert composition_only_r2(points) == pytest.approx(0.971, abs=0.01)

    def test_a_per_element_correction_leaves_every_hull_distance_alone(self, tmp_path):
        """pipeline.md §4a's second claim, on a real 41-phase Sm-Fe-Ti hull.

        Measured 2026-08-27: max |change| = 1.776e-15 eV/atom, while the total
        energies themselves moved by up to 1.25 eV/atom. The correction is real;
        its effect on the filtered quantity is exactly zero.
        """
        from cspflow.reference.hull import Entry, build_hull
        from cspflow.reference.mp import fetch_chemsys

        entries = fetch_chemsys("Fe-Sm-Ti", cache=tmp_path).entries
        deltas = {"Sm": 0.85, "Fe": -0.40, "Ti": 1.25}

        def hull_entries(shift: bool):
            out = []
            for e in entries:
                per_atom = e.e_raw_per_atom
                if shift:
                    per_atom += sum(deltas.get(el, 0.0) * c / e.n_atoms
                                    for el, c in e.counts.items())
                out.append(Entry(label=e.mp_id, counts=e.counts,
                                 energy=per_atom * e.n_atoms, scale="raw",
                                 source="mp", run_type="GGA"))
            return out

        before = build_hull(hull_entries(False)).e_above_hull
        after = build_hull(hull_entries(True)).e_above_hull
        worst = max(abs(after[k] - before[k]) for k in before)
        assert worst < 1e-12, f"per-element correction moved a hull distance by {worst}"
