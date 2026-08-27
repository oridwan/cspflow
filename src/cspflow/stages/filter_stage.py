"""Stage 5 -- the gate between the cheap tier and the expensive one.

This is the barrier in the funnel: everything before it costs GPU-minutes,
everything after it costs ~10³ core-hours per composition. So the stage's job is
not to be clever, it is to be **legible** -- every candidate that does not go
forward has a recorded reason, and `csp status --why` can replay it.

Two things here are less obvious than they look.

**The threshold may be calibrated rather than literal.** If Stage 4 found that
the MLIP compresses hull distances by 1.4x, then screening at 0.10 in MLIP units
silently discards candidates sitting at 0.10 in DFT units. `e_above_hull_max` is
therefore converted through the fitted α/β -- and the conversion adjusts the
*threshold*, never a stored energy (D068). What is stored stays what was
measured; the cutoff is a config value, in provenance, auditable and reversible.

**`max_per_composition` is applied after the hull cut, not before.** Taking the
best N per composition first and then applying the threshold would let a
composition whose candidates are all bad contribute its best three anyway. The
threshold is about whether a structure is worth computing; the per-composition
cap is about not spending the whole budget on one formula.
"""

from __future__ import annotations

from collections import defaultdict

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from .base import StageReport


class FilterStage:
    name = "filter"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.effective_threshold: float | None = None

    def pending(self, store: Store) -> int:
        return len(self._candidates(store))

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("filter is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("filter is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        rows = self._candidates(store)
        if not rows:
            return StageReport(stage=self.name, note="nothing screened to filter")

        cfg = self.cfg.campaign.filter
        threshold, how = self._threshold(store)
        self.effective_threshold = threshold

        # 1. The hull cut. Everything above the threshold is out, with the
        #    number that put it there recorded.
        survivors = []
        for row, e_hull in rows:
            passed = e_hull <= threshold
            store.add_filter_event(
                structure_id=int(row.id), gate="filter:e_above_hull",
                passed=passed, value=e_hull, threshold=threshold,
                detail=how,
            )
            if passed:
                survivors.append((row, e_hull))
            else:
                store.set_structure_state(int(row.id), StructureState.filtered_out,
                                          filter_reason="above the hull threshold")

        # 2. The per-composition cap, applied to what survived the cut.
        by_composition: dict[str, list] = defaultdict(list)
        for row, e_hull in survivors:
            by_composition[row.toatoms().get_chemical_formula()].append((row, e_hull))

        selected, capped = 0, 0
        for formula, members in sorted(by_composition.items()):
            members.sort(key=lambda pair: pair[1])
            for rank, (row, e_hull) in enumerate(members):
                within = rank < cfg.max_per_composition
                store.add_filter_event(
                    structure_id=int(row.id), gate="filter:per_composition",
                    passed=within, value=float(rank + 1),
                    threshold=float(cfg.max_per_composition),
                    detail=f"rank {rank + 1} of {len(members)} for {formula}",
                )
                if within:
                    store.set_structure_state(int(row.id), StructureState.selected)
                    selected += 1
                else:
                    store.set_structure_state(
                        int(row.id), StructureState.filtered_out,
                        filter_reason=f"rank {rank + 1} exceeds max_per_composition",
                    )
                    capped += 1

        note = (f"threshold {threshold:.4f} eV/atom ({how}); "
                f"{selected} selected, {len(rows) - len(survivors)} above the hull, "
                f"{capped} over the per-composition cap")
        return StageReport(stage=self.name, claimed=selected,
                           reconciled=len(rows), note=note)

    # -- internals ---------------------------------------------------------

    def _candidates(self, store: Store) -> list[tuple]:
        """Screened structures that have a hull placement and no verdict yet."""
        placements = {
            int(r["structure_id"]): float(r["e_above_hull"])
            for r in store.sql.execute(
                "SELECT structure_id, e_above_hull FROM hull WHERE hull_type='mlip'")
        }
        out = []
        for row in store.structures(state=StructureState.screened.value):
            e_hull = placements.get(int(row.id))
            if e_hull is not None:
                out.append((row, e_hull))
        return out

    def _threshold(self, store: Store) -> tuple[float, str]:
        """The cutoff to apply, and a one-line account of where it came from.

        `calibrated` falls back to the literal value with the reason stated,
        rather than silently: a threshold that quietly stopped being calibrated
        is exactly the kind of change that alters results with no visible cause.
        """
        cfg = self.cfg.campaign.filter
        literal = cfg.e_above_hull_max
        if cfg.e_above_hull_max_source != "calibrated":
            return literal, "literal"

        row = store.latest_calibration("pilot") or store.latest_calibration("mp")
        alpha, beta = _fit_from(row)
        if alpha is None:
            return literal, "literal (no calibration fit available yet)"
        if alpha == 0:
            return literal, "literal (calibration fit has alpha=0, no information)"

        converted = (literal - (beta or 0.0)) / alpha
        return converted, (f"calibrated: DFT cutoff {literal:.3f} -> MLIP "
                           f"{converted:.4f} via alpha={alpha:.4f}, beta={beta or 0.0:+.4f}")


def _fit_from(row) -> tuple[float | None, float | None]:
    if row is None:
        return None, None
    try:
        return (row["alpha"], row["beta"])
    except (KeyError, IndexError):                         # pragma: no cover
        return None, None
