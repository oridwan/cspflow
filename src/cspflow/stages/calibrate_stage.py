"""Stage 4a -- `calibrate:mp`, the free calibration.

Runs on MP phases the moment they are fetched, **before** the MLIP relaxes
anything, and needs nothing from the candidates. So it can run at the very start,
in parallel with generation: if MatterSim is bad at Sm you learn it in minutes
rather than after a GPU-day of MatterGen.

One forward pass per reference structure. For the whole 36-system magnet
campaign that is ~359 structures, seconds of GPU.

**What it does not do, and why.** It does not fit an energy correction. A
per-element correction is the only physically motivated form, and it moves
`e_above_hull` by ~10⁻¹⁵ eV/atom — exact cancellation along the tie-line,
measured on a real 41-phase Sm–Fe–Ti hull (D067). Fitting one would produce a
number that looks like a result and changes nothing.

**What it cannot replace.** Universal MLIPs of this class are trained on
MP-derived trajectories, so MP phases are close to in-distribution and 4a is
substantially a self-consistency check. It catches gross failure, not subtle
bias. The structures the campaign actually cares about are *generated*, often in
prototypes absent from MP — that is the out-of-distribution set, and only
`calibrate:pilot` (4b) tests the model there. Which is why 4a defaults to
`warn` and 4b to `block`.
"""

from __future__ import annotations

from ..calibrate.parity import ParityPoint, build_report, fit_threshold
from ..config.loader import ResolvedConfig
from ..db.store import Store
from ..reference.mp import ReferenceError, fetch_structures
from .base import StageReport


class CalibrateStage:
    name = "calibrate"
    role = "gpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, engine=None) -> None:
        self.cfg = cfg
        self._engine = engine
        self.last_report = None

    @property
    def engine(self):
        if self._engine is None:
            from ..mlip import for_config

            self._engine = for_config(self.cfg.campaign.screen)
        return self._engine

    def pending(self, store: Store) -> int:
        """Reference entries not yet given an MLIP single point."""
        row = store.sql.execute(
            "SELECT COUNT(*) n FROM reference_entry WHERE e_mlip_static IS NULL"
        ).fetchone()
        return int(row["n"]) if row else 0

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("calibrate is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("calibrate is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        rows = [r for r in store.reference_entries() if r["e_mlip_static"] is None]
        if not rows:
            return StageReport(stage=self.name, note="every reference entry already has one")

        structures = self._structures(store)
        points, evaluated, missing = [], 0, 0

        for row in rows:
            structure = structures.get(row["mp_id"])
            if structure is None:
                missing += 1
                continue
            atoms = _to_atoms(structure)

            # Single point AT MP'S GEOMETRY. Relaxing first would compare
            # E_MLIP(x_MLIP) against E_DFT(x_MP) and fold the geometry error
            # into the energy error irreversibly (D069).
            static = self.engine.single_point(atoms)
            if not static.ok:
                store.update_reference_entry(int(row["id"]), state="failed",
                                             fail_reason=static.error[:200])
                continue

            relaxed = self.engine.relax(atoms) if self.cfg.campaign.reference.relax_with_mlip else None

            store.update_reference_entry(
                int(row["id"]),
                e_mlip_static=static.e_per_atom,
                e_mlip_relaxed=relaxed.e_per_atom if relaxed and relaxed.ok else None,
                volume_drift=relaxed.volume_drift if relaxed and relaxed.ok else None,
                state="static_done" if relaxed is None else "relaxed",
            )
            evaluated += 1

            counts: dict[str, int] = {}
            for symbol in atoms.get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            points.append(ParityPoint(
                label=row["mp_id"], counts=counts,
                e_mlip_per_atom=static.e_per_atom,
                e_dft_per_atom=row["e_dft_raw"],
                volume_mlip=relaxed.volume_after if relaxed and relaxed.ok else None,
                volume_dft=relaxed.volume_before if relaxed and relaxed.ok else None,
            ))

        if not points:
            return StageReport(stage=self.name, claimed=0,
                               note=f"no usable points ({missing} structures unavailable)")

        thresholds = self.cfg.campaign.calibrate.mp.thresholds
        report = build_report(
            points,
            mae_max=thresholds.mae_e_per_atom,
            # 4a has no DFT hull to compare against -- that is 4b's job -- so the
            # hull bound is left wide here rather than being invented.
            mae_hull_max=float("inf"),
            spearman_min=thresholds.spearman_min,
            volume_drift_max=thresholds.max_volume_drift,
        )
        self.last_report = report

        store.add_calibration(
            kind="mp", n_points=report.n, verdict=report.verdict,
            mae_e_per_atom=report.mae_e_per_atom,
            spearman=report.spearman,
            volume_drift=report.mean_volume_drift,
            detail=report.render()[:4000],
        )

        note = f"{report.verdict.upper()}: MAE {report.mae_e_per_atom * 1000:.0f} meV/atom"
        if report.spearman is not None:
            note += f", Spearman {report.spearman:.3f}"
        if missing:
            note += f"; {missing} structure(s) unavailable"
        return StageReport(stage=self.name, claimed=evaluated,
                           reconciled=report.n, note=note)

    def _structures(self, store: Store) -> dict:
        out: dict = {}
        for chemsys in store.chemsystems():
            try:
                out.update(fetch_structures(chemsys))
            except (ReferenceError, Exception):
                continue
        return out


def _to_atoms(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)
