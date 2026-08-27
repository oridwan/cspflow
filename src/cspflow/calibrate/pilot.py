"""Stage 4b -- the calibration that costs DFT, and the one that decides.

4a compares the MLIP against MP's own DFT and is free.  It is also, for an
MP-trained universal MLIP, close to a self-consistency check: MP phases are near
in-distribution, so 4a catches gross failure and not subtle bias.

The structures this campaign actually cares about are *generated*, frequently in
prototypes absent from MP.  That is the out-of-distribution set, and the only way
to measure the model there is to compute some of them properly.  So 4b sits at
the barrier, costs a pilot set of real DFT, and defaults to blocking.

Three things are measured, and the third is the one people forget:

*   energy parity, MAE and RMSE on E/atom;
*   parity on `e_above_hull` -- the quantity actually filtered on, which is a
    *difference* and so does not inherit the energy MAE;
*   **ranking agreement**: whether the DFT top-N is the MLIP top-N. A model can
    have an excellent MAE and still select the wrong twenty structures, and
    selection is all the MLIP is being asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .parity import ParityPoint, ParityReport, Verdict, build_report, spearman


class PilotError(Exception):
    pass


@dataclass
class PilotReport:
    """A parity report plus what it means for selection."""

    parity: ParityReport
    top_n: int = 0
    top_n_overlap: int = 0
    missed: list[str] = field(default_factory=list)
    verdict: Verdict = "pass"
    reasons: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.parity.n

    @property
    def top_n_fraction(self) -> float | None:
        return self.top_n_overlap / self.top_n if self.top_n else None

    def render(self) -> str:
        lines = [f"pilot calibration on {self.parity.n} structures against OUR DFT "
                 f"-- {self.verdict.upper()}",
                 self.parity.render()]
        if self.top_n:
            fraction = self.top_n_fraction or 0.0
            lines.append(f"  selection (what the MLIP is actually for)")
            lines.append(f"    DFT top-{self.top_n} recovered by the MLIP: "
                         f"{self.top_n_overlap}/{self.top_n} ({fraction * 100:.0f}%)")
            if self.missed:
                lines.append(f"    missed: {', '.join(self.missed[:8])}")
        for reason in self.reasons:
            lines.append(f"  {reason}")
        return "\n".join(lines)


def stratified_sample(labels: Sequence[str], values: Sequence[float],
                      n: int) -> list[str]:
    """`n` labels spread evenly across the range of `values`.

    Not the top `n`.  A pilot drawn from the best-ranked candidates measures the
    model only where it already ranks things highly, and the failure that
    matters is a structure the MLIP puts *low* and DFT puts high -- which such a
    pilot cannot see by construction.

    Deterministic: sorted by value, then every k-th taken.  A campaign rerun
    with the same candidates picks the same pilot, so the gate does not move on
    its own.
    """
    if n <= 0:
        raise PilotError(f"pilot_n must be positive, got {n}")
    if len(labels) != len(values):
        raise PilotError("labels and values must be the same length")
    if not labels:
        return []
    order = sorted(range(len(labels)), key=lambda i: (values[i], labels[i]))
    if n >= len(order):
        return [labels[i] for i in order]
    # Evenly spaced positions including both ends of the range.
    step = (len(order) - 1) / (n - 1) if n > 1 else 0.0
    picked = sorted({order[round(i * step)] for i in range(n)})
    # Rounding can collide; fill from what is left, nearest the gaps.
    if len(picked) < n:
        for index in order:
            if index not in picked:
                picked.append(index)
                if len(picked) == n:
                    break
        picked.sort()
    return [labels[i] for i in picked]


def top_n_agreement(points: Sequence[ParityPoint], n: int) -> tuple[int, list[str]]:
    """How many of DFT's best `n` the MLIP also ranks in its best `n`.

    Ranked on `e_above_hull` where both are known, because that is what the
    filter sorts on; on energy per atom otherwise.
    """
    if not points or n <= 0:
        return 0, []
    use_hull = all(p.e_hull_mlip is not None and p.e_hull_dft is not None
                   for p in points)
    key_mlip = (lambda p: p.e_hull_mlip) if use_hull else (lambda p: p.e_mlip_per_atom)
    key_dft = (lambda p: p.e_hull_dft) if use_hull else (lambda p: p.e_dft_per_atom)

    n = min(n, len(points))
    best_dft = [p.label for p in sorted(points, key=key_dft)[:n]]
    best_mlip = {p.label for p in sorted(points, key=key_mlip)[:n]}
    missed = [label for label in best_dft if label not in best_mlip]
    return n - len(missed), missed


def build_pilot_report(points: Sequence[ParityPoint], *,
                       mae_max: float = 0.05,
                       mae_hull_max: float = 0.05,
                       spearman_min: float = 0.90,
                       top_n: int = 0,
                       top_n_min_fraction: float = 0.5) -> PilotReport:
    """Parity, plus the selection check, plus one verdict over both."""
    parity = build_report(points, mae_max=mae_max, mae_hull_max=mae_hull_max,
                          spearman_min=spearman_min,
                          volume_drift_max=float("inf"))
    report = PilotReport(parity=parity, verdict=parity.verdict,
                         reasons=list(parity.reasons))

    if not points:
        report.verdict = "fail"
        return report

    report.top_n = min(top_n, len(points)) if top_n else 0
    if report.top_n:
        overlap, missed = top_n_agreement(points, report.top_n)
        report.top_n_overlap = overlap
        report.missed = missed
        fraction = overlap / report.top_n
        if fraction < top_n_min_fraction:
            report.verdict = "fail"
            report.reasons.append(
                f"the MLIP recovers only {overlap} of DFT's best {report.top_n} "
                f"({fraction * 100:.0f}%): the energies may be close but the "
                f"selection is not, and selection is what the MLIP is for")
        elif fraction < 1.0 and report.verdict == "pass":
            report.verdict = "warn"
            report.reasons.append(
                f"{len(missed)} of DFT's best {report.top_n} fall outside the "
                f"MLIP's own top {report.top_n}")

    return report


def volume_drift_is_not_measured_here() -> str:
    """Why the pilot has no geometry check, stated where someone will look.

    4a compares against MP's *relaxed* geometry, so a volume drift is
    meaningful. The pilot's DFT starts from the MLIP-relaxed geometry, so a
    comparison of the two volumes measures how far VASP moved from where
    MatterSim left it -- which is a useful number and is not the same quantity.
    It is reported by `analyze`, not gated on here.
    """
    return ("geometry is not gated in 4b: the DFT started from the MLIP's own "
            "relaxed cell, so the two volumes are not independent")
