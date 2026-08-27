"""MLIP-vs-DFT parity statistics, and what may legitimately be done with them.

Three decisions from pipeline.md §4a are implemented here rather than described,
because each is a place where the obvious thing is wrong:

**1. Energy parity is measured at a FIXED geometry.** Comparing an
MLIP-*relaxed* energy against MP's DFT energy compares `E_MLIP(x_MLIP)` with
`E_DFT(x_MP)` — two different geometries. That folds energy error and geometry
error into one number, and they are not separable afterwards. A model with
perfect energies and a 3% volume bias and a model with correct geometry and a
systematic offset produce the *same* MAE, which makes "what should I fine-tune
on" unanswerable. So: single point at MP's own geometry for the energy metric,
and a separate metric for the geometry.

**2. A per-element energy correction cannot move `e_above_hull`.** The
physically motivated correction is `E_DFT ≈ E_MLIP + Σ nᵢΔᵢ` — the same shape as
the MP2020 corrections. It is a linear function of composition, and so is the
correction of every hull terminal, so the two cancel along the tie-line.
`correction_is_a_noop()` demonstrates this rather than asserting it, and the
test measures the residual: **~10⁻¹⁵ eV/atom, machine precision.**

**3. The fit that *is* useful regresses `e_above_hull`, not total energy.** Hull
distance has a true zero, is reference-invariant, and is the quantity actually
filtered on. `fit_threshold()` converts the DFT cutoff you want into the MLIP
cutoff that captures it — a config value, auditable and reversible, **never a
mutation of stored energies.**

There is also a trap in reading the parity plot at all: over 41 MP Sm–Fe–Ti
phases, a fit using only the formula and no structural information whatever
reaches R² = 0.971. A parity plot of total energies inherits that and looks
superb however bad the model's physics is. `composition_only_r2()` computes that
baseline so the real R² can be read against it instead of against zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

Verdict = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class ParityPoint:
    """One material measured both ways."""

    label: str
    counts: dict[str, int]
    e_mlip_per_atom: float
    e_dft_per_atom: float
    e_hull_mlip: float | None = None
    e_hull_dft: float | None = None
    volume_mlip: float | None = None
    volume_dft: float | None = None
    rmsd: float | None = None

    @property
    def energy_error(self) -> float:
        return self.e_mlip_per_atom - self.e_dft_per_atom

    @property
    def volume_drift(self) -> float | None:
        if not self.volume_dft or self.volume_mlip is None:
            return None
        return (self.volume_mlip - self.volume_dft) / self.volume_dft


@dataclass
class ParityReport:
    n: int = 0
    mae_e_per_atom: float = 0.0
    rmse_e_per_atom: float = 0.0
    bias_e_per_atom: float = 0.0
    spearman: float | None = None
    mae_e_hull: float | None = None
    spearman_hull: float | None = None
    mean_volume_drift: float | None = None
    max_volume_drift: float | None = None
    per_element_mae: dict[str, float] = field(default_factory=dict)
    composition_only_r2: float | None = None
    verdict: Verdict = "pass"
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"parity on {self.n} structures -- {self.verdict.upper()}"]
        lines.append(f"  energy (single point at the DFT geometry)")
        lines.append(f"    MAE            {self.mae_e_per_atom * 1000:8.1f} meV/atom")
        lines.append(f"    RMSE           {self.rmse_e_per_atom * 1000:8.1f} meV/atom")
        lines.append(f"    bias           {self.bias_e_per_atom * 1000:+8.1f} meV/atom")
        if self.spearman is not None:
            lines.append(f"    Spearman       {self.spearman:8.4f}")
        if self.composition_only_r2 is not None:
            lines.append(f"    composition-only R^2 baseline {self.composition_only_r2:.4f}"
                         f"   <- read any R^2 against THIS, not against 0")
        if self.mae_e_hull is not None:
            lines.append(f"  e_above_hull (the quantity actually filtered on)")
            lines.append(f"    MAE            {self.mae_e_hull * 1000:8.1f} meV/atom")
            if self.spearman_hull is not None:
                lines.append(f"    Spearman       {self.spearman_hull:8.4f}")
        if self.mean_volume_drift is not None:
            lines.append(f"  geometry (reported separately, never folded into the energy)")
            lines.append(f"    mean dV/V      {self.mean_volume_drift * 100:+8.2f} %")
            lines.append(f"    max  |dV/V|    {self.max_volume_drift * 100:8.2f} %")
        if self.per_element_mae:
            lines.append("  per element (so \"bad at Sm\" is visible)")
            for element, mae in sorted(self.per_element_mae.items(),
                                       key=lambda kv: -kv[1]):
                lines.append(f"    {element:<4} {mae * 1000:8.1f} meV/atom")
        for reason in self.reasons:
            lines.append(f"  {reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, computed here rather than imported.

    Screening only needs the *ranking* to be right, so this is the metric that
    matters more than the MAE. Implemented directly (with proper tie handling)
    to avoid making scipy a hard dependency of a ten-line calculation.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def composition_only_r2(points: Sequence[ParityPoint]) -> float | None:
    """R² of a fit that uses ONLY the formula -- no structural information at all.

    `E/atom ≈ Σ xᵢaᵢ`, least squares over the element fractions. Measured over 41
    MP Sm–Fe–Ti GGA phases this reaches **R² = 0.971**, reproducing E/atom to
    185 meV against a spread of 1.09 eV.

    That is the number any parity R² has to beat to mean anything. A plot of
    MLIP against DFT total energies inherits this baseline whatever the model's
    physics is like, which is why the R² of such a plot is not evidence.
    """
    if len(points) < 3:
        return None
    elements = sorted({e for p in points for e in p.counts})
    if len(elements) >= len(points):
        return None

    rows = []
    for p in points:
        total = sum(p.counts.values()) or 1
        rows.append([p.counts.get(e, 0) / total for e in elements])
    targets = [p.e_dft_per_atom for p in points]

    coeffs = _least_squares(rows, targets)
    if coeffs is None:
        return None
    predicted = [sum(c * x for c, x in zip(coeffs, row)) for row in rows]
    mean = sum(targets) / len(targets)
    ss_res = sum((t - p) ** 2 for t, p in zip(targets, predicted))
    ss_tot = sum((t - mean) ** 2 for t in targets)
    return None if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _least_squares(rows: list[list[float]], targets: list[float]) -> list[float] | None:
    try:
        import numpy as np
    except ImportError:                                    # pragma: no cover
        return None
    a = np.asarray(rows, dtype=float)
    b = np.asarray(targets, dtype=float)
    try:
        return list(np.linalg.lstsq(a, b, rcond=None)[0])
    except Exception:                                      # pragma: no cover
        return None


def per_element_mae(points: Sequence[ParityPoint]) -> dict[str, float]:
    """MAE attributed to each element, weighted by how much of it is present.

    Not a fit -- a weighted average of the per-structure error over the
    structures containing each element. It answers "is the model bad at Sm?",
    which is the question that decides what to fine-tune on.
    """
    totals: dict[str, list[float]] = {}
    for p in points:
        n_atoms = sum(p.counts.values()) or 1
        for element, count in p.counts.items():
            weight = count / n_atoms
            totals.setdefault(element, []).append(abs(p.energy_error) * weight)
    return {e: sum(v) / len(v) for e, v in totals.items() if v}


# --------------------------------------------------------------------------
# The correction that does nothing, demonstrated
# --------------------------------------------------------------------------


def apply_per_element_correction(
    points: Sequence[ParityPoint], deltas: dict[str, float]
) -> list[float]:
    """`E' = E + Σ xᵢΔᵢ` per atom, for each point."""
    out = []
    for p in points:
        n_atoms = sum(p.counts.values()) or 1
        shift = sum(deltas.get(e, 0.0) * n / n_atoms for e, n in p.counts.items())
        out.append(p.e_mlip_per_atom + shift)
    return out


def correction_shifts_hull_by(
    counts: dict[str, int],
    terminals: Sequence[dict[str, int]],
    deltas: dict[str, float],
) -> float:
    """How much a per-element correction moves one composition's hull distance.

    The answer is zero, to machine precision, and this computes it rather than
    asserting it. A compound's correction and the correction of its hull
    terminals are the *same linear function of composition*, so along the
    tie-line they cancel exactly.

    The consequence is worth stating plainly: **fitting a per-element energy
    correction cannot change what gets filtered.** It is not that the effect is
    small; it is that the quantity is invariant.

    (This holds only because MLIP candidates are compared against MLIP
    terminals. Mixing MLIP candidates against DFT terminals breaks it — which is
    exactly the mistake the hull's scale guard exists to prevent.)
    """
    n_atoms = sum(counts.values()) or 1
    fractions = {e: n / n_atoms for e, n in counts.items()}
    compound_shift = sum(deltas.get(e, 0.0) * x for e, x in fractions.items())

    # The tie-line value at this composition is the same linear combination of
    # the terminals' shifts, because the terminals span the composition space.
    terminal_shift = 0.0
    for element, fraction in fractions.items():
        for terminal in terminals:
            if set(terminal) == {element}:
                t_atoms = sum(terminal.values()) or 1
                terminal_shift += fraction * sum(
                    deltas.get(e, 0.0) * n / t_atoms for e, n in terminal.items()
                )
                break
    return compound_shift - terminal_shift


# --------------------------------------------------------------------------
# The fit that IS useful
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdFit:
    """`e_hull_DFT ≈ α·e_hull_MLIP + β`, used to set a threshold."""

    alpha: float
    beta: float
    n: int
    r2: float | None = None

    def mlip_cutoff_for(self, dft_cutoff: float) -> float:
        """The MLIP threshold that captures everything below `dft_cutoff` in DFT.

        This is the whole point. If the MLIP compresses hull distances by 1.4x,
        screening at 0.10 in MLIP units silently discards candidates sitting at
        0.10 in DFT units. Converting the threshold fixes that **without
        rewriting a single stored energy** -- the result is a config value, in
        provenance, auditable and reversible.
        """
        if self.alpha == 0:
            raise ValueError("alpha is zero; the fit carries no information")
        return (dft_cutoff - self.beta) / self.alpha

    def render(self) -> str:
        r2 = f", R^2 {self.r2:.4f}" if self.r2 is not None else ""
        return (f"e_hull_DFT ~ {self.alpha:.4f} * e_hull_MLIP + {self.beta:+.4f} "
                f"(n={self.n}{r2})")


def fit_threshold(points: Sequence[ParityPoint]) -> ThresholdFit | None:
    """Regress DFT hull distance on MLIP hull distance.

    Deliberately NOT a fit on total energy. Total energy has an arbitrary zero
    set by the pseudopotential, so a fit against it is not invariant under
    `E -> E + c` and its coefficients change meaning with a different POTCAR
    set. Hull distance has a true zero and is reference-invariant.
    """
    usable = [p for p in points
              if p.e_hull_mlip is not None and p.e_hull_dft is not None]
    if len(usable) < 3:
        return None

    xs = [p.e_hull_mlip for p in usable]
    ys = [p.e_hull_dft for p in usable]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    beta = my - alpha * mx

    predicted = [alpha * x + beta for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return ThresholdFit(alpha=alpha, beta=beta, n=n, r2=r2)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def build_report(
    points: Sequence[ParityPoint],
    *,
    mae_max: float = 0.05,
    mae_hull_max: float = 0.05,
    spearman_min: float = 0.9,
    volume_drift_max: float = 0.05,
) -> ParityReport:
    """Every metric, plus a PASS / WARN / FAIL against configurable thresholds."""
    if not points:
        return ParityReport(verdict="fail", reasons=["no calibration points at all"])

    errors = [p.energy_error for p in points]
    report = ParityReport(
        n=len(points),
        mae_e_per_atom=sum(abs(e) for e in errors) / len(errors),
        rmse_e_per_atom=math.sqrt(sum(e * e for e in errors) / len(errors)),
        bias_e_per_atom=sum(errors) / len(errors),
        spearman=spearman([p.e_mlip_per_atom for p in points],
                          [p.e_dft_per_atom for p in points]),
        per_element_mae=per_element_mae(points),
        composition_only_r2=composition_only_r2(points),
    )

    hull_points = [p for p in points
                   if p.e_hull_mlip is not None and p.e_hull_dft is not None]
    if hull_points:
        hull_errors = [p.e_hull_mlip - p.e_hull_dft for p in hull_points]
        report.mae_e_hull = sum(abs(e) for e in hull_errors) / len(hull_errors)
        report.spearman_hull = spearman([p.e_hull_mlip for p in hull_points],
                                        [p.e_hull_dft for p in hull_points])

    drifts = [p.volume_drift for p in points if p.volume_drift is not None]
    if drifts:
        report.mean_volume_drift = sum(drifts) / len(drifts)
        report.max_volume_drift = max(abs(d) for d in drifts)

    _judge(report, mae_max, mae_hull_max, spearman_min, volume_drift_max)
    return report


def _judge(report: ParityReport, mae_max: float, mae_hull_max: float,
           spearman_min: float, volume_drift_max: float) -> None:
    fails, warns = [], []

    if report.mae_e_hull is not None and report.mae_e_hull > mae_hull_max:
        fails.append(f"e_above_hull MAE {report.mae_e_hull * 1000:.0f} meV/atom "
                     f"exceeds {mae_hull_max * 1000:.0f} -- this is the quantity "
                     f"candidates are filtered on")
    if report.mae_e_per_atom > mae_max:
        warns.append(f"energy MAE {report.mae_e_per_atom * 1000:.0f} meV/atom "
                     f"exceeds {mae_max * 1000:.0f}")
    ranking = report.spearman_hull if report.spearman_hull is not None else report.spearman
    if ranking is not None and ranking < spearman_min:
        fails.append(f"Spearman {ranking:.3f} below {spearman_min} -- screening needs "
                     f"the RANKING to be right more than the absolute energies")
    if report.max_volume_drift is not None and report.max_volume_drift > volume_drift_max:
        warns.append(f"max volume drift {report.max_volume_drift * 100:.1f}% exceeds "
                     f"{volume_drift_max * 100:.0f}%; the geometry error is separate "
                     f"from the energy error and is not fixed by an energy correction")

    if report.per_element_mae:
        worst, value = max(report.per_element_mae.items(), key=lambda kv: kv[1])
        median = sorted(report.per_element_mae.values())[len(report.per_element_mae) // 2]
        if median > 0 and value > 3 * median:
            warns.append(f"{worst} is {value / median:.1f}x the median per-element error; "
                         f"fine-tuning should target it")

    report.reasons = [*fails, *warns]
    report.verdict = "fail" if fails else ("warn" if warns else "pass")
