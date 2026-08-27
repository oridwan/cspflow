"""Putting the 4f moment back, and being explicit that it is a model.

A frozen-4f POTCAR (`Gd_3`, `Sm_3`, `Nd_3`) buries the f electrons in the core.
That is the right choice for energetics -- GGA without +U pins valence 4f at the
Fermi level and produces nonsense -- and it costs exactly one thing: the rare
earth comes back with no moment.  The campaign's own `Gd2Cr4Co20` starts its Gd
at -7.0 mu_B and finishes at -0.248.

So the saturation magnetisation of a rare-earth magnet cannot be read off a
frozen-4f calculation.  It has to be reassembled:

    M_s  =  M_DFT(transition-metal sublattice)  +  sum_RE  sigma * g_J * J

with `sigma = +1` for a light rare earth (J = L - S, the moment adds to the
transition-metal sublattice) and `sigma = -1` for a heavy one (J = L + S, it
subtracts).  Gd is the half-filled case: L = 0, J = S, and its spin couples
antiparallel to the transition metal, so it belongs with the heavy group -- this
is why GdCo5 is a ferrimagnet with a compensation point.

**This is a model, and it is never merged with the computed number.**
`m_dft_raw` and `m_s_reconstructed` are separate columns, each tagged with the
`f_treatment` that produced it.  Merging them is how a Hund's-rule estimate ends
up quoted as a DFT result -- and it is what would have caught Tb2Fe17 at 39 mu_B
before it reached a plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chem import RARE_EARTHS

# g_J * J for the trivalent ion, in Bohr magnetons: the free-ion Hund's-rule
# ground-state moment. La and Lu have empty and full shells; Eu(III) has J = 0.
GJ_J: dict[str, float] = {
    "La": 0.00,   # 4f0
    "Ce": 2.14,   # 4f1   2F5/2   g=6/7  J=5/2
    "Pr": 3.20,   # 4f2   3H4     g=4/5  J=4
    "Nd": 3.27,   # 4f3   4I9/2   g=8/11 J=9/2
    "Pm": 2.40,   # 4f4   5I4     g=3/5  J=4
    "Sm": 0.71,   # 4f5   6H5/2   g=2/7  J=5/2
    "Eu": 0.00,   # 4f6   7F0     J=0
    "Gd": 7.00,   # 4f7   8S7/2   g=2    J=7/2
    "Tb": 9.00,   # 4f8   7F6     g=3/2  J=6
    "Dy": 10.00,  # 4f9   6H15/2  g=4/3  J=15/2
    "Ho": 10.00,  # 4f10  5I8     g=5/4  J=8
    "Er": 9.00,   # 4f11  4I15/2  g=6/5  J=15/2
    "Tm": 7.00,   # 4f12  3H6     g=7/6  J=6
    "Yb": 4.00,   # 4f13  2F7/2   g=8/7  J=7/2
    "Lu": 0.00,   # 4f14
}

# Light: less than half filled, J = L - S, the total moment adds to the
# transition-metal sublattice. Heavy (Gd included): J = L + S, it subtracts.
LIGHT = frozenset({"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu"})
HEAVY = frozenset({"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"})

assert LIGHT | HEAVY == set(GJ_J) == set(RARE_EARTHS), "the 4f table must cover every RE"


class HundError(Exception):
    pass


def sign_for(element: str) -> int:
    """+1 for a light rare earth, -1 for a heavy one."""
    if element in LIGHT:
        return +1
    if element in HEAVY:
        return -1
    raise HundError(f"{element} is not a rare earth; there is no 4f moment to add")


@dataclass
class Reconstruction:
    """`m_s_reconstructed` and everything that went into it."""

    m_s: float
    tm_moment: float
    contributions: dict[str, float] = field(default_factory=dict)
    f_treatment: str = "frozen"
    warnings: list[str] = field(default_factory=list)

    @property
    def f_contribution(self) -> float:
        return sum(self.contributions.values())

    def render(self) -> str:
        parts = [f"TM {self.tm_moment:+.3f}"]
        parts += [f"{el} {v:+.3f}" for el, v in sorted(self.contributions.items())]
        return f"{self.m_s:.3f} = " + " ".join(parts) + f"  [{self.f_treatment}]"


def reconstruct(tm_moment: float, re_counts: dict[str, int], *,
                f_treatment: str = "frozen") -> Reconstruction:
    """`M_s = M_DFT(TM) + sum_RE sigma * g_J * J * n_RE`.

    `tm_moment` is the transition-metal sublattice moment from the projected
    table, not the cell magnetisation: the cell value already contains whatever
    residual the frozen rare earth carries (-0.248 mu_B per Gd, above), and
    adding a full Hund's-rule moment on top of that double-counts it.
    """
    result = Reconstruction(m_s=float(tm_moment), tm_moment=float(tm_moment),
                            f_treatment=f_treatment)

    if f_treatment == "no_reconstruction":
        result.warnings.append(
            "reconstruct_ms is off: m_s_reconstructed is the transition-metal "
            "sublattice moment with no 4f term added. For a heavy rare earth "
            "that is not the saturation magnetisation -- read m_dft_raw and "
            "know what it omits.")
        return result

    if f_treatment != "frozen":
        result.warnings.append(
            f"f_treatment={f_treatment!r}: the 4f moment is already in the DFT "
            f"result, so reconstruction would double-count it. m_s_reconstructed "
            f"is reported equal to the transition-metal sublattice moment and "
            f"should not be used; read m_dft_raw instead.")
        return result

    for element, n in sorted(re_counts.items()):
        if n <= 0:
            continue
        if element not in GJ_J:
            raise HundError(f"no Hund's-rule moment tabulated for {element}")
        contribution = sign_for(element) * GJ_J[element] * n
        result.contributions[element] = contribution
        result.m_s += contribution

    if not re_counts:
        result.warnings.append("no rare earth in this structure; "
                               "m_s_reconstructed equals the DFT sublattice sum")
    return result


def counts_of(symbols: list[str], only: frozenset[str] | None = None) -> dict[str, int]:
    """Element counts over a per-ion symbol list, optionally restricted."""
    counts: dict[str, int] = {}
    for symbol in symbols:
        if only is not None and symbol not in only:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts
