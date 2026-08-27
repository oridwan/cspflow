"""Convex hulls, and the four ways one silently gives the wrong answer.

`e_above_hull` is the number the whole campaign is ranked on, and it is
unusually easy to compute confidently and wrongly. Four guards, each of which
corresponds to a failure that produces a plausible number rather than an error:

1.  **Mixed energy scales.** MP corrected, MP raw and MLIP energies are three
    different scales (Stage 3b). Mixing them inside one hull shifts results by
    O(0.1-1 eV/atom) -- enough to move a candidate from on-the-hull to a few
    tenths above it. Refused, as a hard error.

2.  **A missing elemental reference.** A hull over Sm-Fe with no elemental Sm
    entry is not a hull over Sm-Fe; pymatgen will happily build a diagram from
    whatever it is given, and every `e_above_hull` from it is measured against
    the wrong lower boundary. Refused.

3.  **Mixed functionals.** MP's own data for one chemistry can contain GGA and
    GGA+U entries that share no common energy zero on the raw scale. Refused
    unless the caller says which it wants.

4.  **Duplicate compositions from different provenance.** Two entries for the
    same composition at different energies is not an error -- polymorphs are
    real -- but two entries for the same *material* from two sources is a
    double count, and the lower one silently defines the hull. Reported.

Everything here is pure computation over entries handed in, so it is testable
offline with no MP key, no network and no DFT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from ..chem import canonical_formula, chemsys as chemsys_of

EnergyScale = Literal["raw", "mp_corrected"]


class HullError(Exception):
    """A hull that would have produced a number that looks right and is not."""


@dataclass(frozen=True)
class Entry:
    """One point on the hull: a composition and a total energy on one scale."""

    label: str
    counts: dict[str, int]
    energy: float                      # TOTAL energy for `counts`, not per atom
    scale: EnergyScale = "raw"
    source: str = ""                   # 'mp' | 'ours' | 'mlip'
    run_type: str = ""                 # 'GGA' | 'GGA+U' | ''
    structure_id: int | None = None

    @property
    def n_atoms(self) -> int:
        return sum(self.counts.values())

    @property
    def e_per_atom(self) -> float:
        return self.energy / self.n_atoms

    @property
    def elements(self) -> frozenset[str]:
        return frozenset(e for e, n in self.counts.items() if n > 0)

    @property
    def is_elemental(self) -> bool:
        return len(self.elements) == 1

    @property
    def formula(self) -> str:
        return canonical_formula(self.counts)


@dataclass
class HullResult:
    """Where every entry sits, and what the hull was built from."""

    e_above_hull: dict[str, float] = field(default_factory=dict)
    formation_energy: dict[str, float] = field(default_factory=dict)
    stable: set[str] = field(default_factory=set)
    scale: EnergyScale = "raw"
    chemsys: str = ""
    n_entries: int = 0
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"hull {self.chemsys} on the {self.scale} scale, "
                 f"{self.n_entries} entries, {len(self.stable)} stable"]
        for label in sorted(self.e_above_hull, key=lambda k: self.e_above_hull[k])[:10]:
            mark = "*" if label in self.stable else " "
            lines.append(f"  {mark} {label:<20} {self.e_above_hull[label]:+.4f} eV/atom")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def assert_one_scale(entries: Sequence[Entry]) -> EnergyScale:
    """Every entry on one energy scale, or refuse.

    A warning would not do. The result of mixing is a hull that is wrong by a
    few tenths of an eV per atom, which is the same order as the thresholds the
    campaign filters on -- so the wrong answer is indistinguishable from a right
    one by inspection.
    """
    if not entries:
        raise HullError("cannot build a hull from no entries")
    scales = {e.scale for e in entries}
    if len(scales) > 1:
        by_scale = {s: [e.label for e in entries if e.scale == s][:3] for s in sorted(scales)}
        raise HullError(
            f"entries span {len(scales)} energy scales: {by_scale}. MP corrected, MP raw "
            f"and MLIP energies do not share a zero; a hull mixing them is wrong by "
            f"O(0.1-1 eV/atom), which is the same size as the filter threshold. "
            f"Choose one via reference.energy_scale and convert, or drop the others."
        )
    return scales.pop()


def assert_one_functional(entries: Sequence[Entry]) -> str:
    """One `run_type`, or refuse.

    MP's data for a single chemistry can mix GGA and GGA+U entries, and on the
    raw scale those share no common energy zero. For O/F-free magnets it does
    not arise, which is exactly why it has never bitten -- and why it will, the
    first time someone runs an oxide.
    """
    kinds = {e.run_type for e in entries if e.run_type}
    # An entry with no run_type is "not recorded", which is different from
    # "conflicts". It cannot be checked, so it is reported by build_hull rather
    # than refused here -- refusing would make every hand-built entry illegal.
    if len(kinds) > 1:
        counts = {k: sum(1 for e in entries if e.run_type == k) for k in sorted(kinds)}
        raise HullError(
            f"entries mix functionals {counts}. GGA and GGA+U raw total energies share "
            f"no common zero, so a hull built across them is not comparable. "
            f"Set reference.functionals to pick one."
        )
    return next(iter(kinds), "")


def assert_elemental_references(entries: Sequence[Entry]) -> None:
    """Every element present must have an elemental entry.

    Without one, the hull's lower boundary in that direction is defined by
    whatever compound happens to be lowest, and every `e_above_hull` measured
    against it is wrong -- with no symptom whatsoever. pymatgen will build the
    diagram regardless.
    """
    present: set[str] = set()
    for entry in entries:
        present |= entry.elements
    elemental = {next(iter(e.elements)) for e in entries if e.is_elemental}
    missing = sorted(present - elemental)
    if missing:
        raise HullError(
            f"no elemental reference for {missing} among {len(entries)} entries "
            f"(elements present: {sorted(present)}). Formation energies are measured "
            f"from the elemental references, so without them the hull's boundary is "
            f"set by whichever compound happens to be lowest and every e_above_hull "
            f"is wrong with no visible symptom."
        )


def find_duplicates(entries: Sequence[Entry]) -> list[str]:
    """Same composition from more than one source.  Reported, not refused.

    Two entries at one composition are legitimate -- polymorphs are real, and
    the hull should see both. Two entries for the same *material* from two
    sources are a double count, and the lower one silently defines the hull.
    Only the caller knows which it has, so this reports rather than decides.
    """
    by_formula: dict[str, list[Entry]] = {}
    for entry in entries:
        by_formula.setdefault(entry.formula, []).append(entry)

    notes = []
    for formula, group in sorted(by_formula.items()):
        sources = {e.source for e in group if e.source}
        if len(group) > 1 and len(sources) > 1:
            spread = max(e.e_per_atom for e in group) - min(e.e_per_atom for e in group)
            notes.append(
                f"{formula} appears from {sorted(sources)} with a spread of "
                f"{spread:.4f} eV/atom; the lower one defines the hull"
            )
    return notes


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def build_hull(entries: Iterable[Entry], *, strict_functional: bool = True) -> HullResult:
    """Place every entry on the convex hull of the set.

    Uses pymatgen's `PhaseDiagram`, which is well-tested and which nobody should
    reimplement. Everything above it is the part that is easy to get wrong.
    """
    entries = list(entries)
    scale = assert_one_scale(entries)
    if strict_functional:
        assert_one_functional(entries)
    assert_elemental_references(entries)

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    computed = []
    for entry in entries:
        computed.append(ComputedEntry(
            composition=Composition(entry.counts),
            energy=entry.energy,
            entry_id=entry.label,
        ))

    try:
        diagram = PhaseDiagram(computed)
    except Exception as exc:
        raise HullError(f"pymatgen could not build the phase diagram: {exc}") from exc

    warnings = find_duplicates(entries)
    stated = {e.run_type for e in entries if e.run_type}
    unstated = sum(1 for e in entries if not e.run_type)
    if stated and unstated:
        warnings.append(
            f"{unstated} of {len(entries)} entries record no run_type while others say "
            f"{sorted(stated)}; those cannot be checked for functional consistency"
        )

    result = HullResult(
        scale=scale,
        chemsys=chemsys_of({e for entry in entries for e in entry.elements}),
        n_entries=len(entries),
        warnings=warnings,
    )
    for computed_entry in computed:
        label = str(computed_entry.entry_id)
        above = float(diagram.get_e_above_hull(computed_entry))
        result.e_above_hull[label] = above
        result.formation_energy[label] = float(
            diagram.get_form_energy_per_atom(computed_entry)
        )
        if above <= 1e-9:
            result.stable.add(label)
    return result


def place_on_hull(candidate: Entry, reference: Sequence[Entry]) -> float:
    """`e_above_hull` for one candidate against a fixed reference set.

    Kept separate from `build_hull` because the reference set is cached across
    campaigns (keyed on chemsys + settings_hash) while candidates arrive
    continuously. Rebuilding the diagram per candidate would be correct and
    slow; this is the same thing, spelled so the caching is possible.
    """
    return build_hull([*reference, candidate]).e_above_hull[candidate.label]
