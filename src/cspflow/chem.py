"""Formula parsing, reduction and canonicalisation.

Deliberately dependency-free.  Three reasons, in order of how much they cost
when violated:

1.  **The canonical formula is a database key.**  ``composition`` is unique on
    ``(formula, z, source_name)``, so if two code paths spell the same
    composition differently -- ``Gd1Co10Cr2`` from a directory name and
    ``Co10Cr2Gd1`` from an enumerator -- they become two rows for one material
    and every per-composition count downstream is wrong.  Canonicalisation
    therefore has to be one function, and it has to give the same answer in
    every environment.  pymatgen's ``reduced_formula`` orders by
    electronegativity, which is prettier but depends on pymatgen's data tables
    and version; alphabetical ordering depends on nothing.

2.  A malformed formula must fail loudly at the point of entry, not turn into
    a plausible-looking wrong composition further down.

3.  ``csp ingest`` and ``csp doctor`` must work in an environment without the
    scientific stack installed, so that "the tooling is broken" and "the
    science env is broken" are distinguishable failures.
"""

from __future__ import annotations

from math import gcd
from typing import Iterable

# Rare earths.  Sc and Y are excluded on purpose: they are group-3 metals with
# no f electrons, so neither the ``max_rare_earth`` guard (Stage 0.1) nor the 4f
# treatment (Stage 3d) applies to them.  This is the single definition; the
# config schema re-exports it.
RARE_EARTHS: frozenset[str] = frozenset(
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
)

# Every element symbol, so a typo like `Xx2O3` or a lower-cased `fe` is caught
# at parse time rather than becoming a phantom species that only VASP rejects,
# hours later, with a POTCAR error.
ELEMENTS: frozenset[str] = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg
    Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)


class ChemError(ValueError):
    """A formula that cannot be understood, with the position that broke it."""


def parse_formula(formula: str, *, validate: bool = True) -> dict[str, int]:
    """``'Gd1Co10Cr2'`` -> ``{'Gd': 1, 'Co': 10, 'Cr': 2}``.

    Supports nested parentheses (``Ca3(PO4)2``) and a bare symbol as count 1.
    Counts must be integers: a fractional formula means a disordered structure,
    which Stage 0.3 rejects at ingest rather than carrying to DFT input
    generation where it fails anyway (see pipeline.md sec.0.3).
    """
    counts, pos = _parse_group(formula, 0, validate)
    if pos != len(formula):
        raise ChemError(
            f"cannot parse formula {formula!r}: unexpected {formula[pos]!r} at position {pos}"
        )
    if not counts:
        raise ChemError(f"no elements in formula {formula!r}")
    return counts


def _parse_group(s: str, pos: int, validate: bool) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    while pos < len(s):
        ch = s[pos]
        if ch == ")":
            break
        if ch == "(":
            inner, pos = _parse_group(s, pos + 1, validate)
            if pos >= len(s) or s[pos] != ")":
                raise ChemError(f"unbalanced '(' in formula {s!r}")
            pos += 1
            mult, pos = _read_int(s, pos)
            for el, n in inner.items():
                counts[el] = counts.get(el, 0) + n * mult
            continue
        if ch in " \t_-":            # tolerate separators, e.g. 'Fe-Co' spacing
            pos += 1
            continue
        if not ch.isupper():
            raise ChemError(
                f"cannot parse formula {s!r}: expected an element symbol at position "
                f"{pos}, found {ch!r} (element symbols start with a capital letter)"
            )
        end = pos + 1
        while end < len(s) and s[end].islower():
            end += 1
        symbol = s[pos:end]
        if validate and symbol not in ELEMENTS:
            raise ChemError(f"{symbol!r} in formula {s!r} is not an element symbol")
        n, pos = _read_int(s, end)
        counts[symbol] = counts.get(symbol, 0) + n
    return counts, pos


def _read_int(s: str, pos: int) -> tuple[int, int]:
    """Read an optional integer multiplier; absent means 1."""
    end = pos
    while end < len(s) and s[end].isdigit():
        end += 1
    if end == pos:
        return 1, pos
    return int(s[pos:end]), end


def reduce_counts(counts: dict[str, int]) -> tuple[dict[str, int], int]:
    """``{'Fe': 2, 'Co': 10}`` -> ``({'Fe': 1, 'Co': 5}, 2)``.

    The second value is Z, the number of formula units.  Keeping it means a
    campaign never runs ``FeCo5`` and ``Fe2Co10`` as separate work while still
    being able to say which cell size was requested.
    """
    if not counts:
        raise ChemError("cannot reduce an empty composition")
    if any(n <= 0 for n in counts.values()):
        bad = sorted(el for el, n in counts.items() if n <= 0)
        raise ChemError(f"non-positive counts for {bad}")
    z = 0
    for n in counts.values():
        z = gcd(z, n)
    return {el: n // z for el, n in counts.items()}, z


def canonical_formula(counts: dict[str, int]) -> str:
    """Deterministic formula string: elements alphabetical, counts always shown.

    ``{'Gd': 1, 'Co': 10}`` -> ``'Co10Gd1'``.  The explicit ``1`` is not an
    oversight: it makes the string round-trip through `parse_formula` and makes
    ``Co10Gd1`` and ``Co10Gd`` impossible to both exist as keys.
    """
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def chemsys(counts_or_elements: dict[str, int] | Iterable[str]) -> str:
    """``'-'``-joined sorted element symbols -- the key the funnel groups on."""
    elements = (
        counts_or_elements.keys()
        if isinstance(counts_or_elements, dict)
        else counts_or_elements
    )
    return "-".join(sorted(set(elements)))


def n_rare_earth(counts: dict[str, int]) -> int:
    """How many distinct rare-earth species are present."""
    return sum(1 for el, n in counts.items() if n > 0 and el in RARE_EARTHS)


def n_atoms(counts: dict[str, int]) -> int:
    return sum(counts.values())


def formula_identity(formula: str) -> tuple[str, str, dict[str, int], int]:
    """Parse an arbitrary formula into the canonical row identity.

    Returns ``(canonical_reduced_formula, chemsys, reduced_counts, z)``.  This
    is the single funnel every formula string passes through -- a directory
    name from a legacy campaign, a line in the user's CSV, an enumerated
    candidate -- so that all of them land on the same database key.
    """
    counts = parse_formula(formula)
    reduced, z = reduce_counts(counts)
    return canonical_formula(reduced), chemsys(reduced), reduced, z
