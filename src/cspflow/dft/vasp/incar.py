"""Writing an INCAR, including the four tags that depend on the structure.

The recipe supplies literal tags; this module adds the ones that cannot be
literal — `SYSTEM`, `MAGMOM`, `NBANDS`, `LMAXMIX`, and the `LDAU*` block — and
renders the result. The emitted file is complete and self-describing: every tag
that VASP will act on is in it, with no value left to be inferred.

**`MAGMOM` is where the physics of this project actually lives.** The old/redo
comparison in `/projects/mmi/shuo` differs mainly in whether the rare-earth
moment was initialised parallel or antiparallel to the transition metal, and
that choice was a code edit. Here it is `dft.magnetism.mode`, one named config
value, recorded in provenance:

    none                 no MAGMOM written (ISPIN=1 territory)
    table                the user's own element -> moment mapping
    ferrimagnetic_retm   RE negative, TM positive -- the RE-TM convention
    pymatgen             pymatgen's defaults, for comparison only

`strict` (default true) refuses to write a MAGMOM for a site whose element has
no entry, rather than falling back to a default. A silent default moment is how
a campaign ends up having initialised half its structures the wrong way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ...config.schema import Ldau, Magnetism, RareEarth
from ..recipe import KNOWN_TAGS, tag_value

# The RE-TM ferrimagnetic convention, ported VERBATIM from
# `ter_mag_flow.py:MAGMOM_OVERRIDE` in /projects/mmi/shuo. Its own comment reads:
#
#     RE = negative (4f moments), late 3d TM = positive,
#     early 3d TM = small negative.
#
# The early-3d sign is the part worth not getting wrong. Ti, V and Cr are
# initialised ANTIPARALLEL to Fe/Co/Ni, which is physically motivated -- early
# and late 3d metals couple antiferromagnetically -- and confirmed in the
# campaign's own inputs: `Gd1Co10Cr2` was run with
# `MAGMOM = 2*-7.0 4*-2.0 20*2.0`, i.e. Cr at -2.0 against Co at +2.0.
#
# Magnitudes are starting guesses, not results; VASP relaxes them. The SIGN
# PATTERN is what this table is for, and it is the single choice that dominates
# the difference between the `old` and `redo` campaigns.
FERRI_RETM: dict[str, float] = {
    # rare earths: negative (4f moments)
    "Ce": -1.0, "Pr": -2.0, "Nd": -3.0, "Pm": -4.0, "Sm": -5.0, "Eu": -7.0,
    "Gd": -7.0, "Tb": -6.0, "Dy": -5.0, "Ho": -4.0, "Er": -3.0, "Tm": -2.0,
    "Yb": -1.0,
    # early 3d: small negative, antiparallel to the late 3d
    "Ti": -0.5, "V": -1.0, "Cr": -2.0,
    # Mn sits at the crossover
    "Mn": 0.5,
    # late 3d: positive
    "Fe": 2.2, "Co": 2.0, "Ni": 2.0,
}

# What an element not in the table gets when `strict` is off. 0.6 is pymatgen's
# own fallback and the legacy script's.
DEFAULT_MOMENT = 0.6

class IncarError(Exception):
    pass


@dataclass
class IncarContext:
    """Everything about one structure that the INCAR needs to know."""

    symbols: list[str]                      # per site, in POSCAR order
    formula: str = ""
    zvals: dict[str, float] = field(default_factory=dict)
    f_in_valence: bool = False
    n_electrons: float | None = None


def magmom_for(
    symbols: Sequence[str], magnetism: Magnetism, rare_earth: RareEarth | None = None
) -> list[float] | None:
    """The initial moment for every site, in POSCAR order.

    Returns None for `mode: none`. Raises when `strict` and an element has no
    entry -- because the alternative is a silent default, and a wrong initial
    moment does not announce itself: the SCF converges happily to the wrong
    magnetic state and reports a number.
    """
    if magnetism.mode == "none":
        return None

    table = _table_for(magnetism, rare_earth)
    missing = sorted({s for s in symbols if s not in table
                      and s not in magnetism.site_overrides})
    if missing and magnetism.strict:
        raise IncarError(
            f"magnetism.mode={magnetism.mode!r} has no initial moment for {missing}. "
            f"Refusing to fall back to a default: a wrong initial moment does not "
            f"announce itself -- the SCF converges to the wrong magnetic state and "
            f"reports a number. Add them to dft.magnetism.table, or set "
            f"dft.magnetism.strict: false to accept {DEFAULT_MOMENT} for them."
        )

    return [float(magnetism.site_overrides.get(s, table.get(s, DEFAULT_MOMENT)))
            for s in symbols]


def _table_for(magnetism: Magnetism, rare_earth: RareEarth | None) -> dict[str, float]:
    if magnetism.mode == "table":
        return dict(magnetism.table)
    if magnetism.mode == "ferrimagnetic_retm":
        table = dict(FERRI_RETM)
        if rare_earth is not None and rare_earth.magnetic_order == "ferro":
            # Same magnitudes, RE parallel to TM. The one-line version of the
            # difference between the `old` and `redo` campaigns.
            table = {k: abs(v) for k, v in table.items()}
        elif rare_earth is not None and rare_earth.magnetic_order == "none":
            from ...chem import RARE_EARTHS

            table = {k: (0.0 if k in RARE_EARTHS else v) for k, v in table.items()}
        table.update(magnetism.table)
        return table
    if magnetism.mode == "pymatgen":
        return _pymatgen_table(magnetism)
    raise IncarError(f"unknown magnetism.mode {magnetism.mode!r}")


def _pymatgen_table(magnetism: Magnetism) -> dict[str, float]:
    try:
        from pymatgen.io.vasp.sets import MPRelaxSet  # noqa: F401
        from pymatgen.io.vasp.sets import _load_yaml_config

        config = _load_yaml_config("MPRelaxSet")
        table = dict(config.get("INCAR", {}).get("MAGMOM", {}))
    except Exception:                                      # pragma: no cover
        table = {}
    table.update(magnetism.table)
    return table


def nbands_auto(context: IncarContext, ncore: int = 4) -> int | None:
    """`NBANDS` from the POTCAR ZVALs.  Ported verbatim from `ter_mag_flow.py`:

        nelect = sum(ZVAL[el] * amount)
        nbands = max(int(nelect/2) + max(nions//2, 10), int(0.6 * nelect))
        nbands = ceil(nbands / NCORE) * NCORE

    Its comment says why it exists: "Compute NBANDS from POTCAR ZVAL to avoid
    VASP crash for high-ZVAL systems". Leaving VASP to choose is fine for light
    elements and fails for rare earths, where ZVAL is large enough that VASP's
    own estimate leaves too few empty bands for the smearing to work.

    The `0.6 * nelect` branch is the one that matters for RE-TM: it dominates
    whenever the electron count is high, which is exactly the failing case.

    Rounded up to a multiple of NCORE so VASP does not silently increase it and
    print a warning nobody reads.
    """
    if context.n_electrons is not None:
        nelect = float(context.n_electrons)
    elif context.zvals:
        nelect = sum(context.zvals.get(s, 0.0) for s in context.symbols)
    else:
        return None
    if nelect <= 0:
        return None

    nions = len(context.symbols)
    bands = max(int(nelect / 2) + max(nions // 2, 10), int(0.6 * nelect))
    ncore = max(1, ncore)
    return ((bands + ncore - 1) // ncore) * ncore


def lmaxmix_for(f_in_valence: bool) -> int:
    """6 with 4f in valence, 4 for d-only.  VASP's own default of 2 is wrong for both."""
    return 6 if f_in_valence else 4


def ldau_block(ldau: Ldau, elements: Sequence[str]) -> dict[str, Any]:
    """The LDAU tags, in POSCAR element order.

    Note the coupling the plan flags: turning U on changes which MP entries are a
    valid reference set, because MP's own GGA and GGA+U entries do not share an
    energy zero on the raw scale.
    """
    if not ldau.enabled:
        return {}
    unknown = sorted(set(elements) - set(ldau.u))
    if unknown:
        raise IncarError(
            f"ldau.enabled is true but no U is given for {unknown}. Every element in "
            f"the structure needs an entry (0.0 is a valid answer) -- a missing one "
            f"would silently become 0 and the campaign would mix U and non-U results."
        )
    return {
        "LDAU": ".TRUE.",
        "LDAUTYPE": ldau.ldau_type,
        "LDAUL": [2 if ldau.u.get(e, 0.0) else -1 for e in elements],
        "LDAUU": [float(ldau.u.get(e, 0.0)) for e in elements],
        "LDAUJ": [float(ldau.j.get(e, 0.0)) for e in elements],
        "LDAUPRINT": 1,
    }


def build_incar(
    base: dict[str, Any],
    context: IncarContext,
    *,
    magnetism: Magnetism | None = None,
    rare_earth: RareEarth | None = None,
    ldau: Ldau | None = None,
    nbands: Any = "auto",
    ncore: int = 4,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The recipe's literal tags plus the computed ones, fully resolved."""
    incar: dict[str, Any] = dict(base)

    if context.formula and "SYSTEM" not in incar:
        incar["SYSTEM"] = context.formula

    if "LMAXMIX" not in incar:
        incar["LMAXMIX"] = lmaxmix_for(context.f_in_valence)

    if magnetism is not None and tag_value(incar, "ISPIN", 1) != 1:
        moments = magmom_for(context.symbols, magnetism, rare_earth)
        if moments is not None:
            incar["MAGMOM"] = moments

    if ldau is not None:
        incar.update(ldau_block(ldau, _unique_in_order(context.symbols)))

    if nbands == "auto":
        value = nbands_auto(context, ncore=int(tag_value(incar, "NCORE", ncore) or ncore))
        if value is not None:
            incar["NBANDS"] = value
    elif isinstance(nbands, int):
        incar["NBANDS"] = nbands

    if overrides:
        incar.update(overrides)
    return incar


def _unique_in_order(symbols: Sequence[str]) -> list[str]:
    """POSCAR element order: first appearance, not alphabetical.

    LDAUU and friends are positional -- one value per species in the order the
    POSCAR lists them -- so getting this wrong applies the wrong U to the wrong
    element, silently.
    """
    seen: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.append(s)
    return seen


def render_incar(incar: dict[str, Any], *, comment: str = "") -> str:
    """The INCAR file text.

    MAGMOM is run-length encoded (`4*4.0 1*-5.0`) the way VASP writes it, which
    keeps a 200-atom cell's line readable.
    """
    lines = []
    if comment:
        lines.extend(f"! {line}" for line in comment.splitlines())
    for key in sorted(incar):
        lines.append(f"{key} = {_format(key, incar[key])}")
    return "\n".join(lines) + "\n"


def _format(key: str, value: Any) -> str:
    if isinstance(value, bool):
        # A user writing `LASPH: true` in YAML gets a Python bool here; VASP
        # wants .TRUE./.FALSE. and would reject `True`.
        return ".TRUE." if value else ".FALSE."
    if key.upper() == "MAGMOM" and isinstance(value, (list, tuple)):
        return _run_length(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_format(key, v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _run_length(values: Sequence[float]) -> str:
    if not values:
        return ""
    out, count, current = [], 1, values[0]
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            out.append(f"{count}*{current:g}")
            count, current = 1, value
    out.append(f"{count}*{current:g}")
    return " ".join(out)
