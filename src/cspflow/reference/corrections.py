"""Stage 3b -- which MP2020 corrections apply, and how big they are.

Three energy scales are in play and they must never be mixed inside one convex
hull (pipeline.md sec.3b):

| scale | where it comes from | pymatgen accessor |
|---|---|---|
| MP corrected | raw PBE/PBE+U **plus** `MaterialsProject2020Compatibility` | `ComputedEntry.energy` |
| MP raw | what VASP actually printed | `ComputedEntry.uncorrected_energy` |
| MLIP | MatterSim et al., trained on **MPtrj = raw, uncorrected** | model output |

The rule is: **MLIP ↔ raw, our own VASP ↔ raw, and neither is ever compared
against mp.org's corrected numbers.** `prescreen.py` and `compute_dft_e_hull.py`
already use `uncorrected_energy` and get this right -- but nowhere say why and
nowhere check it, so the next person to touch them will get it wrong.

For the chemistry being run *today* -- pure RE-TM-TM' intermetallics -- every
MP2020 correction is exactly zero and the question is moot. It stops being moot
the moment a composition contains N, O, Si, H or a halide, which Stage 0 lets a
user type at any time. Sm-Fe-N is an obvious next target. This module exists so
that the warning arrives before the generation, not after the DFT.
"""

from __future__ import annotations

from dataclasses import dataclass

# Elements whose presence triggers an MP2020 *composition* ("anion") correction.
# Read from the installed pymatgen's MP2020Compatibility.yaml rather than
# hardcoded belief -- `load_mp2020_tables()` below re-derives them and the test
# suite asserts the two agree, so a pymatgen upgrade that changes the set is a
# test failure rather than a silent shift in every hull.
ANION_CORRECTED = frozenset(
    "Br Cl F H I N O S Sb Se Si Te".split()
)

# Elements that receive a Hubbard-U correction, but ONLY in a compound that also
# contains O or F. An Fe-Co intermetallic gets nothing; Fe2O3 does.
U_CORRECTED = frozenset("Co Cr Fe Mn Mo Ni V W".split())
U_REQUIRES = frozenset({"O", "F"})


@dataclass(frozen=True)
class CorrectionRisk:
    """Whether a chemical system's energies move between the two MP scales."""

    chemsys: str
    anion_elements: tuple[str, ...] = ()
    u_elements: tuple[str, ...] = ()

    @property
    def affected(self) -> bool:
        return bool(self.anion_elements or self.u_elements)

    def render(self) -> str:
        if not self.affected:
            return (f"{self.chemsys}: no MP2020 correction applies -- corrected and raw "
                    f"energies are identical, so the two scales are interchangeable here")
        parts = []
        if self.anion_elements:
            parts.append(f"composition correction on {', '.join(self.anion_elements)}")
        if self.u_elements:
            parts.append(f"Hubbard-U correction on {', '.join(self.u_elements)}")
        return (f"{self.chemsys}: {' and '.join(parts)}. Corrected and raw MP energies "
                f"DIFFER here, by O(0.1-1 eV/atom). Mixing the scales in one hull can "
                f"move a candidate from on-the-hull to a few tenths above it.")


def correction_risk(chemsys: str) -> CorrectionRisk:
    """Which corrections a chemical system triggers.  Pure lookup, no network."""
    elements = tuple(sorted(e for e in chemsys.split("-") if e))
    anions = tuple(e for e in elements if e in ANION_CORRECTED)
    u_elements: tuple[str, ...] = ()
    if U_REQUIRES & set(elements):
        u_elements = tuple(e for e in elements if e in U_CORRECTED)
    return CorrectionRisk(chemsys=chemsys, anion_elements=anions, u_elements=u_elements)


def load_mp2020_tables() -> dict[str, dict[str, float]]:
    """The actual correction values from the installed pymatgen.

    Read rather than hardcoded so that "how big is the correction" is answered
    by the code that will apply it, at its installed version -- not by a number
    copied into a docstring at some point in the past.
    """
    try:
        import os

        import yaml
        from pymatgen.entries import compatibility

        path = os.path.join(os.path.dirname(compatibility.__file__),
                            "MP2020Compatibility.yaml")
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except (ImportError, OSError) as exc:                 # pragma: no cover
        raise RuntimeError(f"cannot read pymatgen's MP2020 table: {exc}") from exc

    return {
        "anion": dict(data.get("Corrections", {}).get("CompositionCorrections", {})),
        "u": dict(data.get("Corrections", {}).get("GGAUMixingCorrections", {})),
    }


def audit_chemsystems(chemsystems: list[str]) -> tuple[list[CorrectionRisk], str]:
    """Report every chemical system whose two MP scales differ.

    Called at Stage 0 (before generation) and Stage 3 (before the hull), because
    the whole value of the check is that it lands before the expensive part.
    """
    risks = [correction_risk(cs) for cs in chemsystems]
    affected = [r for r in risks if r.affected]
    if not affected:
        return risks, (
            f"all {len(risks)} chemical system(s) are correction-free: MP corrected "
            f"and raw energies are identical, so the energy_scale choice cannot "
            f"change any result here"
        )
    lines = [
        f"{len(affected)} of {len(risks)} chemical system(s) trigger MP2020 corrections, "
        f"so corrected and raw MP energies DIFFER for them:"
    ]
    lines += [f"  {r.render()}" for r in affected[:10]]
    if len(affected) > 10:
        lines.append(f"  ... and {len(affected) - 10} more")
    lines.append(
        "  reference.energy_scale decides which is used; it is recorded in provenance "
        "and folded into ref_set_hash, and a hull may never mix the two."
    )
    return risks, "\n".join(lines)
