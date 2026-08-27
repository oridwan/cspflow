"""Stage 7 -- properties, and the hull on our own energies.

An in-process stage.  Everything it does is arithmetic over finished jobs, so it
runs in the driver in seconds and needs no queue.

Two jobs, and they are separate on purpose.

**Properties** are extracted per structure from its own DFT directory: relaxed
volume, spacegroup at a stated tolerance, the cell magnetisation, the
sublattice split, and -- as a distinct column -- the Hund's-rule reconstruction
of the saturation magnetisation.  See `analysis/hund.py` for why the last one
cannot be the same column as the first.

**The DFT hull** is recomputed for a chemical system whenever a new DFT result
lands in it, so `e_above_hull` is never stale.  This is the one number in the
campaign that is not a property of a structure: it depends on every competing
phase in the same system, including other candidates of ours.  A candidate
placed against a hull that was missing a phase computed an hour later is simply
wrong, and re-placing is cheap.

The hull is built on *our* energies plus the MP reference entries on the same
scale.  Mixing our GGA numbers with MP's corrected ones would put the error of
the correction scheme straight into `e_above_hull`; `reference/hull.py` refuses
that outright, and this stage does not try to talk it round.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..analysis.properties import extract
from ..chem import canonical_formula, chemsys
from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..reference.hull import Entry, HullError, build_hull
from .base import StageReport, WorkItem

# Where the DFT stage recorded the directory it wrote.
DIR_KEY = "dft_dir"
# Set once a structure's properties have been extracted, so a second cycle does
# not re-read a 10 MB OUTCAR for every finished structure in the campaign.
DONE_KEY = "analyzed"


class AnalyzeStage:
    name = "analyze"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg

    # -- what is ready -----------------------------------------------------

    def pending(self, store: Store) -> int:
        return len(self._ready(store))

    @staticmethod
    def _ready(store: Store) -> list[Any]:
        return [row for row in store.structures(state=StructureState.dft_done.value)
                if not row.key_value_pairs.get(DONE_KEY)]

    def claim(self, store: Store, budget: int) -> list[WorkItem]:  # pragma: no cover
        raise AssertionError("analyze is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                               # pragma: no cover
        raise AssertionError("analyze is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items) -> None:    # pragma: no cover
        pass

    # -- the work ----------------------------------------------------------

    def run(self, store: Store) -> StageReport:
        extracted, problems = self._extract_all(store)
        systems, hull_note = self._place_all(store)

        note = f"{len(systems)} chemical system(s) placed"
        if hull_note:
            note += f"; {hull_note}"
        if problems:
            note += f"; {len(problems)} structure(s) with warnings"
        return StageReport(stage=self.name, claimed=extracted, reconciled=len(systems),
                           note=note, pending=self.pending(store))

    def _extract_all(self, store: Store) -> tuple[int, list[str]]:
        treatment = self._f_treatment()
        done, problems = 0, []
        for row in self._ready(store):
            sid = int(row.id)
            directory = row.key_value_pairs.get(DIR_KEY)
            if not directory or not Path(directory).is_dir():
                store.update_structure(
                    sid, **{DONE_KEY: True,
                            "analyze_note": f"no DFT directory recorded"[:200]})
                problems.append(f"{sid}: no DFT directory")
                continue

            props = extract(Path(directory), structure_id=sid,
                            z=int(row.key_value_pairs.get("z", 1) or 1),
                            f_treatment=treatment)
            kv = props.as_kv()
            kv[DONE_KEY] = True
            if props.spacegroup_symbol:
                kv["spacegroup_symbol"] = props.spacegroup_symbol
                kv["symprec"] = props.symprec
            if props.warnings:
                kv["analyze_note"] = "; ".join(props.warnings)[:200]
                problems.append(f"{sid}: {props.warnings[0]}")
            store.update_structure(sid, **kv)

            for key, value in (("m_dft_raw", props.m_dft_raw),
                               ("m_s_reconstructed", props.m_s_reconstructed),
                               ("m_spheres", props.m_spheres),
                               ("volume", props.volume),
                               ("spacegroup", props.spacegroup_number)):
                if value is not None:
                    store.add_property(structure_id=sid, key=key, source="dft",
                                       value=float(value))
            for name, value in props.sublattice.items():
                store.add_property(structure_id=sid, key=f"m_{name}", source="dft",
                                   value=float(value))
            done += 1
        return done, problems

    def _f_treatment(self) -> str:
        dft = self.cfg.campaign.dft
        if dft is None or dft.magnetism is None:
            return "frozen"
        rare_earth = getattr(dft.magnetism, "rare_earth", None)
        treatment = getattr(rare_earth, "f_treatment", None) if rare_earth else None
        return getattr(treatment, "value", treatment) or "frozen"

    # -- the hull ----------------------------------------------------------

    def _place_all(self, store: Store) -> tuple[list[str], str]:
        """Rebuild every chemical system that has a DFT energy in it."""
        ours = self._our_entries(store)
        if not ours:
            return [], ""

        placed: list[str] = []
        notes: list[tuple[str, str]] = []
        for system, entries in sorted(ours.items()):
            reference = self._reference_entries(store, system)
            try:
                result = build_hull([*reference, *entries])
            except HullError as exc:
                notes.append((system, str(exc)))
                continue
            for entry in entries:
                if entry.structure_id is None:
                    continue
                store.update_structure(
                    int(entry.structure_id),
                    dft_e_above_hull=float(result.e_above_hull[entry.label]),
                    dft_e_formation=float(result.formation_energy[entry.label]),
                )
                store.add_property(structure_id=int(entry.structure_id),
                                   key="dft_e_above_hull", source="dft",
                                   value=float(result.e_above_hull[entry.label]))
            placed.append(system)
        return placed, _summarise(notes)

    @staticmethod
    def _our_entries(store: Store) -> dict[str, list[Entry]]:
        """Our own finished DFT results, grouped by chemical system."""
        out: dict[str, list[Entry]] = {}
        for row in store.structures(state=StructureState.dft_done.value):
            energy = row.key_value_pairs.get("vasp_energy")
            if energy is None:
                continue
            counts: dict[str, int] = {}
            for symbol in row.toatoms().get_chemical_symbols():
                counts[symbol] = counts.get(symbol, 0) + 1
            entry = Entry(label=f"ours-{row.id}", counts=counts, energy=float(energy),
                          scale="raw", source="ours", run_type="GGA",
                          structure_id=int(row.id))
            out.setdefault(chemsys(counts), []).append(entry)
        return out

    @staticmethod
    def _reference_entries(store: Store, system: str) -> list[Entry]:
        """MP entries for `system` and every sub-system of it.

        Sub-systems are not optional: a binary query returns the binaries and
        no elemental end members, and a hull without its elemental references is
        not a hull. `reference_entries(include_subsystems=True)` is where that
        expansion already lives.
        """
        entries = []
        for row in store.reference_entries(chemsys=system, include_subsystems=True):
            counts = _counts_from_formula(row["formula"])
            per_atom = row["e_dft_raw"]
            if per_atom is None or not counts:
                continue
            # `e_dft_raw` is stored per atom; `Entry.energy` is the total for
            # `counts`. Scaling by the parsed formula's own atom count rather
            # than by the stored `n_atoms` keeps the two self-consistent even
            # when the stored formula is the reduced one.
            entries.append(Entry(label=f"mp-{row['id']}", counts=counts,
                                 energy=float(per_atom) * sum(counts.values()),
                                 scale="raw", source="mp",
                                 run_type=row["run_type"] or "GGA"))
        return entries


def _summarise(notes: list[tuple[str, str]]) -> str:
    """Collapse one reason repeated across many systems into one line.

    The hull guards explain themselves at length, which is right the first time
    and noise the fourth: four chemical systems missing their elemental
    references produced four identical paragraphs on one cycle line.
    """
    if not notes:
        return ""
    grouped: dict[str, list[str]] = {}
    for system, reason in notes:
        grouped.setdefault(_short(reason), []).append(system)
    out = []
    for reason, systems in grouped.items():
        listed = ", ".join(sorted(systems)[:3])
        if len(systems) > 3:
            listed += f" and {len(systems) - 3} more"
        out.append(f"{len(systems)} system(s) not placed ({listed}): {reason}")
    return "; ".join(out)


def _short(reason: str) -> str:
    """The first sentence of a guard's message; the rest is its explanation."""
    head = reason.split(". ")[0].strip()
    # Drop the element list and the entry count, so systems that differ only in
    # which elements they are missing group into one line. The systems
    # themselves are named by the caller.
    head = head.split(" among ")[0]
    bracket = head.find("[")
    head = head[:bracket].strip() if bracket > 0 else head
    # A message cut before its list can end on a dangling preposition.
    while head.split() and head.split()[-1] in {"for", "in", "of", "among", "at"}:
        head = head.rsplit(" ", 1)[0]
    return head


def _counts_from_formula(formula: str) -> dict[str, int]:
    from ..chem import parse_formula

    try:
        return parse_formula(formula)
    except Exception:                                        # pragma: no cover
        return {}
