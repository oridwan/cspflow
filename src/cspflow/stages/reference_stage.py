"""Stage 3 -- fetch the reference phases and place every screened candidate.

In-process: seconds to minutes, no scheduler. The expensive part is the network,
and it is paid once per chemical system across all campaigns because the cache
lives outside any of them.

The stage does two things that are easy to conflate and must not be:

*   It **fetches** MP phases for each chemical system the campaign touches, on
    one `thermo_type` and one energy scale, and stores both energies per entry.
*   It **places** each screened candidate on the hull built from those phases,
    on the MLIP energy scale.

The second is where the subtlety lives. An MLIP energy and an MP raw DFT energy
are not on the same scale either -- MatterSim is trained on MPtrj, which is raw
MP DFT, so they are *close*, but "close" is exactly what Stage 4 exists to
measure rather than assume. The hull placement is therefore recorded as
`hull_type='mlip'`, kept separate from any DFT hull, and never silently compared
against one.
"""

from __future__ import annotations

from ..config.loader import ResolvedConfig
from ..db.store import Store, StructureState
from ..reference.corrections import audit_chemsystems
from ..reference.hull import Entry, HullError, build_hull
from ..reference.mp import ReferenceError, fetch_chemsys
from .base import StageReport, WorkItem


class ReferenceStage:
    name = "reference"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.last_snapshots: dict[str, str] = {}

    def pending(self, store: Store) -> int:
        """Work for this stage: structures to place, or a reference set to fetch.

        Two triggers, not one. The obvious trigger is screened structures with
        no hull placement. The second is a chemical system that has results in
        it and no reference entries at all -- which is the state an *ingested*
        campaign starts in, because its structures arrive already at
        `dft_done` and never pass through screening. Without it the DFT hull in
        `analyze` has nothing to measure against and refuses, correctly, for a
        reason the user cannot act on.
        """
        rows = store.sql.execute(
            "SELECT COUNT(DISTINCT chemsys) n FROM composition").fetchone()
        if not rows or not rows["n"]:
            return 0

        entries = store.sql.execute(
            "SELECT COUNT(*) n FROM reference_entry").fetchone()["n"]
        if not entries and store.count_structures(state=StructureState.dft_done.value):
            return int(rows["n"])

        placed = store.sql.execute("SELECT COUNT(*) n FROM hull").fetchone()["n"]
        screened = store.count_structures(state=StructureState.screened.value)
        return 0 if (screened == 0 or placed >= screened) else screened

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("reference is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("reference is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        reference = self.cfg.campaign.reference
        thermo_type = _thermo_label(reference)
        scale = getattr(reference, "energy_scale", "raw")

        chemsystems = store.chemsystems()
        _, audit = audit_chemsystems(chemsystems)

        fetched, placed, notes = 0, 0, []
        by_chemsys: dict[str, list[Entry]] = {}

        for chemsys in chemsystems:
            try:
                result = fetch_chemsys(chemsys, thermo_type=thermo_type,
                                       energy_scale=scale)
            except ReferenceError as exc:
                notes.append(f"{chemsys}: {exc}")
                continue

            self.last_snapshots[chemsys] = result.snapshot_id
            for entry in result.entries:
                store.add_reference_entry(
                    mp_id=entry.mp_id, chemsys=entry.chemsys,
                    thermo_type=entry.thermo_type, run_type=entry.run_type,
                    formula=entry.formula, n_atoms=entry.n_atoms,
                    e_dft_raw=entry.e_raw_per_atom,
                    e_dft_corrected=entry.e_corrected_per_atom,
                    correction=_correction(entry),
                    snapshot_id=result.snapshot_id, state="fetched",
                )
                fetched += 1
            by_chemsys[chemsys] = result.hull_entries(scale)
            notes.extend(result.warnings)

        placed, place_notes = self._place(store, by_chemsys, scale)
        notes.extend(place_notes)

        note = audit.splitlines()[0] if audit else ""
        if notes:
            note += f"; {len(notes)} note(s)"
        return StageReport(stage=self.name, claimed=fetched, reconciled=placed, note=note)

    def _place(self, store: Store, by_chemsys: dict[str, list[Entry]],
               scale: str) -> tuple[int, list[str]]:
        """Put every screened candidate on its own chemical system's hull."""
        placed, notes = 0, []
        for chemsys, reference in by_chemsys.items():
            candidates = self._candidates(store, chemsys)
            if not candidates:
                continue
            try:
                hull = build_hull([*reference, *candidates])
            except HullError as exc:
                notes.append(f"{chemsys}: {exc}")
                continue

            ref_hash = self.last_snapshots.get(chemsys, "")
            for candidate in candidates:
                sid = candidate.structure_id
                if sid is None:                            # pragma: no cover
                    continue
                store.add_hull(
                    structure_id=sid, hull_type="mlip",
                    energy_scale="raw" if scale == "raw" else "mp_corrected",
                    e_above_hull=hull.e_above_hull[candidate.label],
                    formation_energy=hull.formation_energy.get(candidate.label),
                    ref_set_hash=ref_hash,
                )
                store.update_structure(sid,
                                       e_above_hull_mlip=hull.e_above_hull[candidate.label])
                placed += 1
            notes.extend(hull.warnings)
        return placed, notes

    def _candidates(self, store: Store, chemsys: str) -> list[Entry]:
        """Screened structures in one chemical system, as hull entries.

        The MLIP energy enters as `scale='raw'` because MatterSim is trained on
        MPtrj, which is raw MP DFT. That makes them comparable enough to rank
        on -- and Stage 4 exists precisely to measure how comparable, rather
        than to take it on trust. The placement is labelled `hull_type='mlip'`
        so it can never be mistaken for a DFT hull.
        """
        from ..chem import parse_formula

        out = []
        for row in store.structures(state=StructureState.screened.value):
            energy = row.key_value_pairs.get("mlip_e_per_atom")
            if energy is None:
                continue
            counts = _counts(row)
            if "-".join(sorted(counts)) != chemsys:
                continue
            out.append(Entry(
                label=f"cand-{row.id}", counts=counts,
                energy=float(energy) * sum(counts.values()),
                scale="raw", source="mlip", structure_id=int(row.id),
            ))
        return out


def _counts(row) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in row.toatoms().get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _correction(entry) -> float | None:
    if entry.e_raw_per_atom is None or entry.e_corrected_per_atom is None:
        return None
    return entry.e_corrected_per_atom - entry.e_raw_per_atom


def _thermo_label(reference) -> str:
    """Map the campaign's `thermo_type` enum onto MP's own label."""
    value = getattr(getattr(reference, "thermo_type", None), "value", None)
    return str(value or getattr(reference, "thermo_type", "GGA_GGA+U"))
