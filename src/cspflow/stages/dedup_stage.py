"""Deduplication of screened structures.

**A deviation from pipeline.md, and the reason for it.** The plan groups dedup
inside Stage 2 (`screen` = MLIP relax + dedup). Operationally it cannot live
there: `screen` is a *submitted* stage whose unit of work is a chunk of a few
hundred structures, and duplicates are global -- two copies of one material can
easily land in different array tasks, or in tasks submitted cycles apart. A
per-chunk dedup would deduplicate within chunks and miss exactly the collisions
that matter.

So it is its own in-process stage, between `screen` and `reference` in the
funnel. That placement is not arbitrary either: it must run before the hull is
built, because fifty copies of one structure on a hull do not change the hull's
shape but do change every count, every "how many candidates survived", and every
per-composition budget downstream.

Generated structures are deduplicated by default -- that is the entire point of
Stage 2. **Seeds are not**, unless asked: a curated input list is a place where
two near-identical entries are usually deliberate, and silently merging them
loses which one survived (pipeline.md §0.3).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config.loader import ResolvedConfig
from ..db.store import Origin, Store, StructureState
from .base import StageReport


class DedupStage:
    name = "dedup"
    role = "cpu"
    in_process = True

    def __init__(self, cfg: ResolvedConfig, include_seeds: bool = False) -> None:
        self.cfg = cfg
        self.include_seeds = include_seeds

    def pending(self, store: Store) -> int:
        return len(self._eligible(store))

    def claim(self, store, budget):                        # pragma: no cover
        raise AssertionError("dedup is an in-process stage; the driver calls run()")

    def build(self, items, workdir):                       # pragma: no cover
        raise AssertionError("dedup is an in-process stage; nothing is submitted")

    def reconcile(self, store, job_row, status, items):    # pragma: no cover
        pass

    def run(self, store: Store) -> StageReport:
        rows = self._eligible(store)
        if not rows:
            return StageReport(stage=self.name, note="nothing to compare")

        matcher = self._matcher()
        if matcher is None:                                # pragma: no cover
            return StageReport(stage=self.name,
                               note="pymatgen unavailable; dedup skipped")

        by_formula: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            by_formula[row.toatoms().get_chemical_formula()].append(row)

        dropped, groups, kept_seeds = 0, 0, 0
        for formula, members in sorted(by_formula.items()):
            if len(members) < 2:
                continue
            for survivor, duplicates in self._group(matcher, members):
                if not duplicates:
                    continue
                groups += 1
                for row in duplicates:
                    if row.get("origin") == Origin.seed.value and not self.include_seeds:
                        # Two seeds that match are a *result* -- a relaxed and an
                        # unrelaxed copy of one prototype, say. Recorded, kept.
                        kept_seeds += 1
                        store.add_filter_event(
                            structure_id=int(row.id), gate="dedup:seed_collision",
                            passed=True, detail=f"matches structure {survivor.id}; kept",
                        )
                        continue
                    store.set_structure_state(
                        int(row.id), StructureState.deduped,
                        duplicate_of=int(survivor.id),
                    )
                    store.add_filter_event(
                        structure_id=int(row.id), gate="dedup",
                        passed=False, detail=f"duplicate of structure {survivor.id}",
                    )
                    dropped += 1

        note = f"{groups} duplicate group(s)"
        if kept_seeds:
            note += f", {kept_seeds} seed collision(s) reported and kept"
        return StageReport(stage=self.name, claimed=dropped, reconciled=len(rows),
                           note=note)

    # -- internals ---------------------------------------------------------

    def _eligible(self, store: Store) -> list[Any]:
        return [r for r in store.structures(state=StructureState.screened.value)
                if "duplicate_of" not in r.key_value_pairs]

    def _matcher(self):
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
        except ImportError:                                # pragma: no cover
            return None
        cfg = self.cfg.campaign.screen.dedup.matcher
        return StructureMatcher(ltol=cfg.ltol, stol=cfg.stol, angle_tol=cfg.angle_tol)

    def _group(self, matcher, members: list[Any]):
        """Group by structural identity; the lowest MLIP energy survives each group.

        Uses `StructureMatcher.group_structures`, which is materially faster
        than the naive O(N^2) `fit` loop -- it fingerprints first and only
        compares within candidate groups. At a few hundred structures per
        formula the difference is minutes.

        The survivor is the lowest-energy member rather than the first, so which
        structure survives does not depend on the order rows came out of the
        database.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        index = {}
        structures = []
        for row in members:
            structure = AseAtomsAdaptor.get_structure(row.toatoms())
            index[id(structure)] = row
            structures.append(structure)

        for group in matcher.group_structures(structures):
            rows = [index[id(s)] for s in group]
            rows.sort(key=lambda r: r.key_value_pairs.get("mlip_e_per_atom", float("inf")))
            yield rows[0], rows[1:]
