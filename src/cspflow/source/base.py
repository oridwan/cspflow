"""Shared vocabulary for Stage 0.

The three source modes differ only in what they emit and where they enter the
funnel (pipeline.md sec.0).  They agree on everything else, and that agreement
lives here: the row shapes, the Z expansion, the rejection log, and the work
estimate that gets printed before a single GPU-minute is spent.

A design note worth stating once, because it shapes every mode below.  Stage 0
**never writes to the database while it is enumerating.**  It builds a complete
`SourceResult` in memory first, and a separate step commits it.  Enumeration
over a chemical space can reject millions of candidates, and a half-written
campaign whose enumeration died partway is far harder to reason about than one
that either exists or does not.  It also makes `--dry-run` free: the same code
path runs, and the commit is simply skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..chem import canonical_formula, chemsys, n_atoms as _n_atoms
from ..config.schema import NStructures, Source, SourceDefaults

# How many concrete examples to keep per rejection reason.  Enough to see the
# shape of what was thrown away, few enough that a 10-million-candidate sweep
# does not accumulate a 10-million-entry list.
MAX_EXAMPLES = 5


class SourceError(Exception):
    """A source definition or input file that cannot be turned into rows."""


@dataclass
class RejectionLog:
    """Why candidates were dropped, aggregated.

    Aggregated rather than per-item because the counts are the useful signal:
    "3.2M rejected: 3.1M below min_fraction, 84k over max_atoms, 12 rare-earth
    limit" tells you your constraints are doing what you meant.  A list of 3.2M
    formulas tells you nothing and costs a gigabyte.
    """

    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, reason: str, what: str = "") -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if what:
            got = self.examples.setdefault(reason, [])
            if len(got) < MAX_EXAMPLES:
                got.append(what)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def render(self, indent: str = "  ") -> str:
        if not self.counts:
            return ""
        lines = []
        for reason, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"{indent}{n:>9,}  {reason}")
            for ex in self.examples.get(reason, []):
                lines.append(f"{indent}           e.g. {ex}")
        return "\n".join(lines)


@dataclass(frozen=True)
class EmittedComposition:
    """One `composition` row waiting to be written."""

    formula: str                 # canonical, reduced
    chemsys: str
    counts: dict[str, int]       # reduced counts, per formula unit
    z: int
    n_atoms: int                 # z * atoms per formula unit
    n_target: int                # structures wanted for this row
    source_name: str
    source_mode: str

    def key(self) -> tuple[str, int]:
        return (self.formula, self.z)


@dataclass
class EmittedStructure:
    """One seeded `structure` row waiting to be written (mode 3 only)."""

    atoms: Any                   # ase.Atoms; typed loosely to keep ASE optional here
    formula: str                 # canonical, reduced
    chemsys: str
    counts: dict[str, int]       # reduced counts
    z: int
    n_atoms: int
    path: str
    content_hash: str
    source_name: str
    source_mode: str
    relax: bool = True


@dataclass
class SourceResult:
    """Everything one source produced, before anything is committed."""

    name: str
    mode: str
    compositions: list[EmittedComposition] = field(default_factory=list)
    structures: list[EmittedStructure] = field(default_factory=list)
    rejected: RejectionLog = field(default_factory=RejectionLog)
    warnings: list[str] = field(default_factory=list)

    @property
    def entry_stage(self) -> str:
        return "screen" if self.mode == "structure_list" else "generate"

    def chemsystems(self) -> list[str]:
        seen = {c.chemsys for c in self.compositions}
        seen |= {s.chemsys for s in self.structures}
        return sorted(seen)

    @property
    def n_target_total(self) -> int:
        """Structures this source will ask Stage 1 to generate."""
        return sum(c.n_target for c in self.compositions)

    def render(self) -> str:
        lines = [f"source '{self.name}' (mode={self.mode}, enters at {self.entry_stage})"]
        if self.compositions:
            lines.append(f"  compositions      {len(self.compositions):,}")
            lines.append(f"  chemical systems  {len(self.chemsystems()):,}")
            lines.append(f"  structures wanted {self.n_target_total:,}")
        if self.structures:
            lines.append(f"  seed structures   {len(self.structures):,}")
            lines.append(f"  chemical systems  {len(self.chemsystems()):,}")
        if self.rejected.total:
            lines.append(f"  rejected          {self.rejected.total:,}")
            lines.append(self.rejected.render(indent="    "))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Z expansion -- shared by modes 1 and 2
# --------------------------------------------------------------------------


def expand_z(
    counts: dict[str, int],
    *,
    defaults: SourceDefaults,
    source_name: str,
    source_mode: str,
    z_override: Iterable[int] | None = None,
    max_atoms_override: int | None = None,
    n_structures_override: NStructures | None = None,
    reject: RejectionLog | None = None,
) -> list[EmittedComposition]:
    """One reduced formula -> its `composition` rows, one per admissible Z.

    `counts` must already be reduced.  A Z whose cell exceeds `max_atoms` is
    dropped with a reason rather than silently clipped, because silently
    clipping turns "I asked for Z up to 4" into "I got Z up to 2" with no
    record of it.
    """
    formula = canonical_formula(counts)
    system = chemsys(counts)
    per_fu = _n_atoms(counts)
    cap = max_atoms_override if max_atoms_override is not None else defaults.max_atoms
    n_struct = n_structures_override or defaults.n_structures
    z_values = list(z_override) if z_override is not None else defaults.z.values()

    admissible = []
    for z in z_values:
        total = z * per_fu
        if total > cap:
            if reject is not None:
                reject.add(f"cell over max_atoms ({cap})", f"{formula} Z={z} -> {total} atoms")
            continue
        admissible.append((z, total))

    if not admissible:
        return []

    targets = [n_struct.target_for(total) for _, total in admissible]

    if defaults.n_structures_scope == "total":
        targets = _split_total(n_struct, admissible, targets)

    return [
        EmittedComposition(
            formula=formula, chemsys=system, counts=dict(counts), z=z, n_atoms=total,
            n_target=target, source_name=source_name, source_mode=source_mode,
        )
        for (z, total), target in zip(admissible, targets)
        if target > 0
    ]


def _split_total(
    n_struct: NStructures,
    admissible: list[tuple[int, int]],
    per_z_targets: list[int],
) -> list[int]:
    """`n_structures_scope: total` -- one budget for the formula, split across Z.

    Defined explicitly because "total" has more than one reasonable reading and
    an undocumented choice here silently changes campaign size:

      * `mode: fixed`    -> the budget is `count`, full stop.
      * `mode: per_atom` -> the budget is what the LARGEST cell alone would have
        cost.  Reading it as the smallest would make `z: [1, 4]` cheaper than
        `z: [4, 4]`, which is backwards; reading it as the sum is just `per_z`
        again under another name.

    The budget is then split in proportion to the per-Z targets, by largest
    remainder, so bigger cells get more samples.  Every row keeps at least 1:
    a Z the user asked for should appear in the campaign or be rejected with a
    reason, never be present with a budget of zero.
    """
    if n_struct.mode.value == "fixed":
        budget = int(n_struct.count or 0)
    else:
        budget = max(per_z_targets)

    k = len(per_z_targets)
    budget = max(budget, k)
    weight_sum = sum(per_z_targets) or k
    exact = [budget * t / weight_sum for t in per_z_targets] if weight_sum else [budget / k] * k

    floors = [max(1, int(x)) for x in exact]
    short = budget - sum(floors)
    if short > 0:
        order = sorted(range(k), key=lambda i: -(exact[i] - int(exact[i])))
        for i in range(short):
            floors[order[i % k]] += 1
    return floors
