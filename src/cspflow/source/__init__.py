"""Stage 0 -- turn a campaign's `source:` list into database rows.

The three modes differ in what they emit and where they enter the funnel, and
agree on everything else (pipeline.md sec.0.4): all of them end by grouping to
**chemsys**, which is what the expensive part of the pipeline consumes.  That is
why Stage 3's reference cache is keyed on chemsys and shared across campaigns --
a `structure_list` run and a `chemical_space` run that both touch Sm-Fe-Ti pay
for the reference hull once.

Two properties of this layer are worth stating because the rest of the pipeline
depends on them:

*   **Nothing is written until every source has been expanded.**  Enumeration
    can reject millions of candidates and can fail partway; a campaign database
    that either exists complete or does not exist at all is much easier to
    reason about than one stopped mid-enumeration.  It also makes `--dry-run`
    exercise the identical code path.

*   **Cross-source collisions are reported, never silently merged.**  Running a
    seed list alongside a generated sweep is a built-in control group: if your
    own funnel produces SmFe11Ti at the wrong hull energy, you want to learn it
    from the same run.  That only works if a seed reappearing as a generated
    candidate is recorded as a *result* rather than deduplicated away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import Campaign, Source, SourceMode
from ..db.store import Origin, Store, StructureState
from .base import (
    EmittedComposition,
    EmittedStructure,
    RejectionLog,
    SourceError,
    SourceResult,
    expand_z,
)
from .chemical_space import expand_chemical_space
from .composition_list import expand_composition_list
from .structure_list import expand_structure_list

__all__ = [
    "EmittedComposition",
    "EmittedStructure",
    "RejectionLog",
    "SourceError",
    "SourceResult",
    "SourcePlan",
    "expand_all",
    "expand_source",
    "expand_z",
    "write_plan",
]


def expand_source(source: Source, base_dir: Path | None = None) -> SourceResult:
    """Dispatch one source to its mode."""
    if source.mode is SourceMode.chemical_space:
        return expand_chemical_space(source)
    if source.mode is SourceMode.composition_list:
        return expand_composition_list(source, base_dir)
    if source.mode is SourceMode.structure_list:
        return expand_structure_list(
            source, base_dir, default_max_atoms=source.defaults.max_atoms
        )
    raise SourceError(f"unknown source mode {source.mode!r}")   # pragma: no cover


@dataclass
class SourcePlan:
    """What Stage 0 would write, complete, before anything is written.

    This is the object `csp source --dry-run` prints and `csp source` commits.
    Having the two be the same object is the point: the estimate you approve is
    computed by the code that then does the work, so it cannot drift from it.
    """

    results: list[SourceResult] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)

    @property
    def compositions(self) -> list[EmittedComposition]:
        return [c for r in self.results for c in r.compositions]

    @property
    def structures(self) -> list[EmittedStructure]:
        return [s for r in self.results for s in r.structures]

    def chemsystems(self) -> list[str]:
        seen: set[str] = set()
        for r in self.results:
            seen.update(r.chemsystems())
        return sorted(seen)

    @property
    def n_target_total(self) -> int:
        return sum(r.n_target_total for r in self.results)

    def render(self, gpu_seconds_per_structure: float | None = None) -> str:
        """The work estimate printed before a single GPU-minute is spent."""
        lines = [r.render() for r in self.results]
        lines.append("")
        lines.append("total")
        lines.append(f"  compositions      {len(self.compositions):,}")
        lines.append(f"  seed structures   {len(self.structures):,}")
        lines.append(f"  chemical systems  {len(self.chemsystems()):,}")
        lines.append(f"  structures wanted {self.n_target_total:,}")
        if gpu_seconds_per_structure and self.n_target_total:
            hours = self.n_target_total * gpu_seconds_per_structure / 3600.0
            lines.append(f"  implied GPU time  {hours:,.1f} h at "
                         f"{gpu_seconds_per_structure:g} s/structure")
        if self.collisions:
            lines.append(f"  cross-source      {len(self.collisions):,} collisions "
                         f"(reported, not merged)")
            for c in self.collisions[:5]:
                lines.append(f"      {c}")
            if len(self.collisions) > 5:
                lines.append(f"      ... and {len(self.collisions) - 5} more")
        return "\n".join(lines)


def expand_all(sources: list[Source] | Campaign, base_dir: Path | None = None) -> SourcePlan:
    """Expand every source and pool the results, reporting cross-source overlap."""
    if isinstance(sources, Campaign):
        sources = list(sources.source)

    _assert_names_distinct(sources)

    plan = SourcePlan()
    for source in sources:
        plan.results.append(expand_source(source, base_dir))

    plan.collisions = _cross_source_collisions(plan)
    return plan


def _assert_names_distinct(sources: list[Source]) -> None:
    """Two sources sharing a name make `--by-source` meaningless and collide in the DB.

    `composition` is unique on `(formula, z, source_name)`, so two sources called
    `default` that both emit Co5Fe1 at Z=1 silently become one row -- and the
    control group stops being a control group.
    """
    if len(sources) < 2:
        return
    seen: dict[str, int] = {}
    for s in sources:
        seen[s.name] = seen.get(s.name, 0) + 1
    dupes = sorted(n for n, c in seen.items() if c > 1)
    if dupes:
        raise SourceError(
            f"source names must be distinct, but {dupes} appear more than once. "
            f"Rows are keyed on the source name, so duplicates would merge and "
            f"`csp status --by-source` could not tell them apart."
        )


def _cross_source_collisions(plan: SourcePlan) -> list[str]:
    """Where two sources produced the same composition.  Reported, not merged."""
    owners: dict[tuple[str, int], list[str]] = {}
    for result in plan.results:
        for c in result.compositions:
            owners.setdefault(c.key(), []).append(result.name)
        for s in result.structures:
            owners.setdefault((s.formula, s.z), []).append(result.name)

    out = []
    for (formula, z), names in sorted(owners.items()):
        distinct = sorted(set(names))
        if len(distinct) > 1:
            out.append(f"{formula} Z={z} from {distinct}")
    return out


# --------------------------------------------------------------------------
# Commit
# --------------------------------------------------------------------------


@dataclass
class WriteStats:
    compositions: int = 0
    structures: int = 0

    def render(self) -> str:
        return (f"wrote {self.compositions:,} composition rows, "
                f"{self.structures:,} seed structures")


def write_plan(plan: SourcePlan, store: Store) -> WriteStats:
    """Commit a fully expanded plan.

    Seed structures carry `state='screened'` when `relax: false` and `'new'`
    otherwise, which is the whole of what "enters at Stage 2 rather than Stage 1"
    means mechanically -- there is no branch anywhere downstream, only a row that
    starts further along.
    """
    stats = WriteStats()

    for result in plan.results:
        for c in result.compositions:
            store.add_composition(
                formula=c.formula, chemsys=c.chemsys, z=c.z, n_atoms=c.n_atoms,
                n_target=c.n_target, source_mode=c.source_mode,
                source_name=c.source_name, state="new",
            )
            stats.compositions += 1

        for s in result.structures:
            comp_id = store.add_composition(
                formula=s.formula, chemsys=s.chemsys, z=s.z, n_atoms=s.n_atoms,
                n_target=0, source_mode=s.source_mode, source_name=s.source_name,
                state="generated",
            )
            store.add_structure(
                s.atoms,
                origin=Origin.seed,
                state=StructureState.new if s.relax else StructureState.screened,
                composition_id=comp_id,
                reduced_formula=s.formula,
                source_name=s.source_name,
                source_mode=s.source_mode,
                source_path=s.path,
                content_hash=s.content_hash,
                needs_relax=s.relax,
            )
            stats.structures += 1

    return stats
