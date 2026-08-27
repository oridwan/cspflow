"""Mode 2: explicit formulas in, structures still generated.

Inline for a handful, `from_file` for a sweep; identical downstream.  The whole
of this module is bookkeeping, but three of its decisions change what a campaign
actually runs, so each is spelled out where it is made:

*   what a non-reduced formula like `Fe2Co10` means (sec. `_resolve_item`),
*   when two items are the same work and when they only look it (`_dedup`),
*   and that a formula which cannot be parsed stops the campaign rather than
    being skipped (`SourceError` throughout).

That last one is a deliberate asymmetry with `csp ingest`, which skips bad
directories and reports them.  Ingest reads a directory tree nobody curated and
must tolerate junk; a composition list is something the user wrote, so a line
in it that cannot be parsed is a mistake they want to hear about before the
campaign starts, not a quietly missing result they notice at the end.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ..chem import ChemError, canonical_formula, chemsys, parse_formula, reduce_counts
from ..config.schema import CompositionItem, NStructures, Source, ZRange
from .base import EmittedComposition, SourceError, SourceResult, expand_z


def expand_composition_list(source: Source, base_dir: Path | None = None) -> SourceResult:
    block = source.composition_list
    if block is None:                                   # pragma: no cover - schema guards
        raise ValueError("expand_composition_list called on a source without a block")

    result = SourceResult(name=source.name, mode=source.mode.value)

    items = list(block.items)
    if block.from_file:
        items.extend(_read_csv(_resolve(block.from_file, base_dir)))
    if not items:
        raise SourceError(
            f"source '{source.name}': composition_list produced no items "
            f"(from_file={block.from_file!r} was empty?)"
        )

    resolved = [_resolve_item(item, source.name, result) for item in items]
    kept = _dedup(resolved, result)

    for entry in kept:
        rows = expand_z(
            entry.counts,
            defaults=source.defaults,
            source_name=source.name,
            source_mode=source.mode.value,
            z_override=entry.z_values,
            max_atoms_override=entry.max_atoms,
            n_structures_override=entry.n_structures,
            reject=result.rejected,
        )
        if not rows:
            result.warnings.append(
                f"'{entry.written}' produced no rows: every requested Z exceeds max_atoms"
            )
        result.compositions.extend(rows)

    return result


# --------------------------------------------------------------------------


class _Resolved:
    """One list item after parsing, reduction and override resolution."""

    __slots__ = ("written", "formula", "counts", "z_values", "max_atoms", "n_structures")

    def __init__(self, written: str, formula: str, counts: dict[str, int],
                 z_values: list[int] | None, max_atoms: int | None,
                 n_structures: NStructures | None) -> None:
        self.written = written
        self.formula = formula
        self.counts = counts
        self.z_values = z_values
        self.max_atoms = max_atoms
        self.n_structures = n_structures

    @property
    def z_key(self) -> tuple[int, ...] | None:
        return tuple(self.z_values) if self.z_values is not None else None


def _resolve_item(item: CompositionItem, source_name: str, result: SourceResult) -> _Resolved:
    """Parse one item and decide what a non-reduced formula means.

    `Fe2Co10` reduces to `Co5Fe1` with an intrinsic multiplier of 2, and the
    honest reading of what the user typed is a 12-atom cell -- not the 6-atom
    one they would have got by silently keeping the reduced formula at Z=1.
    So the intrinsic multiplier becomes the Z, and the substitution is reported
    rather than done quietly.

    Writing a non-reduced formula *and* an explicit `z` is refused instead of
    guessed at: `Fe2Co10` with `z: [1, 2]` could mean Z in {1,2} or Z in {2,4},
    the two differ by a factor of two in cell size, and neither reading is
    obviously right.  A campaign that silently picks one is exactly the class of
    error this pipeline exists to remove.
    """
    try:
        counts = parse_formula(item.formula)
        reduced, z_intrinsic = reduce_counts(counts)
    except ChemError as exc:
        raise SourceError(f"source '{source_name}': {exc}") from exc

    formula = canonical_formula(reduced)
    z_values: list[int] | None = None

    if item.z is not None:
        if z_intrinsic != 1:
            raise SourceError(
                f"source '{source_name}': item '{item.formula}' is not a reduced formula "
                f"(it is {formula} x{z_intrinsic}) and also sets z={[item.z.min, item.z.max]}. "
                f"That is ambiguous -- z could be relative to {formula} or to "
                f"{item.formula}. Write the reduced formula '{formula}' with the z range "
                f"you want."
            )
        z_values = item.z.values()
    elif z_intrinsic != 1:
        z_values = [z_intrinsic]
        result.warnings.append(
            f"'{item.formula}' is not reduced; taken as {formula} at Z={z_intrinsic} "
            f"({z_intrinsic * sum(reduced.values())} atoms)"
        )

    return _Resolved(item.formula, formula, reduced, z_values, item.max_atoms, item.n_structures)


def _dedup(entries: list[_Resolved], result: SourceResult) -> list[_Resolved]:
    """Collapse items that are the same work; report items that merely look alike.

    Two distinctions matter here and they are easy to conflate:

    *   Same reduced formula AND same Z -> the same rows would be written twice.
        The later one is dropped and both spellings are named, so `FeCo5` listed
        alongside `Co5Fe1` reports the overlap instead of silently double-
        counting the composition in every summary downstream.
    *   Same reduced formula, different Z -> `FeCo5` and `Fe2Co10` are a 6-atom
        and a 12-atom cell of one material.  They are different work and both are
        kept; the shared formula is reported because a user who wrote both may
        have meant to write one.
    """
    kept: list[_Resolved] = []
    by_key: dict[tuple[str, tuple[int, ...] | None], str] = {}
    by_formula: dict[str, list[str]] = {}

    for entry in entries:
        key = (entry.formula, entry.z_key)
        first = by_key.get(key)
        if first is not None:
            result.warnings.append(
                f"'{entry.written}' appears twice; keeping the first"
                if first == entry.written else
                f"'{entry.written}' duplicates '{first}' after reduction "
                f"(both are {entry.formula}); keeping the first"
            )
            continue
        by_key[key] = entry.written
        by_formula.setdefault(entry.formula, []).append(entry.written)
        kept.append(entry)

    for formula, written in by_formula.items():
        distinct = sorted(set(written))
        if len(distinct) > 1:
            result.warnings.append(
                f"{distinct} all reduce to {formula} and are all kept, as separate "
                f"cells with different Z -- write one of them if that was not intended"
            )
    return kept


def _resolve(path: str, base_dir: Path | None) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() or base_dir is None else (base_dir / p)


def _read_csv(path: Path) -> list[CompositionItem]:
    """`formula[,z_min,z_max,n_structures]`, with `#` comments and an optional header.

    Blank cells mean "inherit from source.defaults", which is why the columns
    are read positionally but each is resolved independently: a row that sets
    only `n_structures` should not have to restate the Z range.
    """
    if not path.exists():
        raise SourceError(f"composition_list.from_file not found: {path}")

    items: list[CompositionItem] = []
    with path.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]

    if rows and rows[0] and rows[0][0].strip().lower() == "formula":
        rows = rows[1:]

    for lineno, row in enumerate(rows, start=1):
        cells = [c.strip() for c in row]
        if not cells or not cells[0]:
            continue
        formula = cells[0]
        z_min = _int_cell(cells, 1, path, lineno, "z_min")
        z_max = _int_cell(cells, 2, path, lineno, "z_max")
        n_struct = _int_cell(cells, 3, path, lineno, "n_structures")

        z: ZRange | None = None
        if z_min is not None or z_max is not None:
            lo = z_min if z_min is not None else (z_max or 1)
            hi = z_max if z_max is not None else lo
            try:
                z = ZRange(min=lo, max=hi)
            except ValueError as exc:
                raise SourceError(f"{path}:{lineno}: {exc}") from exc

        try:
            items.append(CompositionItem(
                formula=formula, z=z,
                n_structures=NStructures(mode="fixed", count=n_struct) if n_struct else None,
            ))
        except ValueError as exc:
            raise SourceError(f"{path}:{lineno}: {exc}") from exc
    return items


def _int_cell(cells: list[str], index: int, path: Path, lineno: int, name: str) -> int | None:
    if index >= len(cells) or not cells[index]:
        return None
    try:
        return int(cells[index])
    except ValueError as exc:
        raise SourceError(
            f"{path}:{lineno}: column {index + 1} ({name}) is {cells[index]!r}, "
            f"which is not an integer"
        ) from exc
