"""What a generator is, and the arithmetic that decides how much it is asked for.

The interface is four things -- a request, an outcome, a way to split a request
into GPU-sized batches, and a way to read what came back.  Everything specific
to MatterGen lives in `mattergen_engine.py`, so a second generator (SymmCD,
LEGO-xtal) is a new file rather than a new branch in this one.

Two rules here are corrections to how the legacy campaign did it, and both are
measured against that campaign's own output rather than asserted:

*   **A split must not inflate the request.**  See `split_batches`.
*   **A result is what came back, not what was written.**  The legacy resume
    check treated the presence of `generated_crystals.extxyz` as completion, so
    a job killed after its first supercell resumed as finished.  Here the
    requested and produced counts are both recorded and compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..chem import canonical_formula


class GeneratorError(Exception):
    """Anything wrong with generation, stated in terms of the request."""


@dataclass(frozen=True)
class GenerationRequest:
    """Ask for `n_requested` structures at exactly this composition.

    `counts` is the *cell* composition, already multiplied by Z.  The source
    stage has expanded Z into separate `composition` rows (one per formula
    unit count), so nothing downstream re-derives supercells -- which is where
    the legacy script's proportional-allocation arithmetic lived.
    """

    composition_id: int
    formula: str
    counts: dict[str, int]
    n_requested: int

    @property
    def n_atoms(self) -> int:
        return sum(self.counts.values())

    def __post_init__(self) -> None:
        if self.n_requested <= 0:
            raise GeneratorError(f"{self.formula}: n_requested must be positive")
        if not self.counts or any(v <= 0 for v in self.counts.values()):
            raise GeneratorError(f"{self.formula}: counts must be positive integers")


@dataclass
class GenerationOutcome:
    """What one request actually produced."""

    composition_id: int
    formula: str
    n_requested: int
    n_produced: int = 0
    batches: list[int] = field(default_factory=list)
    output: str = ""
    seconds: float = 0.0
    error: str = ""
    device: str = ""
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and self.n_produced > 0

    @property
    def complete(self) -> bool:
        """Produced everything that was asked for.

        Separate from `ok` on purpose: a request that returns 190 of 200 is a
        usable result *and* a shortfall, and collapsing the two is exactly how
        the legacy resume check lost work.
        """
        return self.ok and self.n_produced >= self.n_requested

    def as_dict(self) -> dict[str, Any]:
        return {
            "composition_id": self.composition_id, "formula": self.formula,
            "n_requested": self.n_requested, "n_produced": self.n_produced,
            "batches": self.batches, "output": self.output,
            "seconds": round(self.seconds, 2), "error": self.error,
            "device": self.device, "rejected": self.rejected,
        }


def split_batches(target: int, max_batch_size: int) -> list[int]:
    """Split `target` structures into batches that sum to exactly `target`.

    MatterGen's CLI takes `--batch_size` and `--num_batches` and generates their
    *product*, so a target above the batch cap has to be divided.  The obvious
    division is wrong in a way that costs real GPU time:

        num_batches = ceil(target / cap);  generated = cap * num_batches

    which rounds every request up to a multiple of the cap.  Measured on the
    campaign's own `generation_summary.json` -- 1,692 compositions, all 1,692
    reproduced exactly by this arithmetic -- that turned 177,120 requested
    structures into 265,536 generated ones.  **88,416 structures, 49.9% more
    than asked for**, and every one of them was then relaxed and deduplicated
    downstream.

    Dividing evenly instead costs nothing: 108 becomes 54+54 rather than
    100+100, which is both exact and better balanced for GPU memory.
    """
    if target <= 0:
        raise GeneratorError(f"target must be positive, got {target}")
    if max_batch_size <= 0:
        raise GeneratorError(f"max_batch_size must be positive, got {max_batch_size}")
    n = math.ceil(target / max_batch_size)
    base, remainder = divmod(target, n)
    return [base + 1] * remainder + [base] * (n - remainder)


def legacy_batch_total(target: int, max_batch_size: int) -> int:
    """What the legacy split would have generated.  Used by the tests only."""
    if target <= max_batch_size:
        return target
    return max_batch_size * math.ceil(target / max_batch_size)


def read_generated(directory: Path) -> list:
    """Read the structures MatterGen wrote, from the extxyz file.

    MatterGen writes two files: `generated_crystals.extxyz` in the output
    directory, and `generated_crystals_cif.zip` built by round-tripping every
    structure through a **hard-coded `/tmp/gen_{i}.cif`** (mattergen
    `common/utils/eval_utils.py:save_structures`).  Two generation processes
    sharing a node therefore write and read each other's `/tmp/gen_0.cif`
    while building their zips.  It is not `tempfile.gettempdir()`, so `TMPDIR`
    does not move it.

    Reading the extxyz avoids the question entirely, so that is what we read.
    (The campaign's own output was checked for this and is clean -- 25 sampled
    directories, zero wrong-composition entries in either file -- because its
    generation ran sequentially in a single job.  The hazard is latent, not
    realised, but an array job is exactly what would realise it.)

    That same function wraps its writes in `except IOError: print(...)`, so a
    failed save returns normally.  A missing file here is therefore a real
    possible outcome of a "successful" call, and is reported as such.
    """
    import ase.io

    path = Path(directory) / "generated_crystals.extxyz"
    if not path.is_file():
        raise GeneratorError(
            f"no generated_crystals.extxyz in {directory}. MatterGen's save step "
            f"swallows IOError and returns normally, so this can follow an exit "
            f"code of 0 -- check the job's stderr for a write failure."
        )
    if path.stat().st_size == 0:
        raise GeneratorError(f"{path} is empty")
    return list(ase.io.read(str(path), index=":"))


def check_composition(atoms, counts: dict[str, int]) -> str:
    """Empty string if `atoms` has exactly `counts`, else why not.

    CSP-mode generation is conditioned on a target composition, and in the
    campaign's output it held for every structure sampled.  It is still checked
    per structure, because the failure -- a model that ignores the conditioning
    for one chemistry -- produces plausible structures of the wrong compound,
    which nothing further down the funnel would notice.
    """
    got: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        got[symbol] = got.get(symbol, 0) + 1
    if got == dict(counts):
        return ""
    return (f"composition mismatch: asked {canonical_formula(counts)}, "
            f"got {canonical_formula(got)}")


@runtime_checkable
class Generator(Protocol):
    """The whole interface a generator has to provide."""

    name: str

    def preflight(self) -> list[str]:
        """Checks to run before spending queue time.  Returns problems found."""

    def generate(self, request: GenerationRequest, workdir: Path) -> GenerationOutcome:
        """Produce structures for one request, leaving them in `workdir`."""
