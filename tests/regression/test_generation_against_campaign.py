"""The generation arithmetic, checked against the campaign that ran it.

`/projects/mmi/shuo/ter_mag_flow/mattergen_results/ternary_csp_magnets/
generation_summary.json` records, for each of 1,692 compositions, how many
structures were actually produced. That is a complete, independent oracle for
the allocation and batch-splitting logic -- and reproducing it exactly is what
establishes that the two corrections in `generators/base.py` are corrections
rather than guesses.

Skipped when the campaign directory is not mounted.
"""

import json
import math
import re
from pathlib import Path

import pytest

from cspflow.chem import parse_formula
from cspflow.generators import legacy_batch_total, split_batches

CAMPAIGN = Path("/projects/mmi/shuo/ter_mag_flow/mattergen_results/ternary_csp_magnets")
SUMMARY = CAMPAIGN / "generation_summary.json"

pytestmark = pytest.mark.skipif(not SUMMARY.is_file(),
                                reason="campaign generation_summary.json not present")

MAX_ATOMS = 20          # the mp_20 cap the legacy scripts used
MAX_BATCH = 100         # generate_ternary_csp.sh: MAX_BATCH_SIZE
CAP_STRUCTURES_PER_ATOM = 6.0   # see test_the_recorded_parameter_is_not_the_one_used


@pytest.fixture(scope="module")
def entries():
    return json.loads(SUMMARY.read_text())["compositions"]


def legacy_allocation(counts: dict[str, int], structures_per_atom: float) -> int:
    """Reimplementation of `generate_structures_batch.py`, from the script.

    Supercells 1x..Nx up to `max_atoms`; a total request proportional to the
    atoms summed over all of them; distributed back proportionally; then each
    supercell's share rounded *up* to a whole number of `max_batch_size`
    batches and multiplied back out.
    """
    per_cell = sum(counts.values())
    multiplier = MAX_ATOMS // per_cell
    supercells = [per_cell * m for m in range(1, multiplier + 1)]
    total_atoms = sum(supercells)
    n_structures = max(1, round(total_atoms * structures_per_atom))
    produced = 0
    for atoms in supercells:
        share = max(1, round(atoms / total_atoms * n_structures))
        produced += legacy_batch_total(share, MAX_BATCH)
    return produced


def test_the_legacy_allocation_reproduces_the_campaign_exactly(entries):
    """1,692 of 1,692, which is what makes the rest of this file evidence."""
    misses = [e["formula"] for e in entries
              if legacy_allocation(parse_formula(e["formula"]), CAP_STRUCTURES_PER_ATOM)
              != e["structures_generated"]]
    assert misses == [], f"{len(misses)} of {len(entries)} did not reproduce"


def test_the_recorded_parameter_is_not_the_one_used(entries):
    """`generate_ternary_csp.sh` on disk says `STRUCTURES_PER_ATOM=2.0`.

    At 2.0 the script reproduces **0** of 1,692 recorded counts; at 6.0 it
    reproduces all 1,692. The script in the repository is therefore not the one
    that produced the results next to it -- the same class of finding as the
    NBANDS formula, and the reason cspflow writes its resolved generation
    parameters into the manifest beside the output.
    """
    at_two = sum(1 for e in entries
                 if legacy_allocation(parse_formula(e["formula"]), 2.0)
                 == e["structures_generated"])
    assert at_two == 0


def test_the_batch_rounding_inflated_the_campaign_by_half(entries):
    """What the round-up cost, in structures that were never asked for."""
    requested = produced = 0
    for entry in entries:
        counts = parse_formula(entry["formula"])
        per_cell = sum(counts.values())
        supercells = [per_cell * m for m in range(1, MAX_ATOMS // per_cell + 1)]
        total_atoms = sum(supercells)
        requested += max(1, round(total_atoms * CAP_STRUCTURES_PER_ATOM))
        produced += entry["structures_generated"]

    assert requested == 177_120
    assert produced == 265_536
    assert produced - requested == 88_416
    assert 0.498 < (produced - requested) / requested < 0.500


def test_our_split_would_have_asked_for_exactly_what_was_wanted(entries):
    """The same walk, with `split_batches` in place of the round-up."""
    requested = produced = 0
    for entry in entries:
        counts = parse_formula(entry["formula"])
        per_cell = sum(counts.values())
        supercells = [per_cell * m for m in range(1, MAX_ATOMS // per_cell + 1)]
        total_atoms = sum(supercells)
        n_structures = max(1, round(total_atoms * CAP_STRUCTURES_PER_ATOM))
        requested += n_structures
        for atoms in supercells:
            share = max(1, round(atoms / total_atoms * n_structures))
            produced += sum(split_batches(share, MAX_BATCH))

    # Proportional rounding still moves a handful of structures per composition;
    # what is gone is the systematic 50% inflation.
    assert abs(produced - requested) / requested < 0.005


def test_every_recorded_composition_parses_and_fits_the_cap(entries):
    """A cheap guard on the oracle itself, so a bad read fails loudly here."""
    assert len(entries) == 1692
    for entry in entries:
        counts = parse_formula(entry["formula"])
        assert 3 <= len(counts) <= 3          # every one is ternary
        assert 8 <= sum(counts.values()) <= MAX_ATOMS


@pytest.mark.slow
def test_the_written_structures_match_the_recorded_counts(entries):
    """The extxyz files hold what the summary says they hold.

    Read for a sample rather than all 1,692, because this touches ~1.7 GB.
    """
    import ase.io

    sample = entries[::170]
    for entry in sample:
        path = CAMPAIGN / f"{entry['formula']}_structures" / "generated_crystals.extxyz"
        if not path.is_file():                      # pragma: no cover
            pytest.skip(f"{path} not present")
        structures = ase.io.read(str(path), index=":")
        assert len(structures) == entry["structures_generated"]
        wanted = parse_formula(entry["formula"])
        for atoms in structures:
            got: dict[str, int] = {}
            for symbol in atoms.get_chemical_symbols():
                got[symbol] = got.get(symbol, 0) + 1
            # Cells are supercells of the reduced formula, so compare ratios.
            factor = sum(got.values()) // sum(wanted.values())
            assert got == {k: v * factor for k, v in wanted.items()}
