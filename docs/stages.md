# The stages

Nine stages, in one fixed order. `--through`, `--from` and `--only` slice this
list; there is no other control flow.

```
source ──► generate ──► screen ──► dedup ──► reference ──► calibrate ──► filter ──► dft ──► analyze
└─────────────── Phase A: cheap, run to completion ────────────────┘     └ Phase B: expensive, streamed ┘
```

## The two phases

**Phase A is a barrier.** `csp run --through calibrate` finishes for *every*
composition before anything expensive starts. Screening is cheap, so there is
no reason to trickle it, and having all of it done is what makes the next
decision — who is worth DFT — a decision across the whole field rather than
across whatever happened to finish first.

**Phase B is a stream.** `csp run --from filter --watch` submits what fits,
reconciles what finished, and repeats, under a core-hour budget. DFT is
expensive and the queue is finite, so it is throttled rather than dumped.

They are the same driver loop over a different slice. Nothing else distinguishes
them.

---

## 0. `source` — what to search

Expands `source:` into composition rows and seed rows. Enumeration happens
entirely in memory and the plan is printed before anything is written, so
`--dry-run` runs the identical code path — the estimate you approve is produced
by the code that then does the work.

```bash
csp source --dry-run     # print the plan, write nothing
csp source -n 50         # commit only the first 50 rows (the full plan is still reported)
```

See [Choosing your input](sources.md).

## 1. `generate` — candidate structures

MatterGen samples structures for each composition, one job per batch, on a GPU.

Omit the `generate:` block entirely for a `structure_list` campaign; its
absence is what tells the driver this stage has nothing to do.

Generation yield is reported *above* the structure counts in `csp status`,
because a shortfall here is otherwise invisible: the funnel narrows anyway, and
40% fewer candidates entering it looks exactly like a smaller campaign.

## 2. `screen` — MLIP relaxation

Every candidate is relaxed with MatterSim, in batches, on a GPU. This is where
almost all structures die, and it costs GPU-seconds rather than core-hours.

Two gates report separately in the funnel:

* `screen:validate` — the structure is not physical (overlapping atoms, an
  absurd cell)
* `screen:converged` — did not reach `fmax` within `max_steps`

A cell still moving at `max_steps` is **kept and marked**, not thrown away.

## 3. `dedup` — the same crystal, found twice

pymatgen's `StructureMatcher` at the tolerances in `screen.dedup.matcher`.
Generation produces the same structure many times; carrying duplicates into DFT
buys nothing.

`dedup:seed_collision` is reported separately — a generated structure that
matches one of your own seeds is a different fact from two generated structures
matching each other.

## 4. `reference` — the hull to measure against

Fetches Materials Project entries for every chemical system in the campaign and
builds the convex hull.

`snapshot: true` freezes the query, so the hull cannot move under a campaign
mid-run. `thermo_type` is pinned to one functional: MP's summary endpoint
silently mixes them, and SmFe₂ differs by **12.2 eV/atom** between GGA and
r2SCAN — a reference set containing more than one is refused rather than
averaged.

## 5. `calibrate` — does the cheap number predict the expensive one?

The stage that makes the rest defensible. Two checks:

* **`mp`** — free. The MLIP against MP's own DFT at fixed geometry. A sanity
  check; `on_fail: warn` by default.
* **`pilot`** — costs DFT. `pilot_n` structures through *your* recipe, compared
  against their MLIP energies. `on_fail: block` by default, and this is the
  gate that stops a campaign spending 200,000 core-hours on a ranking that does
  not hold.

`spearman_min` matters most: filtering is a **ranking** decision, so rank
correlation is what has to hold — not absolute agreement.

The measured offset also feeds `filter.e_above_hull_max_source: calibrated`,
which shifts the cut by the MLIP-vs-DFT bias instead of pretending there is
none.

## 6. `filter` — who is worth a VASP job

`filter:e_above_hull` and `filter:per_composition`, in that order. The first is
the physics cut; the second keeps the field broad, so one prolific composition
cannot fill the queue.

Every rejection is recorded as an **event**, with the value and the threshold,
which is why `csp status --why` can answer "why is this structure not in my
results" exactly rather than by inference.

## 7. `dft` — VASP

Each selected structure climbs the ladder in [`recipe.yaml`](recipes.md), one
step at a time, each starting from the previous step's relaxed geometry.
Failures are matched to a remedy by cause — a timeout resumes from CONTCAR, an
SCF failure changes algorithm — and the ladder is finite.

The driver throttles against your **live** SLURM limits, not the numbers in
`machine.yaml`, and stops cleanly when `dft.select.budget_core_hours` is spent.

> A VASP run that exits cleanly at the ionic step limit is `done` and **not
> relaxed**. `csp status` reports relaxations separately from job state for
> exactly this reason — in one legacy campaign, 61% of the runs were that case.

## 8. `analyze` — properties and the report

Extracts what `analyze.properties` asks for from each finished calculation —
magnetisation, volume, spacegroup, per-formula-unit quantities — and writes the
candidate table.

See [Reading the results](results.md).

---

## The driver loop

One cycle, whatever the slice:

1. **Reconcile** — ask the scheduler what happened to everything in flight, and
   record it
2. **Advance** — move structures whose stage finished into the next stage
3. **Submit** — claim work that fits the remaining budget and the live limits
4. Repeat (with `--watch`), or stop when idle

Work is claimed with an explicit claim file, and pending work is found by an
explicit `state` string — never by a missing row. That is what makes the run
interruptible: kill it at any point and the same command resumes, because the
database says what state everything is in and nothing is inferred from absence.

```bash
csp run --through calibrate      # Phase A
csp run --from filter --watch    # Phase B, cycling every 300 s
csp run --only screen            # one stage
csp run --dry-run                # report what would be submitted, claim nothing
csp run --max-cycles 3           # bounded
```
