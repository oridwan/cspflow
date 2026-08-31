# Reading the results

## Where things are

```
my-campaign/
├── campaign.yaml
├── report/          report.html + candidates.csv   ← csp report writes here
└── results/  ──────────────────────────────┐
                                            │ symlink, made on the first run
$workdir/                        ←──────────┘
├── campaign.db      everything: structures, states, jobs, energies
├── generate/        one directory per generation job
├── screen/          MLIP batches and their manifests
└── dft/             one directory per VASP calculation
```

`workdir` is on scratch because it gets large. The campaign folder stays small
and holds only what you wrote plus what you would want to keep.

## `csp status`

```
campaign     Fe-Sm
database     /scratch/.../Fe-Sm/campaign.db
compositions 109  across 1 chemical systems
generated    3,034 of 5,348 requested (56.7%)   <- 107 composition(s) short
structures   3034
    dft_done         165
    failed           2
    filtered_out     2867
reference    21 MP entries
jobs
    done             165
    timeout          2
    core-hours       1,276
relaxations
    mattersim:converged      3034
    vasp:relax:converged     52
    vasp:relax:not converged 115   <- not usable as a relaxed geometry
```

Three things in that output are there because they are easy to miss:

**Generation yield is printed above the structure counts.** A shortfall here is
otherwise invisible — the funnel narrows anyway, and 40% fewer candidates
entering it looks exactly like a smaller campaign.

**Relaxations are reported separately from job state.** A VASP run that exits
cleanly at the ionic step limit is `done` and **not relaxed**. In one legacy
campaign 61% of the runs were exactly that, and counting them as successes
would have put unrelaxed geometries in the results table.

**`core-hours`** is what you have actually spent, against
`dft.select.budget_core_hours`.

## `csp status --why <id>` — why is this structure not in my results?

```
structure 12: Fe10Sm
    composition_id       2
    filter_reason        mlip e_above_hull > 0.1
    generator            mattergen
    mlip_e_above_hull    0.20335836121530093
    mlip_e_per_atom      -7.919482838023793
    mlip_relaxed         True
    origin               generated
    spacegroup           10
    state                filtered_out
    wyckoff              ['2n', '2n', '2m', '1h', '2m', '1e', '1a']
  gates
    filter:e_above_hull  FAIL  value=0.20335836121530093 threshold=0.1
```

The gate line is read from a recorded **event**, not reconstructed by
differencing counts. That distinction matters: a structure can leave a state for
more than one reason, and differencing attributes all of them to whichever gate
ran last.

## `csp report`

```bash
csp report                    # report/report.html + report/candidates.csv
csp report --limit 0          # every row, not just the first 2000
csp report -o /somewhere/else
```

`report.html` is self-contained — no server, no build step, no network. Open it
in a browser. It carries the funnel, the summary, and a sortable candidate
table with a definition for every column.

### The funnel

How many structures each gate saw and how many it kept, in the order they ran.
From a real campaign:

```
gate                           seen   passed  rejected
filter:e_above_hull            3034      167      2867
dft:relax:converged             167       52       115
select:candidate                 11       11         0
```

Which gates appear depends on which stages ran: a full campaign also shows
`screen:validate`, `screen:converged`, `dedup`, `dedup:seed_collision` and
`filter:per_composition`.

### `candidates.csv`

| column | meaning |
|---|---|
| `id` | campaign structure id — the one `--why` takes |
| `formula` | reduced formula, alphabetical (`Fe17Sm2`) |
| `n_atoms` | atoms in the cell |
| `spacegroup`, `spacegroup_symbol`, `symprec` | of the relaxed cell, at the tolerance named |
| `e_above_hull_mlip` | eV/atom, MatterSim energies against MP |
| `dft_e_above_hull` | eV/atom, your DFT against MP |
| `dft_e_formation` | eV/atom |
| `volume`, `volume_per_atom` | Å³, relaxed cell |
| `m_dft_raw` | μ_B per cell — **computed**: the cell magnetisation |
| `m_s_reconstructed` | μ_B per cell — **modelled**: TM sublattice + Hund's-rule 4f |
| `f_treatment` | which of those two is the meaningful one |
| `m_per_formula_unit` | μ_B, from `m_dft_raw` |
| `m_per_volume` | μ_B/Å³, from `m_dft_raw` |
| `state` | where the structure stopped |

`m_dft_raw` and `m_s_reconstructed` are the same physical quantity computed two
ways and are **adjacent on purpose**. With `f_treatment: frozen` the 4f moment
is in the core and `m_dft_raw` is missing it; `m_s_reconstructed` adds it back
from Hund's rules. A reader who sees only one of them has been misled, so both
are always written, and `f_treatment` says which one to believe.

> **Check `dft_e_above_hull` against the reference mode.** With
> `reference.mode` not yet wired, a DFT hull measures your candidates against
> MP's reference energies — two absolute scales, ~0.19 eV/atom apart in
> Fe-Sm-Ti against a 0.06 eV/atom threshold. `analyze` warns when a hull mixes
> scales. `e_above_hull_mlip` is unaffected.

## The database

`campaign.db` is one SQLite file, and it is **also** a valid ASE database. The
`systems` table is ASE's; the relational tables beside it are cspflow's.

```bash
ase gui campaign.db                     # browse structures, no export step
sqlite3 campaign.db ".tables"
```

```sql
-- the funnel, from the events that produced it
SELECT gate, COUNT(*) FROM filter_event GROUP BY gate;

-- core-hours by stage
SELECT stage, SUM(core_hours) FROM job GROUP BY stage;
```

Two conventions hold everywhere in it, and both come from measured behaviour:

* **Pending work is an explicit `state` string, never a missing key.** ASE's
  `db.select('~key')` does not return the complement, so "not yet computed" as
  an absent key is a dead end — the failure mode is a driver that silently
  never picks up any work.
* **No `NULL` for "not measured yet".** A failed or skipped measurement carries
  a state and a reason.

`provenance` records the resolved config and code state behind every set of
numbers, so a result can always be traced to the configuration that produced it.
