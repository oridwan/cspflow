# cspflow user guide

Everything you need to run a campaign, in the order you will need it.

## Start here

| | |
|---|---|
| [Installation](installation.md) | Install cspflow, set the Materials Project key, point it at your POTCARs |
| [Quick start](quickstart.md) | A working campaign in ten minutes, from `csp init` to `report.html` |
| [Examples](../examples/) | Three complete campaigns, one per input mode, with sample inputs |

## Reference

| | |
|---|---|
| [Choosing your input](sources.md) | The three ways in: a chemical space, a list of formulas, or structures you already have |
| [`campaign.yaml`](campaign.md) | Every setting in the campaign file: what it does, what it defaults to, and which ones actually matter |
| [Running on your cluster](machines.md) | `machine.yaml`: partitions, modules, walltime, POTCAR trees, the VASP binary |
| [The DFT recipe](recipes.md) | `recipe.yaml`: INCAR tags, k-points, per-step resources, and what happens on a failure |
| [The stages](stages.md) | What each of the nine stages does, and why the run is split into two phases |
| [Reading the results](results.md) | `report.html`, `candidates.csv`, the database, and how to ask why one structure was dropped |
| [Command reference](cli.md) | Every `csp` command and its options |
| [Troubleshooting](troubleshooting.md) | The errors you are most likely to hit, and what they actually mean |

## How it works, in one paragraph

You describe **what to search** in `campaign.yaml`. cspflow enumerates the
compositions, generates candidate structures with a diffusion model, relaxes
every one of them with a machine-learned interatomic potential, throws away
duplicates, places the survivors on a convex hull built from Materials Project
reference phases, **proves the cheap energies rank the same way the expensive
ones do**, and only then spends VASP time — streamed under a core-hour budget,
best candidates first. Everything lands in one SQLite file per campaign, so a
run can be stopped, inspected and resumed at any point.

```
source ──► generate ──► screen ──► dedup ──► reference ──► calibrate ──► filter ──► dft ──► analyze
└─────────────── Phase A: cheap, run to completion ────────────────┘     └ Phase B: expensive, streamed ┘
```
