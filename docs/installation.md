# Installation

## What you need

| | |
|---|---|
| **Python** | 3.10 or newer |
| **A cluster** | SLURM, or `scheduler: local` for a workstation |
| **VASP** | Your own licensed build, plus the POTCAR files. cspflow never ships either. |
| **A Materials Project API key** | Free from [materialsproject.org](https://next-gen.materialsproject.org/api) — reference energies for the convex hull come from there |
| **A GPU** | For structure generation and MLIP screening. Not needed for the DFT half. |

You can install and explore cspflow — write a campaign, enumerate it, inspect
the resolved config and the DFT recipe — with none of the above except Python.
Everything up to `csp source --dry-run` runs on a laptop.

## Install

```bash
pip install cspflow            # the pipeline: config, driver, scheduler, DFT
pip install "cspflow[full]"    # adds pymatgen — needed for real work
```

`[full]` is not really optional for a campaign that runs: pymatgen provides
structure matching (dedup), spacegroup determination, convex-hull construction,
k-point meshes and POTCAR handling. The base install exists so that the config
layer, the CLI and the tests stay usable without a scientific stack.

From a checkout:

```bash
git clone <url> cspflow && cd cspflow
pip install -e ".[full,dev]"
```

## The machine-learning stack

Generation (MatterGen) and screening (MatterSim) are **not** installed by pip,
because they pin each other and pin torch. Trying to add them to an existing
environment usually breaks it. Use the bundled script, which builds one conda
environment holding all three:

```bash
./scripts/build_env.sh cspflow      # python 3.10, torch 2.2.1+cu118
conda activate cspflow
csp version
```

It writes `scripts/env.lock.txt` so the environment can be rebuilt exactly. The
script's header explains every pin — the short version is that mattergen
requires `numpy<2` and mattersim 1.2+ requires `numpy>=2`, so the newest of each
cannot coexist and the working combination is not obvious.

One environment serves every stage. If your stages need different environments,
`machine.yaml` maps each role to one (`conda: {cpu: ..., gpu: ...}`).

## Environment variables

```bash
export MP_API_KEY=...                              # required for the hull
export CSPFLOW_CACHE=/scratch/$USER/cspflow_cache  # optional; default ~/.cache/cspflow
export PMG_VASP_PSP_DIR=/path/to/potcars           # optional; machine.yaml can say instead
```

The MP key is read from the environment and never written to a config file, a
cache or the database. Keep it out of `campaign.yaml`.

`$CSPFLOW_CACHE` holds the frozen Materials Project snapshot a campaign is
measured against. Point it at somewhere with room and a backup — deleting it
means the next run builds a *different* hull.

> **Any `$VAR` you write in a config file must be defined**, or the campaign
> stops with an error naming it. This is deliberate: an empty expansion turns
> `$SCRATCH/work` into `/work`.

## POTCARs

pymatgen expects the POTCAR tree to use particular directory names, which the
distribution tarballs do not. `csp doctor --fix` builds the expected symlink
layout beside your files:

```bash
csp doctor --fix
```

Without it, a `functional: PBE_64` label can silently match a flat-layout
directory, and the recorded label then misdescribes what was actually used.
Nothing is copied and nothing is modified — only symlinks are added.

## Verify

```bash
csp version
csp doctor            # exits non-zero on any hard failure
```

`doctor` resolves a POTCAR for every element in your campaign, reads your live
SLURM limits, checks that each configured module exists, verifies the VASP
binary, and reports what it cannot confirm. Run it before your first submission
and after any cluster change; it is safe to put at the top of a submit script.

## Next

[Quick start →](quickstart.md)
