# Quick start

Ten minutes from nothing to a candidate table. Assumes cspflow is
[installed](installation.md) and `csp doctor` is happy.

## 1. Make a campaign

A campaign is a **folder**, not a file:

```bash
csp init my-campaign -m orion
cd my-campaign
```

```
my-campaign/
├── campaign.yaml   what to search, and how hard   ← every tunable key, listed
├── machine.yaml    partitions, walltime, modules, VASP, POTCAR trees
├── recipe.yaml     the DFT ladder: INCAR tags, k-points, per-step resources
├── inputs/         your own structures or composition lists
├── results/        → workdir on scratch (symlinked on the first run)
└── report/         report.html + candidates.csv
```

`machine.yaml` and `recipe.yaml` are **your copies**, not references into the
installed package. Edit them freely; nothing here is read-only.

`-m` picks which shipped profile to copy: `orion`, `generic_slurm` (portable
SLURM, no site specifics) or `local` (run on this machine, no scheduler).

## 2. Edit four things

Open `campaign.yaml`. Live keys are the ones you must choose; every other key
is there too, commented out, showing the default already in effect at the right
indentation — **deleting the leading `# ` is the whole edit**.

The four that matter on day one:

```yaml
name: my-campaign
workdir: /scratch/$USER/cspflow/my-campaign   # where the database and jobs live

source:
  - mode: chemical_space          # ← what you are searching
    chemical_space:
      groups:
        A: {elements: [Sm, Tb],     pick: 1}
        B: {elements: [Fe, Co, Ni], pick: 1, min_fraction: 0.75}
      max_atoms_formula: 20

filter:
  e_above_hull_max: 0.10          # ← eV/atom; this sets how much DFT you buy
```

Not searching a chemical space? [Choosing your input](sources.md) covers giving
an explicit list of formulas or a folder of structures you already have.

## 3. Check before you spend anything

```bash
csp doctor              # machine, modules, VASP, POTCARs, live SLURM limits
csp source --dry-run    # what would be enumerated — nothing is written
```

`--dry-run` runs the identical code path the real enumeration does, so the
estimate you approve is produced by the code that then does the work:

```
source 'shortlist' (mode=composition_list, enters at generate)
  compositions      34
  chemical systems  5
  structures wanted 1,394
  warning: 'SmFe11Ti' appears twice; keeping the first

total
  compositions      34
  seed structures   0
  chemical systems  5
  structures wanted 1,394

--dry-run: nothing written
```

**Read the structure count before you go on.** A chemical space has a size
multiplier hiding in it — raising `pick` from 1 to 2, or `max_atoms_formula`
from 20 to 30, can move that number by an order of magnitude.

## 4. Phase A — cheap, run to completion

```bash
csp run --through calibrate
```

This generates structures, relaxes them all with the MLIP, drops duplicates,
builds the reference hull, and checks that the cheap energies rank structures
the way real DFT does. It is a **barrier**: it finishes for every composition
before anything expensive begins.

## 5. Phase B — expensive, streamed

```bash
csp run --from filter --watch
```

Selects who is worth a VASP job, submits under a core-hour budget, and keeps
cycling (`--interval`, default 300 s) — reconciling what finished, submitting
what fits. Stop it with Ctrl-C whenever you like; nothing is lost, and the same
command resumes.

## 6. Look at what came out

```bash
csp status                # progress, by stage
csp status --why 1042     # the full life history of one structure
csp report                # report/report.html + report/candidates.csv
```

`report.html` is self-contained — no server, no build step, no network. Open it
in a browser. [Reading the results →](results.md)

## Things worth knowing early

**Commands find the campaign by walking up**, so they work from any folder
inside it, not just the top.

**Any stage slice works**: `--only screen`, `--from dft`, `--through dedup`.
The two phases above are just two slices of one list.

**Change one value without editing a file:**

```bash
csp run --set filter.e_above_hull_max=0.05
csp config show --origins       # every resolved value, and which file set it
```

**Nothing is hidden.** `csp config show --origins` names the file behind every
setting, and `csp recipe` prints the DFT ladder fully resolved — every INCAR
tag literal, nothing deferred to a library default.

## Next

* [Choosing your input](sources.md) — the three ways in
* [`campaign.yaml` reference](campaign.md) — every setting
* [`examples/`](../examples/) — three complete campaigns you can copy
