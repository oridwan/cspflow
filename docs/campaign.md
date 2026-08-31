# `campaign.yaml` reference

One file describes the whole search. This page lists every setting, what it
does, and what it defaults to.

You rarely need to write most of it: `csp init` writes a `campaign.yaml` with
the live keys set and **every other key present but commented out**, at the
right indentation, showing the default already in effect. Deleting the leading
`# ` is the whole edit. `csp init --minimal` opts out and writes the bare ~10
keys instead.

```bash
csp config defaults           # the schema's defaults, in full
csp config show --origins     # your resolved values, and which file set each
```

## The shape of the file

```yaml
name: my-campaign             # required
machine: machine.yaml         # required — a path, or a shipped profile name
workdir: /scratch/$USER/...   # required — database and job directories
archive: /projects/.../arch   # or null to keep nothing

source:      [...]            # required — see sources.md
generate:    {...}            # omit entirely for a structure_list campaign
screen:      {...}
reference:   {...}
calibrate:   {...}
filter:      {...}
dft:         {...}
analyze:     {...}
```

Every block except `name`, `machine`, `workdir` and `source` may be omitted
entirely; the defaults below apply.

**Unknown keys are rejected**, not ignored. A typo stops the campaign with the
key named, rather than silently doing something else.

## Where values come from

Five layers, later winning over earlier:

```
shipped defaults  <  machine profile's `campaign:` block  <  template
                  <  campaign.yaml  <  --set on the command line
```

`csp config show --origins` names the layer behind every resolved value, so
"where did this number come from" never requires reasoning about merge order.

### `$VAR` expansion

`$VAR` and `${VAR}` are expanded in any string, **after** all layers merge.

> **An undefined variable is a hard error**, not an empty string. Expanding
> `$SCRATCH` to `""` would turn `$SCRATCH/work` into `/work`, and that is
> exactly the class of quiet mistake this pipeline exists to remove.

Only YAML a layer actually supplied is expanded — schema defaults are not. This
has one surprising consequence worth knowing: the default MP cache is
`$CSPFLOW_CACHE/mp`, but **writing that literally in your campaign file** turns
an optional environment variable into a required one. Leave it out unless you
have exported `CSPFLOW_CACHE`.

### `--set`

```bash
csp run --set filter.e_above_hull_max=0.05
csp run --set dft.select.max_total=200 --set analyze.report=none
```

Dotted path into mappings; the value is parsed as YAML, so `[GGA]`, `true`,
`0.05` and `null` all mean what they look like. Lists are replaced whole —
there is no list-index syntax.

---

## `source` — what to search

Required, and the one block with real structure to it. See
**[Choosing your input](sources.md)**.

---

## `generate` — candidate structures

Omit this block entirely for a `structure_list` campaign; its absence is what
tells the driver Stage 1 has nothing to do.

```yaml
generate:
  engine: mattergen
  mattergen:
    model: /path/to/checkpoint_dir
    mode: csp
    max_batch_size: 100
    timeout_per_batch: 1800
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | meaning |
|---|---|---|
| `engine` | `mattergen` | the only engine implemented |
| `mattergen.model` | required | checkpoint directory |
| `mattergen.mode` | `csp` | `csp` conditions on composition (what you want); `unconditional` ignores it |
| `mattergen.max_batch_size` | 100 | structures per forward pass — lower it if the GPU runs out of memory |
| `mattergen.timeout_per_batch` | 1800 | seconds before a batch is abandoned |
| `resources` | `role: gpu, gpus: 1` | see [Resources](#resources) |

Generation refuses to run on CPU unless `CSPFLOW_ALLOW_CPU_GENERATION` is set.
Sampling on CPU is roughly 1000x slower, so a real run that took that path would
be killed by its walltime having produced nothing. The escape hatch exists for
smoke tests; which device was actually used is recorded in the results file.

## `screen` — MLIP relaxation

```yaml
screen:
  mlip: mattersim
  mattersim:
    model: MatterSim-v1.0.0-5M.pth
    fmax: 0.01
    max_steps: 500
    batch_size: 32
  dedup:
    matcher: {ltol: 0.2, stol: 0.2, angle_tol: 5.0}
  resources: {role: gpu, gpus: 1, time: "24:00:00"}
```

| key | default | meaning |
|---|---|---|
| `mlip` | `mattersim` | `mattersim`, `mace` or `uma` |
| `mattersim.model` | `MatterSim-v1.0.0-5M.pth` | checkpoint |
| `mattersim.fmax` | 0.01 | eV/Å force convergence |
| `mattersim.max_steps` | 500 | a cell still moving at this point is **kept and marked**, not thrown away |
| `mattersim.batch_size` | 32 | structures per batch |
| `dedup.matcher.ltol` | 0.2 | pymatgen `StructureMatcher` fractional length tolerance |
| `dedup.matcher.stol` | 0.2 | site displacement tolerance |
| `dedup.matcher.angle_tol` | 5.0 | degrees |

Note `max_steps` belongs to `mattersim`, not to `screen` directly.

## `reference` — the hull the candidates are measured against

```yaml
reference:
  functionals: [GGA]
  thermo_type: GGA_GGA+U
  energy_scale: raw
  mode: recompute
  prescreen_mode: mp_energies
  prescreen_hull_max: 0.20
  snapshot: true
  snapshot_id: auto
  relax_with_mlip: true
```

| key | default | meaning |
|---|---|---|
| `functionals` | `[GGA]` | which Materials Project functionals may enter the reference set |
| `thermo_type` | `GGA_GGA+U` | **pinned.** MP mixes functionals silently; a reference set containing more than one is refused rather than averaged |
| `energy_scale` | `raw` | `raw`, or `mp_corrected` to apply MP's anion corrections |
| `mode` | `recompute` | `mp_energies` takes MP's numbers as they are; `recompute` runs the reference phases through **your** DFT so both sides of the hull share one absolute scale |
| `prescreen_mode` | `mp_energies` | what Phase A's cheap hull uses |
| `prescreen_hull_max` | 0.20 | eV/atom, deliberately wide for Phase A |
| `snapshot` | `true` | freeze the MP query so the hull cannot move under you mid-campaign |
| `snapshot_id` | `auto` | `auto` stamps a new one with the date and MP release; a fixed id reproduces an old run exactly |
| `cache` | `$CSPFLOW_CACHE/mp` | where the frozen snapshot lives (see the `$VAR` note above) |
| `recompute_cache` | `$CSPFLOW_CACHE/reference` | recomputed reference energies |
| `relax_with_mlip` | `true` | relax reference phases with the same MLIP, so candidate and reference are treated alike |

> **Read this before trusting a DFT hull.** `mode: recompute` is the default but
> is not yet wired to anything, so a DFT hull currently places your candidates
> against MP's reference energies — two absolute scales, measured ~0.19 eV/atom
> apart in Fe-Sm-Ti against a 0.06 eV/atom selection threshold. `analyze` warns
> when a hull mixes scales. Phase A gates on the MLIP hull and is unaffected.

## `calibrate` — prove the cheap number predicts the expensive one

Two checks. The first is free, the second costs DFT and is the real gate.

```yaml
calibrate:
  mp:
    on_fail: warn
    thresholds: {mae_e_per_atom: 0.05, spearman_min: 0.90, max_volume_drift: 0.05}
  pilot:
    on_fail: block
    pilot_n: 40
    thresholds: {mae_e_per_atom: 0.05, mae_e_hull: 0.05, spearman_min: 0.90}
```

| key | default | meaning |
|---|---|---|
| `mp.on_fail` | `warn` | compares the MLIP against MP's own DFT at fixed geometry — a sanity check, not the gate |
| `mp.thresholds.mae_e_per_atom` | 0.05 | eV/atom |
| `mp.thresholds.spearman_min` | 0.90 | rank correlation |
| `mp.thresholds.max_volume_drift` | 0.05 | fractional |
| `pilot.on_fail` | `block` | runs **your** DFT on a sample; `block` stops Phase B if the MLIP does not predict it well enough |
| `pilot.pilot_n` | 40 | structures to spend on the test |
| `pilot.thresholds` | 0.05 / 0.05 / 0.90 | MAE per atom, MAE in hull distance, Spearman |

`spearman_min` is the number that matters most: filtering is a **ranking**
decision, so rank correlation is what has to hold, not absolute agreement.

## `filter` — who is worth a VASP job

```yaml
filter:
  e_above_hull_max: 0.10
  e_above_hull_max_source: calibrated
  max_per_composition: 5
  spacegroup: {min_number: 1}
```

| key | default | meaning |
|---|---|---|
| `e_above_hull_max` | 0.10 | eV/atom. **The most consequential number in the file** — it sets Phase B's size |
| `e_above_hull_max_source` | `calibrated` | `calibrated` shifts the cut by the measured MLIP-vs-DFT offset; `literal` uses the number as written |
| `max_per_composition` | 5 | keeps the field broad; without it one prolific composition fills the queue |
| `spacegroup.min_number` | 1 | 1 keeps everything. 3 excludes P1 and P-1, which are usually unrelaxed noise |

## `dft` — the expensive half

INCAR tags, k-points and per-step resources live in
[`recipe.yaml`](recipes.md). This block is everything *about* the DFT that
depends on the campaign rather than on the ladder.

```yaml
dft:
  recipe: recipe.yaml
  potcar: {tree: VASP6.4, functional: PBE_64, overrides: {}}
  rare_earth: {f_treatment: frozen, magnetic_order: ferri, reconstruct_ms: true}
  magnetism: {mode: ferrimagnetic_retm, strict: true}
  ldau: {enabled: false, ldau_type: 2, u: {}, j: {}}
  nbands: auto
  incar_overrides: {}
  max_in_flight: 200
  max_concurrent_tasks: 48
  select:
    rank_by: e_above_hull_mlip
    max_per_composition: 3
    max_total: 1500
    budget_core_hours: 200000
```

| key | default | meaning |
|---|---|---|
| `recipe` | `magnets` | a shipped recipe name, or a path (`recipe.yaml` beside the campaign) |
| `potcar.tree` | `VASP6.4` | `VASP6.4` or `VASP5.2` |
| `potcar.functional` | `PBE_64` | pymatgen functional label |
| `potcar.overrides` | `{}` | element → POTCAR symbol, e.g. `{Sm: Sm_3}` |
| `rare_earth.f_treatment` | `frozen` | `frozen` puts 4f in the core (tractable, one convention per campaign); `valence` needs LDA+U |
| `rare_earth.magnetic_order` | `ferri` | `ferri`, `ferro` or `none` |
| `rare_earth.reconstruct_ms` | `true` | add the frozen 4f moment back when reporting |
| `magnetism.mode` | `ferrimagnetic_retm` | RE moment antiparallel to the TM sublattice. Also `none`, `pymatgen`, `table` |
| `magnetism.strict` | `true` | fail rather than let any site take a default MAGMOM — a silent default is a wrong answer |
| `magnetism.table` | `{}` | element → initial moment, for `mode: table` |
| `magnetism.site_overrides` | `{}` | per-site moments |
| `ldau.enabled` | `false` | true only with `f_treatment: valence` |
| `ldau.u` / `ldau.j` | `{}` | element → eV, e.g. `{Sm: 6.0}` |
| `nbands` | `auto` | or an integer |
| `incar_overrides` | `{}` | applied on top of **every** recipe step, e.g. `{NCORE: 8, LREAL: Auto}` |
| `max_in_flight` | 200 | jobs queued at once (QOS-aware) |
| `max_concurrent_tasks` | 48 | the `--array %N` throttle |
| `select.rank_by` | `e_above_hull_mlip` | the order Phase B spends the budget in |
| `select.max_per_composition` | 3 | |
| `select.max_total` | 1500 | |
| `select.budget_core_hours` | 200000 | Phase B stops here, mid-stream, cleanly |

## `analyze` — properties and the report

```yaml
analyze:
  properties: [m_dft_raw, m_s_reconstructed, volume, spacegroup]
  report: html
```

| key | default | meaning |
|---|---|---|
| `properties` | as above | what is extracted from each finished calculation |
| `report` | `html` | `html` or `none`. `csp report` writes `report/` |

---

## Resources

Every stage that submits jobs takes a `resources:` block:

```yaml
resources:
  role: gpu           # a partition role defined in machine.yaml
  ntasks: 64
  cpus_per_task: 1
  gpus: 1
  mem: 64G
  time: "24:00:00"
```

`role` is the indirection that keeps a campaign portable: it names a role
(`cpu`, `gpu`, `bigmem`, …) that [`machine.yaml`](machines.md) maps to real
partition names. Anything you leave out falls back to that machine's defaults.

## Top-level keys

| key | required | meaning |
|---|---|---|
| `name` | yes | campaign name, used in job names and the report title |
| `machine` | yes | path to a machine profile, or a shipped name (`orion`, `generic_slurm`, `local`). A relative path resolves against the campaign folder |
| `workdir` | yes | where `campaign.db` and all job directories go. Put it on scratch — it gets large |
| `archive` | no | where finished output is archived; `null` keeps nothing |
