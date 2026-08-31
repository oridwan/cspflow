# Troubleshooting

Errors you are likely to meet, with what they actually mean. Most of cspflow's
failures are deliberate — it prefers stopping with a named cause to producing a
number nobody can defend.

## Configuration

### `undefined variable(s) ['CSPFLOW_CACHE'] in '$CSPFLOW_CACHE/mp' (at reference.cache)`

A `$VAR` in your config is not exported. Either export it, or write the value
literally.

An undefined variable is a hard error rather than an empty string on purpose:
expanding `$SCRATCH` to `""` turns `$SCRATCH/work` into `/work`.

**The common version of this is self-inflicted.** The default MP cache is
`$CSPFLOW_CACHE/mp`, but *schema defaults are never expanded* — only YAML a
layer actually supplied is. So copying that default into your campaign file
turns an optional environment variable into a required one. Leave it out unless
you have exported `CSPFLOW_CACHE`.

### `Extra inputs are not permitted` / `filtr` — a typo

```
1 validation error for Campaign
filtr
  Extra inputs are not permitted
```

Unknown keys are rejected rather than ignored, so a misspelled key stops the
campaign instead of silently doing nothing. The name in the error is the key
that was not recognised.

### `Field required … source.0.chemical_space.groups.A.pick`

`pick` has **no default** — it is the sweep's size multiplier, and guessing it
would change how much compute you buy by an order of magnitude. Say how many
elements to take from the group.

### `unknown machine 'x'; shipped profiles: ['generic_slurm', 'local', 'orion']. Pass a path to use your own.`

`machine:` takes a shipped name or a path. In a campaign made by `csp init`,
that path is `machine.yaml` beside the campaign file.

### `unknown stage 'screeen'; the funnel is [...]`

`--only`, `--from` and `--through` take a stage name from the funnel, which the
error prints in full.

### "Which file set this value?"

```bash
csp config show --origins
```

Every resolved value with the layer that supplied it. There is no need to
reason about merge order.

## Input

### `column 4 (n_structures) is '80          # the 2:17 magnet', which is not an integer`

In a composition CSV, `#` is honoured **only at the start of a line**. A
trailing comment after data is read as part of the last cell. Put the comment
on its own line.

### `item 'Fe2Co10' is not a reduced formula … and also sets z=[1, 2]. That is ambiguous`

`Fe2Co10` reduces to `Co5Fe1` with an intrinsic multiplier of 2, so `z: [1, 2]`
could mean Z ∈ {1,2} or Z ∈ {2,4} — a factor of two in cell size. Write the
reduced formula with the range you mean.

Writing a non-reduced formula *without* a `z` is fine: it is taken as that Z,
and you are told.

### `'SmFe11Ti' appears twice; keeping the first`

You listed a formula both inline and in the CSV, or twice in one of them.
Inline items are read first and win. This is a warning, not an error — it is a
normal way to make a one-off exception to a list you otherwise maintain in a
file.

### A seed structure is rejected at ingest

Structure files are parsed by **both** pymatgen and ASE, and they must agree.
The hard failures are:

* **a VASP-4 POSCAR with no species line** — pymatgen invents hydrogen and
  helium for these, with nothing but a stderr warning; H and He would then go
  into the MLIP and into VASP
* **a disordered structure** (partial occupancies) — ASE silently discards the
  minority species and hands back a clean-looking ordered cell that is not the
  material in the file
* **the two parsers disagreeing on the composition**

These are errors at ingest rather than warnings because each of them otherwise
surfaces after GPU or DFT time is spent — or never, with a wrong number in the
results table looking exactly like a right one. Fix the file, or drop it.

### `composition_list.from_file not found: …`

Relative paths resolve against the **campaign folder** — the directory holding
`campaign.yaml` — not against wherever you are standing. `inputs/compositions.csv`
means the one in this campaign.

## Cluster

### A job dies with an error naming something unrelated

Almost always a missing module. `module load` of a module that does not exist
writes to stderr and **still exits 0**, so under `set -e` the job dies at the
*next* command.

```bash
csp doctor        # checks every configured module against `module -t avail`
```

Cluster module names change. `cuda/11.8` disappearing is what motivated the
check.

### `srun --mpi=pmi2` starts and then `PMPI_Init` aborts

Intel MPI needs `I_MPI_PMI_LIBRARY` pointed at the PMI library SLURM actually
provides. Without it, the pmi2 plugin rejects every client request with an
error naming neither the variable nor the library. `csp doctor` checks that
every `env` value that looks like a path exists.

### POTCARs resolve but the functional label looks wrong

Run `csp doctor --fix`. pymatgen expects directory *names* the distribution
tarballs do not use; without the symlink layout, `functional: PBE_64` can
silently match a flat-layout directory and the recorded label then misdescribes
what was used. `--fix` only adds symlinks — nothing is copied or modified.

### Jobs are not being submitted

`csp run --dry-run` reports what *would* be submitted and why not, without
claiming anything. The usual causes:

* the core-hour budget is spent (`dft.select.budget_core_hours`)
* your live QOS `max_submit` is the binding limit — the driver throttles
  against what `sacctmgr`/`scontrol` say, not against `machine.yaml`
* `calibrate.pilot.on_fail: block` stopped Phase B (see below)

### A failed job is not retried

Exit 127 is `command not found`, and it is **never** retried: retrying it
unchanged burns a submission slot to reproduce the identical failure. Fix the
path in `machine.yaml`.

Otherwise, a structure gets one attempt per rung of the recipe's `retry`
ladder, plus the original. `csp status --why <id>` names the remedy each
attempt used.

### A retry rule never fires

Check you wrote `when:`, not `on:`. YAML 1.1 parses a bare `on` as the boolean
`True`, so `- on: timeout` becomes `{True: 'timeout'}` and the trigger
disappears. cspflow warns about a retry rule with no `when:`.

## Results

### Phase B will not start: calibration blocked it

`calibrate.pilot` ran your DFT on `pilot_n` structures and the MLIP did not
predict it well enough — usually `spearman_min`. That is the gate doing its
job: it is the difference between spending 200,000 core-hours on a ranking that
holds and one that does not.

Options, in the order worth trying: look at the parity plot before changing any
threshold; raise `pilot_n` if the sample was small; try a different MLIP;
loosen `filter.e_above_hull_max` so the ranking has to hold over a narrower
range. Setting `on_fail: warn` is available and is the choice to make
deliberately, not by reflex.

### `recipe 'x' would inherit VASP defaults for tags that change the physics`

A recipe step omits `ENCUT`, `ISPIN`, `LASPH`, `NELM` or `LORBIT`. The error
says what VASP would do instead for each. See [the recipe
guide](recipes.md#incar--free-form-with-two-guardrails).

### A VASP job is `done` but the structure has no relaxed geometry

A run that exits cleanly at the ionic step limit is `done` and **not relaxed**.
`csp status` reports relaxations separately from job state for this reason, and
flags `not converged` counts with `<- not usable as a relaxed geometry`. Give
the ladder an `ionic_step_limit` rung with a higher `NSW` and
`remedy: resume_from_contcar`.

### `dft_e_above_hull` looks implausible

**Check the reference mode.** `reference.mode` defaults to `recompute` but is
not yet wired to anything, so a DFT hull measures your candidates against MP's
reference energies — two absolute scales, measured ~0.19 eV/atom apart in
Fe-Sm-Ti against a 0.06 eV/atom selection threshold. `analyze` warns when a
hull mixes scales.

Until that is closed, gate on `e_above_hull_mlip`, which is unaffected.

### "Why is this structure not in my results?"

```bash
csp status --why 1042
```

Prints everything known about it and every gate it passed or failed, with the
value and the threshold. Read from recorded events, not reconstructed by
differencing counts — a structure can leave a state for more than one reason.

### `no campaign database at …`

Nothing has run yet. `csp source` writes the first rows.

## Still stuck

* `csp doctor` — most cluster-side problems have a named check
* `csp config show --origins` — most config-side problems are a value coming
  from a layer you forgot about
* `csp recipe` — the DFT ladder fully resolved, with nothing implicit
* [`examples/`](../examples/) — three campaigns known to load, with their
  numbers asserted by the test suite
