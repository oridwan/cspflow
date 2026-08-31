# Command reference

Every command takes `-c/--campaign` (default `campaign.yaml`) and `-s/--set`.

**Commands find the campaign by walking up** from the current directory, so
they work anywhere inside a campaign folder, not just at its top. The campaign
actually used is printed to stderr when it was found that way.

```
csp version    Print the version.
csp init       Create a campaign folder with every knob in it.
csp doctor     Check everything that can be known before a job is submitted.
csp source     Stage 0 — expand `source:` into composition and seed rows.
csp run        The driver loop: reconcile what is in flight, submit what fits, repeat.
csp status     Show campaign progress.
csp report     Write the candidate table and a self-contained HTML dashboard.
csp recipe     Print a recipe fully resolved — every tag literal, nothing deferred.
csp config     Inspect configuration.
csp ingest     Import an existing campaign directory into a cspflow database.
csp adopt      Adopt a finished legacy campaign into cspflow's own layout.
```

---

## `csp init <name>`

Creates a campaign folder: `campaign.yaml`, your own `machine.yaml` and
`recipe.yaml`, an `inputs/` folder and a README.

| option | default | |
|---|---|---|
| `-d, --dir <path>` | `./<name>` | where to create it |
| `-m, --machine <str>` | `orion` | shipped profile to copy: `orion`, `generic_slurm`, `local` |
| `--recipe <str>` | `magnets` | shipped DFT recipe to copy |
| `--here` | | use the current folder instead of creating one |
| `--minimal` | | `campaign.yaml` only, referring to the shipped profile and recipe by name |
| `-o, --out <path>` | | write just the campaign file, at this path |
| `--force` | | overwrite existing files |

Without `--minimal`, `campaign.yaml` contains **every** key: the live ones set,
the rest commented out at the right indentation showing the default already in
effect.

## `csp doctor`

Checks everything knowable before submission and **exits non-zero on any hard
failure**, so it can gate a submit script.

| option | |
|---|---|
| `--elements <str>` | comma-separated, e.g. `Sm,Fe,Ti` — resolve POTCARs for these regardless of the campaign |
| `--fix` | create the POTCAR symlink layout |
| `-m, --machine <str>` | check a different profile |

Resolves every POTCAR, reads live SLURM limits from `sacctmgr`/`scontrol`,
verifies each configured module exists and that the VASP binary is there.

## `csp source`

Expands `source:` into rows.

| option | default | |
|---|---|---|
| `--dry-run` | | print the plan, write nothing |
| `-n, --limit <int>` | | commit only the first N composition rows (the full plan is still reported) |
| `--gpu-seconds <float>` | 0.0 | seconds per generated structure, for a time estimate |
| `--db <path>` | | write here instead of the campaign workdir |

`--limit` is what makes a large chemical space quick to sanity-check: the whole
enumeration is still reported, only the commit is capped.

## `csp run`

| option | default | |
|---|---|---|
| `--through <stage>` | | run stages up to and including this one |
| `--from <stage>` | | run stages from this one on |
| `--only <stage>` | | a single stage |
| `--watch` | | keep cycling instead of stopping when idle |
| `--interval <int>` | 300 | seconds between cycles |
| `--max-cycles <int>` | | stop after this many |
| `--dry-run` | | report what would be submitted, claim nothing |

```bash
csp run --through calibrate       # Phase A
csp run --from filter --watch     # Phase B
csp run --only screen
```

Stages, in order: `source`, `generate`, `screen`, `dedup`, `reference`,
`calibrate`, `filter`, `dft`, `analyze`. A name not in that list is an error
naming the list.

## `csp status`

| option | |
|---|---|
| `--why <int>` | full life history of one structure id, with every gate it passed or failed |

## `csp report`

| option | default | |
|---|---|---|
| `-o, --out <path>` | `<campaign>/report` | directory for `report.html` and `candidates.csv` |
| `--limit <int>` | 2000 | rows in the table; `0` for all |

## `csp recipe [name]`

Prints a recipe fully resolved — every INCAR tag literal, `inherit:` expanded.
With no argument, prints the recipe **this campaign** would actually use.

## `csp config`

```bash
csp config show              # the fully resolved configuration
csp config show --origins    # …and which layer supplied each value
csp config defaults          # the schema's defaults, independent of any campaign
```

## `csp ingest <root>`

Imports an existing campaign directory (VASP outputs on disk) into a cspflow
database.

| option | default | |
|---|---|---|
| `-n, --limit <int>` | | only this many formula directories |
| `--source-name <str>` | `ingested` | the source name recorded |
| `--db <path>` | | write here instead of the campaign workdir |
| `-q, --quiet` | | |

Unlike a composition list, ingest **tolerates junk**: it reads a directory tree
nobody curated, so a bad directory is skipped and reported rather than fatal.

## `csp adopt <flow>`

Adopts a finished legacy campaign into cspflow's layout, reading all of a flow's
result artefacts — not just its VASP directories, which is what `ingest` does —
and writing the campaign cspflow would have written had it run the work itself.
Nothing is recomputed and nothing in the source directory is touched.

| option | default | |
|---|---|---|
| `--dest <path>` | `/scratch/$USER/cspflow_results` | where the campaign directories go |
| `--staging <path>` | `/tmp` | build here first, then move — SQLite on NFS commits ~15x slower |
| `--machine <str>` | `orion` | profile to record in `campaign.yaml` |
| `-n, --limit <int>` | | only this many structures, for a quick check |

This is specific to one project's legacy layout; it is not a general importer.
For arbitrary VASP output trees, use `csp ingest`.
