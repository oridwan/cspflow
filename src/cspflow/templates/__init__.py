"""Campaign scaffolds.

`csp init` writes a campaign *folder*, not a lone file, because the knobs a
user actually reaches for live in three different places -- what to search
(campaign.yaml), where it runs (machine.yaml) and how the DFT is done
(recipe.yaml) -- and two of those used to be buried inside the installed
package where nobody could find, read or edit them.

The annotated campaign file is the single source of truth here: the terse
`--minimal` variant is *derived* from it by dropping the comment lines, so the
two can never drift apart. Every tunable key is present as a comment showing
the default already in effect; uncommenting one changes it. Tier-3 knobs (raw
INCAR tags, retry ladders) live in recipe.yaml, which is now a file in the
campaign folder rather than a name resolved inside site-packages.
"""

from __future__ import annotations

import re
from pathlib import Path

# Placeholders are @TOKENS@ rather than {braces} so the template can contain
# YAML flow mappings -- {elements: [...]} -- without doubling every brace.
CAMPAIGN = """\
# ===========================================================================
#  @NAME@ -- cspflow campaign
#
#  Everything this campaign needs is in this folder:
#
#    campaign.yaml   what to search, and how hard        <- you are here
#    machine.yaml    scheduler, partitions, codes, POTCAR trees
#    recipe.yaml     the DFT ladder: INCAR tags, k-points, resources
#    inputs/         your own structures or composition lists
#    results/        symlink to workdir, made on the first run
#
#  Uncommented keys are live. Commented keys show the default already in
#  effect -- delete the leading "# " to change one; the indentation is already
#  right. To see the resolved value of every key and which file supplied it:
#
#      csp config show --origins
# ===========================================================================

name: @NAME@
machine: @MACHINE@
workdir: /scratch/$USER/cspflow/@NAME@
archive: /projects/mmi/Ridwan/cspflow_archive/@NAME@

# ---------------------------------------------------------------------------
# 1. SOURCE -- what to search
#
#    Three modes, and a campaign may hold several entries at once. The two
#    below the live one are complete and ready to uncomment.
# ---------------------------------------------------------------------------
source:
  - mode: chemical_space
    name: sweep
    chemical_space:
      groups:
        # `pick` has no default on purpose: one element per group gives binary
        # systems, allowing two adds ternaries and a 5-10x larger sweep.
        A: {elements: [Sm, Tb],                pick: 1}
        B: {elements: [Fe, Co, Ni],            pick: 1, min_fraction: 0.75}
        C: {elements: [Ti, V, Cr, Mn, Cu, Zn], pick: 1}
      max_atoms_formula: 20
      max_rare_earth: 1           # rare-earth species per system; null = no limit
    defaults:
      z: {min: 1, max: 2}
      max_atoms: 40
      n_structures: {mode: per_atom, structures_per_atom: 2.0}

#   - mode: composition_list      # exactly these formulas, no sweep
#     name: shortlist
#     composition_list:
#       items:
#         - {formula: Sm2Fe17,  z: [1, 1], max_atoms: 38}
#         - {formula: SmFe11Ti, z: [1, 2], max_atoms: 26,
#            n_structures: {mode: fixed, count: 40}}
#       # from_file: inputs/compositions.csv  # or a CSV: formula[,z_min,z_max,n_structures]

#   - mode: structure_list        # skip generation; enter the funnel at screen
#     name: seeds
#     structure_list:
#       paths: [inputs/seeds]     # POSCAR/CIF paths or globs
#       relax: true               # MLIP-relax them before screening
#       dedup: warn               # warn | drop -- 'warn' keeps curated near-duplicates
#       max_atoms: 40

# ---------------------------------------------------------------------------
# 2. GENERATE -- candidate structures per composition
#     (delete this whole block for a structure_list-only campaign)
# ---------------------------------------------------------------------------
generate:
  engine: mattergen
  mattergen:
    model: /projects/mmi/shuo/MatterGen_checkpoints/18-55-08
#     mode: csp                   # csp | unconditional
#     max_batch_size: 100
#     timeout_per_batch: 1800     # seconds
#   resources: {role: gpu, gpus: 1, time: "24:00:00"}

# ---------------------------------------------------------------------------
# 3. SCREEN + DEDUP -- MLIP relaxation, then throw away the duplicates
# ---------------------------------------------------------------------------
# screen:
#   mlip: mattersim               # mattersim | mace | uma
#   mattersim:
#     model: MatterSim-v1.0.0-5M.pth
#     fmax: 0.01                  # eV/A force convergence
#     max_steps: 500
#     batch_size: 32
#   dedup:
#     matcher: {ltol: 0.2, stol: 0.2, angle_tol: 5.0}
#   resources: {role: gpu, gpus: 1, time: "24:00:00"}

# ---------------------------------------------------------------------------
# 4. REFERENCE -- the convex hull the candidates are measured against
# ---------------------------------------------------------------------------
# reference:
#   functionals: [GGA]
#   thermo_type: GGA_GGA+U        # PINNED. MP mixes functionals silently; a
#                                 # reference set with more than one is refused.
#   energy_scale: raw             # raw | mp_corrected
#   mode: recompute               # mp_energies | recompute
#   prescreen_mode: mp_energies
#   prescreen_hull_max: 0.20      # widened for Phase A
#   snapshot: true                # freeze the MP query so the hull cannot move
#   snapshot_id: auto
#   relax_with_mlip: true
#   cache: $CSPFLOW_CACHE/mp

# ---------------------------------------------------------------------------
# 5. CALIBRATE -- prove the cheap number predicts the expensive one
# ---------------------------------------------------------------------------
# calibrate:
#   mp:                           # free: MP's own DFT, at FIXED geometry
#     on_fail: warn
#     thresholds: {mae_e_per_atom: 0.05, spearman_min: 0.9, max_volume_drift: 0.05}
#   pilot:                        # costs pilot DFT: OUR DFT. the real gate.
#     on_fail: block
#     pilot_n: 40
#     thresholds: {mae_e_per_atom: 0.05, mae_e_hull: 0.05, spearman_min: 0.9}

# ---------------------------------------------------------------------------
# 6. FILTER -- who is worth a VASP job
# ---------------------------------------------------------------------------
filter:
  e_above_hull_max: 0.10          # eV/atom
#   e_above_hull_max_source: calibrated   # literal | calibrated
#   max_per_composition: 5
#   spacegroup: {min_number: 1}   # 3 and up excludes P1 and P-1

# ---------------------------------------------------------------------------
# 7. DFT -- the expensive half. Tags and k-points live in recipe.yaml.
# ---------------------------------------------------------------------------
dft:
  recipe: @RECIPE@
#   potcar:
#     tree: VASP6.4               # VASP6.4 | VASP5.2 -- both supported
#     functional: PBE_64
#     overrides: {}               # element -> POTCAR symbol
#   rare_earth:
#     f_treatment: frozen         # frozen | valence -- one convention per campaign
#     magnetic_order: ferri
#     reconstruct_ms: true
#   magnetism:
#     mode: ferrimagnetic_retm
#     strict: true                # fail if any site would take a default MAGMOM
#   ldau: {enabled: false, ldau_type: 2, u: {}, j: {}}
#   nbands: auto
#   incar_overrides: {}           # free-form; applied on top of the recipe
#   max_in_flight: 200            # jobs queued at once
#   max_concurrent_tasks: 48
#   select:
#     rank_by: e_above_hull_mlip
#     max_per_composition: 3
#     max_total: 1500
#     budget_core_hours: 200000   # Phase B stops here

# ---------------------------------------------------------------------------
# 8. ANALYZE -- what comes out
# ---------------------------------------------------------------------------
# analyze:
#   properties: [m_dft_raw, m_s_reconstructed, volume, spacegroup]
#   report: html
"""

MINIMAL_HEADER = """\
# @NAME@ -- cspflow campaign.  `csp init @NAME@` (without --minimal) writes
# this file with every tunable knob listed alongside it, and copies the
# machine profile and DFT recipe into the campaign folder so both can be
# edited.  `csp config defaults` prints the defaults in force here.
"""

README = """\
# @NAME@

A cspflow campaign. Four files, and you can edit all of them.

| file | what you change there |
|------|-----------------------|
| `campaign.yaml` | what to search, how many structures, the hull cutoff |
| `machine.yaml`  | partition, walltime, modules, VASP binary, POTCAR trees |
| `recipe.yaml`   | the DFT ladder: INCAR tags, k-point density, resources |
| `inputs/`       | your own structures (`structure_list`) or composition lists |

Outputs are not in this folder -- they go to `workdir` on scratch, because
they get large. After the first run, `results/` here is a symlink to it, and
`csp report` writes `report/` here.

## Run it

```bash
conda activate cspflow
csp doctor                    # check machine, codes, POTCARs, env
csp source --dry-run          # what would be searched, before anything runs
csp run --through calibrate   # Phase A: cheap, runs to completion
csp run --from filter --watch # Phase B: expensive, streamed under budget
csp status                    # where everything is
csp report                    # report/report.html + report/candidates.csv
```

Every command finds `campaign.yaml` by walking up from wherever you are, so
these work from any folder inside the campaign, not just the top of it.

## Change one thing without editing a file

```bash
csp run --set filter.e_above_hull_max=0.05
csp config show --origins     # every resolved value, and which file set it
```
"""

INPUTS_README = """\
Put your own input here.

* `structure_list` campaigns read POSCAR/CIF/extxyz from a folder named in
  `campaign.yaml`, e.g. `paths: [inputs/seeds]`.
* Composition lists can live here too, as YAML, and be referenced the same way.

Anything in this folder is yours; cspflow never writes to it.
"""


def _fill(text: str, *, name: str, machine: str = "orion", recipe: str = "magnets") -> str:
    return (text.replace("@NAME@", name)
                .replace("@MACHINE@", machine)
                .replace("@RECIPE@", recipe))


def _strip_comments(text: str) -> str:
    """Drop whole-line comments, keeping trailing ones and collapsing the gaps.

    This is what makes `--minimal` safe: the terse file is the annotated file
    with the annotations removed, so a knob can never exist in one and not the
    other.
    """
    kept = [line for line in text.splitlines() if not re.match(r"\s*#", line)]
    out: list[str] = []
    for line in kept:
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def campaign_yaml(*, name: str, machine: str = "orion", recipe: str = "magnets",
                  minimal: bool = False) -> str:
    """The campaign file: annotated by default, terse on request."""
    body = _fill(CAMPAIGN, name=name, machine=machine, recipe=recipe)
    if not minimal:
        return body
    return _fill(MINIMAL_HEADER, name=name) + "\n" + _strip_comments(body)


def workspace_readme(*, name: str) -> str:
    return _fill(README, name=name)


def inputs_readme() -> str:
    return INPUTS_README


def machine_copy(source: Path, *, name: str) -> str:
    """A machine profile copied into the campaign folder, with a note on top."""
    header = (
        f"# Scheduler profile for the '{name}' campaign.\n"
        f"# Copied from the shipped profile {source.name} so it can be edited\n"
        f"# here -- partition, walltime, modules, VASP binary, POTCAR trees.\n"
        f"# Delete this file and set `machine: {source.stem}` in campaign.yaml\n"
        f"# to go back to the shipped one and pick up its updates.\n"
        f"#\n"
    )
    return header + source.read_text()


def recipe_copy(source: Path, *, name: str) -> str:
    """A DFT recipe copied into the campaign folder, with a note on top."""
    header = (
        f"# DFT recipe for the '{name}' campaign.\n"
        f"# Copied from the shipped recipe {source.name}. Every INCAR tag, the\n"
        f"# k-point density and the per-stage resources are here and editable.\n"
        f"# `csp recipe` prints it fully resolved, with nothing deferred.\n"
        f"# Delete this file and set `dft.recipe: {source.stem}` in campaign.yaml\n"
        f"# to go back to the shipped one.\n"
        f"#\n"
    )
    return header + source.read_text()
