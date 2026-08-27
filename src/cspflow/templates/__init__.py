"""Campaign scaffolds.

`csp init` emits only the Tier-1 keys -- the handful a new user must set.
`--full` adds the Tier-2 knobs, commented out with their defaults shown, so the
answer to "what can I change?" is in the file rather than in the source.
Tier-3 (raw INCAR tags, retry ladders, MP query details) is documented but
never templated.
"""

from __future__ import annotations

TIER1 = """\
# {name} -- cspflow campaign
#
# Only the keys below are required.  `csp init --full` writes the commonly
# tuned ones too; everything else is documented in pipeline.md.
name: {name}
machine: {machine}
workdir: /scratch/$USER/cspflow/{name}
archive: /projects/mmi/Ridwan/cspflow_archive/{name}

source:
  - mode: chemical_space          # chemical_space | composition_list | structure_list
    name: sweep
    chemical_space:
      groups:
        # `pick` has no default on purpose: one element per group gives binary
        # systems, allowing two adds ternaries and a 5-10x larger sweep.
        A: {{elements: [Sm, Tb],                   pick: 1}}
        B: {{elements: [Fe, Co, Ni],               pick: 1, min_fraction: 0.75}}
        C: {{elements: [Ti, V, Cr, Mn, Cu, Zn],    pick: 1}}
      max_atoms_formula: 20
      max_rare_earth: 1           # RE species per assembled system; null = no limit
    defaults:
      z: {{min: 1, max: 2}}
      max_atoms: 40
      n_structures: {{mode: per_atom, structures_per_atom: 2.0}}

generate:
  engine: mattergen
  mattergen:
    model: /projects/mmi/shuo/MatterGen_checkpoints/18-55-08

filter:
  e_above_hull_max: 0.10          # eV/atom

dft:
  recipe: magnets
"""

TIER2 = """
# ---------------------------------------------------------------------------
# Tier 2 -- commonly tuned.  Values shown are the defaults; uncomment to change.
# ---------------------------------------------------------------------------

# screen:
#   mlip: mattersim
#   mattersim: {mattersim_model: MatterSim-v1.0.0-5M.pth, fmax: 0.01, batch_size: 32}
#   dedup:
#     matcher: {ltol: 0.2, stol: 0.2, angle_tol: 5}

# reference:
#   functionals: [GGA]
#   thermo_type: GGA_GGA+U        # PINNED. MP mixes functionals silently; a
#                                 # reference set with more than one is refused.
#   energy_scale: raw             # raw | mp_corrected
#   mode: recompute               # mp_energies | recompute
#   prescreen_hull_max: 0.20      # widened for Phase A
#   snapshot: true                # freeze the MP query so the hull cannot move
#   relax_with_mlip: true

# calibrate:
#   mp:                           # free: MP's own DFT, at FIXED geometry
#     on_fail: warn
#     thresholds: {mae_e_per_atom: 0.05, spearman_min: 0.90, max_volume_drift: 0.05}
#   pilot:                        # costs pilot DFT: OUR DFT. the real gate.
#     on_fail: block
#     pilot_n: 40

# filter:
#   e_above_hull_max_source: calibrated   # literal | calibrated
#   max_per_composition: 5
#   spacegroup: {min_number: 3}

# dft:
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
#   incar_overrides: {}           # free-form; applied on top of the recipe
#   max_in_flight: 200
#   max_concurrent_tasks: 48
#   select: {max_per_composition: 3, max_total: 1500, budget_core_hours: 200000}

# analyze:
#   properties: [m_dft_raw, m_s_reconstructed, volume, spacegroup]
#   report: html
"""


def scaffold(*, name: str, machine: str = "orion", full: bool = False) -> str:
    text = TIER1.format(name=name, machine=machine)
    if full:
        text += TIER2.replace("{mattersim_model: ", "{model: ")
    return text
