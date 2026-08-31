"""Adopt a completed legacy campaign into cspflow's own layout.

`ingest.py` reads one thing: a `VASP_JOBS/` tree.  That was enough to prove the
state model could hold real work, and it is still the right tool when a
directory of VASP runs is all that survives.  It is not enough here.  The four
campaigns in `/projects/mmi/shuo` finished with six artefacts, and five of them
hold results that no directory walk can recover -- the MLIP energy of every
generated structure, which of them were duplicates, the MP reference set the
hull was built against, the DFT hull placement, and the prototype and moment of
every selected candidate.  Ingesting only the VASP directories would throw away
465,000 screened structures and keep 12,000.

So this module reads all six, and writes what cspflow would have written had it
run the campaign itself.  The point is not an archive format: it is that
`csp status`, the hull code, the report writer and the web portal all work
against the result without knowing it was adopted rather than run.

Artefacts read, and where each one lands
----------------------------------------

    mattergen_results/*/generation_summary.json
        per-formula generator ask                    -> composition.n_target

    VASP_JOBS/prescreening_structures.db  (ASE)
        the surviving geometries + MLIP energy,      -> ASE systems rows,
        space group, Pearson symbol, Wyckoff set        relaxation, hull(mlip),
                                                        filter_event

    VASP_JOBS/prescreening_stability.json
        every generated structure_id, pre-dedup      -> campaign_meta counters
                                                        (no geometry survives,
                                                        so no ASE row can exist)

    VASP_JOBS/mp_vaspdft.json  +  mp_mattersim.json
        the MP reference set, both scales            -> reference_entry

    VASP_JOBS/workflow.json  +  VASP_JOBS/<F>/<F>_sNNN/<Step>/
        what was submitted, and what came back       -> job, relaxation,
                                                        filter_event, dft_dir

    VASP_JOBS/dft_stability_results.json
        DFT hull placement + decomposition           -> hull(dft), data blob

    VASP_JOBS/hull_comparison.json
        MLIP-vs-DFT agreement over 2,000 points      -> calibration(kind='mp')

    candidates_w_proto_mag.csv
        the selected set: prototype, moment, volume  -> property, ASE kv

Three things this deliberately does NOT do
------------------------------------------

*It does not copy the VASP outputs.*  They are 549 GB across the four
campaigns and they are already on a read-only filesystem that is not going
anywhere.  `job.workdir` and the `dft_dir` key point into `/projects/mmi/shuo`,
and `campaign_meta['legacy.root']` records the prefix so a move can be repaired
with one UPDATE rather than a re-adoption.

*It does not invent rows for the structures dedup removed.*  93,637 of the
468,146 generated structures are known only by id and MLIP energy; their
geometry was discarded before the database Shuo kept was written.  ASE has no
row without an `Atoms`, and a fabricated cell would be a lie that every later
stage would believe.  The counts go into `campaign_meta` and the report
reconciles them out loud.

*It does not re-run anything.*  Every number written here was computed by the
original campaign.  Where two artefacts disagree -- OUTCAR says
-6.84927090909091 eV/atom and `dft_stability_results.json` says
-6.84927089909091 -- the parsed OUTCAR wins for the structure row and the JSON
wins for the hull row, because that is the number the published hull was
actually built from.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import chem
from .db.store import Origin, Store, StructureState
from .dft.vasp.parse import read_job_directory

SHUO = Path("/projects/mmi/shuo")

# Only a fallback.  The prescreening gate is a per-campaign setting and is read
# from each flow's own `prescreening_stability.json` summary: the two binary
# flows cut at 0.10 eV/atom and the two ternary ones at 0.06, and a hardcoded
# constant that happened to match half of them is exactly how the other half
# gets silently mis-described.
DEFAULT_PRESCREEN_HULL_MAX = 0.10

# Every energy here is a raw GGA total energy on MP's scale, computed with one
# settings set.  Stamped on the hull rows so `assert_hull_consistent` has
# something to check and a later, natively-computed hull cannot silently merge.
# A VASP energy that cannot be a result.  Every real per-atom energy in these
# campaigns runs between -6 and -14 eV; 43 runs of 12,193 came back POSITIVE,
# the worst at +3489.05 eV/atom, which is an electronic loop that diverged and
# was written down anyway.  Zero is used as the bound rather than something
# tuned, because there is no ambiguity to tune away: a bound intermetallic has
# a negative cohesive energy, and nothing between -6 and 0 appears in the data.
DIVERGED_E_PER_ATOM = 0.0

ENERGY_SCALE = "raw"
SETTINGS_HASH = "legacy-shuo-2026"


class LegacyError(Exception):
    """Anything that makes a legacy campaign unreadable."""


@dataclass(frozen=True)
class LegacyFlow:
    """One of Shuo's flows, and the campaign it becomes.

    `groups` is the chemical space as the flow actually ran it, recovered from
    the compositions it generated rather than from the script that generated
    them -- the script holds the intent, the composition list holds the fact,
    and where they differ the fact is what the database has to describe.
    """

    name: str
    source_dir: str
    groups: dict[str, dict[str, Any]]
    z_max: int = 3
    description: str = ""

    @property
    def root(self) -> Path:
        return SHUO / self.source_dir

    @property
    def vasp_jobs(self) -> Path:
        return self.root / "VASP_JOBS"

    def prescreen_hull_max(self) -> float:
        """The MLIP hull cut this flow actually applied, from its own summary.

        `prescreening_structures.db` carries a `passed_prescreening` column that
        is NOT this gate for the ternary flows -- see `_prescreen_gate`.  The
        summary is where the threshold that was used is written down.
        """
        path = self.vasp_jobs / "prescreening_stability.json"
        if not path.is_file():
            return DEFAULT_PRESCREEN_HULL_MAX
        # Matched in the file's first 64 KB rather than parsed: the real files
        # run to 75 MB and the summary is their first key, so this is a read of
        # one block instead of four hundred megabytes of Python objects.  A
        # regex rather than a line scan because the file may be one long line.
        head = path.read_text(errors="ignore")[:65536]
        found = re.search(r'"hull_threshold"\s*:\s*([0-9.eE+-]+)', head)
        return float(found.group(1)) if found else DEFAULT_PRESCREEN_HULL_MAX

    def chemsystems(self) -> list[str]:
        """Every chemical system the flow produced structures for.

        Read from `text_key_values` directly rather than through ASE: it is a
        `SELECT DISTINCT` over an indexed column, and going through `db.select`
        would deserialise 225,000 geometries to answer a question about 36
        strings.
        """
        import sqlite3

        path = self.vasp_jobs / "prescreening_structures.db"
        if not path.is_file():
            return []
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return sorted(r[0] for r in con.execute(
                "SELECT DISTINCT value FROM text_key_values WHERE key='chemsys'"))
        finally:
            con.close()

    def elements(self) -> list[str]:
        out: list[str] = []
        for group in self.groups.values():
            out.extend(group["elements"])
        return out


_RE_SM_TB = {"elements": ["Sm", "Tb"], "pick": 1}
_RE_GD_Y = {"elements": ["Gd", "Y"], "pick": 1}
_TM = {"elements": ["Fe", "Co", "Ni"], "pick": 1, "min_fraction": 0.75}
_SUB = {"elements": ["Ti", "V", "Cr", "Mn", "Cu", "Zn"], "pick": 1}

# Named for what they contain, not for the directory they came from.  Shuo's
# `new_bin_mag`/`new_ter_mag` are Gd+Y, not Y alone, and `bin_mag_flow` is
# Sm+Tb, not Sm alone -- checked against the 186 and 1,692 composition lists.
#
# The separator matters.  A campaign is a PRODUCT OF GROUPS -- {Sm,Tb} x
# {Fe,Co,Ni} -- while a hyphen already means "these elements in one system"
# everywhere else in this codebase and on the website (`Co-Tb-Zn`).  Written
# `sm-tb-binary`, the campaign appears to be about the Sm-Tb binary system, of
# which it contains exactly zero: `max_rare_earth: 1`, so Sm and Tb never
# co-occur in any structure.  Underscore separates groups, hyphen separates
# elements of one system, and the two never mean the same thing.
#
# `X` is the substituent group, following the RE-TM / RE-TM-X notation the
# magnet literature already uses (SmCo5 and Sm2Co17 are RE-TM; SmFe11Ti and
# Nd2Fe14B are RE-TM-X).  Its members are spelled out in every campaign.yaml.
REGISTRY: dict[str, LegacyFlow] = {
    "SmTb_FeCoNi_binary": LegacyFlow(
        name="SmTb_FeCoNi_binary", source_dir="bin_mag_flow",
        groups={"RE": _RE_SM_TB, "TM": _TM},
        description="{Sm,Tb} x {Fe,Co,Ni} -- RE-TM binaries",
    ),
    "SmTb_FeCoNi_X_ternary": LegacyFlow(
        name="SmTb_FeCoNi_X_ternary", source_dir="ter_mag_flow",
        groups={"RE": _RE_SM_TB, "TM": _TM, "X": _SUB},
        description="{Sm,Tb} x {Fe,Co,Ni} x {Ti,V,Cr,Mn,Cu,Zn} -- RE-TM-X ternaries",
    ),
    "GdY_FeCoNi_binary": LegacyFlow(
        name="GdY_FeCoNi_binary", source_dir="new_bin_mag",
        groups={"RE": _RE_GD_Y, "TM": _TM},
        description="{Gd,Y} x {Fe,Co,Ni} -- RE-TM binaries",
    ),
    "GdY_FeCoNi_X_ternary": LegacyFlow(
        name="GdY_FeCoNi_X_ternary", source_dir="new_ter_mag",
        groups={"RE": _RE_GD_Y, "TM": _TM, "X": _SUB},
        description="{Gd,Y} x {Fe,Co,Ni} x {Ti,V,Cr,Mn,Cu,Zn} -- RE-TM-X ternaries",
    ),
}


@dataclass
class AdoptStats:
    """What went in, so the report can reconcile it against what came out."""

    campaign: str = ""
    compositions: int = 0
    structures: int = 0
    generated_pre_dedup: int = 0
    dedup_removed: int = 0
    by_state: dict[str, int] = field(default_factory=dict)
    references: int = 0
    references_with_mlip: int = 0
    jobs: int = 0
    jobs_by_state: dict[str, int] = field(default_factory=dict)
    core_hours: float = 0.0
    dft_converged: int = 0
    dft_step_limit: int = 0
    diverged: int = 0
    dft_hull_rows: int = 0
    candidates: int = 0
    candidate_max_hull: float | None = None
    calibration: str = ""
    missing_dft_dirs: int = 0
    unmatched: list[str] = field(default_factory=list)

    def note_state(self, state: str) -> None:
        self.by_state[state] = self.by_state.get(state, 0) + 1

    def render(self) -> str:
        lines = [
            f"campaign            {self.campaign}",
            f"compositions        {self.compositions:,}   (formula, Z) rows",
            f"structures          {self.structures:,}   ASE rows with geometry",
        ]
        for state, n in sorted(self.by_state.items()):
            lines.append(f"    {state:<15} {n:,}")
        lines += [
            f"generated           {self.generated_pre_dedup:,}   before dedup",
            f"dedup removed       {self.dedup_removed:,}   id + MLIP energy only, no geometry",
            f"reference entries   {self.references:,}   ({self.references_with_mlip:,} with an MLIP energy)",
            f"DFT jobs            {self.jobs:,}",
        ]
        for state, n in sorted(self.jobs_by_state.items()):
            lines.append(f"    {state:<15} {n:,}")
        lines += [
            f"    converged       {self.dft_converged:,}",
            f"    scf diverged    {self.diverged:,}"
            + ("   <- positive energy, discarded" if self.diverged else ""),
            f"    ionic limit     {self.dft_step_limit:,}"
            + ("   <- finished cleanly but NOT relaxed" if self.dft_step_limit else ""),
            f"core-hours          {self.core_hours:,.0f}",
            f"DFT hull rows       {self.dft_hull_rows:,}",
            f"candidates          {self.candidates:,}"
            + (f"   (worst DFT hull {self.candidate_max_hull:.4f} eV/atom)"
               if self.candidate_max_hull is not None else ""),
        ]
        if self.calibration:
            lines.append(f"calibration         {self.calibration}")
        if self.missing_dft_dirs:
            lines.append(f"missing DFT dirs    {self.missing_dft_dirs:,}")
        if self.unmatched:
            lines.append(f"unmatched ids       {len(self.unmatched):,}")
            for u in self.unmatched[:5]:
                lines.append(f"    {u}")
            if len(self.unmatched) > 5:
                lines.append(f"    ... and {len(self.unmatched) - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# campaign.yaml
# ---------------------------------------------------------------------------

def render_config(flow: LegacyFlow, workdir: Path, *, machine: str = "orion",
                  chemsys: str | None = None) -> str:
    """The campaign file that would have produced this campaign.

    Written into the campaign directory rather than kept beside the code, so
    `csp status --campaign <dir>/campaign.yaml` works on the adopted result and
    the settings travel with the numbers they produced.

    With `chemsys`, the campaign is one chemical system: each group is narrowed
    to the elements that system actually contains, which is the sub-space this
    campaign covers and nothing wider.  `Co-Sm` becomes `RE: [Sm], TM: [Co]` --
    still a product of groups, still the same rule, one element per group.
    """
    wanted = set(chemsys.split("-")) if chemsys else None
    groups = []
    for label, group in flow.groups.items():
        elements = [e for e in group["elements"]
                    if wanted is None or e in wanted]
        if not elements:
            continue
        parts = [f"elements: [{', '.join(elements)}]", f"pick: {group['pick']}"]
        if "min_fraction" in group:
            parts.append(f"min_fraction: {group['min_fraction']}")
        groups.append(f"        {label}: {{{', '.join(parts)}}}")
    group_block = "\n".join(groups)
    gate = flow.prescreen_hull_max()
    name = chemsys or flow.name

    heading = (
        f"# {name} -- one chemical system of {flow.name}\n"
        f"#\n"
        f"# A campaign here IS a chemical system, so the name is written the way\n"
        f"# a system is written everywhere else: elements, sorted, hyphenated.\n"
        f"# The family it belongs to is the directory above, whose name is a\n"
        f"# product of element groups and uses underscores for exactly that\n"
        f"# reason -- the two separators never mean the same thing.\n"
        f"#\n"
        f"# Split per system because the MLIP-vs-DFT agreement is a property of\n"
        f"# the system, not of the run: within one of these flows it ranges from\n"
        f"# 0.03 to 0.97, and a single campaign-wide number describes none of it.\n"
        f"# See campaign_meta['calibration.*'] for this system's own score.\n"
        if chemsys else
        f"# {flow.name} -- adopted from /projects/mmi/shuo/{flow.source_dir}\n"
        f"#\n"
        f"# {flow.description}\n"
        f"#\n"
        f"# The name is a product of the groups below, underscore-separated; a\n"
        f"# hyphen would read as a chemical system, which is a different thing\n"
        f"# entirely (this campaign contains no system holding two rare earths).\n"
    )

    return f"""\
{heading}#
# NOT a runnable plan: this is the configuration reconstructed from a campaign
# that has already finished, so that the settings behind the numbers travel
# with them.  `source:` is the space the flow actually covered, recovered from
# its composition list.  Re-running it would regenerate work that already
# exists -- see campaign_meta['legacy.root'] for where the outputs live.
name: {name}
machine: {machine}
workdir: {workdir}

source:
  - mode: chemical_space
    name: sweep
    chemical_space:
      groups:
{group_block}
      max_atoms_formula: 20
      max_rare_earth: 1
    defaults:
      z: {{min: 1, max: {flow.z_max}}}
      max_atoms: 40
      n_structures: {{mode: per_atom, structures_per_atom: 2.0}}

generate:
  engine: mattergen
  mattergen:
    model: /projects/mmi/shuo/MatterGen_checkpoints/18-55-08
    mode: csp

screen:
  mlip: mattersim
  mattersim:
    model: MatterSim-v1.0.0-5M.pth

reference:
  functionals: [GGA]
  thermo_type: GGA_GGA+U
  energy_scale: raw
  mode: mp_energies
  prescreen_hull_max: {gate}

filter:
  e_above_hull_max: {gate}
  e_above_hull_max_source: literal

dft:
  recipe: magnets
  rare_earth:
    f_treatment: frozen

analyze:
  properties: [m_dft_raw, volume, spacegroup]
  report: html
"""


# ---------------------------------------------------------------------------
# reading the legacy artefacts
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise LegacyError(f"{path} is missing")
    with path.open() as handle:
        return json.load(handle)


def _ref_set_hash(entries: Iterable[dict], label: str) -> str:
    """Identity of a reference set: its mp_ids and their energies.

    Two hulls may only be compared when this is equal, so it is derived from
    the numbers rather than from the filename -- a re-fetched `mp_vaspdft.json`
    with one entry changed must not keep the old identity.
    """
    payload = sorted((e.get("mp_id", ""), round(float(e.get("energy", 0.0)), 6))
                     for e in entries)
    digest = hashlib.sha256(json.dumps([label, payload]).encode()).hexdigest()
    return digest[:16]


def _prescreen_gate(flow: LegacyFlow) -> tuple[float, set[str], set[str]]:
    """Which structures passed the MLIP hull gate, from the file that decided it.

    `prescreening_structures.db` has a `passed_prescreening` column and it is
    the wrong answer for both ternary flows.  Those campaigns tightened the cut
    from 0.10 to 0.06 eV/atom and never recomputed the column, so it still
    describes the looser gate:

        flow            threshold   db says   JSON says   got DFT
        bin_mag_flow    0.10          2,021       2,021     2,021
        ter_mag_flow    0.06         26,074       3,305     3,305
        new_bin_mag     0.10          4,536       4,536     4,536
        new_ter_mag     0.06         20,724       2,362     2,362

    The JSON agrees with `workflow.json` -- what was actually submitted -- in
    every case, and the database column agrees with it in two.  Taken from the
    column, the two ternary campaigns each gain ~20,000 structures marked as
    having passed a gate they did not pass and then mysteriously never run,
    which reads as a campaign that abandoned most of its own shortlist.

    Returns `(threshold, passed_ids, known_ids)`.  `known_ids` is what the JSON
    covers at all, so a structure it never saw can fall back to the column
    rather than being silently failed.
    """
    threshold = flow.prescreen_hull_max()
    path = flow.vasp_jobs / "prescreening_stability.json"
    if not path.is_file():
        return threshold, set(), set()

    payload = _load_json(path)
    passed, known = set(), set()
    for record in payload.get("results", []):
        sid = record.get("structure_id")
        if not sid:
            continue
        known.add(sid)
        if record.get("passed_prescreening"):
            passed.add(sid)
    del payload
    return threshold, passed, known


def _generation_summary(flow: LegacyFlow) -> dict[str, int]:
    """formula -> how many structures the generator returned for it."""
    matches = sorted((flow.root / "mattergen_results").glob("*/generation_summary.json"))
    if not matches:
        raise LegacyError(f"no generation_summary.json under {flow.root}/mattergen_results")
    summary = _load_json(matches[0])
    return {item["formula"]: int(item["structures_generated"])
            for item in summary["compositions"]}


# ---------------------------------------------------------------------------
# the phases
# ---------------------------------------------------------------------------

def _adopt_references(flow: LegacyFlow, store: Store, stats: AdoptStats,
                      chemsys: str | None = None) -> tuple[str, str]:
    """MP phases, both scales, into `reference_entry`.

    Shuo filtered MP's GGA_GGA+U thermo documents down to `run_type == 'GGA'`
    before building any hull, so both fields are recorded: the thermo document
    the entries came from, and the run type that was actually kept.  A later
    campaign that keeps GGA+U as well is then distinguishable from this one
    instead of looking identical.
    """
    dft_entries = _load_json(flow.vasp_jobs / "mp_vaspdft.json")
    mlip_path = flow.vasp_jobs / "mp_mattersim.json"
    mlip_entries = _load_json(mlip_path) if mlip_path.is_file() else []

    def _relevant(entries: list[dict]) -> list[dict]:
        """The subset whose elements this system's hull can actually use."""
        if chemsys is None:
            return entries
        elements = set(chemsys.split("-"))
        return [e for e in entries if set(e["composition"]) <= elements]

    dft_hash = _ref_set_hash(_relevant(dft_entries), f"mp_vaspdft:{chemsys or 'all'}")
    mlip_hash = (_ref_set_hash(_relevant(mlip_entries), f"mp_mattersim:{chemsys or 'all'}")
                 if mlip_entries else dft_hash)

    mlip_by_id: dict[str, dict] = {e["mp_id"]: e for e in mlip_entries}

    # A system's hull is built from the phases of that system and of every
    # sub-system: Co-Sm needs elemental Co, elemental Sm and the Co-Sm binaries,
    # and nothing else.  Carrying the whole flow's reference set into every
    # campaign would put Fe and Ni phases into a hull that has no Fe or Ni axis
    # -- pymatgen would either ignore them or refuse, and the ref_set_hash would
    # claim the campaigns shared a reference set they do not.
    wanted = set(chemsys.split("-")) if chemsys else None

    for entry in dft_entries:
        counts = {k: int(round(v)) for k, v in entry["composition"].items()}
        if wanted is not None and not set(counts) <= wanted:
            continue
        n_atoms = sum(counts.values())
        if n_atoms <= 0:
            continue
        fields: dict[str, Any] = {
            "mp_id": entry["mp_id"],
            "chemsys": entry.get("chemsys") or chem.chemsys(counts),
            "thermo_type": "GGA_GGA+U",
            "run_type": "GGA",
            "formula": chem.canonical_formula(counts),
            "n_atoms": n_atoms,
            "e_dft_raw": float(entry["energy"]) / n_atoms,
            "snapshot_id": dft_hash,
            "state": "fetched",
        }
        mlip = mlip_by_id.get(entry["mp_id"])
        if mlip is not None:
            mlip_counts = {k: int(round(v)) for k, v in mlip["composition"].items()}
            mlip_atoms = sum(mlip_counts.values()) or n_atoms
            fields["e_mlip_static"] = float(mlip["energy"]) / mlip_atoms
            fields["state"] = "static_done"
            stats.references_with_mlip += 1
        store.add_reference_entry(**fields)
        stats.references += 1

    return mlip_hash, dft_hash


def _adopt_structures(flow: LegacyFlow, store: Store, stats: AdoptStats,
                      mlip_hash: str, *, limit: int | None,
                      progress: Callable[[int], None] | None,
                      chemsys: str | None = None) -> dict[str, int]:
    """Every surviving geometry, with its MLIP result.  Returns legacy_id -> sid.

    Composition rows are keyed `(formula, z)` because the schema is, and because
    MatterGen returned a mix of cell sizes for every formula it was asked for --
    `Sm1Co10` came back at 11, 22 and 33 atoms.  Collapsing those onto one row
    would make `n_produced` the sum of three different things.
    """
    from ase.db import connect

    source = flow.vasp_jobs / "prescreening_structures.db"
    if not source.is_file():
        raise LegacyError(f"{source} is missing")

    targets = _generation_summary(flow)
    threshold, passed_ids, known_ids = _prescreen_gate(flow)
    comp_ids: dict[tuple[str, int], int] = {}
    comp_meta: dict[tuple[str, int], tuple[str, str, int, int, str]] = {}
    produced: dict[tuple[str, int], int] = {}
    sids: dict[str, int] = {}

    # Selected on the indexed `chemsys` key rather than filtered in Python:
    # the ternary source holds 225,000 rows and `toatoms()` on each one to
    # discard 218,000 of them would cost more than the whole adoption.
    db = connect(str(source))
    selection = f"chemsys={chemsys}" if chemsys else None
    for n, row in enumerate(db.select(selection), start=1):
        if limit is not None and n > limit:
            break
        kvp = row.key_value_pairs
        legacy_id = str(kvp.get("structure_id") or f"row{row.id}")
        formula_dir = str(kvp.get("composition") or "")

        try:
            counts = chem.parse_formula(formula_dir)
        except chem.ChemError:
            stats.unmatched.append(f"{legacy_id}: cannot parse composition {formula_dir!r}")
            continue
        reduced, _ = chem.reduce_counts(counts)
        canonical = chem.canonical_formula(reduced)
        per_unit = chem.n_atoms(reduced)
        z = max(1, round(row.natoms / per_unit)) if per_unit else 1

        key = (canonical, z)
        cid = comp_ids.get(key)
        if cid is None:
            cid = store.add_composition(
                formula=canonical, chemsys=chem.chemsys(reduced), z=z,
                n_atoms=per_unit * z, n_target=int(targets.get(formula_dir, 0)),
                source_mode="chemical_space", source_name="sweep", state="generated",
            )
            comp_ids[key] = cid
            comp_meta[key] = (canonical, chem.chemsys(reduced), z, per_unit, formula_dir)
            stats.compositions += 1
        produced[key] = produced.get(key, 0) + 1

        e_hull = float(kvp["e_above_hull"]) if "e_above_hull" in kvp else None
        # The JSON decided; the column is only consulted for a structure the
        # JSON never saw.  See `_prescreen_gate`.
        passed = (legacy_id in passed_ids if legacy_id in known_ids
                  else bool(kvp.get("passed_prescreening")))
        state = StructureState.selected if passed else StructureState.filtered_out

        kv: dict[str, Any] = {
            "composition_id": cid,
            "reduced_formula": canonical,
            "formula_dir": formula_dir,
            "legacy_id": legacy_id,
            "generator": "mattergen",
            "mlip_relaxed": True,
            # Every row in this database is a dedup survivor: Shuo's merge step
            # ran before the file was written.  Recorded under the same key the
            # dedup stage uses so `csp status` cannot tell the two apart.
            "dedup_checked": True,
        }
        if "e_mattersim" in kvp:
            kv["mlip_e_per_atom"] = float(kvp["e_mattersim"])
        if e_hull is not None:
            kv["mlip_e_above_hull"] = e_hull
        if not passed:
            kv["filter_reason"] = f"mlip e_above_hull > {threshold}"
        for src, dest in (("space_group_number", "spacegroup"),
                          ("density", "density"),
                          ("dof", "degrees_of_freedom")):
            if src in kvp:
                kv[dest] = int(kvp[src]) if dest != "density" else float(kvp[src])
        if kvp.get("pearson_symbol"):
            kv["pearson_symbol"] = str(kvp["pearson_symbol"])
        if kvp.get("wps"):
            kv["wyckoff"] = str(kvp["wps"])[:200]

        sid = store.add_structure(row.toatoms(), origin=Origin.generated,
                                  state=state, **kv)
        sids[legacy_id] = sid
        stats.structures += 1
        stats.note_state(state.value)

        if "e_mattersim" in kvp:
            # The legacy database stores only eV/atom.  The cell total is
            # exactly `e_per_atom * natoms` and a native `screen` writes both,
            # so it is derived rather than left NULL -- otherwise every adopted
            # campaign has 15,000 relaxation rows with a missing energy that
            # nothing is actually missing.
            per_atom = float(kvp["e_mattersim"])
            store.add_relaxation(structure_id=sid, engine="mattersim",
                                 energy=per_atom * row.natoms,
                                 e_per_atom=per_atom, converged=True)
        if e_hull is not None:
            store.add_hull(structure_id=sid, hull_type="mlip", energy_scale=ENERGY_SCALE,
                           e_above_hull=e_hull, ref_set_hash=mlip_hash,
                           settings_hash=SETTINGS_HASH)
            # `filter:e_above_hull` is what `filter_stage` writes for this
            # decision in a native campaign, and Shuo's "prescreening" is the
            # same cut on the same quantity.  Named anything else, a query that
            # asks "what passed the hull gate" answers correctly for one kind
            # of campaign and silently returns nothing for the other.
            store.add_filter_event(structure_id=sid, gate="filter:e_above_hull",
                                   passed=passed, value=e_hull,
                                   threshold=threshold)
        if progress and n % 5000 == 0:
            progress(n)

    # The generator was asked once per formula and chose the cell size itself,
    # so there is no per-(formula, Z) ask on record.  Splitting the formula's
    # ask in proportion to what each Z actually produced is the one distribution
    # that leaves the campaign totals true: sum(n_target) is the number of
    # structures MatterGen was asked for, sum(n_produced) is the number that
    # survived dedup, and `csp status` then reports the dedup loss instead of a
    # fictional 3x shortfall from counting the same ask on three rows.
    per_formula: dict[str, int] = {}
    for key, count in produced.items():
        per_formula[comp_meta[key][4]] = per_formula.get(comp_meta[key][4], 0) + count

    for key, cid in comp_ids.items():
        canonical, chemsys, z, per_unit, formula_dir = comp_meta[key]
        ask = int(targets.get(formula_dir, 0))
        total = per_formula.get(formula_dir, 0)
        share = round(ask * produced.get(key, 0) / total) if total else 0
        store.add_composition(
            formula=canonical, chemsys=chemsys, z=z, n_atoms=per_unit * z,
            n_target=share, source_mode="chemical_space", source_name="sweep",
            state="generated",
        )
        store.set_composition_state(cid, "generated", "", n_produced=produced.get(key, 0))
    return sids


def _adopt_dft(flow: LegacyFlow, store: Store, stats: AdoptStats,
               sids: dict[str, int], *,
               progress: Callable[[int], None] | None,
               chemsys: str | None = None) -> set[str]:
    """What was submitted (`workflow.json`) and what came back (the OUTCARs).

    The two are read together on purpose.  `workflow.json` knows the scheduler
    id and the state the manager last saw; only the directory knows whether the
    physics converged.  A job that finished cleanly at the ionic step limit is
    `RELAX_DONE` in one and `converged=False` in the other, and both facts are
    worth keeping.
    """
    workflow = _load_json(flow.vasp_jobs / "workflow.json")
    entries = workflow.get("structures", {})

    state_map = {
        "RELAX_DONE": "done", "STATIC_DONE": "done", "DONE": "done",
        "RELAX_TMOUT": "timeout", "STATIC_TMOUT": "timeout", "TIMEOUT": "timeout",
        "RELAX_FAIL": "failed", "STATIC_FAIL": "failed", "FAILED": "failed",
        "RELAX_RUNNING": "running", "PENDING": "pending",
    }

    diverged: set[str] = set()
    for n, (legacy_id, record) in enumerate(sorted(entries.items()), start=1):
        # `workflow.json` covers the whole flow; this campaign is one system of
        # it.  Skipped on the record's own `chemsys` rather than on a missing
        # sid, so a genuinely unmatched id is still reported instead of being
        # lost among 200,000 that simply belong to a different campaign.
        if chemsys is not None and record.get("chemsys") != chemsys:
            continue
        sid = sids.get(legacy_id)
        if sid is None:
            stats.unmatched.append(f"{legacy_id}: DFT job with no screened structure")
            continue

        raw_state = str(record.get("state", ""))
        job_state = state_map.get(raw_state, "failed")
        rel = record.get("relax_dir") or ""
        directory = (flow.root / rel) if rel else None
        step = Path(rel).name or "Relax"

        job_id = store.add_job(stage="dft", structure_id=sid, recipe_step=step,
                               workdir=str(directory) if directory else "")
        stats.jobs += 1

        outcome = None
        if directory is not None and directory.is_dir():
            outcome = read_job_directory(directory)
        else:
            stats.missing_dft_dirs += 1

        # An energy that cannot be one.  VASP exited cleanly, wrote an OUTCAR
        # and reported a POSITIVE per-atom energy -- the electronic loop
        # diverged and the number it left behind is not a measurement of
        # anything.  Recorded as a failure with the value in the reason, and
        # never as `vasp_energy`: put in that column it enters the hull, and
        # +3489 eV/atom against reference phases near -8 produces a hull
        # distance of 3495 that sorts, plots and averages like a real one.
        if outcome is not None and outcome.e_per_atom is not None \
                and outcome.e_per_atom >= DIVERGED_E_PER_ATOM:
            diverged.add(legacy_id)
            stats.diverged += 1
            store.update_job(job_id, state="failed", slurm_id=outcome.slurm_id,
                             core_hours=outcome.core_hours,
                             exit_reason="scf_diverged", attempt=1)
            stats.jobs_by_state["failed"] = stats.jobs_by_state.get("failed", 0) + 1
            stats.core_hours += outcome.core_hours
            store.add_filter_event(
                structure_id=sid, gate=f"dft:{step.lower()}:converged", passed=False,
                detail=f"scf diverged: {outcome.e_per_atom:+.2f} eV/atom")
            store.set_structure_state(
                sid, StructureState.failed,
                dft_dir=str(directory.resolve()),
                legacy_job_state=raw_state,
                dft_fail_reason=(f"{step}: SCF diverged, VASP reported "
                                 f"{outcome.e_per_atom:+.2f} eV/atom")[:200])
            if progress and n % 500 == 0:
                progress(n)
            continue

        if outcome is not None:
            store.update_job(job_id, state=outcome.state, slurm_id=outcome.slurm_id,
                             core_hours=outcome.core_hours,
                             exit_reason=outcome.exit_reason, attempt=1)
            stats.jobs_by_state[outcome.state] = stats.jobs_by_state.get(outcome.state, 0) + 1
            stats.core_hours += outcome.core_hours
            if outcome.converged:
                stats.dft_converged += 1
            elif outcome.exit_reason == "ionic_step_limit":
                stats.dft_step_limit += 1

            # Only when there is an energy, matching `dft_stage`.  A relaxation
            # row with a NULL energy records nothing the `job` row does not
            # already say, and it would make `csp status`'s relaxation counts
            # disagree between an adopted campaign and a native one.
            if outcome.energy is not None:
                store.add_relaxation(structure_id=sid, engine=f"vasp:{step.lower()}",
                                     energy=outcome.energy,
                                     e_per_atom=outcome.e_per_atom,
                                     converged=outcome.converged,
                                     n_steps=outcome.n_ionic_steps)
            store.add_filter_event(
                # Lowercase to match the recipe's own stage names (`relax`,
                # `static`), which is what `dft_stage` interpolates here.  The
                # legacy directory is `Relax`; the gate is not the directory.
                structure_id=sid, gate=f"dft:{step.lower()}:converged",
                passed=outcome.converged, value=float(outcome.n_ionic_steps),
                threshold=float(outcome.step_limit) if outcome.step_limit else None,
                detail=outcome.exit_reason,
            )

            kv: dict[str, Any] = {
                "dft_dir": str(directory.resolve()),
                "dft_step": 1,
                "dft_attempt": 1,
                "converged": outcome.converged,
                "legacy_job_state": raw_state,
            }
            if outcome.energy is not None:
                kv["vasp_energy"] = outcome.energy
            if outcome.e_per_atom is not None:
                kv["e_per_atom"] = outcome.e_per_atom
            if outcome.magnetisation is not None:
                kv["magnetisation"] = outcome.magnetisation

            new_state = (StructureState.dft_done if outcome.state == "done"
                         else StructureState.failed)
            if new_state is StructureState.failed:
                kv["dft_fail_reason"] = f"{step}: {outcome.exit_reason or raw_state}"[:200]
            store.set_structure_state(sid, new_state, **kv)
        else:
            store.update_job(job_id, state=job_state, exit_reason="no directory on disk",
                             slurm_id=str(record.get("relax_job_id", "")), attempt=1)
            stats.jobs_by_state[job_state] = stats.jobs_by_state.get(job_state, 0) + 1
            store.set_structure_state(sid, StructureState.failed,
                                      dft_fail_reason="DFT directory not on disk",
                                      legacy_job_state=raw_state)

        if progress and n % 500 == 0:
            progress(n)

    # `by_state` was counted while the screening states were written; the DFT
    # phase moved a few thousand of those rows, so it is recounted from the
    # database rather than adjusted in place.
    stats.by_state = {
        r["state"]: r["n"] for r in store.sql.execute(
            "SELECT value AS state, COUNT(*) n FROM text_key_values "
            "WHERE key='state' GROUP BY value")
    }
    return diverged


def _adopt_dft_hull(flow: LegacyFlow, store: Store, stats: AdoptStats,
                    sids: dict[str, int], dft_hash: str,
                    diverged: set[str] | None = None) -> None:
    """The published DFT hull placement, and what each structure decomposes into.

    The decomposition goes into the ASE `data=` blob rather than a table: it is
    a variable-length list of (phase, fraction, energy) that is displayed and
    never queried, which is exactly the case the blob exists for.
    """
    payload = _load_json(flow.vasp_jobs / "dft_stability_results.json")
    for entry in payload.get("results", []):
        sid = sids.get(entry["structure_id"])
        if sid is None:
            continue
        # The published hull placement for a diverged run is arithmetic on a
        # number that was never a measurement: Gd1Ni19_s013 is listed at
        # 3495.22 eV/atom above the hull.  Dropped rather than stored and
        # flagged -- a hull row is a placement, and there is no placement here.
        if diverged and entry["structure_id"] in diverged:
            continue
        e_hull = entry.get("energy_above_hull")
        if e_hull is None:
            continue
        store.add_hull(structure_id=sid, hull_type="dft", energy_scale=ENERGY_SCALE,
                       e_above_hull=float(e_hull), ref_set_hash=dft_hash,
                       settings_hash=SETTINGS_HASH)
        store.add_property(structure_id=sid, key="dft_e_above_hull", source="dft",
                           value=float(e_hull))
        kv = {"dft_e_above_hull": float(e_hull)}
        data = {}
        if entry.get("decomposition"):
            data["decomposition"] = entry["decomposition"]
        store.update_structure(sid, data=data or None, **kv)
        stats.dft_hull_rows += 1


def _adopt_candidates(flow: LegacyFlow, store: Store, stats: AdoptStats,
                      sids: dict[str, int]) -> list[dict[str, str]]:
    """The selected set: prototype, moment, volume.

    `source` on each property row is the honest part.  The moment and volume are
    computed (`dft`); the AFLOW prototype is a match against a library, so it is
    a `model` -- reading them back later, nothing has to remember which was
    which.
    """
    path = flow.root / "candidates_w_proto_mag.csv"
    if not path.is_file():
        return []

    numeric = {
        "total_mag": ("m_dft_raw", "dft"),
        "cell_volume": ("volume", "dft"),
        "mag_per_vol": ("m_per_volume", "dft"),
        "spg_num": ("spacegroup", "dft"),
    }
    textual = {
        "aflow_proto": ("aflow_prototype", "model"),
        "aflow_anrl": ("aflow_anrl", "model"),
        "pearson_symbol": ("pearson_symbol", "model"),
        "match_type": ("proto_match_type", "model"),
    }

    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            sid = sids.get(record["structure_id"])
            if sid is None:
                # The CSV covers the whole flow; this campaign is one system of
                # it.  Not an error, and not this campaign's row -- appending it
                # regardless would give every system a copy of the flow's entire
                # candidate table.
                continue
            rows.append(record)

            kv: dict[str, Any] = {"candidate": True}
            for column, (key, source) in numeric.items():
                raw = (record.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                store.add_property(structure_id=sid, key=key, source=source, value=value)
                kv[key] = int(value) if key == "spacegroup" else value
            for column, (key, source) in textual.items():
                raw = (record.get(column) or "").strip()
                if not raw:
                    continue
                store.add_property(structure_id=sid, key=key, source=source,
                                   text_value=raw[:200])
                kv[key] = raw[:200]

            # No threshold: the candidate table is not a hull cut.  Every
            # candidate sits under 0.10 eV/atom, but so do 1,262 of
            # bin_mag_flow's 2,021 DFT results and only 104 were selected --
            # the rest of the rule is an AFLOW prototype match and a moment,
            # and writing 0.10 here would assert a gate that does not explain
            # the set.  The value is recorded; the rule is described.
            store.add_filter_event(
                structure_id=sid, gate="select:candidate", passed=True,
                value=float(record["dft_e_hull"]) if record.get("dft_e_hull") else None,
                detail="in the campaign's candidate table: DFT hull distance "
                       "plus an AFLOW prototype match and a computed moment")
            store.update_structure(sid, **kv)
            stats.candidates += 1
            hull = kv.get("dft_e_above_hull")
            if hull is None:
                try:
                    hull = float(record["dft_e_hull"])
                except (KeyError, TypeError, ValueError):
                    hull = None
            if hull is not None:
                stats.candidate_max_hull = (hull if stats.candidate_max_hull is None
                                            else max(stats.candidate_max_hull, hull))
    return rows


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, with ties averaged.  None below 8 points.

    Implemented here rather than pulled from scipy because `legacy.py` is
    imported by the CLI and the CLI must stay importable on a login node with
    nothing but the standard library plus ASE.
    """
    n = len(xs)
    if n < 8:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def _adopt_calibration(flow: LegacyFlow, store: Store, stats: AdoptStats,
                       provenance_id: int) -> None:
    """MLIP-vs-DFT agreement, as Stage 4a would have recorded it.

    Shuo's comparison is the same measurement cspflow calls `calibrate.mp`:
    both energies at a fixed geometry over a few thousand points.  Recording it
    under that name means `csp status` reports a calibration for this campaign
    instead of an empty table, and the verdict is derived from cspflow's own
    thresholds rather than copied from a script that had different ones.
    """
    # Computed from this campaign's own pairs rather than copied from the
    # flow-wide `hull_comparison.json`.  That file reports one number for a
    # whole flow, and the whole point of splitting per system is that the
    # number is not one number: inside a single flow it runs from 0.03 to 0.97.
    # Diverged runs never had a hull row written, so they are already absent.
    pairs = store.sql.execute(
        "SELECT m.e_above_hull AS mlip, d.e_above_hull AS dft "
        "FROM hull m JOIN hull d ON m.structure_id = d.structure_id "
        "WHERE m.hull_type='mlip' AND d.hull_type='dft'"
    ).fetchall()
    if not pairs:
        return

    xs = [float(r["mlip"]) for r in pairs]
    ys = [float(r["dft"]) for r in pairs]
    spearman = _spearman(xs, ys)
    mae = sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs)

    # cspflow's own gate, applied to a legacy result rather than copied from a
    # script that used different thresholds.  `fail` is not a criticism of the
    # campaign -- it says this system's screening did not rank well enough to
    # justify trusting its MLIP column where the DFT column is blank.
    verdict = "pass"
    if spearman is None or spearman < 0.90:
        verdict = "warn"
    if mae > 0.05 or (spearman is not None and spearman < 0.30):
        verdict = "fail"

    store.add_calibration(
        kind="mp", n_points=len(xs), verdict=verdict, provenance_id=provenance_id,
        mae_e_hull=mae, spearman=spearman,
        detail=(f"MLIP vs DFT hull distance over {len(xs)} structures of this "
                f"system; diverged runs excluded")[:500],
    )
    stats.calibration = (
        f"n={len(xs)} mae={mae:.4f} rho={spearman:.3f} -> {verdict}"
        if spearman is not None else f"n={len(xs)} mae={mae:.4f} -> {verdict}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

LEGACY_FILES = (
    "VASP_JOBS/mp_vaspdft.json",
    "VASP_JOBS/mp_mattersim.json",
    "VASP_JOBS/hull_comparison.json",
    "VASP_JOBS/energy_method_comparison.json",
    "VASP_JOBS/workflow.json",
    "candidates_w_proto_mag.csv",
    "candidates_w_proto.csv",
)


def _copy_legacy_files(flow: LegacyFlow, out: Path) -> None:
    """The small source artefacts, verbatim, beside the database they became.

    Not the 549 GB of VASP output and not the 75 MB prescreening dump: the
    files here are the ones a reviewer would want to diff against the tables,
    and together they are under 3 MB.
    """
    out.mkdir(parents=True, exist_ok=True)
    for rel in LEGACY_FILES:
        src = flow.root / rel
        if src.is_file():
            shutil.copy2(src, out / Path(rel).name)
    for summary in sorted((flow.root / "mattergen_results").glob("*/generation_summary.json")):
        shutil.copy2(summary, out / "generation_summary.json")
        break


def _seal(db_path: Path) -> None:
    """Fold the write-ahead log back into the file before the directory moves.

    A campaign runs in WAL mode, so it is three files.  Moving all three is
    fine; arriving with an un-checkpointed `-wal` is not, and the way that
    fails is `Store._check_sidecars` refusing to open the campaign -- or, if the
    `.db` is the one that goes missing, SQLite reporting a bare "disk I/O
    error".  One checkpoint costs a second and removes the whole class.

    The `gc.collect()` is load-bearing, and was found the hard way.  `Store.ase`
    hands out a fresh ASE connection per write and never closes it explicitly;
    the underlying SQLite handle is released when that object is collected, not
    when `Store.close()` returns.  So immediately after a run that wrote
    structures the file still has a live reader, `PRAGMA journal_mode=DELETE`
    fails with "database is locked" against a database nothing appears to have
    open, and the `-wal`/`-shm` pair survives into the moved directory.  With
    the collection first, SQLite tidies both sidecars away by itself.
    """
    import gc
    import sqlite3

    gc.collect()

    # Autocommit, and every PRAGMA's rows drained: a PRAGMA that returns rows
    # holds a read transaction open until it is read, and the next PRAGMA on
    # the same connection then fails on the lock it is itself holding.
    con = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        con.execute("PRAGMA journal_mode=DELETE").fetchall()
    finally:
        con.close()
    for sidecar in (Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if sidecar.exists():
            sidecar.unlink()


def _adopt_one(flow: LegacyFlow, chemsys: str, build: Path, final: Path, *,
               machine: str, limit: int | None,
               say: Callable[[str], None]) -> AdoptStats:
    """One chemical system, as its own campaign."""
    build.mkdir(parents=True)
    stats = AdoptStats(campaign=chemsys)

    config_text = render_config(flow, final, machine=machine, chemsys=chemsys)
    (build / "campaign.yaml").write_text(config_text)

    store = Store.create(build / "campaign.db", campaign=chemsys)
    # A rebuild-from-source, so a crash costs a re-run and never a wrong answer.
    # Roughly 3x on the write path, and the only pragma that is safe here:
    # `journal_mode=MEMORY` is faster still and deadlocks, because `Store.ase`
    # opens a second connection to this same file for every structure written
    # and an in-memory rollback journal is not shared between connections.
    store.sql.execute("PRAGMA synchronous=OFF")

    with store:
        provenance_id = store.add_provenance(
            config_hash=hashlib.sha256(config_text.encode()).hexdigest()[:16],
            settings_hash=SETTINGS_HASH, machine=machine,
            code_version="legacy-adopt",
            resolved_config={"legacy_flow": flow.source_dir,
                             "family": flow.name, "chemsys": chemsys,
                             "adopted": True},
        )
        for key, value in (
            ("legacy.root", str(flow.root)),
            ("legacy.flow", flow.source_dir),
            ("family", flow.name),
            ("chemsys", chemsys),
            ("legacy.description", flow.description),
            ("legacy.adopted_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("legacy.vasp_outputs", "not copied; dft_dir points into legacy.root"),
            ("legacy.mlip", "MatterSim-v1.0.0-5M.pth"),
            ("legacy.prescreen_hull_max", str(flow.prescreen_hull_max())),
        ):
            store.set_meta(key, value)

        mlip_hash, dft_hash = _adopt_references(flow, store, stats, chemsys=chemsys)
        store.set_meta("ref_set_hash.mlip", mlip_hash)
        store.set_meta("ref_set_hash.dft", dft_hash)

        sids = _adopt_structures(flow, store, stats, mlip_hash, limit=limit,
                                 progress=None, chemsys=chemsys)
        if not sids:
            raise LegacyError(f"{chemsys}: no structures in the source database")

        store.set_meta("legacy.unique", str(stats.structures))

        diverged = _adopt_dft(flow, store, stats, sids, progress=None, chemsys=chemsys)
        _adopt_dft_hull(flow, store, stats, sids, dft_hash, diverged)
        store.assert_hull_consistent(dft_hash)

        rows = _adopt_candidates(flow, store, stats, sids)
        if stats.candidate_max_hull is not None:
            store.set_meta("legacy.candidate_max_dft_e_above_hull",
                           f"{stats.candidate_max_hull:.5f}")

        _adopt_calibration(flow, store, stats, provenance_id)

    _seal(build / "campaign.db")
    if rows:
        report = build / "report"
        report.mkdir(exist_ok=True)
        with (report / "candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # `dft/` is where a native campaign's job directories live.  Left empty and
    # named, so the layout matches and the README explains where they are.
    (build / "dft").mkdir(exist_ok=True)
    say(f"    {chemsys:<12} {stats.structures:>7,} structures  "
        f"{stats.candidates:>4} candidates  {stats.calibration}")
    return stats


def adopt(flow: LegacyFlow, dest: Path, *, staging: Path | None = None,
          limit: int | None = None, machine: str = "orion",
          progress: Callable[[str], None] | None = None) -> AdoptStats:
    """Write one legacy flow as a family of per-system campaigns.

        dest/<flow.name>/            the family -- a product of element groups
            <chemsys>/               one campaign, one chemical system
                campaign.yaml
                campaign.db
                dft/  legacy/  report/

    A campaign is one chemical system because the question campaigns are judged
    by has that granularity.  The MLIP-vs-DFT rank correlation that decides
    whether a screen was worth running ranges from 0.03 to 0.97 inside a single
    flow -- `Co-Y` at 0.687 and `Gd-Ni` at 0.027 were the same run, the same
    settings, the same code.  Pooled they report 0.375, which describes neither
    and hides that the screen worked for one rare earth and not the other.
    cspflow stores one calibration per campaign, so one campaign per system is
    what makes that verdict storable at the granularity it is true at.

    `staging` exists because of a measurement, not a preference: SQLite commits
    at 300 rows/s on the scratch NFS mount and at 4,700 rows/s on node-local
    disk, and this writes on the order of two million rows.  Built in place the
    adoption takes hours; built locally and moved as one directory it takes
    minutes.
    """
    final = dest / flow.name
    build = (staging / flow.name) if staging else final
    if build.exists() or final.exists():
        raise LegacyError(
            f"{build if build.exists() else final} already exists. Adoption is a "
            f"rebuild, not an update: remove it first, or pass a different "
            f"destination.")
    build.mkdir(parents=True)

    say = progress or (lambda _msg: None)
    systems = flow.chemsystems()
    if not systems:
        raise LegacyError(f"{flow.root}: no chemical systems in the source database")
    say(f"  {len(systems)} chemical systems")

    family = AdoptStats(campaign=flow.name)
    per_system: list[tuple[str, AdoptStats]] = []
    for chemsys in systems:
        stats = _adopt_one(flow, chemsys, build / chemsys, final / chemsys,
                           machine=machine, limit=limit, say=say)
        per_system.append((chemsys, stats))
        family.compositions += stats.compositions
        family.structures += stats.structures
        family.references += stats.references
        family.references_with_mlip += stats.references_with_mlip
        family.jobs += stats.jobs
        family.core_hours += stats.core_hours
        family.dft_converged += stats.dft_converged
        family.dft_step_limit += stats.dft_step_limit
        family.diverged += stats.diverged
        family.dft_hull_rows += stats.dft_hull_rows
        family.candidates += stats.candidates
        family.missing_dft_dirs += stats.missing_dft_dirs
        family.unmatched.extend(stats.unmatched)
        for state, n in stats.by_state.items():
            family.by_state[state] = family.by_state.get(state, 0) + n
        for state, n in stats.jobs_by_state.items():
            family.jobs_by_state[state] = family.jobs_by_state.get(state, 0) + n

    # Everything the flow generated, including what dedup discarded, which is a
    # property of the flow and not of any one system.
    prescreen = _load_json(flow.vasp_jobs / "prescreening_stability.json").get("summary", {})
    family.generated_pre_dedup = int(prescreen.get("total_structures") or 0)
    family.dedup_removed = max(0, family.generated_pre_dedup - family.structures)

    _copy_legacy_files(flow, build / "legacy")
    _write_family_index(flow, build, per_system, family)

    if staging:
        say(f"  moving {build} -> {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(build), str(final))
    family.calibration = _rank_line(per_system)
    return family


def _rank_line(per_system: list[tuple[str, AdoptStats]]) -> str:
    scored = [(s.calibration, cs) for cs, s in per_system if s.calibration]
    return f"{len(scored)} systems scored" if scored else ""


def _write_family_index(flow: LegacyFlow, build: Path,
                        per_system: list[tuple[str, AdoptStats]],
                        family: AdoptStats) -> None:
    """A machine-readable roll-up beside the systems it summarises.

    Written so that "which systems are good and which are bad" is answerable
    without opening 36 databases -- by the website, by a shell script, and by
    anyone browsing the directory.
    """
    systems = []
    for chemsys, stats in per_system:
        systems.append({
            "chemsys": chemsys,
            "n_structures": stats.structures,
            "n_compositions": stats.compositions,
            "n_dft": stats.jobs,
            "n_converged": stats.dft_converged,
            "n_diverged": stats.diverged,
            "n_candidates": stats.candidates,
            "calibration": stats.calibration,
        })
    payload = {
        "family": flow.name,
        "description": flow.description,
        "legacy_flow": flow.source_dir,
        "prescreen_hull_max": flow.prescreen_hull_max(),
        "adopted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totals": {
            "n_systems": len(per_system),
            "n_structures": family.structures,
            "n_compositions": family.compositions,
            "n_dft": family.jobs,
            "n_converged": family.dft_converged,
            "n_diverged": family.diverged,
            "n_candidates": family.candidates,
            "generated_pre_dedup": family.generated_pre_dedup,
            "dedup_removed": family.dedup_removed,
        },
        "systems": systems,
    }
    (build / "family.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
