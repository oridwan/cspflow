"""Recipes: an ordered list of DFT stages, each with a complete INCAR.

**No pymatgen input sets.** Decided in §12.5, and for two reasons beyond the
user's preference, both of which have already bitten this project:

1. `MPRelaxSet`'s values are *pymatgen version state*, not campaign state. An
   upgrade silently moves `ENCUT`, the `LDAU` tables or `SIGMA` mid-campaign, and
   provenance recording `base_set: MPRelaxSet` reproduces nothing.
2. It injects physics nobody asked for — `LDAU` tables keyed on element *and*
   anion, a default `MAGMOM`, `ISPIN`, `LMAXMIX` — silently.

**The trap in dropping it.** `MPRelaxSet` was supplying values VASP does not
default sensibly. Delete it without writing them down and you do not get user
control, you get VASP's defaults:

| tag | needed | VASP's default | consequence |
|---|---|---|---|
| `ENCUT` | 520 eV | `max(ENMAX)` over the POTCARs | **changes with composition** |
| `ISPIN` | 2 | 1, non-spin-polarised | every moment is zero |
| `LMAXMIX` | 6 (f in valence) / 4 (d) | 2 | wrong mixing; false convergence |
| `LASPH` | `.TRUE.` | `.FALSE.` | aspherical terms off; matters for 4f |
| `NELM` | 200 | 60 | SCF gives up and the job exits "successfully" |
| `LORBIT` | 11 | unset | no site moments; Stage 7 has nothing to read |

`validate_recipe()` refuses a recipe missing any of these, naming what VASP
would have done instead.

Three rules govern a recipe file:

*   **`inherit:` copies, it does not defer.** A resolved stage contains every
    tag literally; there is never a value that requires knowing pymatgen to
    predict.
*   **Structure-dependent tags are the only non-literal ones** — `MAGMOM`,
    `NBANDS`, `LDAU*`, `SYSTEM`, and `LMAXMIX` when it is derived from the
    POTCARs. Each is written into the emitted INCAR, so the file on disk is
    still complete and self-describing.
*   **Any tag may be added.** `incar:` is a free-form mapping with no whitelist.
    Unknown tags are written as given and warned about, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RECIPE_DIR = Path(__file__).resolve().parent / "recipes"

# Tags VASP does not default sensibly. Each maps to what it would do instead.
REQUIRED_TAGS: dict[str, str] = {
    "ENCUT": "VASP would use max(ENMAX) over the POTCARs, which CHANGES WITH "
             "COMPOSITION -- measured 267.9 eV for Sm_3/Fe/Ti and 295.4 eV for "
             "anything with Cu. A hull built on that compares incomparable numbers, "
             "and the error does not cancel in a formation energy",
    "ISPIN": "VASP would use 1 (non-spin-polarised): every moment would be zero",
    "LMAXMIX": "VASP would use 2, which is wrong mixing for d and f states -- slow "
               "or falsely-converged SCF",
    "LASPH": "VASP would use .FALSE., turning off aspherical corrections that matter "
             "for 4f and for any MAE work",
    "NELM": "VASP would use 60; the SCF gives up early and the job exits successfully "
            "having not converged",
    "LORBIT": "VASP would leave it unset: no site-projected moments, so Stage 7 has "
              "nothing to analyse",
}

# Tags whose value is computed per structure rather than written in the recipe.
# `LMAXMIX` is here because it is derived from the POTCARs -- 6 when 4f is in
# valence, 4 for d-only chemistries -- and getting that from the actual
# pseudopotentials is more reliable than asking the user to keep it in step with
# `dft.rare_earth.f_treatment`. Setting it explicitly in a recipe still wins.
COMPUTED_TAGS = frozenset({"MAGMOM", "NBANDS", "SYSTEM", "LMAXMIX", "LDAUU",
                           "LDAUJ", "LDAUL", "LDAU", "LDAUTYPE", "LDAUPRINT"})

# A conservative list of real VASP tags, used only to warn. Deliberately not a
# whitelist: an unrecognised tag is written out as given, because refusing one
# would make the "add any tag you like" rule false.
KNOWN_TAGS = frozenset("""
ADDGRID AEXX AGGAC AGGAX ALGO AMIN AMIX AMIX_MAG ANDERSEN_PROB ANTIRES APACO
BMIX BMIX_MAG CH_LSPEC CH_NEDOS CH_SIGMA CLL CLN CLZ CMBJ CMBJA CMBJB CSHIFT
DEPER DIPOL EBREAK EDIFF EDIFFG EFIELD EFIELD_PEAD EINT EMAX EMIN ENAUG ENCUT
ENCUTFOCK ENCUTGW ENINI EPSILON ESTOP EVENONLY EVENONLYGW FERDO FERWE FINDIFF
GGA GGA_COMPAT HFLMAX HFRCUT HFSCREEN HILLS_BIN HILLS_H HILLS_W I_CONSTRAINED_M
IALGO IBAND IBRION ICHARG ICHIBARE ICORELEVEL IDIPOL IEPSILON IGPAR IMAGES
IMIX INCREM INIMIX INIWAV IPEAD ISIF ISMEAR ISPIN ISTART ISYM IVDW IWAVPR
KBLOCK KGAMMA KPAR KPOINT_BSE KPUSE LADDER LAECHG LAMBDA LANGEVIN_GAMMA LASPH
LASYNC LATTICE_CONSTRAINTS LBERRY LBLUEOUT LBONE LCALCEPS LCALCPOL LCHARG
LCHIMAG LCORR LDAU LDAUJ LDAUL LDAUPRINT LDAUTYPE LDAUU LDIAG LDIPOL LEFG
LELF LEPSILON LFOCKAEDFT LHARTREE LHFCALC LHYPERFINE LKPROJ LLRAUG LMAXFOCK
LMAXFOCKAE LMAXMIX LMAXPAW LMAXTAU LMONO LNABLA LNMR_SYM_RED LNONCOLLINEAR
LOPTICS LORBIT LORBMOM LPARD LPEAD LPLANE LREAL LRPA LSCAAWARE LSCALAPACK
LSCALU LSCSGRAD LSELFENERGY LSEPB LSEPK LSORBIT LSPECTRAL LSPECTRALGW LSUBROT
LTHOMAS LUSE_VDW LVDW_EWALD LVHAR LVTOT LWANNIER90 LWAVE LWRITE_MMN_AMN
LZEROZ M_CONSTR MAGMOM MAXMEM MAXMIX MDALGO METAGGA MIXPRE ML_LMLFF NBANDS
NBANDSGW NBANDSO NBANDSV NBLK NBLOCK NBMOD NCORE NCRPA_BANDS NDAV NEDOS NELECT
NELM NELMDL NELMIN NFREE NGX NGXF NGY NGYF NGZ NGZF NKRED NKREDX NKREDY NKREDZ
NLSPLINE NMAXFOCKAE NOMEGA NOMEGAR NPACO NPAR NPPSTR NSIM NSUBSYS NSW NTAUPAR
NUPDOWN NWRITE ODDONLY ODDONLYGW OFIELD_A OFIELD_KAPPA OFIELD_Q6_FAR
OFIELD_Q6_NEAR OMEGAMAX OMEGAMIN OMEGATL PARAM1 PARAM2 PFLAT PHON_NSTRUCT
PLEVEL PMASS POMASS POTIM PREC PRECFOCK PROUTINE PSTRESS PTHRESHOLD QSPIRAL
QUAD_EFG RANDOM_SEED ROPT RWIGS SAXIS SCSRAD SHAKEMAXITER SHAKETOL SIGMA
SMASS SMEARINGS SPRING STEP_MAX STEP_SIZE SYMPREC SYSTEM TEBEG TEEND TIME
TSUBSYS VALUE_MAX VALUE_MIN VCUTOFF VDW_A1 VDW_A2 VDW_C6 VDW_CNRADIUS VDW_D
VDW_R0 VDW_RADIUS VDW_S6 VDW_S8 VDW_SR VOSKOWN WC WEIMIN ZVAL
""".split())


# The retry ladder keys on `when:`, deliberately NOT `on:`. YAML 1.1 treats a
# bare `on` as the boolean True, so `- on: timeout` parses as `{True: 'timeout'}`
# and the key vanishes. Found while writing the shipped recipe. The same trap
# catches `off`, `yes` and `no` -- which is also why INCAR booleans are written
# as `.TRUE.`/`.FALSE.` strings and why the writer accepts Python bools and
# converts them, since a user writing `LASPH: true` gets a Python bool here.
RETRY_KEY = "when"


class RecipeError(Exception):
    """A recipe that would produce inputs whose meaning is not what it says."""


@dataclass
class Kpoints:
    scheme: str = "reciprocal_density"
    value: Any = 64

    def as_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "value": self.value}


@dataclass
class RecipeStage:
    """One fully resolved DFT stage.  Nothing here defers to anything."""

    name: str
    incar: dict[str, Any] = field(default_factory=dict)
    kpoints: Kpoints = field(default_factory=Kpoints)
    resources: dict[str, Any] = field(default_factory=dict)
    retry: list[dict[str, Any]] = field(default_factory=list)

    def with_overrides(self, overrides: dict[str, Any]) -> "RecipeStage":
        return RecipeStage(
            name=self.name, incar={**self.incar, **overrides},
            kpoints=self.kpoints, resources=dict(self.resources),
            retry=list(self.retry),
        )


@dataclass
class Recipe:
    name: str
    stages: list[RecipeStage] = field(default_factory=list)
    source: Path | None = None

    def stage(self, name: str) -> RecipeStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise RecipeError(
            f"recipe {self.name!r} has no stage {name!r}; it has "
            f"{[s.name for s in self.stages]}"
        )

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]


def load_recipe(name_or_path: str) -> Recipe:
    """Load a shipped recipe by name, or any YAML file by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = RECIPE_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        shipped = sorted(p.stem for p in RECIPE_DIR.glob("*.yaml"))
        raise RecipeError(
            f"no recipe at {path}. Shipped recipes: {shipped}. "
            f"`dft.recipe` takes a shipped name or a path to a YAML file."
        )

    data = yaml.safe_load(path.read_text()) or {}
    return build_recipe(data, name=path.stem, source=path)


def build_recipe(data: dict[str, Any], *, name: str = "recipe",
                 source: Path | None = None) -> Recipe:
    """Resolve a recipe mapping into literal stages.

    `inherit:` is applied here, once, by copying. After this function there is
    no indirection left anywhere in the recipe -- which is what makes
    `csp dft --dry-run` able to print exactly what will land on disk.
    """
    raw_stages = data.get("stages")
    if not raw_stages:
        raise RecipeError(f"recipe {name!r} defines no stages")

    resolved: dict[str, RecipeStage] = {}
    order: list[RecipeStage] = []

    for raw in raw_stages:
        stage_name = raw.get("name")
        if not stage_name:
            raise RecipeError(f"recipe {name!r} has a stage with no 'name'")

        parent = raw.get("inherit")
        if parent is not None:
            if parent not in resolved:
                raise RecipeError(
                    f"stage {stage_name!r} inherits from {parent!r}, which is not "
                    f"defined above it. Stages inherit only from earlier stages, so "
                    f"that resolution is a single pass with no cycles possible."
                )
            base = resolved[parent]
            incar = {**base.incar, **(raw.get("incar") or {})}
            kpoints = _kpoints(raw.get("kpoints")) or base.kpoints
            resources = {**base.resources, **(raw.get("resources") or {})}
            retry = raw.get("retry") or list(base.retry)
        else:
            incar = dict(raw.get("incar") or {})
            kpoints = _kpoints(raw.get("kpoints")) or Kpoints()
            resources = dict(raw.get("resources") or {})
            retry = list(raw.get("retry") or [])

        stage = RecipeStage(name=stage_name, incar=incar, kpoints=kpoints,
                            resources=resources, retry=retry)
        resolved[stage_name] = stage
        order.append(stage)

    return Recipe(name=data.get("name", name), stages=order, source=source)


def _kpoints(raw: Any) -> Kpoints | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return Kpoints(scheme=raw.get("scheme", "reciprocal_density"),
                       value=raw.get("value", 64))
    raise RecipeError(f"kpoints must be a mapping, got {type(raw).__name__}")


def validate_recipe(recipe: Recipe) -> list[str]:
    """Refuse a recipe that would silently inherit a VASP default.

    Returns warnings; raises `RecipeError` on anything that would change the
    physics without saying so.
    """
    warnings: list[str] = []
    problems: list[str] = []

    for stage in recipe.stages:
        missing = [tag for tag in REQUIRED_TAGS
                   if tag not in stage.incar and tag not in COMPUTED_TAGS]
        for tag in missing:
            problems.append(f"stage '{stage.name}' does not set {tag}. "
                            f"{REQUIRED_TAGS[tag]}.")

        for tag in stage.incar:
            if tag.upper() not in KNOWN_TAGS and tag not in COMPUTED_TAGS:
                warnings.append(
                    f"stage '{stage.name}': {tag} is not a VASP tag this version "
                    f"recognises. It will be written to the INCAR as given -- "
                    f"there is no whitelist -- but check the spelling."
                )
        warnings.extend(check_retry_keys(Recipe(name=recipe.name, stages=[stage])))
        if tag_value(stage.incar, "NSW", 0) == 0 and tag_value(stage.incar, "IBRION", -1) > 0:
            warnings.append(
                f"stage '{stage.name}': IBRION > 0 with NSW = 0 does no ionic steps"
            )

    if problems:
        raise RecipeError(
            f"recipe {recipe.name!r} would inherit VASP defaults for tags that "
            f"change the physics:\n  " + "\n  ".join(problems)
        )
    return warnings


def check_retry_keys(recipe: Recipe) -> list[str]:
    """Catch a retry rule whose trigger silently became a YAML boolean."""
    notes = []
    for stage in recipe.stages:
        for rule in stage.retry:
            if RETRY_KEY not in rule:
                keys = sorted(str(k) for k in rule)
                notes.append(
                    f"stage '{stage.name}': a retry rule has no '{RETRY_KEY}:' key "
                    f"(it has {keys}). If you wrote 'on:', note that YAML parses a "
                    f"bare `on` as the boolean True, so the key disappears. Use "
                    f"'{RETRY_KEY}:'."
                )
    return notes


def tag_value(incar: dict[str, Any], tag: str, default: Any = None) -> Any:
    for key, value in incar.items():
        if key.upper() == tag.upper():
            return value
    return default
