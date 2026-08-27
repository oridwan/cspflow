"""Pydantic models for a cspflow campaign -- the single source of truth.

Two rules run through this file and are worth stating once:

1.  ``extra="forbid"`` everywhere.  A mistyped key is an error, not a silently
    ignored line.  The failure this prevents is a campaign that runs to
    completion having quietly ignored the setting you cared about.

2.  Physics values are not invented here.  Where the plan says a value must be
    explicit (``pick``, the INCAR tags, the reference functional), the field is
    required or the default is a deliberate, documented choice recorded in
    provenance -- never a convenience fallback.  See pipeline.md sec.9,
    "What 'fully robust' means here", gate 2.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Rare earths, used by the ``max_rare_earth`` guard (Stage 0.1) and by the
# 4f treatment in Stage 3d.  Sc and Y are excluded: they are group-3 metals with
# no f electrons, so the 4f machinery does not apply to them.
RARE_EARTHS: frozenset[str] = frozenset(
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
)


class Base(BaseModel):
    """Common config: reject unknown keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Stage 0 -- source
# --------------------------------------------------------------------------


class SourceMode(str, Enum):
    chemical_space = "chemical_space"
    composition_list = "composition_list"
    structure_list = "structure_list"


class NStructuresMode(str, Enum):
    fixed = "fixed"
    per_atom = "per_atom"


class NStructures(Base):
    """How many structures to generate for one (composition, Z) row."""

    mode: NStructuresMode = NStructuresMode.per_atom
    count: int | None = Field(None, gt=0, description="mode=fixed: structures per row")
    structures_per_atom: float | None = Field(
        2.0, gt=0, description="mode=per_atom: multiplied by the atom count"
    )

    @model_validator(mode="after")
    def _one_of(self) -> "NStructures":
        if self.mode is NStructuresMode.fixed and self.count is None:
            raise ValueError("n_structures.mode='fixed' requires 'count'")
        if self.mode is NStructuresMode.per_atom and self.structures_per_atom is None:
            raise ValueError("n_structures.mode='per_atom' requires 'structures_per_atom'")
        return self

    def target_for(self, n_atoms: int) -> int:
        if self.mode is NStructuresMode.fixed:
            return int(self.count)  # type: ignore[arg-type]
        return max(1, round(self.structures_per_atom * n_atoms))  # type: ignore[operator]


class ZRange(Base):
    """Formula units of the *reduced* formula to enumerate."""

    min: int = Field(1, ge=1)
    max: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "ZRange":
        if self.max < self.min:
            raise ValueError(f"z.max ({self.max}) < z.min ({self.min})")
        return self

    def values(self) -> list[int]:
        return list(range(self.min, self.max + 1))


class SourceDefaults(Base):
    """Inherited by modes 1 and 2; per-item overrides win."""

    z: ZRange = Field(default_factory=ZRange)
    max_atoms: int = Field(40, gt=0, description="hard cap on Z * atoms per formula unit")
    n_structures: NStructures = Field(default_factory=NStructures)
    n_structures_scope: Literal["per_z", "total"] = "per_z"


class ElementGroup(Base):
    """One group in a chemical space, e.g. ``{Fe, Co, Ni}`` at >= 0.75 fraction.

    ``pick`` is REQUIRED and has no default on purpose.  ``{Fe,Co,Ni}-{Y,Gd}``
    is ambiguous: one element per group gives 6 binary systems, allowing two
    from a group adds ternaries and a 5-10x larger sweep.  Left implicit this is
    a silent campaign-size multiplier, so the user states it.  See pipeline.md
    sec.0.1.
    """

    elements: list[str] = Field(..., min_length=1)
    pick: int | list[int] = Field(
        ..., description="how many elements to take from this group; int or list of arities"
    )
    min_fraction: float | None = Field(None, ge=0.0, le=1.0)
    max_fraction: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("elements")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            dupes = sorted({e for e in v if v.count(e) > 1})
            raise ValueError(f"duplicate elements in group: {dupes}")
        return v

    @field_validator("pick")
    @classmethod
    def _positive(cls, v: int | list[int]) -> int | list[int]:
        arities = [v] if isinstance(v, int) else v
        if not arities:
            raise ValueError("pick must not be empty")
        for a in arities:
            if a < 1:
                raise ValueError(f"pick must be >= 1, got {a}")
        return v

    @model_validator(mode="after")
    def _checks(self) -> "ElementGroup":
        if (
            self.min_fraction is not None
            and self.max_fraction is not None
            and self.min_fraction > self.max_fraction
        ):
            raise ValueError(
                f"min_fraction ({self.min_fraction}) > max_fraction ({self.max_fraction})"
            )
        for a in self.arities():
            if a > len(self.elements):
                raise ValueError(
                    f"pick={a} exceeds the {len(self.elements)} elements in the group"
                )
        return self

    def arities(self) -> list[int]:
        return [self.pick] if isinstance(self.pick, int) else list(self.pick)


class ChemicalSpace(Base):
    """Mode 1: element groups plus per-group ratio constraints."""

    groups: dict[str, ElementGroup] = Field(..., min_length=1)
    max_atoms_formula: int = Field(20, gt=0, description="cap on the reduced formula itself")
    max_rare_earth: int | None = Field(
        1,
        ge=0,
        description=(
            "max rare-earth species in the ASSEMBLED system.  Deliberately separate "
            "from `pick`, which is arity within one group: nothing forces the rare "
            "earths into their own group, so a user writing {Sm, Tb, Fe} with "
            "pick=[1,2] would otherwise generate Sm-Tb-Fe unasked.  null = no limit."
        ),
    )


class CompositionItem(Base):
    """One explicit formula in mode 2, with optional per-item overrides."""

    formula: str = Field(..., min_length=1)
    z: ZRange | None = None
    max_atoms: int | None = Field(None, gt=0)
    n_structures: NStructures | None = None

    @field_validator("z", mode="before")
    @classmethod
    def _z_shorthand(cls, v: Any) -> Any:
        """Accept ``z: [1, 2]`` as shorthand for ``z: {min: 1, max: 2}``."""
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"z as a list must be [min, max], got {list(v)}")
            return {"min": v[0], "max": v[1]}
        return v


class CompositionList(Base):
    """Mode 2: explicit formulas, still generated."""

    items: list[CompositionItem] = Field(default_factory=list)
    from_file: str | None = Field(
        None, description="CSV: formula[,z_min,z_max,n_structures]"
    )

    @model_validator(mode="after")
    def _something(self) -> "CompositionList":
        if not self.items and self.from_file is None:
            raise ValueError("composition_list needs either 'items' or 'from_file'")
        return self


class StructureList(Base):
    """Mode 3: structures in, NO generation.

    ``dedup`` defaults to ``warn`` deliberately.  For generated structures,
    dropping duplicates is the entire point of Stage 2.  For a curated input
    list it is a hazard: two seeds you supplied on purpose -- a relaxed and an
    unrelaxed copy of one prototype, say -- would be silently merged and you
    would never learn which survived.  See pipeline.md sec.0.3.
    """

    paths: list[str] = Field(..., min_length=1, description="POSCAR/CIF paths or globs")
    relax: bool = Field(True, description="MLIP-relax the seed before DFT")
    dedup: Literal["warn", "drop"] = "warn"
    max_atoms: int | None = Field(None, gt=0, description="defaults to source.defaults.max_atoms")


class Source(Base):
    """One entry in the ``source:`` list.

    ``name`` is required when several sources are present: without it, "did this
    candidate come from the sweep or from my seed list?" is unanswerable once the
    rows are pooled, and the control group stops being a control group.
    """

    mode: SourceMode
    name: str = Field("default", min_length=1)
    defaults: SourceDefaults = Field(default_factory=SourceDefaults)
    chemical_space: ChemicalSpace | None = None
    composition_list: CompositionList | None = None
    structure_list: StructureList | None = None

    @model_validator(mode="after")
    def _block_matches_mode(self) -> "Source":
        blocks = {
            SourceMode.chemical_space: self.chemical_space,
            SourceMode.composition_list: self.composition_list,
            SourceMode.structure_list: self.structure_list,
        }
        if blocks[self.mode] is None:
            raise ValueError(f"source.mode='{self.mode.value}' requires a '{self.mode.value}:' block")
        extra = [m.value for m, b in blocks.items() if b is not None and m is not self.mode]
        if extra:
            raise ValueError(
                f"source.mode='{self.mode.value}' but these blocks are also set: {extra}. "
                "Use a separate list entry for each source rather than one entry with several blocks."
            )
        return self

    @property
    def entry_stage(self) -> Literal["generate", "screen"]:
        """Where this source's rows enter the funnel.

        Mode 3 is an entry point, not a branch: it emits structures directly and
        skips generation, but everything from Stage 2 on is composition-agnostic
        and needs no change.
        """
        return "screen" if self.mode is SourceMode.structure_list else "generate"


# --------------------------------------------------------------------------
# Stages 1-2 -- generate, screen
# --------------------------------------------------------------------------


class Resources(Base):
    """Scheduler ask for one stage.  Resolved against the machine profile."""

    role: str = Field("cpu", description="machine-profile partition role: cpu | gpu | ...")
    ntasks: int | None = Field(None, gt=0)
    cpus_per_task: int | None = Field(None, gt=0)
    gpus: int | None = Field(None, ge=0)
    mem: str | None = None
    time: str = "24:00:00"

    @field_validator("time")
    @classmethod
    def _walltime(cls, v: str) -> str:
        # SLURM accepts several forms; we require D-HH:MM:SS or HH:MM:SS so that
        # the driver can compare walltimes against QOS limits numerically.
        import re

        if not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", v):
            raise ValueError(f"time must be [D-]HH:MM:SS, got {v!r}")
        return v


class MatterGen(Base):
    model: str = Field(..., description="checkpoint directory")
    mode: Literal["csp", "unconditional"] = "csp"
    max_batch_size: int = Field(100, gt=0)
    timeout_per_batch: int = Field(1800, gt=0, description="seconds")


class Generate(Base):
    engine: Literal["mattergen"] = "mattergen"
    mattergen: MatterGen | None = None
    resources: Resources = Field(default_factory=lambda: Resources(role="gpu", gpus=1))

    @model_validator(mode="after")
    def _engine_block(self) -> "Generate":
        if self.engine == "mattergen" and self.mattergen is None:
            raise ValueError("generate.engine='mattergen' requires a 'mattergen:' block")
        return self


class MatterSim(Base):
    model: str = "MatterSim-v1.0.0-5M.pth"
    fmax: float = Field(0.01, gt=0, description="eV/A force convergence")
    max_steps: int = Field(500, gt=0)
    batch_size: int = Field(32, gt=0)


class StructureMatcherCfg(Base):
    ltol: float = Field(0.2, gt=0)
    stol: float = Field(0.2, gt=0)
    angle_tol: float = Field(5.0, gt=0)


class Dedup(Base):
    matcher: StructureMatcherCfg = Field(default_factory=StructureMatcherCfg)


class Screen(Base):
    mlip: Literal["mattersim", "mace", "uma"] = "mattersim"
    mattersim: MatterSim = Field(default_factory=MatterSim)
    dedup: Dedup = Field(default_factory=Dedup)
    resources: Resources = Field(default_factory=lambda: Resources(role="gpu", gpus=1))


# --------------------------------------------------------------------------
# Stage 3 -- reference
# --------------------------------------------------------------------------


class ThermoType(str, Enum):
    """MP's own functional label.

    This must be pinned.  MP's /materials/summary/ endpoint returns
    ``uncorrected_energy_per_atom`` from whichever functional it prefers per
    material, with NO field in the response saying which -- SmFe2 comes back as
    -7.1966 (GGA) or -19.4095 (r2SCAN) depending on the material.  Mixing them
    inside one chemsys puts a 5-12 eV/atom discontinuity into the hull.  See
    pipeline.md sec.3.1.
    """

    GGA_GGA_U = "GGA_GGA+U"
    R2SCAN = "R2SCAN"
    GGA_GGA_U_R2SCAN = "GGA_GGA+U_R2SCAN"


class Reference(Base):
    functionals: list[Literal["GGA", "GGA+U"]] = Field(default_factory=lambda: ["GGA"])
    thermo_type: ThermoType = Field(
        ThermoType.GGA_GGA_U,
        description="PINNED. A reference set containing more than one is refused.",
    )
    energy_scale: Literal["raw", "mp_corrected"] = "raw"
    mode: Literal["mp_energies", "recompute"] = "recompute"
    prescreen_mode: Literal["mp_energies", "recompute"] = "mp_energies"
    prescreen_hull_max: float = Field(0.20, gt=0, description="eV/atom, widened for Phase A")
    snapshot: bool = Field(
        True,
        description=(
            "Freeze the MP query.  Without it, e_above_hull moves with no change "
            "to any of our inputs, which is indistinguishable from a bug."
        ),
    )
    snapshot_id: str = Field("auto", description="'auto' = new, stamped with date + MP release")
    cache: str = "$CSPFLOW_CACHE/mp"
    recompute_cache: str = "$CSPFLOW_CACHE/reference"
    relax_with_mlip: bool = True


# --------------------------------------------------------------------------
# Stage 4 -- calibrate
# --------------------------------------------------------------------------


class OnFail(str, Enum):
    off = "off"
    warn = "warn"
    block = "block"


def _on_fail(value: Any) -> Any:
    """Undo YAML 1.1's boolean coercion of `off`.

    PyYAML implements YAML 1.1, where bare `off`, `no` and `false` are all the
    boolean False -- so `on_fail: off`, written exactly as the documentation
    says, reaches pydantic as `False` and is rejected with a message about an
    enum that mentions neither YAML nor booleans.

    The same trap bit the DFT retry ladder, where `on:` became `True:` and a
    retry rule silently had no condition (D065). There it was caught by
    refusing the key; here the value is simply translated, because `off` is a
    legitimate thing to write and the user is not wrong.
    """
    if value is False:
        return "off"
    if value is True:
        raise ValueError(
            "on_fail must be 'off', 'warn' or 'block'. YAML 1.1 reads a bare "
            "`on` as the boolean True; quote it if you meant a word.")
    return value


class CalibrateMPThresholds(Base):
    mae_e_per_atom: float = Field(0.05, gt=0, description="single-point MLIP on MP geometry")
    spearman_min: float = Field(0.90, ge=-1.0, le=1.0)
    max_volume_drift: float = Field(
        0.05, gt=0, description="geometry error, kept as its own number, never folded into the MAE"
    )


class CalibratePilotThresholds(Base):
    mae_e_per_atom: float = Field(0.05, gt=0)
    mae_e_hull: float = Field(0.05, gt=0)
    spearman_min: float = Field(0.90, ge=-1.0, le=1.0)


class CalibrateMP(Base):
    """4a: free.  MP's own DFT is the yardstick, at FIXED geometry.

    Single-point, never relaxed-vs-relaxed: comparing E_MLIP(x_MLIP) against
    E_DFT(x_MP) folds energy error and geometry error into one inseparable MAE,
    and the two have opposite consequences.  A uniform energy offset largely
    cancels in a hull; a volume bias does not and is fatal for screening.
    """

    on_fail: OnFail = OnFail.warn
    thresholds: CalibrateMPThresholds = Field(default_factory=CalibrateMPThresholds)

    _fix_on_fail = field_validator("on_fail", mode="before")(_on_fail)


class CalibratePilot(Base):
    """4b: costs pilot DFT.  OUR DFT is the yardstick.  This is the real gate.

    MP phases are near in-distribution for MP-trained universal MLIPs, so 4a is
    largely a self-consistency check.  The generated structures are the
    out-of-distribution set.
    """

    on_fail: OnFail = OnFail.block
    pilot_n: int = Field(40, gt=0, description="screened candidates given pilot DFT")
    thresholds: CalibratePilotThresholds = Field(default_factory=CalibratePilotThresholds)

    _fix_on_fail = field_validator("on_fail", mode="before")(_on_fail)


class Calibrate(Base):
    mp: CalibrateMP = Field(default_factory=CalibrateMP)
    pilot: CalibratePilot = Field(default_factory=CalibratePilot)


# --------------------------------------------------------------------------
# Stage 5 -- filter
# --------------------------------------------------------------------------


class SpacegroupFilter(Base):
    min_number: int = Field(1, ge=1, le=230)


class Filter(Base):
    e_above_hull_max: float = Field(0.10, gt=0, description="eV/atom")
    e_above_hull_max_source: Literal["literal", "calibrated"] = Field(
        "calibrated",
        description=(
            "'calibrated' rescales the THRESHOLD by the fitted alpha/beta from "
            "Stage 4a rather than rewriting any stored energy.  If the MLIP "
            "compresses hull distances by 1.4x, screening at 0.10 in MLIP units "
            "silently discards candidates sitting at 0.10 in DFT units."
        ),
    )
    max_per_composition: int = Field(5, gt=0)
    spacegroup: SpacegroupFilter = Field(default_factory=SpacegroupFilter)


# --------------------------------------------------------------------------
# Stage 6 -- dft
# --------------------------------------------------------------------------


class PotcarCfg(Base):
    """Both POTCAR trees stay available; the choice is the user's per campaign.

    Whichever is chosen, every resolved POTCAR is pinned by md5_header_hash in
    provenance, so the two trees can never be mixed inside one hull -- that is a
    hard refusal, not a warning.  See pipeline.md sec.3c.5.
    """

    tree: Literal["VASP6.4", "VASP5.2"] = "VASP6.4"
    functional: str = Field("PBE_64", description="pymatgen functional label")
    overrides: dict[str, str] = Field(
        default_factory=dict, description="element -> POTCAR symbol, e.g. {Gd: Gd_3}"
    )


class FTreatment(str, Enum):
    """Frozen 4f (RE_3 across the whole series) or f in valence.

    Both are supported and switchable per campaign.  ``frozen`` ships as the
    default for the screening funnel: it is faster, converges, and introduces no
    free U parameter.  ``valence`` is what you need for real moments and any
    future MAE work.  The one thing that is forbidden is mixing them within one
    campaign -- which is exactly what MPRelaxSet does across the RE series.
    """

    frozen = "frozen"
    valence = "valence"


class RareEarth(Base):
    f_treatment: FTreatment = FTreatment.frozen
    magnetic_order: Literal["ferri", "ferro", "none"] = "ferri"
    reconstruct_ms: bool = Field(
        True, description="add Hund's-rule 4f moments back in Stage 7, reported separately"
    )


class Magnetism(Base):
    mode: Literal["none", "pymatgen", "table", "ferrimagnetic_retm"] = "ferrimagnetic_retm"
    strict: bool = Field(
        True, description="fail if any site would take a default MAGMOM"
    )
    table: dict[str, float] = Field(default_factory=dict, description="element -> initial moment")
    site_overrides: dict[str, float] = Field(default_factory=dict)


class Ldau(Base):
    enabled: bool = False
    u: dict[str, float] = Field(default_factory=dict, description="element -> U (eV)")
    j: dict[str, float] = Field(default_factory=dict)
    ldau_type: int = Field(2, ge=1, le=4)


class Select(Base):
    """Which screened candidates get DFT, and how much they may cost."""

    rank_by: str = "e_above_hull_mlip"
    max_per_composition: int = Field(3, gt=0)
    max_total: int = Field(1500, gt=0)
    budget_core_hours: int = Field(200_000, gt=0)


class Dft(Base):
    recipe: str = Field("magnets", description="name in dft/recipes/, or a path to a YAML file")
    potcar: PotcarCfg = Field(default_factory=PotcarCfg)
    rare_earth: RareEarth = Field(default_factory=RareEarth)
    magnetism: Magnetism = Field(default_factory=Magnetism)
    ldau: Ldau = Field(default_factory=Ldau)
    nbands: Literal["auto"] | int = "auto"
    incar_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Campaign-level INCAR overrides applied on top of the recipe, for every "
            "stage.  Free-form: there is no whitelist.  Unknown tags are written as "
            "given; unrecognised ones warn but are never silently dropped."
        ),
    )
    max_in_flight: int = Field(200, gt=0, description="jobs submitted at once (QOS-aware)")
    max_concurrent_tasks: int = Field(48, gt=0, description="the --array %N throttle")
    select: Select = Field(default_factory=Select)


# --------------------------------------------------------------------------
# Stage 7 -- analyze
# --------------------------------------------------------------------------


class Analyze(Base):
    properties: list[str] = Field(
        default_factory=lambda: ["m_dft_raw", "m_s_reconstructed", "volume", "spacegroup"]
    )
    report: Literal["html", "none"] = "html"

    @model_validator(mode="after")
    def _never_merge_moments(self) -> "Analyze":
        """m_dft_raw and m_s_reconstructed are reported side by side, never merged.

        The reconstructed value is a model layered on DFT, not a computed
        result.  Collapsing them into one column is how a Hund's-rule estimate
        ends up quoted as a DFT number.
        """
        props = set(self.properties)
        if "m_s" in props:
            raise ValueError(
                "'m_s' is ambiguous: use 'm_dft_raw' (computed) and/or "
                "'m_s_reconstructed' (Hund's-rule model). They are never merged."
            )
        return self


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


class Campaign(Base):
    """A whole campaign.  This is the only file a user edits."""

    name: str = Field(..., min_length=1)
    machine: str = Field(..., min_length=1, description="machine profile name or path")
    workdir: str = Field(..., min_length=1)
    archive: str | None = Field(
        None,
        description=(
            "db + report mirrored here at every stage boundary, so a scratch purge "
            "costs compute but never provenance"
        ),
    )

    source: list[Source] = Field(..., min_length=1)
    generate: Generate | None = None
    screen: Screen = Field(default_factory=Screen)
    reference: Reference = Field(default_factory=Reference)
    calibrate: Calibrate = Field(default_factory=Calibrate)
    filter: Filter = Field(default_factory=Filter)
    dft: Dft = Field(default_factory=Dft)
    analyze: Analyze = Field(default_factory=Analyze)

    @field_validator("source", mode="before")
    @classmethod
    def _accept_bare_mapping(cls, v: Any) -> Any:
        """A single source may be written as a bare mapping instead of a 1-list."""
        if isinstance(v, dict):
            return [v]
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "Campaign":
        names = [s.name for s in self.source]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"duplicate source names {dupes}. Every source needs a distinct 'name:' -- "
                "it is stamped on each row it emits, and without it a pooled campaign "
                "cannot say which source a candidate came from."
            )
        if len(self.source) > 1 and any(s.name == "default" for s in self.source):
            raise ValueError(
                "with more than one source, every entry needs an explicit 'name:' "
                "(one is still using the placeholder 'default')"
            )
        if self.needs_generation and self.generate is None:
            modes = sorted({s.mode.value for s in self.source if s.entry_stage == "generate"})
            raise ValueError(
                f"source mode(s) {modes} generate structures, so a 'generate:' block is required"
            )
        return self

    @property
    def needs_generation(self) -> bool:
        return any(s.entry_stage == "generate" for s in self.source)

    @property
    def rare_earth_elements(self) -> set[str]:
        """Rare earths mentioned anywhere in the campaign's chemical spaces."""
        found: set[str] = set()
        for s in self.source:
            if s.chemical_space:
                for g in s.chemical_space.groups.values():
                    found |= RARE_EARTHS & set(g.elements)
        return found


# --------------------------------------------------------------------------
# Machine profile -- the portability answer (pipeline.md sec.6.5)
# --------------------------------------------------------------------------


class Partition(Base):
    name: str = Field(..., description="site partition name(s), comma-separated for SLURM")
    qos: str | None = None
    account: str | None = None
    constraint: str | None = None
    exclude: str | None = None


class PartitionLimits(Base):
    max_submit: int | None = Field(None, gt=0)
    max_cpus: int | None = Field(None, gt=0)
    max_gpus: int | None = Field(None, gt=0)
    max_walltime: str | None = None


class MachineDefaults(Base):
    nodes: int = Field(1, gt=0)
    ntasks: int = Field(16, gt=0)
    cpus_per_task: int = Field(1, gt=0)
    mem: str = "32G"


class Codes(Base):
    vasp_std: str | None = None
    vasp_gam: str | None = None
    vasp_ncl: str | None = None
    mpi_launcher: str = "srun --mpi=pmi2"


class Machine(Base):
    """A site profile.  Everything site-specific lives here and nowhere else."""

    scheduler: Literal["slurm", "local"] = "slurm"
    partitions: dict[str, Partition] = Field(default_factory=dict)
    defaults: MachineDefaults = Field(default_factory=MachineDefaults)
    limits: dict[str, PartitionLimits] = Field(default_factory=dict)
    modules: dict[str, list[str]] = Field(default_factory=dict)
    env: dict[str, str | int] = Field(default_factory=dict)
    codes: Codes = Field(default_factory=Codes)
    potcar_root: str | None = Field(
        None, description="pymatgen-layout tree (PMG_VASP_PSP_DIR)"
    )
    potcar_dirs: dict[str, str] = Field(
        default_factory=dict, description="functional label -> pymatgen directory name"
    )
    potcar_trees: dict[str, str] = Field(
        default_factory=dict, description="tree label (VASP6.4/VASP5.2) -> functional label"
    )
    conda: dict[str, str] = Field(default_factory=dict, description="role -> env name")
    scratch: str = "/scratch/$USER"

    def partition_for(self, role: str) -> Partition:
        if role not in self.partitions:
            known = sorted(self.partitions) or ["<none defined>"]
            raise KeyError(
                f"machine profile has no partition for role {role!r}; known roles: {known}"
            )
        return self.partitions[role]
