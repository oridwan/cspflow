"""Preflight checks.

`csp doctor` exists so that everything which can be known before a job is
submitted is known before a job is submitted.  Its hard failures are the ones
that would otherwise surface as a wrong number rather than an error: an
unresolvable POTCAR, a functional label that misdescribes what is on disk, two
4f conventions in one campaign.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config.loader import ResolvedConfig
from .config.schema import RARE_EARTHS, Machine
from .dft.vasp import potcar as pc

Status = Literal["ok", "warn", "fail", "fixed", "skip"]

_MARK = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "fixed": "FIX ", "skip": "--  "}


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    rows: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"[{_MARK[self.status]}] {self.name}"
        if self.detail:
            head += f": {self.detail}"
        return "\n".join([head, *(f"        {r}" for r in self.rows)])


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def render(self) -> str:
        body = "\n".join(c.render() for c in self.checks)
        n_fail = sum(c.status == "fail" for c in self.checks)
        n_warn = sum(c.status == "warn" for c in self.checks)
        verdict = (
            f"\n{n_fail} failure(s), {n_warn} warning(s)."
            if n_fail or n_warn
            else "\nAll checks passed."
        )
        return body + "\n" + verdict


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_potcar_layout(machine: Machine, *, fix: bool = False) -> Check:
    """The symlink tree that makes a functional label mean what it says.

    Without it, `functional: PBE_64` fails and `PBE_54` succeeds by flat-layout
    fallback -- so the recorded provenance says PBE_54 for what are in fact
    VASP 6.4 potentials.
    """
    sources = {
        "PBE_64": "/projects/mmi/Ridwan/potcarFiles/VASP6.4/potpaw_PBE",
        "PBE_52": "/projects/mmi/Ridwan/potcarFiles/VASP5.2/potpaw_PBE",
    }
    sources = {k: v for k, v in sources.items() if k in machine.potcar_dirs}
    if not sources:
        return Check("POTCAR layout", "skip", "machine profile defines no potcar_dirs")
    actions = pc.ensure_pmg_layout(machine, sources, create=fix)
    worst: Status = "ok"
    for status, _ in actions:
        if status == "fail":
            worst = "fail"
        elif status in ("warn", "fixed") and worst == "ok":
            worst = status  # type: ignore[assignment]
    detail = {
        "ok": "labels match what is on disk",
        "fixed": "symlink tree created",
        "warn": "see below",
        "fail": "run `csp doctor --fix` to create the symlink tree",
    }[worst]
    return Check("POTCAR layout", worst, detail,
                 [f"{_MARK[s]} {m}" for s, m in actions])


def check_potcars(cfg: ResolvedConfig, elements: list[str]) -> list[Check]:
    """Resolve, hash and sanity-check every POTCAR the campaign needs."""
    dft = cfg.campaign.dft
    try:
        infos, errors = pc.resolve_all(
            elements, cfg.machine, tree=dft.potcar.tree,
            f_treatment=dft.rare_earth.f_treatment,
            overrides=dft.potcar.overrides,
        )
    except pc.PotcarError as exc:
        return [Check("POTCAR resolution", "fail", str(exc))]

    rows = [f"{'element':<8} {'symbol':<8} {'TITEL':<26} {'ZVAL':>6} {'ENMAX':>8}  hash"]
    for i in infos:
        rows.append(
            f"{i.element:<8} {i.symbol:<8} {i.titel:<26} {i.zval:6.1f} {i.enmax:8.3f}  {i.short_hash}"
        )
    for err in errors:
        rows.append(f"FAIL {err}")

    checks = [
        Check(
            f"POTCAR resolution ({dft.potcar.tree}, f_treatment={dft.rare_earth.f_treatment.value})",
            "fail" if errors else "ok",
            f"{len(errors)} unresolved" if errors else f"{len(infos)} resolved",
            rows,
        )
    ]

    # A vacuous pass is worse than no answer: if nothing resolved, or there is
    # only one rare earth, there is no convention to be consistent about.
    rare = [i for i in infos if i.element in RARE_EARTHS]
    if not infos:
        checks.append(Check("4f convention", "skip", "no POTCARs resolved"))
    elif len(rare) < 2:
        checks.append(Check("4f convention", "skip",
                            f"{len(rare)} rare earth(s) in this campaign -- nothing to compare"))
    else:
        try:
            pc.assert_one_f_convention(infos)
            checks.append(Check(
                "4f convention", "ok",
                f"one convention across {len(rare)} rare earths: "
                + ", ".join(f"{i.element}({i.symbol}, ZVAL {i.zval:g})" for i in rare)))
        except pc.PotcarError as exc:
            checks.append(Check("4f convention", "fail", str(exc)))

    if infos:
        default_encut = pc.max_enmax(infos)
        encut = _campaign_encut(cfg)
        rows = [
            f"VASP would default ENCUT to max(ENMAX) = {default_encut:.3f} eV for THIS "
            f"composition set",
            "that default moves with composition, so a hull built on it compares "
            "incomparable numbers",
        ]
        if encut is None:
            checks.append(Check("ENCUT", "warn",
                                "no ENCUT in the resolved recipe -- it must be explicit", rows))
        elif encut < default_encut:
            checks.append(Check("ENCUT", "warn",
                                f"ENCUT={encut:g} eV is below max(ENMAX)={default_encut:.1f} eV",
                                rows))
        else:
            checks.append(Check("ENCUT", "ok", f"explicit at {encut:g} eV "
                                               f"(>= max ENMAX {default_encut:.1f} eV)"))
    return checks


def _campaign_encut(cfg: ResolvedConfig) -> float | None:
    value = cfg.campaign.dft.incar_overrides.get("ENCUT")
    return float(value) if value is not None else None


def check_vasp(machine: Machine) -> Check:
    path = machine.codes.vasp_std
    if not path:
        return Check("VASP binary", "skip", "no vasp_std in machine profile")
    resolved = shutil.which(path) or (path if Path(path).is_file() else None)
    if resolved is None:
        return Check("VASP binary", "fail", f"{path} not found")
    if not os.access(resolved, os.X_OK):
        return Check("VASP binary", "fail", f"{resolved} is not executable")
    return Check("VASP binary", "ok", resolved)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_scheduler(machine: Machine) -> Check:
    if machine.scheduler == "local":
        return Check("scheduler", "ok", "local (no queue)")
    if shutil.which("sinfo") is None:
        return Check("scheduler", "warn", "slurm configured but sinfo not on PATH")

    out = _run(["sinfo", "-h", "-o", "%P|%a|%l|%D"])
    if out is None:
        return Check("scheduler", "warn", "sinfo failed")
    live = {}
    for line in out.strip().splitlines():
        name, avail, walltime, nodes = (line.split("|") + ["", "", ""])[:4]
        live[name.rstrip("*")] = (avail, walltime, nodes)

    rows, worst = [], "ok"
    for role, part in machine.partitions.items():
        wanted = [p for p in part.name.split(",") if p]
        if not wanted:
            rows.append(f"{role:<10} (site default)")
            continue
        missing = [p for p in wanted if p not in live]
        if missing:
            rows.append(f"{role:<10} {part.name}  MISSING: {missing}")
            worst = "fail"
        else:
            info = ", ".join(
                f"{p}[{live[p][0]}, {live[p][2]} nodes, max {live[p][1]}]" for p in wanted
            )
            rows.append(f"{role:<10} {info}")
    return Check("partitions", worst, f"{len(live)} partitions visible", rows)


def check_qos_limits() -> Check:
    """Live submit limits, so the driver throttles against reality."""
    user = os.environ.get("USER", "")
    out = _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                "format=Account,Partition,QOS,MaxSubmitJobs,MaxJobs,GrpTRES"])
    if out is None:
        return Check("QOS limits", "warn",
                     "sacctmgr unavailable; the driver will fall back to profile limits")
    rows = [line for line in out.strip().splitlines() if line.strip()]
    if not rows:
        return Check("QOS limits", "warn", f"no associations reported for user {user!r}")
    return Check("QOS limits", "ok", f"{len(rows)} association(s)",
                 ["Account|Partition|QOS|MaxSubmit|MaxJobs|GrpTRES", *rows[:12]])


def check_optional_deps(cfg: ResolvedConfig) -> Check:
    """Engines are only needed if the campaign actually uses them."""
    wanted: dict[str, str] = {}
    if cfg.campaign.needs_generation and cfg.campaign.generate:
        wanted[cfg.campaign.generate.engine] = "generation"
    wanted[cfg.campaign.screen.mlip] = "screening"
    wanted["pymatgen"] = "structure handling"

    rows, worst = [], "ok"
    for mod, why in sorted(wanted.items()):
        found = importlib.util.find_spec(mod) is not None
        rows.append(f"{mod:<12} {'present' if found else 'MISSING':<8} ({why})")
        if not found:
            worst = "warn"
    return Check("engines", worst,
                 "missing engines block only the stages that use them", rows)


def check_paths(cfg: ResolvedConfig) -> Check:
    rows, worst = [], "ok"
    workdir = Path(cfg.campaign.workdir)
    parent = workdir if workdir.exists() else workdir.parent
    if not parent.exists():
        rows.append(f"workdir {workdir} -- parent {parent} does not exist")
        worst = "fail"
    elif not os.access(parent, os.W_OK):
        rows.append(f"workdir {workdir} -- {parent} is not writable")
        worst = "fail"
    else:
        rows.append(f"workdir {workdir} (writable)")
    if cfg.campaign.archive:
        arch = Path(cfg.campaign.archive)
        ap = arch if arch.exists() else arch.parent
        ok = ap.exists() and os.access(ap, os.W_OK)
        rows.append(f"archive {arch} ({'writable' if ok else 'NOT writable'})")
        if not ok:
            worst = "warn" if worst == "ok" else worst
    return Check("paths", worst, "", rows)


# --------------------------------------------------------------------------


def run(cfg: ResolvedConfig, *, elements: list[str] | None = None, fix: bool = False) -> Report:
    report = Report()
    report.add(Check("config", "ok",
                     f"{cfg.campaign.name}  hash {cfg.short_hash}  machine {cfg.machine_path.name}"))
    report.add(check_paths(cfg))
    report.add(check_potcar_layout(cfg.machine, fix=fix))
    if elements:
        report.add(*check_potcars(cfg, elements))
    else:
        report.add(Check("POTCAR resolution", "skip",
                         "no elements yet -- run `csp source` first, or pass --elements"))
    report.add(check_vasp(cfg.machine))
    report.add(check_scheduler(cfg.machine))
    report.add(check_qos_limits())
    report.add(check_optional_deps(cfg))
    return report
