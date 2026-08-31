"""The `csp` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from . import __version__, doctor as doctor_mod
from .config.loader import ConfigError, load_campaign, resolve_machine_path
from .config.schema import Campaign
from .db.store import Store, StoreError
from .ingest import IngestError, ingest_campaign
from .legacy import REGISTRY as LEGACY_REGISTRY, LegacyError, adopt as adopt_legacy
from .driver import STAGE_ORDER, Driver, DriverError, DriverOptions
from .scheduler import for_machine
from .source import SourceError, expand_all, write_plan
from .stages import IMPLEMENTED, PLANNED, build_registry
from .worker import WorkerError, run_generate_task, run_screen_task
from . import templates

app = typer.Typer(
    name="csp",
    help="High-throughput crystal structure prediction and first-principles discovery.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

DEFAULT_CAMPAIGN = "campaign.yaml"

CampaignOpt = Annotated[
    Path, typer.Option("--campaign", "-c", help="campaign YAML file")
]
SetOpt = Annotated[
    Optional[list[str]], typer.Option("--set", "-s", help="override, e.g. -s filter.e_above_hull_max=0.2")
]


def _die(message: str) -> None:
    typer.secho(str(message), fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _find_campaign(path: Path) -> Path:
    """Locate the campaign file, walking up from the working directory.

    A campaign is a folder, so every command should work from inside it or from
    any folder beneath it -- the way git works anywhere in a checkout. Only the
    unqualified default name is searched for; if the user named a file, that is
    the file, and a missing one is an error rather than a hunt.
    """
    if path.is_file() or path.is_absolute() or str(path) != DEFAULT_CAMPAIGN:
        return path
    here = Path.cwd()
    for folder in here.parents:
        candidate = folder / DEFAULT_CAMPAIGN
        if candidate.is_file():
            typer.secho(f"# campaign: {candidate}", fg=typer.colors.BLUE, err=True)
            return candidate
    return path


def _load(campaign: Path, sets: list[str] | None, machine: str | None = None):
    try:
        return load_campaign(_find_campaign(campaign), sets=sets, machine=machine)
    except ConfigError as exc:
        _die(str(exc))


def _link_results(campaign_file: Path, workdir: Path) -> None:
    """Put a `results` symlink beside campaign.yaml pointing at the workdir.

    Output lives on scratch because it gets large, which normally means the
    config and the thing it produced are in two unrelated corners of the
    filesystem. One symlink keeps them one `cd` apart.
    """
    link = _find_campaign(campaign_file).resolve().parent / "results"
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(workdir, target_is_directory=True)
    except OSError:
        pass          # a read-only or exotic filesystem is not a reason to stop


def _db_path(cfg) -> Path:
    return cfg.campaign_db


# --------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"cspflow {__version__}")


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="campaign name")],
    directory: Annotated[Optional[Path], typer.Option("--dir", "-d", help="where to create it (default: ./<name>)")] = None,
    machine: Annotated[str, typer.Option("--machine", "-m", help="shipped profile to copy: orion, generic_slurm, local")] = "orion",
    recipe: Annotated[str, typer.Option("--recipe", help="shipped DFT recipe to copy")] = "magnets",
    here: Annotated[bool, typer.Option("--here", help="use the current folder instead of creating one")] = False,
    minimal: Annotated[bool, typer.Option("--minimal", help="campaign.yaml only, referring to the shipped profile and recipe")] = False,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="write just the campaign file, at this path")] = None,
    force: Annotated[bool, typer.Option("--force", help="overwrite existing files")] = False,
) -> None:
    """Create a campaign folder with every knob in it.

    A campaign is a folder, not a file. Alongside campaign.yaml this puts your
    own copy of the machine profile and the DFT recipe -- the two things that
    used to be buried in site-packages where they could be neither found nor
    edited -- plus an inputs/ folder for your own structures. Nothing here is
    read-only and nothing is hidden: `--minimal` opts back out to a single file
    that refers to the shipped profile and recipe by name.
    """
    from .dft.recipe import RECIPE_DIR

    if out is not None:                     # single-file mode
        if out.exists() and not force:
            _die(f"{out} already exists (use --force to overwrite)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(templates.campaign_yaml(name=name, machine=machine,
                                               recipe=recipe, minimal=minimal))
        typer.echo(f"wrote {out}")
        typer.echo("Next: edit it, then run `csp doctor`.")
        return

    root = Path.cwd() if here else (directory or Path(name))
    written: list[Path] = []

    def _write(rel: str, text: str) -> None:
        path = root / rel
        if path.exists() and not force:
            _die(f"{path} already exists (use --force to overwrite)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)

    if minimal:
        _write(DEFAULT_CAMPAIGN, templates.campaign_yaml(
            name=name, machine=machine, recipe=recipe, minimal=True))
    else:
        try:
            machine_src = resolve_machine_path(machine)
        except ConfigError as exc:
            _die(str(exc))
        recipe_src = RECIPE_DIR / f"{recipe}.yaml"
        if not recipe_src.is_file():
            shipped = sorted(f.stem for f in RECIPE_DIR.glob("*.yaml"))
            _die(f"unknown recipe {recipe!r}; shipped: {shipped}")

        _write(DEFAULT_CAMPAIGN, templates.campaign_yaml(
            name=name, machine="machine.yaml", recipe="recipe.yaml"))
        _write("machine.yaml", templates.machine_copy(machine_src, name=name))
        _write("recipe.yaml", templates.recipe_copy(recipe_src, name=name))
        _write("inputs/README.md", templates.inputs_readme())
        _write("README.md", templates.workspace_readme(name=name))

    typer.secho(f"\ncampaign {name} in {root}/", fg=typer.colors.GREEN, bold=True)
    for path in written:
        rel = path.relative_to(root)
        typer.echo(f"  {str(rel):<18} {_BLURB.get(str(rel), '')}")
    typer.echo("\nNext:")
    if not here:
        typer.echo(f"  cd {root}")
    typer.echo("  $EDITOR campaign.yaml     # elements, cutoffs, how many structures")
    typer.echo("  csp doctor                # check the machine before submitting anything")
    typer.echo("  csp source --dry-run      # what would be searched")


_BLURB = {
    DEFAULT_CAMPAIGN: "what to search, and how hard",
    "machine.yaml": "partitions, walltime, modules, VASP, POTCARs",
    "recipe.yaml": "the DFT ladder: INCAR, k-points, resources",
    "inputs/README.md": "your own structures and composition lists go here",
    "README.md": "what to edit, what to run",
}


@config_app.command("show")
def config_show(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    origins: Annotated[bool, typer.Option("--origins", help="show which layer supplied each value")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="emit JSON")] = False,
) -> None:
    """Print the fully resolved configuration."""
    cfg = _load(campaign, set_)
    payload = cfg.campaign.model_dump(mode="json")
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    typer.echo(f"# machine:     {cfg.machine_path}")
    typer.echo(f"# config_hash: {cfg.config_hash}")
    if origins:
        typer.echo("\n# where each value came from:")
        for path in sorted(cfg.origins):
            typer.echo(f"#   {path:<48} {cfg.origins[path]}")


@config_app.command("defaults")
def config_defaults() -> None:
    """Print the schema's default values.

    Derived from the pydantic models rather than a checked-in file, so the
    defaults shown here cannot drift from the defaults actually applied.
    """
    skeleton = {
        "name": "<required>",
        "machine": "<required>",
        "workdir": "<required>",
        "source": "<required: a list of source entries>",
    }
    for field, info in Campaign.model_fields.items():
        if field in skeleton:
            continue
        if info.default_factory is not None:
            value = info.default_factory()
            skeleton[field] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        else:
            skeleton[field] = info.default
    typer.echo(yaml.safe_dump(skeleton, sort_keys=False, default_flow_style=False))


@app.command()
def doctor(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    machine: Annotated[Optional[str], typer.Option("--machine", "-m")] = None,
    elements: Annotated[Optional[str], typer.Option("--elements", help="comma-separated, e.g. Sm,Fe,Ti")] = None,
    fix: Annotated[bool, typer.Option("--fix", help="create the POTCAR symlink layout")] = False,
) -> None:
    """Check everything that can be known before a job is submitted.

    Exits non-zero on any hard failure, so it can gate a submission script.
    """
    cfg = _load(campaign, set_, machine)

    els: list[str] = []
    if elements:
        els = [e.strip() for e in elements.split(",") if e.strip()]
    else:
        db = _db_path(cfg)
        if db.is_file():
            try:
                with Store.open(db) as store:
                    els = sorted({e for cs in store.chemsystems() for e in cs.split("-")})
            except StoreError:
                els = []

    report = doctor_mod.run(cfg, elements=els, fix=fix)
    typer.echo(report.render())
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    root: Annotated[Path, typer.Argument(help="existing campaign directory to import")],
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="only this many formula directories")] = None,
    source_name: Annotated[str, typer.Option("--source-name")] = "ingested",
    db: Annotated[Optional[Path], typer.Option("--db", help="write here instead of the campaign workdir")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Import an existing campaign directory into a cspflow database.

    `--limit` caps the number of formula directories, which is what makes this
    quick to sanity-check: a handful exercises every code path that all of them
    would.
    """
    cfg = _load(campaign, set_)
    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), Path(cfg.campaign.workdir))

    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    seen = 0

    def progress(formula: str) -> None:
        nonlocal seen
        seen += 1
        if not quiet:
            typer.echo(f"  [{seen}] {formula}", err=True)

    with store:
        try:
            stats = ingest_campaign(root, store, limit=limit, source_name=source_name,
                                    progress=None if quiet else progress)
        except IngestError as exc:
            _die(str(exc))
    typer.echo(stats.render())
    typer.echo(f"\nwrote {target}")


@app.command()
def adopt(
    flow: Annotated[str, typer.Argument(help=f"which legacy campaign: {', '.join(LEGACY_REGISTRY)}, or 'all'")],
    dest: Annotated[Path, typer.Option("--dest", help="where the campaign directories go")] = Path("/scratch/$USER/cspflow_results"),
    staging: Annotated[Optional[Path], typer.Option("--staging", help="build here first, then move (SQLite on NFS commits ~15x slower)")] = Path("/tmp"),
    machine: Annotated[str, typer.Option("--machine", help="machine profile to record in campaign.yaml")] = "orion",
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="only this many structures, for a quick check")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Adopt a finished legacy campaign into cspflow's own layout.

    Reads all six of a legacy flow's result artefacts -- not just its VASP
    directories, which is what `csp ingest` does -- and writes the campaign
    cspflow would have written had it run the work itself.  Nothing is
    recomputed and nothing in the source directory is touched.

    The VASP outputs are 549 GB and stay where they are; `dft_dir` and
    `job.workdir` point at them, and `campaign_meta['legacy.root']` records the
    prefix so a move needs one UPDATE rather than a re-adoption.
    """
    import os

    dest = Path(os.path.expandvars(str(dest)))
    names = list(LEGACY_REGISTRY) if flow == "all" else [flow]
    unknown = [n for n in names if n not in LEGACY_REGISTRY]
    if unknown:
        _die(f"unknown legacy campaign(s) {unknown}. Known: {', '.join(LEGACY_REGISTRY)}")

    def say(message: str) -> None:
        if not quiet:
            typer.echo(message, err=True)

    for name in names:
        say(f"{name}  <-  {LEGACY_REGISTRY[name].root}")
        try:
            stats = adopt_legacy(LEGACY_REGISTRY[name], dest, staging=staging,
                          limit=limit, machine=machine,
                          progress=None if quiet else say)
        except LegacyError as exc:
            _die(str(exc))
        typer.echo(stats.render())
        typer.echo(f"\nwrote {dest / name}\n")


@app.command()
def source(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="print the plan, write nothing")] = False,
    db: Annotated[Optional[Path], typer.Option("--db", help="write here instead of the campaign workdir")] = None,
    gpu_seconds: Annotated[float, typer.Option("--gpu-seconds", help="seconds per generated structure, for the time estimate")] = 0.0,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="commit only the first N composition rows")] = None,
) -> None:
    """Stage 0 -- expand `source:` into composition and seed rows.

    Enumeration happens entirely in memory and the plan is printed before
    anything is written, so `--dry-run` runs the identical code path and the
    estimate you approve is produced by the code that then does the work.

    `--limit` truncates the committed rows, which is what makes a large chemical
    space quick to sanity-check: the full enumeration is still reported, only the
    commit is capped.
    """
    cfg = _load(campaign, set_)
    base = cfg.base_dir

    try:
        plan = expand_all(cfg.campaign, base)
    except SourceError as exc:
        _die(str(exc))

    typer.echo(plan.render(gpu_seconds_per_structure=gpu_seconds or None))

    if dry_run:
        typer.echo("\n--dry-run: nothing written")
        return

    if limit is not None:
        for result in plan.results:
            result.compositions = result.compositions[:limit]

    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), Path(cfg.campaign.workdir))
    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    with store:
        store.add_provenance(
            config_hash=cfg.config_hash,
            machine=str(cfg.machine_path),
            resolved_config=cfg.campaign.model_dump(mode="json"),
        )
        stats = write_plan(plan, store)
    typer.echo("")
    typer.echo(stats.render())
    typer.echo(f"wrote {target}")


@app.command()
def run(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    through: Annotated[Optional[str], typer.Option("--through", help="run stages up to and including this one")] = None,
    from_: Annotated[Optional[str], typer.Option("--from", help="run stages from this one on")] = None,
    only: Annotated[Optional[str], typer.Option("--only", help="a single stage")] = None,
    watch: Annotated[bool, typer.Option("--watch", help="keep cycling instead of stopping when idle")] = False,
    interval: Annotated[int, typer.Option("--interval", help="seconds between cycles")] = 300,
    max_cycles: Annotated[Optional[int], typer.Option("--max-cycles")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="report what would be submitted, claim nothing")] = False,
    db: Annotated[Optional[Path], typer.Option("--db")] = None,
) -> None:
    """The driver loop: reconcile what is in flight, submit what fits, repeat.

    `--through calibrate` is Phase A -- cheap, run to completion for every
    composition. `--from filter --watch` is Phase B -- expensive, streamed under
    a core-hour budget. The two are the same loop over a different stage slice.
    """
    cfg = _load(campaign, set_)
    base = cfg.base_dir
    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    _link_results(Path(campaign), Path(cfg.campaign.workdir))

    wanted = _stage_slice(through, from_, only)
    runnable = [name for name in wanted if name in IMPLEMENTED]
    skipped = [name for name in wanted if name not in IMPLEMENTED]
    if skipped:
        typer.secho(
            "not yet implemented, skipped: "
            + ", ".join(f"{n} ({PLANNED.get(n, '?')})" for n in skipped),
            fg=typer.colors.YELLOW, err=True,
        )
    if not runnable:
        _die("none of the requested stages are implemented yet")

    store = Store.open(target) if target.is_file() else Store.create(
        target, campaign=cfg.campaign.name, config_hash=cfg.config_hash
    )
    options = DriverOptions(interval=interval, max_cycles=max_cycles,
                            stages=runnable, dry_run=dry_run)
    with store:
        try:
            driver = Driver(cfg, store, for_machine(cfg.machine, dry_run=dry_run),
                            build_registry(cfg, base), options,
                            emit=lambda msg: typer.echo(msg))
            driver.run(watch=watch)
        except (DriverError, SourceError) as exc:
            _die(str(exc))
    typer.echo(f"\ndatabase {target}")


def _stage_slice(through: str | None, from_: str | None, only: str | None) -> list[str]:
    """Turn --through/--from/--only into a contiguous slice of the funnel."""
    for name in (through, from_, only):
        if name is not None and name not in STAGE_ORDER:
            _die(f"unknown stage {name!r}; the funnel is {STAGE_ORDER}")
    if only:
        return [only]
    start = STAGE_ORDER.index(from_) if from_ else 0
    stop = STAGE_ORDER.index(through) + 1 if through else len(STAGE_ORDER)
    if stop <= start:
        _die(f"--from {from_} comes after --through {through}; that selects nothing")
    return STAGE_ORDER[start:stop]


@app.command()
def recipe(
    name: Annotated[Optional[str], typer.Argument(help="shipped recipe name or a path; default: this campaign's")] = None,
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
) -> None:
    """Print a recipe fully resolved -- every tag literal, nothing deferred.

    With no argument this prints the recipe the campaign in this folder would
    actually use, which is the question being asked most of the time. This is
    what `inherit:` copying buys: there is no value here whose meaning requires
    knowing pymatgen to predict.
    """
    from .dft.recipe import RecipeError, load_recipe, validate_recipe
    from .dft.vasp.incar import render_incar

    base: Path | None = None
    if name is None:
        found = _find_campaign(campaign)
        if found.is_file():
            cfg = _load(found, None)
            name, base = cfg.campaign.dft.recipe, cfg.base_dir
        else:
            name = "magnets"

    try:
        loaded = load_recipe(name, base)
        warnings = validate_recipe(loaded)
    except RecipeError as exc:
        _die(str(exc))

    typer.echo(f"recipe {loaded.name}  ({loaded.source})")
    for stage in loaded.stages:
        typer.echo(f"\n=== {stage.name} ===")
        typer.echo(render_incar(stage.incar).rstrip())
        typer.echo(f"kpoints   {stage.kpoints.as_dict()}")
        typer.echo(f"resources {stage.resources}")
        if stage.retry:
            typer.echo(f"retry     {[r.get('when') for r in stage.retry]}")
    typer.echo("\n# MAGMOM, NBANDS, LMAXMIX, SYSTEM and LDAU* are computed per")
    typer.echo("# structure and written into the emitted INCAR.")
    for w in warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW, err=True)


@app.command("screen-worker", hidden=True)
def screen_worker(
    manifest: Annotated[Path, typer.Option("--manifest", help="written by the screen stage")],
    task_id: Annotated[Optional[int], typer.Option("--task-id", help="defaults to $SLURM_ARRAY_TASK_ID")] = None,
) -> None:
    """Relax one chunk of a screen manifest.  Run by array tasks, not by hand.

    Hidden because it is an implementation detail of the screen stage: it takes
    a manifest the stage wrote and writes a results file the driver reads. It is
    exposed as a command only because that is how a SLURM array task invokes it.
    """
    try:
        out = run_screen_task(manifest, task_id)
    except WorkerError as exc:
        _die(str(exc))
    typer.echo(str(out))


@app.command()
def report(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="directory for report.html and candidates.csv")] = None,
    limit: Annotated[int, typer.Option("--limit", help="rows in the table; 0 for all")] = 2000,
) -> None:
    """Write the candidate table and a self-contained HTML dashboard.

    This is the direct answer to "seeing results is difficult": one file that
    opens in a browser with no server, no build step and no network, plus the
    same table as CSV for anything downstream.
    """
    from .report.candidates import candidate_rows, funnel, write_csv
    from .report.html import write as write_html

    cfg = _load(campaign, set_)
    db = _db_path(cfg)
    if not db.is_file():
        _die(f"no campaign database at {db}. Run `csp init` and then a stage.")

    directory = out or (cfg.base_dir / "report")
    with Store.open(db) as store:
        rows = candidate_rows(store, limit=limit or None)
        csv_path = write_csv(rows, directory / "candidates.csv")
        html_path = write_html(store, directory / "report.html",
                               title=cfg.campaign.name, limit=limit or None)
        counts = funnel(store)

    typer.echo(counts.render())
    typer.echo("")
    typer.echo(f"{len(rows)} candidate(s)")
    typer.echo(f"table  {csv_path}")
    typer.echo(f"report {html_path}")
    if not rows:
        typer.secho("no candidates yet -- nothing has reached dft_done",
                    fg=typer.colors.YELLOW, err=True)


@app.command("generate-worker", hidden=True)
def generate_worker(
    manifest: Annotated[Path, typer.Option("--manifest", help="written by the generate stage")],
    task_id: Annotated[Optional[int], typer.Option("--task-id", help="defaults to $SLURM_ARRAY_TASK_ID")] = None,
) -> None:
    """Generate structures for one chunk of a generate manifest.

    Hidden for the same reason as `screen-worker`: it exists so that a SLURM
    array task has something to invoke, and takes all of its instructions from
    the manifest the stage wrote.
    """
    try:
        out = run_generate_task(manifest, task_id)
    except WorkerError as exc:
        _die(str(exc))
    typer.echo(str(out))


@app.command()
def status(
    campaign: CampaignOpt = Path(DEFAULT_CAMPAIGN),
    set_: SetOpt = None,
    why: Annotated[Optional[int], typer.Option("--why", help="full life history of one structure id")] = None,
) -> None:
    """Show campaign progress."""
    cfg = _load(campaign, set_)
    db = _db_path(cfg)
    if not db.is_file():
        _die(f"no campaign database at {db}. Run `csp init` and then a stage.")

    with Store.open(db) as store:
        if why is not None:
            _print_history(store, why)
            return
        s = store.summary()
        typer.echo(f"campaign     {s['campaign']}")
        typer.echo(f"database     {db}")
        typer.echo(f"compositions {s['compositions']}  across {s['chemsystems']} chemical systems")
        # Generation yield is reported before the structure counts because a
        # shortfall here is invisible below it: the funnel narrows anyway, and
        # 40% fewer candidates entering it looks exactly like a smaller campaign.
        gen = store.generation_yield()
        # Only when something was actually asked for: a seeded campaign has
        # compositions in state `generated` with nothing requested, and
        # "0 of 0 requested (0.0%)" reads as a failure rather than as silence.
        if gen["requested"]:
            pct = 100.0 * gen["produced"] / gen["requested"] if gen["requested"] else 0.0
            line = (f"generated    {gen['produced']:,} of {gen['requested']:,} "
                    f"requested ({pct:.1f}%)")
            if gen["short"]:
                line += f"   <- {gen['short']} composition(s) short"
            typer.echo(line)
        typer.echo(f"structures   {s['structures']}")
        for state, n in sorted(s["structures_by_state"].items()):
            typer.echo(f"    {state:<16} {n}")
        typer.echo(f"reference    {s['reference_entries']} MP entries")
        if s["jobs"]:
            typer.echo("jobs")
            for state, n in sorted(s["jobs"].items()):
                typer.echo(f"    {state:<16} {n}")
            typer.echo(f"    {'core-hours':<16} {s['core_hours']:,.0f}")
        if s["relaxations"]:
            # Reported separately from job state on purpose: a VASP run that
            # exits cleanly at the ionic step limit is `done` and NOT relaxed.
            # 61% of redo-new-ter-mag was exactly that.
            typer.echo("relaxations")
            for key, n in sorted(s["relaxations"].items()):
                flag = "   <- not usable as a relaxed geometry" if "not converged" in key else ""
                typer.echo(f"    {key:<24} {n}{flag}")


def _print_history(store: Store, sid: int) -> None:
    try:
        row = store.get_structure(sid)
    except StoreError as exc:
        _die(str(exc))
    typer.echo(f"structure {sid}: {row.formula}")
    for key, value in sorted(row.key_value_pairs.items()):
        typer.echo(f"    {key:<20} {value}")
    events = store.filter_events(sid)
    if events:
        typer.echo("  gates")
        for e in events:
            verdict = "pass" if e["passed"] else "FAIL"
            typer.echo(f"    {e['gate']:<20} {verdict:<5} value={e['value']} threshold={e['threshold']}")
    props = store.properties(sid)
    if props:
        typer.echo("  properties")
        for p in props:
            typer.echo(f"    {p['key']:<20} {p['value']}  ({p['source']})")
    jobs = [j for j in store.jobs() if j["structure_id"] == sid]
    if jobs:
        typer.echo("  jobs")
        for j in jobs:
            typer.echo(
                f"    {j['stage']}/{j['recipe_step'] or '-':<8} {j['state']:<9} "
                f"attempt {j['attempt']}  {j['exit_reason']}"
            )


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        typer.secho("interrupted", fg=typer.colors.YELLOW, err=True)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
