"""The `csp` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from . import __version__, doctor as doctor_mod
from .config.loader import ConfigError, load_campaign
from .config.schema import Campaign
from .db.store import Store, StoreError
from .ingest import IngestError, ingest_campaign
from .driver import STAGE_ORDER, Driver, DriverError, DriverOptions
from .scheduler import for_machine
from .source import SourceError, expand_all, write_plan
from .stages import IMPLEMENTED, PLANNED, build_registry
from .worker import WorkerError, run_screen_task
from .templates import scaffold

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


def _load(campaign: Path, sets: list[str] | None, machine: str | None = None):
    try:
        return load_campaign(campaign, sets=sets, machine=machine)
    except ConfigError as exc:
        _die(str(exc))


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
    out: Annotated[Path, typer.Option("--out", "-o", help="where to write the campaign file")] = Path(DEFAULT_CAMPAIGN),
    machine: Annotated[str, typer.Option("--machine", "-m")] = "orion",
    full: Annotated[bool, typer.Option("--full", help="include commented Tier-2 knobs")] = False,
    force: Annotated[bool, typer.Option("--force", help="overwrite an existing file")] = False,
) -> None:
    """Write a starter campaign file.

    Emits only the ~10 Tier-1 keys by default; `--full` adds the commonly tuned
    Tier-2 knobs, commented out with their defaults shown.
    """
    if out.exists() and not force:
        _die(f"{out} already exists (use --force to overwrite)")
    out.write_text(scaffold(name=name, machine=machine, full=full))
    typer.echo(f"wrote {out}")
    typer.echo("Next: edit it, then run `csp doctor`.")


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
    base = Path(campaign).resolve().parent

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
    base = Path(campaign).resolve().parent
    target = db or _db_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)

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
