"""The only thing that opens a campaign database.

One SQLite file holds both halves of the state model: ASE owns `systems` (one
row per structure), and the tables in schema.sql own campaign state.  Keeping
every read and write behind this class is what lets the integrity rules be
enforced at write time rather than hoped for -- in particular the refusal to mix
energy scales or settings inside one hull, which no analysis path can then
bypass.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.db import connect as ase_connect

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
SCHEMA_VERSION = "2"

# ASE key_value_pairs accept only these.  A list, dict or None raises
# ValueError deep inside ASE; we catch it at the boundary with a message that
# says what to do instead.
_ASE_SCALARS = (str, int, float, bool)

# ASE also reserves key *names*: every element symbol plus about forty of its own
# row attributes (`formula`, `energy`, `magmom`, `natoms`, `id`, `user`, `age`,
# `fmax`, ...).  Writing one raises a bare `ValueError: Bad key: formula` from
# four frames inside ASE, which says nothing about why a perfectly ordinary word
# is not allowed -- so the check is hoisted here, where the reason can be given.
try:                                                     # pragma: no cover - ASE layout
    from ase.db.core import reserved_keys as _ASE_RESERVED
except ImportError:                                      # pragma: no cover
    _ASE_RESERVED = frozenset()


class StoreError(Exception):
    """Anything wrong with the campaign database."""


class StructureState(str, Enum):
    """Explicit states.  Pending work is found by these, never by a missing key.

    ASE's ``db.select('~key')`` does not return the complement of
    ``db.select('key')``, so "not yet computed" cannot be expressed as an absent
    key.  Every stage writes its state here instead.
    """

    new = "new"
    screening = "screening"
    screened = "screened"
    deduped = "deduped"
    filtered_out = "filtered_out"
    selected = "selected"
    dft_queued = "dft_queued"
    dft_running = "dft_running"
    dft_done = "dft_done"
    failed = "failed"


class Origin(str, Enum):
    generated = "generated"
    mp = "mp"
    seed = "seed"


@dataclass(frozen=True)
class CompositionRow:
    id: int
    formula: str
    chemsys: str
    z: int
    n_atoms: int
    n_target: int
    n_produced: int
    source_mode: str
    source_name: str
    state: str
    fail_reason: str = ""


def _clean_kv(kv: dict[str, Any]) -> dict[str, Any]:
    """Validate key-value pairs destined for ASE, with an actionable message."""
    out: dict[str, Any] = {}
    for key, value in kv.items():
        if value is None:
            raise StoreError(
                f"key {key!r} is None. ASE key_value_pairs cannot hold null; a value "
                f"that is not yet measured must be represented by a `state`, and a "
                f"failure by state='failed' plus a reason."
            )
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, _ASE_SCALARS):
            raise StoreError(
                f"key {key!r} is a {type(value).__name__}. ASE key_value_pairs hold only "
                f"str/int/float/bool; put structured values in the `data=` blob instead "
                f"(they round-trip, but are not queryable)."
            )
        if key in _ASE_RESERVED:
            raise StoreError(
                f"key {key!r} is reserved by ASE (it reserves every element symbol "
                f"plus its own row attributes such as formula/energy/magmom/natoms/"
                f"id/user). Writing it raises `ValueError: Bad key: {key}` from inside "
                f"ASE. Prefix or qualify the name -- e.g. 'reduced_formula', "
                f"'vasp_energy' -- so it cannot collide with a column ASE owns."
            )
        if _looks_like_a_formula(key):
            raise StoreError(
                f"key {key!r} parses as a chemical formula. ASE warns about this rather "
                f"than refusing it, and the consequence is silent: db.select({key!r}) "
                f"returns rows CONTAINING those elements, not rows carrying this key, so "
                f"the query looks like it works and returns the wrong set. Rename the key."
            )
        out[key] = value
    return out


def _looks_like_a_formula(key: str) -> bool:
    try:
        from ase.formula import Formula
    except ImportError:                                  # pragma: no cover
        return False
    try:
        Formula(key, strict=True)
    except (ValueError, KeyError):
        return False
    return True


def _git_sha(repo: Path | None = None) -> str:
    root = repo or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class Store:
    """A campaign database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._sql: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def sql(self) -> sqlite3.Connection:
        if self._sql is None:
            self._sql = sqlite3.connect(str(self.path), timeout=30.0)
            self._sql.row_factory = sqlite3.Row
            self._sql.execute("PRAGMA journal_mode=WAL")
            self._sql.execute("PRAGMA foreign_keys=ON")
            self._sql.execute("PRAGMA busy_timeout=30000")
        return self._sql

    @property
    def ase(self):
        """A fresh ASE connection.

        ASE opens and closes per operation; holding one open across a long stage
        would keep a write lock that the array workers then block on.
        """
        return ase_connect(str(self.path), serial=True)

    @staticmethod
    def _check_sidecars(path: Path) -> None:
        """Refuse an orphaned WAL, with the fix named.

        The database runs in WAL mode, so it is really three files: `x.db`,
        `x.db-wal` and `x.db-shm`. Deleting only `x.db` -- the obvious way to
        start a campaign over -- leaves the other two, and SQLite then fails
        with a bare

            OperationalError: disk I/O error

        which says nothing whatever about the cause. Hit while testing `csp run`
        after `rm -f campaign.db`, which is exactly what a user would type.
        """
        orphans = [p for p in (Path(f"{path}-wal"), Path(f"{path}-shm")) if p.exists()]
        if orphans and not path.exists():
            names = ", ".join(p.name for p in orphans)
            raise StoreError(
                f"{path} does not exist but its write-ahead log does ({names}). "
                f"SQLite reports this as a bare 'disk I/O error'. The database is "
                f"three files in WAL mode; remove the leftovers too:\n"
                f"    rm -f {path}-wal {path}-shm"
            )

    @classmethod
    def create(cls, path: str | Path, *, campaign: str, config_hash: str = "") -> "Store":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._check_sidecars(path)
        store = cls(path)
        store.sql.executescript(SCHEMA_PATH.read_text())
        # Touch the ASE side so both halves exist from the start; otherwise the
        # first structure write creates ASE's tables at an arbitrary later time.
        _ = store.ase.count()
        store.set_meta("schema_version", SCHEMA_VERSION)
        store.set_meta("campaign", campaign)
        store.set_meta("config_hash", config_hash)
        store.sql.commit()
        return store

    @classmethod
    def open(cls, path: str | Path) -> "Store":
        path = Path(path)
        cls._check_sidecars(path)
        if not path.is_file():
            raise StoreError(f"no campaign database at {path}. Run `csp init` first.")
        store = cls(path)
        found = store.meta("schema_version")
        if found is None:
            raise StoreError(f"{path} is not a cspflow database (no schema_version).")
        if found != SCHEMA_VERSION:
            raise StoreError(
                f"{path} has schema version {found}, this cspflow expects "
                f"{SCHEMA_VERSION}. Migration is required; refusing to guess."
            )
        return store

    def close(self) -> None:
        if self._sql is not None:
            self._sql.commit()
            self._sql.close()
            self._sql = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.sql.execute(
            "INSERT INTO campaign_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.sql.commit()

    def meta(self, key: str) -> str | None:
        """Read campaign metadata.

        Returns None rather than raising when the table is absent: this is how
        `open` distinguishes a cspflow database from any other SQLite file, and
        a raw OperationalError there would be a worse message than the one
        `open` gives.
        """
        try:
            row = self.sql.execute(
                "SELECT value FROM campaign_meta WHERE key=?", (key,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["value"] if row else None

    # -- structures (ASE side) --------------------------------------------

    def add_structure(
        self,
        atoms: Atoms,
        *,
        origin: Origin | str,
        state: StructureState | str = StructureState.new,
        data: dict[str, Any] | None = None,
        **kv: Any,
    ) -> int:
        kv["origin"] = origin.value if isinstance(origin, Origin) else origin
        kv["state"] = state.value if isinstance(state, StructureState) else state
        return int(self.ase.write(atoms, data=data or {}, **_clean_kv(kv)))

    def update_structure(self, sid: int, *, data: dict[str, Any] | None = None, **kv: Any) -> None:
        if data is not None:
            self.ase.update(sid, data=data, **_clean_kv(kv))
        else:
            self.ase.update(sid, **_clean_kv(kv))

    def get_structure(self, sid: int):
        try:
            return self.ase.get(id=sid)
        except KeyError as exc:
            raise StoreError(f"no structure with id {sid}") from exc

    def structures(self, selection: str | None = None, **kwargs: Any) -> Iterator[Any]:
        yield from self.ase.select(selection, **kwargs)

    def count_structures(self, selection: str | None = None, **kwargs: Any) -> int:
        return int(self.ase.count(selection, **kwargs))

    def structure_ids(self, selection: str | None = None, **kwargs: Any) -> list[int]:
        return [int(r.id) for r in self.ase.select(selection, **kwargs)]

    def set_structure_state(self, sid: int, state: StructureState | str, **kv: Any) -> None:
        self.update_structure(sid, state=state, **kv)

    # -- compositions ------------------------------------------------------

    def add_composition(
        self,
        *,
        formula: str,
        chemsys: str,
        z: int,
        n_atoms: int,
        n_target: int,
        source_mode: str,
        source_name: str,
        state: str = "new",
    ) -> int:
        cur = self.sql.execute(
            "INSERT INTO composition "
            "(formula, chemsys, z, n_atoms, n_target, source_mode, source_name, state) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(formula, z, source_name) DO UPDATE SET "
            "  n_target=excluded.n_target, n_atoms=excluded.n_atoms",
            (formula, chemsys, z, n_atoms, n_target, source_mode, source_name, state),
        )
        self.sql.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.sql.execute(
            "SELECT id FROM composition WHERE formula=? AND z=? AND source_name=?",
            (formula, z, source_name),
        ).fetchone()
        return int(row["id"])

    def compositions(self, *, state: str | None = None, chemsys: str | None = None) -> list[CompositionRow]:
        q = "SELECT * FROM composition WHERE 1=1"
        args: list[Any] = []
        if state is not None:
            q += " AND state=?"
            args.append(state)
        if chemsys is not None:
            q += " AND chemsys=?"
            args.append(chemsys)
        q += " ORDER BY id"
        return [
            CompositionRow(
                id=r["id"], formula=r["formula"], chemsys=r["chemsys"], z=r["z"],
                n_atoms=r["n_atoms"], n_target=r["n_target"],
                n_produced=r["n_produced"], source_mode=r["source_mode"],
                source_name=r["source_name"], state=r["state"],
                fail_reason=r["fail_reason"],
            )
            for r in self.sql.execute(q, args)
        ]

    def set_composition_state(self, cid: int, state: str, fail_reason: str = "",
                              n_produced: int | None = None) -> None:
        """Move a composition to a new state, optionally recording its yield.

        `n_produced` is separate from the state on purpose.  "generated" says
        the generator ran and returned something; it does not say it returned
        what was asked for.  A campaign that quietly gets 60% of its requested
        structures back looks identical, at the state level, to one that gets
        100% -- so the number is stored rather than inferred.
        """
        if n_produced is None:
            self.sql.execute(
                "UPDATE composition SET state=?, fail_reason=? WHERE id=?",
                (state, fail_reason, cid))
        else:
            self.sql.execute(
                "UPDATE composition SET state=?, fail_reason=?, n_produced=? WHERE id=?",
                (state, fail_reason, int(n_produced), cid))
        self.sql.commit()

    def generation_yield(self) -> dict[str, int]:
        """Requested versus produced across every composition that has run.

        Reported by `csp status`, because a shortfall here is invisible further
        down: the funnel narrows anyway, and 40% fewer candidates entering it
        looks exactly like a campaign that was always going to be small.
        """
        row = self.sql.execute(
            "SELECT COUNT(*) AS n, "
            "       COALESCE(SUM(n_target), 0)   AS requested, "
            "       COALESCE(SUM(n_produced), 0) AS produced, "
            "       COALESCE(SUM(CASE WHEN n_produced < n_target THEN 1 ELSE 0 END), 0) AS short "
            "FROM composition WHERE state IN ('generated','done')").fetchone()
        return {"compositions": int(row["n"]), "requested": int(row["requested"]),
                "produced": int(row["produced"]), "short": int(row["short"])}

    def chemsystems(self) -> list[str]:
        return [r["chemsys"] for r in self.sql.execute(
            "SELECT DISTINCT chemsys FROM composition ORDER BY chemsys")]

    # -- provenance --------------------------------------------------------

    def add_provenance(
        self,
        *,
        config_hash: str,
        settings_hash: str = "",
        machine: str = "",
        code_version: str = "",
        resolved_config: dict[str, Any] | None = None,
    ) -> int:
        git_sha = _git_sha()
        blob = json.dumps(resolved_config or {}, sort_keys=True)
        self.sql.execute(
            "INSERT OR IGNORE INTO provenance "
            "(config_hash, settings_hash, git_sha, code_version, machine, resolved_config_json) "
            "VALUES (?,?,?,?,?,?)",
            (config_hash, settings_hash, git_sha, code_version, machine, blob),
        )
        self.sql.commit()
        row = self.sql.execute(
            "SELECT id FROM provenance WHERE config_hash=? AND settings_hash=? AND git_sha=?",
            (config_hash, settings_hash, git_sha),
        ).fetchone()
        return int(row["id"])

    # -- hull --------------------------------------------------------------

    def add_hull(
        self,
        *,
        structure_id: int,
        hull_type: str,
        energy_scale: str,
        e_above_hull: float,
        ref_set_hash: str,
        formation_energy: float | None = None,
        settings_hash: str = "",
    ) -> int:
        """Record a hull placement.

        The (hull_type, energy_scale, ref_set_hash, settings_hash) tuple is part
        of the identity.  `assert_hull_consistent` is what actually enforces
        that two scales never enter one hull; this method stores the labels that
        make the check possible.
        """
        cur = self.sql.execute(
            "INSERT INTO hull (structure_id, hull_type, energy_scale, e_above_hull, "
            "formation_energy, ref_set_hash, settings_hash) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(structure_id, hull_type, energy_scale, ref_set_hash) DO UPDATE SET "
            "  e_above_hull=excluded.e_above_hull, formation_energy=excluded.formation_energy, "
            "  settings_hash=excluded.settings_hash",
            (structure_id, hull_type, energy_scale, e_above_hull, formation_energy,
             ref_set_hash, settings_hash),
        )
        self.sql.commit()
        return int(cur.lastrowid or 0)

    def assert_hull_consistent(self, ref_set_hash: str) -> None:
        """Refuse a hull built from mixed energy scales or mixed settings.

        This is the write-time form of the rule stated in Stages 3b and 3c: two
        energies with different settings_hash may never enter the same hull.
        Enforced here so no analysis path can bypass it.
        """
        rows = self.sql.execute(
            "SELECT DISTINCT energy_scale, settings_hash FROM hull WHERE ref_set_hash=?",
            (ref_set_hash,),
        ).fetchall()
        scales = {r["energy_scale"] for r in rows}
        settings = {r["settings_hash"] for r in rows}
        if len(scales) > 1:
            raise StoreError(
                f"hull {ref_set_hash[:12]} mixes energy scales {sorted(scales)}. "
                f"Raw and MP-corrected energies may not enter the same hull."
            )
        if len(settings) > 1:
            raise StoreError(
                f"hull {ref_set_hash[:12]} mixes settings_hash {sorted(s[:12] for s in settings)}. "
                f"Energies from different INCAR/POTCAR settings may not enter the same hull."
            )

    # -- reference ---------------------------------------------------------

    def add_reference_entry(self, **fields: Any) -> int:
        cols = [
            "mp_id", "chemsys", "thermo_type", "run_type", "formula", "n_atoms",
            "e_dft_raw", "e_dft_corrected", "correction", "e_mlip_static",
            "e_mlip_relaxed", "volume_drift", "rmsd", "structure_id", "snapshot_id",
            "state", "fail_reason",
        ]
        unknown = set(fields) - set(cols)
        if unknown:
            raise StoreError(f"unknown reference_entry field(s): {sorted(unknown)}")
        present = [c for c in cols if c in fields]
        sql = (
            f"INSERT INTO reference_entry ({','.join(present)}) "
            f"VALUES ({','.join('?' * len(present))}) "
            f"ON CONFLICT(mp_id, thermo_type, snapshot_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in present if c != "mp_id")
        )
        self.sql.execute(sql, [fields[c] for c in present])
        self.sql.commit()
        row = self.sql.execute(
            "SELECT id FROM reference_entry WHERE mp_id=? AND thermo_type=? AND snapshot_id=?",
            (fields["mp_id"], fields["thermo_type"], fields.get("snapshot_id", "")),
        ).fetchone()
        return int(row["id"])

    def update_reference_entry(self, entry_id: int, **fields: Any) -> None:
        """Amend an existing reference entry in place.

        Distinct from `add_reference_entry`, which is an upsert and therefore
        has to supply every NOT NULL column. That makes it unable to express
        "record the MLIP energy for an entry that already exists": SQLite
        evaluates the INSERT before the ON CONFLICT clause, so the row fails on
        `chemsys` being NULL even though the conflicting row has one.

        Found when Stage 4a tried exactly that.
        """
        if not fields:
            return
        allowed = {
            "run_type", "formula", "n_atoms", "e_dft_raw", "e_dft_corrected",
            "correction", "e_mlip_static", "e_mlip_relaxed", "volume_drift",
            "rmsd", "structure_id", "state", "fail_reason",
        }
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise StoreError(
                f"unknown or immutable reference_entry field(s): {unknown}. "
                f"Identity columns (mp_id, thermo_type, snapshot_id, chemsys) are not "
                f"amendable -- a row whose identity changed is a different row."
            )
        assignments = ", ".join(f"{k}=?" for k in fields)
        self.sql.execute(f"UPDATE reference_entry SET {assignments} WHERE id=?",
                         [*fields.values(), entry_id])
        self.sql.commit()

    def reference_entries(
        self, *, chemsys: str | None = None, include_subsystems: bool = False
    ) -> list[sqlite3.Row]:
        """Reference phases, optionally including every sub-system.

        `include_subsystems` matters more than it looks, and defaults to False
        only because an exact-match query is the less surprising default.

        An elemental Fe entry's `chemsys` is `'Fe'`, not `'Fe-Sm'`. So
        `reference_entries(chemsys='Fe-Sm')` returns the binary compounds and
        **no elemental references at all** -- which is precisely the set the
        hull's elemental guard (D058) refuses, and precisely the trap D061 had
        to fix at the MP API level. It reappears here because the database
        stores each entry under its own chemical system, correctly.

        Anything building a hull wants `include_subsystems=True`.
        """
        q = "SELECT * FROM reference_entry"
        args: list[Any] = []
        if chemsys is not None:
            wanted = self._subsystems(chemsys) if include_subsystems else [chemsys]
            q += f" WHERE chemsys IN ({','.join('?' * len(wanted))})"
            args.extend(wanted)
        return list(self.sql.execute(q + " ORDER BY id", args))

    @staticmethod
    def _subsystems(chemsys: str) -> list[str]:
        from itertools import combinations

        elements = sorted(e for e in chemsys.split("-") if e)
        return ["-".join(c) for n in range(1, len(elements) + 1)
                for c in combinations(elements, n)]

    def assert_single_thermo_type(self) -> str:
        """MP silently mixes functionals; a reference set may contain only one.

        SmFe2 comes back as -7.1966 eV/atom under GGA and -19.4095 under
        r2SCAN, with no field in MP's summary response saying which you got.
        """
        rows = self.sql.execute(
            "SELECT DISTINCT thermo_type FROM reference_entry"
        ).fetchall()
        kinds = sorted(r["thermo_type"] for r in rows)
        if len(kinds) > 1:
            raise StoreError(
                f"reference set mixes functionals {kinds}. MP's summary endpoint "
                f"returns whichever functional it prefers per material with no field "
                f"saying which; a mixed set puts a multi-eV/atom discontinuity into "
                f"the hull. Pin reference.thermo_type and refetch."
            )
        return kinds[0] if kinds else ""

    # -- jobs --------------------------------------------------------------

    def add_job(self, *, stage: str, structure_id: int | None = None,
                recipe_step: str = "", workdir: str = "",
                provenance_id: int | None = None) -> int:
        cur = self.sql.execute(
            "INSERT INTO job (structure_id, stage, recipe_step, workdir, provenance_id) "
            "VALUES (?,?,?,?,?)",
            (structure_id, stage, recipe_step, workdir, provenance_id),
        )
        self.sql.commit()
        return int(cur.lastrowid or 0)

    def update_job(self, job_id: int, **fields: Any) -> None:
        allowed = {
            "slurm_id", "array_task_id", "state", "attempt", "workdir",
            "core_hours", "exit_reason", "remedy", "submitted_at", "finished_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StoreError(f"unknown job field(s): {sorted(unknown)}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.sql.execute(f"UPDATE job SET {sets} WHERE id=?", [*fields.values(), job_id])
        self.sql.commit()

    def orphan_jobs(self) -> int:
        """Job rows written but never stamped with a scheduler id.

        The driver writes its rows before submitting so that a process killed
        mid-submission leaves evidence rather than nothing; this is how that
        evidence is read back. A non-zero count means a job may be running that
        no cycle will ever reconcile.
        """
        row = self.sql.execute(
            "SELECT COUNT(*) AS n FROM job WHERE slurm_id='' AND state='pending'"
        ).fetchone()
        return int(row["n"])

    def assert_job_id_is_new(self, slurm_id: str, stage: str, workdir: str) -> None:
        """Refuse a scheduler id that already belongs to some other submission.

        One array submission legitimately writes many job rows under a single
        id -- that is how the funnel's chunked stages work -- so an id is not
        unique by itself.  What must never happen is the *same* id naming two
        different submissions, because reconciliation groups rows by id and
        would then apply one job's outcome to the other's rows.

        Real SLURM ids are globally unique, so this only ever fires for a local
        run whose scheduler restarted its counter.  It is checked here anyway:
        nothing downstream can tell a reused id from an array, and the failure
        it produces is silent.
        """
        row = self.sql.execute(
            "SELECT stage, workdir FROM job WHERE slurm_id=? AND "
            "(stage!=? OR workdir!=?) LIMIT 1", (slurm_id, stage, workdir)).fetchone()
        if row is not None:
            raise StoreError(
                f"scheduler id {slurm_id!r} is already recorded for stage "
                f"{row['stage']!r} in {row['workdir']!r}; this submission is "
                f"stage {stage!r} in {workdir!r}. Two submissions under one id "
                f"would be reconciled against each other."
            )

    def jobs(self, *, state: str | None = None, stage: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM job WHERE 1=1", []
        if state is not None:
            q += " AND state=?"
            args.append(state)
        if stage is not None:
            q += " AND stage=?"
            args.append(stage)
        return list(self.sql.execute(q + " ORDER BY id", args))

    def count_jobs_by_state(self, stage: str | None = None) -> dict[str, int]:
        q = "SELECT state, COUNT(*) n FROM job"
        args: list[Any] = []
        if stage is not None:
            q += " WHERE stage=?"
            args.append(stage)
        return {r["state"]: r["n"] for r in self.sql.execute(q + " GROUP BY state", args)}

    # -- filter events, properties, calibration ----------------------------

    def add_filter_event(self, *, structure_id: int, gate: str, passed: bool,
                         value: float | None = None, threshold: float | None = None,
                         detail: str = "") -> None:
        self.sql.execute(
            "INSERT INTO filter_event (structure_id, gate, passed, value, threshold, detail) "
            "VALUES (?,?,?,?,?,?)",
            (structure_id, gate, int(passed), value, threshold, detail),
        )
        self.sql.commit()

    def filter_events(self, structure_id: int) -> list[sqlite3.Row]:
        return list(self.sql.execute(
            "SELECT * FROM filter_event WHERE structure_id=? ORDER BY id", (structure_id,)))

    def add_property(self, *, structure_id: int, key: str, source: str,
                     value: float | None = None, text_value: str = "") -> None:
        self.sql.execute(
            "INSERT INTO property (structure_id, key, value, text_value, source) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(structure_id, key, source) DO UPDATE SET "
            "  value=excluded.value, text_value=excluded.text_value",
            (structure_id, key, value, text_value, source),
        )
        self.sql.commit()

    def properties(self, structure_id: int) -> list[sqlite3.Row]:
        return list(self.sql.execute(
            "SELECT * FROM property WHERE structure_id=? ORDER BY key", (structure_id,)))

    def add_calibration(self, *, kind: str, n_points: int, verdict: str,
                        provenance_id: int | None = None, **metrics: Any) -> int:
        allowed = {"mae_e_per_atom", "mae_e_hull", "spearman", "volume_drift",
                   "alpha", "beta", "detail"}
        unknown = set(metrics) - allowed
        if unknown:
            raise StoreError(f"unknown calibration field(s): {sorted(unknown)}")
        cols = ["kind", "n_points", "verdict", "provenance_id", *metrics]
        vals = [kind, n_points, verdict, provenance_id, *metrics.values()]
        cur = self.sql.execute(
            f"INSERT INTO calibration ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )
        self.sql.commit()
        return int(cur.lastrowid or 0)

    def latest_calibration(self, kind: str) -> sqlite3.Row | None:
        return self.sql.execute(
            "SELECT * FROM calibration WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()

    # -- relaxations -------------------------------------------------------

    def add_relaxation(self, *, structure_id: int, engine: str, energy: float | None = None,
                       e_per_atom: float | None = None, converged: bool = False,
                       n_steps: int = 0, volume_before: float | None = None,
                       volume_after: float | None = None,
                       provenance_id: int | None = None) -> int:
        cur = self.sql.execute(
            "INSERT INTO relaxation (structure_id, engine, energy, e_per_atom, converged, "
            "n_steps, volume_before, volume_after, provenance_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (structure_id, engine, energy, e_per_atom, int(converged), n_steps,
             volume_before, volume_after, provenance_id),
        )
        self.sql.commit()
        return int(cur.lastrowid or 0)

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "campaign": self.meta("campaign"),
            "structures": self.count_structures(),
            "compositions": self.sql.execute("SELECT COUNT(*) n FROM composition").fetchone()["n"],
            "chemsystems": len(self.chemsystems()),
            "reference_entries": self.sql.execute(
                "SELECT COUNT(*) n FROM reference_entry").fetchone()["n"],
            "jobs": self.count_jobs_by_state(),
            "structures_by_state": {
                r["state"]: r["n"] for r in self.sql.execute(
                    "SELECT value AS state, COUNT(*) n FROM text_key_values "
                    "WHERE key='state' GROUP BY value")
            },
            "relaxations": self.relaxation_outcomes(),
            "core_hours": float(self.sql.execute(
                "SELECT COALESCE(SUM(core_hours), 0.0) h FROM job").fetchone()["h"]),
        }

    def relaxation_outcomes(self) -> dict[str, int]:
        """Converged vs. not, per engine -- never folded into a single "done".

        This is D027 surfaced where a user will actually see it. Over 106 jobs
        in `redo-new-ter-mag`, 100% exited cleanly and only 39% reached required
        accuracy; a status line reading "106 done" describes the process
        faithfully and the science not at all.
        """
        out: dict[str, int] = {}
        for row in self.sql.execute(
            "SELECT engine, converged, COUNT(*) n FROM relaxation "
            "GROUP BY engine, converged ORDER BY engine"
        ):
            key = f"{row['engine']}:{'converged' if row['converged'] else 'not converged'}"
            out[key] = row["n"]
        return out
