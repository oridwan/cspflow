"""Ingest of a legacy campaign directory.

Built on a synthetic fixture rather than the real 444-formula campaign: a
handful of directories exercises every code path that all of them would, and
the suite stays fast enough to run on every change. One opt-in test reads a
small slice of the real campaign to confirm the fixture is not a fiction.
"""

from pathlib import Path

import pytest

from cspflow.db.store import Store
from cspflow.dft.vasp.parse import read_job_directory, read_oszicar, read_outcar_status
from cspflow.ingest import IngestError, ingest_campaign, parse_formula

REAL = Path("/projects/mmi/shuo/redo-new-ter-mag")
has_real = pytest.mark.skipif(not REAL.is_dir(), reason="legacy campaign not present")


# --------------------------------------------------------------------------
# fixture builder
# --------------------------------------------------------------------------

POSCAR = """\
Gd Co
1.0
  5.0 0.0 0.0
  0.0 5.0 0.0
  0.0 0.0 5.0
Gd Co
1 2
Direct
  0.0 0.0 0.0
  0.5 0.5 0.0
  0.0 0.5 0.5
"""


def _oszicar(steps: int, e0: float = -25.0, mag: float = 3.5) -> str:
    lines = []
    for i in range(1, steps + 1):
        lines.append(f"RMM:   4    -0.2E+03   -0.4E-06   -0.2E-05  3816   0.25E-02")
        lines.append(
            f"{i:4d} F= {e0 - 0.001:.8E} E0= {e0:.8E}  d E =0.3E-05  mag=  {mag:.4f}"
        )
    return "\n".join(lines) + "\n"


def _outcar(*, n_atoms: int, converged: bool, finished: bool, elapsed: float = 3600.0) -> str:
    head = f"""\
 vasp.6.4.3 test
   NIONS = {n_atoms}
   NBANDS = 40
"""
    body = "\n".join(" free energy line" for _ in range(50))
    tail = ""
    if converged:
        tail += "\n reached required accuracy - stopping structural energy minimisation\n"
    if finished:
        tail += (
            "\n General timing and accounting informations for this job:\n"
            f"                  Elapsed time (sec): {elapsed}\n"
        )
    return head + body + tail


def make_job(directory: Path, *, n_atoms=3, steps=10, nsw=100, converged=True,
             finished=True, slurm="12345678", potcars=("Gd_3", "Co")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "POSCAR").write_text(POSCAR)
    (directory / "CONTCAR").write_text(POSCAR)
    (directory / "INCAR").write_text(f"NSW = {nsw}\nENCUT = 520\nNCORE = 4\nISPIN = 2\n")
    (directory / "OSZICAR").write_text(_oszicar(steps))
    (directory / "OUTCAR").write_text(
        _outcar(n_atoms=n_atoms, converged=converged, finished=finished)
    )
    (directory / "POTCAR").write_text(
        "".join(f"   TITEL  = PAW_PBE {p} 06Sep2000\n" for p in potcars)
    )
    if finished:
        (directory / "VASP_DONE").write_text("")
    (directory / f"vasp_{slurm}.out").write_text("ok\n")
    return directory


@pytest.fixture()
def campaign_dir(tmp_path):
    """Four structures spanning every outcome the real campaign contains."""
    root = tmp_path / "legacy" / "VASP_JOBS"
    make_job(root / "Gd1Co2" / "Gd1Co2_s001" / "Relax", steps=58, converged=True)
    make_job(root / "Gd1Co2" / "Gd1Co2_s002" / "Relax", steps=100, nsw=100, converged=False)
    make_job(root / "Gd1Fe2" / "Gd1Fe2_s001" / "Relax", steps=40, converged=False,
             finished=False)                                  # killed mid-run
    d = root / "Gd1Fe2" / "Gd1Fe2_s002" / "Relax"
    d.mkdir(parents=True)
    (d / "INCAR").write_text("NSW = 100\n")                   # no OUTCAR at all
    return tmp_path / "legacy"


# --------------------------------------------------------------------------
# formula parsing
# --------------------------------------------------------------------------


def test_parse_formula():
    chemsys, counts = parse_formula("Gd1Co10Cr2")
    assert chemsys == "Co-Cr-Gd"
    assert counts == {"Gd": 1, "Co": 10, "Cr": 2}


def test_parse_formula_implicit_one():
    _, counts = parse_formula("SmFe11Ti")
    assert counts == {"Sm": 1, "Fe": 11, "Ti": 1}


def test_parse_formula_two_letter_elements():
    chemsys, counts = parse_formula("Nd2Fe14B")
    assert counts == {"Nd": 2, "Fe": 14, "B": 1}
    assert chemsys == "B-Fe-Nd"


def test_parse_formula_rejects_junk():
    with pytest.raises(IngestError):
        parse_formula("1234")


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------


def test_oszicar_reads_the_last_ionic_step(tmp_path):
    p = tmp_path / "OSZICAR"
    p.write_text(_oszicar(58, e0=-189.30176, mag=23.2242))
    r = read_oszicar(p)
    assert r.n_ionic_steps == 58
    assert r.e0 == pytest.approx(-189.30176)
    assert r.magnetisation == pytest.approx(23.2242)


def test_oszicar_missing_file_is_not_an_error(tmp_path):
    assert read_oszicar(tmp_path / "nope").n_ionic_steps == 0


def test_outcar_separates_finished_from_converged(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text(_outcar(n_atoms=13, converged=False, finished=True))
    st = read_outcar_status(p)
    assert st.finished is True and st.converged is False
    assert st.hit_step_limit is True
    assert st.n_atoms == 13


def test_outcar_nions_comes_from_the_header(tmp_path):
    """NIONS is printed in the header, so a tail-only read misses it."""
    p = tmp_path / "OUTCAR"
    p.write_text(_outcar(n_atoms=26, converged=True, finished=True) + "x" * 500_000)
    assert read_outcar_status(p).n_atoms == 26


# --------------------------------------------------------------------------
# job classification -- the distinction that matters
# --------------------------------------------------------------------------


def test_converged_job(campaign_dir):
    o = read_job_directory(campaign_dir / "VASP_JOBS/Gd1Co2/Gd1Co2_s001/Relax")
    assert o.state == "done" and o.converged is True
    assert o.exit_reason == ""
    assert o.slurm_id == "12345678"
    assert o.potcar_symbols == ["Gd_3", "Co"]


def test_ionic_step_limit_is_done_but_not_converged(campaign_dir):
    """The case that makes VASP_DONE untrustworthy: VASP exited cleanly having
    never reached the force criterion."""
    o = read_job_directory(campaign_dir / "VASP_JOBS/Gd1Co2/Gd1Co2_s002/Relax")
    assert o.state == "done"
    assert o.converged is False
    assert o.exit_reason == "ionic_step_limit"
    assert o.unconverged_but_finished is True


def test_killed_job_is_a_timeout(campaign_dir):
    o = read_job_directory(campaign_dir / "VASP_JOBS/Gd1Fe2/Gd1Fe2_s001/Relax")
    assert o.state == "timeout" and "epilogue" in o.exit_reason


def test_missing_outcar(campaign_dir):
    o = read_job_directory(campaign_dir / "VASP_JOBS/Gd1Fe2/Gd1Fe2_s002/Relax")
    assert o.state == "failed" and o.exit_reason == "no OUTCAR"


def test_core_hours_scale_with_ncore(campaign_dir):
    o = read_job_directory(campaign_dir / "VASP_JOBS/Gd1Co2/Gd1Co2_s001/Relax")
    assert o.core_hours == pytest.approx(4.0)          # 3600 s * NCORE 4 / 3600


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="t") as s:
        yield s


def test_ingest_counts(campaign_dir, store):
    stats = ingest_campaign(campaign_dir, store)
    assert stats.formulas == 2
    assert stats.jobs == 4
    assert stats.converged == 1
    assert stats.hit_step_limit == 1
    assert stats.jobs_by_state == {"done": 2, "timeout": 1, "failed": 1}
    # Three structures, not four: the fourth directory has no POSCAR or CONTCAR
    # so there is no geometry to store.
    assert stats.structures == 3


def test_a_structure_with_no_geometry_is_skipped_but_its_jobs_are_kept(campaign_dir, store):
    """Core-hours were still spent, so the job is recorded unattached rather
    than dropped -- but a row with no geometry must never enter the database."""
    stats = ingest_campaign(campaign_dir, store)
    assert any("no readable geometry" in s for s in stats.skipped)
    assert store.count_structures() == 3
    orphans = [j for j in store.jobs() if j["structure_id"] is None]
    assert len(orphans) == 1 and orphans[0]["state"] == "failed"


def test_limit_caps_formula_directories(campaign_dir, store):
    """What makes this usable as a quick check rather than a batch job."""
    stats = ingest_campaign(campaign_dir, store, limit=1)
    assert stats.formulas == 1
    assert stats.structures == 2


def test_ingest_writes_structures_readable_by_ase(campaign_dir, store):
    ingest_campaign(campaign_dir, store)
    assert store.count_structures() == 3
    assert store.count_structures(converged=True) == 1
    assert store.count_structures(converged=False) == 2


def test_ingest_records_energy_and_moment(campaign_dir, store):
    ingest_campaign(campaign_dir, store)
    row = next(iter(store.structures(converged=True)))
    assert row.vasp_energy == pytest.approx(-25.0)
    assert row.magnetisation == pytest.approx(3.5)


def test_ingest_records_a_convergence_gate_per_job(campaign_dir, store):
    """`csp status --why` must be able to say *why* a structure is unusable."""
    ingest_campaign(campaign_dir, store)
    sids = store.structure_ids(converged=False)
    events = [e for sid in sids for e in store.filter_events(sid)]
    assert any(e["gate"] == "relax:converged" and not e["passed"] for e in events)
    assert any(e["detail"] == "ionic_step_limit" for e in events)


def test_ingest_is_idempotent(campaign_dir, store):
    a = ingest_campaign(campaign_dir, store)
    b = ingest_campaign(campaign_dir, store)
    assert a.formulas == b.formulas
    # compositions upsert; structures are append-only by design (a re-ingest is
    # a second observation, not a correction), so only compositions are checked.
    assert len(store.compositions()) == len(set(
        (c.formula, c.z, c.source_name) for c in store.compositions()))


def test_ingest_skips_a_structure_with_no_step_directory(campaign_dir, store):
    (campaign_dir / "VASP_JOBS/Gd1Co2/Gd1Co2_s003").mkdir()
    stats = ingest_campaign(campaign_dir, store)
    assert any("no recipe-step" in s for s in stats.skipped)


def test_missing_vasp_jobs_directory(tmp_path, store):
    with pytest.raises(IngestError, match="no VASP_JOBS"):
        ingest_campaign(tmp_path / "nothing-here", store)


def test_render_flags_unconverged(campaign_dir, store):
    out = ingest_campaign(campaign_dir, store).render()
    assert "hit ionic limit" in out and "NOT relaxed" in out


# --------------------------------------------------------------------------
# one check against the real thing
# --------------------------------------------------------------------------


@has_real
def test_small_slice_of_the_real_campaign(store):
    """Five formula directories, not 444 -- enough to prove the fixture is real."""
    stats = ingest_campaign(REAL, store, limit=5)
    assert stats.formulas == 5
    assert stats.structures > 0
    assert stats.jobs == stats.structures
    # Every job in this campaign exited cleanly; the interesting split is
    # convergence, not process state.
    assert set(stats.jobs_by_state) == {"done"}
    assert stats.converged + stats.hit_step_limit == stats.jobs
    assert stats.hit_step_limit > 0, "expected some runs to have hit NSW"
