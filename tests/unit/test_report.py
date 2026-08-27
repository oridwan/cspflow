"""Stage 8 -- the candidate table and the dashboard.

The tests are mostly about labelling. A report is where a number stops being an
internal value and becomes something someone quotes, so the properties worth
asserting are that every column says what it is, that the modelled column is
marked as modelled, and that a missing value stays missing rather than becoming
a zero.
"""

import csv
import re
from pathlib import Path

import pytest
from ase import Atoms

from cspflow.db.store import Origin, Store, StructureState
from cspflow.report.candidates import COLUMNS, candidate_rows, funnel, write_csv
from cspflow.report.html import MODELLED, render, write


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "c.db", campaign="demo") as s:
        yield s


def add(store, state=StructureState.dft_done, **kv):
    atoms = Atoms("Fe2Sm", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
                  cell=[4, 4, 4], pbc=True)
    return store.add_structure(atoms, origin=Origin.generated, state=state, **kv)


# -- the table -------------------------------------------------------------

def test_the_two_magnetisations_are_separate_columns():
    names = [name for name, _ in COLUMNS]
    assert "m_dft_raw" in names and "m_s_reconstructed" in names
    assert "m_s" not in names
    # Adjacent, so a reader cannot see one without the other.
    assert names.index("m_s_reconstructed") == names.index("m_dft_raw") + 1


def test_every_column_carries_a_meaning():
    assert all(meaning.strip() for _, meaning in COLUMNS)


def test_rows_are_sorted_by_the_dft_hull_where_there_is_one(store):
    far = add(store, dft_e_above_hull=0.5)
    near = add(store, dft_e_above_hull=0.01)
    mlip_only = add(store, e_above_hull_mlip=0.001)
    rows = candidate_rows(store)
    assert [r["id"] for r in rows] == [near, far, mlip_only]


def test_a_missing_value_stays_missing(store):
    add(store)
    row = candidate_rows(store)[0]
    assert row["m_dft_raw"] is None
    assert row["dft_e_above_hull"] is None


def test_the_csv_names_every_column_in_its_header(tmp_path, store):
    add(store, m_dft_raw=4.0, m_s_reconstructed=-4.0, volume=64.0)
    path = write_csv(candidate_rows(store), tmp_path / "c.csv")
    text = path.read_text()
    assert "# m_dft_raw: mu_B per cell, computed" in text
    assert "MODELLED" in text
    body = [line for line in text.splitlines() if not line.startswith("#")]
    parsed = list(csv.DictReader(body))
    assert parsed[0]["m_dft_raw"] == "4.0"
    assert parsed[0]["m_s_reconstructed"] == "-4.0"


# -- the funnel ------------------------------------------------------------

def test_the_funnel_counts_come_from_the_recorded_events(store):
    sid = add(store, state=StructureState.screened)
    store.add_filter_event(structure_id=sid, gate="screen:converged", passed=True)
    store.add_filter_event(structure_id=sid, gate="filter:e_above_hull", passed=False)
    counts = {gate: (seen, passed) for gate, seen, passed in funnel(store).rows}
    assert counts["screen:converged"] == (1, 1)
    assert counts["filter:e_above_hull"] == (1, 0)


def test_the_funnel_lists_known_gates_in_the_order_they_run(store):
    sid = add(store)
    for gate in ("filter:e_above_hull", "screen:validate", "dedup"):
        store.add_filter_event(structure_id=sid, gate=gate, passed=True)
    assert [g for g, _, _ in funnel(store).rows] == [
        "screen:validate", "dedup", "filter:e_above_hull"]


# -- the page --------------------------------------------------------------

def test_the_page_is_self_contained(store, tmp_path):
    add(store, m_dft_raw=4.0, m_s_reconstructed=-4.0)
    page = render(store)
    assert "http://" not in page and "https://" not in page
    assert "<link" not in page and "src=" not in page


def test_the_modelled_column_is_marked_in_the_page(store):
    add(store, m_dft_raw=4.0, m_s_reconstructed=-4.0)
    page = render(store)
    assert "m_s_reconstructed" in MODELLED
    header = re.search(r"<th class=modelled data-col=\"m_s_reconstructed\"", page)
    assert header is not None
    assert "never merged" in page


def test_the_page_says_which_number_is_computed(store):
    add(store, m_dft_raw=4.0, m_s_reconstructed=-4.0)
    page = render(store)
    assert "cell magnetisation VASP reports" in page
    assert "frozen-4f POTCAR leaves out" in page


def test_a_missing_value_renders_as_a_dash_not_a_zero(store):
    add(store)
    assert "&mdash;" in render(store)


def test_the_page_is_written_where_asked(store, tmp_path):
    path = write(store, tmp_path / "sub" / "report.html", title="demo")
    assert path.is_file()
    assert "demo" in path.read_text()


def test_a_campaign_with_no_candidates_still_renders(store):
    page = render(store)
    assert "<table" in page
