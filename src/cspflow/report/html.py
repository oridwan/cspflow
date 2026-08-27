"""One self-contained HTML file.

Self-contained is a requirement, not a preference: the report is read from a
laptop over a mounted filesystem or emailed as an attachment, and a page that
fetches a stylesheet is a page that renders as unstyled text half the time. No
external requests, no build step, no JavaScript beyond the table sort.

The design rule is the same one that runs through the rest of the campaign: a
number is never shown without what it is. The two magnetisation columns are
adjacent and differently shaded, and the header says which one is computed.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..db.store import Store
from .candidates import COLUMNS, Funnel, candidate_rows, funnel

# Columns holding a modelled rather than a computed number. Shown, but marked.
MODELLED = {"m_s_reconstructed"}
# Columns whose value is only meaningful with a tolerance or a scale beside it.
QUALIFIED = {"spacegroup", "e_above_hull_mlip", "dft_e_above_hull"}

_CSS = """
:root { --fg:#1a1a1a; --bg:#fff; --line:#d8d8d8; --muted:#666;
        --model:#fff6e5; --head:#f4f4f6; --good:#1a7f37; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#16181c; --line:#333; --muted:#9aa0a6;
          --model:#3a2f1a; --head:#22252b; --good:#4ac26b; }
}
body { background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.5rem;
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.4rem; margin:0 0 .2rem; }
h2 { font-size:1.05rem; margin:2rem 0 .6rem; font-weight:600; }
.sub { color:var(--muted); margin:0 0 1.5rem; font-size:.9rem; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { padding:.32rem .5rem; border-bottom:1px solid var(--line);
        text-align:right; white-space:nowrap; }
th { background:var(--head); text-align:right; position:sticky; top:0;
     cursor:pointer; user-select:none; font-weight:600; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
td.modelled, th.modelled { background:var(--model); }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:6px; }
.note { color:var(--muted); font-size:.85rem; margin:.5rem 0 0; }
.funnel td:first-child { text-align:left; }
.kept { color:var(--good); }
dl { display:grid; grid-template-columns:max-content 1fr; gap:.15rem .9rem;
     font-size:.85rem; color:var(--muted); margin:.4rem 0 0; }
dt { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--fg); }
"""

_SORT = """
document.querySelectorAll('th[data-col]').forEach(function (th) {
  th.addEventListener('click', function () {
    var table = th.closest('table'), body = table.tBodies[0];
    var index = Array.prototype.indexOf.call(th.parentNode.children, th);
    var dir = th.dataset.dir === 'asc' ? -1 : 1;
    th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = a.cells[index].dataset.v, y = b.cells[index].dataset.v;
      var nx = parseFloat(x), ny = parseFloat(y);
      if (!isNaN(nx) && !isNaN(ny)) { return (nx - ny) * dir; }
      return String(x).localeCompare(String(y)) * dir;
    });
    rows.forEach(function (r) { body.appendChild(r); });
  });
});
"""


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "&mdash;"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return html.escape(str(value))


def _sort_value(value: Any) -> str:
    """What the sort compares. Missing sorts last under either direction."""
    if value is None or value == "":
        return "1e30"
    return str(value)


def render(store: Store, *, title: str = "cspflow", limit: int | None = 2000) -> str:
    rows = candidate_rows(store, limit=limit)
    counts = funnel(store)
    summary = store.summary()
    yield_ = store.generation_yield()
    return _page(title, summary, yield_, counts, rows)


def _page(title: str, summary: dict, yield_: dict, counts: Funnel,
          rows: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
             f"<h1>{html.escape(str(summary.get('campaign') or title))}</h1>",
             f"<p class=sub>{len(rows)} candidates &middot; "
             f"{summary['structures']:,} structures &middot; "
             f"{summary['compositions']:,} compositions &middot; "
             f"{summary['core_hours']:,.0f} core-hours &middot; {stamp}</p>"]

    if yield_["compositions"]:
        pct = 100.0 * yield_["produced"] / max(yield_["requested"], 1)
        parts.append(f"<h2>Generation</h2><p class=note>"
                     f"{yield_['produced']:,} structures produced of "
                     f"{yield_['requested']:,} requested ({pct:.1f}%); "
                     f"{yield_['short']} composition(s) short.</p>")

    parts.append("<h2>Funnel</h2><div class=scroll><table class=funnel><thead><tr>"
                 "<th>gate</th><th>seen</th><th>passed</th><th>rejected</th>"
                 "</tr></thead><tbody>")
    for gate, seen, passed in counts.rows:
        parts.append(f"<tr><td>{html.escape(gate)}</td><td>{seen:,}</td>"
                     f"<td class=kept>{passed:,}</td><td>{seen - passed:,}</td></tr>")
    parts.append("</tbody></table></div>")

    parts.append("<h2>Candidates</h2><div class=scroll><table><thead><tr>")
    for name, meaning in COLUMNS:
        css = " class=modelled" if name in MODELLED else ""
        parts.append(f'<th{css} data-col="{name}" title="{html.escape(meaning)}">'
                     f"{html.escape(name)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for name, _ in COLUMNS:
            value = row.get(name)
            css = " class=modelled" if name in MODELLED else ""
            parts.append(f'<td{css} data-v="{html.escape(_sort_value(value))}">'
                         f"{_cell(value)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")

    parts.append("<h2>Columns</h2><dl>")
    for name, meaning in COLUMNS:
        parts.append(f"<dt>{html.escape(name)}</dt><dd>{html.escape(meaning)}</dd>")
    parts.append("</dl>")
    parts.append("<p class=note>Shaded columns are modelled, not computed. "
                 "<code>m_dft_raw</code> is the cell magnetisation VASP reports; "
                 "<code>m_s_reconstructed</code> adds the Hund&rsquo;s-rule 4f "
                 "moment back to the transition-metal sublattice, which a "
                 "frozen-4f POTCAR leaves out. For a heavy rare earth the two "
                 "routinely differ by more than a factor of two and often in "
                 "sign. They are never merged.</p>")
    parts.append(f"<script>{_SORT}</script>")
    return "\n".join(parts)


def write(store: Store, path: Path, *, title: str = "cspflow",
          limit: int | None = 2000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(store, title=title, limit=limit))
    return path
