# Example campaigns

Three complete campaigns, one per source mode. Each is a folder you can copy
and run; every block is populated with a realistic value and commented with
what else it could have been.

| | mode | the question it answers | input |
|---|---|---|---|
| [`1-chemical-space/`](1-chemical-space/) | `chemical_space` | "search this region of the periodic table" | element groups in the YAML |
| [`2-composition-list/`](2-composition-list/) | `composition_list` | "I know which formulas I want" | inline items **and** [`inputs/compositions.csv`](2-composition-list/inputs/compositions.csv) |
| [`3-structure-list/`](3-structure-list/) | `structure_list` | "I already have the structures" | five real POSCARs in [`inputs/seeds/`](3-structure-list/inputs/seeds/) |

Check any of them without running anything:

```bash
csp source --dry-run -c examples/1-chemical-space/campaign.yaml
```

```
1-chemical-space     3,384 compositions   36 chemical systems   165,456 structures
2-composition-list      34 compositions    5 chemical systems     1,394 structures
3-structure-list         0 compositions    1 chemical system          5 seeds
```

Those numbers are asserted by `tests/unit/test_examples.py`, so an example that
stops matching its own description fails the suite rather than misleading you.

## To use one

```bash
cp -r examples/2-composition-list my-campaign
cd my-campaign
$EDITOR campaign.yaml
csp doctor
```

The examples name a shipped machine profile (`machine: orion`) rather than
carrying a copy, because three copies of the same profile in one repository is
noise. A real campaign made with `csp init` gets its own editable
`machine.yaml` and `recipe.yaml` in the folder — that is the difference between
these read-me examples and a working campaign.

## What each one is really demonstrating

**1 — chemical_space** is the mode with a size multiplier hidden in it. `pick`
has no default for that reason, and `max_atoms_formula` is the dial that moves
the composition count fastest. Run `--dry-run` before you believe any sweep.

**2 — composition_list** shows both ways of giving the list at once. Inline
items are read *before* the file, and a formula in both warns and keeps the
first — so the file is the list you maintain and the inline items are the
exceptions you are making today. The example does this on purpose with
`SmFe11Ti` and warns about it when you run it.

**3 — structure_list** has no `generate:` block, which is what tells the driver
Stage 1 has nothing to do. Its seeds are five DFT-relaxed Sm-Fe structures from
a finished campaign — Sm₂Fe₁₇ (hR19), SmFe₁₂ (tI26), SmFe₅, SmFe₄, SmFe₃ — so
the parsing, composition-derivation and `max_atoms` gate all run against real
files. It also carries a commented second source that turns the campaign into a
seeds-versus-generated control group.

## The CSV format

`formula[,z_min,z_max,n_structures]`. A header row is optional, blank cells
inherit from `source.defaults`, and `#` comments must be on a line of their own
— a trailing comment after data is read as part of the last cell.
