"""Formula parsing, reduction and canonicalisation.

The canonical formula is a database key -- `composition` is unique on
`(formula, z, source_name)` -- so these tests are guarding an invariant, not a
formatting preference: two spellings of one composition must not become two
rows.
"""

import pytest

from cspflow import chem


class TestParse:
    def test_explicit_counts(self):
        assert chem.parse_formula("Gd1Co10Cr2") == {"Gd": 1, "Co": 10, "Cr": 2}

    def test_implicit_one(self):
        assert chem.parse_formula("SmFe11Ti") == {"Sm": 1, "Fe": 11, "Ti": 1}

    def test_single_element(self):
        assert chem.parse_formula("Fe") == {"Fe": 1}

    def test_parentheses(self):
        assert chem.parse_formula("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}

    def test_nested_parentheses(self):
        assert chem.parse_formula("K2(Mg(SO4)2)3") == {"K": 2, "Mg": 3, "S": 6, "O": 24}

    def test_repeated_element_accumulates(self):
        assert chem.parse_formula("FeO Fe2O3") == {"Fe": 3, "O": 4}

    @pytest.mark.parametrize(
        "bad, fragment",
        [
            ("Xx2O3", "not an element"),
            ("fe2", "capital letter"),
            ("Ca(PO4", "unbalanced"),
            ("2Fe", "capital letter"),
            ("", "no elements"),
        ],
    )
    def test_rejects(self, bad, fragment):
        with pytest.raises(chem.ChemError, match=fragment):
            chem.parse_formula(bad)

    def test_unknown_element_can_be_allowed(self):
        """`validate=False` exists for parsing keys, not for accepting bad input."""
        assert chem.parse_formula("Xx2", validate=False) == {"Xx": 2}


class TestReduce:
    def test_reduces_and_reports_z(self):
        assert chem.reduce_counts({"Fe": 2, "Co": 10}) == ({"Fe": 1, "Co": 5}, 2)

    def test_already_reduced(self):
        assert chem.reduce_counts({"Fe": 1, "Co": 5}) == ({"Fe": 1, "Co": 5}, 1)

    def test_coprime_stays(self):
        assert chem.reduce_counts({"Fe": 2, "Co": 3}) == ({"Fe": 2, "Co": 3}, 1)

    def test_rejects_empty(self):
        with pytest.raises(chem.ChemError):
            chem.reduce_counts({})

    def test_rejects_non_positive(self):
        with pytest.raises(chem.ChemError, match="non-positive"):
            chem.reduce_counts({"Fe": 0, "Co": 1})


class TestCanonical:
    def test_alphabetical_with_explicit_ones(self):
        assert chem.canonical_formula({"Gd": 1, "Co": 10}) == "Co10Gd1"

    def test_spelling_independent(self):
        """The point of the whole module: input order must not reach the key."""
        a = chem.canonical_formula({"Gd": 1, "Co": 10, "Cr": 2})
        b = chem.canonical_formula({"Cr": 2, "Co": 10, "Gd": 1})
        assert a == b == "Co10Cr2Gd1"

    def test_round_trips(self):
        counts = {"Sm": 1, "Fe": 11, "Ti": 1}
        assert chem.parse_formula(chem.canonical_formula(counts)) == counts

    def test_explicit_one_is_not_cosmetic(self):
        """`Co10Gd` and `Co10Gd1` must not both be possible keys."""
        assert chem.canonical_formula({"Co": 10, "Gd": 1}).endswith("Gd1")


class TestChemsys:
    def test_sorted_and_joined(self):
        assert chem.chemsys({"Gd": 1, "Co": 10}) == "Co-Gd"

    def test_accepts_an_element_iterable(self):
        assert chem.chemsys(["Fe", "Co", "Fe"]) == "Co-Fe"


class TestRareEarth:
    def test_counts_species_not_atoms(self):
        assert chem.n_rare_earth({"Gd": 4, "Fe": 1}) == 1

    def test_two_species(self):
        assert chem.n_rare_earth({"Sm": 1, "Tb": 1, "Fe": 10}) == 2

    def test_excludes_scandium_and_yttrium(self):
        """Group-3 metals with no f electrons: neither guard nor 4f applies."""
        assert chem.n_rare_earth({"Y": 2, "Sc": 1, "Fe": 14}) == 0
        assert "Y" not in chem.RARE_EARTHS and "Sc" not in chem.RARE_EARTHS

    def test_all_fourteen_plus_lanthanum(self):
        assert len(chem.RARE_EARTHS) == 15


class TestFormulaIdentity:
    def test_directory_name_and_generated_name_agree(self):
        """A legacy `Fe2Co10` directory and a generated `Co5Fe1` are one row."""
        legacy = chem.formula_identity("Fe2Co10")
        generated = chem.formula_identity("Co5Fe1")
        assert legacy[0] == generated[0] == "Co5Fe1"
        assert legacy[3] == 2 and generated[3] == 1

    def test_returns_chemsys_and_counts(self):
        formula, system, counts, z = chem.formula_identity("Gd1Co10Cr2")
        assert (formula, system, z) == ("Co10Cr2Gd1", "Co-Cr-Gd", 1)
        assert counts == {"Gd": 1, "Co": 10, "Cr": 2}
