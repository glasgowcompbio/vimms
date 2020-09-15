# some other more general tests

from vimms.Chemicals import Formula


class TestFormula:
    def test_atom_counts(self):
        formula_string = 'Cl'
        f = Formula(formula_string)
        assert f.atoms['Cl'] == 1
        assert f.atoms['C'] == 0

        formula_string = 'C6C7H2O'
        f = Formula(formula_string)
        assert f.atoms['C'] == 13
        assert f.atoms['H'] == 2
        assert f.atoms['O'] == 1
