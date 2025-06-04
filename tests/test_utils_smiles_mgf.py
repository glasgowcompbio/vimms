import tempfile

import vimms.Utils as U


def test_packline_appends():
    out = []
    returned = U.packline(out, 'hi')
    assert returned is out
    assert out == ['hi\n']
    U.packline(out, 'there')
    assert out == ['hi\n', 'there\n']


def test_smiles_to_formula():
    assert U.smiles_to_formula('O') == 'H2O'
    assert U.smiles_to_formula('CCO') == 'C2H6O'
    assert U.smiles_to_formula('C1/') is None


def test_mgf_to_database(tmp_path):
    mgf_text = """BEGIN IONS
SPECTRUMID=1
SMILES=O
PEPMASS=100
100 1
END IONS
BEGIN IONS
SPECTRUMID=2
SMILES=C1/
PEPMASS=200
110 5
END IONS
"""
    mgf_file = tmp_path / 'file.mgf'
    mgf_file.write_text(mgf_text)
    db = U.mgf_to_database(mgf_file)
    assert len(db) == 1
    d = db[0]
    assert d.name == '1'
    assert d.chemical_formula == 'H2O'

