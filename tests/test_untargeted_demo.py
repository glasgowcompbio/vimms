
from demo.untargeted.generate_chemicals import generate_chemicals
from demo.untargeted.generate_dataset import (
    create_design,
    generate_mzml_files,
    generate_ground_truth_table,
    write_ground_truth_mgf,
    setup_simulation,
)


def test_generate_chemicals_length():
    chems = generate_chemicals(10)
    assert len(chems) == 10
    # ensure each chemical has expected attributes
    for chem in chems:
        assert hasattr(chem, 'mass')
        assert hasattr(chem, 'rt')


def test_create_design():
    design = create_design(3)
    assert design.samples['case'] == ['case_1', 'case_2', 'case_3']
    assert len(design.samples['control']) == 3


def test_setup_simulation_returns_data():
    chems, design = setup_simulation(20, 2)
    assert len(chems) == 20
    assert len(design.samples['case']) == 2


def test_generate_mzml_files(tmp_path):
    chems, design = setup_simulation(3, 1)
    generate_mzml_files(chems, design, tmp_path, max_rt=20, top_n=1)
    assert (tmp_path / 'case' / 'case_1.mzML').is_file()


def test_generate_ground_truth_table():
    chems, design = setup_simulation(2, 1)
    df = generate_ground_truth_table(chems, design)
    assert {'sample', 'compound_id', 'mz_min', 'mz_max'} <= set(df.columns)
    assert len(df) == 4  # 2 samples (case/control) * 2 chemicals


def test_write_ground_truth_mgf(tmp_path):
    chems = generate_chemicals(2)
    out_file = tmp_path / "lib.mgf"
    write_ground_truth_mgf(chems, out_file)
    text = out_file.read_text()
    assert text.startswith("BEGIN IONS")
