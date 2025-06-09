
from demo.untargeted.generate_chemicals import generate_chemicals
from demo.untargeted.generate_dataset import create_design, setup_simulation


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
