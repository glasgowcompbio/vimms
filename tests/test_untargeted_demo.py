
from demo.untargeted.generate_chemicals import generate_chemicals


def test_generate_chemicals_length():
    chems = generate_chemicals(10)
    assert len(chems) == 10
    # ensure each chemical has expected attributes
    for chem in chems:
        assert hasattr(chem, 'mass')
        assert hasattr(chem, 'rt')
