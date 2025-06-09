from pathlib import Path

from vimms.ChemicalSamplers import DatabaseFormulaSampler, UniformRTAndIntensitySampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import ADDUCT_DICT_POS_MH, load_obj, save_obj


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def generate_chemicals(n_chemicals: int = 100):
    """Return a list of simulated chemicals for testing."""
    hmdb = load_obj(FIXTURES / "hmdb_compounds.p")
    fs = DatabaseFormulaSampler(hmdb, min_mz=100, max_mz=1000)
    ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=180)
    cmc = ChemicalMixtureCreator(
        fs, rt_and_intensity_sampler=ri, adduct_prior_dict=ADDUCT_DICT_POS_MH
    )
    return cmc.sample(n_chemicals, 2)


def main():
    chemicals = generate_chemicals()
    save_obj(chemicals, str(FIXTURES / "demo_chemicals.p"))


if __name__ == "__main__":
    main()
