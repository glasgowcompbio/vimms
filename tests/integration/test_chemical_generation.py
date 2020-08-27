# test chemical generaion
import unittest
from pathlib import Path

import numpy as np
import pytest
from vimms.Common import *
from vimms.ChemicalSamplers import *
from vimms.Chemicals import ChemicalMixtureCreator

np.random.seed(1)

### define some useful constants ###

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(Path(DIR_PATH, 'fixtures'))
HMDB = load_obj(Path(BASE_DIR, 'hmdb_compounds.p'))
OUT_DIR = Path(DIR_PATH, 'results')

MZ_RANGE = [(300,600)]
RT_RANGE = [(300,500)]

N_CHEMICALS = 10

MGF_FILE = Path(BASE_DIR, 'small_mgf.mgf')

def check_chems(chem_list):
    assert len(chem_list) == N_CHEMICALS
    for chem in chem_list:
        assert chem.mass >= MZ_RANGE[0][0]
        assert chem.mass <= MZ_RANGE[0][1]
        assert chem.rt >= RT_RANGE[0][0]
        assert chem.rt <= RT_RANGE[0][1]


class TestDatabaseCreation:
    def test_hmdb_creation(self):
        
        
        hf = DatabaseFormulaSampler(HMDB)
        cc = ChemicalMixtureCreator(hf)
        d = cc.sample(MZ_RANGE, RT_RANGE, N_CHEMICALS, 2)

        check_chems(d)

    def test_mz_creation(self):

        hf = UniformMZFormulaSampler()
        cc = ChemicalMixtureCreator(hf)
        d = cc.sample(MZ_RANGE, RT_RANGE, N_CHEMICALS, 2)

        check_chems(d)


    def test_ms2_uniform(self):
        
        hf = DatabaseFormulaSampler(HMDB)
        cs = CRPMS2Sampler()
        cc = ChemicalMixtureCreator(hf, ms2_sampler=cs)
        d = cc.sample(MZ_RANGE, RT_RANGE, N_CHEMICALS, 2)
        check_chems(d)

    def test_ms2_mgf(self):
        hf = DatabaseFormulaSampler(HMDB)
        cs = MGFMS2Sampler(MGF_FILE)
        cc = ChemicalMixtureCreator(hf, ms2_sampler=cs)
        d = cc.sample(MZ_RANGE, RT_RANGE, N_CHEMICALS, 2)
        check_chems(d)
