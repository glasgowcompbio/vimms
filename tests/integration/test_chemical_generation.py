# test chemical generaion
import unittest
from pathlib import Path

import numpy as np
import pytest
from vimms.Common import *
from vimms.ChemicalSamplers import *
from vimms.Chemicals import ChemicalMixtureCreator, MultipleMixtureCreator
from vimms.Noise import NoPeakNoise
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

    def test_multiple_chems(self):
        hf = DatabaseFormulaSampler(HMDB)
        cc = ChemicalMixtureCreator(hf)
        d = cc.sample(MZ_RANGE, RT_RANGE, N_CHEMICALS, 2)
        

        group_list = ['control', 'control', 'case', 'case']
        group_dict = {'case': {'missing_probability': 0,
                                'changing_probability': 0}}

        # missing noise
        peak_noise = NoPeakNoise()

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()

        for c in cl:
            check_chems(c)
            # with these settings all chemicals should be in all lists with identical intensities
            originals = [f.original_chemical for f in c]
            assert len(set(originals)) == len(d)
            for f in c:
                assert f.max_intensity == f.original_chemical.max_intensity
        
        
        group_dict = {'case': {'missing_probability': 1.,'changing_probability': 0}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i,c in enumerate(cl):
            if group_list[i] == 'case':
                assert len(c) == 0

        # test the case that if the missing probability is 1 all are missing
        group_dict = {'case': {'missing_probability': 1.,'changing_probability': 0}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i,c in enumerate(cl):
            if group_list[i] == 'case':
                assert len(c) == 0

        # test the case that changing probablity is 1 changes everything
        group_dict = {'case': {'missing_probability': 0.,'changing_probability': 1.}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i,c in enumerate(cl):
            if group_list[i] == 'case':
                for f in c:
                    assert not f.max_intensity == f.original_chemical.max_intensity
        
        