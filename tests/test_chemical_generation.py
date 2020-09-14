# test chemical generaion
import os
from pathlib import Path

import numpy as np
import pytest
from mass_spec_utils.library_matching.gnps import load_mgf

from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, DatabaseFormulaSampler, UniformMZFormulaSampler, \
    CRPMS2Sampler, MGFMS2Sampler, MZMLMS2Sampler, ExactMatchMS2Sampler, MZMLRTandIntensitySampler, MZMLFormulaSampler, \
    MZMLChromatogramSampler
from vimms.Chemicals import ChemicalMixtureCreator, MultipleMixtureCreator, ChemicalMixtureFromMZML
from vimms.Common import load_obj, ADDUCT_DICT_POS_MH
from vimms.Noise import NoPeakNoise
from vimms.Roi import RoiParams
from vimms.Utils import write_msp, mgf_to_database

np.random.seed(1)

### define some useful constants ###

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(Path(DIR_PATH, 'fixtures'))
HMDB = load_obj(Path(BASE_DIR, 'hmdb_compounds.p'))
OUT_DIR = Path(DIR_PATH, 'results')

MZ_RANGE = [(300, 600)]
RT_RANGE = [(300, 500)]

N_CHEMICALS = 10

MGF_FILE = Path(BASE_DIR, 'small_mgf.mgf')
MZML_FILE = Path(BASE_DIR, 'small_mzml.mzML')


@pytest.fixture(scope="module")
def simple_dataset():
    ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
    hf = DatabaseFormulaSampler(HMDB)
    cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, adduct_prior_dict=ADDUCT_DICT_POS_MH)
    d = cc.sample(N_CHEMICALS, 2)
    return d


@pytest.fixture(scope="module")
def simple_no_database_dataset():
    ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
    hf = UniformMZFormulaSampler()
    cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, adduct_prior_dict=ADDUCT_DICT_POS_MH)
    d = cc.sample(N_CHEMICALS, 2)
    return d


def check_chems(chem_list):
    assert len(chem_list) == N_CHEMICALS
    for chem in chem_list:
        assert chem.mass >= MZ_RANGE[0][0]
        assert chem.mass <= MZ_RANGE[0][1]
        assert chem.rt >= RT_RANGE[0][0]
        assert chem.rt <= RT_RANGE[0][1]


class TestDatabaseCreation:
    def test_hmdb_creation(self):

        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri)
        d = cc.sample(N_CHEMICALS, 2)

        check_chems(d)

    def test_mz_creation(self):
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        hf = UniformMZFormulaSampler(min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri)
        d = cc.sample(N_CHEMICALS, 2)

        check_chems(d)

    def test_ms2_uniform(self):

        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cs = CRPMS2Sampler()
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, ms2_sampler=cs)
        d = cc.sample(N_CHEMICALS, 2)
        check_chems(d)

    def test_ms2_mgf(self):
        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cs = MGFMS2Sampler(MGF_FILE)
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, ms2_sampler=cs)
        d = cc.sample(N_CHEMICALS, 2)
        check_chems(d)

    def test_multiple_chems(self):
        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri)
        d = cc.sample(N_CHEMICALS, 2)

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

        group_dict = {'case': {'missing_probability': 1., 'changing_probability': 0}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i, c in enumerate(cl):
            if group_list[i] == 'case':
                assert len(c) == 0

        # test the case that if the missing probability is 1 all are missing
        group_dict = {'case': {'missing_probability': 1., 'changing_probability': 0}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i, c in enumerate(cl):
            if group_list[i] == 'case':
                assert len(c) == 0

        # test the case that changing probablity is 1 changes everything
        group_dict = {'case': {'missing_probability': 0., 'changing_probability': 1.}}

        mm = MultipleMixtureCreator(d, group_list, group_dict, intensity_noise=peak_noise)

        cl = mm.generate_chemical_lists()
        for i, c in enumerate(cl):
            if group_list[i] == 'case':
                for f in c:
                    assert not f.max_intensity == f.original_chemical.max_intensity


class TestMS2Sampling:
    def test_mzml_ms2(self):
        min_n_peaks = 50
        ms = MZMLMS2Sampler(MZML_FILE, min_n_peaks=min_n_peaks)
        ud = UniformMZFormulaSampler()
        cm = ChemicalMixtureCreator(ud, ms2_sampler=ms)
        d = cm.sample(N_CHEMICALS, 2)

        for chem in d:
            assert len(chem.children) >= min_n_peaks


class TestMSPWriting:

    def test_msp_writer_known_formula(self, simple_dataset):
        out_file = 'simple_known_dataset.msp'
        write_msp(simple_dataset, out_file, out_dir=OUT_DIR)
        assert (os.path.exists(Path(OUT_DIR, out_file)))

    def test_msp_writer_unknown_formula(self, simple_no_database_dataset):
        out_file = 'simple_unknown_dataset.msp'
        write_msp(simple_no_database_dataset, out_file, out_dir=OUT_DIR)
        assert (os.path.exists(Path(OUT_DIR, out_file)))


class TestLinkedCreation:

    def test_linked_ms1_ms2_creation(self):
        # make a database from an mgf
        database = mgf_to_database(MGF_FILE, id_field="SPECTRUMID")
        hd = DatabaseFormulaSampler(database)
        # ExactMatchMS2Sampler needs to be given the same mgf file
        # and both need to use the same field in the MGF as the unique ID
        mm = ExactMatchMS2Sampler(MGF_FILE, id_field="SPECTRUMID")
        cm = ChemicalMixtureCreator(hd, ms2_sampler=mm)
        dataset = cm.sample(N_CHEMICALS, 2)

        # check each chemical to see if it has the correct number of peaks
        records = load_mgf(MGF_FILE, id_field="SPECTRUMID")
        for chem in dataset:
            orig_spec = records[chem.database_accession]
            assert len(chem.children) > 0
            assert len(orig_spec.peaks) == len(chem.children)


class TestChemicalsFromMZML():
    def test_chemical_mixture_from_mzml(self):
        roi_params = RoiParams(min_intensity=10, min_length=5)
        cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
        d = cm.sample(None, 2)

    def test_rt_from_mzml(self):
        ri = MZMLRTandIntensitySampler(MZML_FILE)
        fs = MZMLFormulaSampler(MZML_FILE)
        cs = MZMLChromatogramSampler(MZML_FILE)
        cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        cm.sample(100, 2)
