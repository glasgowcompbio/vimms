# test chemical generation

import os
from pathlib import Path

import pytest
from mass_spec_utils.library_matching.gnps import load_mgf

from tests.conftest import HMDB, MGF_FILE, MZML_FILE, OUT_DIR, check_mzML, check_non_empty_MS2
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, DatabaseFormulaSampler, \
    UniformMZFormulaSampler, CRPMS2Sampler, MGFMS2Sampler, MZMLMS2Sampler, ExactMatchMS2Sampler, \
    MZMLRTandIntensitySampler, MZMLFormulaSampler, MZMLChromatogramSampler, MzMLScanTimeSampler
from vimms.Chemicals import ChemicalMixtureCreator, MultipleMixtureCreator, ChemicalMixtureFromMZML
from vimms.Common import ADDUCT_DICT_POS_MH, POSITIVE, set_log_level_warning, \
    DEFAULT_SCAN_TIME_DICT
from vimms.Controller import SimpleMs1Controller, TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import NoPeakNoise
from vimms.Roi import RoiBuilderParams
from vimms.Utils import write_msp, mgf_to_database

# define some useful constants

MZ_RANGE = [(300, 600)]
RT_RANGE = [(300, 500)]
N_CHEMS = 10


@pytest.fixture(scope="module")
def simple_dataset():
    ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
    hf = DatabaseFormulaSampler(HMDB)
    cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri,
                                adduct_prior_dict=ADDUCT_DICT_POS_MH)
    d = cc.sample(N_CHEMS, 2)
    return d


@pytest.fixture(scope="module")
def simple_no_database_dataset():
    ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
    hf = UniformMZFormulaSampler()
    cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri,
                                adduct_prior_dict=ADDUCT_DICT_POS_MH)
    d = cc.sample(N_CHEMS, 2)
    return d


def check_chems(chem_list):
    assert len(chem_list) == N_CHEMS
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
        d = cc.sample(N_CHEMS, 2)

        check_chems(d)

    def test_mz_creation(self):
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        hf = UniformMZFormulaSampler(min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri)
        d = cc.sample(N_CHEMS, 2)

        check_chems(d)

    def test_ms2_uniform(self):

        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cs = CRPMS2Sampler()
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, ms2_sampler=cs)
        d = cc.sample(N_CHEMS, 2)
        check_chems(d)

    def test_ms2_mgf(self):
        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cs = MGFMS2Sampler(MGF_FILE)
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri, ms2_sampler=cs)
        d = cc.sample(N_CHEMS, 2)
        check_chems(d)

    def test_multiple_chems(self):
        hf = DatabaseFormulaSampler(HMDB, min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
        ri = UniformRTAndIntensitySampler(min_rt=RT_RANGE[0][0], max_rt=RT_RANGE[0][1])
        cc = ChemicalMixtureCreator(hf, rt_and_intensity_sampler=ri)
        d = cc.sample(N_CHEMS, 2)

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
            originals = [f.base_chemical for f in c]
            assert len(set(originals)) == len(d)
            for f in c:
                assert f.max_intensity == f.base_chemical.max_intensity

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
                    assert not f.max_intensity == f.base_chemical.max_intensity


class TestMS2Sampling:
    def test_mzml_ms2(self):
        min_n_peaks = 50
        ms = MZMLMS2Sampler(MZML_FILE, min_n_peaks=min_n_peaks)
        ud = UniformMZFormulaSampler()
        cm = ChemicalMixtureCreator(ud, ms2_sampler=ms)
        d = cm.sample(N_CHEMS, 2)

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
        dataset = cm.sample(N_CHEMS, 2)

        # check each chemical to see if it has the correct number of peaks
        records = load_mgf(MGF_FILE, id_field="SPECTRUMID")
        for chem in dataset:
            orig_spec = records[chem.database_accession]
            assert len(chem.children) > 0
            assert len(orig_spec.peaks) == len(chem.children)


class TestChemicalsFromMZML():
    def test_chemical_mixture_from_mzml(self):
        roi_params = RoiBuilderParams(min_roi_intensity=10, min_roi_length=5)
        cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
        d = cm.sample(None, 2)
        assert len(d) == len(cm.good_rois)

    def test_rt_from_mzml(self):
        ri = MZMLRTandIntensitySampler(MZML_FILE)
        fs = MZMLFormulaSampler(MZML_FILE)
        cs = MZMLChromatogramSampler(MZML_FILE)
        cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        cm.sample(100, 2)

    def test_fullscan_from_mzml(self, chems_from_mzml):
        ionisation_mode = POSITIVE
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(ionisation_mode, chems_from_mzml)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'fullscan_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_topn_from_mzml(self, chems_from_mzml):
        ionisation_mode = POSITIVE
        N = 10
        isolation_width = 0.7
        mz_tol = 0.01
        rt_tol = 15
        min_ms1_intensity = 10
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    min_ms1_intensity)
        ms = IndependentMassSpectrometer(ionisation_mode, chems_from_mzml)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        check_non_empty_MS2(controller)
        filename = 'topn_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_mz_rt_i_from_mzml(self, chem_mz_rt_i_from_mzml):
        ionisation_mode = POSITIVE
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(ionisation_mode, chem_mz_rt_i_from_mzml)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'fullscan_mz_rt_i_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestScanTiming():
    def test_default_scan_time(self, chems_from_mzml):
        ionisation_mode = POSITIVE
        N = 10
        isolation_width = 0.7
        mz_tol = 0.01
        rt_tol = 15
        min_ms1_intensity = 10
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    min_ms1_intensity)

        # run simulation using default scan times
        ms = IndependentMassSpectrometer(ionisation_mode, chems_from_mzml,
                                         scan_duration=DEFAULT_SCAN_TIME_DICT)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'test_scan_time_default.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_scan_time_from_mzml(self):
        ionisation_mode = POSITIVE
        N = 10
        isolation_width = 0.7
        mz_tol = 0.01
        rt_tol = 15
        min_ms1_intensity = 10
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    min_ms1_intensity)

        # extract chemicals from mzML
        roi_params = RoiBuilderParams(min_roi_intensity=10, min_roi_length=5)
        cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
        chems = cm.sample(None, 2)

        # extract timing from mzML and sample one value each time when generating a scan duration
        sd = MzMLScanTimeSampler(MZML_FILE, use_mean=False)
        ms = IndependentMassSpectrometer(ionisation_mode, chems, scan_duration=sd)

        # run simulation
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'test_scan_time_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_mean_scan_time_from_mzml(self):
        ionisation_mode = POSITIVE
        N = 10
        isolation_width = 0.7
        mz_tol = 0.01
        rt_tol = 15
        min_ms1_intensity = 10
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    min_ms1_intensity)

        # extract chemicals from mzML
        roi_params = RoiBuilderParams(min_roi_intensity=10, min_roi_length=5)
        cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
        chems = cm.sample(None, 2)

        # extract mean timing per scan level from mzML
        sd = MzMLScanTimeSampler(MZML_FILE, use_mean=True)
        ms = IndependentMassSpectrometer(ionisation_mode, chems, scan_duration=sd)

        # run simulation
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'test_scan_time_mean_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)
