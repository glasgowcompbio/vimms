import csv
import os
import unittest
from pathlib import Path
from loguru import logger

import numpy as np
import pytest

from vimms.ChemicalSamplers import UniformMZFormulaSampler, UniformRTAndIntensitySampler, GaussianChromatogramSampler, \
    EvenMZFormulaSampler, ConstantChromatogramSampler, MZMLFormulaSampler, MZMLRTandIntensitySampler, MZMLChromatogramSampler, \
        FixedMS2Sampler, DatabaseFormulaSampler
from vimms.Chemicals import ChemicalCreator, ChemicalMixtureCreator, ChemicalMixtureFromMZML
from vimms.Common import load_obj, set_log_level_warning, set_log_level_debug, GET_MS2_BY_PEAKS, ADDUCT_DICT_POS_MH, \
    GET_MS2_BY_SPECTRA, POSITIVE, NEGATIVE, DEFAULT_ISOLATION_WIDTH
from vimms.Controller import TopNController, PurityController, TopN_RoiController, AIF, \
    TopN_SmartRoiController, WeightedDEWController, FixedScansController, \
    SWATH, DiaController, AdvancedParams, TargetedController, Target, create_targets_from_toxid, \
    MultiIsolationController
from vimms.Controller.fullscan import SimpleMs1Controller
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer, ScanParameters
from vimms.Noise import GaussianPeakNoise, GaussianPeakNoiseLevelSpecific, UniformSpikeNoise
from vimms.Roi import RoiParams

np.random.seed(1)

### define some useful constants ###

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(Path(DIR_PATH, 'fixtures'))
HMDB = load_obj(Path(BASE_DIR, 'hmdb_compounds.p'))
OUT_DIR = Path(DIR_PATH, 'results')

ROI_SOURCES = [str(Path(BASE_DIR, 'beer_t10_simulator_files'))]
# MIN_MS1_INTENSITY = 1.75E5
MIN_MS1_INTENSITY = 1
RT_RANGE = [(0, 1200)]
CENTRE_RANGE = 600
MIN_RT = RT_RANGE[0][0]
MAX_RT = RT_RANGE[0][1]
MZ_RANGE = [(0, 1050)]
N_CHEMS = 10

BEER_CHEMS = load_obj(Path(BASE_DIR, 'QCB_22May19_1.p'))

# this is a temporary hack until beer_chems are updated
for b in BEER_CHEMS:
    b.adducts = {POSITIVE: b.adducts}

BEER_MIN_BOUND = 550
BEER_MAX_BOUND = 650

MZML_FILE = Path(BASE_DIR, 'small_mzml.mzML')


### define some useful methods ###

def get_rt_bounds(dataset, centre):
    rts = [ds.rt for ds in dataset]
    min_bound = max([rt for rt in rts if rt < centre], default=centre) - 60
    max_bound = min([rt for rt in rts if rt > centre], default=centre) + 60
    return (min_bound, max_bound)


def run_environment(env):
    # set the log level to WARNING so we don't see too many messages when environment is running
    set_log_level_warning()
    # run the simulation
    logger.info('Running simulation')
    env.run()
    logger.info('Done')
    # set the log level back to DEBUG
    set_log_level_debug()


def check_mzML(env, out_dir, filename):
    out_file = os.path.join(out_dir, filename)
    logger.info('Writing out mzML')
    env.write_mzML(out_dir, filename)
    logger.info('Done')
    assert os.path.exists(out_file)


def check_non_empty_MS1(controller):
    return check_non_empty(controller, 1)


def check_non_empty_MS2(controller):
    return check_non_empty(controller, 2)


def check_non_empty(controller, ms_level):
    non_empty = 0
    for scan in controller.scans[ms_level]:
        if scan.num_peaks > 0:
            non_empty += 1
        if scan.ms_level == 2:
            assert scan.scan_params is not None
            assert scan.scan_params.get(ScanParameters.PRECURSOR_MZ) is not None
    assert non_empty > 0


### define some useful test fixtures ###

@pytest.fixture(scope="module")
def fullscan_ps():
    return load_obj(Path(BASE_DIR, 'peak_sampler_mz_rt_int_beerqcb_fullscan.p'))


@pytest.fixture(scope="module")
def fullscan_dataset(fullscan_ps):
    chems = ChemicalCreator(fullscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 1)


@pytest.fixture(scope="module")
def fragscan_ps():
    return load_obj(Path(BASE_DIR, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))


@pytest.fixture(scope="module")
def fragscan_dataset_peaks(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 2,
                        get_children_method=GET_MS2_BY_PEAKS)


@pytest.fixture(scope="module")
def fragscan_dataset_peaks_onlyMH(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 1,
                        get_children_method=GET_MS2_BY_PEAKS, adduct_prior_dict=ADDUCT_DICT_POS_MH)


@pytest.fixture(scope="module")
def fragscan_dataset_spectra(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 2,
                        get_children_method=GET_MS2_BY_SPECTRA)


@pytest.fixture(scope="module")
def simple_dataset():
    um = UniformMZFormulaSampler(min_mz=515, max_mz=516)
    ri = UniformRTAndIntensitySampler(min_rt=150, max_rt=160)
    cs = GaussianChromatogramSampler(sigma=100)
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(1, 2)


@pytest.fixture(scope="module")
def ten_chems():
    um = UniformMZFormulaSampler(min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
    ri = UniformRTAndIntensitySampler(min_rt=200, max_rt=300)
    cs = GaussianChromatogramSampler()
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(10, 2)


@pytest.fixture(scope="module")
def two_fixed_chems():
    em = EvenMZFormulaSampler()
    ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
    cs = ConstantChromatogramSampler()
    cm = ChemicalMixtureCreator(em, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(2, 2)


@pytest.fixture(scope="module")
def even_chems():
    # four evenly spaced chems for more advanced SWATH testing
    em = EvenMZFormulaSampler()
    ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
    cs = ConstantChromatogramSampler()
    cm = ChemicalMixtureCreator(em, rt_and_intensity_sampler=ri, chromatogram_sampler=cs,
                                adduct_prior_dict=ADDUCT_DICT_POS_MH)
    return cm.sample(4, 2)


@pytest.fixture(scope="module")
def chems_from_mzml():
    roi_params = RoiParams(min_intensity=10, min_length=5)
    cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
    return cm.sample(None, 2)


@pytest.fixture(scope="module")
def chem_mz_rt_i_from_mzml():
    fs = MZMLFormulaSampler(MZML_FILE)
    ri = MZMLRTandIntensitySampler(MZML_FILE)
    cs = MZMLChromatogramSampler(MZML_FILE)
    cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(500, 2)


### tests starts from here ###

class TestIonisationMode:
    def test_positive_fixed(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler()
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        dataset = cm.sample(3,2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(POSITIVE, dataset, None)
        controller = TopNController(POSITIVE, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        env =  Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        ms1_mz_vals = [int(m) for m in controller.scans[1][0].mzs]
        
        expected_vals = [101,201,301]
        for i,m in  enumerate(ms1_mz_vals):
            assert m == expected_vals[i]

        expected_frags = set([81,91,181,191, 281, 291])
        for scan in controller.scans[2]:
            for m in scan.mzs:
                assert int(m) in expected_frags

    def test_negative_fixed(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler()
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        dataset = cm.sample(3,2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(NEGATIVE, dataset, None)
        controller = TopNController(NEGATIVE, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        env =  Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        ms1_mz_vals = [int(m) for m in controller.scans[1][0].mzs]
        
        expected_vals = [98,198,298]
        for i,m in  enumerate(ms1_mz_vals):
            assert m == expected_vals[i]
        
        expected_frags = set([88,78,188,178, 288, 278])
        for scan in controller.scans[2]:
            for m in scan.mzs:
                assert int(m) in expected_frags

    def test_multiple_adducts(self):
        fs = DatabaseFormulaSampler(HMDB)
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        adduct_prior_dict = {POSITIVE: {'M+H': 100, 'M+Na': 100, 'M+K': 100}}
        cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs, adduct_prior_dict=adduct_prior_dict)

        n_adducts = len(adduct_prior_dict[POSITIVE])
        n_chems = 5
        dataset = cm.sample(n_chems,2)
        

        for c in dataset:
            c.isotopes = [(c.mass, 1, "Mono")]

        # should be 15 peaks or less all the time
        # some adducts might not be sampled if the probability is less than 0.2
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(POSITIVE, dataset, None)
        env =  Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        for scan in controller.scans[1]:
            assert len(scan.mzs) <= n_chems * n_adducts


class TestMS1Controller:
    """
    Tests the MS1 controller that does MS1 full-scans only with the simulated mass spec class.
    """

    def test_ms1_controller_with_simulated_chems(self, fragscan_dataset_peaks, fullscan_ps):
        logger.info('Testing MS1 controller with simulated chemicals')

        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        logger.info('RT bounds %s %s' % (min_bound, max_bound))
        assert len(fragscan_dataset_peaks) == N_CHEMS

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, fragscan_dataset_peaks, fullscan_ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'ms1_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_ms1_controller_with_qcbeer_chems(self, fullscan_ps):
        logger.info('Testing MS1 controller with QC beer chemicals')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS, fullscan_ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'ms1_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_peaks_in_range(self, fragscan_dataset_peaks, fullscan_ps):

        min_mz = 100.
        max_mz = 200.

        logger.info('Testing MS1 controller with narrow m/z range')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS, fullscan_ps)
        params = AdvancedParams()
        params.default_ms1_scan_window = (min_mz, max_mz)
        controller = SimpleMs1Controller(params=params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'ms1_controller_qcbeer_chems_narrow.mzML'
        check_mzML(env, OUT_DIR, filename)


        for scan_level, scans in controller.scans.items():
            for s in scans:
                assert min(s.mzs) >= min_mz
                assert max(s.mzs) <= max_mz



class TestTopNControllerSpectra:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    Fragment spectra are generated via the "spectra" method
    """

    def test_TopN_controller_with_simulated_chems(self, fragscan_dataset_spectra, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals -- no noise')
        assert len(fragscan_dataset_spectra) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_spectra, fragscan_ps)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_spectra, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        filename = 'topN_controller_simulated_chems_no_noise_spectra.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestTopNController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    def test_TopN_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals -- no noise')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        filename = 'topN_controller_simulated_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_simulated_chems_and_noise(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals -- with noise')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and Top-N controller
        mz_noise = GaussianPeakNoise(0.1)
        intensity_noise = GaussianPeakNoise(1000.)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, mz_noise=mz_noise,
                                                intensity_noise=intensity_noise)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_simulated_chems_with_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_beer_chems_and_scan_duration_dict(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals passing in the scan durations')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        ps = None
        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller and passing in the scan_duration dict
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, ps, scan_duration_dict=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise_with_scan_duration.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_simulated_chems_onlyMH(self, fragscan_dataset_peaks_onlyMH, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals -- no noise, only MH adducts')
        assert len(fragscan_dataset_peaks_onlyMH) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks_onlyMH, fragscan_ps)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks_onlyMH, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'topN_controller_simulated_chems_no_noise_onlyMH.mzML'
        check_mzML(env, OUT_DIR, filename)

class TestExclusion:
    def test_TopN_controller_with_beer_chems_and_initial_exclusion_list(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals and an initial exclusion list')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        initial_exclusion_list = None
        for i in range(3):
            mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
            controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY,
                                        initial_exclusion_list=initial_exclusion_list)
            print('exclude = %d' % len(controller.exclusion_list))
            env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
            run_environment(env)

            if initial_exclusion_list is None:
                initial_exclusion_list = []
            initial_exclusion_list = initial_exclusion_list + controller.all_exclusion_items

            # check that there is at least one non-empty MS2 scan
            check_non_empty_MS2(controller)

            # write simulated output to mzML file
            filename = 'topN_controller_qcbeer_exclusion_%d.mzML' % i
            check_mzML(env, OUT_DIR, filename)

    def test_exclusion_simple_data(self):
        # three chemicals, both will get fragmented 
        # first time around and exclusion such  that neither
        # should be fragmented second time
        fs = EvenMZFormulaSampler()
        ch = ConstantChromatogramSampler()
        rti = UniformRTAndIntensitySampler(min_rt=0, max_rt=5)
        cs = ChemicalMixtureCreator(fs, chromatogram_sampler=ch, rt_and_intensity_sampler=rti)
        n_chems = 3
        dataset = cs.sample(n_chems, 2)
        ionisation_mode = POSITIVE
        initial_exclusion_list = None
        min_ms1_intensity = 0
        N = 10
        mz_tol = 10
        rt_tol = 30
        isolation_width = 1
        all_controllers = []
        for i in range(2):
            mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, None)
            controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                                        initial_exclusion_list=initial_exclusion_list)
            print('exclude = %d' % len(controller.exclusion_list))
            env = Environment(mass_spec, controller, 0, 20, progress_bar=True)
            run_environment(env)

            if initial_exclusion_list is None:
                initial_exclusion_list = []
            initial_exclusion_list = initial_exclusion_list + controller.all_exclusion_items
            all_controllers.append(controller)
        assert len(all_controllers[0].scans[2]) == n_chems
        assert len(all_controllers[1].scans[2]) == 0


class TestMultipleMS2Windows:
    def test_data_generation(self, two_fixed_chems):
        assert len(two_fixed_chems) == 2
        assert two_fixed_chems[0].mass == 100
        assert two_fixed_chems[1].mass == 200
        assert len(two_fixed_chems[0].children) > 0
        assert len(two_fixed_chems[1].children) > 0

    def test_acquisition(self, two_fixed_chems):
        mz_to_target = [chem.mass + 1.0 for chem in two_fixed_chems]
        schedule = []
        # env = Environment()
        isolation_width = DEFAULT_ISOLATION_WIDTH
        mz_tol = 0.1
        rt_tol = 15

        min_rt = 110
        max_rt = 112

        ionisation_mode = POSITIVE

        controller = FixedScansController()
        mass_spec = IndependentMassSpectrometer(ionisation_mode, two_fixed_chems, None)
        env = Environment(mass_spec, controller, min_rt, max_rt)

        ms1_scan = env.get_default_scan_params()
        ms2_scan_1 = env.get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width, mz_tol, rt_tol)
        ms2_scan_2 = env.get_dda_scan_param(mz_to_target[1], 0.0, None, isolation_width, mz_tol, rt_tol)
        ms2_scan_3 = env.get_dda_scan_param(mz_to_target, [0.0, 0.0], None, isolation_width, mz_tol, rt_tol)

        schedule = [ms1_scan, ms2_scan_1, ms2_scan_2, ms2_scan_3]
        controller.set_tasks(schedule)
        set_log_level_warning()
        env.run()
        assert len(controller.scans[2]) == 3

        n_peaks = []
        for scan in controller.scans[2]:
            n_peaks.append(scan.num_peaks)

        assert n_peaks[0] > 0
        assert n_peaks[1] > 0
        assert n_peaks[2] == n_peaks[0] + n_peaks[1]
        env.write_mzML(OUT_DIR, 'multi_windows.mzML')


class TestPurityController:
    """
    Tests the Purity controller that is used for purity experiments
    """

    def test_purity_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing purity controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        # set different isolation widths, Ns, dynamic exclusion RT and mz tolerances at different timepoints
        isolation_widths = [1, 1, 1, 1]
        N = [5, 10, 15, 20]
        rt_tol = [15, 30, 60, 120]
        mz_tol = [10, 5, 15, 20]
        scan_param_changepoints = [300, 600, 900]  # the timepoints when we will change the 4 parameters above
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and purity controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        controller = PurityController(ionisation_mode, N, scan_param_changepoints, isolation_widths, mz_tol, rt_tol,
                                      MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'purity_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_purity_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing purity controller with QC beer chemicals')

        isolation_window = [1]  # the isolation window in Dalton around a selected precursor ion
        N = [5]
        rt_tol = [10]
        mz_tol = [10]
        min_ms1_intensity = 1.75E5
        scan_param_changepoints = None
        n_purity_scans = N[0]
        purity_shift = 0.2
        purity_threshold = 1

        # these settings change the Mass Spec type. They arent necessary to run the Top-N ROI Controller
        isolation_transition_window = 'gaussian'
        isolation_transition_window_params = [0.5]

        purity_add_ms1 = True  # this seems to be the broken bit
        purity_randomise = True

        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS, fragscan_ps,
                                                isolation_transition_window=isolation_transition_window,
                                                isolation_transition_window_params=isolation_transition_window_params)
        controller = PurityController(mass_spec, N, scan_param_changepoints, isolation_window, mz_tol, rt_tol,
                                      min_ms1_intensity, n_purity_scans, purity_shift, purity_threshold,
                                      purity_add_ms1=purity_add_ms1, purity_randomise=purity_randomise)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'purity_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def test_roi_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        for f in fragscan_dataset_peaks:
            assert len(f.children) > 0

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 2
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'roi_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_roi_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing ROI controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'roi_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestSMARTROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def test_smart_roi_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                             min_roi_intensity, min_roi_length, N, rt_tol,
                                             min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'smart_roi_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_smart_roi_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing ROI controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                             min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'smart_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestTopNShiftedController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the beer chems.
    """

    def test_TopN_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals')
        test_shift = 0
        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps,
                                                scan_duration_dict=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY,
                                    ms1_shift=test_shift)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_shifted_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestTopNExcludingShiftedController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the beer chems.
    """

    def test_TopN_excluding_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing excluding Top-N controller with QC beer chemicals')
        test_shift = 0
        isolation_width = 1
        N = 10
        mz_tol = 10
        ionisation_mode = POSITIVE
        exclusion_t_0 = 15.0
        rt_tol = 120
        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps,
                                                scan_duration_dict=scan_duration_dict)
        controller = WeightedDEWController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY,
                                           ms1_shift=test_shift,
                                           exclusion_t_0=exclusion_t_0,
                                           log_intensity=True)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_excluding_shifted_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestAIFControllers:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    def test_AIF_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals')

        # create some chemical object
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        min_mz = 100
        max_mz = 500

        # shorten  the rt range for quicker tests
        min_rt = 0
        max_rt = 400

        scan_time_dict = {1: 0.12, 2: 0.06}

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, params=params)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_simulated_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

        # create a simulated mass spec with noise and Top-N controller
        logger.info('With noise')
        mz_noise = GaussianPeakNoiseLevelSpecific({2: 0.01})
        intensity_noise = GaussianPeakNoiseLevelSpecific({2: 1000.})
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps,
                                                scan_duration_dict=scan_time_dict, mz_noise=mz_noise,
                                                intensity_noise=intensity_noise)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, params=params)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_simulated_chems_with_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_AIF_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE
        min_mz = 100
        max_mz = 500

        min_rt = 0
        max_rt = 500

        # create a simulated mass spec without noise and Top-N controller
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, params=params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_aif_msdial_experiment_file(self):
        min_mz = 200
        max_mz = 300
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, params=params)
        out_file = Path(OUT_DIR, 'AIF_experiment.txt')
        controller.write_msdial_experiment_file(out_file)

        assert os.path.exists(out_file)
        with open(out_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t', lineterminator=os.linesep)
            rows = []
            for row in reader:
                rows.append(row)
        ce = params.ms1_source_cid_energy
        expected_row = ['1', 'ALL', min_mz, max_mz, "{}eV".format(ce), ce, 1]
        for i, val in enumerate(expected_row):
            assert rows[-1][i] == str(val)

    def test_aif_with_fixed_chems(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler(n_frags=2)
        cs = ConstantChromatogramSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=1)
        cs = ChemicalMixtureCreator(fs, ms2_sampler=ms, chromatogram_sampler=cs, rt_and_intensity_sampler=ri)
        d = cs.sample(1, 2)

        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy)
        ionisation_mode = POSITIVE
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d, None)
        env = Environment(mass_spec, controller, 10, 20, progress_bar=True)

        set_log_level_warning()
        env.run()

        for i, s in enumerate(controller.scans[1]):
            if i % 2 == 1:
                # odd scan, AIF, should  have two peaks at 81 and 91
                integer_mzs = [int(i) for i in s.mzs]
                integer_mzs.sort()
                assert integer_mzs[0] == 81
                assert integer_mzs[1] == 91
            else:
                # even scan, MS1 - should have a single peak at integer value of 101
                integer_mzs = [int(i) for i in s.mzs]
                assert integer_mzs[0] == 101


class TestSWATH:
    def test_swath(self, ten_chems):
        min_mz = 100
        max_mz = 1000
        width = 100
        scan_overlap = 10

        ionisation_mode = POSITIVE

        controller = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        scan_time_dict = {1: 0.124, 2: 0.124}

        spike_noise = UniformSpikeNoise(0.1, 1)

        mass_spec = IndependentMassSpectrometer(ionisation_mode, ten_chems, None, scan_duration_dict=scan_time_dict,
                                                spike_noise=spike_noise)

        env = Environment(mass_spec, controller, 200, 300, progress_bar=True)

        set_log_level_warning()

        env.run()

        check_non_empty_MS2(controller)

        filename = 'SWATH_ten_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_swath_more(self, even_chems):
        """
        Tests SWATH by making even chemicals and then 
        varying the SWATH window so that in the first example
        each chemical is in its own window, in the second each window holds two chems
        and in the third, one window holds them all
        """
        ionisation_mode = POSITIVE
        min_mz = 50
        max_mz = 460
        width = 100
        scan_overlap = 0
        controller = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems, None, scan_duration_dict=scan_time_dict)
        env = Environment(mass_spec, controller, 200, 300, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check the scans
        ms2_scans = controller.scans[2]
        for i in range(4):
            assert len(ms2_scans[i].mzs) == len(even_chems[i].children)

        width = 200
        controller2 = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems, None, scan_duration_dict=scan_time_dict)
        env = Environment(mass_spec, controller2, 200, 300, progress_bar=True)
        env.run()

        ms2_scans2 = controller2.scans[2]

        assert len(ms2_scans2[0].mzs) == len(even_chems[0].children) + len(even_chems[1].children)
        assert len(ms2_scans2[1].mzs) == len(even_chems[2].children) + len(even_chems[3].children)

        width = 400
        controller3 = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems, None, scan_duration_dict=scan_time_dict)
        env = Environment(mass_spec, controller3, 200, 300, progress_bar=True)
        env.run()

        ms2_scans3 = controller3.scans[2]
        assert len(ms2_scans3[0].mzs) == sum([len(c.children) for c in even_chems])
        assert len(ms2_scans3[0].mzs) == sum([len(s.mzs) for s in ms2_scans2[:2]])

    def test_swath_msdial_experiment_file(self):
        min_mz = 50
        max_mz = 440
        width = 100
        scan_overlap = 0
        controller = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        out_file = Path(OUT_DIR, 'swath_experiment.txt')
        controller.write_msdial_experiment_file(out_file)

        assert os.path.exists(out_file)
        with open(out_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t', lineterminator=os.linesep)
            rows = []
            for row in reader:
                rows.append(row)
        expected_row = ['4', 'SWATH', '350', '450']
        for i, val in enumerate(expected_row):
            assert rows[-1][i] == val


class TestDiaController:
    """
    Tests for the DiaController that implements the nested and tree DIA methods
    """

    def test_NestedDiaController_even(self, simple_dataset, fragscan_ps):
        logger.info('Testing NestedDiaController even')

        # some parameters
        window_type = 'even'
        kaufmann_design = 'nested'
        num_windows = 64
        scan_overlap = 0
        ionisation_mode = POSITIVE
        scan_time_dict = {1: 0.12, 2: 0.06}
        min_rt = 0
        max_rt = 400
        min_mz = 100
        max_mz = 1000

        # run controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows, scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'nested_dia_even.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_NestedDiaController_percentile(self, simple_dataset, fragscan_ps):
        logger.info('Testing NestedDiaController percentile')

        # some parameters
        window_type = 'percentile'
        kaufmann_design = 'nested'
        num_windows = 64
        scan_overlap = 0
        ionisation_mode = POSITIVE
        scan_time_dict = {1: 0.12, 2: 0.06}
        min_rt = 0
        max_rt = 400
        min_mz = 100
        max_mz = 1000

        # run controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows, scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'nested_dia_percentile.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_TreeDiaController_even(self, simple_dataset, fragscan_ps):
        logger.info('Testing TreeDiaController even')

        # some parameters
        window_type = 'even'
        kaufmann_design = 'tree'
        num_windows = 64
        scan_overlap = 0
        ionisation_mode = POSITIVE
        scan_time_dict = {1: 0.12, 2: 0.06}
        min_rt = 0
        max_rt = 400
        min_mz = 100
        max_mz = 1000

        # run controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows, scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'tree_dia_even.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_TreeDiaController_percentile(self, simple_dataset, fragscan_ps):
        logger.info('Testing TreeDiaController percentile')

        # some parameters
        window_type = 'percentile'
        kaufmann_design = 'tree'
        num_windows = 64
        scan_overlap = 0
        ionisation_mode = POSITIVE
        scan_time_dict = {1: 0.12, 2: 0.06}
        min_rt = 0
        max_rt = 400
        min_mz = 100
        max_mz = 1000

        # run controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset, fragscan_ps,
                                                scan_duration_dict=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows, scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'tree_dia_percentile.mzml'
        check_mzML(env, OUT_DIR, filename)


class TestFixedScansController:
    """
    Tests the FixedScansController that sends a scheduled number of scans
    """

    def test_FixedScansController(self, two_fixed_chems):
        logger.info('Testing FixedScansController')
        mz_to_target = [chem.mass + 1.0 for chem in two_fixed_chems]
        isolation_width = DEFAULT_ISOLATION_WIDTH
        mz_tol = 0.1
        rt_tol = 15
        min_rt = 110
        max_rt = 112
        ionisation_mode = POSITIVE

        controller = FixedScansController(schedule=None)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, two_fixed_chems, None)
        env = Environment(mass_spec, controller, min_rt, max_rt)

        ms1_scan = env.get_default_scan_params()
        ms2_scan_1 = env.get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width, mz_tol, rt_tol)
        ms2_scan_2 = env.get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width, mz_tol, rt_tol)
        ms2_scan_3 = env.get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width, mz_tol, rt_tol)
        schedule = [ms1_scan, ms2_scan_1, ms2_scan_2, ms2_scan_3]
        controller.set_tasks(schedule)
        set_log_level_warning()
        env.run()

        assert len(controller.scans[1]) == 1
        assert len(controller.scans[2]) == 3
        for scan in controller.scans[2]:
            assert scan.num_peaks > 0
        env.write_mzML(OUT_DIR, 'fixedScansController.mzML')


class TestChemsFromMZML:
    def test_fullscan_from_mzml(self, chems_from_mzml):
        ionisation_mode = POSITIVE
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(ionisation_mode, chems_from_mzml, None)
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
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)
        ms = IndependentMassSpectrometer(ionisation_mode, chems_from_mzml, None)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        check_non_empty_MS2(controller)
        filename = 'topn_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_mz_rt_i_from_mzml(self, chem_mz_rt_i_from_mzml):
        ionisation_mode = POSITIVE
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(ionisation_mode, chem_mz_rt_i_from_mzml, None)
        env = Environment(ms, controller, 500, 600, progress_bar=True)
        set_log_level_warning()
        env.run()
        filename = 'fullscan_mz_rt_i_from_mzml.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestTargetedController:
    def test_targeted(self):
        fs = EvenMZFormulaSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=10)
        cr = ConstantChromatogramSampler()
        ms = FixedMS2Sampler()
        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cr, ms2_sampler=ms)
        d = cs.sample(2, 2)  # sample chems with m/z = 100 and 200
        ionisation_mode = POSITIVE
        targets = []
        targets.append(Target(101, 100, 102, 10, 20))
        targets.append(Target(201, 200, 202, 10, 20))
        ce_values = [10, 20, 30]
        n_replicates = 4
        controller = TargetedController(targets, ce_values, n_replicates=n_replicates, limit_acquisition=True)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d, None)
        env = Environment(mass_spec, controller, 5, 25, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that we go all the scans we wanted
        for ms_level in controller.scans:
            assert len(controller.scans[ms_level]) > 0
        set_log_level_debug()
        target_counts = {t: {c: 0 for c in ce_values} for t in targets}

        for s in controller.scans[2]:
            params = s.scan_params
            pmz = params.get(ScanParameters.PRECURSOR_MZ)[0].precursor_mz
            filtered_targets = list(
                filter(lambda x: x.min_rt <= s.rt and x.max_rt >= s.rt and x.min_mz <= pmz and x.max_mz >= pmz,
                       targets))
            assert len(filtered_targets) == 1
            target = filtered_targets[0]
            ce = params.get(ScanParameters.COLLISION_ENERGY)
            target_counts[target][ce] += 1

        for t in target_counts:
            for ce, count in target_counts[t].items():
                assert count == n_replicates

    def test_target_creation(self):
        toxid_file = Path(BASE_DIR, 'StdMix1_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        toxid_file = Path(BASE_DIR, 'StdMix2_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        toxid_file = Path(BASE_DIR, 'StdMix3_pHILIC_Current.csv')
        targets = create_targets_from_toxid(toxid_file)
        assert len(targets) > 0
        set_log_level_debug()
        logger.debug(targets[-1].mz)


class TestMultiIsolationController:
    def test_multiple_isolation(self):
        N = 3
        fs = EvenMZFormulaSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=10)
        cr = ConstantChromatogramSampler()
        ms = FixedMS2Sampler()
        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cr, ms2_sampler=ms)
        d = cs.sample(3, 2)  # sample chems with m/z = 100 and 200
        ionisation_mode = POSITIVE
        controller = MultiIsolationController(N)
        ms = IndependentMassSpectrometer(POSITIVE, d, None)
        env = Environment(ms, controller, 10, 20, progress_bar=True)
        set_log_level_warning()
        env.run()

        assert len(controller.scans[1]) > 0
        assert len(controller.scans[2]) > 0

        # look at the first block of MS2 scans
        # and check that they are the correct super-positions
        mm = {}
        # first three scans hit the individual precursors
        mm[(0,)] = controller.scans[2][0]
        mm[(1,)] = controller.scans[2][1]
        mm[(2,)] = controller.scans[2][2]
        # next three should hit the pairs
        mm[(0,1)] = controller.scans[2][3]
        mm[(0,2)] = controller.scans[2][4]
        mm[(1,2)] = controller.scans[2][5]
        # final should hit all three
        mm[(0,1,2)] = controller.scans[2][6]

        for key, value in mm.items():
            actual_mz_vals = set(mm[key].mzs)
            expected_mz_vals = set()
            for k in key:
                for m in mm[(k,)].mzs:
                    expected_mz_vals.add(m)
            assert expected_mz_vals == actual_mz_vals

if __name__ == '__main__':
    unittest.main()
