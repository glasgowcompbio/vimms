import unittest
from pathlib import Path

import pytest

from vimms.Chemicals import ChemicalCreator, GET_MS2_BY_PEAKS, GET_MS2_BY_SPECTRA
from vimms.Common import *
from vimms.Controller import SimpleMs1Controller, TopNController, PurityController, TopN_RoiController, \
    TopN_SmartRoiController, ExcludingTopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer

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
BEER_MIN_BOUND = 550
BEER_MAX_BOUND = 650


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
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 1,
                        get_children_method=GET_MS2_BY_PEAKS)


@pytest.fixture(scope="module")
def fragscan_dataset_spectra(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 1,
                        get_children_method=GET_MS2_BY_SPECTRA)


### tests starts from here ###

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

    def test_peaks_in_range(self,fragscan_dataset_peaks, fullscan_ps):

        min_mz = 100.
        max_mz = 200.

        logger.info('Testing MS1 controller with narrow m/z range')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS, fullscan_ps)
        controller = SimpleMs1Controller(default_ms1_scan_window=(min_mz, max_mz))

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        for scan_level, scans in controller.scans.items():
            for s in scans[1:]:
                assert min(s.mzs) >= min_mz
                assert max(s.mzs) <= max_mz
        
        # write simulated output to mzML file
        filename = 'ms1_controller_qcbeer_chems_narrow.mzML'
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=True)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, ps, add_noise=False,
                                                scan_duration_dict=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise_with_scan_duration.mzML'
        check_mzML(env, OUT_DIR, filename)


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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=True)
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

        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS, fragscan_ps, add_noise=True,
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

    def test_roi_controller_with_simulated_chems(self, fragscan_dataset_spectra, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset_spectra) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_spectra, fragscan_ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_spectra, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'roi_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestSMARTROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def test_smart_roi_controller_with_simulated_chems(self, fragscan_dataset_spectra, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        len(fragscan_dataset_spectra) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_spectra, fragscan_ps, add_noise=True)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                             min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_spectra, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps, add_noise=True)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                             min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps, add_noise=False,
                                                scan_duration_dict=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY,
                                    ms1_shift=test_shift)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps, add_noise=False,
                                                scan_duration_dict=scan_duration_dict)
        controller = ExcludingTopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY,
                                             ms1_shift=test_shift,
                                             exclusion_t_0=exclusion_t_0,
                                             log_intensity=True)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'topN_excluding_shifted_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)


if __name__ == '__main__':
    unittest.main()
