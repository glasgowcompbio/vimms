import sys
import unittest

sys.path.append('..')
sys.path.append('C:\\Users\\joewa\\Work\\git\\pymzm')  # FIXME: termporary hack
sys.path.append('/Users/simon/git/pymzm')

from pathlib import Path

import pytest

from vimms.Chemicals import ChemicalCreator, GET_MS2_BY_PEAKS, GET_MS2_BY_SPECTRA
from vimms.MassSpec import IndependentMassSpectrometer

from vimms.Controller import SimpleMs1Controller, TopNController, PurityController, TopN_RoiController, \
    TopN_SmartRoiController, ExcludingTopNController
from vimms.Environment import Environment
from vimms.Common import *

dir_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.abspath(Path(dir_path, 'fixtures'))
hmdb = load_obj(Path(base_dir, 'hmdb_compounds.p'))
out_dir = Path(dir_path, 'results')

ROI_Sources = [str(Path(base_dir, 'beer_t10_simulator_files'))]
min_ms1_intensity = 1.75E5
min_ms1_intensity = 1
rt_range = [(0, 1200)]
centre_range = 600
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]
mz_range = [(0, 1050)]
n_chems = 10

beer_chems = load_obj(Path(base_dir, 'QCB_22May19_1.p'))
beer_min_bound = 550
beer_max_bound = 650


def get_rt_bounds(dataset, centre):
    rts = [ds.rt for ds in dataset]
    min_bound = max([rt for rt in rts if rt < centre], default=centre) - 60
    max_bound = min([rt for rt in rts if rt > centre], default=centre) + 60
    return (min_bound, max_bound)


@pytest.fixture(scope="module")
def fullscan_ps():
    return load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fullscan.p'))


@pytest.fixture(scope="module")
def fullscan_dataset(fullscan_ps):
    chems = ChemicalCreator(fullscan_ps, ROI_Sources, hmdb)
    return chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, 1)


@pytest.fixture(scope="module")
def fragscan_ps():
    return load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))


@pytest.fixture(scope="module")
def fragscan_dataset_peaks(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_Sources, hmdb)
    return chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, 1,
                        get_children_method=GET_MS2_BY_PEAKS)


@pytest.fixture(scope="module")
def fragscan_dataset_spectra(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_Sources, hmdb)
    return chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, 1,
                        get_children_method=GET_MS2_BY_SPECTRA)


class TestMS1Controller:
    """
    Tests the MS1 controller that does MS1 full-scans only with the simulated mass spec class.
    """

    def test_ms1_controller_with_simulated_chems(self, fragscan_dataset_peaks, fullscan_ps):
        logger.info('Testing MS1 controller with simulated chemicals')
        print('Testing here')

        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, centre_range)
        logger.info('RT bounds %s %s' % (min_bound, max_bound))
        assert len(fragscan_dataset_peaks) == n_chems

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, fragscan_dataset_peaks, fullscan_ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        logger.info('Running simulation')
        env.run()
        logger.info('Done')
        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'ms1_controller_simulated_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        logger.info('Writing out mzML')
        env.write_mzML(out_dir, filename)
        logger.info('Done')
        assert os.path.exists(out_file)
        print()

    def test_ms1_controller_with_qcbeer_chems(self, fullscan_ps):
        logger.info('Testing MS1 controller with QC beer chemicals')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, beer_chems, fullscan_ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'ms1_controller_qcbeer_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestTopNController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    def test_TopN_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing Top-N controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == n_chems

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, centre_range)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_controller_simulated_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)

        # create a simulated mass spec with noise and Top-N controller
        logger.info('With noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=True)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_controller_simulated_chems_with_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()

    def test_TopN_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, fragscan_ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestPurityController:
    """
    Tests the Purity controller that is used for purity experiments
    """

    def test_purity_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing purity controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == n_chems

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
                                      min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, centre_range)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'purity_controller_simulated_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()

    def test_purity_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing purity controller with QC beer chemicals')

        isolation_window = [1]  # the isolation window in Dalton around a selected precursor ion
        N = [5]
        rt_tol = [10]
        mz_tol = [10]
        min_ms1_intensity = 1.75E5
        scan_param_changepoints = None
        rt_range = [(0, 400)]
        min_rt = rt_range[0][0]
        max_rt = rt_range[0][1]
        n_purity_scans = N[0]
        purity_shift = 0.2
        purity_threshold = 1

        # these settings change the Mass Spec type. They arent necessary to run the Top-N ROI Controller
        isolation_transition_window = 'gaussian'
        isolation_transition_window_params = [0.5]

        purity_add_ms1 = True  # this seems to be the broken bit
        purity_randomise = True

        mass_spec = IndependentMassSpectrometer(POSITIVE, beer_chems, fragscan_ps, add_noise=True,
                                                isolation_transition_window=isolation_transition_window,
                                                isolation_transition_window_params=isolation_transition_window_params)
        controller = PurityController(mass_spec, N, scan_param_changepoints, isolation_window, mz_tol, rt_tol,
                                      min_ms1_intensity, n_purity_scans, purity_shift, purity_threshold,
                                      purity_add_ms1=purity_add_ms1, purity_randomise=purity_randomise)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'purity_controller_qcbeer_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def test_roi_controller_with_simulated_chems(self, fragscan_dataset_spectra, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset_spectra) == n_chems

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_spectra, fragscan_ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_spectra, centre_range)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'roi_controller_simulated_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, fragscan_ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'roi_controller_qcbeer_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestSMARTROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def test_smart_roi_controller_with_simulated_chems(self, fragscan_dataset_spectra, fragscan_ps):
        logger.info('Testing ROI controller with simulated chemicals')
        len(fragscan_dataset_spectra) == n_chems

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_spectra, fragscan_ps, add_noise=True)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                             min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_spectra, centre_range)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'smart_roi_controller_simulated_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, fragscan_ps, add_noise=True)
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                             min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'smart_controller_qcbeer_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestTopNShiftedController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    # def test_TopN_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):

    #     test_shift = 3

    #     logger.info('Testing Top-N controller with simulated chemicals')
    #     assert len(fragscan_dataset_peaks) == n_chems

    #     isolation_width = 1
    #     N = 10
    #     rt_tol = 15
    #     mz_tol = 10
    #     ionisation_mode = POSITIVE

    #     # create a simulated mass spec without noise and Top-N controller
    #     logger.info('Without noise')
    #     mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, fragscan_dataset_peaks, add_noise=False)
    #     controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

    #     # create an environment to run both the mass spec and controller
    #     min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, centre_range)
    #     env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

    #     # set the log level to WARNING so we don't see too many messages when environment is running
    #     set_log_level_warning()

    #     # run the simulation
    #     env.run()

    #     # set the log level back to DEBUG
    #     set_log_level_debug()

    #     # write simulated output to mzML file
    #     filename = 'topN_shifted_controller_simulated_chems_no_noise.mzML'
    #     out_file = os.path.join(out_dir, filename)
    #     env.write_mzML(out_dir, filename)
    #     assert os.path.exists(out_file)

    #     # create a simulated mass spec with noise and Top-N controller
    #     logger.info('With noise')
    #     mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=True)
    #     controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

    #     # create an environment to run both the mass spec and controller
    #     env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

    #     # set the log level to WARNING so we don't see too many messages when environment is running
    #     set_log_level_warning()

    #     # run the simulation
    #     env.run()

    #     # set the log level back to DEBUG
    #     set_log_level_debug()

    #     # write simulated output to mzML file
    #     filename = 'topN_shifted_controller_simulated_chems_with_noise.mzML'
    #     out_file = os.path.join(out_dir, filename)
    #     env.write_mzML(out_dir, filename)
    #     assert os.path.exists(out_file)
    #     print()

    def test_TopN_controller_with_beer_chems(self, fragscan_ps):
        test_shift = 0

        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, fragscan_ps, add_noise=False,
                                                scan_duration_dict=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                                    ms1_shift=test_shift)

        min_rt = 500
        max_rt = 600

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_shifted_controller_qcbeer_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


class TestTopNExcludingShiftedController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    # def test_excluded_TopN_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):

    #     test_shift = 0

    #     logger.info('Testing Top-N controller with simulated chemicals')

    #     assert len(fragscan_dataset_peaks) == n_chems

    #     isolation_width = 1
    #     N = 10
    #     rt_tol = 60
    #     exclusion_t_0 = 15
    #     mz_tol = 10
    #     ionisation_mode = POSITIVE

    #     # create a simulated mass spec without noise and Top-N controller
    #     logger.info('Without noise')
    #     mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=False)
    #     controller = ExcludingTopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

    #     # create an environment to run both the mass spec and controller
    #     min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, centre_range)
    #     env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

    #     # set the log level to WARNING so we don't see too many messages when environment is running
    #     set_log_level_warning()

    #     # run the simulation
    #     env.run()

    #     # set the log level back to DEBUG
    #     set_log_level_debug()

    #     # write simulated output to mzML file
    #     filename = 'topN_excluding_controller_simulated_chems_no_noise.mzML'
    #     out_file = os.path.join(out_dir, filename)
    #     env.write_mzML(out_dir, filename)
    #     assert os.path.exists(out_file)

    #     # create a simulated mass spec with noise and Top-N controller
    #     logger.info('With noise')
    #     mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps, add_noise=True)
    #     controller = ExcludingTopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

    #     # create an environment to run both the mass spec and controller
    #     env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

    #     # set the log level to WARNING so we don't see too many messages when environment is running
    #     set_log_level_warning()

    #     # run the simulation
    #     env.run()

    #     # set the log level back to DEBUG
    #     set_log_level_debug()

    #     # write simulated output to mzML file
    #     filename = 'topN_excluding_controller_simulated_chems_with_noise.mzML'
    #     out_file = os.path.join(out_dir, filename)
    #     env.write_mzML(out_dir, filename)
    #     assert os.path.exists(out_file)
    #     print()

    def test_TopN_excluding_controller_with_beer_chems(self, fragscan_ps):
        test_shift = 0

        logger.info('Testing excluding Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10

        mz_tol = 10
        ionisation_mode = POSITIVE

        exclusion_t_0 = 15.0
        rt_tol = 120

        min_rt = 500
        max_rt = 600

        scan_duration_dict = {1: 0.2, 2: 0.1}
        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, fragscan_ps, add_noise=False,
                                                scan_duration_dict=scan_duration_dict)
        controller = ExcludingTopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                                             ms1_shift=test_shift,
                                             exclusion_t_0=exclusion_t_0,
                                             log_intensity=True)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, beer_min_bound, beer_max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_excluding_shifted_controller_qcbeer_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        assert os.path.exists(out_file)
        print()


if __name__ == '__main__':
    unittest.main()
