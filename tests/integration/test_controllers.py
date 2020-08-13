import sys
import unittest

sys.path.append('..')
sys.path.append('C:\\Users\\joewa\\Work\\git\\pymzm') # FIXME: termporary hack
sys.path.append('/Users/simon/git/pymzm')

from pathlib import Path

from vimms.Chemicals import ChemicalCreator, GET_MS2_BY_PEAKS, GET_MS2_BY_SPECTRA
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import SimpleMs1Controller, TopNController, PurityController, TopN_RoiController, AIF
from vimms.Environment import Environment
from vimms.Common import *

dir_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.abspath(Path(dir_path, 'fixtures'))
hmdb = load_obj(Path(base_dir, 'hmdb_compounds.p'))
out_dir = Path(dir_path, 'results')

ROI_Sources = [str(Path(base_dir, 'beer_t10_simulator_files'))]
min_ms1_intensity = 1.75E5
rt_range = [(0, 1200)]
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]
mz_range = [(0, 1050)]
n_chems = 500

beer_chems = load_obj(Path(base_dir, 'QCB_22May19_1.p'))


class TestMS1Controller(unittest.TestCase):
    """
    Tests the MS1 controller that does MS1 full-scans only with the simulated mass spec class.
    """

    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fullscan.p'))
        self.ms_level = 1

    def test_ms1_controller_with_simulated_chems(self):
        logger.info('Testing MS1 controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level)
        self.assertEqual(len(dataset), n_chems)

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, dataset, self.ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'ms1_controller_simulated_chems.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_ms1_controller_with_qcbeer_chems(self):
        logger.info('Testing MS1 controller with QC beer chemicals')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, beer_chems, self.ps)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()


class TestTopNController(unittest.TestCase):
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))
        self.ms_level = 1

    def test_TopN_controller_with_simulated_chems(self):
        logger.info('Testing Top-N controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level,
                               get_children_method=GET_MS2_BY_PEAKS)
        self.assertEqual(len(dataset), n_chems)

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_co ntroller_simulated_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))

        # create a simulated mass spec with noise and Top-N controller
        logger.info('With noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=True)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_TopN_controller_with_beer_chems(self):
        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, self.ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()


class TestPurityController(unittest.TestCase):
    """
    Tests the Purity controller that is used for purity experiments
    """

    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))
        self.ms_level = 1

    def test_purity_controller_with_simulated_chems(self):
        logger.info('Testing purity controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level,
                               get_children_method=GET_MS2_BY_PEAKS)
        self.assertEqual(len(dataset), n_chems)

        # set different isolation widths, Ns, dynamic exclusion RT and mz tolerances at different timepoints
        isolation_widths = [1, 1, 1, 1]
        N = [5, 10, 15, 20]
        rt_tol = [15, 30, 60, 120]
        mz_tol = [10, 5, 15, 20]
        scan_param_changepoints = [300, 600, 900]  # the timepoints when we will change the 4 parameters above
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and purity controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=True)
        controller = PurityController(ionisation_mode, N, scan_param_changepoints, isolation_widths, mz_tol, rt_tol,
                                      min_ms1_intensity)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_purity_controller_with_beer_chems(self):
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

        mass_spec = IndependentMassSpectrometer(POSITIVE, beer_chems, self.ps, add_noise=True,
                                                isolation_transition_window=isolation_transition_window,
                                                isolation_transition_window_params=isolation_transition_window_params)
        controller = PurityController(mass_spec, N, scan_param_changepoints, isolation_window, mz_tol, rt_tol,
                                      min_ms1_intensity, n_purity_scans, purity_shift, purity_threshold,
                                      purity_add_ms1=purity_add_ms1, purity_randomise=purity_randomise)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()


class TestROIController(unittest.TestCase):
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on selecting regions of interests
    (rather than the top-N most intense peaks)
    """

    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))
        self.ms_level = 1

    def test_roi_controller_with_simulated_chems(self):
        logger.info('Testing ROI controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level,
                               get_children_method=GET_MS2_BY_SPECTRA)
        self.assertEqual(len(dataset), n_chems)

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_roi_controller_with_beer_chems(self):
        logger.info('Testing ROI controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, self.ps, add_noise=True)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                        min_roi_intensity, min_roi_length, N, rt_tol)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

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
        self.assertTrue(os.path.exists(out_file))
        print()



class TestTopNShiftedController(unittest.TestCase):
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """
    
    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))
        self.ms_level = 1

    def test_TopN_controller_with_simulated_chems(self):

        test_shift = 3

        logger.info('Testing Top-N controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level,
                               get_children_method=GET_MS2_BY_PEAKS)
        self.assertEqual(len(dataset), n_chems)

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_shifted_controller_simulated_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))

        # create a simulated mass spec with noise and Top-N controller
        logger.info('With noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=True)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'topN_shifted_controller_simulated_chems_with_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_TopN_controller_with_beer_chems(self):

        test_shift = 3

        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, self.ps, add_noise=False)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift = test_shift)

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
        self.assertTrue(os.path.exists(out_file))
        print()

class TestDIAControllers(unittest.TestCase):
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the simulated mass spec class.
    """

    def setUp(self):
        self.ps = load_obj(Path(base_dir, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))
        self.ms_level = 1

    def test_AIF_controller_with_simulated_chems(self):
        logger.info('Testing Top-N controller with simulated chemicals')

        # create some chemical objects
        chems = ChemicalCreator(self.ps, ROI_Sources, hmdb)
        dataset = chems.sample(mz_range, rt_range, min_ms1_intensity, n_chems, self.ms_level,
                               get_children_method=GET_MS2_BY_PEAKS)
        self.assertEqual(len(dataset), n_chems)

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

        scan_time_dict = {1:0.12, 2:0.06}

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=False, scan_duration_dict = scan_time_dict)
        controller = AIF(min_mz,max_mz)
        
        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_simulated_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))

        # create a simulated mass spec with noise and Top-N controller
        logger.info('With noise')
        
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset, self.ps, add_noise=True, scan_duration_dict = scan_time_dict)
        controller = AIF(min_mz,max_mz)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_simulated_chems_with_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))
        print()

    def test_AIF_controller_with_beer_chems(self):
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
        scan_time_dict = {1:0.124,2:0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, beer_chems, self.ps, add_noise=False, scan_duration_dict = scan_time_dict)
        controller = AIF(min_mz,max_mz)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_qcbeer_chems_no_noise.mzML'
        out_file = os.path.join(out_dir, filename)
        env.write_mzML(out_dir, filename)
        self.assertTrue(os.path.exists(out_file))
        print()




if __name__ == '__main__':
    unittest.main()
