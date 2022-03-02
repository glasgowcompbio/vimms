import csv
import os
from pathlib import Path

from loguru import logger

from tests.conftest import N_CHEMS, get_rt_bounds, CENTRE_RANGE, check_mzML, \
    OUT_DIR, BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND, check_non_empty_MS2
from vimms.ChemicalSamplers import EvenMZFormulaSampler, FixedMS2Sampler, \
    ConstantChromatogramSampler, UniformRTAndIntensitySampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug
from vimms.Controller import AdvancedParams, AIF, SWATH, DiaController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import GaussianPeakNoiseLevelSpecific, UniformSpikeNoise


class TestAIFControllers:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the
    simulated mass spec class.
    """

    def test_AIF_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing Top-N controller with simulated chemicals')

        # create some chemical object
        assert len(fragscan_dataset) == N_CHEMS

        # isolation_width = 1
        # N = 10
        # rt_tol = 15
        # mz_tol = 10
        ionisation_mode = POSITIVE

        min_mz = 100
        max_mz = 500

        # shorten  the rt range for quicker tests
        # min_rt = 0
        # max_rt = 400

        scan_time_dict = {1: 0.12, 2: 0.06}

        # create a simulated mass spec without noise and Top-N controller
        logger.info('Without noise')
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset,
                                                scan_duration=scan_time_dict)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, advanced_params=params)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages when
        # environment is running
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset,
                                                mz_noise=mz_noise,
                                                intensity_noise=intensity_noise,
                                                scan_duration=scan_time_dict)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, advanced_params=params)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)

        # set the log level to WARNING so we don't see too many messages
        # when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()

        # set the log level back to DEBUG
        set_log_level_debug()

        # write simulated output to mzML file
        filename = 'AIF_simulated_chems_with_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_AIF_controller_with_beer_chems(self):
        logger.info('Testing Top-N controller with QC beer chemicals')

        # isolation_width = 1
        # N = 10
        # rt_tol = 15
        # mz_tol = 10
        ionisation_mode = POSITIVE
        min_mz = 100
        max_mz = 500

        # min_rt = 0
        # max_rt = 500

        # create a simulated mass spec without noise and Top-N controller
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS,
                                                scan_duration=scan_time_dict)
        params = AdvancedParams(default_ms1_scan_window=[min_mz, max_mz])
        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy, advanced_params=params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND,
                          progress_bar=True)

        # set the log level to WARNING so we don't see too many messages
        # when environment is running
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
        controller = AIF(ms1_source_cid_energy, advanced_params=params)
        out_file = Path(OUT_DIR, 'AIF_experiment.txt')
        controller.write_msdial_experiment_file(out_file)

        assert os.path.exists(out_file)
        with open(out_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t', lineterminator=os.linesep)
            rows = []
            for row in reader:
                rows.append(row)
        ce = ms1_source_cid_energy
        expected_row = ['1', 'ALL', min_mz, max_mz, "{}eV".format(ce), ce, 1]
        for i, val in enumerate(expected_row):
            assert rows[-1][i] == str(val)

    def test_aif_with_fixed_chems(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler(n_frags=2)
        cs = ConstantChromatogramSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=1)
        cs = ChemicalMixtureCreator(fs, ms2_sampler=ms, chromatogram_sampler=cs,
                                    rt_and_intensity_sampler=ri)
        d = cs.sample(1, 2)

        ms1_source_cid_energy = 30
        controller = AIF(ms1_source_cid_energy)
        ionisation_mode = POSITIVE
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d)
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

        mass_spec = IndependentMassSpectrometer(ionisation_mode, ten_chems,
                                                spike_noise=spike_noise,
                                                scan_duration=scan_time_dict)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems,
                                                scan_duration=scan_time_dict)
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems,
                                                scan_duration=scan_time_dict)
        env = Environment(mass_spec, controller2, 200, 300, progress_bar=True)
        env.run()

        ms2_scans2 = controller2.scans[2]

        assert len(ms2_scans2[0].mzs) == len(even_chems[0].children) + len(even_chems[1].children)
        assert len(ms2_scans2[1].mzs) == len(even_chems[2].children) + len(even_chems[3].children)

        width = 400
        controller3 = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)
        scan_time_dict = {1: 0.124, 2: 0.124}
        mass_spec = IndependentMassSpectrometer(ionisation_mode, even_chems,
                                                scan_duration=scan_time_dict)
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

    def test_NestedDiaController_even(self, simple_dataset):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset,
                                                scan_duration=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows,
                                   scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'nested_dia_even.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_NestedDiaController_percentile(self, simple_dataset):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset,
                                                scan_duration=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows,
                                   scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'nested_dia_percentile.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_TreeDiaController_even(self, simple_dataset):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset,
                                                scan_duration=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows,
                                   scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'tree_dia_even.mzml'
        check_mzML(env, OUT_DIR, filename)

    def test_TreeDiaController_percentile(self, simple_dataset):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, simple_dataset,
                                                scan_duration=scan_time_dict)
        controller = DiaController(min_mz, max_mz, window_type, kaufmann_design, num_windows,
                                   scan_overlap=scan_overlap)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
        set_log_level_warning()
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'tree_dia_percentile.mzml'
        check_mzML(env, OUT_DIR, filename)
