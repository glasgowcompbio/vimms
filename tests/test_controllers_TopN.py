import os

import pymzml
from loguru import logger

from tests.conftest import N_CHEMS, MIN_MS1_INTENSITY, get_rt_bounds, CENTRE_RANGE, \
    run_environment, \
    check_non_empty_MS2, check_mzML, OUT_DIR, BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND, HMDB
from vimms.ChemicalSamplers import EvenMZFormulaSampler, FixedMS2Sampler, \
    UniformRTAndIntensitySampler, \
    ConstantChromatogramSampler, DatabaseFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, NEGATIVE, ScanParameters
from vimms.Controller import TopNController, SimpleMs1Controller, WeightedDEWController, \
    AdvancedParams
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import GaussianPeakNoise


class TestNegative:
    def test_neg(self, even_chems):
        mass_spec = IndependentMassSpectrometer(NEGATIVE, even_chems)
        N = 10
        controller = TopNController(NEGATIVE, N, 0.7, 10, 15, 0, force_N=True)
        env = Environment(mass_spec, controller, 200, 300, progress_bar=True)
        run_environment(env)

        for level in controller.scans:
            for scan in controller.scans[level]:
                assert scan.scan_params.get(ScanParameters.POLARITY) == NEGATIVE
        ms1_peaks = [int(m) for m in controller.scans[1][0].mzs]
        ms1_peaks.sort()
        assert 98 in ms1_peaks
        assert 198 in ms1_peaks
        assert 298 in ms1_peaks
        assert 398 in ms1_peaks

        filename = 'topn_negative.mzML'
        check_mzML(env, OUT_DIR, filename)

        # load the file and check polarity in the mzml

        run = pymzml.run.Reader(os.path.join(OUT_DIR, filename))
        for n, spec in enumerate(run):
            assert spec.get('MS:1000129')  # this is the negative scan accession


class TestTopNForcedN:
    """
    Test the TopN controller when N is forced to be N. I.e. always fragment enough
    """

    def test_TopN_forceN(self, ten_chems):
        mass_spec = IndependentMassSpectrometer(POSITIVE, ten_chems)
        N = 20
        controller = TopNController(POSITIVE, N, 0.7, 10, 15, 0, force_N=True)
        env = Environment(mass_spec, controller, 200, 300, progress_bar=True)
        run_environment(env)

        all_scans = controller.scans[1] + controller.scans[2]
        # sort by RT
        all_scans.sort(key=lambda x: x.rt)
        ms1_pos = []
        for i, s in enumerate(all_scans):
            if s.ms_level == 1:
                ms1_pos.append(i)

        for i, mp in enumerate(ms1_pos[:-1]):
            assert ms1_pos[i + 1] - (mp + 1) == N


class TestTopNController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the
    simulated mass spec class.
    """

    def test_TopN_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing Top-N controller with simulated chemicals -- no noise')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        filename = 'topN_controller_simulated_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_simulated_chems_and_noise(self, fragscan_dataset):
        logger.info('Testing Top-N controller with simulated chemicals -- with noise')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and Top-N controller
        mz_noise = GaussianPeakNoise(0.1)
        intensity_noise = GaussianPeakNoise(1000.)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset,
                                                mz_noise=mz_noise,
                                                intensity_noise=intensity_noise)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_simulated_chems_with_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_beer_chems(self):
        logger.info('Testing Top-N controller with QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_TopN_controller_with_beer_chems_and_scan_duration_dict(self):
        logger.info('Testing Top-N controller with QC beer chemicals '
                    'passing in the scan durations')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # ps = None
        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller and passing
        # in the scan_duration dict
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS,
                                                scan_duration=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_controller_qcbeer_chems_no_noise_with_scan_duration.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestTopNAdvanced:
    def test_TopN_controller_advanced_params(self):
        # set some values that are not the defaults, so we know they're passed correctly
        params = AdvancedParams(
            default_ms1_scan_window=(10.0, 2000.0),
            ms1_agc_target=100000,
            ms1_max_it=500,
            ms1_collision_energy=200,
            ms1_orbitrap_resolution=100000,
            ms1_activation_type='CID',
            ms1_mass_analyser='IonTrap',
            ms1_isolation_mode='IonTrap',
            ms1_source_cid_energy=10,
            ms2_agc_target=50000,
            ms2_max_it=250,
            ms2_collision_energy=300,
            ms2_orbitrap_resolution=100000,
            ms2_activation_type='CID',
            ms2_mass_analyser='IonTrap',
            ms2_isolation_mode='IonTrap',
            ms2_source_cid_energy=20
        )

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY, advanced_params=params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that some of the scan parameters returned are actually what we set
        # ms1 check
        scan = controller.scans[1][0]
        scan_params = scan.scan_params
        assert scan_params.get(ScanParameters.FIRST_MASS) == params.default_ms1_scan_window[0]
        assert scan_params.get(ScanParameters.LAST_MASS) == params.default_ms1_scan_window[1]
        assert scan_params.get(ScanParameters.AGC_TARGET) == params.ms1_agc_target
        assert scan_params.get(ScanParameters.MAX_IT) == params.ms1_max_it
        assert scan_params.get(ScanParameters.COLLISION_ENERGY) == params.ms1_collision_energy
        assert scan_params.get(ScanParameters.ORBITRAP_RESOLUTION) == params.ms1_orbitrap_resolution # noqa
        assert scan_params.get(ScanParameters.ACTIVATION_TYPE) == params.ms1_activation_type
        assert scan_params.get(ScanParameters.MASS_ANALYSER) == params.ms1_mass_analyser
        assert scan_params.get(ScanParameters.ISOLATION_MODE) == params.ms1_isolation_mode
        assert scan_params.get(ScanParameters.SOURCE_CID_ENERGY) == params.ms1_source_cid_energy

        # ms2 check
        scan = controller.scans[2][0]
        scan_params = scan.scan_params
        assert scan_params.get(ScanParameters.AGC_TARGET) == params.ms2_agc_target
        assert scan_params.get(ScanParameters.MAX_IT) == params.ms2_max_it
        assert scan_params.get(ScanParameters.COLLISION_ENERGY) == params.ms2_collision_energy
        assert scan_params.get(ScanParameters.ORBITRAP_RESOLUTION) == params.ms2_orbitrap_resolution # noqa
        assert scan_params.get(ScanParameters.ACTIVATION_TYPE) == params.ms2_activation_type
        assert scan_params.get(ScanParameters.MASS_ANALYSER) == params.ms2_mass_analyser
        assert scan_params.get(ScanParameters.ISOLATION_MODE) == params.ms2_isolation_mode
        assert scan_params.get(ScanParameters.SOURCE_CID_ENERGY) == params.ms2_source_cid_energy


class TestTopNControllerSpectra:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the
    simulated mass spec class. Fragment spectra are generated via the "spectra" method
    """

    def test_TopN_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing Top-N controller with simulated chemicals -- no noise')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        filename = 'topN_controller_simulated_chems_no_noise_spectra.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestIonisationMode:
    def test_positive_fixed(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler()
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri,
                                    chromatogram_sampler=cs)
        dataset = cm.sample(3, 2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(POSITIVE, dataset)
        controller = TopNController(POSITIVE, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)
        env = Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        ms1_mz_vals = [int(m) for m in controller.scans[1][0].mzs]

        expected_vals = [101, 201, 301]
        for i, m in enumerate(ms1_mz_vals):
            assert m == expected_vals[i]

        expected_frags = set([81, 91, 181, 191, 281, 291])
        for scan in controller.scans[2]:
            for m in scan.mzs:
                assert int(m) in expected_frags

    def test_negative_fixed(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler()
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri,
                                    chromatogram_sampler=cs)
        dataset = cm.sample(3, 2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(NEGATIVE, dataset)
        controller = TopNController(NEGATIVE, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)
        env = Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        ms1_mz_vals = [int(m) for m in controller.scans[1][0].mzs]

        expected_vals = [98, 198, 298]
        for i, m in enumerate(ms1_mz_vals):
            assert m == expected_vals[i]

        expected_frags = set([88, 78, 188, 178, 288, 278])
        for scan in controller.scans[2]:
            for m in scan.mzs:
                assert int(m) in expected_frags

    def test_multiple_adducts(self):
        fs = DatabaseFormulaSampler(HMDB)
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        adduct_prior_dict = {POSITIVE: {'M+H': 100, 'M+Na': 100, 'M+K': 100}}
        cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs,
                                    adduct_prior_dict=adduct_prior_dict,
                                    adduct_proportion_cutoff=0.0)

        n_adducts = len(adduct_prior_dict[POSITIVE])
        n_chems = 5
        dataset = cm.sample(n_chems, 2)

        for c in dataset:
            c.isotopes = [(c.mass, 1, "Mono")]

        # should be 15 peaks or less all the time
        # some adducts might not be sampled if the probability is less than 0.2
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(POSITIVE, dataset)
        env = Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        for scan in controller.scans[1]:
            assert len(scan.mzs) <= n_chems * n_adducts


class TestExclusion:
    def test_TopN_controller_with_beer_chems_and_initial_exclusion_list(self):
        logger.info('Testing Top-N controller with QC beer chemicals and '
                    'an initial exclusion list')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        initial_exclusion_list = []
        for i in range(3):
            mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
            controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                        MIN_MS1_INTENSITY,
                                        initial_exclusion_list=initial_exclusion_list)
            env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND,
                              progress_bar=True)
            run_environment(env)

            mz_intervals = list(controller.exclusion.exclusion_list.boxes_mz.items())
            rt_intervals = list(controller.exclusion.exclusion_list.boxes_rt.items())
            unique_items_mz = set(i.data for i in mz_intervals)
            unique_items_rt = set(i.data for i in rt_intervals)
            assert len(unique_items_mz) == len(unique_items_rt)

            initial_exclusion_list = list(unique_items_mz)

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
        initial_exclusion_list = []
        min_ms1_intensity = 0
        N = 10
        mz_tol = 10
        rt_tol = 30
        isolation_width = 1
        all_controllers = []
        for i in range(3):
            mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset)
            controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                        min_ms1_intensity,
                                        initial_exclusion_list=initial_exclusion_list)
            env = Environment(mass_spec, controller, 0, 20, progress_bar=True)
            run_environment(env)

            mz_intervals = list(controller.exclusion.exclusion_list.boxes_mz.items())
            rt_intervals = list(controller.exclusion.exclusion_list.boxes_rt.items())
            unique_items_mz = set(i.data for i in mz_intervals)
            unique_items_rt = set(i.data for i in rt_intervals)
            assert len(unique_items_mz) == len(unique_items_rt)

            initial_exclusion_list = list(unique_items_mz)

            all_controllers.append(controller)
        assert len(all_controllers[0].scans[2]) == n_chems
        assert len(all_controllers[1].scans[2]) == 0
        assert len(all_controllers[2].scans[2]) == 0


class TestTopNShiftedController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with
    the beer chems.
    """

    def test_TopN_controller_with_beer_chems(self):
        logger.info('Testing Top-N controller with QC beer chemicals')
        test_shift = 0
        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        scan_duration_dict = {1: 0.2, 2: 0.1}

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS,
                                                scan_duration=scan_duration_dict)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY, ms1_shift=test_shift)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND,
                          progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_shifted_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestWeightedDEWController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans
    with the beer chems.
    """

    def test_WeightedDEW_controller_with_beer_chems(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS,
                                                scan_duration=scan_duration_dict)
        controller = WeightedDEWController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                           MIN_MS1_INTENSITY, ms1_shift=test_shift,
                                           exclusion_t_0=exclusion_t_0, log_intensity=True)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND,
                          progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'topN_weighted_dew_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)
