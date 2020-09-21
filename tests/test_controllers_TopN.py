from loguru import logger

from tests.conftest import N_CHEMS, MIN_MS1_INTENSITY, get_rt_bounds, CENTRE_RANGE, run_environment, \
    check_non_empty_MS2, check_mzML, OUT_DIR, BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND, HMDB
from vimms.ChemicalSamplers import EvenMZFormulaSampler, FixedMS2Sampler, UniformRTAndIntensitySampler, \
    ConstantChromatogramSampler, DatabaseFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, NEGATIVE
from vimms.Controller import TopNController, SimpleMs1Controller, PurityController, WeightedDEWController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import GaussianPeakNoise



class TestTopNForcedN:
    """
    Test the TopN controller when N is forced to be N. I.e. always fragment enough
    """
    def test_TopN_forceN(self, ten_chems):
        mass_spec = IndependentMassSpectrometer(POSITIVE, ten_chems, None)
        N = 20
        controller = TopNController(POSITIVE, N, 0.7, 10, 15, 0, force_N=True)
        env = Environment(mass_spec, controller, 200, 300, progress_bar=True)
        run_environment(env)

        all_scans = controller.scans[1] + controller.scans[2]
        # sort by RT
        all_scans.sort(key = lambda x: x.rt)
        ms1_pos = []
        for i, s in enumerate(all_scans):
            if s.ms_level == 1:
                ms1_pos.append(i)
        
        for i, mp in enumerate(ms1_pos[:-1]):
            assert ms1_pos[i+1] - (mp + 1) == N


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


class TestIonisationMode:
    def test_positive_fixed(self):
        fs = EvenMZFormulaSampler()
        ms = FixedMS2Sampler()
        ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
        cs = ConstantChromatogramSampler()
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        dataset = cm.sample(3, 2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(POSITIVE, dataset, None)
        controller = TopNController(POSITIVE, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
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
        cm = ChemicalMixtureCreator(fs, ms2_sampler=ms, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        dataset = cm.sample(3, 2)

        N = 10
        isolation_width = 0.7
        mz_tol = 10
        rt_tol = 15

        ms = IndependentMassSpectrometer(NEGATIVE, dataset, None)
        controller = TopNController(NEGATIVE, N, isolation_width, mz_tol, rt_tol, MIN_MS1_INTENSITY)
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
                                    adduct_prior_dict=adduct_prior_dict, adduct_proportion_cutoff=0.0)

        n_adducts = len(adduct_prior_dict[POSITIVE])
        n_chems = 5
        dataset = cm.sample(n_chems, 2)

        for c in dataset:
            c.isotopes = [(c.mass, 1, "Mono")]

        # should be 15 peaks or less all the time
        # some adducts might not be sampled if the probability is less than 0.2
        controller = SimpleMs1Controller()
        ms = IndependentMassSpectrometer(POSITIVE, dataset, None)
        env = Environment(ms, controller, 102, 110, progress_bar=True)
        set_log_level_warning()
        env.run()
        for scan in controller.scans[1]:
            assert len(scan.mzs) <= n_chems * n_adducts


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


class TestWeightedDEWController:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans with the beer chems.
    """

    def test_WeightedDEW_controller_with_beer_chems(self, fragscan_ps):
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
        filename = 'topN_weighted_dew_controller_qcbeer_chems_no_noise.mzML'
        check_mzML(env, OUT_DIR, filename)