from loguru import logger

from tests.conftest import N_CHEMS, MIN_MS1_INTENSITY, get_rt_bounds, CENTRE_RANGE, run_environment, \
    check_non_empty_MS2, check_mzML, OUT_DIR, BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND
from vimms.Box import LocatorGrid, IdentityDrift, AllOverlapGrid
from vimms.Common import POSITIVE, ROI_EXCLUSION_WEIGHTED_DEW
from vimms.Controller import TopN_RoiController, TopN_SmartRoiController, NonOverlapController, \
    IntensityNonOverlapController, FlexibleNonOverlapController, RoiBuilder
from vimms.Environment import Environment
from vimms.GridEstimator import GridEstimator
from vimms.MassSpec import IndependentMassSpectrometer


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


class TestNonOverlapController:
    def test_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        grid = GridEstimator(LocatorGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size), IdentityDrift())
        controller = NonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                          min_roi_intensity, min_roi_length, N, grid,
                                          rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing non-overlap controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(LocatorGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = NonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                          min_roi_intensity, min_roi_length, N, grid,
                                          rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems_and_smartROI_rules(self, fragscan_ps):
        logger.info('Testing non-overlap controller with QC beer chemicals and SmartROI rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(LocatorGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = NonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                          min_roi_intensity, min_roi_length, N, grid,
                                          rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                          roi_type=RoiBuilder.ROI_TYPE_SMART,
                                          reset_length_seconds=1e6,
                                          intensity_increase_factor=10,
                                          drop_perc=0.1 / 100
                                          )

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self, fragscan_ps):
        logger.info('Testing non-overlap controller with QC beer chemicals and WeightedDEW rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 120
        exclusion_t_0 = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(LocatorGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = NonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                          min_roi_intensity, min_roi_length, N, grid,
                                          rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                          exclusion_method=ROI_EXCLUSION_WEIGHTED_DEW, exclusion_t_0=exclusion_t_0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems_weighteddew.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestIntensityNonOverlapController:
    def test_intensity_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing intensity non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size), IdentityDrift())
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                   min_roi_intensity, min_roi_length, N, grid,
                                                   rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing intensity non-overlap controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                   min_roi_intensity, min_roi_length, N, grid,
                                                   rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems_and_smartROI_rules(self, fragscan_ps):
        logger.info('Testing intensity non-overlap controller with QC beer chemicals and SmartROI rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                   min_roi_intensity, min_roi_length, N, grid,
                                                   rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                                   roi_type=RoiBuilder.ROI_TYPE_SMART,
                                                   reset_length_seconds=1e6,
                                                   intensity_increase_factor=10,
                                                   drop_perc=0.1 / 100
                                                   )

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self, fragscan_ps):
        logger.info('Testing intensity non-overlap controller with QC beer chemicals and WeightedDEW rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 120
        exclusion_t_0 = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                   min_roi_intensity, min_roi_length, N, grid,
                                                   rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                                   exclusion_method=ROI_EXCLUSION_WEIGHTED_DEW,
                                                   exclusion_t_0=exclusion_t_0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_qcbeer_chems_weighteddew.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestFlexibleNonOverlapController:
    def test_intensity_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset_peaks, fragscan_ps):
        logger.info('Testing flexible non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset_peaks) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset_peaks, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset_peaks, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size), IdentityDrift())
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                  min_roi_intensity, min_roi_length, N, grid,
                                                  rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems(self, fragscan_ps):
        logger.info('Testing flexible non-overlap controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                  min_roi_intensity, min_roi_length, N, grid,
                                                  rt_tol=rt_tol, min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems_and_smartROI_rules(self, fragscan_ps):
        logger.info('Testing flexible non-overlap controller with QC beer chemicals and SmartROI rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                  min_roi_intensity, min_roi_length, N, grid,
                                                  rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                                  roi_type=RoiBuilder.ROI_TYPE_SMART,
                                                  reset_length_seconds=1e6,
                                                  intensity_increase_factor=10,
                                                  drop_perc=0.1 / 100
                                                  )

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self, fragscan_ps):
        logger.info('Testing flexible non-overlap controller with QC beer chemicals and WeightedDEW rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 120
        exclusion_t_0 = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS, fragscan_ps)
        grid = GridEstimator(AllOverlapGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size),
                             IdentityDrift())
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, mz_tol, MIN_MS1_INTENSITY,
                                                  min_roi_intensity, min_roi_length, N, grid,
                                                  rt_tol=rt_tol, min_roi_length_for_fragmentation=0,
                                                  exclusion_method=ROI_EXCLUSION_WEIGHTED_DEW,
                                                  exclusion_t_0=exclusion_t_0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_qcbeer_chems_weighteddew.mzML'
        check_mzML(env, OUT_DIR, filename)
