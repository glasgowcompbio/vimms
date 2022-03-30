import numpy as np
from loguru import logger

from vimms.Common import POSITIVE, ROI_EXCLUSION_WEIGHTED_DEW
from vimms.MassSpec import IndependentMassSpectrometer, Scan
from vimms.Environment import Environment
from vimms.Roi import RoiBuilder, RoiBuilderParams, SmartRoiParams
from vimms.Box import BoxGrid
from vimms.BoxManager import BoxSplitter, BoxManager
from vimms.Controller import (
    TopN_RoiController, TopN_SmartRoiController, 
    NonOverlapController, IntensityNonOverlapController, FlexibleNonOverlapController
)
from tests.conftest import (
    N_CHEMS, MIN_MS1_INTENSITY, get_rt_bounds, CENTRE_RANGE,
    run_environment, check_non_empty_MS2, check_mzML, OUT_DIR, 
    BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND
)


def delete_point(scans, mz, rt):
    # find scan
    filtered = [sc for sc in scans if sc.rt == rt]
    scan = filtered[0]

    # delete point
    idx = np.where(scan.mzs == mz)[0][0]
    scan.mzs = np.delete(scan.mzs, idx)
    scan.intensities = np.delete(scan.intensities, idx)
    scan.num_peaks = len(scan.mzs)


class TestRoiBuilder:
    """
    Tests the codes that performs ROI building
    """

    def test_simple(self):
        """
        Test the simple case of building ROIs without any gaps in the scans
        """

        # create some scans in 3 contiguous ROIs
        ms_level = 1
        mzs = np.array([100, 101, 102])
        intensities = np.array([25, 50, 75])

        scans = []
        for rt in range(20):
            sc = Scan(1, mzs, intensities, ms_level, rt)
            scans.append(sc)

        # Build ROIs
        roi_params = RoiBuilderParams()
        roi_builder = RoiBuilder(roi_params)
        for sc in scans:
            roi_builder.update_roi(sc)
        rois = roi_builder.get_good_rois()
        assert len(rois) == 3

    def test_gaps(self):
        """
        Test the case of building ROIs with gaps in the scans
        """

        # create some scans with gaps
        ms_level = 1
        mzs = np.array([100, 101, 102])
        intensities = np.array([25, 50, 75])

        scans = []
        for rt in range(20):
            sc = Scan(1, mzs, intensities, ms_level, rt)
            scans.append(sc)

        # Introduce one gap
        delete_point(scans, 100, 10)

        # Build ROIs with no gap-filling.
        # We should get 4 ROIs as one ROI (at m/z 100) has been broken into two parts.
        roi_params = RoiBuilderParams(max_gaps_allowed=0)
        roi_builder = RoiBuilder(roi_params)
        for sc in scans:
            roi_builder.update_roi(sc)
        rois = roi_builder.get_good_rois()
        assert len(rois) == 4

        # Build ROIs with `max_skips_allowed` set to 1.
        # Here we still get 3 ROIs due to gap-filling.
        roi_params = RoiBuilderParams(max_gaps_allowed=1)
        roi_builder = RoiBuilder(roi_params)
        for sc in scans:
            roi_builder.update_roi(sc)
        rois = roi_builder.get_good_rois()
        assert len(rois) == 3

        # Introduce more gaps
        delete_point(scans, 100, 5)
        delete_point(scans, 100, 15)
        delete_point(scans, 102, 10)
        delete_point(scans, 102, 11)
        delete_point(scans, 102, 12)

        # Build ROIs with no gap-filling.
        # We should get 7 ROIs as the bottom ROI (at m/z 100) is now split into 4 parts, and
        # the top ROI (at m/z 102) is split into two with large gaps.
        roi_params = RoiBuilderParams(max_gaps_allowed=0)
        roi_builder = RoiBuilder(roi_params)
        for sc in scans:
            roi_builder.update_roi(sc)
        rois = roi_builder.get_good_rois()
        assert len(rois) == 7

        # Build ROIs with `max_skips_allowed` set to 3.
        # Here we still get 3 ROIs due to gap-filling.
        roi_params = RoiBuilderParams(max_gaps_allowed=3)
        roi_builder = RoiBuilder(roi_params)
        for sc in scans:
            roi_builder.update_roi(sc)
        rois = roi_builder.get_good_rois()
        assert len(rois) == 3


class TestROIController:
    """
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on
    selecting regions of interests (rather than the top-N most intense peaks)
    """

    def test_roi_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset) == N_CHEMS

        for f in fragscan_dataset:
            assert len(f.children) > 0

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 2
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)

        # def __init__(self,
        #              ionisation_mode,
        #              isolation_width,
        #              N,
        #              mz_tol,
        #              rt_tol,
        #              min_ms1_intensity,
        #              roi_params,
        #              min_roi_length_for_fragmentation=0,
        #              ms1_shift=0,
        #              advanced_params=None,
        #              exclusion_method=ROI_EXCLUSION_DEW,
        #              exclusion_t_0=None):

        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = TopN_RoiController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                        MIN_MS1_INTENSITY, roi_params)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'roi_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = TopN_RoiController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                        MIN_MS1_INTENSITY, roi_params)

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
    Tests the ROI controller that performs fragmentations and dynamic exclusions based on
    selecting regions of interests (rather than the top-N most intense peaks)
    """

    def test_smart_roi_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing ROI controller with simulated chemicals')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)

        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                             MIN_MS1_INTENSITY, roi_params, smartroi_params,
                                             min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'smart_roi_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_smart_roi_controller_with_beer_chems(self):
        logger.info('Testing ROI controller with QC beer chemicals')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)

        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                             MIN_MS1_INTENSITY, roi_params, smartroi_params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'smart_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestNonOverlapController:
    def test_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)

        grid = BoxManager(
            box_geometry=BoxGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = NonOverlapController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                          MIN_MS1_INTENSITY, roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)

        grid = BoxManager(
            box_geometry=BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = NonOverlapController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                          MIN_MS1_INTENSITY, roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems_and_smartROI_rules(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)

        grid = BoxManager(
            box_geometry=BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = NonOverlapController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                          MIN_MS1_INTENSITY, roi_params, grid,
                                          smartroi_params=smartroi_params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)

        grid = BoxManager(
            box_geometry=BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = NonOverlapController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                          MIN_MS1_INTENSITY, roi_params, grid,
                                          exclusion_method=ROI_EXCLUSION_WEIGHTED_DEW,
                                          exclusion_t_0=exclusion_t_0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'non_overlap_qcbeer_chems_weighteddew.mzML'
        check_mzML(env, OUT_DIR, filename)


class TestIntensityNonOverlapController:
    def test_intensity_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing intensity non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)
        grid = BoxManager(
            box_geometry=BoxGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size),
            box_splitter=BoxSplitter(split=True)
        )

        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                   rt_tol, MIN_MS1_INTENSITY, roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                   rt_tol, MIN_MS1_INTENSITY, roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems_and_smartROI_rules(self):
        logger.info('Testing intensity non-overlap controller with QC beer chemicals '
                    'and SmartROI rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                   rt_tol, MIN_MS1_INTENSITY, roi_params, grid,
                                                   smartroi_params=smartroi_params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'intensity_non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_intensity_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self):
        logger.info('Testing intensity non-overlap controller with QC beer chemicals '
                    'and WeightedDEW rules')

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = IntensityNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                   rt_tol, MIN_MS1_INTENSITY, roi_params, grid,
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
    def test_intensity_nonoverlap_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing flexible non-overlap controller with simulated chemicals')
        assert len(fragscan_dataset) == N_CHEMS

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 50
        min_roi_length = 0
        ionisation_mode = POSITIVE
        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, fragscan_dataset)
        grid = BoxManager(
            box_geometry=BoxGrid(min_bound, max_bound, rt_box_size, 0, 3000, mz_box_size),
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = FlexibleNonOverlapController(
            ionisation_mode, isolation_width, N, mz_tol, rt_tol, MIN_MS1_INTENSITY,
            roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        assert len(controller.scans[2]) > 0

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems(self):
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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                  rt_tol, MIN_MS1_INTENSITY, roi_params, grid)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems_and_smartROI_rules(self):
        logger.info('Testing flexible non-overlap controller with QC beer chemicals and '
                    'SmartROI rules')

        isolation_width = 1  # the isolation window in Dalton around a selected precursor ion
        N = 10
        rt_tol = 15
        mz_tol = 10
        min_roi_intensity = 5000
        min_roi_length = 10
        ionisation_mode = POSITIVE
        rt_box_size, mz_box_size = 1, 0.3

        # create a simulated mass spec with noise and ROI controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                  rt_tol, MIN_MS1_INTENSITY, roi_params, grid,
                                                  smartroi_params=smartroi_params)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'flexible_non_overlap_qcbeer_chems_smartroi.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_flexible_non_overlap_controller_with_beer_chems_and_weighteddew_rules(self):
        logger.info('Testing flexible non-overlap controller with QC beer chemicals and '
                    'WeightedDEW rules')

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
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS)
        bg = BoxGrid(BEER_MIN_BOUND, BEER_MAX_BOUND, rt_box_size, 0, 3000, mz_box_size)
        grid = BoxManager(
            box_geometry=bg,
            box_splitter=BoxSplitter(split=True)
        )
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        controller = FlexibleNonOverlapController(ionisation_mode, isolation_width, N, mz_tol,
                                                  rt_tol, MIN_MS1_INTENSITY, roi_params, grid,
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
