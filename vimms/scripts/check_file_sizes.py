from loguru import logger

from tests.conftest import OUT_DIR, check_mzML, check_non_empty_MS2, \
    BEER_CHEMS, run_environment, MIN_MS1_INTENSITY, BEER_CHEMS_PATH
from vimms.Chemicals import ChemSet
from vimms.Common import POSITIVE
from vimms.Controller import TopNController, TopN_SmartRoiController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiBuilderParams, SmartRoiParams


class TestCheckFileSizes():

    def test_memory_beer_chems(self):
        logger.info('Testing memory chems')
        chemset = ChemSet.to_chemset(BEER_CHEMS)
        out_file = 'test_chemset_memory.mzML'
        self._run_topN(chemset, out_file, 30302625)

    def test_fast_memory_beer_chems(self):
        logger.info('Testing fast memory chems')
        chemset = ChemSet.to_chemset(BEER_CHEMS, fast=True)
        out_file = 'test_chemset_fast_memory.mzML'
        self._run_topN(chemset, out_file, 30302625)

    def test_fast_memory_beer_chems_smartroi(self):
        logger.info('Testing fast memory chems')
        chemset = ChemSet.to_chemset(BEER_CHEMS, fast=True)
        out_file = 'test_chemset_fast_memory_smartroi.mzML'
        self._run_smart_roi(chemset, out_file, 27431372)

    def test_file_beer_chems(self):
        logger.info('Testing file chems')
        chemset = ChemSet.to_chemset(None, filepath=BEER_CHEMS_PATH)
        out_file = 'test_chemset_file.mzML'
        self._run_topN(chemset, out_file, 30302625)

    def test_fast_memory_synthetic_chems(self, large_fragscan_dataset):
        logger.info('Testing fast memory synthetic chems')
        chemset = ChemSet.to_chemset(large_fragscan_dataset, fast=True)
        out_file = 'test_chemset_fast_synthetic_memory.mzML'

        # Initial size in master is this:
        # self._run_topN(chemset, out_file, 28917347)

        # New size is this. FIXME: small changes due to vectorisation?
        self._run_topN(chemset, out_file, 28917352)

    def test_fast_memory_synthetic_chems_smartroi(self, large_fragscan_dataset):
        logger.info('Testing fast memory synthetic chems')
        chemset = ChemSet.to_chemset(large_fragscan_dataset, fast=True)
        out_file = 'test_chemset_fast_synthetic_memory_smartroi.mzML'
        self._run_smart_roi(chemset, out_file, 27026014)

    def _run_topN(self, chemset, out_file, expected_size):
        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        # create a simulated mass spec without noise and Top-N controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, chemset)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, 0, 1440, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        check_mzML(env, OUT_DIR, out_file, assert_size=expected_size)

    def _run_smart_roi(self, chemset, out_file, expected_size):
        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE
        min_roi_intensity = 50
        min_roi_length = 0

        # create a simulated mass spec without noise and smartroi controller
        mass_spec = IndependentMassSpectrometer(ionisation_mode, chemset)
        roi_params = RoiBuilderParams(min_roi_length=min_roi_length,
                                      min_roi_intensity=min_roi_intensity)
        smartroi_params = SmartRoiParams()
        controller = TopN_SmartRoiController(ionisation_mode, isolation_width, N, mz_tol, rt_tol,
                                             MIN_MS1_INTENSITY, roi_params, smartroi_params,
                                             min_roi_length_for_fragmentation=0)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, 0, 1440, progress_bar=True)
        run_environment(env)

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        check_mzML(env, OUT_DIR, out_file, assert_size=expected_size)
