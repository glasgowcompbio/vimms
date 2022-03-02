from loguru import logger

from tests.conftest import get_rt_bounds, CENTRE_RANGE, N_CHEMS, run_environment, \
    check_mzML, OUT_DIR, BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND
from vimms.Common import POSITIVE
from vimms.Controller import SimpleMs1Controller, AdvancedParams
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


class TestMS1Controller:
    """
    Tests the MS1 controller that does MS1 full-scans only with the simulated mass spec class.
    """

    def test_ms1_controller_with_simulated_chems(self, fragscan_dataset):
        logger.info('Testing MS1 controller with simulated chemicals')

        min_bound, max_bound = get_rt_bounds(fragscan_dataset, CENTRE_RANGE)
        logger.info('RT bounds %s %s' % (min_bound, max_bound))
        assert len(fragscan_dataset) == N_CHEMS

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, fragscan_dataset)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_bound, max_bound, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'ms1_controller_simulated_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_ms1_controller_with_qcbeer_chems(self):
        logger.info('Testing MS1 controller with QC beer chemicals')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS)
        controller = SimpleMs1Controller()

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        run_environment(env)

        # write simulated output to mzML file
        filename = 'ms1_controller_qcbeer_chems.mzML'
        check_mzML(env, OUT_DIR, filename)

    def test_peaks_in_range(self):

        min_mz = 100.
        max_mz = 200.

        logger.info('Testing MS1 controller with narrow m/z range')

        # create a simulated mass spec and MS1 controller
        mass_spec = IndependentMassSpectrometer(POSITIVE, BEER_CHEMS)
        params = AdvancedParams()
        params.default_ms1_scan_window = (min_mz, max_mz)
        controller = SimpleMs1Controller(advanced_params=params)

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
