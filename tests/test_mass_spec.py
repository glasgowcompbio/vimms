from loguru import logger

from tests.conftest import MIN_MS1_INTENSITY, check_non_empty_MS2, check_mzML, OUT_DIR, \
    BEER_CHEMS, BEER_MIN_BOUND, BEER_MAX_BOUND
from vimms.Common import POSITIVE
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer, TaskManager


class TestSimulatedMassSpec:
    """
    Tests the Top-N controller that does standard DDA Top-N fragmentation scans
    with the simulated mass spec class.
    """

    def test_mass_spec(self):
        logger.info('Testing mass spec using the Top-N controller and QC beer chemicals')

        isolation_width = 1
        N = 10
        rt_tol = 15
        mz_tol = 10
        ionisation_mode = POSITIVE

        task_manager = TaskManager(buffer_size=3)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, BEER_CHEMS,
                                                task_manager=task_manager)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                    MIN_MS1_INTENSITY)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, BEER_MIN_BOUND, BEER_MAX_BOUND, progress_bar=True)
        # run_environment(env)
        env.run()

        # check that there is at least one non-empty MS2 scan
        check_non_empty_MS2(controller)

        # write simulated output to mzML file
        filename = 'test_mass_spec.mzML'
        check_mzML(env, OUT_DIR, filename)
