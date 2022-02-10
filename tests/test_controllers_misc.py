from loguru import logger

from tests.conftest import OUT_DIR
from vimms.ChemicalSamplers import EvenMZFormulaSampler, UniformRTAndIntensitySampler, \
    ConstantChromatogramSampler, FixedMS2Sampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import DEFAULT_ISOLATION_WIDTH, POSITIVE, set_log_level_warning, \
    get_default_scan_params, get_dda_scan_param
from vimms.Controller import FixedScansController, MultiIsolationController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


class TestMultipleMS2Windows:
    def test_data_generation(self, two_fixed_chems):
        assert len(two_fixed_chems) == 2
        assert two_fixed_chems[0].mass == 100
        assert two_fixed_chems[1].mass == 200
        assert len(two_fixed_chems[0].children) > 0
        assert len(two_fixed_chems[1].children) > 0

    def test_acquisition(self, two_fixed_chems):
        mz_to_target = [chem.mass + 1.0 for chem in two_fixed_chems]
        schedule = []
        # env = Environment()
        isolation_width = DEFAULT_ISOLATION_WIDTH
        mz_tol = 0.1
        rt_tol = 15

        min_rt = 110
        max_rt = 112

        ionisation_mode = POSITIVE

        controller = FixedScansController()
        mass_spec = IndependentMassSpectrometer(ionisation_mode, two_fixed_chems)
        env = Environment(mass_spec, controller, min_rt, max_rt)

        ms1_scan = get_default_scan_params(polarity=ionisation_mode)
        ms2_scan_1 = get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width, mz_tol,
                                        rt_tol, polarity=ionisation_mode)
        ms2_scan_2 = get_dda_scan_param(mz_to_target[1], 0.0, None, isolation_width, mz_tol,
                                        rt_tol, polarity=ionisation_mode)
        ms2_scan_3 = get_dda_scan_param(mz_to_target, [0.0, 0.0], None, isolation_width, mz_tol,
                                        rt_tol, polarity=ionisation_mode)

        schedule = [ms1_scan, ms2_scan_1, ms2_scan_2, ms2_scan_3]
        controller.set_tasks(schedule)
        set_log_level_warning()
        env.run()
        assert len(controller.scans[2]) == 3

        n_peaks = []
        for scan in controller.scans[2]:
            n_peaks.append(scan.num_peaks)

        assert n_peaks[0] > 0
        assert n_peaks[1] > 0
        assert n_peaks[2] == n_peaks[0] + n_peaks[1]
        env.write_mzML(OUT_DIR, 'multi_windows.mzML')


class TestFixedScansController:
    """
    Tests the FixedScansController that sends a scheduled number of scans
    """

    def test_FixedScansController(self, two_fixed_chems):
        logger.info('Testing FixedScansController')
        mz_to_target = [chem.mass + 1.0 for chem in two_fixed_chems]
        isolation_width = DEFAULT_ISOLATION_WIDTH
        mz_tol = 0.1
        rt_tol = 15
        min_rt = 110
        max_rt = 112
        ionisation_mode = POSITIVE

        controller = FixedScansController(schedule=None)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, two_fixed_chems)
        env = Environment(mass_spec, controller, min_rt, max_rt)

        ms1_scan = get_default_scan_params(polarity=ionisation_mode)
        ms2_scan_1 = get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width,
                                        mz_tol, rt_tol, polarity=ionisation_mode)
        ms2_scan_2 = get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width,
                                        mz_tol, rt_tol, polarity=ionisation_mode)
        ms2_scan_3 = get_dda_scan_param(mz_to_target[0], 0.0, None, isolation_width,
                                        mz_tol, rt_tol, polarity=ionisation_mode)
        schedule = [ms1_scan, ms2_scan_1, ms2_scan_2, ms2_scan_3]
        controller.set_tasks(schedule)
        set_log_level_warning()
        env.run()

        assert len(controller.scans[1]) == 1
        assert len(controller.scans[2]) == 3
        for scan in controller.scans[2]:
            assert scan.num_peaks > 0
        env.write_mzML(OUT_DIR, 'fixedScansController.mzML')


class TestMultiIsolationController:
    def test_multiple_isolation(self):
        N = 3
        fs = EvenMZFormulaSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=10)
        cr = ConstantChromatogramSampler()
        ms = FixedMS2Sampler()
        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri,
                                    chromatogram_sampler=cr, ms2_sampler=ms)
        d = cs.sample(3, 2)  # sample chems with m/z = 100 and 200
        # ionisation_mode = POSITIVE
        controller = MultiIsolationController(N)
        ms = IndependentMassSpectrometer(POSITIVE, d)
        env = Environment(ms, controller, 10, 20, progress_bar=True)
        set_log_level_warning()
        env.run()

        assert len(controller.scans[1]) > 0
        assert len(controller.scans[2]) > 0

        # look at the first block of MS2 scans
        # and check that they are the correct super-positions
        mm = {}
        # first three scans hit the individual precursors
        mm[(0,)] = controller.scans[2][0]
        mm[(1,)] = controller.scans[2][1]
        mm[(2,)] = controller.scans[2][2]
        # next three should hit the pairs
        mm[(0, 1)] = controller.scans[2][3]
        mm[(0, 2)] = controller.scans[2][4]
        mm[(1, 2)] = controller.scans[2][5]
        # final should hit all three
        mm[(0, 1, 2)] = controller.scans[2][6]

        for key, value in mm.items():
            actual_mz_vals = set(mm[key].mzs)
            expected_mz_vals = set()
            for k in key:
                for m in mm[(k,)].mzs:
                    expected_mz_vals.add(m)
            assert expected_mz_vals == actual_mz_vals
