import os

from loguru import logger

from tests.conftest import OUT_DIR
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, UniformMZFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator, MultipleMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Utils import write_msp
from vimms.scripts.check_ms2_matches import main as ms2_main
from vimms.scripts.optimal_performance import setup_scans, make_edges_chems, \
    reducedUnweightedMaxMatchingFromLists


class TestMS2Matching:
    def test_ms2_matching(self):
        rti = UniformRTAndIntensitySampler(min_rt=10, max_rt=20)
        fs = UniformMZFormulaSampler()
        adduct_prior_dict = {POSITIVE: {'M+H': 1}}

        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=rti,
                                    adduct_prior_dict=adduct_prior_dict)
        d = cs.sample(300, 2)

        group_list = ['control', 'control', 'case', 'case']
        group_dict = {}
        group_dict['control'] = {'missing_probability': 0.0, 'changing_probability': 0.0}
        group_dict['case'] = {'missing_probability': 0.0, 'changing_probability': 1.0}

        mm = MultipleMixtureCreator(d, group_list, group_dict)

        cl = mm.generate_chemical_lists()

        N = 10
        isolation_width = 0.7
        mz_tol = 0.001
        rt_tol = 30
        min_ms1_intensity = 0

        set_log_level_warning()

        output_folder = os.path.join(OUT_DIR, 'ms2_matching')
        write_msp(d, 'mmm.msp', out_dir=output_folder)

        initial_exclusion_list = []
        for i, chem_list in enumerate(cl):
            controller = TopNController(
                POSITIVE, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                initial_exclusion_list=initial_exclusion_list)
            ms = IndependentMassSpectrometer(POSITIVE, chem_list)
            env = Environment(ms, controller, 10, 30, progress_bar=True)
            env.run()
            env.write_mzML(output_folder, '{}.mzML'.format(i))

            mz_intervals = list(controller.exclusion.exclusion_list.boxes_mz.items())
            rt_intervals = list(controller.exclusion.exclusion_list.boxes_rt.items())
            unique_items_mz = set(i.data for i in mz_intervals)
            unique_items_rt = set(i.data for i in rt_intervals)
            assert len(unique_items_mz) == len(unique_items_rt)

            initial_exclusion_list = list(unique_items_mz)
            logger.warning(len(initial_exclusion_list))

        set_log_level_debug()
        msp_file = os.path.join(output_folder, 'mmm.msp')
        # check with just the first file
        a, b = ms2_main(os.path.join(output_folder, '0.mzML'), msp_file, 1, 0.7)
        # check with all
        c, d = ms2_main(output_folder, os.path.join(output_folder, 'mmm.msp'), 1, 0.7)

        assert b == d
        assert c > a


class TestChemEdges:
    def test_chem_edges(self, ten_chems):
        set_log_level_debug()
        min_ms1_intensity = 1e3
        min_rt = 200
        max_rt = 300
        N = 10
        scan_duration_dict = {1: 0.6, 2: 0.2}
        scan_levels, scan_start_times = setup_scans(scan_duration_dict, N, min_rt, max_rt)
        edges = make_edges_chems(ten_chems, scan_start_times, scan_levels, min_ms1_intensity)

        scan_names, box_names, _ = zip(*edges)
        scanSet = set(scan_names)
        boxSet = set(box_names)
        reduced_edges = list(zip(scan_names, box_names))
        matchList, size = reducedUnweightedMaxMatchingFromLists(scanSet, boxSet, reduced_edges)
        print("The matching has size: {}".format(size))
