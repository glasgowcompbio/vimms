import os
import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import argparse

import numpy as np
import pylab as plt
from loguru import logger

from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController, AdvancedParams
from vimms.Environment import Environment
from vimms.Common import POSITIVE, create_if_not_exist, \
    set_log_level_warning, set_log_level_debug, save_obj
from vimms.scripts.openms_evaluate import extract_boxes, evaluate_fragmentation
from vimms.scripts.topN_test import get_input_filenames, extract_chems, extract_scan_timing


class TopNParameters:
    def __init__(self):
        # The minimum retention time for Top-N.
        self.MIN_RT = 0
        # The maximum retention time for Top-N.
        self.MAX_RT = 7200
        # The isolation window for Top-N.
        self.ISOLATION_WINDOW = 0.7
        # The Top N value.
        self.N = 15
        # The retention time tolerance for Top-N.
        self.RT_TOL = 30
        # The mass to charge ratio tolerance for Top-N.
        self.MZ_TOL = 10
        # The minimum MS1 intensity for Top-N.
        self.MIN_MS1_INTENSITY = 5000
        # The start of the default MS1 scan window.
        self.DEFAULT_MS1_SCAN_WINDOW_START = 310.0
        # The end of the default MS1 scan window.
        self.DEFAULT_MS1_SCAN_WINDOW_END = 2000.0
        # The number of times to exclude after in DEW parameters.
        self.EXCLUDE_AFTER_N_TIMES = 2
        # The exclude t0 value in DEW parameters.
        self.EXCLUDE_T0 = 15
        # Whether to perform deisotoping or not.
        self.DEISOTOPE = True
        # The start of the charge range for filtering.
        self.CHARGE_RANGE_START = 2
        # The end of the charge range for filtering.
        self.CHARGE_RANGE_END = 6
        # The minimum fit score from ms_deconvolve.
        self.MIN_FIT_SCORE = 80
        # Penalty factor for ms_deconvolve.
        self.PENALTY_FACTOR = 1.5


class TopNParametersBuilder:
    def __init__(self):
        self.topNParameters = TopNParameters()

    def set_MIN_RT(self, value):
        self.topNParameters.MIN_RT = value
        return self

    def set_MAX_RT(self, value):
        self.topNParameters.MAX_RT = value
        return self

    def set_ISOLATION_WINDOW(self, value):
        self.topNParameters.ISOLATION_WINDOW = value
        return self

    def set_N(self, value):
        self.topNParameters.N = value
        return self

    def set_RT_TOL(self, value):
        self.topNParameters.RT_TOL = value
        return self

    def set_MZ_TOL(self, value):
        self.topNParameters.MZ_TOL = value
        return self

    def set_MIN_MS1_INTENSITY(self, value):
        self.topNParameters.MIN_MS1_INTENSITY = value
        return self

    def set_DEFAULT_MS1_SCAN_WINDOW_START(self, value):
        self.topNParameters.DEFAULT_MS1_SCAN_WINDOW_START = value
        return self

    def set_DEFAULT_MS1_SCAN_WINDOW_END(self, value):
        self.topNParameters.DEFAULT_MS1_SCAN_WINDOW_END = value
        return self

    def set_EXCLUDE_AFTER_N_TIMES(self, value):
        self.topNParameters.EXCLUDE_AFTER_N_TIMES = value
        return self

    def set_EXCLUDE_T0(self, value):
        self.topNParameters.EXCLUDE_T0 = value
        return self

    def set_DEISOTOPE(self, value):
        self.topNParameters.DEISOTOPE = value
        return self

    def set_CHARGE_RANGE_START(self, value):
        self.topNParameters.CHARGE_RANGE_START = value
        return self

    def set_CHARGE_RANGE_END(self, value):
        self.topNParameters.CHARGE_RANGE_END = value
        return self

    def set_MIN_FIT_SCORE(self, value):
        self.topNParameters.MIN_FIT_SCORE = value
        return self

    def set_PENALTY_FACTOR(self, value):
        self.topNParameters.PENALTY_FACTOR = value
        return self

    def build(self):
        return self.topNParameters


def run_simulation(args, dataset, st, out_dir, out_file, pbar=False):
    # Top-N parameters
    rt_range = [(args.MIN_RT, args.MAX_RT)]
    min_rt = rt_range[0][0]
    max_rt = rt_range[0][1]
    isolation_window = args.ISOLATION_WINDOW
    N = args.N
    rt_tol = args.RT_TOL
    mz_tol = args.MZ_TOL
    min_ms1_intensity = args.MIN_MS1_INTENSITY
    min_fit_score = args.MIN_FIT_SCORE
    penalty_factor = args.PENALTY_FACTOR
    default_ms1_scan_window = (
        args.DEFAULT_MS1_SCAN_WINDOW_START, args.DEFAULT_MS1_SCAN_WINDOW_END)

    # DEW, isotope and charge filtering parameters
    exclude_after_n_times = args.EXCLUDE_AFTER_N_TIMES
    exclude_t0 = args.EXCLUDE_T0
    deisotope = args.DEISOTOPE
    charge_range = (args.CHARGE_RANGE_START, args.CHARGE_RANGE_END)

    # create controller and mass spec objects
    params = AdvancedParams(default_ms1_scan_window=default_ms1_scan_window)
    mass_spec = IndependentMassSpectrometer(POSITIVE, dataset, scan_duration=st)
    controller = TopNController(
        POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity,
        advanced_params=params, exclude_after_n_times=exclude_after_n_times,
        exclude_t0=exclude_t0, deisotope=deisotope, charge_range=charge_range,
        min_fit_score=min_fit_score, penalty_factor=penalty_factor)

    # create an environment to run both the mass spec and controller
    env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=pbar)

    # set the log level to WARNING so we don't see too many messages when environment is running
    set_log_level_warning()

    # run the simulation
    env.run()
    set_log_level_debug()
    env.write_mzML(out_dir, out_file)


def plot_heatmaps(coverage_array, intensity_array, out_dir):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    cax1 = axs[0].imshow(coverage_array, cmap='hot', interpolation='nearest')
    axs[0].set_title('Coverage Proportion')
    fig.colorbar(cax1, ax=axs[0])

    cax2 = axs[1].imshow(intensity_array, cmap='hot', interpolation='nearest')
    axs[1].set_title('Intensity Proportion')
    fig.colorbar(cax2, ax=axs[1])

    # Save the figure
    fig.savefig(os.path.join(out_dir, 'heatmap.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimise controllers on proteomics data using ViMMS')
    parser.add_argument('seed_file', type=str)
    parser.add_argument('--method', type=str, default='topN')

    parser.add_argument('--at_least_one_point_above', type=float, default=1E5,
                        help='The minimum intensity value for ROI extraction.')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='The number of bins to sample scan durations from.')

    parser.add_argument('--out_dir', type=str, default='topN_optimise',
                        help='The directory where the output files will be stored.')
    parser.add_argument('--openms_dir', type=str, default=None)
    parser.add_argument('--openms_ini_file', type=str, default=None)
    parser.add_argument('--isolation_width', type=float, default=0.7, help="Isolation width for fragmentation.")
    args = parser.parse_args()

    csv_file = extract_boxes(args.seed_file, args.openms_dir, args.openms_ini_file)

    # check input and output paths
    assert os.path.isfile(args.seed_file), 'Input mzML file %s is not found!' % args.seed_file
    out_dir = os.path.abspath(args.out_dir)
    create_if_not_exist(out_dir)

    # Format output file names
    chem_file, st_file = get_input_filenames(args.at_least_one_point_above, out_dir)

    # extract chems and scan timing from mzml file
    dataset = extract_chems(args.seed_file, chem_file, args.at_least_one_point_above)
    st = extract_scan_timing(args.seed_file, st_file, args.num_bins)

    N_values = [5, 10, 15, 20, 25, 30]
    RT_TOL_values = [5, 10, 15, 30, 60, 120, 180, 240, 300]
    pbar = True

    results = {}
    coverage_array = np.zeros((len(N_values), len(RT_TOL_values)))
    intensity_array = np.zeros((len(N_values), len(RT_TOL_values)))

    for i, n in enumerate(N_values):
        for j, rt_tol in enumerate(RT_TOL_values):
            params = (TopNParametersBuilder()
                      .set_N(n)
                      .set_RT_TOL(rt_tol)
                      .build())

            # your simulation code here...
            out_file = f'topN_N_{params.N}_DEW_{params.RT_TOL}.mzML'
            run_simulation(params, dataset, st, out_dir, out_file, pbar)

            mzml_file = os.path.join(out_dir, out_file)
            logger.info(f'Now processing fragmentation file {mzml_file}')
            eva = evaluate_fragmentation(csv_file, mzml_file, args.isolation_width)
            report = eva.evaluation_report()

            key = (n, rt_tol)
            results[key] = report
            print(key, eva.summarise())

            coverage_prop = report['cumulative_coverage_proportion']
            intensity_prop = report['cumulative_intensity_proportion']
            coverage_array[i, j] = coverage_prop[0]
            intensity_array[i, j] = intensity_prop[0]

    save_obj(results, os.path.join(out_dir, 'topN_optimise_results.p'))
    save_obj(coverage_array, os.path.join(out_dir, 'topN_coverage_array.p'))
    save_obj(intensity_array, os.path.join(out_dir, 'topN_intensity_array.p'))

    plot_heatmaps(coverage_array, intensity_array, out_dir)