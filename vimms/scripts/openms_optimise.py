import os
import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import argparse

import numpy as np
import pylab as plt
import seaborn as sns
from loguru import logger
import optuna

from optuna.visualization import plot_optimization_history, plot_param_importances

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


class TopNSimulator:
    def __init__(self, args, out_dir, pbar=False):
        self.args = args
        self.out_dir = out_dir
        self.pbar = pbar

        # for grid search
        self.N_values = [5, 10, 15, 20, 25, 30]
        self.RT_TOL_values = [5, 10, 15, 30, 60, 120, 180, 240, 300]
        self.EXCLUDE_T0_value = 15
        self.results = {}
        self.coverage_array = np.zeros((len(self.N_values), len(self.RT_TOL_values)))
        self.intensity_array = np.zeros((len(self.N_values), len(self.RT_TOL_values)))

    def simulate(self, n, rt_tol, exclude_t0):
        params = (TopNParametersBuilder()
                  .set_N(n)
                  .set_RT_TOL(rt_tol)
                  .set_EXCLUDE_T0(exclude_t0)
                  .build())

        # your simulation code here...
        out_file = f'topN_N_{params.N}_DEW_{params.RT_TOL}_exclude_t0_{params.EXCLUDE_T0}.mzML'
        self.run_simulation(params, dataset, st, self.out_dir, out_file)
        mzml_file = os.path.join(self.out_dir, out_file)

        logger.debug(f'Now processing fragmentation file {mzml_file}')
        eva = evaluate_fragmentation(csv_file, mzml_file, params.ISOLATION_WINDOW)
        logger.debug(f'N={n} RT_TOL={rt_tol} exclude_t0={exclude_t0}')
        logger.debug(eva.summarise(min_intensity=params.MIN_MS1_INTENSITY))

        report = eva.evaluation_report(min_intensity=params.MIN_MS1_INTENSITY)
        return report

    def run_simulation(self, params, dataset, st, out_dir, out_file):
        # Top-N parameters
        rt_range = [(params.MIN_RT, params.MAX_RT)]
        min_rt = rt_range[0][0]
        max_rt = rt_range[0][1]
        isolation_window = params.ISOLATION_WINDOW
        N = params.N
        rt_tol = params.RT_TOL
        mz_tol = params.MZ_TOL
        min_ms1_intensity = params.MIN_MS1_INTENSITY
        min_fit_score = params.MIN_FIT_SCORE
        penalty_factor = params.PENALTY_FACTOR
        default_ms1_scan_window = (
            params.DEFAULT_MS1_SCAN_WINDOW_START, params.DEFAULT_MS1_SCAN_WINDOW_END)

        # DEW, isotope and charge filtering parameters
        exclude_after_n_times = params.EXCLUDE_AFTER_N_TIMES
        exclude_t0 = params.EXCLUDE_T0
        deisotope = params.DEISOTOPE
        charge_range = (params.CHARGE_RANGE_START, params.CHARGE_RANGE_END)

        # create controller and mass spec objects
        params = AdvancedParams(default_ms1_scan_window=default_ms1_scan_window)
        mass_spec = IndependentMassSpectrometer(POSITIVE, dataset, scan_duration=st)
        controller = TopNController(
            POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity,
            advanced_params=params, exclude_after_n_times=exclude_after_n_times,
            exclude_t0=exclude_t0, deisotope=deisotope, charge_range=charge_range,
            min_fit_score=min_fit_score, penalty_factor=penalty_factor)

        # create an environment to run both the mass spec and controller
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=self.pbar)

        # set the log level to WARNING so we don't see too many messages when environment is running
        set_log_level_warning()

        # run the simulation
        env.run()
        set_log_level_debug()
        env.write_mzML(out_dir, out_file)

    def grid_search(self):
        logger.debug(f'Performing grid search using N={self.N_values} and rt_tol={self.RT_TOL_values}')
        exclude_t0 = self.EXCLUDE_T0_value
        for i, n in enumerate(self.N_values):
            for j, rt_tol in enumerate(self.RT_TOL_values):
                # simulate and evaluate the combination of N and RT_TOL
                report = self.simulate(n, rt_tol, exclude_t0)
                self.results[(n, rt_tol, exclude_t0)] = report

                # store the results
                coverage_prop = report['cumulative_coverage_proportion']
                intensity_prop = report['cumulative_intensity_proportion']
                self.coverage_array[i, j] = coverage_prop[0]
                self.intensity_array[i, j] = intensity_prop[0]

    def save_grid_search_results(self):
        logger.debug(f'Saving grid search results to {self.out_dir}')

        # save pickled results
        data = {
            'topN_optimise_results.p': self.results,
            'topN_coverage_array.p': self.coverage_array,
            'topN_intensity_array.p': self.intensity_array
        }
        for filename, data_obj in data.items():
            save_obj(data_obj, os.path.join(self.out_dir, filename))

        # save heatmap
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        data = [
            (self.coverage_array, 'Coverage Proportion'),
            (self.intensity_array, 'Intensity Proportion')
        ]
        for i, (array, title) in enumerate(data):
            sns.heatmap(array, ax=axs[i], cbar_ax=axs[i].inset_axes([1.05, 0.1, 0.05, 0.8]))
            axs[i].set_title(title)
            axs[i].set_xticklabels(self.RT_TOL_values)
            axs[i].set_yticklabels(self.N_values)
            axs[i].set_xlabel('RT TOL')
            axs[i].set_ylabel('N')

        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, 'heatmap.png'), dpi=300)

    def objective(self, trial):
        # define the space for hyperparameters
        n = trial.suggest_int('N', 5, 30, step=5)
        rt_tol = trial.suggest_int('RT_TOL', 5, 300, step=5)
        exclude_t0 = trial.suggest_int('EXCLUDE_t0', 5, 60, step=5)

        # simulate and evaluate the combination of N and RT_TOL
        report = self.simulate(n, rt_tol, exclude_t0)
        self.results[(n, rt_tol, exclude_t0)] = report

        # Access args.optimize
        if self.args.optuna_optimise == 'coverage_prop':
            return report['cumulative_coverage_proportion'][0]  # Optuna minimizes by default
        elif self.args.optuna_optimise == 'intensity_prop':
            return report['cumulative_intensity_proportion'][0]  # Optuna minimizes by default
        else:
            raise ValueError(f"Invalid optimisation choice: {self.args.optuna_optimise}. "
                             f"Choose 'coverage_prop' or 'intensity_prop'.")


def save_study(study, results, out_dir):
    trial = study.best_trial
    logger.info(f'Number of finished trials: {len(study.trials)}')
    logger.info(f'Best trial value: {trial.value}')
    logger.info('Best trial params: ')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')

    # save pickled results
    save_obj(results, os.path.join(out_dir, 'topN_optimise_results.p'))

    # Write report csv and plots
    study.trials_dataframe().to_csv(os.path.join(out_dir, f'study.csv'))
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(out_dir, f'study_optimisation_history.png'))
    fig2 = plot_param_importances(study)
    fig2.write_image(os.path.join(out_dir, f'study_param_importances.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimise controllers on proteomics data using ViMMS')

    # chemical extraction and simulation parameters
    parser.add_argument('seed_file', type=str)
    parser.add_argument('--method', type=str, default='topN')  # valid choices are 'topN', 'SmartROI' or 'WeightedDEW'
    parser.add_argument('--at_least_one_point_above', type=float, default=1E5,
                        help='The minimum intensity value for ROI extraction.')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='The number of bins to sample scan durations from.')
    parser.add_argument('--pbar', type=bool, default=True, help='Show progress bar during simulation.')

    # evaluation parameters
    parser.add_argument('--out_dir', type=str, default='topN_optimise',
                        help='The directory where the output files will be stored.')
    parser.add_argument('--openms_dir', type=str, default=None)
    parser.add_argument('--openms_ini_file', type=str, default=None)
    parser.add_argument('--isolation_width', type=float, default=0.7, help="Isolation width for fragmentation.")

    # optimisation parameters
    parser.add_argument('--optuna_use', type=bool, default=False, help='Use Optuna for optimisation.')
    parser.add_argument('--optuna_optimise', type=str, default='intensity_prop',
                        help="For optuna, optimise for either 'coverage_prop' or 'intensity_prop'.")
    parser.add_argument('--optuna_n_trials', type=int, default=100,
                        help='For optuna, the number of trials.')

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

    simulator = TopNSimulator(args, out_dir, pbar=args.pbar)
    if args.optuna_use:
        study = optuna.create_study(direction='maximize')
        study.optimize(simulator.objective, n_trials=args.optuna_n_trials)
        save_study(study, simulator.results, out_dir)

    else:  # grid search
        simulator.grid_search()
        simulator.save_grid_search_results()
