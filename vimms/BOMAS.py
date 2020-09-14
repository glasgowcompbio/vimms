import itertools as it
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pylab as plt
from loguru import logger
from pyDOE import lhs
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# from vimms.Common import *
# from vimms.Controller import *
# from vimms.Environment import *
from vimms.Common import POSITIVE, load_obj, save_obj
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.FeatureExtraction import extract_roi
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.PythonMzmine import peak_scoring
from vimms.PythonMzmine import pick_peaks, controller_score

MZMINE_COMMAND = 'C:\\Users\\Vinny\\work\\MZmine-2.40.1\\MZmine-2.40.1\\startMZmine_Windows.bat'
XML_TEMPLATE_MS1 = 'C:\\Users\\Vinny\\work\\vimms\\batch_files\\mzmine_batch_ms1.xml'
XML_TEMPLATE_MS2 = 'C:\\Users\\Vinny\\work\\vimms\\batch_files\\mzmine_batch_ms2.xml'
QCB_XML_TEMPLATE_MS1 = 'C:\\Users\\Vinny\\work\\vimms\\batch_files\\QCB_mzmine_batch_ms1.xml'
QCB_XML_TEMPLATE_MS2 = 'C:\\Users\\Vinny\\work\\vimms\\batch_files\\QCB_mzmine_batch_ms2.xml'

QCB_MZML2CHEMS_DICT = {'min_ms1_intensity': 1.75E5,
                       'mz_tol': 2,
                       'mz_units': 'ppm',
                       'min_length': 1,
                       'min_intensity': 0,
                       'start_rt': 0,
                       'stop_rt': 1560}

MADELEINE_MZML2CHEMS_DICT = {'min_ms1_intensity': 1.75E5,
                             'mz_tol': 10,
                             'mz_units': 'ppm',
                             'min_length': 1,
                             'min_intensity': 0,
                             'start_rt': 0,
                             'stop_rt': 600}

QCB_TOP_N_CONTROLLER_PARAM_DICT = {"ionisation_mode": POSITIVE,
                                   "mz_tol": 10,
                                   "min_ms1_intensity": 1.75E5,
                                   "rt_range": [(0, 1560)],
                                   "isolation_width": 1}

MADELEINE_TOP_N_CONTROLLER_PARAM_DICT = {"ionisation_mode": POSITIVE,
                                         "mz_tol": 10,
                                         "min_ms1_intensity": 1.75E5,
                                         "rt_range": [(0, 600)],
                                         "isolation_width": 1}

QCB_SCORE_PARAM_DICT = {'min_ms1_intensity': 1.75E5,
                        'matching_mz_tol': 30,
                        'matching_rt_tol': 10}


class BaseOptimiser(object):
    def __init__(self, base_dir, ps,
                 controller_method='TopNController',
                 chem_param_dict=QCB_MZML2CHEMS_DICT,
                 controller_param_dict=QCB_TOP_N_CONTROLLER_PARAM_DICT,
                 score_param_dict=QCB_SCORE_PARAM_DICT,
                 min_ms1_intensity=1.75E5,
                 ms1_mzml=None,
                 ms1_picked_peaks_file=None,
                 dataset_file=None,
                 add_noise=True,
                 xml_template_ms1=XML_TEMPLATE_MS1,
                 xml_template_ms2=XML_TEMPLATE_MS2,
                 mzmine_command=MZMINE_COMMAND,
                 use_parallel=False):  # TODO: change to True once implemented

        self.xml_template_ms1 = xml_template_ms1
        self.xml_template_ms2 = xml_template_ms2
        self.mzmine_command = mzmine_command
        self.controller_method = controller_method
        self.controller_param_dict = controller_param_dict
        self.score_param_dict = score_param_dict
        self.use_parallel = use_parallel

        self.method_name = self.__class__.__name__
        self.min_ms1_intensity = min_ms1_intensity

        self.base_name = self._get_base_name(ms1_mzml, ms1_picked_peaks_file, dataset_file)
        self.output_dir, self.ms2_dir, self.picked_peaks_dir = self._get_directories(base_dir, controller_method)
        self.dataset, self.ms1_picked_peaks_file = self._get_data(ms1_mzml, ms1_picked_peaks_file, dataset_file, ps,
                                                                  chem_param_dict)

        # set up mass spec
        self.mass_spec = IndependentMassSpectrometer(POSITIVE, self.dataset, ps, add_noise=add_noise)

    def run(self):
        # get initial params and set up pandas dataframe
        self.initial_params, self.results = self._get_initial_params()

        # run optimisation algorithm
        self.results = self.run_initial_simulations()
        self.results = self.run_optimiser_simulations()

        # save results
        df_name = self.base_name + '_df_results'
        self.results.to_pickle(self.output_dir + '\\' + df_name + '.p')

    def run_initial_simulations(self):
        if self.use_parallel:
            # TODO: not working yet
            # results = run_parallel_controller(self.controller_method, search_param_dict, base_param_dict)
            # return results
            pass
        else:
            results = self.results
            for idx in range(len(self.initial_params)):
                next_flex_param_dict = self.initial_params[idx]
                new_results = self._get_controller_score(self.mass_spec, self.ms1_picked_peaks_file,
                                                         self.controller_method,
                                                         self.controller_param_dict, next_flex_param_dict, idx)
                results.loc[len(self.results)] = new_results
            return results

    def run_optimiser_simulations(self):
        NotImplementedError()

    def _get_parallel_params(self):
        NotImplementedError()

    def _run_parrallel(self):
        NotImplementedError()

    def _get_base_name(self, ms1_mzml, ms1_picked_peaks_file, dataset_file):
        if ms1_mzml is not None:
            base_name = Path(ms1_mzml).stem
        elif ms1_picked_peaks_file is not None:
            base_name = Path(ms1_picked_peaks_file).stem
        elif dataset_file is not None:
            base_name = Path(dataset_file).stem
        else:
            sys.exit('No data provided')
        return base_name

    def _get_directories(self, base_dir, controller_method):
        # make and set directories
        output_dir = base_dir + '\\' + self.base_name + '_' + self.method_name + '_' + controller_method
        os.mkdir(output_dir)
        ms2_dir = output_dir + '\\ms2_dir'
        os.mkdir(ms2_dir)
        picked_peaks_dir = output_dir + '//picked_peaks'
        os.mkdir(picked_peaks_dir)
        return output_dir, ms2_dir, picked_peaks_dir

    def _get_data(self, ms1_mzml, ms1_picked_peaks_file, dataset_file, ps, chem_param_dict):
        # Load data
        if dataset_file is None:
            datasets = extract_roi([ms1_mzml], None, None, None, ps, param_dict=chem_param_dict)
            dataset = datasets[0]
        else:
            dataset = load_obj(dataset_file)
        if ms1_picked_peaks_file is None and self.method_name != 'RepeatedExperiment':
            pick_peaks([ms1_mzml], xml_template=self.xml_template_ms1, output_dir=self.picked_peaks_dir,
                       mzmine_command=self.mzmine_command)
            ms1_picked_peaks_file = self.picked_peaks_dir + '\\' + self.base_name + '.csv'

        return dataset, ms1_picked_peaks_file

    def _get_initial_params(self):
        NotImplementedError()

    def _get_controller_score(self, mass_spec, ms1_picked_peaks_file, controller_method,
                              controller_param_dict, next_flex_param_dict, idx):
        env, controller_name = self._run_controller(mass_spec, controller_method, controller_param_dict,
                                                    next_flex_param_dict, idx)
        ms2_picked_peaks_file = self.frag_peak_picking(controller_name)
        score = self._get_score(env, ms1_picked_peaks_file, ms2_picked_peaks_file, next_flex_param_dict)
        return score

    def _run_controller(self, mass_spec, controller_method, controller_param_dict, next_flex_param_dict, idx):
        controller = self._get_controller(controller_method, controller_param_dict, next_flex_param_dict)
        env = Environment(mass_spec, controller, controller_param_dict['rt_range'][0][0],
                          controller_param_dict['rt_range'][0][1], progress_bar=True)
        env.run()
        controller_name = 'controller_' + "{:05d}".format(idx)
        env.write_mzML(self.ms2_dir, controller_name + '.mzml')
        save_obj(controller, os.path.join(self.ms2_dir, controller_name + '.p'))
        return env, controller_name

    def frag_peak_picking(self, controller_name):
        file_list = [os.path.join(self.ms2_dir, controller_name + '.mzml')]
        pick_peaks(file_list, xml_template=self.xml_template_ms2, output_dir=self.picked_peaks_dir,
                   mzmine_command=self.mzmine_command)
        ms2_picked_peaks_file = self.picked_peaks_dir + '\\' + controller_name + '_pp.csv'
        return ms2_picked_peaks_file

    def _get_score(self, env, ms1_picked_peaks_file, ms2_picked_peaks_file, next_flex_param_dict):
        score_count, score_int, prec, rec, f1 = peak_scoring(env.controller, ms1_picked_peaks_file,
                                                             ms2_picked_peaks_file,
                                                             self.dataset, self.min_ms1_intensity,
                                                             self.score_param_dict)
        if self.scoring_method is None:
            return [score_count, score_int, prec, rec, f1] + list(next_flex_param_dict.values())
        elif self.scoring_method == 'Peak Count':
            return [score_count] + list(next_flex_param_dict.values())
        elif self.scoring_method == 'Log Peak Intensity':
            return [score_int] + list(next_flex_param_dict.values())

    def _get_controller(self, controller_method, controller_param_dict, flex_controller_param_dict):
        if controller_method == 'TopNController':
            controller = TopNController(controller_param_dict["ionisation_mode"],
                                        # TODO: make this more general so it can find params from either dictionary
                                        flex_controller_param_dict['N'],
                                        controller_param_dict["isolation_width"],
                                        controller_param_dict["mz_tol"],
                                        flex_controller_param_dict['DEW'],
                                        controller_param_dict["min_ms1_intensity"])
        # elif controller_method == 'Repeated_RoiController':
        #     controller = Repeated_RoiController(controller_param_dict["ionisation_mode"],
        #                                         controller_param_dict["isolation_width"],
        #                                         controller_param_dict["mz_tol"],
        #                                         controller_param_dict["min_ms1_intensity"],
        #                                         controller_param_dict["min_roi_intensity"],
        #                                         controller_param_dict["min_roi_length"],
        #                                         controller_param_dict["N"],
        #                                         controller_param_dict["rt_tol"],
        #                                         controller_param_dict["min_roi_length_for_fragmentation"],
        #                                         flex_controller_param_dict["peak_df"])
        return controller  # TODO: add more controller options


class GridSearch(BaseOptimiser):
    def __init__(self, flex_controller_param_dict, *args, **kwargs):
        self.flex_controller_param_dict = flex_controller_param_dict
        self.scoring_method = None
        super().__init__(*args, **kwargs)

    def _get_initial_params(self):
        # gets whole parameter list
        allNames = sorted(self.flex_controller_param_dict)
        combinations = list(it.product(*(self.flex_controller_param_dict[Name] for Name in allNames)))
        dictionaries = [dict(zip(allNames, combinations[i])) for i in range(len(combinations))]
        col_names = ['Peak Count', 'Log Peak Intensity', 'Precision', 'Recall', 'F1'] + list(
            dictionaries[0].keys())  # this code is repeated, tryi
        df = pd.DataFrame(columns=col_names)
        return dictionaries, df

    def run_optimiser_simulations(self):
        return self.results


class BOMAS(BaseOptimiser):
    def __init__(self, scoring_method, N_init, N_BO, gp_kernel, gp_noise_sd, theta_range, *args, **kwargs):
        self.scoring_method = scoring_method
        if self.scoring_method == 'Peak Count' or self.scoring_method == 'Log Peak Intensity' or self.scoring_method == 'F1':
            self.N_init = N_init
            self.N_BO = N_BO
            self.gp_kernel = gp_kernel
            self.gp_noise_sd = gp_noise_sd
            self.theta_range = theta_range
            super().__init__(*args, **kwargs)
        else:
            sys.exit('Invalid scoring method. Must be Peak Count, Log Peak Intensity, F1')

    def _get_initial_params(self):
        theta_array = [list(self.theta_range[list(self.theta_range.keys())[i]][0]) for i in
                       range(len(self.theta_range))]
        scaled_theta = lhs(len(theta_array), samples=self.N_init, criterion='center')
        scaled_theta = np.array([(
                                         scaled_theta[:, i] * (theta_array[i][1] - theta_array[i][0]) + theta_array[i][
                                     0]).tolist()
                                 for i in range(len(theta_array))])
        if 'N' in self.theta_range.keys():
            which_N = list(np.where(np.array(list(self.theta_range.keys())) == 'N')[0])[0]
            scaled_theta[which_N] = np.round(scaled_theta[which_N], 0)
        dictionaries = [dict(zip(self.theta_range.keys(), scaled_theta[:, i])) for i in range(len(scaled_theta[0]))]
        col_names = [self.scoring_method] + list(dictionaries[0].keys())
        df = pd.DataFrame(columns=col_names)
        return dictionaries, df

    def _get_next_BO_params(self):
        noise = 0.1  # TODO: set this up properly at the start
        bounds = np.array(
            [list(self.theta_range[list(self.theta_range.keys())[i]][0]) for i in range(len(self.theta_range))])
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)
        y = self.results[self.scoring_method].values
        theta = self.results[self.results.columns[1:]].values
        self.gpr.fit(theta, y)
        theta_next = propose_location(expected_improvement, theta, y, self.gpr,
                                      bounds)  # TODO: add bounds + other parameters to start
        theta_next_dict = dict(zip(self.theta_range.keys(), theta_next.flatten().tolist()))
        if 'N' in list(self.theta_range.keys()):
            theta_next_dict['N'] = round(theta_next_dict['N'])
        return theta_next_dict

    def run_optimiser_simulations(self):
        results = self.results
        for i in range(self.N_BO):
            idx = self.N_init + i
            next_flex_param_dict = self._get_next_BO_params()
            new_results = self._get_controller_score(self.mass_spec, self.ms1_picked_peaks_file, self.controller_method,
                                                     self.controller_param_dict, next_flex_param_dict, idx)
            results.loc[len(self.results)] = new_results
        return results

        # X_init = np.array(self.results[self.results.columns[1:]].values)
        # y_init = -np.array(self.results[self.scoring_method].values)
        # gpr = GaussianProcessRegressor(kernel=self.gp_kernel, alpha=self.gp_noise_sd ** 2)
        # results = gp_minimize(lambda x: -self.optimiser_function(np.array(x))[0],
        #                       self.gp_bounds.tolist(),
        #                       base_estimator=gpr,
        #                       acq_func='EI',  # expected improvement
        #                       xi=0.01,  # exploitation-exploration trade-off
        #                       n_calls=self.N_BO,  # number of iterations
        #                       n_random_starts=0,  # initial samples are provided
        #                       x0=X_init.tolist(),  # initial samples
        #                       y0=y_init.ravel())
        # return results  # TODO: change to return the results in the right format

    # def optimiser_function(self, x):  # TODO: this and function above are the wrong way around
    # needs to take a vector x and return y
    # will need to convert x into dictionary format
    # get idx
    # new_results = self._get_controller_score(self.mass_spec, self.ms1_picked_peaks_file, self.controller_method,
    #                                         self.controller_param_dict, next_flex_param_dict, idx)
    # return right bit of new results


def create_controller(controller_method, search_param_dict, base_param_dict):
    # TODO: make this function create the relevant controller based on the inputs
    # Can be used in the parallel and non parallel version of the controllers
    NotImplementedError()


def Heatmap_GP(BOMAS_object):  # TODO: allow you to specify the columns
    X = BOMAS_object.results[BOMAS_object.results.columns[1:]].values
    names = list(BOMAS_object.results[BOMAS_object.results.columns[1:]].keys())

    # Input space
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max())
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max())

    x1x2 = np.array(list(product(x1, x2)))
    y_pred, MSE = BOMAS_object.gpr.predict(x1x2, return_std=True)

    X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)
    Zp = np.reshape(y_pred, (50, 50))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_aspect('equal')
    cf = ax.contourf(X0p, X1p, Zp)
    fig.colorbar(cf, ax=ax)
    plt.xlabel(names[0])
    plt.ylabel(names[1])

    plt.show()


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model. Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. xi: Exploitation-exploration trade-off parameter. Returns: Expected improvements at points X. '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    # sigma = sigma.reshape(-1, X_sample.shape[1])  # TODO: What is this meant to do?

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    ''' Proposes the next sampling point by optimizing the acquisition function. Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. Returns: Location of the acquisition function maximum. '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


def load_scores(colnames, peak_files, ms2_dir, dataset_file, ms1_picked_peaks_file, score_param_dict):
    dataset = load_obj(dataset_file)  # TODO: this needs updating
    results = pd.DataFrame(columns=colnames)
    for i in range(len(peak_files)):
        ms2_picked_peaks_file = peak_files[i]
        controller = load_obj(ms2_dir + Path(peak_files[i]).stem[:-3] + '.p')
        score = controller_score(controller, dataset, ms1_picked_peaks_file, ms2_picked_peaks_file,
                                 score_param_dict['min_ms1_intensity'],
                                 score_param_dict['matching_mz_tol'],
                                 score_param_dict['matching_rt_tol'])
        new_entry = [score, controller.N, controller.rt_tol]
        results.loc[len(results)] = new_entry
        logger.debug(i)
    return results


def GetScaledValues(n_samples, theta_ranges):
    values = lhs(len(theta_ranges), samples=n_samples, criterion='center')
    scaled_values = np.array([(values[:, i] * (theta_ranges[i][1] - theta_ranges[i][0]) + theta_ranges[i][0]).tolist()
                              for i in range(len(theta_ranges))])
    return scaled_values
