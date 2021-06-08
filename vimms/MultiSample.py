from vimms.Box import *
from vimms.Common import *
from vimms.Controller import TopNController, TopN_RoiController, TopNBoxRoiController, \
    NonOverlapController
from vimms.Environment import Environment
from vimms.Evaluation import evaluate_multiple_simulated_env
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiAligner


# import numpy as np
#
# from vimms.BOMAS import BaseOptimiser, QCB_MZML2CHEMS_DICT, XML_TEMPLATE_MS2, MZMINE_COMMAND
# from vimms.Common import POSITIVE
# from vimms.PythonMzmine import get_base_scoring_df, get_max_intensity
#
# QCB_ROI_CONTROLLER_PARAM_DICT = {"ionisation_mode": POSITIVE,
#                                  "isolation_width": 1,
#                                  "mz_tol": 10,
#                                  "min_ms1_intensity": 1.75E5,
#                                  "min_roi_intensity": 0,
#                                  "min_roi_length": 0,
#                                  "N": 10,
#                                  "rt_tol": 10,
#                                  "min_roi_length_for_fragmentation": 1,
#                                  "rt_range": [(0, 1560)]}
#
#
# QCB_XML_TEMPLATE_MS2 = 'C:\\Users\\Vinny\\work\\vimms\\batch_files\\QCB_mzmine_batch_ms2.xml'
#
# # TODO: set up a multisample environment?
#
#
# class RepeatedExperiment(BaseOptimiser):
#     def __init__(self, n_repeats, base_dir, ps,
#                  # controller_method='Repeated_RoiController',
#                  chem_param_dict=QCB_MZML2CHEMS_DICT,
#                  controller_param_dict=QCB_ROI_CONTROLLER_PARAM_DICT,
#                  # score_param_dict=QCB_SCORE_PARAM_DICT,
#                  min_ms1_intensity=1.75E5,
#                  # ms1_mzml=None,
#                  # ms1_picked_peaks_file=None,
#                  dataset_file=None,
#                  add_noise=True,
#                  # xml_template_ms1=XML_TEMPLATE_MS1,
#                  xml_template_ms2=XML_TEMPLATE_MS2,
#                  mzmine_command=MZMINE_COMMAND):
#         super().__init__(base_dir, ps, 'Repeated_RoiController', chem_param_dict, controller_param_dict, None,
#                          min_ms1_intensity,  None, None, dataset_file, add_noise, None,
#                          xml_template_ms2, mzmine_command)
#         self.n_repeats = n_repeats
#         self.peaks_fragmented = [0]
#         self.prop_peaks_fragmented = [0]
#
#     def run(self):
#         next_flex_param_dict = {'peak_df': None}
#         for i in range(self.n_repeats):
#             env, controller_name = self._run_controller(self.mass_spec, self.controller_method,
#                                                         self.controller_param_dict, next_flex_param_dict, i)
#             if i == 0:
#                 ms2_picked_peaks_file = self.frag_peak_picking(controller_name)
#                 self.peak_df = get_base_scoring_df(ms2_picked_peaks_file)
#                 self.peak_df['frag_status'] = [0 for j in range(len(self.peak_df.iloc[:, 0]))]
#             # calculate peak frag status fo current iteration
#             self.peak_df = self._update_peak_df(env.controller, self.peak_df)
#             next_flex_param_dict = {'peak_df': self.peak_df}
#             # updates scores
#             self.peaks_fragmented.append(sum(self.peak_df['frag_status']))
#             self.prop_peaks_fragmented.append(sum(self.peak_df['frag_status'])/len(self.peak_df['frag_status']))
#
#     def _update_peak_df(self, controller, peak_df):
#         scoring_count = np.array([0 for i in range(len(peak_df.index))])
#         frag_scans = controller.scans[2]
#         for scan in frag_scans:
#             query_mz_window = scan.scan_params.compute_isolation_windows()[0][0]
#             # check whether in an MS2 peak
#             min_rt_check_ms2 =peak_df['rt min'] <= scan.rt
#             max_rt_check_ms2 = scan.rt <= peak_df['rt max']
#             mz_check1 = (peak_df['m/z min'] >= query_mz_window[0]) & (
#                         query_mz_window[1] >= peak_df['m/z min'])
#             mz_check2 = (peak_df['m/z max'] >= query_mz_window[0]) & (
#                         query_mz_window[1] >= peak_df['m/z max'])
#             mz_check3 = (peak_df['m/z min'] <= query_mz_window[0]) & (
#                         peak_df['m/z max'] >= query_mz_window[1])
#             ms2_mz_check = mz_check1 | mz_check2 | mz_check3
#             idx = np.nonzero(np.array(min_rt_check_ms2) & np.array(max_rt_check_ms2) & np.array(ms2_mz_check))[0]
#             # record scores
#             if len(idx) > 0:
#                 scoring_count[idx] = 1
#         peak_df['frag_status'] = list(map(max, zip(peak_df['frag_status'], list(scoring_count))))
#         return peak_df
#
#
# class CaseControlExperiment(BaseOptimiser):
#     def __init__(self, base_dir, ps,
#                  mzml_files, cc_statuses,
#                  chem_param_dict=QCB_MZML2CHEMS_DICT,
#                  controller_param_dict=QCB_ROI_CONTROLLER_PARAM_DICT,
#                  min_ms1_intensity=1.75E5,
#                  dataset_file=None,
#                  add_noise=True,
#                  xml_template_ms2=XML_TEMPLATE_MS2,
#                  mzmine_command=MZMINE_COMMAND):
#         super().__init__(base_dir, ps, 'Repeated_RoiController', chem_param_dict, controller_param_dict, None,
#                          min_ms1_intensity,  None, None, dataset_file, add_noise, None,
#                          xml_template_ms2, mzmine_command)
#         self.mzml_files = mzml_files
#         self.cc_statuses = cc_statuses
#
#     def run(self):
#         next_flex_param_dict = {'peak_df': None}
#         for i in range(self.n_repeats):
#             env, controller_name = self._run_controller(self.mass_spec, self.controller_method,
#                                                         self.controller_param_dict, next_flex_param_dict, i)
#             ms2_picked_peaks_file = self.frag_peak_picking(controller_name)
#             if i == 0:
#                 self.peak_df = get_base_scoring_df(ms2_picked_peaks_file)
#                 self.peak_df['diff_prob'] = [0.5 for j in range(len(self.peak_df.iloc[:, 0]))]
#             else:
#                 self.peak_df['diff_prob'] = self._update_peak_df(env.controller, ms2_picked_peaks_file)
#             next_flex_param_dict = {'peak_df': self.peak_df}
#
#     def _update_peak_df(self, current_peak_df, new_peak_df):
#         if current_peak_df is None
#             peak_df
#
#
#
#         diff_prob = [0.5 for j in range(len(self.peak_df.iloc[:, 0]))]
#         return diff_prob
#
#
#


def top_n_experiment(datasets, base_chemicals, rt_range, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                     ionisation_mode=POSITIVE):
    env_list = []
    for i in range(len(datasets)):
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None)
        controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift=0,
                                    initial_exclusion_list=None, force_N=False)
        env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=True)
        env.run()
        env_list.append(env)
    final_evaluation = evaluate_multiple_simulated_env(env_list, base_chemicals=base_chemicals)
    return env_list, final_evaluation


def top_n_roi_experiment(datasets, base_chemicals, rt_range, isolation_width, mz_tol, min_ms1_intensity,
                         min_roi_intensity,
                         min_roi_length, N, rt_tol, ionisation_mode=POSITIVE):
    env_list = []
    for i in range(len(datasets)):
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None)
        controller = TopN_RoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                                        min_roi_length, N=N, rt_tol=rt_tol)
        env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=True)
        env.run()
        env_list.append(env)
    final_evaluation = evaluate_multiple_simulated_env(env_list, base_chemicals=base_chemicals)
    return env_list, final_evaluation


def non_overlap_experiment(datasets, base_chemicals, rt_range, isolation_width, mz_tol, min_ms1_intensity,
                           min_roi_intensity, min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, rt_box_size,
                           mz_box_size, ionisation_mode=POSITIVE):
    env_list = []
    grid = GridEstimator(LocatorGrid(rt_range[0], rt_range[1], rt_box_size, 0, 3000, mz_box_size), IdentityDrift())
    for i in range(len(datasets)):
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None)
        controller = NonOverlapController(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
            min_roi_length, N, grid, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation)
        env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=True)
        env.run()
        env_list.append(env)
    final_evaluation = evaluate_multiple_simulated_env(env_list, base_chemicals=base_chemicals)
    return env_list, final_evaluation


def top_n_box_experiment(datasets, base_chemicals, rt_range, boxes_params, dataset_group_list, isolation_width, mz_tol,
                         min_ms1_intensity, min_roi_intensity, min_roi_length, N, rt_tol, ionisation_mode=POSITIVE):
    env_list = []
    aligner = RoiAligner()
    boxes = None
    boxes_intensity = None
    for i in range(len(datasets)):
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None)
        controller = TopNBoxRoiController(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
                                          min_roi_intensity,
                                          min_roi_length, boxes_params=boxes_params, boxes=boxes,
                                          boxes_intensity=boxes_intensity, N=N, rt_tol=rt_tol)
        env = Environment(mass_spec, controller, rt_range[0], rt_range[1], progress_bar=True)
        env.run()
        env_list.append(env)
        rois = env.controller.live_roi + env.controller.dead_roi
        aligner.add_sample(rois, 'sample_' + str(i), dataset_group_list[i])
        boxes = aligner.get_boxes()
        boxes_intensity = aligner.get_max_frag_intensities()
    final_evaluation = evaluate_multiple_simulated_env(env_list, base_chemicals=base_chemicals)
    return env_list, final_evaluation
