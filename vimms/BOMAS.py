import os
import time

import numpy as np
from mass_spec_utils.data_import.mzmine import load_picked_boxes, \
    map_boxes_to_scans
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Agent import TopNDEWAgent
from vimms.Box import IdentityDrift, LocatorGrid, AllOverlapGrid
from vimms.Common import load_obj, POSITIVE, ROI_TYPE_NORMAL, ROI_EXCLUSION_DEW
from vimms.Controller import TopN_SmartRoiController, WeightedDEWController, \
    TopN_RoiController, \
    NonOverlapController, IntensityNonOverlapController, TopNBoxRoiController, \
    FlexibleNonOverlapController, \
    FixedScansController, AgentBasedController, TopNController
from vimms.DsDA import get_schedule, dsda_get_scan_params, create_dsda_schedule
from vimms.Environment import Environment
from vimms.Evaluation import evaluate_multi_peak_roi_aligner
from vimms.Evaluation import evaluate_multiple_simulated_env
from vimms.GridEstimator import CaseControlGridEstimator, GridEstimator
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import FrequentistRoiAligner, RoiAligner


def run_coverage_evaluation(box_file, mzml_file, half_isolation_window):
    boxes = load_picked_boxes(box_file)
    mz_file = MZMLFile(mzml_file)
    scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes,
                                                  half_isolation_window=half_isolation_window)
    coverage = len(boxes2scans) / len(boxes)
    return coverage


def run_env(mass_spec, controller, min_rt, max_rt, mzml_file):
    env = Environment(mass_spec, controller, min_rt, max_rt)
    env.run()
    env.write_mzML(None, mzml_file)
    chems = [event.chem.__repr__() for event in
             env.mass_spec.fragmentation_events if event.ms_level > 1]
    chemical_coverage = len(np.unique(np.array(chems))) / len(
        env.mass_spec.chemicals)
    return chemical_coverage


###################################################################################################
# Evaluation methods
###################################################################################################


def top_n_evaluation(param_dict):
    mass_spec = load_obj(param_dict['mass_spec_file'])
    params = load_obj(param_dict['params_file'])
    topn = TopNController(param_dict['ionisation_mode'], param_dict['N'],
                          param_dict['isolation_width'],
                          param_dict['mz_tol'], param_dict['rt_tol'],
                          param_dict['min_ms1_intensity'], advanced_params=params)
    chemical_coverage = run_env(mass_spec, topn, param_dict['min_rt'],
                                param_dict['max_rt'],
                                param_dict['save_file_name'])
    coverage = run_coverage_evaluation(param_dict['box_file'],
                                       param_dict['save_file_name'],
                                       param_dict['half_isolation_window'])
    print('coverage', coverage)
    print('chemical_coverage', chemical_coverage)
    if param_dict['coverage_type'] == 'coverage':
        return coverage
    else:
        return chemical_coverage


# def smart_roi_evaluation(param_dict):
#     mass_spec = load_obj(param_dict['mass_spec_file'])
#     params = load_obj(param_dict['params_file'])
#     smartroi = TopN_SmartRoiController(param_dict['ionisation_mode'],
#                                        param_dict['isolation_window'],
#                                        param_dict['mz_tol'],
#                                        param_dict['min_ms1_intensity'],
#                                        param_dict['min_roi_intensity'],
#                                        param_dict['min_roi_length'],
#                                        param_dict['N'], param_dict['rt_tol'],
#                                        param_dict[
#                                            'min_roi_length_for_fragmentation'],
#                                        param_dict['reset_length_seconds'],
#                                        param_dict['iif'],
#                                        drop_perc=param_dict['dp'] / 100,
#                                        ms1_shift=0, params=params)
#     chemical_coverage = run_env(mass_spec, smartroi, param_dict['min_rt'],
#                                 param_dict['max_rt'],
#                                 param_dict['save_file_name'])
#     coverage = run_coverage_evaluation(param_dict['box_file'],
#                                        param_dict['save_file_name'],
#                                        param_dict['half_isolation_window'])
#     print('coverage', coverage)
#     print('chemical_coverage', chemical_coverage)
#     if param_dict['coverage_type'] == 'coverage':
#         return coverage
#     else:
#         return chemical_coverage


def smart_roi_evaluation(param_dict):
    mass_spec = load_obj(param_dict['mass_spec_file'])
    params = load_obj(param_dict['params_file'])
    smart_roi = TopN_SmartRoiController(param_dict['ionisation_mode'],
                                        param_dict['isolation_width'],
                                        param_dict['mz_tol'],
                                        param_dict['min_ms1_intensity'],
                                        param_dict['min_roi_intensity'],
                                        param_dict['min_roi_length'],
                                        N=param_dict['N'],
                                        rt_tol=param_dict['rt_tol'],
                                        min_roi_length_for_fragmentation=param_dict[
                                            'min_roi_length_for_fragmentation'],
                                        reset_length_seconds=param_dict[
                                            'reset_length_seconds'],
                                        intensity_increase_factor=param_dict[
                                            'intensity_increase_factor'],
                                        drop_perc=param_dict['drop_perc'],
                                        ms1_shift=param_dict['ms1_shift'],
                                        advanced_params=params)
    run_env(mass_spec, smart_roi, param_dict['min_rt'], param_dict['max_rt'],
            param_dict['save_file_name'])
    coverage = run_coverage_evaluation(param_dict['box_file'],
                                       param_dict['save_file_name'],
                                       param_dict['half_isolation_window'])
    return coverage


def weighted_dew_evaluation(param_dict):
    mass_spec = load_obj(param_dict['mass_spec_file'])
    params = load_obj(param_dict['params_file'])
    weighted_dew = WeightedDEWController(param_dict['ionisation_mode'],
                                         param_dict['N'],
                                         param_dict['isolation_width'],
                                         param_dict['mz_tol'],
                                         param_dict['rt_tol'],
                                         param_dict['min_ms1_intensity'],
                                         exclusion_t_0=param_dict[
                                             'exclusion_t_0'],
                                         log_intensity=param_dict[
                                             'log_intensity'], advanced_params=params)
    run_env(mass_spec, weighted_dew, param_dict['min_rt'], param_dict['max_rt'],
            param_dict['save_file_name'])
    coverage = run_coverage_evaluation(param_dict['box_file'],
                                       param_dict['save_file_name'],
                                       param_dict['half_isolation_window'])
    return coverage


###################################################################################################
# Experiment evaluation methods
###################################################################################################


def top_n_experiment_evaluation(datasets, min_rt, max_rt, N, isolation_window,
                                mz_tol, rt_tol, min_ms1_intensity,
                                base_chemicals=None, mzmine_files=None,
                                rt_tolerance=100, experiment_dir=None,
                                progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = TopNController(POSITIVE, N, isolation_window, mz_tol,
                                        rt_tol, min_ms1_intensity, ms1_shift=0,
                                        initial_exclusion_list=None,
                                        force_N=False)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def top_n_exclusion_experiment_evaluation(datasets, min_rt, max_rt, N,
                                          isolation_window, mz_tol, rt_tol,
                                          min_ms1_intensity,
                                          base_chemicals=None,
                                          mzmine_files=None, rt_tolerance=100,
                                          experiment_dir=None,
                                          progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        agent = TopNDEWAgent(POSITIVE, N, isolation_window, mz_tol, rt_tol,
                             min_ms1_intensity, remove_exclusion=False)
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = AgentBasedController(agent)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def top_n_roi_experiment_evaluation(datasets, min_rt, max_rt, N,
                                    isolation_window, mz_tol, rt_tol,
                                    min_ms1_intensity, min_roi_intensity,
                                    min_roi_length, base_chemicals=None,
                                    mzmine_files=None, rt_tolerance=100,
                                    experiment_dir=None, progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = TopN_RoiController(POSITIVE, isolation_window, mz_tol,
                                            min_ms1_intensity,
                                            min_roi_intensity,
                                            min_roi_length, N=N, rt_tol=rt_tol)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def smart_roi_experiment_evaluation(datasets, min_rt, max_rt, N,
                                    isolation_window, mz_tol, rt_tol,
                                    min_ms1_intensity, min_roi_intensity,
                                    min_roi_length,
                                    min_roi_length_for_fragmentation,
                                    reset_length_seconds,
                                    intensity_increase_factor,
                                    drop_perc, ms1_shift, base_chemicals=None,
                                    mzmine_files=None,
                                    rt_tolerance=100, experiment_dir=None,
                                    progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = TopN_SmartRoiController(
                POSITIVE, isolation_window, mz_tol, min_ms1_intensity, min_roi_intensity,
                min_roi_length, N=N, rt_tol=rt_tol,
                min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                reset_length_seconds=reset_length_seconds,
                intensity_increase_factor=intensity_increase_factor,
                drop_perc=drop_perc,
                ms1_shift=ms1_shift)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def weighted_dew_experiment_evaluation(datasets, min_rt, max_rt, N,
                                       isolation_window, mz_tol, r, t0,
                                       min_ms1_intensity, base_chemicals=None,
                                       mzmine_files=None, rt_tolerance=100,
                                       experiment_dir=None, progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = WeightedDEWController(POSITIVE, N, isolation_window,
                                               mz_tol, r, min_ms1_intensity,
                                               exclusion_t_0=t0,
                                               log_intensity=True)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def box_controller_experiment_evaluation(datasets, group_list, min_rt, max_rt,
                                         N, isolation_window,
                                         mz_tol, rt_tol, min_ms1_intensity,
                                         min_roi_intensity, min_roi_length,
                                         boxes_params, base_chemicals=None,
                                         mzmine_files=None, rt_tolerance=100,
                                         experiment_dir=None,
                                         progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        boxes = []
        boxes_intensity = []
        aligner = RoiAligner()
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = TopNBoxRoiController(POSITIVE, isolation_window,
                                              mz_tol, min_ms1_intensity,
                                              min_roi_intensity,
                                              min_roi_length,
                                              boxes_params=boxes_params,
                                              boxes=boxes,
                                              boxes_intensity=boxes_intensity,
                                              N=N, rt_tol=rt_tol)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            rois = env.controller.live_roi + env.controller.dead_roi
            aligner.add_sample(rois, 'sample_' + str(i), group_list[i])
            boxes = aligner.get_boxes()
            boxes_intensity = aligner.get_max_frag_intensities()
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


# change roi_type to ROI_TYPE_SMART to toggle smartroi
# change exclusion_method to ROI_EXCLUSION_WEIGHTED_DEW and specify exclusion_t_0 to
# toggle weighteddew
def non_overlap_experiment_evaluation(datasets, min_rt, max_rt, N,
                                      isolation_window, mz_tol, rt_tol,
                                      min_ms1_intensity,
                                      min_roi_intensity, min_roi_length,
                                      rt_box_size, mz_box_size,
                                      min_roi_length_for_fragmentation,
                                      base_chemicals=None, mzmine_files=None,
                                      rt_tolerance=100, experiment_dir=None,
                                      roi_type=ROI_TYPE_NORMAL,
                                      reset_length_seconds=1e6,
                                      intensity_increase_factor=10,
                                      drop_perc=0.1 / 100,
                                      exclusion_method=ROI_EXCLUSION_DEW,
                                      exclusion_t_0=None, progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        grid = GridEstimator(
            LocatorGrid(min_rt, max_rt, rt_box_size, 0, 3000, mz_box_size),
            IdentityDrift())
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = NonOverlapController(
                POSITIVE, isolation_window, mz_tol, min_ms1_intensity,
                min_roi_intensity,
                min_roi_length, N, grid, rt_tol=rt_tol,
                min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                intensity_increase_factor=intensity_increase_factor,
                drop_perc=drop_perc,
                exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


# change roi_type to ROI_TYPE_SMART to toggle smartroi
# change exclusion_method to ROI_EXCLUSION_WEIGHTED_DEW and specify exclusion_t_0 to
# toggle weighteddew
def intensity_non_overlap_experiment_evaluation(datasets, min_rt, max_rt, N,
                                                isolation_window, mz_tol,
                                                rt_tol, min_ms1_intensity,
                                                min_roi_intensity,
                                                min_roi_length,
                                                rt_box_size, mz_box_size,
                                                min_roi_length_for_fragmentation,
                                                scoring_params={'theta1': 1},
                                                base_chemicals=None,
                                                mzmine_files=None,
                                                rt_tolerance=100,
                                                experiment_dir=None,
                                                roi_type=ROI_TYPE_NORMAL,
                                                reset_length_seconds=1e6,
                                                intensity_increase_factor=10,
                                                drop_perc=0.1 / 100,
                                                exclusion_method=ROI_EXCLUSION_DEW,
                                                exclusion_t_0=None,
                                                progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        grid = GridEstimator(
            AllOverlapGrid(min_rt, max_rt, rt_box_size, 0, 3000, mz_box_size),
            IdentityDrift())
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = IntensityNonOverlapController(
                POSITIVE, isolation_window, mz_tol, min_ms1_intensity,
                min_roi_intensity,
                min_roi_length, N, grid, rt_tol=rt_tol,
                min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                scoring_params=scoring_params,
                roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                intensity_increase_factor=intensity_increase_factor,
                drop_perc=drop_perc,
                exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


# change roi_type to ROI_TYPE_SMART to toggle smartroi
# change exclusion_method to ROI_EXCLUSION_WEIGHTED_DEW and specify exclusion_t_0 to
# toggle weighteddew
def flexible_non_overlap_experiment_evaluation(datasets, min_rt, max_rt, N,
                                               isolation_window, mz_tol,
                                               rt_tol, min_ms1_intensity,
                                               min_roi_intensity,
                                               min_roi_length,
                                               rt_box_size, mz_box_size,
                                               min_roi_length_for_fragmentation,
                                               scoring_params=None,
                                               base_chemicals=None,
                                               mzmine_files=None,
                                               rt_tolerance=100,
                                               experiment_dir=None,
                                               roi_type=ROI_TYPE_NORMAL,
                                               reset_length_seconds=1e6,
                                               intensity_increase_factor=10,
                                               drop_perc=0.1 / 100,
                                               exclusion_method=ROI_EXCLUSION_DEW,
                                               exclusion_t_0=None,
                                               progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        grid = GridEstimator(
            AllOverlapGrid(min_rt, max_rt, rt_box_size, 0, 3000, mz_box_size),
            IdentityDrift())
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        if scoring_params['theta3'] != 0:
            register_all_roi = True
        else:
            register_all_roi = False
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = FlexibleNonOverlapController(
                POSITIVE, isolation_window, mz_tol, min_ms1_intensity,
                min_roi_intensity,
                min_roi_length, N, grid, rt_tol=rt_tol,
                register_all_roi=register_all_roi,
                min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                scoring_params=scoring_params,
                roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                intensity_increase_factor=intensity_increase_factor,
                drop_perc=drop_perc,
                exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None


def case_control_non_overlap_experiment_evaluation(datasets, min_rt, max_rt, N,
                                                   isolation_window, mz_tol,
                                                   rt_tol, min_ms1_intensity,
                                                   min_roi_intensity,
                                                   min_roi_length,
                                                   rt_box_size, mz_box_size,
                                                   min_roi_length_for_fragmentation,
                                                   scoring_params=None,
                                                   base_chemicals=None,
                                                   mzmine_files=None,
                                                   rt_tolerance=100,
                                                   experiment_dir=None,
                                                   box_method='mean',
                                                   roi_type=ROI_TYPE_NORMAL,
                                                   reset_length_seconds=1e6,
                                                   intensity_increase_factor=10,
                                                   drop_perc=0.1 / 100,
                                                   exclusion_method=ROI_EXCLUSION_DEW,
                                                   exclusion_t_0=None,
                                                   progress_bar=False):
    if base_chemicals is not None or mzmine_files is not None:
        env_list = []
        grid = CaseControlGridEstimator(
            AllOverlapGrid(min_rt, max_rt, rt_box_size, 0, 3000, mz_box_size),
            IdentityDrift(), rt_tolerance=rt_tolerance, box_method=box_method)
        mzml_files = []
        source_files = ['sample_' + str(i) for i in range(len(datasets))]
        for i in range(len(datasets)):
            mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
            controller = FlexibleNonOverlapController(
                POSITIVE, isolation_window, mz_tol, min_ms1_intensity,
                min_roi_intensity,
                min_roi_length, N, grid, rt_tol=rt_tol,
                min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                scoring_params=scoring_params,
                roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                intensity_increase_factor=intensity_increase_factor,
                drop_perc=drop_perc,
                exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
            env = Environment(mass_spec, controller, min_rt, max_rt,
                              progress_bar=progress_bar)
            env.run()
            if progress_bar is False:
                print('Processed dataset ' + str(i))
            env_list.append(env)
            if base_chemicals is None:
                file_link = os.path.join(experiment_dir,
                                         source_files[i] + '.mzml')
                mzml_files.append(file_link)
                env.write_mzML(experiment_dir, source_files[i] + '.mzml')
        if base_chemicals is not None:
            evaluation = evaluate_multiple_simulated_env(env_list,
                                                         base_chemicals=base_chemicals)
        else:
            roi_aligner = FrequentistRoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files, True)
        return env_list, evaluation
    else:
        return None, None


def dsda_experiment_evaluation(datasets, base_dir, min_rt, max_rt, N,
                               isolation_window, mz_tol, rt_tol,
                               min_ms1_intensity, mzmine_files=None,
                               rt_tolerance=100, progress_bar=False):
    data_dir = os.path.join(base_dir, 'Data')
    schedule_dir = os.path.join(base_dir, 'settings')
    mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[
        0])  # necessary to get timings for schedule
    create_dsda_schedule(mass_spec, N, min_rt, max_rt, base_dir)
    print('Please open and run R script now')
    time.sleep(1)
    template_file = os.path.join(base_dir, 'DsDA_Timing_schedule.csv')
    env_list = []
    mzml_files = []
    source_files = ['sample_' + "%03d" % i for i in range(len(datasets))]
    for i in range(len(datasets)):
        mass_spec = IndependentMassSpectrometer(POSITIVE, datasets[i])
        if i == 0:
            controller = TopNController(POSITIVE, N, isolation_window, mz_tol,
                                        rt_tol, min_ms1_intensity,
                                        ms1_shift=0,
                                        initial_exclusion_list=None,
                                        force_N=False)
        else:
            print('Looking for next schedule')
            new_schedule = get_schedule(i, schedule_dir)
            print('Found next schedule')
            time.sleep(1)
            schedule_param_list = dsda_get_scan_params(new_schedule,
                                                       template_file,
                                                       isolation_window, mz_tol,
                                                       rt_tol)
            controller = FixedScansController(schedule=schedule_param_list)
        env = Environment(mass_spec, controller, min_rt, max_rt,
                          progress_bar=progress_bar)
        env.run()
        if progress_bar is False:
            print('Processed dataset ' + str(i))
        env_list.append(env)
        file_link = os.path.join(data_dir, source_files[i] + '.mzml')
        mzml_files.append(file_link)
        print("Processed ", i + 1, " files")
        env.write_mzML(data_dir, source_files[i] + '.mzml')
        print("Waiting for R to process .mzML files")
        if mzmine_files is None:
            evaluation = evaluate_multiple_simulated_env(env_list)
        else:
            roi_aligner = RoiAligner(rt_tolerance=rt_tolerance)
            for i in range(len(mzml_files)):
                roi_aligner.add_picked_peaks(mzml_files[i], mzmine_files[i],
                                             source_files[i], 'mzmine')
            evaluation = evaluate_multi_peak_roi_aligner(roi_aligner,
                                                         source_files)
        return env_list, evaluation
    else:
        return None, None
