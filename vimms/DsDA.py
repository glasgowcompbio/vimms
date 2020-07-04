import glob
import os

import numpy as np
import pandas as pd
from loguru import logger

from vimms.Common import load_obj


def get_schedule(n, schedule_dir):
    while True:
        files = sorted(glob.glob(os.path.join(schedule_dir, '*.csv')))
        if len(files) == n:
            last_file = files[-1]
            try:
                schedule = pd.read_csv(last_file)
                if schedule.shape[0] == 11951:
                    logger.debug("Schedule Found")
                    return last_file
            except:
                pass


def fragmentation_performance_chemicals(controller_directory, min_acceptable_intensity, controller_file_spec="*.p"):
    global total_matched_chemicals
    os.chdir(controller_directory)
    file_names = glob.glob(controller_file_spec)
    n_samples = len(file_names)
    controllers = []
    all_chemicals = []
    for controller_index in range(n_samples):
        controller = load_obj(file_names[controller_index])
        controllers.append(controller)
        all_chemicals.extend(controller.environment.mass_spec.chemicals)
    all_rts = [chem.rt for chem in all_chemicals]
    chemicals_found_total = np.unique(all_rts)
    sample_chemical_start_rts = [[] for i in range(n_samples)]
    sample_chemical_start_rts_total = []
    for i in range(n_samples):
        for event in controllers[i].mass_spec.fragmentation_events:
            if event.ms_level == 2:
                if controllers[i].mass_spec._get_intensity(event.chem, event.query_rt, 0,
                                                           0) > min_acceptable_intensity:
                    sample_chemical_start_rts[i].append(event.chem.rt)
        sample_chemical_start_rts[i] = np.unique(np.array(sample_chemical_start_rts[i])).tolist()
        # at this point we have collected the RTs of the all the chemicals that
        # have been fragmented above the min_intensity threshold
        flatten_rts = []
        for l in sample_chemical_start_rts[0:(i + 1)]:
            flatten_rts.extend(l)
        sample_chemical_start_rts_total.append(len(np.unique(np.array(flatten_rts))))
        total_matched_chemicals = sample_chemical_start_rts_total
        logger.debug("Completed Controller", i + 1)
    return chemicals_found_total, total_matched_chemicals


def create_frag_dicts(controller_directory, aligned_chemicals_location, min_acceptable_intensity,
                      controller_file_spec="*.p"):
    os.chdir(controller_directory)
    file_names = glob.glob(controller_file_spec)
    params = []
    for controller_index in range(len(file_names)):
        params.append({
            'controller_directory': controller_directory + file_names[controller_index],
            'min_acceptable_intensity': min_acceptable_intensity,
            'aligned_chemicals_location': aligned_chemicals_location
        })
    return params


def fragmentation_performance_aligned(param_dict):
    controller = load_obj(param_dict["controller_directory"])
    min_acceptable_intensity = param_dict["min_acceptable_intensity"]
    aligned_chemicals = pd.read_csv(param_dict["aligned_chemicals_location"])
    n_chemicals_aligned = len(aligned_chemicals["mzmed"])
    chemicals_found = 0

    events = np.array([event for event in controller.environment.mass_spec.fragmentation_events if event.ms_level == 2])
    event_query_rts = np.array([event.query_rt for event in events])
    event_query_mzs = np.array([controller.environment.mass_spec._get_mz(event.chem, event.query_rt, 0, 0) for event in events])

    chemicals_found = [0 for i in range(n_chemicals_aligned)]

    for aligned_index in range(n_chemicals_aligned):

        rtmin = aligned_chemicals['peak_rtmin'][aligned_index]
        rtmax = aligned_chemicals['peak_rtmax'][aligned_index]
        mzmin = aligned_chemicals['peak_mzmin'][aligned_index]
        mzmax = aligned_chemicals['peak_mzmax'][aligned_index]
        rtmin_check = event_query_rts > rtmin
        rtmax_check = event_query_rts < rtmax
        mzmin_check = event_query_mzs > mzmin
        mzmax_check = event_query_mzs < mzmax
        idx = np.nonzero(rtmin_check & rtmax_check & mzmin_check & mzmax_check)[0]

        for i in idx:
            event = events[i]
            inten = controller.environment.mass_spec._get_intensity(event.chem, event.query_rt, 0, 0)
            if inten > min_acceptable_intensity:
                chemicals_found[aligned_index] = 1
                break
    return chemicals_found


def multi_sample_fragmentation_performance_aligned(params):
    chemicals_found_multi = np.array(list(map(fragmentation_performance_aligned, params)))
    total_chemicals_found = []

    for i in range(len(chemicals_found_multi)):
        total_chemicals_found.append((chemicals_found_multi[0:(1 + i)].sum(axis=0) > 0).sum())

    return total_chemicals_found
