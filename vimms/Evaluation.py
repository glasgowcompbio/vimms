from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_import.mzmine import load_picked_boxes, map_boxes_to_scans, PickedBox

import numpy as np
import csv


def evaluate_simulated_env(env, min_fragmentation_intensity_for_coverage=0.0,
                                    min_fragmentation_intensity_for_intensity=0.0):
    # Evaluates a single simulated injection against the chemicals present in that injection
    chems = env.mass_spec.chemicals
    max_coverage = len(chems)
    max_coverage_intensity = sum([chem.max_intensity for chem in chems])
    coverage = np.array([0 for i in range(max_coverage)])
    coverage_intensity = np.array([0.0 for i in range(max_coverage)])
    for event in env.mass_spec.fragmentation_events:
        if event.ms_level > 1:
            if event.chem in chems:
                chem_idx = np.where(np.array(chems) == event.chem)
                parent_intensity = event.parents_intensity[0]
                if parent_intensity >= min_fragmentation_intensity_for_coverage:
                    coverage[chem_idx] = 1
                if parent_intensity >= min_fragmentation_intensity_for_intensity:
                    coverage_intensity[chem_idx] = max(coverage_intensity[chem_idx], parent_intensity)
    coverage_prop = sum(coverage) / max_coverage
    coverage_intensity_prop = sum(coverage_intensity) / max_coverage_intensity
    chemicals_fragmented = np.array(chems)[np.where(coverage == 1)]
    results_dict = {'coverage': coverage,
                    'intensity': coverage_intensity,
                    'coverage_proportion': coverage_prop,
                    'intensity_proportion': coverage_intensity_prop,
                    'chemicals_fragmented': chemicals_fragmented}
    return results_dict


def evaluate_multiple_simulated_env(env_list, base_chemicals, min_fragmentation_intensity_for_coverage=0.0,
                                    min_fragmentation_intensity_for_intensity=0.0):
    # Evaluates_multiple simulated injections against a base set of chemicals that were used to derive the datasets
    max_coverage = len(base_chemicals)
    coverage_intensity = [0.0 for i in range(max_coverage)]
    # this needs to be calculated in a different way in case the intensities are different in different injections
    for env in env_list:
        env_chemicals = np.array([chem.base_chemical for chem in env.mass_spec.chemicals])
        env_chemical_intensities = np.array([chem.max_intensity for chem in env.mass_spec.chemicals])
        for i in range(max_coverage):
            if base_chemicals[i] in env_chemicals:
                coverage_intensity[i] = max(coverage_intensity[i],
                                            env_chemical_intensities[np.where(env_chemicals == base_chemicals[i])])
    max_coverage_intensity = sum(coverage_intensity)
    coverage = np.array([[0 for i in range(max_coverage)] for env in env_list])
    coverage_intensity = np.array([[0.0 for i in range(max_coverage)] for env in env_list])
    for i, env in enumerate(env_list):
        for event in env.mass_spec.fragmentation_events:
            if event.ms_level > 1:
                if event.chem.base_chemical in base_chemicals:
                    chem_idx = np.where(np.array(base_chemicals) == event.chem.base_chemical)
                    parent_intensity = event.parents_intensity[0]
                    if parent_intensity >= min_fragmentation_intensity_for_coverage:
                        coverage[i][chem_idx] = 1
                    if parent_intensity >= min_fragmentation_intensity_for_intensity:
                        coverage_intensity[i][chem_idx] = max(coverage_intensity[i][chem_idx], parent_intensity)
    coverage_prop = [sum(cov)/max_coverage for cov in coverage]
    coverage_intensity_prop = [(sum(cov_int)/max_coverage_intensity)[0] for cov_int in coverage_intensity]
    chemicals_fragmented = [np.array(base_chemicals)[np.where(cov == 1)] for cov in coverage]
    # cumulative results
    cumulative_coverage = [list(coverage[0])]
    cumulative_intensity = [list(coverage_intensity[0])]
    cumulative_coverage_prop = [sum(coverage[0])/max_coverage]
    cumulative_intensity_prop = [(sum(coverage_intensity[0])/max_coverage_intensity)[0]]
    for i in range(1,len(env_list)):
        cumulative_coverage.append([max(*l) for l in zip(cumulative_coverage[-1], coverage[i])])
        cumulative_intensity.append([max(*l) for l in zip(cumulative_intensity[-1], coverage_intensity[i])])
        cumulative_coverage_prop.append(sum(cumulative_coverage[-1])/max_coverage)
        cumulative_intensity_prop.append((sum(cumulative_intensity[-1])/max_coverage_intensity)[0])
    # the results
    results_dict = {'coverage': coverage,
                    'intensity': coverage_intensity,
                    'coverage_proportion': coverage_prop,
                    'intensity_proportion': coverage_intensity_prop,
                    'chemicals_fragmented': chemicals_fragmented,
                    'cumulative_coverage': cumulative_coverage,
                    'cumulative_intensity': cumulative_intensity,
                    'cumulative_coverage_proportion': cumulative_coverage_prop,
                    'cumulative_intensity_proportion': cumulative_intensity_prop}

    return results_dict


def evaluate_mzml(mzml_file, picked_peaks_file, half_isolation_window):
    boxes = load_picked_boxes(picked_peaks_file)
    mz_file = MZMLFile(mzml_file)
    scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes, half_isolation_window=half_isolation_window)
    coverage = len(boxes2scans)
    return coverage


def load_xcms_boxes(box_file):
    boxes = load_picked_boxes(box_file)
    for box in boxes:
        box.rt_in_seconds /= 60.
        box.rt_range_in_seconds = [r/60. for r in box.rt_range_in_seconds]
        box.rt /= 60.
        box.rt_range = [r/60. for r in box.rt_range]
    return boxes


def load_peakonly_boxes(box_file):
    boxes = []
    with open(box_file,'r') as f:
        reader = csv.reader(f)
        heads = next(reader)
        for line in reader:
            peak_id = int(line[0])
            peak_mz = float(line[1])
            rt_min = 60.0*float(line[2])
            rt_max = 60.0*float(line[3])
            height = float(line[4])
            new_box = PickedBox(peak_id,peak_mz,rt_max+rt_min/2,peak_mz,peak_mz,rt_min,rt_max,height = height)
            boxes.append(new_box)
    boxes.sort(key=lambda x: x.rt)
    return boxes

