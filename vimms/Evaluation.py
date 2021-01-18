from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_import.mzmine import load_picked_boxes, map_boxes_to_scans, PickedBox

import numpy as np
import csv
from functools import reduce
import itertools
from collections import Counter


def evaluate_simulated_env(env, min_intensity=0.0, base_chemicals=None):
    '''Evaluates a single simulated injection against the chemicals present in that injection'''
    true_chems = env.mass_spec.chemicals if base_chemicals == None else base_chemicals
    fragmented = {} #map chem to highest observed intensity
    for event in env.mass_spec.fragmentation_events:
        if(event.ms_level > 1):
            chem = event.chem if event.chem.base_chemical is None else event.chem.base_chemical
            fragmented[chem] = max(event.parents_intensity[0], fragmented.get(chem, 0))
    num_frags = sum(1 for event in env.mass_spec.fragmentation_events if event.ms_level > 1)
    coverage = np.array([fragmented.get(chem, -1) >= min_intensity for chem in true_chems])
    raw_intensities = np.array([fragmented.get(chem, 0) for chem in true_chems])
    coverage_intensities = raw_intensities * (raw_intensities >= min_intensity)
    
    max_coverage = sum(1 for chem in true_chems if chem in fragmented)
    coverage_prop = np.sum(coverage) / max_coverage
    max_coverage_intensity = sum(chem.max_intensity for chem in true_chems if chem in fragmented)
    coverage_intensity_prop = np.sum(coverage_intensities) / max_coverage_intensity
    chemicals_fragmented = np.array(true_chems)[coverage.nonzero()]
    
    return {
            'num_frags': num_frags,
            'fragmented': fragmented,
            'coverage': coverage,
            'raw_intensity' : raw_intensities,
            'intensity': coverage_intensities,
            'coverage_proportion': coverage_prop,
            'intensity_proportion': coverage_intensity_prop,
            'chemicals_fragmented': chemicals_fragmented
    }

def evaluate_multiple_simulated_env(env_list, base_chemicals, min_intensity=0.0):
    '''Evaluates_multiple simulated injections against a base set of chemicals that were used to derive the datasets'''
    results = [evaluate_simulated_env(env, min_intensity=min_intensity, base_chemicals=base_chemicals) for env in env_list]
    num_frags = [r["num_frags"] for r in results]
    fragmented = [r["fragmented"] for r in results]
    
    coverage = [r["coverage"] for r in results]
    observed_chems = set(itertools.chain(*(env.mass_spec.chemicals for env in env_list)))
    max_coverage = sum(any(chem in observed for observed in observed_chems) for chem in base_chemicals)
    coverage_prop = [np.sum(cov) / max_coverage for cov in coverage]
    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_coverage_prop = [np.sum(cov) / max_coverage for cov in cumulative_coverage]
    
    raw_intensities = [r["raw_intensity"] for r in results]
    cumulative_raw_intensities = list(itertools.accumulate(coverage, np.fmax))
    
    coverage_intensities = [r["intensity"] for r in results]
    max_coverage_intensity = sum(reduce(np.fmax, coverage_intensities))
    coverage_intensities_prop = [np.sum(c_i) / max_coverage_intensity for c_i in coverage_intensities]
    cumulative_coverage_intensities = list(itertools.accumulate(coverage, np.fmax))
    cumulative_coverage_intensities_prop = [np.sum(c_i) / max_coverage_intensity for c_i in cumulative_coverage_intensities]
    
    chemicals_fragmented = [r["chemicals_fragmented"] for r in results]
    times_fragmented = np.sum([r["coverage"] for r in results], axis=0)
    times_fragmented_summary = Counter(times_fragmented)
    
    return {
            'num_frags': num_frags,
            'fragmented': fragmented,
            'coverage': coverage,
            'raw_intensities': raw_intensities,
            'intensity': coverage_intensities,
            
            'coverage_proportion': coverage_prop,
            'intensity_proportion': coverage_intensities_prop,
            
            'chemicals_fragmented': chemicals_fragmented,
            'times_fragmented': times_fragmented,
            'times_fragmented_summary': times_fragmented_summary,
            
            'cumulative_coverage': cumulative_coverage,
            'cumulative_raw_intensity': cumulative_raw_intensities,
            'cumulative_intensity': cumulative_coverage_intensities,
            'cumulative_coverage_proportion': cumulative_coverage_prop,
            'cumulative_intensity_proportion': cumulative_coverage_intensities_prop
    }

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

