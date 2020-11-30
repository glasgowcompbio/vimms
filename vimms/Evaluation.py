from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_import.mzmine import load_picked_boxes,map_boxes_to_scans

import numpy as np


def evaluate_simulated_env(env, base_chemicals=None):
    chems = env.mass_spec.chemicals
    max_coverage = len(chems)
    max_coverage_intensity = sum([chem.max_intensity for chem in chems])
    coverage = np.array([0 for i in range(max_coverage)])
    coverage_intensity = np.array([0.0 for i in range(max_coverage)])
    for event in env.mass_spec.fragmentation_events:
        if event.ms_level > 1:
            if event.chem in chems:
                chem_idx = np.where(np.array(chems) == event.chem)
                coverage[chem_idx] = 1
                coverage_intensity[chem_idx] = max(coverage_intensity[chem_idx], event.parents_intensity[0])
    coverage_prop = sum(coverage) / max_coverage
    coverage_intensity_prop = sum(coverage_intensity) / max_coverage_intensity
    chemicals_fragmented = np.array(chems)[np.where(coverage == 1)]
    if base_chemicals is not None:
        base_chemicals_coverage = [(chem in chemicals_fragmented)*1 for chem in base_chemicals]
    else:
        base_chemicals_coverage = None
    return {'coverage': coverage, 'coverage_intensity': coverage_intensity, 'coverage_prop': coverage_prop,
            'coverage_intensity_prop': coverage_intensity_prop, 'chemicals_fragmented': chemicals_fragmented,
            'base_chemicals_coverage': base_chemicals_coverage}


def evaluate_mzml(mzml_file, picked_peaks_file, half_isolation_window):
    boxes = load_picked_boxes(picked_peaks_file)
    mz_file = MZMLFile(mzml_file)
    scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes, half_isolation_window=half_isolation_window)
    coverage = len(boxes2scans)
    return coverage

