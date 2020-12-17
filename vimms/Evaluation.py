from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_import.mzmine import load_picked_boxes, map_boxes_to_scans, PickedBox

import numpy as np
import csv


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
        #base_chemicals_intensity = [coverage_intensity[np.where(chem in chemicals_fragmented)] for chem in base_chemicals]

        for chem in base_chemicals:
            where_chem = np.where(chem in chemicals_fragmented)
            inten = coverage_intensity[np.where(chem in chemicals_fragmented)]
    else:
        base_chemicals_coverage = None
        base_chemicals_intensity = None
    return {'coverage': coverage, 'coverage_intensity': coverage_intensity, 'coverage_prop': coverage_prop,
            'coverage_intensity_prop': coverage_intensity_prop, 'chemicals_fragmented': chemicals_fragmented,
            'base_chemicals_coverage': base_chemicals_coverage, 'base_chemicals_intensity': base_chemicals_intensity}


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

