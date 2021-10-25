import csv
import itertools
from collections import Counter
from functools import reduce
import statsmodels.api as sm
import numpy as np

from mass_spec_utils.data_import.mzmine import load_picked_boxes, map_boxes_to_scans, PickedBox
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Roi import get_precursor_intensities

def evaluate_simulated_env(env, min_intensity=0.0, base_chemicals=None):
    '''Evaluates a single simulated injection against the chemicals present in that injection'''
    true_chems = env.mass_spec.chemicals if base_chemicals is None else base_chemicals
    fragmented = {}  # map chem to highest observed intensity
    for event in env.mass_spec.fragmentation_events:
        if (event.ms_level > 1):
            chem = event.chem if event.chem.base_chemical is None else event.chem.base_chemical
            fragmented[chem] = max(event.parents_intensity[0], fragmented.get(chem, 0))
    num_frags = sum(1 for event in env.mass_spec.fragmentation_events if event.ms_level > 1)
    coverage = np.array([fragmented.get(chem, -1) >= min_intensity for chem in true_chems])
    raw_intensities = np.array([fragmented.get(chem, 0) for chem in true_chems])
    coverage_intensities = raw_intensities * (raw_intensities >= min_intensity)

    max_coverage = len(true_chems)
    coverage_prop = np.sum(coverage) / max_coverage
    chemicals_fragmented = np.array(true_chems)[coverage.nonzero()]

    if base_chemicals is None:
        max_possible_intensities = [chem.max_intensity for chem in true_chems]
    else:
        max_possible_intensities = []
        ms_chem_parents = np.array([chem.base_chemical for chem in env.mass_spec.chemicals])
        for chem in base_chemicals:
            if chem in ms_chem_parents:
                max_intensity = np.array(env.mass_spec.chemicals)[np.where(ms_chem_parents == chem)][0].max_intensity
            else:
                max_intensity = 0.0
            max_possible_intensities.append(max_intensity)
    which_non_zero = np.where(np.array(max_possible_intensities) > 0.0)
    coverage_intensity_prop = np.nanmean(np.array(coverage_intensities[which_non_zero]) /
                                         np.array(max_possible_intensities)[which_non_zero])

    return {
        'num_frags': num_frags,
        'fragmented': fragmented,
        'coverage': coverage,
        'raw_intensity': raw_intensities,
        'intensity': coverage_intensities,
        'coverage_proportion': coverage_prop,
        'intensity_proportion': coverage_intensity_prop,
        'chemicals_fragmented': chemicals_fragmented,
        'max_possible_intensities': max_possible_intensities
    }


def evaluate_multiple_simulated_env(env_list, min_intensity=0.0, group_list=None):
    '''Evaluates_multiple simulated injections against a base set of chemicals that were used to derive the datasets'''
    all_chems = np.array(list(itertools.chain(*[env.mass_spec.chemicals for env in env_list])))
    base_chemicals = list(set([chem.base_chemical for chem in all_chems]))
    results = [evaluate_simulated_env(env, min_intensity=min_intensity, base_chemicals=base_chemicals) for env in
               env_list]
    num_frags = [r["num_frags"] for r in results]
    fragmented = [r["fragmented"] for r in results]
    max_possible_intensities = [r["max_possible_intensities"] for r in results]

    coverage = [r["coverage"] for r in results]
    observed_chems = set(chem if chem.base_chemical is None else chem.base_chemical for env in env_list for chem in env.mass_spec.chemicals)
    
    max_coverage = sum(chem in observed_chems for chem in base_chemicals)
    coverage_prop = [np.sum(cov) / max_coverage for cov in coverage]
    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_coverage_prop = [np.sum(cov) / max_coverage for cov in cumulative_coverage]

    raw_intensities = [r["raw_intensity"] for r in results]
    cumulative_raw_intensities = list(itertools.accumulate(raw_intensities, np.fmax))

    coverage_intensities = [r["intensity"] for r in results]
    max_coverage_intensity = reduce(np.fmax, max_possible_intensities)
    coverage_intensities_prop = [np.nanmean(c_i / max_coverage_intensity) for c_i in coverage_intensities]
    cumulative_coverage_intensities = list(itertools.accumulate(coverage_intensities, np.fmax))
    cumulative_coverage_intensities_prop = [np.nanmean(c_i / max_coverage_intensity) for c_i in
                                            cumulative_coverage_intensities]
    cumulative_raw_intensities_prop = [np.nanmean(c_i / max_coverage_intensity) for c_i in cumulative_raw_intensities]

    chemicals_fragmented = [r["chemicals_fragmented"] for r in results]
    times_fragmented = np.sum([r["coverage"] for r in results], axis=0)
    times_fragmented_summary = Counter(times_fragmented)

    if group_list is not None:
        datasets = [env.mass_spec.chemicals for env in env_list]
        true_pvalues = calculate_chemical_p_values(datasets, group_list, base_chemicals)
    else:
        true_pvalues = None

    return {
        'num_frags': num_frags,
        'fragmented': fragmented,
        'coverage': coverage,
        'raw_intensity': raw_intensities,
        'intensity': coverage_intensities,

        'coverage_proportion': coverage_prop,
        'intensity_proportion': coverage_intensities_prop,

        'chemicals_fragmented': chemicals_fragmented,
        'times_fragmented': times_fragmented,
        'times_fragmented_summary': times_fragmented_summary,

        'cumulative_coverage': cumulative_coverage,
        'cumulative_raw_intensity': cumulative_raw_intensities,
        'cumulative_raw_intensity_proportion': cumulative_raw_intensities_prop,
        'cumulative_intensity': cumulative_coverage_intensities,
        'cumulative_coverage_proportion': cumulative_coverage_prop,
        'cumulative_intensity_proportion': cumulative_coverage_intensities_prop,

        'max_possible_intensities': max_possible_intensities,

        'true_pvalues': true_pvalues
    }


def calculate_chemical_p_values(datasets, group_list, base_chemicals):
    # only accepts case control currently
    p_values = []
    # create y here
    categories = np.unique(np.array(group_list))
    if len(categories) < 2:
        pass
    elif len(categories):
        x = np.array([1 for i in group_list])
        if 'control' in categories:
            control_type = 'control'
        else:
            control_type = categories[0]
        x[np.where(np.array(group_list) == control_type)] = 0
        x = sm.add_constant(x)
    else:
        pass
    # create each x and calculate p-value
    ds_parents = [[chem.base_chemical for chem in ds] for ds in datasets]
    for chem in base_chemicals:
        y = []
        for i, ds in enumerate(ds_parents):
            if chem in base_chemicals:
                new_chem = np.array(datasets[i])[np.where(np.array(ds) == chem)[0]][0]
                intensity = np.log(new_chem.max_intensity + 1)
            else:
                intensity = 0.0
            y.append(intensity)
        model = sm.OLS(y, x)
        p_values.append(model.fit(disp=0).pvalues[1])
    return p_values


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
        box.rt_range_in_seconds = [r / 60. for r in box.rt_range_in_seconds]
        box.rt /= 60.
        box.rt_range = [r / 60. for r in box.rt_range]
    return boxes


def load_peakonly_boxes(box_file):
    boxes = []
    with open(box_file, 'r') as f:
        reader = csv.reader(f)
        heads = next(reader)
        for line in reader:
            peak_id = int(line[0])
            peak_mz = float(line[1])
            rt_min = 60.0 * float(line[2])
            rt_max = 60.0 * float(line[3])
            height = float(line[4])
            new_box = PickedBox(peak_id, peak_mz, rt_max + rt_min / 2, peak_mz, peak_mz, rt_min, rt_max, height=height)
            boxes.append(new_box)
    boxes.sort(key=lambda x: x.rt)
    return boxes


def evaluate_peak_roi_aligner(roi_aligner, source_file, evaluation_mzml_file=None, half_isolation_width=0):
    coverage = []
    coverage_intensities = []
    max_possible_intensities = []
    included_peaksets = []
    for i, peakset in enumerate(roi_aligner.peaksets):
        source_files = [peak.source_file for peak in peakset.peaks]
        if source_file in source_files:
            which_peak = np.where(source_file == np.array(source_files))[0][0]
            max_possible_intensities.append(peakset.peaks[which_peak].intensity)
            if evaluation_mzml_file is not None:
                boxes = list(np.array(roi_aligner.list_of_boxes)[np.where(roi_aligner.sample_names == source_file)])
                scans2boxes, boxes2scans = map_boxes_to_scans(evaluation_mzml_file, boxes, half_isolation_window=half_isolation_width)
                precursor_intensities, scores = get_precursor_intensities(boxes2scans, boxes, 'max')
                temp_max_possible_intensities = max_possible_intensities
                max_possible_intensities = [max(*l) for l in zip(precursor_intensities, temp_max_possible_intensities)]
                # TODO: actually check that this works
            fragint = np.array(roi_aligner.peaksets2fragintensities[peakset])[which_peak]
            coverage_intensities.append(fragint)
            if fragint > 1:
                coverage.append(1)
            else:
                coverage.append(0)
            included_peaksets.append(i)
        else:
            coverage.append(0)  # standard coverage
            coverage_intensities.append(0.0)  # fragmentation intensity
            max_possible_intensities.append(0.0)  # max possible intensity (so highest observed ms1 intensity)
    included_peaksets = np.array(included_peaksets)
    coverage = np.array(coverage)
    coverage_intensities = np.array(coverage_intensities)
    max_possible_intensities = np.array(max_possible_intensities)
    coverage_prop = sum(coverage[included_peaksets]) / len(coverage[included_peaksets])
    coverage_intensity_prop = np.nanmean(coverage_intensities[included_peaksets] / max_possible_intensities[included_peaksets])

    return {
        'coverage': coverage,
        'intensity': coverage_intensities,
        'coverage_proportion': coverage_prop,
        'intensity_proportion': coverage_intensity_prop,
        'max_possible_intensities': max_possible_intensities,
        'included_peaksets': included_peaksets
    }


def evaluate_multi_peak_roi_aligner(frequentist_roi_aligner, source_files, casecontrol=False):
    results = [evaluate_peak_roi_aligner(frequentist_roi_aligner, file) for file in source_files]
    coverage = [r['coverage'] for r in results]
    coverage_intensities = [r['intensity'] for r in results]
    max_possible_intensities_experiment = [r['max_possible_intensities'] for r in results]
    max_possible_intensities = reduce(np.fmax, max_possible_intensities_experiment)
    cumulative_coverage_intensities = list(itertools.accumulate(coverage_intensities, np.fmax))
    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_coverage_prop = [np.sum(cov) / len(frequentist_roi_aligner) for cov in cumulative_coverage]
    cumulative_coverage_intensities_prop = [np.nanmean(c_i / max_possible_intensities) for c_i in
                                            cumulative_coverage_intensities]
    if casecontrol:
        pvalues = frequentist_roi_aligner.get_p_values(casecontrol)
    else:
        pvalues = [None for ps in frequentist_roi_aligner.peaksets]

    return {
        'coverage': coverage,
        'intensity': coverage_intensities,
        'cumulative_coverage_intensities': cumulative_coverage_intensities,
        'cumulative_coverage': cumulative_coverage,
        'cumulative_coverage_prop': cumulative_coverage_prop,
        'cumulative_coverage_intensities_prop': cumulative_coverage_intensities_prop,

        'max_possible_intensities': max_possible_intensities,
        'pvalues': pvalues
        #TODO: something is wrong with intensity proportion in the single version of this code - think it goes above 1

    }

