import csv
import itertools
from collections import Counter, OrderedDict
from functools import reduce

import numpy as np
import statsmodels.api as sm
from mass_spec_utils.data_import.mzmine import load_picked_boxes, \
    map_boxes_to_scans, PickedBox
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Box import (
    Point, Interval, GenericBox,
    BoxLineSweeper
)
from vimms.Common import path_or_mzml


class EvaluationData():
    """
    A container class that wraps the Environment class, storing only relevant data for evaluation.
    This is useful for pickling/unpickling for evaluation as it will be much smaller.
    """
    def __init__(self, env):
        """
        Create an EvaluationData container
        Args:
            env: An instance of [vimms.Environment.Environment] object
        """
        # for compatibility with evaluate_simulated_env
        self.mass_spec = self.DummyMassSpec(env.mass_spec)
        self.controller = self.DummyController(env.controller)

        # for convenience
        self.chemicals = self.mass_spec.chemicals
        self.fragmentation_events = self.mass_spec.fragmentation_events
        self.scans = self.controller.scans

    class DummyMassSpec():
        def __init__(self, mass_spec):
            self.chemicals = mass_spec.chemicals
            self.fragmentation_events = mass_spec.fragmentation_events

    class DummyController():
        def __init__(self, controller):
            self.scans = controller.scans


def evaluate_simulated_env(env, min_intensity=0.0, base_chemicals=None):
    """Evaluates a single simulated injection against the chemicals present in that injection"""
    true_chems = env.mass_spec.chemicals if base_chemicals is None else base_chemicals
    fragmented = {}  # map chem to highest observed intensity
    for event in env.mass_spec.fragmentation_events:
        if (event.ms_level > 1):
            chem = event.chem.get_original_parent()
            fragmented[chem] = max(event.parents_intensity[0],
                                   fragmented.get(chem, 0))
    num_frags = sum(
        1 for event in env.mass_spec.fragmentation_events if event.ms_level > 1)
    coverage = np.array(
        [fragmented.get(chem, -1) >= min_intensity for chem in true_chems])
    raw_intensities = np.array([fragmented.get(chem, 0)
                                for chem in true_chems])
    coverage_intensities = raw_intensities * (raw_intensities >= min_intensity)

    max_coverage = len(true_chems)
    coverage_prop = np.sum(coverage) / max_coverage
    chemicals_fragmented = np.array(true_chems)[coverage.nonzero()]

    if base_chemicals is None:
        max_possible_intensities = np.array(
            [chem.max_intensity for chem in true_chems])
    else:
        true_intensities = {chem.get_original_parent(): chem.max_intensity for
                            chem in env.mass_spec.chemicals}
        max_possible_intensities = np.array(
            [true_intensities.get(chem, 0.0) for chem in true_chems])

    which_non_zero = max_possible_intensities > 0.0
    coverage_intensity_prop = np.mean(
        np.array(coverage_intensities[which_non_zero]) /
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


def evaluate_multiple_simulated_env(env_list, min_intensity=0.0,
                                    group_list=None):
    """Evaluates_multiple simulated injections against a base set of chemicals that
    were used to derive the datasets"""
    all_chems = [chem for env in env_list for chem in env.mass_spec.chemicals]
    observed_chems = set(chem.get_original_parent() for chem in all_chems)
    base_chemicals = list(observed_chems)

    results = [evaluate_simulated_env(env, min_intensity=min_intensity,
                                      base_chemicals=base_chemicals) for env in
               env_list]

    num_frags = [r["num_frags"] for r in results]
    fragmented = [r["fragmented"] for r in results]
    max_possible_intensities = [r["max_possible_intensities"] for r in results]

    coverage = [r["coverage"] for r in results]
    max_coverage = sum(chem in observed_chems for chem in base_chemicals)
    coverage_prop = [np.sum(cov) / max_coverage for cov in coverage]
    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_coverage_prop = [np.sum(cov) / max_coverage for cov in
                                cumulative_coverage]

    raw_intensities = [r["raw_intensity"] for r in results]
    cumulative_raw_intensities = list(
        itertools.accumulate(raw_intensities, np.fmax))

    coverage_intensities = [r["intensity"] for r in results]
    max_coverage_intensity = reduce(np.fmax, max_possible_intensities)
    coverage_intensities_prop = [np.mean(c_i / max_coverage_intensity) for c_i
                                 in coverage_intensities]
    cumulative_coverage_intensities = list(
        itertools.accumulate(coverage_intensities, np.fmax))
    which_non_zero = max_coverage_intensity > 0.0
    cumulative_coverage_intensities_prop = [
        np.mean(c_i[which_non_zero] / max_coverage_intensity[which_non_zero])
        for
        c_i in
        cumulative_coverage_intensities]
    cumulative_raw_intensities_prop = [
        np.mean(c_i[which_non_zero] / max_coverage_intensity[which_non_zero])
        for c_i in
        cumulative_raw_intensities]

    chemicals_fragmented = [r["chemicals_fragmented"] for r in results]
    times_fragmented = np.sum([r["coverage"] for r in results], axis=0)
    times_fragmented_summary = Counter(times_fragmented)

    if group_list is not None:
        datasets = [env.mass_spec.chemicals for env in env_list]
        true_pvalues = calculate_chemical_p_values(datasets, group_list,
                                                   base_chemicals)
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
                new_chem = \
                    np.array(datasets[i])[np.where(np.array(ds) == chem)[0]][0]
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
    scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes,
                                                  half_isolation_window=half_isolation_window)
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
        # heads = next(reader)
        for line in reader:
            peak_id = int(line[0])
            peak_mz = float(line[1])
            rt_min = 60.0 * float(line[2])
            rt_max = 60.0 * float(line[3])
            height = float(line[4])
            new_box = PickedBox(peak_id, peak_mz, rt_max + rt_min / 2, peak_mz,
                                peak_mz, rt_min, rt_max, height=height)
            boxes.append(new_box)
    boxes.sort(key=lambda x: x.rt)
    return boxes


def evaluate_peak_roi_aligner(roi_aligner, source_file,
                              evaluation_mzml_file=None,
                              half_isolation_width=0):
    coverage, coverage_intensities, max_possible_intensities, included_peaksets = [], [], [], []

    for i, peakset in enumerate(roi_aligner.peaksets):
        source_files = {peak.source_file: i for i, peak in
                        enumerate(peakset.peaks)}
        if source_file in source_files:
            which_peak = source_files[source_file]
            max_possible_intensities.append(
                peakset.peaks[which_peak].intensity)
            if evaluation_mzml_file is not None:
                boxes = [box for box, name in zip(roi_aligner.list_of_boxes,
                                                  roi_aligner.sample_names) if
                         name == source_file]
                scans2boxes, boxes2scans = map_boxes_to_scans(
                    evaluation_mzml_file, boxes,
                    half_isolation_window=half_isolation_width)
                precursor_intensities, scores = get_precursor_intensities(
                    boxes2scans, boxes, 'max')
                temp_max_possible_intensities = max_possible_intensities
                max_possible_intensities = [max(*obj) for obj in
                                            zip(precursor_intensities,
                                                temp_max_possible_intensities)]
                # TODO: actually check that this works
            fragint = roi_aligner.peaksets2fragintensities[peakset][which_peak]
            coverage_intensities.append(fragint)
            included_peaksets.append(i)
        else:
            coverage_intensities.append(0.0)  # fragmentation intensity
            max_possible_intensities.append(
                0.0)  # max possible intensity (so highest observed ms1 intensity)
    included_peaksets = np.array(included_peaksets)
    max_possible_intensities = np.array(max_possible_intensities)
    coverage_intensities = np.array(coverage_intensities)
    coverage = coverage_intensities > 1
    coverage_prop = sum(coverage[included_peaksets]) / len(
        coverage[included_peaksets])
    coverage_intensity_prop = np.mean(
        coverage_intensities[included_peaksets] / max_possible_intensities[
            included_peaksets])

    return {
        'coverage': coverage,
        'intensity': coverage_intensities,
        'coverage_proportion': coverage_prop,
        'intensity_proportion': coverage_intensity_prop,
        'max_possible_intensities': max_possible_intensities,
        'included_peaksets': included_peaksets
    }


def evaluate_multi_peak_roi_aligner(frequentist_roi_aligner, source_files,
                                    casecontrol=False):
    results = [evaluate_peak_roi_aligner(frequentist_roi_aligner, file) for file
               in source_files]
    coverage = [r['coverage'] for r in results]
    coverage_intensities = [r['intensity'] for r in results]
    max_possible_intensities_experiment = [r['max_possible_intensities'] for r
                                           in results]
    max_possible_intensities = reduce(np.fmax,
                                      max_possible_intensities_experiment)
    cumulative_coverage_intensities = list(
        itertools.accumulate(coverage_intensities, np.fmax))
    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_coverage_prop = [np.sum(cov) / len(max_possible_intensities) for
                                cov in cumulative_coverage]
    cumulative_coverage_intensities_prop = [
        np.mean(c_i / max_possible_intensities) for c_i in
        cumulative_coverage_intensities]
    coverage_times_fragmented = [sum(i) for i in zip(*coverage)]
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
        'coverage_times_fragmented': coverage_times_fragmented,
        'max_possible_intensities': max_possible_intensities,
        'pvalues': pvalues
    }


def get_precursor_intensities(boxes2scans, boxes, method):
    assert method in ['max', 'first']
    precursor_intensities = []
    scores = []
    for i, box in enumerate(boxes):
        if box in boxes2scans:
            scans = boxes2scans[box]
            scans = sorted(scans, key=lambda s: s.rt_in_seconds)

            # A box can be linked to multiple ms2 scans.
            # Here we get all the ms2 scans that overlap a box.
            # For each ms2 scan, we then find its precursor intensity using the last ms1 scan
            box_intensities = []
            for ms2_scan in scans:
                precursor = ms2_scan.previous_ms1.get_precursor(
                    ms2_scan.precursor_mz)
                if precursor is not None:
                    box_intensities.append(
                        precursor[1])  # precursor is (mz, intensity)

            if method == 'max':
                # for each box, get the max precursor intensity
                if len(box_intensities) > 0:
                    intensity = max(box_intensities)
                    score = intensity / box.height
                    precursor_intensities.append(intensity)
                    scores.append(score)
                else:
                    precursor_intensities.append(0.0)
                    scores.append(0.0)

            elif method == 'first':
                # for each box, get the first precursor intensity (smallest RT)
                intensity = box_intensities[0]
                score = intensity / box.height
                precursor_intensities.append(intensity)
                scores.append(score)
        else:
            precursor_intensities.append(0.0)
            scores.append(0.0)

    precursor_intensities = np.array(precursor_intensities)
    scores = np.array(scores)
    return precursor_intensities, scores


def _new_window(rt, mz, isolation_width):
    width = isolation_width / 2
    return Interval(
        rt,
        rt,
        mz - width,
        mz + width
    )


def _new_peak_info():
    return {
        "current_intensities": [],
        "max_intensity": 0.0,
        "times_fragmented": 0,
        "fragmentation_intensity": 0.0
    }


def peak_info_map(mzmls, boxes, isolation_width=None):
    geom = BoxLineSweeper()
    box_map = {GenericBox.from_pickedbox(b): b for b in boxes}
    geom.register_boxes(list(box_map.keys()))

    lookups = []
    for mzml in mzmls:
        mzml = path_or_mzml(mzml)
        lookup = OrderedDict([
            (GenericBox.from_pickedbox(b), _new_peak_info()) for b in boxes
        ])
        for s in mzml.scans:
            geom.set_active_boxes(s.rt_in_seconds)
            if (s.ms_level == 1):
                for b in lookup.keys():
                    lookup[b]["current_intensities"] = []

                for mz, intensity in s.peaks:
                    related_boxes = geom.point_in_which_boxes(
                        Point(s.rt_in_seconds, mz)
                    )

                    for b in related_boxes:
                        lookup[b]["current_intensities"].append((mz, intensity))
                        lookup[b]["max_intensity"] = max(intensity, lookup[b]["max_intensity"])
            else:
                mz = s.precursor_mz
                if (isolation_width is None):
                    related_boxes = geom.point_in_which_boxes(
                        Point(s.rt_in_seconds, mz)
                    )

                    for b in related_boxes:
                        candidates = [
                            cint for cmz, cint in lookup[b]["current_intensities"]
                            if cmz >= mz - 1E-10 and cmz <= mz + 1E-10
                        ]
                        lookup[b]["times_fragmented"] += 1
                        lookup[b]["fragmentation_intensity"] = max(
                            max(candidates + [0.0]),
                            lookup[b]["fragmentation_intensity"]
                        )
                else:
                    related_boxes = geom.interval_covers_which_boxes(
                        _new_window(s.rt_in_seconds, mz, isolation_width)
                    )

                    for b in related_boxes:
                        lookup[b]["times_fragmented"] += 1
                        lookup[b]["fragmentation_intensity"] = max(
                            max([it for _, it in lookup[b]["current_intensities"]] + [0.0]),
                            lookup[b]["fragmentation_intensity"]
                        )
        lookups.append(lookup)

    return [
        {box_map[b]: results for b, results in lookup.items()}
        for lookup in lookups
    ]


def _lookups_to_array(lookups, attr):
    return np.array([
        [inner[attr] for _, inner in d.items()]
        for d in lookups
    ])


def evaluation_report(lookups, min_intensity=0.0):
    raw_intensities = _lookups_to_array(lookups, "fragmentation_intensity")
    coverage_intensities = raw_intensities * (raw_intensities >= min_intensity)
    coverage = np.array(coverage_intensities, dtype=np.bool)
    max_possible_intensities = _lookups_to_array(lookups, "max_intensity")

    chemicals_fragmented = [
        [ch for ch, inner in d.items() if inner["times_fragmented"] > 0]
        for d in lookups
    ]
    times_fragmented = np.sum(_lookups_to_array(lookups, "times_fragmented"), axis=0)
    times_fragmented_summary = Counter(times_fragmented)
    times_covered = np.sum(coverage, axis=0)
    times_covered_summary = Counter(times_covered)

    cumulative_coverage = list(itertools.accumulate(coverage, np.logical_or))
    cumulative_raw_intensities = list(itertools.accumulate(raw_intensities, np.fmax))
    cumulative_coverage_intensities = list(itertools.accumulate(coverage_intensities, np.fmax))

    num_chems = max_possible_intensities.shape[1]
    coverage_prop = np.sum(coverage, axis=1) / num_chems
    cumulative_coverage_prop = np.sum(cumulative_coverage, axis=1) / num_chems

    max_coverage_intensities = reduce(np.fmax, max_possible_intensities)
    which_non_zero = max_coverage_intensities > 0.0
    coverage_intensity_prop = [
        np.mean(c_i[which_non_zero] / max_coverage_intensities[which_non_zero])
        for c_i in coverage_intensities
    ]
    cumulative_raw_intensities_prop = [
        np.mean(c_i[which_non_zero] / max_coverage_intensities[which_non_zero])
        for c_i in cumulative_raw_intensities
    ]
    cumulative_coverage_intensities_prop = [
        np.mean(c_i[which_non_zero] / max_coverage_intensities[which_non_zero])
        for c_i in cumulative_coverage_intensities
    ]

    return {
        "coverage": coverage,
        "raw_intensity": raw_intensities,
        "intensity": coverage_intensities,
        "max_possible_intensity": max_possible_intensities,

        "chemicals_fragmented": chemicals_fragmented,
        "times_fragmented": times_fragmented,
        "times_fragmented_summary": times_fragmented_summary,
        "times_covered": times_covered,
        "times_covered_summary": times_covered_summary,

        "cumulative_coverage": cumulative_coverage,
        "cumulative_raw_intensity": cumulative_raw_intensities,
        "cumulative_intensity": cumulative_coverage_intensities,

        "coverage_prop": list(coverage_prop),
        "intensity_prop": coverage_intensity_prop,
        "cumulative_raw_intensity_prop": cumulative_raw_intensities_prop,
        "cumulative_coverage_prop": list(cumulative_coverage_prop),
        "cumulative_coverage_intensity_prop": cumulative_coverage_intensities_prop
    }


def evaluate_real(mzmls, boxes, isolation_width=None, min_intensity=0.0):
    return evaluation_report(peak_info_map(mzmls, boxes, isolation_width=isolation_width),
                             min_intensity=min_intensity)
