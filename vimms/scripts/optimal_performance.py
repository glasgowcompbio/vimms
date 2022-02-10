# computes optimal performance
import argparse
import bisect
import os
import sys

import networkx as nx
import numpy as np
from loguru import logger
from mass_spec_utils.data_import.mzmine import load_picked_boxes
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Common import POSITIVE
from vimms.Roi import Roi

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


def get_times(mzfile_object):
    times = {1: [], 2: []}
    for i, scan in enumerate(mzfile_object.scans[1:]):
        scan_level = mzfile_object.scans[i].ms_level
        elapsed_time = scan.rt_in_seconds - \
                       mzfile_object.scans[i].rt_in_seconds
        if elapsed_time > 0:
            times[scan_level].append(elapsed_time)
    return times


def setup_scans(time_dict, N, min_rt, max_rt):
    scan_levels = []
    scan_start_times = []
    time = min_rt
    while True:
        # add an MS1
        scan_levels.append(1)
        scan_start_times.append(time)
        time += time_dict[1]
        if time > max_rt:
            break
        # add N MS2
        for n in range(N):
            scan_levels.append(2)
            scan_start_times.append(time)
            time += time_dict[2]
            if time > max_rt:
                break
    return scan_levels, scan_start_times


# flake8: noqa: C901
def add_rois_to_boxes(boxes, mzfile, verbose=True):
    # reset in case we run more than once
    for box in boxes:
        box.roi = None

    for i, scan in enumerate(mzfile.scans):
        if scan.ms_level == 2:
            continue  # skip MS2 scans
        if verbose and i % 1000 == 0:
            print(i, len(mzfile.scans))

        scan_rt = scan.rt_in_seconds

        # find the boxes that are active at this scan
        ok_boxes = list(
            filter(lambda x: x.rt_range_in_seconds[0] <= scan_rt and
                             x.rt_range_in_seconds[1] >= scan_rt, boxes))
        if len(ok_boxes) == 0:
            continue  # no boxes now, onto the next scan

        mz_list, intensity_list = zip(*scan.peaks)
        for box in ok_boxes:

            min_idx = bisect.bisect_right(mz_list, box.mz_range[0])
            max_idx = bisect.bisect_right(mz_list, box.mz_range[1])
            sub_peaks = []
            for i in range(min_idx, max_idx):
                sub_peaks.append((mz_list[i], intensity_list[i]))
            if len(sub_peaks) > 0:
                sub_peaks.sort(key=lambda x: x[1],
                               reverse=True)  # sort by descending intensity
                peak_mz = sub_peaks[0][0]
                peak_intensity = sub_peaks[0][1]
                peak_rt = scan_rt

                if box.roi is None:
                    box.roi = Roi(peak_mz, peak_rt, peak_intensity)
                else:
                    box.roi.add(peak_mz, peak_rt, peak_intensity)

    n_with_roi = 0
    for box in boxes:
        if box.roi is not None:
            n_with_roi += 1
    print("Of {} boxes, {} have ROIs".format(len(boxes), n_with_roi))


def get_intensity(roi, rt, interpolate=False):
    if rt < roi.rt_list[0] or rt > roi.rt_list[-1]:
        return 0
    else:
        pos = bisect.bisect_right(roi.rt_list, rt)
        before_pos = pos - 1
        after_pos = pos
        if interpolate:
            prop = (rt - roi.rt_list[before_pos]) / (
                    roi.rt_list[after_pos] - roi.rt_list[before_pos])
            return roi.intensity_list[before_pos] + prop * (
                    roi.intensity_list[after_pos] - roi.intensity_list[
                before_pos])
        else:
            return roi.intensity_list[before_pos]


def make_edges_chems(chems, scan_start_times, scan_levels, min_ms1_intensity,
                     chrom_min=0.001):
    # this currently only works for mono-isotope and M+H
    logger.warning('Making graph edges from chemicals only uses the '
                   'monoisotopic M+H adduct')
    chem_id = 0
    edges = []
    # find the minimum delta t and then set the step for determining the end
    # of peaks to be half of this
    # min_scan_delta = min([scan_start_times[i+1] - scan_start_times[i]
    #                       for i in range(len(scan_start_times)-1)])
    # delta_t = min_scan_delta / 2
    for chemical in chems:

        adduct = 'M+H'
        which_isotope = 0

        # skip chems that start after the end of acquisition
        rt_start = chemical.rt
        if rt_start > scan_start_times[-1]:
            continue

        # # find the end rt of chemicals
        # rt_end = rt_start + delta_t
        # while chemical.chromatogram.get_relative_intensity(
        #         rt_end - chemical.rt) is not None and \
        #         chemical.chromatogram.get_relative_intensity(
        #             rt_end - chemical.rt) > chrom_min and \
        #         rt_end <= scan_start_times[-1]:
        #     rt_end += delta_t

        # get the intensity at this RT
        adduct_intensity = {a: i for a, i in chemical.adducts[POSITIVE]}
        max_intensity = chemical.isotopes[which_isotope][1] * adduct_intensity[
            adduct] * chemical.max_intensity

        # standard loop as in the ROI case
        spo = bisect.bisect_right(scan_start_times, rt_start)
        can_fragment = False  # until we see an MS1
        while spo < len(scan_start_times):
            # get relative intensity at this time point
            rel_intensity = chemical.chromatogram.get_relative_intensity(
                scan_start_times[spo] - chemical.rt)
            if rel_intensity is None:
                rel_intensity = 0.0
            intensity = max_intensity * rel_intensity
            if scan_levels[spo] == 1:
                if intensity >= min_ms1_intensity:
                    can_fragment = True
                else:
                    can_fragment = False
            if scan_levels[spo] == 2:
                if can_fragment:
                    edges.append(
                        ("S{}".format(spo), "B{}".format(chem_id), chemical))
            spo += 1
            if rel_intensity < chrom_min:
                break
        chem_id += 1
    return edges


def make_edges(boxes, scan_start_times, scan_levels, min_ms1_intensity):
    edges = []

    for box in boxes:
        peak_id = box.peak_id
        rt_start = box.rt_range_in_seconds[0]
        rt_end = box.rt_range_in_seconds[1]
        spo = bisect.bisect_right(scan_start_times, rt_start)
        can_fragment = False  # until we see an MS1
        while spo < len(scan_start_times) and scan_start_times[spo] < rt_end:
            if scan_levels[spo] == 1:
                if get_intensity(box.roi,
                                 scan_start_times[spo]) >= min_ms1_intensity:
                    can_fragment = True
                else:
                    can_fragment = False
            if scan_levels[spo] == 2:
                if can_fragment:
                    edges.append(
                        ("S{}".format(spo), "B{}".format(peak_id), box))
            spo += 1
    return edges


def reducedUnweightedMaxMatchingFromLists(scanSet, boxSet, edgeList):
    G = nx.Graph()  # create an empty graph
    # add both lists of vertices as nodes to graph
    G.add_nodes_from(scanSet, bipartite=0)
    G.add_nodes_from(boxSet, bipartite=1)
    G.add_edges_from(edgeList)  # add edges to graph from list
    print('There are {} scans and {} boxes'.format(len(scanSet), len(boxSet)))
    print('There are {} edges in total'.format(len(edgeList)))
    top_nodes = {n for n, d in G.nodes(data=True) if
                 d['bipartite'] == 0}  # scans
    # call the matching algorithms
    matching = nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes)
    size = 0
    scanList = list(scanSet)
    matchList = [(s, '-') for s in scanList]

    items_list = list(matching.items())[:len(matching) // 2]

    for scan, box in items_list:
        ind = scanList.index(scan)
        matchList[ind] = (scan, box)  # update entry in the matchList
        size = size + 1

    return matchList, size


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute optimal performance')
    parser.add_argument('--N', dest='N', default=10, type=int)
    parser.add_argument('--ms1_time', dest='ms1_time', default=0.6, type=float)
    parser.add_argument('--ms2_time', dest='ms2_time', default=0.2, type=float)
    parser.add_argument('box_file', type=str)
    parser.add_argument('--mzml_file', dest='mzml_file',
                        default=None, type=str)
    parser.add_argument('--timing_file', dest='timing_file', default=None,
                        type=str)
    parser.add_argument('--min_rt', dest='min_rt', default=0.0, type=float)
    parser.add_argument('--max_rt', dest='max_rt', default=26.0 * 60.0,
                        type=float)
    parser.add_argument('--min_ms1_intensity', dest='min_ms1_intensity',
                        default=5e3, type=float)
    parser.add_argument('--time_factor', dest='time_factor', default=1,
                        type=float)
    args = parser.parse_args()

    print("ARGUMENTS:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print()
    print()
    time_dict = {}
    if args.timing_file is None:
        print("No timing file provided")
        time_dict[1] = args.ms1_time
        time_dict[2] = args.ms2_time
        print("MS1 time = {}, MS2 time = {}".format(
            time_dict[1], time_dict[2]))
    else:
        print("Extracting times from {}".format(args.timing_file))
        times = get_times(MZMLFile(args.timing_file))
        for ms_level in [1, 2]:
            time_dict[ms_level] = np.array(times[ms_level]).mean()
        print("MS1 time = {}, MS2 time = {}".format(
            time_dict[1], time_dict[2]))

    if not args.time_factor == 1.:
        print("Changing times by factor {}".format(args.time_factor))
        for ms_level in time_dict:
            time_dict[ms_level] *= args.time_factor
        print("MS1 time = {}, MS2 time = {}".format(
            time_dict[1], time_dict[2]))

    print()
    print()
    scan_levels, scan_start_times = setup_scans(time_dict, args.N, args.min_rt,
                                                args.max_rt)
    print("Created {} scan times and levels".format(len(scan_levels)))

    if args.mzml_file is None:
        print("No mzml file provided, assuming standard naming convention")
        tokens = args.box_file.split(os.sep)
        final_part = tokens[-1]
        final_part = final_part.split('_box')[0]
        final_part += '.mzML'
        mzml_file_name = os.path.join(os.sep.join(tokens[:-1]), final_part)
        print("Assuming {}".format(mzml_file_name))
        if os.path.isfile(mzml_file_name):
            print("File exists")
        else:
            print("File does not exist, please specify an mzml file")
            sys.exit(0)
    else:
        mzml_file_name = args.mzml_file

    print()
    print()
    print("Loading: {}".format(mzml_file_name))
    mzml_file = MZMLFile(mzml_file_name)

    print()
    print()
    print("Loading: {}".format(args.box_file))
    boxes = load_picked_boxes(args.box_file)
    print("Loaded {} boxes".format(len(boxes)))

    print()
    print()
    print("Adding ROIs to boxes")
    add_rois_to_boxes(boxes, mzml_file)

    print()
    print()
    print("Making edges")
    edges = make_edges(boxes, scan_start_times, scan_levels,
                       args.min_ms1_intensity)
    print("{} edges made".format(len(edges)))

    scan_names, box_names, _ = zip(*edges)
    scanSet = set(scan_names)
    boxSet = set(box_names)
    reduced_edges = list(zip(scan_names, box_names))

    print()
    print()
    print("Doing matching")
    matchList, size = reducedUnweightedMaxMatchingFromLists(scanSet, boxSet,
                                                            reduced_edges)
    print("The matching has size: {}".format(size))

    print("Finished!")
