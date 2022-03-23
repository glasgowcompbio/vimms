import copy
import math

import numpy as np
from tqdm.auto import tqdm

from vimms.Common import ScanParameters


class DiaAnalyser():
    """
    Class for deconvolving basic DIA methods. Only works with basic simulated data
    """

    def __init__(self, controller, min_intensity=0):
        self.controller = controller
        self.scans = controller.scans
        self.dataset = controller.environment.mass_spec.chemicals
        self.chemicals_identified = 0
        self.ms2_matched = 0
        self.entropy = 0
        # TODO: fix this (ie make it so it can be controller in controller and then here
        self.ms1_range = np.array([0, 1000])
        self.min_intensity = min_intensity

        self.ms1_scan_times = np.array([scan.rt for scan in self.scans[1]])
        self.ms2_scan_times = np.array([scan.rt for scan in self.scans[2]])
        self.ms1_mzs = [
            self.controller.environment.mass_spec._get_all_mz_peaks(
                self.dataset[i], self.dataset[i].rt + 0.01, 1,
                [[(0, 1000)]])[
                0][0] for i in range(len(self.dataset))]
        self.ms1_start_rt = np.array([data.rt for data in self.dataset])
        self.ms1_end_rt = np.array(
            [data.rt + data.chromatogram.max_rt for data in self.dataset])
        self.first_scans, self.last_scans = self._get_scan_times()

        self.chemical_locations = []

        with tqdm(total=len(self.dataset)) as pbar:
            for chem_num in range(len(self.dataset)):
                chemical_location = self._get_chemical_location(chem_num)
                chemical_time = [
                    (self.first_scans[chem_num], self.last_scans[chem_num])]
                self.chemical_locations.append(chemical_location)
                num_ms1_options = 0
                for i in range(len(chemical_location)):
                    mz_location = np.logical_and(
                        np.array(self.ms1_mzs) > chemical_location[i][0],
                        np.array(self.ms1_mzs) <= chemical_location[i][1])
                    time_location = np.logical_and(
                        self.ms1_start_rt <= chemical_time[0][1],
                        self.ms1_end_rt >= chemical_time[0][0])
                    num_ms1_options += sum(mz_location * time_location)
                if num_ms1_options == 0:
                    self.entropy += -len(self.dataset[chem_num].children) * len(
                        self.dataset) * math.log(
                        1 / len(self.dataset))
                else:
                    self.entropy += -len(self.dataset[
                                             chem_num].children) * num_ms1_options * math.log(
                        1 / num_ms1_options)
                    if num_ms1_options == 1:
                        self.chemicals_identified += 1
                        self.ms2_matched += len(
                            self.dataset[chem_num].children)
                pbar.update(1)
            pbar.close()

    def _get_scan_times(self):
        max_time = self.scans[1][-1].rt
        if self.scans[2] != []:
            max_time = max(self.scans[1][-1].rt, self.scans[2][-1].rt) + 1
        first_scans = [max_time for i in self.dataset]
        last_scans = [max_time for i in self.dataset]
        for chem_num in range(len(self.dataset)):
            relevant_times = self.ms1_scan_times[
                (self.ms1_start_rt[chem_num] < self.ms1_scan_times) & (
                        self.ms1_scan_times < self.ms1_end_rt[chem_num])]
            for time in relevant_times:
                intensity = \
                    self.controller.environment.mass_spec._get_all_mz_peaks(
                        self.dataset[chem_num], time, 1,
                        [[(0, 1000)]])[0][
                        1]  # TODO: Make MS1 range more general
                if intensity > self.min_intensity:
                    first_scans[chem_num] = min(first_scans[chem_num], time)
                    last_scans[chem_num] = time
        return first_scans, last_scans

    def _get_chemical_location(self, chem_num):
        # find location where ms2s of chemical can be narrowed down to
        which_scans = np.where(np.logical_and(
            np.array(self.ms2_scan_times) > self.first_scans[chem_num],
            np.array(self.ms2_scan_times) < self.last_scans[chem_num]))
        chemical_scans = np.array(self.scans[2])[which_scans]
        if chemical_scans.size == 0:
            possible_locations = [(0, 1000)]  # TODO: Make this more general
        else:
            locations = [scan.scan_params.get(ScanParameters.ISOLATION_WINDOWS)
                         for scan in chemical_scans]
            scan_times = [scan.rt for scan in chemical_scans]
            split_points = np.unique(
                np.array(list(sum(sum(sum(locations, []), []), ()))))
            split_points = np.unique(
                np.concatenate((split_points, self.ms1_range)))
            mid_points = [(split_points[i] + split_points[i + 1]) / 2 for i in
                          range(len(split_points) - 1)]
            possible_mid_points = self._get_mid_points_in_location(chem_num,
                                                                   mid_points,
                                                                   locations,
                                                                   scan_times)
            possible_locations = self._get_possible_locations(
                possible_mid_points, split_points)
        return possible_locations

    def _get_mid_points_in_location(self, chem_num, mid_points, locations,
                                    scan_times):
        # find mid points which satisfying scans locations
        current_mid_points = mid_points
        for i in range(len(locations)):
            chem_scanned = isinstance(
                self.controller.environment.mass_spec._get_all_mz_peaks(
                    self.dataset[chem_num], scan_times[i], 2,
                    locations[i]),
                list)
            new_mid_points = []
            for j in range(len(current_mid_points)):
                if chem_scanned == self._in_window(current_mid_points[j],
                                                   locations[i]):
                    new_mid_points.append(current_mid_points[j])
            current_mid_points = new_mid_points
        return current_mid_points

    def _get_possible_locations(self, possible_mid_points, split_points):
        # find locations where possible mid points can be in, then simplify locations
        possible_locations = []
        for i in range(len(possible_mid_points)):
            min_location = max(
                np.array(split_points)[np.where(
                    np.array(split_points) < possible_mid_points[i])].tolist())
            max_location = min(
                np.array(split_points)[np.where(
                    np.array(split_points) >= possible_mid_points[i])].tolist())
            possible_locations.extend([(min_location, max_location)])
            # TODO: need to simplify still
        return possible_locations

    def _in_window(self, mid_point, locations):
        for window in locations[0]:
            if (mid_point > window[0] and mid_point <= window[1]):
                return True
        return False


class RestrictedDiaAnalyser():
    """
    Class for deconvolving toy DIA examples
    """

    def __init__(self, controller):
        self.entropy = []
        self.chemicals_identified = []
        self.ms2_matched = []
        self.scan_num = []
        temp_controller = copy.deepcopy(controller)
        start = len(temp_controller.scans[2])
        for num_ms2_scans in range(start, -1, -1):
            temp_controller.scans[2] = temp_controller.scans[2][0:num_ms2_scans]
            analyser = DiaAnalyser(temp_controller)
            self.entropy.append(analyser.entropy)
            self.chemicals_identified.append(analyser.chemicals_identified)
            self.ms2_matched.append(analyser.ms2_matched)
            self.scan_num.append(num_ms2_scans + 1)
        self.entropy.reverse()
        self.chemicals_identified.reverse()
        self.ms2_matched.reverse()
        self.scan_num.reverse()


class DiaWindows():
    """
    Class for creating windows for basic, tree and nested DIA methods. Method is used in
    DiaController in Controller/dia. Basic methods are approximately equal to a SWATH method
    with no overlapping windows
    """

    # flake8: noqa: C901
    def __init__(self, ms1_mzs, ms1_range, dia_design, window_type,
                 kaufmann_design, extra_bins, num_windows=None,
                 range_slack=0.01):
        ms1_range_difference = ms1_range[0][1] - ms1_range[0][0]
        # set the number of windows for kaufmann method
        if dia_design == "kaufmann":
            num_windows = 64
        # dont allow extra bins for basic method
        if dia_design == "basic" and extra_bins > 0:
            raise ValueError("Cannot have extra bins with 'basic' dia design.")
        # find bin walls and extra bin walls
        if window_type == "even":
            internal_bin_walls = [ms1_range[0][0]]
            for window_index in range(0, num_windows):
                internal_bin_walls.append(
                    ms1_range[0][0] + ((window_index + 1) / num_windows) * ms1_range_difference)
            internal_bin_walls[0] = internal_bin_walls[0] - range_slack * ms1_range_difference
            internal_bin_walls[-1] = internal_bin_walls[-1] + range_slack * ms1_range_difference
            internal_bin_walls_extra = None
            if extra_bins > 0:
                internal_bin_walls_extra = [ms1_range[0][0]]
                for window_index in range(0, num_windows * (2 ** extra_bins)):
                    internal_bin_walls_extra.append(ms1_range[0][0] + ((window_index + 1) / (
                                num_windows * (2 ** extra_bins))) * ms1_range_difference)
                internal_bin_walls_extra[0] = internal_bin_walls_extra[
                                                  0] - range_slack * ms1_range_difference
                internal_bin_walls_extra[-1] = internal_bin_walls_extra[
                                                   -1] + range_slack * ms1_range_difference
        elif window_type == "percentile":
            internal_bin_walls = np.percentile(ms1_mzs, np.arange(0, 100 + 100 / num_windows,
                                                                  100 / num_windows)).tolist()
            internal_bin_walls[0] = internal_bin_walls[0] - range_slack * ms1_range_difference
            internal_bin_walls[-1] = internal_bin_walls[-1] + range_slack * ms1_range_difference
            internal_bin_walls_extra = None
            if extra_bins > 0:
                internal_bin_walls_extra = np.percentile(ms1_mzs, np.arange(0, 100 + 100 / (
                            num_windows * (2 ** extra_bins)), 100 / (num_windows * (
                            2 ** extra_bins)))).tolist()
                internal_bin_walls_extra[0] = internal_bin_walls_extra[
                                                  0] - range_slack * ms1_range_difference
                internal_bin_walls_extra[-1] = internal_bin_walls_extra[
                                                   -1] + range_slack * ms1_range_difference
        else:
            raise ValueError(
                "Incorrect window_type selected. Must be 'even' or 'percentile'.")
            # convert bin walls and extra bin walls into locations to scan
        if dia_design == "basic":
            self.locations = []
            for window_index in range(0, num_windows):
                self.locations.append(
                    [[(internal_bin_walls[window_index], internal_bin_walls[window_index + 1])]])
        elif dia_design == "kaufmann":
            self.locations = KaufmannWindows(internal_bin_walls,
                                             internal_bin_walls_extra,
                                             kaufmann_design,
                                             extra_bins).locations
        else:
            raise ValueError(
                "Incorrect dia_design selected. Must be 'basic' or 'kaufmann'.")


class KaufmannWindows():
    """
    Class for creating windows for tree and nested DIA methods
    """

    def __init__(self, bin_walls, bin_walls_extra, kaufmann_design,
                 extra_bins=0):
        self.locations = []
        if kaufmann_design == "nested":
            n_locations_internal = 4
            for i in range(0, 8):
                self.locations.append(
                    [[(bin_walls[(0 + i * 8)], bin_walls[(8 + i * 8)])]])
        elif kaufmann_design == "tree":
            n_locations_internal = 3
            self.locations.append([[(bin_walls[0], bin_walls[32])]])
            self.locations.append([[(bin_walls[32], bin_walls[64])]])
            self.locations.append([[(bin_walls[16], bin_walls[48])]])
            self.locations.append([[(bin_walls[8], bin_walls[24]),
                                    (bin_walls[40], bin_walls[56])]])
        else:
            raise ValueError("not a valid design")
        locations_internal = [[[]] for i in
                              range(n_locations_internal + extra_bins)]
        for i in range(0, 4):
            locations_internal[0][0].append(
                (bin_walls[(4 + i * 16)], bin_walls[(12 + i * 16)]))
            locations_internal[1][0].append(
                (bin_walls[(2 + i * 16)], bin_walls[(6 + i * 16)]))
            locations_internal[1][0].append(
                (bin_walls[(10 + i * 16)], bin_walls[(14 + i * 16)]))
            locations_internal[2][0].append(
                (bin_walls[(1 + i * 16)], bin_walls[(3 + i * 16)]))
            locations_internal[2][0].append(
                (bin_walls[(9 + i * 16)], bin_walls[(11 + i * 16)]))
            if kaufmann_design == "nested":
                locations_internal[3][0].append(
                    (bin_walls[(5 + i * 16)], bin_walls[(7 + i * 16)]))
                locations_internal[3][0].append(
                    (bin_walls[(13 + i * 16)], bin_walls[(15 + i * 16)]))
            else:
                locations_internal[2][0].append(
                    (bin_walls[(5 + i * 16)], bin_walls[(7 + i * 16)]))
                locations_internal[2][0].append(
                    (bin_walls[(13 + i * 16)], bin_walls[(15 + i * 16)]))
        if extra_bins > 0:
            for j in range(extra_bins):
                for i in range(64 * (2 ** j)):
                    locations_internal[n_locations_internal + j][0].append(
                        (bin_walls_extra[int(
                            0 + i * ((2 ** extra_bins) / (2 ** j)))],
                         bin_walls_extra[int(
                             ((2 ** extra_bins) / (2 ** j)) / 2 + i * (
                                     (2 ** extra_bins) / (2 ** j)))]))
        self.locations.extend(locations_internal)
