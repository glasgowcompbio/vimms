import numpy as np


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
