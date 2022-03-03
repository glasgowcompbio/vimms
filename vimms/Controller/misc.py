"""
This file provides the implementation of various miscellaneous controllers, some are
pretty experimental.
"""
import copy
import itertools
import math
import os
import subprocess

import numpy as np
from loguru import logger

from vimms.Common import INITIAL_SCAN_ID, get_default_scan_params, \
    get_dda_scan_param, DEFAULT_ISOLATION_WIDTH
from vimms.Controller.base import Controller


class FixedScansController(Controller):
    """
    A controller which takes a schedule of scans, converts them into
    tasks in queue
    """

    def __init__(self, schedule=None, advanced_params=None):
        """
        Creates a FixedScansController that accepts a list of schedule of
        scan parameters
        :param schedule: a list of ScanParameter objects
        :param advanced_params: mass spec advanced parameters, if any
        """
        super().__init__(advanced_params=advanced_params)
        self.tasks = None
        self.initial_task = None
        if schedule is not None and len(schedule) > 0:
            # if schedule is provided, set it
            self.set_tasks(schedule)

    def get_initial_tasks(self):
        """
        Returns all the remaining scan parameter objects to be pushed to
        the mass spec queue
        :return: all the remaining tasks
        """
        # the remaining scan parameters in the schedule must have been set
        assert self.tasks is not None
        return self.tasks

    def get_initial_scan_params(self):
        """
        Returns the initial scan parameter object to send when
        acquisition starts
        :return: the initial task
        """
        # the first scan parameters in the schedule must have been set
        assert self.initial_task is not None
        return self.initial_task

    def set_tasks(self, schedule):
        """
        Set the fixed schedule of tasks in this controller
        :param schedule: a list of scan parameter objects
        :return: None
        """
        assert isinstance(schedule, list)
        self.initial_task = schedule[0]  # used for sending the first scan
        self.tasks = schedule[1:]  # used for sending all the other scans

    def handle_scan(self, scan, current_size, pending_size):
        # simply record every scan that we've received, but return no new tasks
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)
        return []

    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        pass


class MS2PlannerController(FixedScansController):
    """
    A controller that interfaces with MS2Planner, as described in:

    Zuo, Zeyuan, et al. "MS2Planner: improved fragmentation spectra coverage in
    untargeted mass spectrometry by iterative optimized data acquisition."
    Bioinformatics 37.Supplement_1 (2021): i231-i236.
    """

    @staticmethod
    def mzmine2ms2planner(inpath, outpath):
        """
        Transform mzmine2 box file to ms2planner default format.

        Args:
            inpath:
            outpath:

        Returns: None

        """

        records = []
        with open(inpath, "r") as f:
            fields = {}
            for i, name in enumerate(f.readline().split(",")):
                if (name not in fields):
                    fields[name] = list()
                fields[name].append(i)

            mz = fields["row m/z"][0]
            rt = fields["row retention time"][0]
            charges = next(idxes for fd, idxes in fields.items() if
                           fd.strip().endswith("Peak charge"))
            intensities = next(idxes for fd, idxes in fields.items() if
                               fd.strip().endswith("Peak height"))

            for ln in f:
                sp = ln.split(",")
                for charge, intensity in zip(charges, intensities):
                    records.append([
                        sp[mz],
                        str(float(sp[rt]) * 60),
                        sp[charge],
                        "1",
                        sp[intensity]
                    ])

        out_headers = ["Mass [m/z]", "retention_time", "charge", "Blank",
                       "Sample"]
        with open(outpath, "w+") as f:
            f.write(",".join(out_headers) + "\n")
            for r in records:
                f.write(",".join(r) + "\n")

    @staticmethod
    def minimise_single(x, target):
        if (target < 0):
            return 0
        c = int(target // x)
        return min(c, c + 1, key=lambda c: abs(target - c * x))

    @staticmethod
    def minimise_distance(target, *args):
        """
        Solve argmin(a1, a2 ... an)(a1x1 + ... + anxn - t) for
        non-negative integer a1...an and non-negative reals x1...xn, t
        using backtracking search. i.e. Schedule tasks of different fixed
        lengths s.t. the last task ends as close to the target time
        as possible.

        Args:
            target:
            *args:

        Returns: the best coefficients

        """
        best_coefficients = (float("inf"), [])
        stack = [MS2PlannerController.minimise_single(args[0], target)] if len(
            args) > 0 else []
        while (stack != []):
            remainder = target - sum(s * a for s, a in zip(stack, args))
            for i in range(len(stack), len(args)):
                c = MS2PlannerController.minimise_single(args[i], remainder)
                stack.append(c)
                remainder -= c * args[i]
            dist = abs(remainder)
            if (not math.isclose(dist, best_coefficients[0]) and dist <
                    best_coefficients[0]):
                best_coefficients = (
                    dist, copy.copy(stack))
            # if(dist < best_coefficients[0]): best_coefficients = (
            #     dist, copy.copy(stack))
            # if(dist < best_coefficients[0]):
            #    if(math.isclose(dist, best_coefficients[0])):
            #        print(f"IS CLOSE, DIST: {dist}, "
            #              f"CHAMP DIST: {best_coefficients[0]}, "
            #              f"STACK: {stack}, CHAMPION: {best_coefficients[1]}")
            #    best_coefficients = (dist, copy.copy(stack))
            stack.pop()
            while (stack != [] and stack[-1] <= 0):
                stack.pop()
            if (stack != []):
                stack[-1] -= 1
        return best_coefficients[1]

    @staticmethod
    def parse_ms2planner(fpath):
        schedules = []
        fields = ["mz_centre", "mz_isolation", "duration", "rt_start",
                  "rt_end", "intensity", "apex_rt", "charge"]
        with open(fpath, "r") as f:
            for path in f:
                schedules.append([])
                for scan in path.strip().split("\t")[1:]:
                    schedules[-1].append(
                        dict(zip(fields, map(float, scan.split(" ")))))
        return schedules

    @staticmethod
    def sched_dict2params(schedule, scan_duration_dict):
        """
        Scan_duration_dict matches the format of MS scan_duration_dict
        with _fixed_ scan lengths.

        Args:
            schedule:
            scan_duration_dict:

        Returns: new schedule

        """
        time, new_sched = 0, []
        srted = sorted(schedule, key=lambda s: s["rt_start"])
        print("Schedule times: {}".format([s["rt_start"] for s in srted]))
        print(f"NUM SCANS IN SCHEDULE FILE: {len(schedule)}")
        # new_sched.append(get_default_scan_params())
        # scan_duration_dict = {1: 0.2, 2: 0.2}
        id_count = INITIAL_SCAN_ID
        for ms2 in srted:
            filler = MS2PlannerController.minimise_distance(
                ms2["rt_start"] - time, scan_duration_dict[1],
                scan_duration_dict[2])
            print(f"filler_scans: {filler}")
            for i in range(filler[0]):
                sp = get_default_scan_params()
                new_sched.append(sp)
                id_count += 1
            for i in range(filler[1]):
                # print(f"sid: {id_count}")
                new_sched.append(get_dda_scan_param(0, 0.0, id_count,
                                                    ms2["mz_isolation"] * 2,
                                                    0.0, 0.0))
                id_count += 1
            new_sched.append(
                get_dda_scan_param(ms2["mz_centre"], 0.0, id_count,
                                   ms2["mz_isolation"] * 2, 0.0, 0.0))
            id_count += 1
            times = [time, scan_duration_dict[1] * filler[0],
                     scan_duration_dict[2] * filler[1]]
            time += sum(c * scan_duration_dict[i + 1] for i, c in
                        enumerate(filler)) + scan_duration_dict[2]
            print(f"Start time: {times[0]}, MS1 duration: {times[1]}, "
                  f"MS2 duration: {times[2]}, End time: {time}")
            print(f"schedule_length: {len(new_sched)}")
        print(f"Durations: {scan_duration_dict}")
        return new_sched

    @staticmethod
    def from_fullscan(ms2planner_dir,
                      fullscan_file,
                      fullscan_mzmine_table,
                      out_file,
                      intensity_threshold,
                      intensity_ratio,
                      num_injections,
                      intensity_accu,
                      restriction,
                      isolation,
                      delay,
                      min_rt,
                      max_rt,
                      scan_duration_dict,
                      params=None,
                      cluster_method="kNN",
                      userpython="python"):

        converted = os.path.join(os.path.dirname(out_file),
                                 "mzmine2ms2planner.txt")
        MS2PlannerController.mzmine2ms2planner(
            fullscan_mzmine_table, converted)
        subprocess.run(
            [
                userpython,
                os.path.join(ms2planner_dir, "path_finder.py"),
                "curve",
                converted,
                out_file,
                str(intensity_threshold),
                str(intensity_ratio),
                str(num_injections),
                "-infile_raw", str(fullscan_file),
                "-intensity_accu", str(intensity_accu),
                "-restriction", str(restriction[0]), str(restriction[1]),
                "-isolation", str(isolation),
                "-delay", str(delay),
                "-min_scan", str(min_rt),
                "-max_scan", str(max_rt),
                "-cluster", str(cluster_method)
            ]
        )
        schedules = [
            MS2PlannerController.sched_dict2params(sch, scan_duration_dict) for
            sch in
            MS2PlannerController.parse_ms2planner(out_file)]
        with open(os.path.join(os.path.dirname(out_file), "scan_params.txt"),
                  "w+") as f:
            for i, schedule in enumerate(schedules):
                f.write(f"SCHEDULE {i}\n\n")
                f.write("".join(
                    f"SCAN {j}: {scan}\n\n" for j, scan in enumerate(schedule))
                )
        return [MS2PlannerController(schedule=schedule, params=params) for
                schedule in schedules]


class MatchingController(FixedScansController):
    """
    A pre-scheduled controller that performs maximum matching to obtain the largest
    coverage
    """
    @staticmethod
    def from_matching(matching, isolation_width, params=None):
        return [MatchingController(schedule=schedule, params=params) for
                schedule in
                matching.make_schedules(isolation_width)]


class MultiIsolationController(Controller):
    """
    A controller used to test multiple isolations in a single MS2 scan.
    """
    def __init__(self, N, isolation_width=DEFAULT_ISOLATION_WIDTH,
                 advanced_params=None):
        """
        Initialise a multi-isolation controller
        Args:
            N: the number of precursor ions to fragment
            isolation_width: isolation width, in Dalton
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
        """
        super().__init__(advanced_params=advanced_params)
        assert N > 1
        self.N = N
        self.isolation_width = isolation_width
        self.mz_tol = 10
        self.rt_tol = 15

    def _make_scan_order(self, N):
        # makes a list of tuples, each saying which precuror idx in the sorted
        # list should be in which MS2 scan
        initial_idx = range(N)
        scan_order = []
        for L in range(1, len(initial_idx) + 1):
            for subset in itertools.combinations(initial_idx, L):
                scan_order.append(subset)
        return scan_order

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        # fragmented_count = 0
        if self.scan_to_process is not None:
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            # rt = self.scan_to_process.rt
            idx = np.argsort(intensities)[::-1]
            precursor_scan_id = self.scan_to_process.scan_id
            scan_order = self._make_scan_order(min(self.N, len(mzs)))

            for subset in scan_order:
                mz = []
                intensity = []
                for s in subset:
                    mz.append(mzs[idx[s]])
                    intensity.append(mzs[idx[s]])
                dda_scan_params = self.get_ms2_scan_params(
                    mz, intensity, precursor_scan_id, self.isolation_width,
                    self.mz_tol, self.rt_tol)

                new_tasks.append(dda_scan_params)
                self.current_task_id += 1

            ms1_scan_params = self.get_ms1_scan_params()
            self.current_task_id += 1
            self.next_processed_scan_id = self.current_task_id
            new_tasks.append(ms1_scan_params)

        return new_tasks

    def update_state_after_scan(self, scan):
        pass
