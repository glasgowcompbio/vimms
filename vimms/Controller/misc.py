"""
This file provides the implementation of various miscellaneous controllers, some are
pretty experimental.
"""
import copy
import itertools
import math
import os
import subprocess
from statistics import mean
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from loguru import logger

from vimms.Common import (
    INITIAL_SCAN_ID, get_default_scan_params,
    get_dda_scan_param, DEFAULT_ISOLATION_WIDTH,
    ScanParameters
)
from vimms.PeakPicking import MZMineParams
from vimms.Controller.base import Controller, WrapperController


class TaskFilter():
    '''
        Default object that can be used with FixedScansController to update
        schedule dynamically
    '''
    def __init__(self, ms1_length, ms2_length, skip_margin=0.5, add_margin=1.2):
        self.ms1_length = ms1_length
        self.ms2_length = ms2_length
        self.min_length = min(ms1_length, ms2_length)
        
        self.skip_margin = skip_margin
        self.add_margin = add_margin
        
    @staticmethod
    def make_ms2(task_idx, tasks, precursor_id, scan_id):
        dist = 1
        max_dist = max(task_idx, len(tasks) - task_idx - 1)
        while(dist <= max_dist):
            
            right_idx = task_idx + dist
            if(right_idx < len(tasks) and tasks[right_idx].get(ScanParameters.MS_LEVEL) == 2):
                task_idx = right_idx
                break
            
            left_idx = task_idx - dist
            if(left_idx > -1 and tasks[left_idx].get(ScanParameters.MS_LEVEL) == 2):
                task_idx = left_idx
                break
            
            dist += 1
            
        if(dist <= max_dist):
            template_scan = tasks[task_idx]
            precursor_mz = template_scan.get(ScanParameters.PRECURSOR_MZ)[0].precursor_mz
            isolation_width = template_scan.get(ScanParameters.ISOLATION_WIDTH)[0]
        else:
            precursor_mz = 100.0
            isolation_width = 1.0
    
        return get_dda_scan_param(
            precursor_mz, 
            0.0, 
            precursor_id,
            isolation_width,
            0.0, 
            0.0,
            scan_id=scan_id
        )
        
    def get_task(self, scan, scan_id, precursor_id, task_idx, expected_rts, tasks):
        actual_rt = scan.rt
        expected_rt = expected_rts[task_idx]
        rt_dist = expected_rt - actual_rt

        if(rt_dist > self.add_margin * self.min_length):
            if(self.ms1_length > self.ms2_length):
                if(rt_dist > self.add_margin * self.ms1_length):
                    new_task = get_default_scan_params(scan_id=precursor_id)
                else:
                    new_task = self.make_ms2(task_idx, tasks, precursor_id, scan_id)
            else:
                if(rt_dist > self.add_margin * self.ms1_length):
                    new_task = self.make_ms2(task_idx, tasks, precursor_id, scan_id)
                else:
                    new_task = get_default_scan_params(scan_id=precursor_id)
            return task_idx, new_task
        else:
            if(task_idx >= len(tasks) - 1): return task_idx, tasks[task_idx]
            
            while(actual_rt >= expected_rts[task_idx + 1] - self.skip_margin * self.ms2_length):
                task_idx += 1
                if(task_idx >= len(tasks) - 1): return task_idx, tasks[task_idx]
            
            return task_idx + 1, tasks[task_idx]

class FixedScansController(Controller):
    """
    A controller which takes a schedule of scans, converts them into
    tasks in queue
    """

    def __init__(self, schedule=None, advanced_params=None, expected_rts=None, task_filter=None):
        """
        Creates a FixedScansController that accepts a list of schedule of
        scan parameters
        :param schedule: a list of ScanParameter objects
        :param advanced_params: mass spec advanced parameters, if any
        :param expected_rts: gives times tasks are expected to appear at
                             needed to update tasks dynamically with task_filter
        :param task_filter: object that examines the task list and adds or deletes 
                            tasks to ensure schedule remains in sync with the actual
                            RT
        """
        super().__init__(advanced_params=advanced_params)
        self.tasks = None
        self.initial_task = None
        self.task_idx = 0
        self.expected_rts = expected_rts
        self.task_filter = task_filter
        
        if schedule is not None and len(schedule) > 0:
            # if schedule is provided, set it
            self.set_tasks(schedule)
            self.scan_id = schedule[0].get(ScanParameters.SCAN_ID)
            self.precursor_id = None

    def get_initial_tasks(self):
        """
        Returns all the remaining scan parameter objects to be pushed to
        the mass spec queue
        :return: all the remaining tasks
        """
        # the remaining scan parameters in the schedule must have been set
        assert self.tasks is not None
        if(self.task_filter is None):
            return self.tasks
        else:
            return []

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
        return self._process_scan(scan)

    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        if(self.task_filter is None):
            return []
        else:
            self.task_idx, new_task = self.task_filter.get_task(
                scan, 
                self.scan_id, 
                self.precursor_id, 
                self.task_idx, 
                self.expected_rts, 
                self.tasks
            )
            self.scan_id += 1
            if(new_task.get(ScanParameters.MS_LEVEL) == 1):
                self.precursor_id = self.scan_id
            return [new_task]
        

class DsDAController(WrapperController):
    """
        A controller which allows running the DsDA  (Dataset-Dependent Acquisition) 
        method.
        
        See the original publication for a description of DsDA:
        
        Broeckling, Hoyes, et al. "Comprehensive Tandem-Mass-Spectrometry coverage 
        of complex samples enabled by Data-Set-Dependent acquisition." 
        Analytical Chemistry. 90, 8020–8027 (2018).
    """

    def __init__(self, dsda_state, mzml_name, advanced_params=None, task_filter=None):
        """
            Initialise a new DsDAController instance.
            
            Args:
                dsda_state: An instance of [vimms.DsDA.DsDAState][], wrapping a 
                live R process running DsDA.
                mzml_name: The name of the .mzML file to write for this injection.
                advanced_params: a [vimms.Controller.base.AdvancedParams][] object 
                    that contains advanced parameters to control the mass spec. 
                    See [vimms.Controller.base.AdvancedParams][] for defaults.
                task_filter: Object that examines the task list and adds or deletes 
                    tasks to ensure schedule remains in sync with the actual RT.
        """
        self.dsda_state = dsda_state
        self.mzml_name = mzml_name
        self.task_filter = task_filter
        
        if(dsda_state.file_num == 0):
            self.controller = self.dsda_state.get_base_controller()
        else:
            schedule_params, rts = self.dsda_state.get_scan_params()
            self.controller = FixedScansController(
                schedule=schedule_params,
                advanced_params=advanced_params,
                expected_rts=rts,
                task_filter=task_filter
            )
            
        print(self.controller)
        super().__init__()

    def after_injection_cleanup(self):
        self.dsda_state.register_mzml(self.mzml_name)


class MS2PlannerController(FixedScansController):
    """
    A controller that interfaces with MS2Planner, as described in:

    Zuo, Zeyuan, et al. "MS2Planner: improved fragmentation spectra coverage in
    untargeted mass spectrometry by iterative optimized data acquisition."
    Bioinformatics 37.Supplement_1 (2021): i231-i236.
    """

    @staticmethod
    def boxfile2ms2planner(reader, inpath, outpath):
        """
        Transform peak-picked box file to ms2planner default format.

        Args:
            inpath: Path to input box file.
            outpath: Path to output file used in MS2Planner input.

        Returns: None
        """
        
        out_headers = ["Mass [m/z]", "retention_time", "charge", "Blank", "Sample"]
        
        records = []
        fs_names, line_ls = reader.read_aligned_csv(inpath)
        for i, (row_fields, mzml_fields) in enumerate(line_ls):
            row = []
            if(len(list(mzml_fields.keys())) > 1):
                raise NotImplementedError(
                    "MS2Planner controller doesn't currently handle aligned experiment"
                )
            #not sure if it even makes sense to try and use an aligned file with
            #MS2Planner
            #but handle the file as if it was aligned in case this 
            #more general code will be useful later
            statuses = ((mzml, inner["status"].upper()) for mzml, inner in mzml_fields.items())
            mzmls = [mzml for mzml, s in statuses if s == "DETECTED" or s == "ESTIMATED"]
            if(mzmls != []):
                records.append([
                    row_fields["row m/z"],
                    float(row_fields["row retention time"]) * 60,
                    mzml_fields[mzmls[0]]["charge"],
                    0.0,
                    mean(float(mzml_fields[mzml]["height"]) for mzml in mzmls)
                ])
                
        records.sort(key=lambda r: r[1])
        
        with open(outpath, "w+") as f:
            f.write(",".join(out_headers) + "\n")
            for r in records:
                f.write(",".join(str(field) for field in r) + "\n")
                
    @staticmethod
    def mzmine2ms2planner(inpath, outpath):
        """
        Transform MZMine2 box file to ms2planner default format.

        Args:
            inpath: Path to input MZMine2 file.
            outpath: Path to output file used in MS2Planner input.

        Returns: None

        """
        
        return MS2PlannerController.boxfile2ms2planner(MZMineParams, inpath, outpath)

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
        stack = [MS2PlannerController.minimise_single(args[0], target)] if len(args) > 0 else []
        while (stack != []):
            remainder = target - sum(s * a for s, a in zip(stack, args))
            for i in range(len(stack), len(args)):
                c = MS2PlannerController.minimise_single(args[i], remainder)
                stack.append(c)
                remainder -= c * args[i]
            dist = abs(remainder)
            if (not math.isclose(dist, best_coefficients[0]) and dist < best_coefficients[0]):
                best_coefficients = (dist, copy.copy(stack))
            stack.pop()
            while (stack != [] and stack[-1] <= 0):
                stack.pop()
            if (stack != []):
                stack[-1] -= 1
        return best_coefficients[1]

    @staticmethod
    def parse_ms2planner(fpaths):
        fields = ["mz_centre", "mz_isolation", "duration", "rt_start",
                  "rt_end", "intensity", "apex_rt", "charge"]
                 
        schedules = []
        for fpath in fpaths:
            schedule = []
            with open(fpath, "r") as f:
                f.readline()
                for ln in f:
                    schedule.append(
                        dict(zip(fields, (float(x) for x in ln.split(","))))
                    )
            schedules.append(schedule)
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
        time = scan_duration_dict[1]
        new_sched = [get_default_scan_params(scan_id=INITIAL_SCAN_ID)]
        precursor_id = INITIAL_SCAN_ID
        id_count = INITIAL_SCAN_ID + 1
        
        srted = sorted(schedule, key=lambda s: s["rt_start"])
        print("Schedule times: {}".format([s["rt_start"] for s in srted]))
        print(f"NUM SCANS IN SCHEDULE FILE: {len(schedule)}")
        for ms2 in srted:
        
            if(ms2["rt_start"] - time < scan_duration_dict[1]):
                target = ms2["rt_start"] - time
            else:
                target = ms2["rt_start"] - scan_duration_dict[1] - time
                
            num_ms1, num_ms2 = MS2PlannerController.minimise_distance(
                target, 
                scan_duration_dict[1],
                scan_duration_dict[2]
            )
            
            if(ms2["rt_start"] - time >= scan_duration_dict[1]):
                num_ms1 += 1
            num_ms2 += 1 #add the actual scan
                
            print(f"num_scans: {(num_ms1, num_ms2)}")
            
            filler_diff = num_ms1 - num_ms2
            fillers = [
                1 if filler_diff > 0 else 2
                for i in range(abs(filler_diff))
            ]
            fillers.extend([1, 2] * min(num_ms1, num_ms2))
                
            for ms_level in fillers:
                # print(f"sid: {id_count}")
                if(ms_level == 1):
                    precursor_id = id_count
                    new_sched.append(get_default_scan_params(scan_id=precursor_id))
                else:
                    new_sched.append(
                        get_dda_scan_param(
                            ms2["mz_centre"], 
                            0.0, 
                            precursor_id,
                            ms2["mz_isolation"],
                            0.0, 
                            0.0,
                            scan_id=id_count
                        )
                    )
                id_count += 1
            
            times = [
                time, 
                scan_duration_dict[1] * num_ms1,
                scan_duration_dict[2] * num_ms2
            ]
            time = sum(times)
            
            print(f"Start time: {times[0]}, MS1 duration: {times[1]}, "
                  f"MS2 duration: {times[2]}, End time: {time}")
            print(f"schedule_length: {len(new_sched)}")
        print(f"Durations: {scan_duration_dict}")
        
        return new_sched

    @staticmethod
    def from_fullscan(ms2planner_dir,
                      fullscan_mzmine_table,
                      out_file,
                      intensity_threshold,
                      intensity_ratio,
                      num_injections,
                      intensity_accu,
                      isolation,
                      delay,
                      min_scan_len,
                      max_scan_len,
                      scan_duration_dict,
                      mode="apex",
                      fullscan_file=None,
                      restriction=None,
                      cluster_method="kNN",
                      userpython="python",
                      advanced_params=None):

        converted = os.path.join(os.path.dirname(out_file), "mzmine2ms2planner.txt")
        MS2PlannerController.mzmine2ms2planner(fullscan_mzmine_table, converted)
        
        process_args = [
            userpython,
            os.path.join(ms2planner_dir, "path_finder.py"),
            mode,
            converted,
            out_file,
            str(intensity_threshold),
            str(intensity_ratio),
            str(num_injections),
            "-intensity_accu", str(intensity_accu),
            "-isolation", str(isolation),
            "-delay", str(delay),
            "-min_scan", str(min_scan_len),
            "-max_scan", str(max_scan_len),
        ]
        
        if(mode.lower() == "curve"):
            if(fullscan_file is None or restriction is None):
                raise ValueError(
                    """fullscan_file and restriction arguments must be 
                       supplied for curve mode!"""
                )
            
            process_args.extend([
                "-infile_raw", str(fullscan_file),
                "-restriction", str(restriction[0]), str(restriction[1]),
                "-cluster", str(cluster_method)
            ])
                
        elif(mode.lower() != "apex"):
            raise ValueError("Only curve and apex are supported as modes!")
        
        subprocess.run(process_args)
        out_files = [
            f"{'.'.join(out_file.split('.')[:-1])}_{mode.lower()}_path_{i+1}.csv"
            for i in range(num_injections)
        ]
        schedules = [
            MS2PlannerController.sched_dict2params(sch, scan_duration_dict) 
            for sch in MS2PlannerController.parse_ms2planner(out_files)
        ]
        with open(os.path.join(os.path.dirname(out_file), "scan_params.txt"),
                  "w+") as f:
            for i, schedule in enumerate(schedules):
                f.write(f"SCHEDULE {i}\n\n")
                f.write("".join(
                    f"SCAN {j}: {scan}\n\n" for j, scan in enumerate(schedule))
                )
        return [
            MS2PlannerController(schedule=schedule, advanced_params=advanced_params) 
            for schedule in schedules
        ]


class MatchingController(FixedScansController):
    """
    A pre-scheduled controller that performs maximum matching to obtain the largest
    coverage
    """
    @classmethod
    def from_matching(cls, matching, isolation_width, advanced_params=None, task_filter=None):
        return [
            MatchingController(
                schedule=schedule, 
                advanced_params=advanced_params,
                expected_rts=rts,
                task_filter=task_filter
            ) 
            for schedule, rts in zip(*matching.make_schedules(isolation_width))
        ]


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
