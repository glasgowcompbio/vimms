import copy
import glob
import os
import time
import itertools
import pathlib
import socket
import subprocess

import numpy as np
import pandas as pd
from loguru import logger

from vimms.Common import load_obj, get_default_scan_params, \
    get_dda_scan_param, INITIAL_SCAN_ID


def get_schedule(n, schedule_dir, sleep=True):
    while True:
        files = sorted(glob.glob(os.path.join(schedule_dir, '*.csv')))
        if len(files) == n:
            last_file = files[-1]
            try:
                # schedule = pd.read_csv(last_file)
                if sleep:
                    time.sleep(10)
                return last_file
            except Exception:
                pass


def fragmentation_performance_chemicals(controller_directory,
                                        min_acceptable_intensity,
                                        controller_file_spec="*.p"):
    global total_matched_chemicals
    os.chdir(controller_directory)
    file_names = glob.glob(controller_file_spec)
    n_samples = len(file_names)
    controllers = []
    all_chemicals = []
    for controller_index in range(n_samples):
        controller = load_obj(file_names[controller_index])
        controllers.append(controller)
        all_chemicals.extend(controller.environment.mass_spec.chemicals)
    all_rts = [chem.rt for chem in all_chemicals]
    chemicals_found_total = np.unique(all_rts)
    sample_chemical_start_rts = [[] for i in range(n_samples)]
    sample_chemical_start_rts_total = []
    for i in range(n_samples):
        for event in controllers[i].mass_spec.fragmentation_events:
            if event.ms_level == 2:
                intensity = controllers[i].mass_spec._get_intensity(
                    event.chem, event.query_rt, 0, 0)
                if intensity > min_acceptable_intensity:
                    sample_chemical_start_rts[i].append(event.chem.rt)
        sample_chemical_start_rts[i] = np.unique(
            np.array(sample_chemical_start_rts[i])).tolist()
        # at this point we have collected the RTs of the all the chemicals that
        # have been fragmented above the min_intensity threshold
        flatten_rts = []
        for obj in sample_chemical_start_rts[0:(i + 1)]:
            flatten_rts.extend(obj)
        sample_chemical_start_rts_total.append(
            len(np.unique(np.array(flatten_rts))))
        total_matched_chemicals = sample_chemical_start_rts_total
        logger.debug("Completed Controller", i + 1)
    return chemicals_found_total, total_matched_chemicals


def create_frag_dicts(controller_directory, aligned_chemicals_location,
                      min_acceptable_intensity,
                      controller_file_spec="*.p"):
    os.chdir(controller_directory)
    file_names = glob.glob(controller_file_spec)
    params = []
    for controller_index in range(len(file_names)):
        params.append({
            'controller_directory': controller_directory + file_names[
                controller_index],
            'min_acceptable_intensity': min_acceptable_intensity,
            'aligned_chemicals_location': aligned_chemicals_location
        })
    return params


def fragmentation_performance_aligned(param_dict):
    controller = load_obj(param_dict["controller_directory"])
    min_acceptable_intensity = param_dict["min_acceptable_intensity"]
    aligned_chemicals = pd.read_csv(param_dict["aligned_chemicals_location"])
    n_chemicals_aligned = len(aligned_chemicals["mzmed"])
    chemicals_found = 0

    events = np.array([event for event in
                       controller.environment.mass_spec.fragmentation_events if
                       event.ms_level == 2])
    event_query_rts = np.array([event.query_rt for event in events])

    # FIXME: mass_spec._get_mz() has been removed
    # event_query_mzs = np.array(
    #     [controller.environment.mass_spec._get_mz(
    #         event.chem, event.query_rt, 0, 0) for event in events])

    # Check with Ross that get_ms1_peaks_from_chemical below is what we want
    # (generateing MS1 peaks from a chemical).
    event_query_mzs = np.array(
        [controller.environment.mass_spec.get_chemical_mz_ms1(
            event.chem, event.query_rt, 0, 0) for event in events])
    assert False

    chemicals_found = [0 for i in range(n_chemicals_aligned)]

    for aligned_index in range(n_chemicals_aligned):

        rtmin = aligned_chemicals['peak_rtmin'][aligned_index]
        rtmax = aligned_chemicals['peak_rtmax'][aligned_index]
        mzmin = aligned_chemicals['peak_mzmin'][aligned_index]
        mzmax = aligned_chemicals['peak_mzmax'][aligned_index]
        rtmin_check = event_query_rts > rtmin
        rtmax_check = event_query_rts < rtmax
        mzmin_check = event_query_mzs > mzmin
        mzmax_check = event_query_mzs < mzmax
        idx = np.nonzero(
            rtmin_check & rtmax_check & mzmin_check & mzmax_check)[0]

        for i in idx:
            event = events[i]
            inten = controller.environment.mass_spec._get_intensity(
                event.chem, event.query_rt, 0, 0)
            if inten > min_acceptable_intensity:
                chemicals_found[aligned_index] = 1
                break
    return chemicals_found


def multi_sample_fragmentation_performance_aligned(params):
    chemicals_found_multi = np.array(
        list(map(fragmentation_performance_aligned, params)))
    total_chemicals_found = []

    for i in range(len(chemicals_found_multi)):
        total_chemicals_found.append(
            (chemicals_found_multi[0:(1 + i)].sum(axis=0) > 0).sum())

    return total_chemicals_found


def dsda_get_scan_params(schedule_file, template_file, isolation_width, mz_tol, rt_tol):
    """
        Reads output files from the DsDA R process and turns them into
        a list of scans for a ViMMS controller.
        
        Args:
            schedule_file: Path to DsDA output file to be read.
            template_file: Path to file containing schedule of scan times for the 
                DsDA series.
            isolation_width: Isolation width for scans to use.
            mz_tol: m/z tolerance for scans to use.
            rt_tol: RT tolerance for scans to use.
            
        Returns: A list of [vimms.Common.ScanParameters][].
    """
    scan_list = []
    schedule = pd.read_csv(schedule_file)
    template = pd.read_csv(template_file)
    masses = schedule['targetMass']
    types = template['type']
    scan_id = INITIAL_SCAN_ID
    precursor_scan_id = None
    for i in range(schedule.shape[0]):
        if types[i] != 'msms':
            precursor_scan_id = scan_id
            scan_params = get_default_scan_params(scan_id=scan_id)
        else:
            assert not precursor_scan_id is None
            mz = 100 if np.isnan(masses[i]) else masses[i]
            #TODO: Should mz_tol and rt_tol be here???
            scan_params = get_dda_scan_param(mz, 0.0, precursor_scan_id,
                                             isolation_width, mz_tol, rt_tol,
                                             scan_id=scan_id)
        scan_list.append(scan_params)
        scan_id += 1
    return scan_list


def create_dsda_schedule(mass_spec, N, min_rt, max_rt, base_dir):
    timings = [min_rt]
    total_time = mass_spec.scan_duration_dict[1]
    timing_sequence = [0.0, mass_spec.scan_duration_dict[1]] + [
        mass_spec.scan_duration_dict[2] for i in range(N)]
    timing_sequence = list(np.cumsum(timing_sequence))
    scan_numbers = [N + 2]
    scan_types = ['lm']
    scan_types_sequence = ['ms'] + ['msms' for i in range(N + 1)]
    while total_time < max_rt:
        timings.extend([x + total_time for x in timing_sequence])
        total_time += timing_sequence[-1] + mass_spec.scan_duration_dict[2]
        scan_numbers.extend(list(range(1, N + 2)))
        scan_types.extend(scan_types_sequence)
    d = {'rt': timings, 'f': scan_numbers, 'type': scan_types}
    df = pd.DataFrame(data=d)
    df.to_csv(path_or_buf=os.path.join(base_dir, 'DsDA_Timing_schedule.csv'),
              index=False)


class DsDAState():
    """
        An object which provides a wrapper over a live DsDA process. DsDA is
        run using subprocess from a version of the original R script modified 
        to accept command-line input. 
        
        DsDA requires the names of .mzMLs produced by the method (to process the 
        next item in the series) and outputs a .csv schedule file. A socket 
        connection is therefore established between this Python process and the 
        R subprocess, to pass names of input and output files. 
    """
    def __init__(self,
                 out_dir,
                 dsda_loc, 
                 base_controller,
                 min_rt,
                 max_rt, 
                 scan_duration_dict,
                 rscript_loc="RScript", 
                 port=6011, 
                 dsda_params={}):
        """
            Initialise a DsDAState.
            
            Args:
                out_dir: Directory for DsDA to write .mzMLs to. A subdirectory
                    suffixed with the port name will be created for this
                    DsDAState to use as a working directory.
                dsda_loc: Location of the R script containing DsDA.
                base_controller: Controller class used to run the first injection
                    in order to start the DsDA sequence.
                min_rt: RT to start creating schedule from.
                max_rt: RT which terminates schedule creation once it is passed.
                scan_duration_dict: Indexable object where [n] will return the
                    length of an MSn scan - scan lengths are used for schedule
                    creation.
                rscript_loc: Path to the "Rscript" utility packaged with R. By
                    default assumes it can be found via the "Rscript" environment
                    variable.
                port: Port to use for the socket connection. Must not conflict
                    with other services running on that port (e.g. another 
                    DsDAState).
                dsda_params: Dictionary of keyword parameters to pass to the
                    DsDA R script via subprocess.
        """
                 
        self.out_dir = os.path.abspath(out_dir)
        self.dsda_loc = dsda_loc
        self.base_controller = base_controller
        self.min_rt, self.max_rt = min_rt, max_rt
        self.scan_duration_dict = scan_duration_dict
        self.rscript_loc = rscript_loc
        self.port = port
        self.dsda_params = dsda_params
        
        self.file_num = 0
        self.time_schedule = self.create_dsda_schedule(self.base_controller.N)
        self.mzml_names = []
        self.schedule_names = []
        
    @staticmethod
    def get_scan_times(schedule_file):
        """
            Read all the RT values out of a DsDA schedule file.
        
            Args:
                schedule_file: Path to the DsDA schedule file.
            
            Returns: List of RTs for each scan.
        """
        with open(schedule_file, 'r') as f:
            rt_idx = f.readline().split(",").index("rt")
            return [float(ln.split(",")[rt_idx]) for ln in f]
        
    def create_dsda_schedule(self, N):
        """
            Create a schedule file for the whole DsDA sequence with
            rt, index of scan in the current duty cycle and type of scan
            ("ms" or "msms").
        
            Args:
                N: Number of MS2 scans following each MS1.
        
            Returns: Path to the created schedule file.
        """
        
        pathlib.Path(
            os.path.join(self.out_dir, f"dsda_{self.port}")
        ).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(self.out_dir, f"dsda_{self.port}", "DsDA_Timing_schedule.csv")
        
        with open(fname, "w") as f:
            f.write("rt,f,type\n")
            f.write(f"0.0,{N+2},lm\n")
            
            scan_no, rt = 0, max(self.min_rt, self.scan_duration_dict[1])
            while(rt < self.max_rt):
                pos = scan_no % (N + 1)
                f.write(f"{rt},{pos + 1},{'msms' if pos > 0 else 'ms'}\n")
                scan_no += 1
                rt += self.scan_duration_dict[2 if pos > 0 else 1]
        
        return fname
        
    def __enter__(self):
        """
            On DsDAState creation, ensure:
                * A concurrent R process for DsDAState to wrap is created.
                * A socket connection is established between the DsDAState 
                    instance and the R process.
        """
    
        #R will silently fail if we don't do this - there may be better way to return errors?
        if(not os.path.exists(self.dsda_loc)):
            raise FileNotFoundError("DsDA R script not found")    
    
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("localhost", self.port))
        self.socket.settimeout(300)
        self.socket.listen()
        
        child = subprocess.Popen(
            [
                self.rscript_loc, 
                self.dsda_loc
            ] + [
                arg for arg in itertools.chain(
                    *(
                        (f"--{k}", str(v)) for k, v in self.dsda_params.items()
                        if not v is None
                    )
                )
            ] + [
                f"{self.port}",
                os.path.join(self.out_dir, f"dsda_{self.port}"),
                self.time_schedule
            ],
            stdin=subprocess.PIPE
        )
        (self.child_socket, self.address) = self.socket.accept()
        return self
        
    def __exit__(self, type, value, traceback):
        """
            On DsDAState deletion, ensure:
                * DsDA R process is terminated.
                * Socket is destroyed.  
        """
        try:
            self.child_socket.send(b"q\r\n")
            self.child_socket.shutdown(socket.SHUT_RDWR)
        except:
            raise
        finally:
            self.socket.close()
        
    def register_mzml(self, mzml_name):
        """
            Register the name of the next .mzML in the DsDA sequence
            after it has been written.
            
            Args:
                mzml_name: Name of .mzML file to register.
        """
        self.mzml_names.append(mzml_name)
        self.file_num += 1

    def get_scan_params(self):
        """
            Sends the DsDA process the latest .mzML to process. Then, from
            the file output by DsDA creates a list of scans for a ViMMS
            controller to use.
            
            Returns: A tuple of a list of [vimms.Common.ScanParameters][] and
                a list of RTs for the scans.
        """
        mzml_path = os.path.join(self.out_dir, self.mzml_names[self.file_num - 1])
        
        try:
            self.child_socket.send((mzml_path + "\r\n").encode("utf-8"))
            schedule_path = self.child_socket.recv(4096).decode("utf-8").strip()
        except ConnectionError as err:
            raise type(err)(
                """An unexpected issue was encountered with the connection
                   to the DsDA process. Perhaps it crashed and has written
                   an error log?"""
            )
        
        self.schedule_names.append(
            os.path.basename(schedule_path)
        )
        
        schedule_params = dsda_get_scan_params(
            schedule_path,
            os.path.join(self.out_dir, f"dsda_{self.port}", "DsDA_Timing_schedule.csv"),
            self.base_controller.isolation_width, 
            self.base_controller.mz_tol,
            self.base_controller.rt_tol
        )
        
        rts = self.get_scan_times(self.time_schedule)
        
        return schedule_params, rts
        
    def get_base_controller(self):
        """
            Returns the first controller to use in the DsDA sequence
            (in a way that respects mutable state).
        
            Returns: Deep copy of controller object.
        """
        return copy.deepcopy(self.base_controller)