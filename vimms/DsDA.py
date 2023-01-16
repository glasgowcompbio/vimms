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
            scan_params = get_default_scan_params(scan_id=precursor_scan_id)
        else:
            assert precursor_scan_id is not None
            mz = 100 if np.isnan(masses[i]) else masses[i]
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
                 
        self.out_dir = os.path.abspath(out_dir)
        self.dsda_loc = dsda_loc
        self.base_controller = base_controller
        self.rscript_loc = rscript_loc
        self.port = port
        self.dsda_params = dsda_params
        
        self.file_num = 0
        self.time_schedule = self.create_dsda_schedule(
            scan_duration_dict, 
            self.base_controller.N,
            min_rt,
            max_rt
        )
        self.mzml_names = []
        self.schedule_names = []
        
    def create_dsda_schedule(self, scan_duration_dict, N, min_rt, max_rt):
        pathlib.Path(os.path.join(self.out_dir, "dsda")).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(self.out_dir, "dsda", "DsDA_Timing_schedule.csv")
        
        with open(fname, "w") as f:
            f.write("rt,f,type\n")
            f.write(f"0.0,{N+2},lm\n")
            
            scan_no, rt = 0, scan_duration_dict[1]
            while(rt < max_rt):
                pos = scan_no % (N + 1)
                f.write(f"{rt},{pos + 1},{'msms' if pos > 0 else 'ms'}\n")
                scan_no += 1
                rt += scan_duration_dict[2 if pos > 0 else 1]
        
        return fname
        
    def __enter__(self):
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
                arg for arg in itertools.chain(*(
                                    (f"--{k}", str(v)) for k, v in self.dsda_params.items()
                                ))
            ] + [
                f"{self.port}",
                os.path.join(self.out_dir, "dsda"),
                self.time_schedule
            ],
            stdin=subprocess.PIPE
        )
        (self.child_socket, self.address) = self.socket.accept()
        return self
        
    def __exit__(self, type, value, traceback):
        try:
            self.child_socket.send(b"q\r\n")
        except:
            raise
        finally:
            self.socket.close()
        
    def register_mzml(self, mzml_name):
        self.mzml_names.append(mzml_name)
        self.file_num += 1

    def get_scan_params(self):
        mzml_path = os.path.join(self.out_dir, self.mzml_names[self.file_num - 1])
        self.child_socket.send((mzml_path + "\r\n").encode("utf-8"))
        schedule_path = self.child_socket.recv(4096).decode('utf-8').strip()
        self.schedule_names.append(
            os.path.basename(schedule_path)
        )
        
        schedule_params = dsda_get_scan_params(
            schedule_path,
            os.path.join(self.out_dir, "dsda", "DsDA_Timing_schedule.csv"),
            self.base_controller.isolation_width, 
            self.base_controller.mz_tol,
            self.base_controller.rt_tol
        )
        
        return schedule_params
        
    def get_base_controller(self):
        return copy.deepcopy(self.base_controller)