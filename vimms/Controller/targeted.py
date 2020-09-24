# vimms.Controller.targeted

import csv
from molmass import Formula
from mass_spec_utils.adduct_calculator.adduct_rules import AdductTransformer

from vimms.Controller.base import Controller
from vimms.Common import DEFAULT_ISOLATION_WIDTH
from vimms.MassSpec import ScanParameters
from loguru import logger

def create_targets_from_toxid(toxid_file_name, file_rt_units='minutes', mz_delta=10, rt_delta=60., polarity_filter=['+'], adducts_to_use=['[M+H]+', '[M+K]+', '[M+Na]+']):
    """
    Note: mz_delta is in ppm
    """
    target_list = []
    
    with open(str(toxid_file_name), 'r') as f:
        reader = csv.reader(f)
        line = [None]
        while len(line) == 0 or not line[0] == 'Index':
            line = next(reader)
        # we will now be in the data
        at = AdductTransformer()

        for line in reader:
            if len(line) == 0 or line[0] == '-': # empty line, or undetected compound
                continue
            name = line[1]
            formula = line[2]
            polarity = line[3]
            if not polarity in polarity_filter:
                continue
            expected_rt = float(line[5])
            if file_rt_units == 'minutes':
                expected_rt *= 60.
            for val in line[8:]:
                assert val == '-' or val  ==  ''
            metadata = {'name': name, 'formula': formula, 'polarity': polarity, \
                        'expected_rt': expected_rt}
            
            for adduct in adducts_to_use:
                theoretical_mz = at.mass2ion(Formula(formula).isotope.mass, adduct)
                min_mz = theoretical_mz - theoretical_mz * mz_delta / 1e6
                max_mz = theoretical_mz + theoretical_mz * mz_delta / 1e6
                min_rt = expected_rt - rt_delta
                max_rt = expected_rt + rt_delta
                new_target = Target(theoretical_mz, min_mz, max_mz, min_rt, max_rt, name=name, metadata=metadata, adduct=adduct)
                target_list.append(new_target)

    return target_list
    

class Target(object):
    def __init__(self, mz, min_mz, max_mz, min_rt, max_rt, name=None, adduct=None,metadata=None):
        self.mz = mz
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_rt = min_rt
        self.max_rt = max_rt

        self.name = name
        self.metadata = metadata
        self.adduct = adduct
    
    def active(self, mz_intensity, rt, min_intensity_for_fragmentation):
        # check if there is a peak inside this box
        # if there is, return true, else return false
        # mzi is a zip of the mz and intensity lists from a scan
        if rt < self.min_rt or rt >  self.max_rt:
            return False
        sub_mzi = list(filter(lambda x: x[0] >= self.min_mz and x[0] <= self.max_mz and x[1] >= min_intensity_for_fragmentation, mz_intensity))
        if len(sub_mzi) > 0:
            return True
        else:
            return False
    
    def __str__(self):
        if self.name is not None:
            return "{}{} (m/z: {}->{}, rt: {}->{})".format(self.name, self.adduct, self.min_mz, self.max_mz, self.min_rt, self.max_rt)
        else:
            return "(m/z: {}->{}, rt: {}->{})".format(self.min_mz, self.max_mz, self.min_rt, self.max_rt)

class TargetedController(Controller):
    """
    A controller that is given a list of m/z and RT values to target
    Attempts to acquire n_replicates of each target at each CE
    """
    def __init__(self, targets, ce_values, N=10, n_replicates=1, min_ms1_intensity=5e3, isolation_width=DEFAULT_ISOLATION_WIDTH, params=None, limit_acquisition=True):
        super().__init__(params=params)
        self.targets = targets
        self.ce_values = ce_values
        self.n_replicates = n_replicates
        self.N = N
        self.isolation_width = isolation_width
        self.min_ms1_intensity = min_ms1_intensity
        self.limit_acquisition = limit_acquisition

        # these will be removed sometime
        self.mz_tol = 10
        self.rt_tol = 10

        self.target_counts = {}
        for t in self.targets:
            self.target_counts[t] = {c: 0 for c in self.ce_values}

        self.scan_record = [] # keeps track of which scan is which
        self.seen_targets = set()

    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:
            precursor_scan_id = self.scan_to_process.scan_id
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt
            mzi = list(zip(mzs, intensities))

            active_targets = list(filter(lambda x: x.active(mzi, rt, self.min_ms1_intensity), self.targets))
            self.seen_targets.update(active_targets)

            target_list = []
            for t in active_targets:
                for ce in self.target_counts[t]:
                    if self.limit_acquisition and self.target_counts[t][ce] == self.n_replicates:
                        continue
                    else:
                        target_list.append((t, ce, self.target_counts[t][ce]))

            
            if len(target_list) > 0:
                target_list.sort(key = lambda x: x[2]) # prioritise by how far we are below the number of repetitions we want
                # make some MS2 scans, upto N
                for i in range(min(len(target_list), self.N)):
                    t, ce, _ = target_list[i]
                    metadata = dict(t.metadata) # copy
                    metadata['adduct'] = t.adduct
                    dda_scan_params = self.get_ms2_scan_params(t.mz, 1e3, precursor_scan_id, self.isolation_width, \
                                                        self.mz_tol, self.rt_tol, metadata=metadata)
                    dda_scan_params.set(ScanParameters.COLLISION_ENERGY, ce)
                    new_tasks.append(dda_scan_params)
                    self.current_task_id += 1
                    self.target_counts[t][ce] += 1
                    self.scan_record.append([self.current_task_id, t, ce])
            
            # make the MS1 scan
            ms1_scan_params = self.get_ms1_scan_params()
            self.current_task_id += 1
            self.next_processed_scan_id = self.current_task_id
            self.scan_record.append([self.current_task_id, None, None])
            new_tasks.append(ms1_scan_params)
        return new_tasks

    def summarise_activity(self, output_method):
        output_method("Summary of targets")
        output_method("=================")
        found = set()
        found_names = set()
        unique_names = set([t.name for t in self.targets])
        for target in self.targets:
            output_method(target)
            for c in self.ce_values:
                output_method('\t{} -> {} scans'.format(c, self.target_counts[target][c]))
                if self.target_counts[target][c] > 0:
                    found.add(target)
                    found_names.add(target.name)
        output_method("SUMMARY")
        output_method("========")
        output_method("{} out of {} have one or more scans".format(len(found), len(self.targets)))
        output_method("{} of the {} unique names have more than one scan".format(len(found_names), len(unique_names)))


