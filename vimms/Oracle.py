# Oracle.py
import numpy as np
from loguru import logger
from vimms.Exclusion import BoxHolder, ExclusionItem
from vimms.Common import ScanParameters


# some example oracles...

class AbstractOracle(object):
    def __init__(self):
        self.task_list = []
    def next_tasks(self, scan, controller, current_task_id):
        raise NotImplementedError
    def update(self, last_scan, controller):
        raise NotImplementedError

class FullScanOracle(AbstractOracle):
    def next_tasks(self, scan, controller, current_task_id):
        new_tasks = []
        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        current_task_id += 1
        next_processed_scan_id = current_task_id
        self.task_list.append((controller, ms1_scan_params))
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass
    
class TopNOracle(AbstractOracle):
    def __init__(self, N, min_ms1_intensity):
        super().__init__()
        self.N = N
        self.min_ms1_intensity = min_ms1_intensity
        self.mz_tol = 10
        self.rt_tol = 15
        self.isolation_width = 0.7
        self.exclusion = BoxHolder()

    def next_tasks(self, scan, controller, current_task_id):
        new_tasks = []
        mzs = scan.mzs
        intensities = scan.intensities
        assert mzs.shape == intensities.shape
        rt = scan.rt
        idx = np.argsort(intensities)[::-1]


        fragmented_count = 0
        for i in idx:
            if fragmented_count >= self.N:
                break
            mz = mzs[i]
            intensity = intensities[i]

            if intensity < self.min_ms1_intensity:
                break
        
            if self.exclusion.is_in_box(mz, rt): # will always return false in this controller, but used in children
                continue

            precursor_scan_id = scan.scan_id
            dda_scan_params = controller.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
                                                        self.mz_tol, self.rt_tol)
            new_tasks.append(dda_scan_params)
            self.task_list.append((controller, dda_scan_params))
            fragmented_count += 1
            current_task_id += 1
        
        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        self.task_list.append((controller, ms1_scan_params))
        current_task_id += 1
        next_processed_scan_id = current_task_id
        
        return new_tasks, current_task_id, next_processed_scan_id
    
    def update(self, last_scan, controller):
        pass

class TopNDEWOracle(TopNOracle):
    def __init__(self, N, min_ms1_intensity, mz_tol, rt_tol):
        super().__init__(N, min_ms1_intensity)
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
    
    def update(self, last_scan, controller):
        rt = last_scan.rt
        if last_scan.ms_level >= 2:  # if ms-level is 2, it's a custom scan and we should always know its scan parameters
            assert last_scan.scan_params is not None
            for precursor in last_scan.scan_params.get(ScanParameters.PRECURSOR_MZ):
                # add dynamic exclusion item to the exclusion list to prevent the same precursor ion being fragmented
                # multiple times in the same mz and rt window
                # Note: at this point, fragmentation has occurred and time has been incremented! so the time when
                # items are checked for dynamic exclusion is the time when MS2 fragmentation occurs
                # TODO: we need to add a repeat count too, i.e. how many times we've seen a fragment peak before
                #  it gets excluded (now it's basically 1)

                # TODO: check if already excluded and, if so, just move the time
                mz = precursor.precursor_mz
                mz_lower = mz * (1 - self.mz_tol / 1e6)
                mz_upper = mz * (1 + self.mz_tol / 1e6)
                rt_lower = rt - self.rt_tol
                rt_upper = rt + self.rt_tol
                x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper,
                          frag_at=rt)
                self.exclusion.add_box(x)
        
        