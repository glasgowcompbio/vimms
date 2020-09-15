import itertools
import numpy as np
from loguru import logger

from vimms.Controller.base import Controller
from vimms.Common import DEFAULT_ISOLATION_WIDTH

class FixedScansController(Controller):
    """
    A controller which takes a schedule of scans, converts them into tasks in queue
    """

    def __init__(self, schedule=None, params=None):
        """
        Creates a FixedScansController that accepts a list of schedule of scan parameters
        :param schedule: a list of ScanParameter objects
        :param params: mass spec advanced parameters, if any
        """
        super().__init__(params=params)
        self.tasks = None
        self.initial_task = None
        if schedule is not None and len(schedule) > 0:
            # if schedule is provided, set it
            self.set_tasks(schedule)

    def get_initial_tasks(self):
        """
        Returns all the remaining scan parameter objects to be pushed to the mass spec queue
        :return: all the remaining tasks
        """
        assert self.tasks is not None  # the remaining scan parameters in the schedule must have been set
        return self.tasks

    def get_initial_scan_params(self):
        """
        Returns the initial scan parameter object to send when acquisition starts
        :return: the initial task
        """
        assert self.initial_task is not None  # the first scan parameters in the schedule must have been set
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

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # simply record every scan that we've received, but return no new tasks
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)
        return []

    def update_state_after_scan(self, last_scan):
        pass

class MultiIsolationController(Controller):
    def __init__(self, N, isolation_width=DEFAULT_ISOLATION_WIDTH, params=None):
        super().__init__(params=params)
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
        for L in range(1, len(initial_idx)+1):
            for subset in itertools.combinations(initial_idx, L):
                scan_order.append(subset)
        return scan_order
    
    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        fragmented_count = 0
        if self.scan_to_process is not None:
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt
            idx = np.argsort(intensities)[::-1]
            precursor_scan_id = self.scan_to_process.scan_id
            scan_order = self._make_scan_order(min(self.N,len(mzs)))

            for subset in scan_order:
                mz = []
                intensity = []
                for s in subset:
                    mz.append(mzs[idx[s]])
                    intensity.append(mzs[idx[s]])
                dda_scan_params = self.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
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
