# Agent.py
import collections
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from vimms.Exclusion import TopNExclusion


class AbstractAgent(ABC):
    """An abstract class to represent an Agent.

    Agents has a scope outside the controller,
    and it can be handy way to enable a controller that has a global state across runs.
    """

    def __init__(self):
        self.task_list = []

    @abstractmethod
    def next_tasks(self, scan_to_process, controller, current_task_id):
        """Schedule the next action

        Subclasses should implement this to handle scans in the appropriate way.

        Arguments:
            scan_to_process (vimms.MassSpec.Scan): the next scan to process
            controller (vimms.Controller.base.Controller): parent controller class for this agent
            current_task_id (int): the current task ID

        Returns:
            (tuple): a tuple containing:

                new_tasks(vimms.Common.ScanParameters): a list of new tasks to run
                current_task_id (int): the id of the current task
                next_processed_scan_id (int): the id of the next incoming scan to process
        """
        pass

    @abstractmethod
    def update(self, last_scan, controller):
        """Updates agent internal state based on the last scan

        Arguments:
            last_scan: the last scan that has been processed
            controller: the parent controller class that contains this agent
        """
        pass

    @abstractmethod
    def act(self, scan_to_process):
        """Acts on a scan

        Arguments:
            scan_to_process: the scan to process
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal state of this Agent
        """
        pass


class FullScanAgent(AbstractAgent):
    def __init__(self):
        """Create a full-scan agent that performs only MS1 scans.
        """
        super().__init__()

    def next_tasks(self, scan_to_process, controller, current_task_id):
        new_tasks = []
        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        current_task_id += 1
        next_processed_scan_id = current_task_id
        self.task_list.append((controller, ms1_scan_params))
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass

    def act(self, scan_to_process):
        pass

    def reset(self):
        pass


class TopNDEWAgent(AbstractAgent):
    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                 min_ms1_intensity):
        """Create a Top-N agent that performs the standard Top-N fragmentation typically seen
        in Data-Dependant Acquisition (DDA) process.

        Arguments:
            ionisation_mode (string): the ionisation mode, either POSITIVE or NEGATIVE.
            N (int): the number of top-N most intense precursors to fragment.
            isolation_width (float): the isolation width, in Dalton.
            mz_tol (float): m/z tolerance for dynamic exclusion
            rt_tol (float): retention time tolerance (in seconds) for dynamic exclusion.
            min_ms1_intensity (float): the minimum intensity of MS1 (precursor) peak to fragment.
        """
        super().__init__()
        self.ionisation_mode = ionisation_mode
        self.N = N
        self.isolation_width = isolation_width
        self.min_ms1_intensity = min_ms1_intensity
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
        self.exclusion = TopNExclusion()
        self.seen_actions = collections.Counter()

    def next_tasks(self, scan_to_process, controller, current_task_id):
        self.act(scan_to_process)
        new_tasks, current_task_id, next_processed_scan_id = \
            self._schedule_tasks(controller, current_task_id, scan_to_process)
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass

    def act(self, scan_to_process):
        pass

    def reset(self):
        self.exclusion = TopNExclusion()
        self.seen_actions = collections.Counter()

    def _schedule_tasks(self, controller, current_task_id, scan_to_process):
        new_tasks = []
        fragmented_count = 0
        mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)

        # loop over points in decreasing intensity
        idx = np.argsort(intensities)[::-1]
        ms2_tasks = []
        for i in idx:
            mz = mzs[i]
            intensity = intensities[i]

            # stopping criteria is after we've fragmented N ions or
            # we found ion < min_intensity
            if fragmented_count >= self.N:
                logger.debug(
                    'Time %f Top-%d ions have been selected' % (rt, self.N))
                break

            if intensity < self.min_ms1_intensity:
                logger.debug(
                    'Time %f Minimum intensity threshold %f reached at %f, %d'
                    % (rt, self.min_ms1_intensity, intensity, fragmented_count)
                )
                break

            # skip ion in the dynamic exclusion list of the mass spec
            is_exc, weight = self.exclusion.is_excluded(mz, rt)
            if is_exc:
                continue

            # create a new ms2 scan parameter to be sent to the mass spec
            precursor_scan_id = scan_to_process.scan_id
            dda_scan_params = controller.get_ms2_scan_params(
                mz, intensity, precursor_scan_id, self.isolation_width,
                self.mz_tol, self.rt_tol)
            new_tasks.append(dda_scan_params)
            ms2_tasks.append(dda_scan_params)
            fragmented_count += 1
            current_task_id += 1

        # add ms1 at the end
        ms1_scan_params = controller.get_ms1_scan_params()
        current_task_id += 1
        next_processed_scan_id = current_task_id
        new_tasks.append(ms1_scan_params)

        # create new exclusion items based on the scheduled ms2 tasks
        self.exclusion.update(scan_to_process, ms2_tasks)
        return new_tasks, current_task_id, next_processed_scan_id

    def _get_mzs_rt_intensities(self, scan_to_process):
        mzs = scan_to_process.mzs
        intensities = scan_to_process.intensities
        rt = scan_to_process.rt
        assert mzs.shape == intensities.shape
        return mzs, rt, intensities
