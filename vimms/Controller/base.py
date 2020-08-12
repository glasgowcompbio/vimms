from collections import defaultdict

import pylab as plt
from loguru import logger

from vimms.Common import INITIAL_SCAN_ID


class Controller(object):
    def __init__(self):
        self.scans = defaultdict(list)  # key: ms level, value: list of scans for that level
        self.make_plot = False
        self.scan_to_process = None
        self.environment = None
        self.next_processed_scan_id = INITIAL_SCAN_ID
        self.initial_scan_id = INITIAL_SCAN_ID
        self.current_task_id = self.initial_scan_id

    def set_environment(self, env):
        self.environment = env

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # record every scan that we've received
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)

        # plot scan if there are peaks
        if scan.num_peaks > 0:
            self._plot_scan(scan)

        # we get an ms1 scan and it has some peaks AND all the pending tasks have been sent and processed AND
        # this ms1 scan is a custom scan we'd sent before (not a method scan)
        # then store it for fragmentation next time
        self.last_scan = scan
        logger.debug(
            'scan.scan_id = %d, self.next_processed_scan_id = %d' % (scan.scan_id, self.next_processed_scan_id))
        if scan.scan_id == self.next_processed_scan_id:
            self.scan_to_process = scan
            self.pending_tasks = pending_tasks_size
            logger.debug('next processed scan %d has arrived' % self.next_processed_scan_id)
        else:
            self.scan_to_process = None

        logger.debug('outgoing_queue_size = %d, pending_tasks_size = %d' % (outgoing_queue_size, pending_tasks_size))
        logger.debug('scan.scan_params = %s' % scan.scan_params)
        logger.debug('scan_to_process = %s' % self.scan_to_process)

        # implemented by subclass
        new_tasks = self._process_scan(scan)
        return new_tasks

    def update_state_after_scan(self, last_scan):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def _process_scan(self, scan):
        raise NotImplementedError()

    def _plot_scan(self, scan):
        if self.make_plot:
            plt.figure()
            for i in range(scan.num_peaks):
                x1 = scan.mzs[i]
                x2 = scan.mzs[i]
                y1 = 0
                y2 = scan.intensities[i]
                a = [[x1, y1], [x2, y2]]
                plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
            plt.title('Scan {0} {1}s -- {2} peaks'.format(scan.scan_id, scan.rt, scan.num_peaks))
            plt.show()
