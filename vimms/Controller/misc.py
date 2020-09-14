from loguru import logger

from vimms.Controller.base import Controller


class FixedScansController(Controller):
    """
    A controller which takes a schedule of scans, converts them into tasks in queue
    """

    def __init__(self, schedule, params=None):
        super().__init__(params=params)
        if len(schedule) > 0:
            self.set_tasks(schedule)

    def get_initial_tasks(self):
        return self.tasks

    def get_initial_scan_params(self):
        return self.initial_task

    def set_tasks(self, schedule):
        assert isinstance(schedule, list)
        self.initial_task = schedule[0] # used for sending the first scan
        self.tasks = schedule[1:] # used for sending all the other scans

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # simply record every scan that we've received, but return no new tasks
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)
        return []

    def update_state_after_scan(self, last_scan):
        pass
