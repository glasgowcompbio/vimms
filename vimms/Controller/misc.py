from loguru import logger

from vimms.Controller.base import Controller


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
