"""
This file implements controllers that produce only full-scan (MS1) data, so no MSN fragmentation
is performed at all.
"""

from vimms.Controller import Controller


class IdleController(Controller):
    """
    A controller that doesn't do any controlling.
    Mostly used as a skeleton code to illustrate the code structure in ViMMS controllers.
    """

    def __init__(self, advanced_params=None):
        """
        Initialise an idle controller
        Args:
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
        """
        super().__init__(advanced_params=advanced_params)

    def _process_scan(self, scan):
        new_tasks = []
        return new_tasks

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass


class SimpleMs1Controller(Controller):
    """
    A simple MS1 controller which does a full scan of the chemical sample,
    but no fragmentation
    """

    def __init__(self, advanced_params=None):
        """
        Initialise a full-scan MS1 controller
        Args:
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
        """
        super().__init__(advanced_params=advanced_params)

    def _process_scan(self, scan):
        if self.scan_to_process is not None:
            task = self.get_ms1_scan_params()
            self.current_task_id += 1
            self.next_processed_scan_id = self.current_task_id
            self.scan_to_process = None  # set this scan as has been processed
            return [task]

    def update_state_after_scan(self, last_scan):
        pass
