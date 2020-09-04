from vimms.Controller import Controller


class IdleController(Controller):
    """
    A controller that doesn't do any controlling.
    """

    def __init__(self, params=None):
        super().__init__(params=params)

    def _process_scan(self, scan):
        new_tasks = []
        return new_tasks

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass


class SimpleMs1Controller(Controller):
    """
    A simple MS1 controller which does a full scan of the chemical sample, but no fragmentation
    """

    def __init__(self, params=None):
        super().__init__(params=params)

    def _process_scan(self, scan):
        task = self.get_ms1_scan_params()
        return [task]

    def update_state_after_scan(self, last_scan):
        pass
