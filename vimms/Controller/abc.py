from vimms.Controller.base import Controller


class AgentBasedController(Controller):
    def __init__(self, ionisation_mode, agent, params=None):
        super().__init__(params=params)
        self.agent = agent
        self.ionisation_mode = ionisation_mode  # check we need this

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:
            new_tasks, self.current_task_id, self.next_processed_scan_id = self.agent.next_tasks(self.scan_to_process,
                                                                                                 self,
                                                                                                 self.current_task_id)
        return new_tasks

    def update_state_after_scan(self, last_scan):
        self.agent.update(last_scan, self)
