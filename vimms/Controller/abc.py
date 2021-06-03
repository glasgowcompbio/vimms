from vimms.Controller.base import Controller


class AgentBasedController(Controller):
    def __init__(self, agent, params=None):
        super().__init__(params=params)
        self.agent = agent

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:
            new_tasks, self.current_task_id, self.next_processed_scan_id = self.agent.next_tasks(self.scan_to_process,
                                                                                                 self,
                                                                                                 self.current_task_id)
            self.scan_to_process = None # has been processed
        return new_tasks

    def update_state_after_scan(self, last_scan):
        self.agent.update(last_scan, self)
