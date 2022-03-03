"""
This file contains the implementation of agent-based controllers.
An agent is a useful abstraction that allows us to maintain
global state outside a controller. This is useful when we need to track objects, e.g.
precursors, fragmented counts, regions of interests etc, across injections.
"""

from vimms.Controller.base import Controller


class AgentBasedController(Controller):
    """
    A class that implements an agent-based controller.
    """
    def __init__(self, agent, advanced_params=None):
        """Initialises an agent-based controller.

        Arguments:
            agent: an instance of the [vimms.Agent.AbstractAgent][] class.
            advanced_params: optional advanced parameters for the mass spec.
        """
        super().__init__(advanced_params=advanced_params)
        self.agent = agent

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:
            new_tasks, self.current_task_id, self.next_processed_scan_id = \
                self.agent.next_tasks(self.scan_to_process, self,
                                      self.current_task_id)
            self.scan_to_process = None  # has been processed
        return new_tasks

    def update_state_after_scan(self, last_scan):
        self.agent.update(last_scan, self)
