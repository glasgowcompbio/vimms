import numpy as np
import pandas as pd
from loguru import logger

from vimms.Common import DEFAULT_ISOLATION_WIDTH
from vimms.Controller.base import Controller


class FixedScansController(Controller):
    """
    A controller which takes a schedule of scans, converts them into tasks in queue
    """

    def __init__(self, ionisation_mode, schedule,
                 isolation_width=DEFAULT_ISOLATION_WIDTH,
                 params=None):
        super().__init__(params=params)

        self.ionisation_mode = ionisation_mode
        self.isolation_width = isolation_width
        self.schedule = schedule

    def get_initial_tasks(self):
        if isinstance(self.schedule, list):  # allows sending list of tasks
            tasks = self.schedule
        else:
            tasks = self._load_tasks(self.schedule)
        return tasks

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # record every scan that we've received
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)

        return []

    def update_state_after_scan(self, last_scan):
        pass

    def _load_tasks(self, schedule):
        """
        Converts schedule to tasks
        :param schedule: a DataFrame of scheduled scans
        :return: a list of mass spec tasks
        """
        new_tasks = []
        for idx, row in schedule.iterrows():
            if row['ms_level'] == 1:
                self.last_ms1_id = self.current_task_id
                agc_target = self._get_value_or_default(row, 'agc_target', self.params.ms1_agc_target)
                max_it = self._get_value_or_default(row, 'max_it', self.params.ms1_max_it)
                collision_energy = self._get_value_or_default(row, 'collision_energy', self.params.ms1_collision_energy)
                orbitrap_resolution = self._get_value_or_default(row, 'orbitrap_resolution',
                                                                 self.params.ms1_orbitrap_resolution)
                activation_type = self._get_value_or_default(row, 'activation_type', self.params.ms1_activation_type)
                mass_analyser = self._get_value_or_default(row, 'mass_analyser', self.params.ms1_mass_analyser)
                isolation_mode = self._get_value_or_default(row, 'isolation_mode', self.params.ms1_isolation_mode)
                ms1_scan_params = self.environment.get_default_scan_params(agc_target=agc_target,
                                                                           max_it=max_it,
                                                                           collision_energy=collision_energy,
                                                                           orbitrap_resolution=orbitrap_resolution,
                                                                           activation_type=activation_type,
                                                                           mass_analyser=mass_analyser,
                                                                           isolation_mode=isolation_mode)
                new_tasks.append(ms1_scan_params)
            else:
                precursor_scan_id = self.last_ms1_id
                precursor_mz = row['precursor_mz']
                precursor_intensity = 100
                isolation_width = self._get_value_or_default(row, 'isolation_width', self.isolation_width)
                agc_target = self._get_value_or_default(row, 'agc_target', self.params.ms2_agc_target)
                max_it = self._get_value_or_default(row, 'max_it', self.params.ms2_max_it)
                collision_energy = self._get_value_or_default(row, 'collision_energy', self.params.ms2_collision_energy)
                orbitrap_resolution = self._get_value_or_default(row, 'orbitrap_resolution',
                                                                 self.params.ms2_orbitrap_resolution)
                activation_type = self._get_value_or_default(row, 'activation_type', self.params.ms2_activation_type)
                mass_analyser = self._get_value_or_default(row, 'mass_analyser', self.params.ms2_mass_analyser)
                isolation_mode = self._get_value_or_default(row, 'isolation_mode', self.params.ms2_isolation_mode)
                dda_scan_params = self.environment.get_dda_scan_param(precursor_mz, precursor_intensity,
                                                                      precursor_scan_id,
                                                                      isolation_width, 0, 0,
                                                                      agc_target=agc_target,
                                                                      max_it=max_it,
                                                                      collision_energy=collision_energy,
                                                                      orbitrap_resolution=orbitrap_resolution,
                                                                      activation_type=activation_type,
                                                                      mass_analyser=mass_analyser,
                                                                      isolation_mode=isolation_mode)
                new_tasks.append(dda_scan_params)
            self.current_task_id += 1
        return new_tasks

    def _get_value_or_default(self, row, key, default):
        try:
            value = row[key]
        except KeyError:
            value = default
        return value


class ScheduleGenerator(object):
    """
    A class to generate scheduled tasks as a dataframe for FixedScansController
    """

    def __init__(self, initial_ms1, end_ms1, precursor_mz, num_topN_blocks, N,
                 ms1_mass_analyser, ms2_mass_analyser,
                 activation_type, isolation_mode):
        self.schedule = self._generate_schedule(initial_ms1, end_ms1, precursor_mz, num_topN_blocks, N,
                                                ms1_mass_analyser, ms2_mass_analyser,
                                                activation_type, isolation_mode)

    def estimate_max_time(self):
        # assume ms1 scans will take 0.4 seconds, ms2 scans will take 0.2 seconds
        ms1_scan_time = 0.4
        ms2_scan_time = 0.2
        count_ms1_scan = np.sum(self.schedule['ms_level'] == 1)
        count_ms2_scan = np.sum(self.schedule['ms_level'] == 2)
        max_time = count_ms1_scan * ms1_scan_time + count_ms2_scan * ms2_scan_time
        max_time = max_time + 30
        return max_time

    def _generate_schedule(self, initial_ms1, end_ms1, precursor_mz, num_topN_blocks, N,
                           ms1_mass_analyser, ms2_mass_analyser,
                           activation_type, isolation_mode):
        # generate the initial MS1 scans
        initial_ms1_tasks = self._generate_ms1_tasks(initial_ms1, ms1_mass_analyser, activation_type, isolation_mode)

        # generate num_topN_blocks of Top-N blocks
        ms2_blocks = self._generate_TopN_tasks(precursor_mz, num_topN_blocks, N, ms1_mass_analyser, ms2_mass_analyser,
                                               activation_type, isolation_mode)

        # generate the ending MS1 tasks
        end_ms1_tasks = self._generate_ms1_tasks(end_ms1, ms1_mass_analyser, activation_type, isolation_mode)

        # convert to pandas dataframe
        data = initial_ms1_tasks + ms2_blocks + end_ms1_tasks
        schedule = pd.DataFrame(data,
                                columns=['ms_level', 'precursor_mz', 'mass_analyser', 'activation_type',
                                         'isolation_mode'])
        return schedule

    def _generate_ms1_tasks(self, repeat, ms1_mass_analyser, activation_type, isolation_mode):
        return [(1, None, ms1_mass_analyser, activation_type, isolation_mode)] * repeat

    def _generate_TopN_tasks(self, precursor_mz, num_topN_blocks, N, ms1_mass_analyser, ms2_mass_analyser,
                             activation_type, isolation_mode):
        ms1_task = self._generate_ms1_tasks(1, ms1_mass_analyser, activation_type, isolation_mode)
        ms2_tasks = [(2, precursor_mz, ms2_mass_analyser, activation_type, isolation_mode)] * N
        return (ms1_task + ms2_tasks) * num_topN_blocks
