import pandas as pd

from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION
from vimms.Controller import Controller
from vimms.Controller.fullscan import SimpleMs1Controller


class ScanParameterController(SimpleMs1Controller):
    def __init__(self,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 agc_values = [1e5,2e5,5e5,1e6],
                 maxit_values = [50,100,250,500]):
        super().__init__(ms1_agc_target=ms1_agc_target,
                 ms1_max_it=ms1_max_it,
                 ms1_collision_energy=ms1_collision_energy,
                 ms1_orbitrap_resolution=ms1_orbitrap_resolution)
        self.agc_values = agc_values
        self.maxit_values = maxit_values
        self.scan_number = 0
        self.scan_param_list = []
        # make a list with all combinations of parameters
        for a in self.agc_values:
            for b in self.maxit_values:
                self.scan_param_list.append((a,b))

    def _process_scan(self, scan):

        n_scan_params = len(self.scan_param_list)
        agc,maxit = self.scan_param_list[self.scan_number % n_scan_params]

        task = self.environment.get_default_scan_params(agc_target=agc,
                                                        max_it=maxit,
                                                        collision_energy=self.ms1_collision_energy,
                                                        orbitrap_resolution=self.ms1_orbitrap_resolution)
        self.scan_number += 1

        return [task]


class FixedScheduleController(Controller):
    """
    A controller which takes a schedule of scans, converts them into tasks in queue
    """

    def __init__(self, schedule_file, isolation_width, mz_tol, mass_spec_ionisation_mode, N, rt_tol,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__()
        self.first_scan = True
        self.isolation_width = isolation_width
        self.mz_tol = mz_tol
        self.mass_spec_ionisation_mode = mass_spec_ionisation_mode
        self.schedule = pd.read_csv(schedule_file, header=None, names=["ms_level", "mz", "time", "box_id", "box_file"])
        self.schedule = self.schedule.iloc[range(0, self.schedule.shape[0], 2)]
        self.scheduled_tasks = self._process_schedule()
        self.N = N
        self.rt_tol = rt_tol

        # advanced parameters
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution

    def _process_scan(self, scan):
        if self.first_scan:
            self.first_scan = False
            return self.scheduled_tasks
        else:
            return []

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass

    def _process_schedule(self):
        # converts schedule into tasks and pushes them to the queue
        tasks = []
        for i in range(self.schedule.shape[0]):
            if self.schedule.iloc[i]['ms_level'] == 'MS1':
                new_task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                    max_it=self.ms1_max_it,
                                                                    collision_energy=self.ms1_collision_energy,
                                                                    orbitrap_resolution=self.ms1_orbitrap_resolution)
            else:
                precursor_scan_id = self.scan_to_process.scan_id
                mz = self.schedule.iloc[i]['mz']
                intensity = 0
                rt_tol = 0
                new_task = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                               self.isolation_width, self.mz_tol, rt_tol,
                                                               agc_target=self.ms2_agc_target,
                                                               max_it=self.ms2_max_it,
                                                               collision_energy=self.ms2_collision_energy,
                                                               orbitrap_resolution=self.ms2_orbitrap_resolution)
            tasks.append(new_task)
        return tasks