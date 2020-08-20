from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS1_SCAN_WINDOW, DEFAULT_MS1_ACTIVATION_TYPE, DEFAULT_MS1_MASS_ANALYSER, \
    DEFAULT_MS1_ISOLATION_MODE
from vimms.Controller import Controller


class IdleController(Controller):
    """
    A controller that doesn't do any controlling.
    """

    def __init__(self):
        super().__init__()

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

    def __init__(self,
                 # advanced parameters
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 default_ms1_scan_window=DEFAULT_MS1_SCAN_WINDOW):
        super().__init__()
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution
        self.ms1_activation_type = ms1_activation_type
        self.ms1_mass_analyser = ms1_mass_analyser
        self.ms1_isolation_mode = ms1_isolation_mode
        self.default_ms1_scan_window = default_ms1_scan_window

    def _process_scan(self, scan):
        task = self.environment.get_default_scan_params(default_ms1_scan_window=self.default_ms1_scan_window,
                                                        agc_target=self.ms1_agc_target,
                                                        max_it=self.ms1_max_it,
                                                        collision_energy=self.ms1_collision_energy,
                                                        orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                        activation_type=self.ms1_activation_type,
                                                        mass_analyser=self.ms1_mass_analyser,
                                                        isolation_mode=self.ms1_isolation_mode)
        return [task]

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass
