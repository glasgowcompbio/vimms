from collections import defaultdict

import pylab as plt
from loguru import logger

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW, DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, \
    DEFAULT_MS1_COLLISION_ENERGY, DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS1_ACTIVATION_TYPE, \
    DEFAULT_MS1_MASS_ANALYSER, DEFAULT_MS1_ISOLATION_MODE, DEFAULT_SOURCE_CID_ENERGY, DEFAULT_MS2_AGC_TARGET, \
    DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS2_ACTIVATION_TYPE, \
    DEFAULT_MS2_MASS_ANALYSER, DEFAULT_MS2_ISOLATION_MODE, INITIAL_SCAN_ID
from vimms.MassSpec import ScanParameters


class AdvancedParams(object):
    def __init__(self,
                 default_ms1_scan_window=DEFAULT_MS1_SCAN_WINDOW,
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 ms1_source_cid_energy=DEFAULT_SOURCE_CID_ENERGY,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE,
                 ms2_source_cid_energy=DEFAULT_SOURCE_CID_ENERGY):
        self.default_ms1_scan_window = default_ms1_scan_window

        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution
        self.ms1_activation_type = ms1_activation_type
        self.ms1_mass_analyser = ms1_mass_analyser
        self.ms1_isolation_mode = ms1_isolation_mode
        self.ms1_source_cid_energy = ms1_source_cid_energy

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution
        self.ms2_activation_type = ms2_activation_type
        self.ms2_mass_analyser = ms2_mass_analyser
        self.ms2_isolation_mode = ms2_isolation_mode
        self.ms2_source_cid_energy = ms2_source_cid_energy


class Controller(object):
    def __init__(self, params=None):
        if params is None:
            self.params = AdvancedParams()
        else:
            self.params = params

        self.scans = defaultdict(list)  # key: ms level, value: list of scans for that level
        self.make_plot = False
        self.scan_to_process = None
        self.environment = None
        self.next_processed_scan_id = INITIAL_SCAN_ID
        self.initial_scan_id = INITIAL_SCAN_ID
        self.current_task_id = self.initial_scan_id

    def get_ms1_scan_params(self, metadata=None):
        task = self.environment.get_default_scan_params(default_ms1_scan_window=self.params.default_ms1_scan_window,
                                                        agc_target=self.params.ms1_agc_target,
                                                        max_it=self.params.ms1_max_it,
                                                        collision_energy=self.params.ms1_collision_energy,
                                                        source_cid_energy=self.params.ms1_source_cid_energy,
                                                        orbitrap_resolution=self.params.ms1_orbitrap_resolution,
                                                        activation_type=self.params.ms1_activation_type,
                                                        mass_analyser=self.params.ms1_mass_analyser,
                                                        isolation_mode=self.params.ms1_isolation_mode,
                                                        metadata=metadata)
        return task

    def get_ms2_scan_params(self, mz, intensity, precursor_scan_id, isolation_width, mz_tol, rt_tol, metadata=None):
        task = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                   isolation_width, mz_tol, rt_tol,
                                                   agc_target=self.params.ms2_agc_target,
                                                   max_it=self.params.ms2_max_it,
                                                   collision_energy=self.params.ms2_collision_energy,
                                                   source_cid_energy=self.params.ms2_source_cid_energy,
                                                   orbitrap_resolution=self.params.ms2_orbitrap_resolution,
                                                   activation_type=self.params.ms2_activation_type,
                                                   mass_analyser=self.params.ms2_mass_analyser,
                                                   isolation_mode=self.params.ms2_isolation_mode,
                                                   metadata=metadata)
        return task

    def get_initial_tasks(self):
        """
        Gets the initial tasks to load immediately into the mass spec (before acquisition starts)
        :return: an empty list of tasks, unless overridden by subclass
        """
        return []

    def get_initial_scan_params(self):
        """
        Gets the initial scan parameters to send to the mass spec that starts the whole process.
        Will default to sending an MS1 scan with whatever parameters passed in self.params
        Subclasses can override this to return different types of scans.
        :return: the MS1
        """
        return self.get_ms1_scan_params()

    def set_environment(self, env):
        self.environment = env

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # record every scan that we've received
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)

        # plot scan if there are peaks
        if scan.num_peaks > 0:
            self._plot_scan(scan)

        # we get an ms1 scan and it has some peaks AND all the pending tasks have been sent and processed AND
        # this ms1 scan is a custom scan we'd sent before (not a method scan)
        # then store it for fragmentation next time
        self.last_scan = scan
        logger.debug(
            'scan.scan_id = %d, self.next_processed_scan_id = %d' % (scan.scan_id, self.next_processed_scan_id))
        if scan.scan_id == self.next_processed_scan_id:
            self.scan_to_process = scan
            self.pending_tasks = pending_tasks_size
            logger.debug('next processed scan %d has arrived' % self.next_processed_scan_id)
        else:
            self.scan_to_process = None

        logger.debug(
            'outgoing_queue_size = %d, pending_tasks_size = %d' % (outgoing_queue_size, pending_tasks_size))
        logger.debug('scan.scan_params = %s' % scan.scan_params)
        logger.debug('scan_to_process = %s' % self.scan_to_process)

        # implemented by subclass
        new_tasks = self._process_scan(scan)
        return new_tasks

    def update_state_after_scan(self, last_scan):
        raise NotImplementedError()

    def _process_scan(self, scan):
        raise NotImplementedError()

    def _plot_scan(self, scan):
        if self.make_plot:
            plt.figure()
            for i in range(scan.num_peaks):
                x1 = scan.mzs[i]
                x2 = scan.mzs[i]
                y1 = 0
                y2 = scan.intensities[i]
                a = [[x1, y1], [x2, y2]]
                plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
            plt.title('Scan {0} {1}s -- {2} peaks'.format(scan.scan_id, scan.rt, scan.num_peaks))
            plt.show()

    def _check_scan(self, params):

        # checks that the conditions that are checked in 
        # vimms-fusion MS class pass here
        collision_energy = params.get(ScanParameters.COLLISION_ENERGY)
        orbitrap_resolution = params.get(ScanParameters.ORBITRAP_RESOLUTION)
        activation_type = params.get(ScanParameters.ACTIVATION_TYPE)
        mass_analyser = params.get(ScanParameters.MASS_ANALYSER)
        isolation_mode = params.get(ScanParameters.ISOLATION_MODE)
        agc_target = params.get(ScanParameters.AGC_TARGET)
        max_it = params.get(ScanParameters.MAX_IT)
        source_cid_energy = params.get(ScanParameters.SOURCE_CID_ENERGY)
        polarity = params.get(ScanParameters.POLARITY)
        first_mass = params.get(ScanParameters.FIRST_MASS)
        last_mass = params.get(ScanParameters.LAST_MASS)

        assert collision_energy is not None
        assert orbitrap_resolution is not None
        assert activation_type is not None
        assert mass_analyser is not None
        assert isolation_mode is not None
        assert agc_target is not None
        assert max_it is not None
        assert polarity is not None
        assert first_mass is not None
        assert last_mass is not None
