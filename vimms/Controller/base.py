import time
from collections import defaultdict

import pandas as pd
from loguru import logger

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW, DEFAULT_MS1_AGC_TARGET, \
    DEFAULT_MS1_MAXIT, \
    DEFAULT_MS1_COLLISION_ENERGY, DEFAULT_MS1_ORBITRAP_RESOLUTION, \
    DEFAULT_MS1_ACTIVATION_TYPE, \
    DEFAULT_MS1_MASS_ANALYSER, DEFAULT_MS1_ISOLATION_MODE, \
    DEFAULT_SOURCE_CID_ENERGY, DEFAULT_MS2_AGC_TARGET, \
    DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS2_ACTIVATION_TYPE, \
    DEFAULT_MS2_MASS_ANALYSER, DEFAULT_MS2_ISOLATION_MODE, INITIAL_SCAN_ID, \
    ScanParameters, get_default_scan_params, \
    get_dda_scan_param


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

        self.scans = defaultdict(
            list)  # key: ms level, value: list of scans for that level
        self.scan_to_process = None
        self.environment = None
        self.next_processed_scan_id = INITIAL_SCAN_ID
        self.initial_scan_id = INITIAL_SCAN_ID
        self.current_task_id = self.initial_scan_id
        self.processing_times = []
        self.last_ms1_rt = 0.0

    def get_ms1_scan_params(self, metadata=None):
        task = get_default_scan_params(
            polarity=self.environment.mass_spec.ionisation_mode,
            default_ms1_scan_window=self.params.default_ms1_scan_window,
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

    def get_ms2_scan_params(self, mz, intensity, precursor_scan_id,
                            isolation_width, mz_tol, rt_tol, metadata=None):
        task = get_dda_scan_param(
            mz, intensity, precursor_scan_id, isolation_width, mz_tol, rt_tol,
            agc_target=self.params.ms2_agc_target,
            max_it=self.params.ms2_max_it,
            collision_energy=self.params.ms2_collision_energy,
            source_cid_energy=self.params.ms2_source_cid_energy,
            orbitrap_resolution=self.params.ms2_orbitrap_resolution,
            activation_type=self.params.ms2_activation_type,
            mass_analyser=self.params.ms2_mass_analyser,
            isolation_mode=self.params.ms2_isolation_mode,
            polarity=self.environment.mass_spec.ionisation_mode,
            metadata=metadata)
        return task

    def get_initial_tasks(self):
        """
        Gets the initial tasks to load immediately into the mass spec
        (before acquisition starts)
        :return: an empty list of tasks, unless overridden by subclass
        """
        return []

    def get_initial_scan_params(self):
        """
        Gets the initial scan parameters to send to the mass spec that
        starts the whole process. Will default to sending an MS1 scan with
        whatever parameters passed in self.params. Subclasses can override
        this to return different types of scans.
        :return: the MS1
        """
        return self.get_ms1_scan_params()

    def set_environment(self, env):
        self.environment = env

    def handle_scan(self, scan, current_size, pending_size):
        logger.debug('tasks to be sent = %d' % (current_size))
        logger.debug('tasks sent but not received = %d' % (pending_size))

        # record every scan that we've received
        self.scans[scan.ms_level].append(scan)

        # update ms1 time (used for ROI matching)
        if scan.ms_level == 1:
            self.last_ms1_rt = scan.rt
            self.last_ms1_scan = scan

        # we get an ms1 scan and it has some peaks AND all the pending tasks
        # have been sent and processed AND this ms1 scan is a custom scan
        # we'd sent before (not a method scan) then store it for
        # fragmentation next time
        logger.debug(
            'scan.scan_id = %d, self.next_processed_scan_id = %d' % (
                scan.scan_id, self.next_processed_scan_id))
        if scan.scan_id == self.next_processed_scan_id:
            self.scan_to_process = scan
            logger.debug('Next processed scan %d has arrived' %
                         self.next_processed_scan_id)
        else:
            self.scan_to_process = None
        logger.debug('scan_to_process = %s' % self.scan_to_process)
        logger.debug('scan.scan_params = %s' % scan.scan_params)

        # implemented by subclass
        if self.scan_to_process is not None:
            # track how long each scan takes to process
            start = time.time()
            new_tasks = self._process_scan(scan)
            elapsed = time.time() - start
            self.processing_times.append(elapsed)
        else:
            # this scan is not the one we want to process, but here we
            # pass it to _process_scan anyway in case the subclass wants
            # to do something with it
            new_tasks = self._process_scan(scan)
        return new_tasks

    def update_state_after_scan(self, last_scan):
        raise NotImplementedError()

    def dump_scans(self, output_method):
        all_scans = self.scans[1] + self.scans[2]
        all_scans.sort(key=lambda x: x.scan_id)  # sort by scan_id
        out_list = []
        for scan in all_scans:
            # ignore any scan that we didn't send (no scan_params)
            if scan.scan_params is not None:
                out = {
                    'scan_id': scan.scan_id,
                    'num_peaks': scan.num_peaks,
                    'rt': scan.rt,
                    'ms_level': scan.ms_level
                }
                # add all the scan params to out
                out.update(scan.scan_params.get_all())
                out_list.append(out)

        # dump to csv
        df = pd.DataFrame(out_list)
        output_method(df.to_csv(index=False, line_terminator='\n'))

    def _process_scan(self, scan):
        raise NotImplementedError()

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
        # source_cid_energy = params.get(ScanParameters.SOURCE_CID_ENERGY)
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

    def after_injection_cleanup(self):
        pass
