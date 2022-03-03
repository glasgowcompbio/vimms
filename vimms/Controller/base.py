"""
Contains the base abstract controller class, as well as other objects that are necessary for
controllers to function.
"""
import time
from collections import defaultdict

from abc import ABC, abstractmethod

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


class AdvancedParams():
    """
    An object that stores advanced parameters used to control the mass spec
    e.g. AGC target, collision energy, orbitrap resolution etc.

    When ViMMS is connected to an actual mass spec instrument (Orbitrap Fusion in our case)
    via IAPI, most of these values are directly passed to the mass spec as they are.
    Generally you can leave these settings to their default.

    In simulated use, most of these values are not used and therefore they won't affect simulated
    results.
    """
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
        """
        Create an advanced parameter object

        Args:
            default_ms1_scan_window: the m/z window to perform MS1 scan
            ms1_agc_target: automatic gain control target for MS1 scan
            ms1_max_it: maximum time to acquire ions for MS1 scan
            ms1_collision_energy: the collision energy used for MS1 scan
            ms1_orbitrap_resolution: the Orbitrap resolution used for MS1 scan
            ms1_activation_type: the activation type for MS1 scan, either CID or HCD
            ms1_mass_analyser: the mass analyser to use for MS1 scan, either IonTrap or Orbitrap
            ms1_isolation_mode: the isolation mode for MS1 scan, either None, Quadrupole, IonTrap
            ms1_source_cid_energy: source CID energy
            ms2_agc_target: automatic gain control target for MS2 scan
            ms2_max_it: maximum time to acquire ions for MS2 scan
            ms2_collision_energy: the collision energy used for MS2 scan
            ms2_orbitrap_resolution: the Orbitrap resolution used for MS2 scan
            ms2_activation_type: the activation type for MS2 scan, either CID or HCD
            ms2_mass_analyser: the mass analyser to use for MS2 scan, either IonTrap or Orbitrap
            ms2_isolation_mode: the isolation mode for MS2 scan, either None, Quadrupole, IonTrap
            ms2_source_cid_energy: source CID energy
        """
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


class Controller(ABC):
    """
    Abtract base class for controllers.
    """
    def __init__(self, advanced_params=None):
        """
        Initialise a base Controller class.

        Args:
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
        """
        if advanced_params is None:
            self.advanced_params = AdvancedParams()
        else:
            self.advanced_params = advanced_params

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
        """
        Generate a default scan parameter for MS1 scan. The generated scan parameter object
        is typically passed to the mass spec (whether real or simulated) that produces
        the actual MS1 scan.
        Args:
            metadata: any additional metadata to include

        Returns: a [vimms.Common.ScanParameters][] object that describes the MS1 scan to generate.

        """
        task = get_default_scan_params(
            polarity=self.environment.mass_spec.ionisation_mode,
            default_ms1_scan_window=self.advanced_params.default_ms1_scan_window,
            agc_target=self.advanced_params.ms1_agc_target,
            max_it=self.advanced_params.ms1_max_it,
            collision_energy=self.advanced_params.ms1_collision_energy,
            source_cid_energy=self.advanced_params.ms1_source_cid_energy,
            orbitrap_resolution=self.advanced_params.ms1_orbitrap_resolution,
            activation_type=self.advanced_params.ms1_activation_type,
            mass_analyser=self.advanced_params.ms1_mass_analyser,
            isolation_mode=self.advanced_params.ms1_isolation_mode,
            metadata=metadata)
        return task

    def get_ms2_scan_params(self, mz, intensity, precursor_scan_id,
                            isolation_width, mz_tol, rt_tol, metadata=None):
        """
        Generate a default scan parameter for MS2 scan. The generated scan parameter object
        is typically passed to the mass spec (whether real or simulated) that produces
        the actual MS2 scan.

        Args:
            mz: the m/z of the precursor ion to fragment
            intensity: the intensity of the precursor ion to fragment
            precursor_scan_id: the associated MS1 scan ID that contains this precursor ion
            isolation_width: isolation width, in Dalton
            mz_tol: m/z tolerance for dynamic exclusion (TODO: this shouldn't be here)
            rt_tol: RT tolerance for dynamic exclusion (TODO: this shouldn't be here)
            metadata: any additional metadata to include

        Returns: a [vimms.Common.ScanParameters][] object that describes the MS2 scan to generate.

        """
        task = get_dda_scan_param(
            mz, intensity, precursor_scan_id, isolation_width, mz_tol, rt_tol,
            agc_target=self.advanced_params.ms2_agc_target,
            max_it=self.advanced_params.ms2_max_it,
            collision_energy=self.advanced_params.ms2_collision_energy,
            source_cid_energy=self.advanced_params.ms2_source_cid_energy,
            orbitrap_resolution=self.advanced_params.ms2_orbitrap_resolution,
            activation_type=self.advanced_params.ms2_activation_type,
            mass_analyser=self.advanced_params.ms2_mass_analyser,
            isolation_mode=self.advanced_params.ms2_isolation_mode,
            polarity=self.environment.mass_spec.ionisation_mode,
            metadata=metadata)
        return task

    def get_initial_tasks(self):
        """
        Gets the initial tasks to load immediately into the mass spec
        (before acquisition starts)

        Returns: an empty list of tasks, unless overridden by subclass

        """
        return []

    def get_initial_scan_params(self):
        """
        Gets the initial scan parameters to send to the mass spec that
        starts the whole process. Will default to sending an MS1 scan with
        whatever parameters passed in self.params. Subclasses can override
        this to return different types of scans.

        Returns: a [vimms.Common.ScanParameters][] object describing the initial scan to make.

        """
        return self.get_ms1_scan_params()

    def set_environment(self, env):
        """
        Set the environment used to run this controller

        Args:
            env: an [vimms.Environment.Environment] object or its subclasses.

        Returns: None

        """
        self.environment = env

    def handle_scan(self, scan, current_size, pending_size):
        """
        Basic codes to handle an incoming scan, which is generally the same for all controllers.

        Args:
            scan: A new [vimms.MassSpec.Scan][] to process.
            current_size: current size of task buffer
            pending_size: pending size of task buffer

        Returns: a list of new [vimms.Common.ScanParameters][] describing what to do next.

        """
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

    @abstractmethod
    def update_state_after_scan(self, last_scan):
        """Update internal state after a scan has been processed.

        Arguments:
            last_scan ([vimms.MassSpec.Scan][]): the last-processed object.
        """
        pass

    @abstractmethod
    def _process_scan(self, scan):
        """Process incoming scan

        Arguments:
            scan: A new [vimms.MassSpec.Scan][] to process.
        """
        pass

    def dump_scans(self, output_method):
        """
        Dump all scans to the output format.
        Useful for debugging.

        Args:
            output_method: a function that accepts scan information as a CSV string from pandas

        Returns: None

        """
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

    def _check_scan(self, params):
        """
        Checks that the conditions that are checked in
        vimms-fusion MS class pass here. This is done to ensure that the values are
        all in a consistent state.

        Args:
            params: a [vimms.Common.ScanParameters][] object to check.

        Returns:

        """

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
        """
        Clean-up method at the end of each injection.

        Returns: None

        """
        pass
