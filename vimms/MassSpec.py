import math

import numpy as np
import scipy
from events import Events
from loguru import logger

from vimms.Common import (
    adduct_transformation, DEFAULT_SCAN_TIME_DICT,
    INITIAL_SCAN_ID, ScanParameters
)
from vimms.Noise import NoPeakNoise


class Peak():
    """
    A class to represent an empirical or sampled scan-level peak object
    """

    def __init__(self, mz, rt, intensity, ms_level):
        """
        Creates a peak object

        Args:
            mz: mass-to-charge value
            rt: retention time value
            intensity: intensity value
            ms_level: MS level
        """
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.ms_level = ms_level

    def __repr__(self):
        return 'Peak mz=%.4f rt=%.2f intensity=%.2f ms_level=%d' % (
            self.mz, self.rt, self.intensity, self.ms_level)

    def __eq__(self, other):
        if not isinstance(other, Peak):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return math.isclose(self.mz, other.mz) and \
               math.isclose(self.rt, other.rt) and \
               math.isclose(self.intensity, other.intensity) and \
               self.ms_level == other.ms_level


class Scan():
    """
    A class to store scan information
    """

    def __init__(self, scan_id, mzs, intensities, ms_level, rt,
                 scan_duration=None, scan_params=None, parent=None,
                 fragevent=None):
        """
        Creates a scan

        Args:
            scan_id: current scan id
            mzs: an array of mz values
            intensities: an array of intensity values
            ms_level: the ms level of this scan
            rt: the retention time of this scan
            scan_duration: how long this scan takes, if known.
            scan_params: the parameters used to generate this scan, if known
            parent: parent precursor peak, if known
            fragevent: fragmentation event associated to this scan, if any
        """
        assert len(mzs) == len(intensities)
        self.scan_id = scan_id

        # ensure that mzs and intensites are sorted by their mz values
        p = mzs.argsort()
        self.mzs = mzs[p]
        self.intensities = intensities[p]

        self.ms_level = ms_level
        self.rt = rt
        self.num_peaks = len(mzs)

        self.scan_duration = scan_duration
        self.scan_params = scan_params
        self.parent = parent
        self.fragevent = fragevent
        
    @classmethod
    def from_mzmlscan(self, scan):
        mzs, intensities = zip(*scan.peaks)
        return Scan(
            scan_id = scan.scan_no,
            mzs = np.array(mzs),
            intensities = np.array(intensities),
            ms_level = scan.ms_level,
            rt = scan.rt_in_seconds
        )

    def __repr__(self):
        return 'Scan %d num_peaks=%d rt=%.2f ms_level=%d' % (
            self.scan_id, self.num_peaks, self.rt, self.ms_level)


class ScanEvent():
    """
    A class to store fragmentation events. Mostly used for benchmarking purpose
    """

    def __init__(self, chem, query_rt, ms_level, peaks, scan_id,
                 parents_intensity=None, parent_adduct=None,
                 parent_isotope=None, precursor_mz=None, isolation_window=None,
                 scan_params=None):
        """
        Creates a fragmentation event

        Args:
            chem: the chemical that were fragmented
            query_rt: the time when fragmentation occurs
            ms_level: MS level of fragmentation
            peaks: the set of peaks produced during the fragmentation event
            scan_id: the scan id linked to this fragmentation event
            parents_intensity: the intensity of the chemical that was
                               fragmented at the time it was fragmented
            parent_adduct: the adduct that was fragmented of the chemical
            parent_isotope: the isotope that was fragmented of the chemical
            precursor_mz: the precursor mz of the scan
            isolation_window: the isolation window of the scan
            scan_params: the scan parameter settings that were used
        """
        self.chem = chem
        self.query_rt = query_rt
        self.ms_level = ms_level
        self.peaks = peaks
        self.scan_id = scan_id
        self.parents_intensity = parents_intensity  # only ms2
        self.parent_adduct = parent_adduct  # only ms2
        self.parent_isotope = parent_isotope  # only ms2
        self.precursor_mz = precursor_mz
        self.isolation_window = isolation_window
        self.scan_params = scan_params

    def __repr__(self):
        return 'MS%d ScanEvent for %s at %f' % (
            self.ms_level, self.chem, self.query_rt)


class TaskManager():
    """
    A class to track how many new tasks (scan commands) that we can send,
    given the buffer size of the mass spec.
    """

    def __init__(self, buffer_size=5):
        """
        Initialises the task manager.

        At any point, this class will ensure that there is not more than this
        number of tasks enqueued on the mass spec, see
        https://github.com/thermofisherlsms/iapi/issues/22.

        Args:
            buffer_size: maximum buffer or queue size on the mass spec.
        """
        self.current_tasks = []
        self.pending_tasks = []
        self.buffer_size = buffer_size

    def add_current(self, tasks):
        """
        Add to the list of current tasks (ready to send)

        Args:
            tasks: list of current tasks

        Returns: None

        """
        self.current_tasks.extend(tasks)

    def add_pending(self, tasks):
        """
        Add to the list of pending tasks (sent but not received)

        Args:
            tasks: list of pending tasks

        Returns: None

        """
        self.pending_tasks.extend(tasks)

    def remove_pending(self, completed_task):
        """
        Remove a completed task from the list of pending tasks

        Args:
            completed_task: a newly completed task

        Returns: None

        """
        self.pending_tasks = [t for t in self.pending_tasks if
                              t != completed_task]

    def to_send(self):
        """
        Select current tasks that could be sent to the mass spec,
        ensuring that buffer_size on the mass spec is not exceeded.

        Returns: None

        """
        batch_size = self.buffer_size - self.pending_size()
        batch = []
        while self.current_size() > 0 and len(batch) < batch_size:
            batch.append(self.pop_current())
        return batch

    def pop_current(self):
        """
        Remove the first current task (ready to send)

        Returns: a current task

        """
        return self.current_tasks.pop(0)

    def pop_pending(self):
        """
        Remove the first pending task (sent but not received)

        Returns: a pending task

        """
        return self.pending_tasks.pop(0)

    def peek_current(self):
        """
        Get the first current task (ready to send) without removing it

        Returns: a current task

        """
        return self.current_tasks[0]

    def peek_pending(self):
        """
        Get the first pending task (sent but not received) without removing it

        Returns: a pending task

        """
        return self.pending_tasks[0]

    def current_size(self):
        """
        Get the size of current tasks

        Returns: the size of current tasks

        """
        return len(self.current_tasks)

    def pending_size(self):
        """
        Get the size of pending tasks

        Returns: the size of pending tasks

        """
        return len(self.pending_tasks)


class IndependentMassSpectrometer():
    """
    A class that represents (synchronous) mass spectrometry process.
    Independent here refers to how the intensity of each peak in a scan is
    independent of each other i.e. there's no ion supression effect.
    """
    MS_SCAN_ARRIVED = 'MsScanArrived'
    ACQUISITION_STREAM_OPENING = 'AcquisitionStreamOpening'
    ACQUISITION_STREAM_CLOSED = 'AcquisitionStreamClosing'
    STATE_CHANGED = 'StateChanged'

    def __init__(self, ionisation_mode, chemicals, mz_noise=None,
                 intensity_noise=None, spike_noise=None,
                 isolation_transition_window='rectangular',
                 isolation_transition_window_params=None,
                 scan_duration=DEFAULT_SCAN_TIME_DICT, task_manager=None):
        """
        Creates a mass spec object.

        Args:
            ionisation_mode: POSITIVE or NEGATIVE
            chemicals: a list of Chemical objects in the dataset
            mz_noise: noise to apply to m/z values of generated peaks.
                      Should be an instance of [vimms.Noise.NoPeakNoise][] or
                      others that inherit from it.
            intensity_noise: noise to apply to intensity values of generated peaks.
                             Should be an instance of [vimms.Noise.NoPeakNoise][] or
                             others that inherit from it.
            spike_noise: spike noise in the generated spectra. Should be either None, or
                         set an instance of [vimms.Noise.UniformSpikeNoise][] if needed.
            isolation_transition_window: transition window for isolating peaks
            isolation_transition_window_params: parameters for isolation
            scan_duration: a dictionary of scan time for each MS level, or
                           an instance of [vimms.ChemicalSamplers.ScanTimeSampler][].
            task_manager: an instance of [vimms.MassSpec.TaskManager] to manage tasks, or None.
        """

        # current scan index and internal time
        self.idx = INITIAL_SCAN_ID  # same as the real mass spec
        self.time = 0

        # current task queue
        self.task_manager = task_manager if task_manager is not None \
            else TaskManager()
        self.environment = None

        self.events = Events((self.MS_SCAN_ARRIVED,
                              self.ACQUISITION_STREAM_OPENING,
                              self.ACQUISITION_STREAM_CLOSED,
                              self.STATE_CHANGED,))
        self.event_dict = {
            self.MS_SCAN_ARRIVED: self.events.MsScanArrived,
            self.ACQUISITION_STREAM_OPENING: self.events.AcquisitionStreamOpening,  # noqa
            self.ACQUISITION_STREAM_CLOSED: self.events.AcquisitionStreamClosing,  # noqa
            self.STATE_CHANGED: self.events.StateChanged
        }

        # the list of all chemicals in the dataset
        self.chemicals = chemicals
        self.ionisation_mode = ionisation_mode

        # stores the chromatograms start and end rt for quick retrieval
        chem_rts = np.array([chem.rt for chem in self.chemicals])
        self.chrom_min_rts = np.array(
            [chem.chromatogram.min_rt for chem in self.chemicals]) + chem_rts
        self.chrom_max_rts = np.array(
            [chem.chromatogram.max_rt for chem in self.chemicals]) + chem_rts

        # whether to add noise to the generated peaks, the default is no noise
        self.mz_noise = mz_noise
        self.intensity_noise = intensity_noise
        if self.mz_noise is None:
            self.mz_noise = NoPeakNoise()
        if self.intensity_noise is None:
            self.intensity_noise = NoPeakNoise()
        self.spike_noise = spike_noise

        self.fragmentation_events = []  # which chemicals produce which peaks

        self.isolation_transition_window = isolation_transition_window
        self.isolation_transition_window_params = isolation_transition_window_params  # noqa

        self.scan_duration_dict = scan_duration

    ###########################################################################
    # Public methods
    ###########################################################################

    def set_environment(self, env):
        self.environment = env

    def step(self, params=None, call_controller=True):
        """
        Performs one step of a mass spectrometry process

        Args:
            params: initial set of tasks from the controller
            call_controller: whether to actually call the controller or not

        Returns: a newly generated scan

        """
        # if no params passed in, then try to get one from the queue
        if params is None:
            # Get the next set of params from the outgoing tasks.
            # This could return an empty list if there's nothing there.
            params_list = self.get_params()
        else:
            logger.debug('Got an initial task from controller')
            params_list = [params]  # initial param passed from the controller

        # Send params away. In the simulated case, no sending actually occurs,
        # instead we just track these params we've sent by adding them to
        # self.environment.pending_tasks
        if len(params_list) > 0:
            logger.debug('new tasks ready to send = %d' % len(params_list))
            self.send_params(params_list)

        # pick up the last param that has been sent and generate a new scan
        new_scan = None
        if self.task_manager.pending_size() > 0:
            params = self.task_manager.pop_pending()

            # Make scan using the param that has been sent
            # Note that in the real mass spec, the scan generation
            # likely won't happen immediately after the param is sent.
            new_scan = self._get_scan(self.time, params)

            # dispatch the generated scan to controller
            if call_controller:
                self.dispatch_scan(new_scan)
        return new_scan

    def dispatch_scan(self, scan):
        """
        Notify the controller that a new scan has been generated
        at this point, the MS_SCAN_ARRIVED event handler in the
        controller is called and the processing queue will be updated
        with new sets of scan parameters to do.

        Args:
            scan: a newly generated scan.

        Returns: None

        """
        self.fire_event(self.MS_SCAN_ARRIVED, scan)

        # sample scan duration and increase internal time
        try:
            next_scan_param = self.task_manager.peek_current()
        except IndexError:
            next_scan_param = None

        current_level = scan.ms_level
        current_scan_duration = self._increase_time(current_level,
                                                    next_scan_param)
        scan.scan_duration = current_scan_duration

    def get_params(self):
        """
        Retrieves a new set of scan parameters from the processing queue

        Returns: A new set of scan parameters from the queue if available,
                 otherwise it returns nothing (default scan set in actual MS)

        """
        params_list = self.task_manager.to_send()
        logger.debug('Selected %d tasks to send' % (len(params_list)))
        return params_list

    def send_params(self, params_list):
        """
        Send parameters to the instrument.

        In the real IAPI mass spec, we would send these params
        to the instrument, but here we just store them in the list of pending tasks
        to be processed later

        Args:
            params_list: the list of scan parameters to send.

        Returns: None

        """
        self.task_manager.add_pending(params_list)
        logger.debug('Successfully sent %d tasks' % (len(params_list)))

    def fire_event(self, event_name, arg=None):
        """
        Simulates sending an event

        Args:
            event_name: the event name
            arg: the event parameter

        Returns: None

        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')

        # pretend to fire the event
        # actually here we just runs the event handler method directly
        e = self.event_dict[event_name]
        if arg is not None:
            e(arg)
        else:
            e()

    def register_event(self, event_name, handler):
        """
        Register event handler

        Args:
            event_name: the event name
            handler: the event handler

        Returns: None

        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e += handler  # register a new event handler for e

    def clear_events(self):
        """
        Clear event handlers

        Returns: None

        """
        for key in self.event_dict:
            self.clear_event(key)

    def clear_event(self, event_name):
        """
        Clears event handler for a given event name

        Args:
            event_name: the event name

        Returns: None

        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e.targets = []

    def close(self):
        """
        Close this mass spec

        Returns: None

        """
        logger.debug('Unregistering event handlers')
        self.clear_events()

    ###########################################################################
    # Private methods
    ###########################################################################

    def _increase_time(self, current_level, next_scan_param):
        """
        Look into the queue, find out what the next scan ms_level is, and
        compute the scan duration.
        Only applicable for simulated mass spec, since the real mass spec
        can generate its own scan duration.

        Args:
            current_level:  the current MS level
            next_scan_param: the next scan parameter in the queue

        Returns: the scan duration of the current scan

        """
        self.idx += 1

        # sample scan duration from dictionary
        if type(self.scan_duration_dict) is dict:
            val = self.scan_duration_dict[current_level]
            current_scan_duration = val() if callable(
                val) else val  # is it a function, or a value?

        else:  # assume it's an object
            scan_sampler = self.scan_duration_dict

            # if queue is empty, the next one is an MS1 scan by default
            next_level = next_scan_param.get(
                ScanParameters.MS_LEVEL) if next_scan_param is not None else 1

            # pass both current and next MS level when sampling scan duration
            current_scan_duration = scan_sampler.sample(current_level,
                                                        next_level)

        self.time += current_scan_duration
        logger.debug(
            'scan_duration=%f time %f' % (current_scan_duration, self.time))
        return current_scan_duration

    ###########################################################################
    # Scan generation methods
    ###########################################################################

    # flake8: noqa: C901
    def _get_scan(self, scan_time, params):
        """
        Constructs a scan at a particular timepoint

        Args:
            scan_time: the timepoint
            params: a mass spectrometry scan at that time

        Returns:

        """
        frag = None

        min_measurement_mz = params.get(ScanParameters.FIRST_MASS)
        max_measurement_mz = params.get(ScanParameters.LAST_MASS)

        ms1_source_collision_energy = params.get(
            ScanParameters.SOURCE_CID_ENERGY)

        scan_mzs = []  # all the mzs values in this scan
        scan_intensities = []  # all the intensity values in this scan
        ms_level = params.get(ScanParameters.MS_LEVEL)
        if ms_level == 1:  # if ms1 then we scan the whole range of m/z
            isolation_windows = params.get(ScanParameters.ISOLATION_WINDOWS)
            if isolation_windows is None:
                isolation_windows = [
                    [(min_measurement_mz, max_measurement_mz)]]
            if not isolation_windows[0][0][0] == min_measurement_mz or not \
                    isolation_windows[0][0][
                        1] == max_measurement_mz:
                logger.warning(
                    "MS1 scan: MS1 isolation window and measurement "
                    "mz range mismatch")

        else:
            # if ms2 then we check if the isolation window parameter
            # is specified
            # isolation_windows = params.get(ScanParameters.ISOLATION_WINDOWS)
            # if not then we compute from the precursor mz and isolation width
            # if isolation_windows is None:
            isolation_windows = params.compute_isolation_windows()

        # if the scan id is specified in the params, use it
        # otherwise use the one that has been incremented from the previous one
        scan_id = params.get(ScanParameters.SCAN_ID)
        if scan_id is None:
            scan_id = self.idx

        # for all chemicals that come out from the column
        # coupled to the mass spec
        idx = self._get_chem_indices(scan_time)

        # the following is to ensure we generate fragment data when
        # we have a collision energe >0
        use_ms_level = ms_level
        if ms_level == 1 and ms1_source_collision_energy > 0:
            use_ms_level = 2

        frag_events = []
        for i in idx:
            chemical = self.chemicals[i]
            # mzs is a list of (mz, intensity) for the different
            # adduct/isotopes combinations of a chemical
            mzs = self._get_all_mz_peaks(chemical, scan_time, use_ms_level,
                                         isolation_windows)
            peaks = []
            peaks_ms1_intensities = []
            peaks_which_isotopes = []
            peaks_which_adducts = []
            if mzs is not None:
                chem_mzs = []
                chem_intensities = []
                for items in mzs:
                    peak_mz, peak_intensity, peak_ms1_int, peak_isotope, \
                    peak_adduct = items
                    if (min_measurement_mz <= peak_mz <= max_measurement_mz) \
                            and peak_intensity > 0:
                        chem_mzs.append(peak_mz)
                        chem_intensities.append(peak_intensity)
                        p = Peak(peak_mz, scan_time, peak_intensity,
                                 use_ms_level)
                        peaks.append(p)
                        peaks_ms1_intensities.append(peak_ms1_int)
                        peaks_which_isotopes.append(peak_isotope)
                        peaks_which_adducts.append(peak_adduct)
                scan_mzs.extend(chem_mzs)
                scan_intensities.extend(chem_intensities)
            # for benchmarking purpose
            if len(peaks) > 0:
                frag = ScanEvent(chemical, scan_time, use_ms_level, peaks,
                                 scan_id,
                                 parents_intensity=peaks_ms1_intensities,
                                 parent_adduct=peaks_which_adducts,
                                 parent_isotope=peaks_which_isotopes,
                                 precursor_mz=params.get(
                                     ScanParameters.PRECURSOR_MZ),
                                 isolation_window=isolation_windows,
                                 scan_params=params)
                frag_events.append(frag)
        self.fragmentation_events.extend(frag_events)

        # FIXME: this here means spike noise is not a chemical and cannot be fragmented
        # (even though it appears in the scan). Is this the best thing to do?
        if self.spike_noise is not None:
            spike_mzs, spike_intensities = self.spike_noise.sample(
                min_measurement_mz, max_measurement_mz)
            scan_mzs.extend(spike_mzs)
            scan_intensities.extend(spike_intensities)

        scan_mzs = np.array(scan_mzs)
        scan_intensities = np.array(scan_intensities)

        if len(frag_events) == 0: # for compatibility with old codes
            frag_events = None

        sc = Scan(scan_id, scan_mzs, scan_intensities, ms_level, scan_time,
                  scan_duration=None, scan_params=params,
                  fragevent=frag_events)

        # Note: at this point, the scan duration is not set yet because
        # we don't know what the next scan is going to be
        # We will set it later in the get_next_scan() method after
        # we've notified the controller that this scan is produced.
        return sc

    def _get_chem_indices(self, query_rt):
        rtmin_check = self.chrom_min_rts <= query_rt
        rtmax_check = query_rt <= self.chrom_max_rts
        idx = np.nonzero(rtmin_check & rtmax_check)[0]
        return idx

    def _get_all_mz_peaks(self, chemical, query_rt, ms_level,
                          isolation_windows):
        # check if the chemical RT matches the current query RT
        if not self._rt_match(chemical, query_rt):
            return None

        # if yes, then gather all the peaks from all the isotopes and
        # adduct of this chemical
        mz_peaks = []
        for which_isotope in range(len(chemical.isotopes)):
            for which_adduct in range(len(self._get_adducts(chemical))):
                peaks = self._get_mz_peaks(chemical, query_rt, ms_level,
                                           isolation_windows, which_isotope,
                                           which_adduct)
                mz_peaks.extend(peaks)

        # if no peaks generated, then just return None
        if len(mz_peaks) == 0:
            return None
        # apply noise if any
        noisy_mz_peaks = []
        for i in range(len(mz_peaks)):
            original_mz = mz_peaks[i][0]
            original_intensity = mz_peaks[i][1]
            noisy_mz = self.mz_noise.get(original_mz, ms_level)
            noisy_intensity = self.intensity_noise.get(original_intensity,
                                                       ms_level)
            noisy_mz_peaks.append((noisy_mz, noisy_intensity, mz_peaks[i][2],
                                   mz_peaks[i][3], mz_peaks[i][4]))
        return noisy_mz_peaks

    def _get_mz_peaks(self, chemical, query_rt, ms_level, isolation_windows,
                      which_isotope, which_adduct):
        # EXAMPLE OF USE OF DEFINITION: if we wants to do an ms2 scan on a
        # chemical. we would first have ms_level=2 and the chemicals
        # ms_level =1. So we would go to the "else". We then check the ms1
        # window matched. It then would loop through
        # the children who have ms_level = 2. So we then go to second elif
        # and return the mz and intensity of each ms2 fragment
        mz_peaks = []
        if ms_level == 1 and chemical.ms_level == 1:  # fragment ms1 peaks
            # returns ms1 peaks if chemical is has ms_level = 1 and
            # scan is an ms1 scan
            if not (which_isotope > 0 and which_adduct > 0):
                # rechecks isolations window if not monoisotopic and
                # "M + H" adduct
                if self._isolation_match(chemical, query_rt,
                                         isolation_windows[0], which_isotope,
                                         which_adduct):
                    intensity = self._get_intensity(
                        chemical, query_rt, which_isotope, which_adduct)
                    mz = self._get_mz(chemical, query_rt, which_isotope,
                                      which_adduct)
                    mz_peaks.extend([(mz, intensity, None, None, None)])
        elif ms_level == chemical.ms_level:
            # returns ms2 fragments if chemical and scan are both ms2,
            # returns ms3 fragments if chemical and scan are both ms3, etc, etc
            ms1_intensity = self._get_intensity(chemical.parent, query_rt,
                                                which_isotope, which_adduct)
            intensity = self._get_intensity(chemical, query_rt, which_isotope,
                                            which_adduct)
            mz = self._get_mz(chemical, query_rt, which_isotope, which_adduct)
            if self.isolation_transition_window == 'gaussian':
                parent_mz = self._get_mz(chemical.parent, query_rt,
                                         which_isotope, which_adduct)
                norm_dist = scipy.stats.norm(
                    0, self.isolation_transition_window_params[0])
                scale_factor = norm_dist.pdf(
                    parent_mz - sum(isolation_windows[ms_level - 2][0]) / 2)
                scale_factor /= scipy.stats.norm(
                    0, self.isolation_transition_window_params[0]).pdf(0)
                intensity *= scale_factor
            return_values = [(mz, intensity, ms1_intensity, which_isotope,
                              which_adduct)]
            return return_values
            # return extra information here for logging
            # TODO: Potential improve how the isotope spectra are generated
        else:
            # check isolation window for ms2+ scans, queries children
            # if isolation windows ok
            isolated = self._isolation_match(
                chemical, query_rt, isolation_windows[chemical.ms_level - 1],
                which_isotope, which_adduct)
            if isolated and chemical.children is not None:
                for i in range(len(chemical.children)):
                    mz_peaks.extend(
                        self._get_mz_peaks(chemical.children[i], query_rt,
                                           ms_level, isolation_windows,
                                           which_isotope, which_adduct))
            else:
                return []
        return mz_peaks

    def _get_adducts(self, chemical):
        if chemical.ms_level == 1:
            if self.ionisation_mode in chemical.adducts:
                return chemical.adducts[self.ionisation_mode]
            else:
                return []
        else:
            return self._get_adducts(chemical.parent)

    def _rt_match(self, chemical, query_rt):
        return chemical.ms_level != 1 or chemical.chromatogram._rt_match(
            query_rt - chemical.rt)

    def _get_intensity(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            intensity = chemical.isotopes[which_isotope][1] * \
                        self._get_adducts(chemical)[which_adduct][1] * \
                        chemical.max_intensity
            return intensity * chemical.chromatogram.get_relative_intensity(
                query_rt - chemical.rt)
        else:
            prop = chemical.parent_mass_prop
            if isinstance(prop, np.ndarray):
                prop = prop[0]
            intensity = self._get_intensity(
                chemical.parent, query_rt, which_isotope, which_adduct)
            intensity = intensity * prop * chemical.prop_ms2_mass
            return intensity

    def _get_mz(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            return (adduct_transformation(chemical.isotopes[which_isotope][0],
                                          self._get_adducts(chemical)[
                                              which_adduct][0]) +
                    chemical.chromatogram.get_relative_mz(
                        query_rt - chemical.rt))
        else:
            ms1_parent = chemical
            while ms1_parent.ms_level != 1:
                ms1_parent = chemical.parent
            isotope_transformation = ms1_parent.isotopes[which_isotope][0] - \
                                     ms1_parent.isotopes[0][0]
            # TODO: Needs improving
            return (adduct_transformation(chemical.isotopes[0][0],
                                          self._get_adducts(chemical)[
                                              which_adduct][
                                              0]) + isotope_transformation)

    def _isolation_match(self, chemical, query_rt, isolation_windows,
                         which_isotope, which_adduct):
        # assumes list is formated like:
        # [(min_1,max_1),(min_2,max_2),...]
        for window in isolation_windows:
            if window[0] < self._get_mz(chemical, query_rt, which_isotope,
                                        which_adduct) <= window[1]:
                return True
        return False
