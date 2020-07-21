from pathlib import Path

from loguru import logger
from tqdm import tqdm

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW, DEFAULT_ISOLATION_WIDTH, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, POSITIVE, DEFAULT_MSN_SCAN_WINDOW
from vimms.Controller import TopNController, PurityController, FixedScheduleController
from vimms.MassSpec import ScanParameters, IndependentMassSpectrometer, Precursor
from vimms.MzmlWriter import MzmlWriter


class Environment(object):
    def __init__(self, mass_spec, controller, min_time, max_time, progress_bar=True, out_dir=None, out_file=None):
        """
        Initialises a synchronous environment to run the mass spec and controller
        :param mass_spec: An instance of Mass Spec object
        :param controller: An instance of Controller object
        :param min_time: start time
        :param max_time: end time
        :param progress_bar: True if a progress bar is to be shown
        """
        self.mass_spec = mass_spec
        self.controller = controller
        self.min_time = min_time
        self.max_time = max_time
        self.progress_bar = progress_bar
        self.out_dir = out_dir
        self.out_file = out_file
        self.pending_tasks = []

    def run(self):
        """
        Runs the mass spec and controller
        :return: None
        """
        # reset mass spec and set some initial values for each run
        self.mass_spec.reset()
        self.controller.reset()
        self._set_initial_values()

        # register event handlers from the controller
        self.mass_spec.register_event(IndependentMassSpectrometer.MS_SCAN_ARRIVED, self.add_scan)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING,
                                      self.handle_acquisition_open)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED,
                                      self.handle_acquisition_closing)
        self.mass_spec.register_event(IndependentMassSpectrometer.STATE_CHANGED,
                                      self.handle_state_changed)

        # run mass spec
        bar = tqdm(total=self.max_time - self.min_time, initial=0) if self.progress_bar else None
        self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING)

        # set this to add initial MS1 scan
        initial_scan = True
        try:
            # perform one step of mass spec up to max_time
            while self.mass_spec.time < self.max_time:
                # controller._process_scan() is called here immediately when a scan is produced within a step
                scan = self.mass_spec.step(initial_scan)
                if initial_scan:
                    # no longer initial scan
                    initial_scan = False

                # update controller internal states AFTER a scan has been generated and handled
                self.controller.update_state_after_scan(scan)
                # increment progress bar
                self._update_progress_bar(bar, scan)
        except Exception as e:
            raise e
        finally:
            self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED)
            self.mass_spec.close()
            self.close_progress_bar(bar)
        self.write_mzML(self.out_dir, self.out_file)

    def handle_acquisition_open(self):
        logger.debug('Acquisition open')

    def handle_acquisition_closing(self):
        logger.debug('Acquisition closing')

    def handle_state_changed(self, state):
        logger.debug('State changed!')

    def _update_progress_bar(self, pbar, scan):
        """
        Updates progress bar based on elapsed time
        :param elapsed: Elapsed time to increment the progress bar
        :param pbar: progress bar object
        :param scan: the newly generated scan
        :return: None
        """
        if pbar is not None:
            N, DEW = self._get_N_DEW(self.mass_spec.time)
            if N is not None and DEW is not None:
                msg = '(%.3fs) ms_level=%d N=%d DEW=%d' % (self.mass_spec.time, scan.ms_level, N, DEW)
            else:
                msg = '(%.3fs) ms_level=%d' % (self.mass_spec.time, scan.ms_level)
            if pbar.n + scan.scan_duration < pbar.total:
                pbar.update(scan.scan_duration)
            pbar.set_description(msg)

    def close_progress_bar(self, bar):
        if bar is not None:
            try:
                bar.close()
            except Exception as e:
                logger.warning('Failed to close progress bar: %s' % str(e))
                pass

    def add_scan(self, scan):
        """
        Adds a newly generated scan. In this case, immediately we process it in the controller without saving the scan.
        :param scan: A newly generated scan
        :return: None
        """
        # check the status of the last block of pending tasks we sent to determine if their corresponding scans
        # have actually been performed by the mass spec
        completed_task = scan.scan_params
        if completed_task is not None:  # should not be none for custom scans that we sent
            self.pending_tasks = [t for t in self.pending_tasks if t != completed_task]

        # handle the scan immediately by passing it to the controller,
        # and get new set of tasks from the controller
        outgoing_queue_size = len(self.mass_spec.get_processing_queue())
        pending_tasks_size = len(self.pending_tasks)
        tasks = self.controller.handle_scan(scan, outgoing_queue_size, pending_tasks_size)
        self.pending_tasks.extend(tasks)  # track pending tasks here

        # immediately push new tasks to mass spec queue
        self.add_tasks(tasks)
        return len(tasks)

    def add_tasks(self, outgoing_scan_params):
        """
        Stores new tasks from the controller. In this case, immediately we pass the new tasks to the mass spec.
        :param outgoing_scan_params: new tasks to send
        :return: None
        """
        for new_task in outgoing_scan_params:
            self.mass_spec.add_to_processing_queue(new_task)

    def write_mzML(self, out_dir, out_file):
        """
        Writes mzML to output file
        :param out_dir: output directory
        :param out_file: output filename
        :return: None
        """
        if out_file is None:  # if no filename provided, just quits
            return
        else:
            if out_dir is None:  # no out_dir, use only out_file
                mzml_filename = Path(out_file)
            else:  # both our_dir and out_file are provided
                mzml_filename = Path(out_dir, out_file)

        logger.debug('Writing mzML file to %s' % mzml_filename)
        try:
            precursor_information = self.controller.precursor_information
        except AttributeError:
            precursor_information = None
        writer = MzmlWriter('my_analysis', self.controller.scans, precursor_information)
        writer.write_mzML(mzml_filename)
        logger.debug('mzML file successfully written!')

    def _set_initial_values(self):
        """
        Sets initial environment, mass spec start time, default scan parameters and other values
        :return: None
        """
        self.controller.set_environment(self)
        self.mass_spec.set_environment(self)
        self.mass_spec.time = self.min_time

        N, DEW = self._get_N_DEW(self.mass_spec.time)
        if N is not None:
            self.mass_spec.current_N = N
        if DEW is not None:
            self.mass_spec.current_DEW = DEW

    def get_default_scan_params(self, agc_target=DEFAULT_MS1_AGC_TARGET, max_it=DEFAULT_MS1_MAXIT,
                 collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION):
        """
        Gets the default method scan parameters. Now it's set to do MS1 scan only.
        :return: the default scan parameters
        """
        default_scan_params = ScanParameters()
        default_scan_params.set(ScanParameters.MS_LEVEL, 1)
        default_scan_params.set(ScanParameters.ISOLATION_WINDOWS, [[DEFAULT_MS1_SCAN_WINDOW]])
        default_scan_params.set(ScanParameters.ISOLATION_WIDTH, DEFAULT_ISOLATION_WIDTH)
        default_scan_params.set(ScanParameters.COLLISION_ENERGY, collision_energy)
        default_scan_params.set(ScanParameters.ORBITRAP_RESOLUTION, orbitrap_resolution)
        default_scan_params.set(ScanParameters.AGC_TARGET, agc_target)
        default_scan_params.set(ScanParameters.MAX_IT, max_it)
        default_scan_params.set(ScanParameters.POLARITY, self.mass_spec.ionisation_mode)
        default_scan_params.set(ScanParameters.FIRST_MASS, DEFAULT_MS1_SCAN_WINDOW[0])
        default_scan_params.set(ScanParameters.LAST_MASS, DEFAULT_MS1_SCAN_WINDOW[1])
        return default_scan_params

    def get_dda_scan_param(self, mz, intensity, precursor_scan_id, isolation_width, mz_tol, rt_tol,
                           agc_target=DEFAULT_MS2_AGC_TARGET, max_it=DEFAULT_MS2_MAXIT,
                           collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                           orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        dda_scan_params = ScanParameters()
        dda_scan_params.set(ScanParameters.MS_LEVEL, 2)

        # create precursor object, assume it's all singly charged
        precursor_charge = +1 if (self.mass_spec.ionisation_mode == POSITIVE) else -1
        precursor = Precursor(precursor_mz=mz, precursor_intensity=intensity,
                              precursor_charge=precursor_charge, precursor_scan_id=precursor_scan_id)
        dda_scan_params.set(ScanParameters.PRECURSOR_MZ, precursor)

        # set the full-width isolation width, in Da
        dda_scan_params.set(ScanParameters.ISOLATION_WIDTH, isolation_width)

        # define dynamic exclusion parameters
        dda_scan_params.set(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL, mz_tol)
        dda_scan_params.set(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL, rt_tol)

        # define other fragmentation parameters
        dda_scan_params.set(ScanParameters.COLLISION_ENERGY, collision_energy)
        dda_scan_params.set(ScanParameters.ORBITRAP_RESOLUTION, orbitrap_resolution)
        dda_scan_params.set(ScanParameters.AGC_TARGET, agc_target)
        dda_scan_params.set(ScanParameters.MAX_IT, max_it)
        dda_scan_params.set(ScanParameters.POLARITY, self.mass_spec.ionisation_mode)
        dda_scan_params.set(ScanParameters.FIRST_MASS, DEFAULT_MSN_SCAN_WINDOW[0])

        # dynamically scale the upper mass
        charge = 1
        wiggle_room = 1.1
        last_mass = precursor.precursor_mz * charge * wiggle_room
        dda_scan_params.set(ScanParameters.LAST_MASS, last_mass)
        return dda_scan_params

    def _get_N_DEW(self, time):
        """
        Gets the current N and DEW depending on which controller type it is
        :return: The current N and DEW values, None otherwise
        """
        if isinstance(self.controller, PurityController):
            current_N, current_rt_tol, idx = self.controller._get_current_N_DEW(time)
            return current_N, current_rt_tol
        elif isinstance(self.controller, TopNController):
            return self.controller.N, self.controller.rt_tol
        elif isinstance(self.controller, FixedScheduleController):
            return self.controller.N, self.controller.rt_tol
        else:
            return None, None


