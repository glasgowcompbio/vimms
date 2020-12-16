from pathlib import Path

from loguru import logger
from tqdm import tqdm

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW, DEFAULT_ISOLATION_WIDTH, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, POSITIVE, DEFAULT_MSN_SCAN_WINDOW, \
    DEFAULT_MS1_MASS_ANALYSER, DEFAULT_MS1_ACTIVATION_TYPE, DEFAULT_MS1_ISOLATION_MODE, DEFAULT_MS2_MASS_ANALYSER, \
    DEFAULT_MS2_ISOLATION_MODE, DEFAULT_SOURCE_CID_ENERGY, ScanParameters, Precursor
from vimms.Controller import TopNController, PurityController
from vimms.MassSpec import IndependentMassSpectrometer
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
        self.bar = tqdm(total=self.max_time - self.min_time, initial=0) if self.progress_bar else None

    def run(self):
        """
        Runs the mass spec and controller
        :return: None
        """
        # set some initial values for each run
        self._set_initial_values()

        # register event handlers from the controller
        self.mass_spec.register_event(IndependentMassSpectrometer.MS_SCAN_ARRIVED, self.add_scan)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING,
                                      self.handle_acquisition_open)
        self.mass_spec.register_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED,
                                      self.handle_acquisition_closing)
        self.mass_spec.register_event(IndependentMassSpectrometer.STATE_CHANGED,
                                      self.handle_state_changed)

        # initial scan should be generated here when the acquisition opens
        self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING)

        # main loop to the simulate scan generation process of the mass spec
        try:
            # perform one step of mass spec up to max_time
            while self.mass_spec.time < self.max_time:
                # unless no more scan scheduled by the controller, then stop the simulated run
                scan = self._one_step()
                if scan is None:
                    break
        except Exception as e:
            raise e
        finally:
            self.mass_spec.fire_event(IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED)
            self.mass_spec.close()
            self.close_progress_bar()
        self.write_mzML(self.out_dir, self.out_file)

    def _one_step(self, params=None):
        # controller._process_scan() is called here immediately when a scan is produced within a step
        scan = self.mass_spec.step(params=params)
        if scan is not None:
            # update controller internal states AFTER a scan has been generated and handled
            self.controller.update_state_after_scan(scan)
            # increment progress bar
            self._update_progress_bar(scan)
        return scan

    def handle_acquisition_open(self):
        logger.debug('Acquisition open')
        # send the initial custom scan to start the custom scan generation process
        params = self.get_initial_scan_params()
        self._one_step(params=params)

    def handle_acquisition_closing(self):
        logger.debug('Acquisition closing')

    def handle_state_changed(self, state):
        logger.debug('State changed!')

    def _update_progress_bar(self, scan):
        """
        Updates progress bar based on elapsed time
        :param scan: the newly generated scan
        :return: None
        """
        if self.bar is not None and scan.scan_duration is not None:
            N, DEW = self._get_N_DEW(self.mass_spec.time)
            if N is not None and DEW is not None:
                msg = '(%.3fs) ms_level=%d N=%d DEW=%d' % (self.mass_spec.time, scan.ms_level, N, DEW)
            else:
                msg = '(%.3fs) ms_level=%d' % (self.mass_spec.time, scan.ms_level)
            if self.bar.n + scan.scan_duration < self.bar.total:
                self.bar.update(scan.scan_duration)
            self.bar.set_description(msg)

    def close_progress_bar(self):
        if self.bar is not None:
            try:
                self.bar.close()
            except Exception as e:
                logger.warning('Failed to close progress bar: %s' % str(e))
                pass

    def add_scan(self, scan):
        """
        Adds a newly generated scan. In this case, immediately we process it in the controller without saving the scan.
        :param scan: A newly generated scan
        :return: None
        """
        logger.debug('Time %f Received %s' % (scan.rt, scan))

        # check the status of the last block of pending tasks we sent to determine if their corresponding scans
        # have actually been performed by the mass spec
        completed_task = scan.scan_params
        if completed_task is not None:  # should not be none for custom scans that we sent
            self.mass_spec.task_manager.remove_pending(completed_task)

        # handle the scan immediately by passing it to the controller,
        # and get new set of tasks from the controller
        current_size = self.mass_spec.task_manager.current_size()
        pending_size = self.mass_spec.task_manager.pending_size()
        tasks = self.controller.handle_scan(scan, current_size, pending_size)

        # immediately push newly generated tasks to mass spec queue
        self.mass_spec.task_manager.add_current(tasks)

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
        writer = MzmlWriter('my_analysis', self.controller.scans)
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

        # add the initial tasks from the controller to the mass spec task manager
        self.mass_spec.task_manager.add_current(self.controller.get_initial_tasks())

        N, DEW = self._get_N_DEW(self.mass_spec.time)
        if N is not None:
            self.mass_spec.current_N = N
        if DEW is not None:
            self.mass_spec.current_DEW = DEW

    def get_initial_scan_params(self):
        return self.controller.get_initial_scan_params()

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
        else:
            return None, None
