import os
from pathlib import Path

import pylab as plt
from loguru import logger
from tqdm.auto import tqdm

from vimms.Common import save_obj
from vimms.Evaluation import EvaluationData
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.MzmlWriter import MzmlWriter


class Environment():
    def __init__(self, mass_spec, controller, min_time, max_time,
                 progress_bar=True, out_dir=None, out_file=None,
                 save_eval=False, check_exists=False):
        """
        Initialises a synchronous environment to run the mass spec and
        controller

        Args:
            mass_spec: An instance of [vimms.MassSpec.IndependentMassSpectrometer] object
            controller: An instance of [vimms.Controller.base.Controller] object
            min_time: start time for simulation
            max_time: end time for simulation
            progress_bar: True if a progress bar is to be shown, False otherwise
            out_dir: output directory to write mzML to
            out_file: output mzML file
            save_eval: whether to save evaluation information
        """
        self.mass_spec = mass_spec
        self.controller = controller
        self.min_time = min_time
        self.max_time = max_time
        self.progress_bar = progress_bar
        self.out_dir = out_dir
        self.out_file = out_file
        self.pending_tasks = []
        self.bar = tqdm(total=self.max_time - self.min_time,
                        initial=0) if self.progress_bar else None
        self.save_eval = save_eval
        self.check_exists = check_exists

    def run(self):
        """
        Runs the mass spec and controller

        Returns: None

        """
        mzml_filename = self._get_out_file(self.out_dir, self.out_file)
        if self.check_exists and mzml_filename is not None and mzml_filename.is_file():
            logger.warning('Already exists %s' % mzml_filename)
            return

        # set some initial values for each run
        self._set_initial_values()

        # register event handlers from the controller
        self.mass_spec.register_event(
            IndependentMassSpectrometer.MS_SCAN_ARRIVED, self.add_scan)
        self.mass_spec.register_event(
            IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING,
            self.handle_acquisition_open)
        self.mass_spec.register_event(
            IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED,
            self.handle_acquisition_closing)
        self.mass_spec.register_event(
            IndependentMassSpectrometer.STATE_CHANGED,
            self.handle_state_changed)

        # initial scan should be generated here when the acquisition opens
        self.mass_spec.fire_event(
            IndependentMassSpectrometer.ACQUISITION_STREAM_OPENING)

        # main loop to the simulate scan generation process of the mass spec
        try:
            # perform one step of mass spec up to max_time
            while self.mass_spec.time < self.max_time:
                # unless no more scan scheduled by the controller,
                # then stop the simulated run
                scan = self._one_step()
                if scan is None:
                    break
        except Exception as e:
            raise e
        finally:
            self.mass_spec.fire_event(
                IndependentMassSpectrometer.ACQUISITION_STREAM_CLOSED)
            self.mass_spec.close()
            self.close_progress_bar()
        self.write_mzML(self.out_dir, self.out_file)
        if self.save_eval:
            self.write_eval_data(self.out_dir, self.out_file)

    def _one_step(self, params=None):
        """
        Simulates one step of the mass spectrometry process, and also
        calling the controller on the result.

        Args:
            params: parameters to the mass spec

        Returns: a newly generated scan

        """
        # controller._process_scan() is called here immediately when
        # a scan is produced within a step
        scan = self.mass_spec.step(params=params)
        if scan is not None:
            # update controller internal states AFTER a scan has been
            # generated and handled
            self.controller.update_state_after_scan(scan)
            # increment progress bar
            self._update_progress_bar(scan)
        return scan

    def handle_acquisition_open(self):
        """
        Handle acquisition open event
        Returns: None

        """
        logger.debug('Acquisition open')
        # send the initial custom scan to start the custom scan
        # generation process
        params = self.get_initial_scan_params()
        self._one_step(params=params)

    def handle_acquisition_closing(self):
        """
        Handle acquisition close event

        Returns: None

        """
        logger.debug('Acquisition closing')
        self.controller.after_injection_cleanup()

    def handle_state_changed(self, state):
        """
        Handle event for any state change on the mass spec

        Args:
            state: a new state, could be any value

        Returns: None

        """
        logger.debug('State changed!')

    def _update_progress_bar(self, scan):
        """
        Updates progress bar based on elapsed time

        Args:
            scan: the newly generated scan

        Returns: None

        """
        if self.bar is not None and scan.scan_duration is not None:
            msg = '(%.3fs) ms_level=%d' % (self.mass_spec.time, scan.ms_level)
            if self.bar.n + scan.scan_duration < self.bar.total:
                self.bar.update(scan.scan_duration)
            self.bar.set_description(msg)

    def close_progress_bar(self):
        """
        Close the progress bar, typically when acquisition has finished

        Returns: None

        """
        if self.bar is not None:
            try:
                self.bar.close()
                self.bar = None
            except Exception as e:
                logger.warning('Failed to close progress bar: %s' % str(e))
                pass

    def add_scan(self, scan):
        """
        Adds a newly generated scan. In this case, immediately we process it
        in the controller without saving the scan.

        Args:
            scan: A newly generated scan

        Returns: None

        """
        logger.debug('Time %f Received %s' % (scan.rt, scan))

        # check the status of the last block of pending tasks we sent to
        # determine if their corresponding scans have actually been performed
        # by the mass spec
        completed_task = scan.scan_params
        if completed_task is not None:
            # should not be none for custom scans that we sent
            self.mass_spec.task_manager.remove_pending(completed_task)

        # handle the scan immediately by passing it to the controller,
        # and get new set of tasks from the controller
        current_size = self.mass_spec.task_manager.current_size()
        pending_size = self.mass_spec.task_manager.pending_size()
        tasks = self.controller.handle_scan(scan, current_size, pending_size)

        # immediately push newly generated tasks to mass spec queue
        self.mass_spec.task_manager.add_current(tasks)

    def _get_out_file(self, out_dir, out_file):
        """
        Generates the output mzML filename based on out_dir and out_file

        Args:
            out_dir: the output directory
            out_file: the output filename

        Returns: the combined out_dir and out_file

        """
        if out_file is None:  # if no filename provided, just quits
            return None
        else:
            if out_dir is None:  # no out_dir, use only out_file
                mzml_filename = Path(out_file)
            else:  # both our_dir and out_file are provided
                mzml_filename = Path(out_dir, out_file)
            return mzml_filename

    def write_mzML(self, out_dir, out_file):
        """
        Writes mzML to output file

        Args:
            out_dir: output directory
            out_file: output filename

        Returns: None

        """
        mzml_filename = self._get_out_file(out_dir, out_file)
        logger.debug('Writing mzML file to %s' % mzml_filename)
        if mzml_filename is not None:
            writer = MzmlWriter('my_analysis', self.controller.scans)
            writer.write_mzML(mzml_filename)
            logger.debug('mzML file successfully written!')

    def write_eval_data(self, out_dir, out_file):
        """
        Writes evaluation data to a pickle file in way that works in methods in Evaluation.py

        Args:
            out_dir: output directory
            out_file: output filename

        Returns: None

        """
        mzml_filename = self._get_out_file(out_dir, out_file)
        if mzml_filename is not None:
            eval_filename = os.path.splitext(mzml_filename)[0] + '.p' # replace .mzML with .p
            logger.debug('Writing evaluation data to %s' % eval_filename)
            eval_data = EvaluationData(self)
            save_obj(eval_data, eval_filename)

    def _set_initial_values(self):
        """
        Sets initial environment, mass spec start time, default
        scan parameters and other values

        Returns: None

        """
        self.controller.set_environment(self)
        self.mass_spec.set_environment(self)
        self.mass_spec.time = self.min_time

        # add the initial tasks from the controller to the mass spec
        # task manager
        self.mass_spec.task_manager.add_current(
            self.controller.get_initial_tasks())

    def get_initial_scan_params(self):
        """
        Get the initial scan parameters before acquisition is even started.
        Useful when we have controllers with pre-scheduled tasks.

        Returns: the list of initial scan parameters from the controller

        """
        return self.controller.get_initial_scan_params()

    def save(self, outname):
        """
        Save certain information from this environment.
        Currently only save scans, but we could save more.

        Args:
            outname: output file to save

        Returns: None

        """
        data_to_save = {
            'scans': self.controller.scans,
            # etc
        }
        save_obj(data_to_save, outname)

    def plot_scan(self, scan):
        """
        Plot a scan
        Args:
            scan: a [vimms.MassSpec.Scan][] object.

        Returns: None

        """
        plt.figure()
        for i in range(scan.num_peaks):
            x1 = scan.mzs[i]
            x2 = scan.mzs[i]
            y1 = 0
            y2 = scan.intensities[i]
            a = [[x1, y1], [x2, y2]]
            plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
        plt.title('Scan {0} {1}s -- {2} peaks'.format(scan.scan_id, scan.rt,
                                                      scan.num_peaks))
        plt.xlabel('m/z')
        plt.ylabel('Intensities')
        plt.show()
