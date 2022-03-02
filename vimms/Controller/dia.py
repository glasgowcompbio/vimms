import csv
import os

from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, \
    DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, \
    DEFAULT_MS2_AGC_TARGET, \
    DEFAULT_MS2_MAXIT, \
    DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS2_ISOLATION_MODE, \
    DEFAULT_MS2_ACTIVATION_TYPE, \
    DEFAULT_MS2_MASS_ANALYSER, create_if_not_exist, ScanParameters, \
    get_default_scan_params, get_dda_scan_param
from vimms.Controller import Controller
from vimms.DIA import DiaWindows


class AIF(Controller):
    def __init__(self, ms1_source_cid_energy, advanced_params=None):
        super().__init__(advanced_params=advanced_params)
        self.scan_number = self.initial_scan_id
        self.ms1_source_cid_energy = ms1_source_cid_energy

    def write_msdial_experiment_file(self, filename):
        heads = ['ID', 'MS Type', 'Start m/z', 'End m/z', 'Name', 'CE',
                 'DecTarget(1:Yes, 0:No)']
        start = self.advanced_params.default_ms1_scan_window[0]
        stop = self.advanced_params.default_ms1_scan_window[1]
        ce = self.ms1_source_cid_energy
        ms1_row = ['0', 'SCAN', start, stop, "0eV", 0, 0]
        aif_row = ['1', 'ALL', start, stop, "{}eV".format(ce), ce, 1]

        out_dir = os.path.dirname(filename)
        create_if_not_exist(out_dir)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t', dialect='excel')
            writer.writerow(heads)
            writer.writerow(ms1_row)
            writer.writerow(aif_row)

    # method required by super-class
    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        # method called when a scan arrives that requires action
        # normally means that we should schedule some more
        # in DIA we don't need to actually look at the peaks
        # in the scan, just schedule the next block

        # For all ions fragmentation, when we receive the last scan of
        # the previous block, we make a new block. Each block is an MS1 scan
        # followed by an MS2 scan where the MS2 fragmens everything
        scans = []

        if self.scan_to_process is not None:
            # make the MS1 scan with source cid energy applied
            aif_scan = self.get_ms1_scan_params()
            aif_scan.set(ScanParameters.SOURCE_CID_ENERGY,
                         self.ms1_source_cid_energy)
            self._check_scan(aif_scan)

            scans.append(aif_scan)
            self.scan_number += 1  # increase every time we make a scan

            # make the MS1 scan with no energy applied
            ms1_scan = self.get_ms1_scan_params()
            self._check_scan(ms1_scan)

            scans.append(ms1_scan)
            self.scan_number += 1
            self.next_processed_scan_id = self.scan_number

            # set this ms1 scan as has been processed
            self.scan_to_process = None

        return scans


class SWATH(Controller):
    def __init__(self, min_mz, max_mz,
                 width, scan_overlap=0,
                 advanced_params=None):
        super().__init__(advanced_params=advanced_params)
        self.width = width
        self.scan_overlap = scan_overlap
        self.min_mz = min_mz  # scan from this mz
        self.max_mz = max_mz  # scan to this mz

        self.scan_number = self.initial_scan_id
        self.exp_info = []  # experimental information - isolation windows

    def write_msdial_experiment_file(self, filename):
        heads = ['Experiment', 'MS Type', 'Min m/z', 'Max m/z']
        start_mz, stop_mz = self._get_start_stop()
        ms1_mz_range = self.advanced_params.default_ms1_scan_window
        ms1_row = ['0', 'SCAN', ms1_mz_range[0], ms1_mz_range[1]]
        swath_rows = []
        for i, start in enumerate(start_mz):
            stop = stop_mz[i]
            new_row = [i + 1, 'SWATH', start, stop]
            swath_rows.append(new_row)

        out_dir = os.path.dirname(filename)
        create_if_not_exist(out_dir)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t', dialect='excel')
            writer.writerow(heads)
            writer.writerow(ms1_row)
            for row in swath_rows:
                writer.writerow(row)

    def _get_start_stop(self):
        start = self.min_mz
        start_mz = []
        stop_mz = []
        while start < self.max_mz:
            start_mz.append(start)
            stop_mz.append(start + self.width)
            start += self.width - self.scan_overlap
        return start_mz, stop_mz

    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:

            precursor_scan_id = self.scan_to_process.scan_id

            start_mz, stop_mz = self._get_start_stop()

            isolation_width = self.width

            precursor_mz_list = []
            for i, start in enumerate(start_mz):
                precursor_mz = (stop_mz[i] + start) / 2.
                precursor_mz_list.append(precursor_mz)

            mz_tol = 10  # not used
            rt_tol = 15  # these are not used
            for mz in precursor_mz_list:
                dda_scan_params = self.get_ms2_scan_params(mz, 0,
                                                           precursor_scan_id,
                                                           isolation_width,
                                                           mz_tol, rt_tol)
                # push this dda scan to the mass spec queue
                new_tasks.append(dda_scan_params)

            # make the MS1 scan
            task = self.get_ms1_scan_params()
            new_tasks.append(task)

            self.scan_number += len(precursor_mz_list) + 1
            self.next_processed_scan_id = self.scan_number
        return new_tasks


class DiaController(Controller):
    # Class for doing tree and nested DIA methods.
    # Also has a SWATH type controller, but reccommend to use SWATH class
    # above. Method uses windows methods from DIA.py to create the pattern
    # of windows needed to run the controllers.
    # Note: the following method used multiple simultaneous isolation windows
    def __init__(self, min_mz, max_mz,  # TODO: add scan overlap to DiaWindows
                 window_type, kaufmann_design, num_windows, scan_overlap=0,
                 extra_bins=0, dia_design='kaufmann',
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ):
        super().__init__()
        self.dia_design = dia_design
        self.window_type = window_type
        self.kaufmann_design = kaufmann_design
        self.extra_bins = extra_bins
        self.num_windows = num_windows
        self.scan_overlap = scan_overlap
        self.min_mz = min_mz  # scan from this mz
        self.max_mz = max_mz  # scan to this mz
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution
        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution
        self.ms2_isolation_mode = ms2_isolation_mode
        self.ms2_activation_type = ms2_activation_type
        self.ms2_mass_analyser = ms2_mass_analyser

        self.scan_number = self.initial_scan_id
        self.exp_info = []  # experimental information - isolation windows

    def update_state_after_scan(self, last_scan):
        pass

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.scan_to_process is not None:

            mz_tol = 10  # not used
            rt_tol = 15  # these are not used

            precursor_scan_id = self.scan_to_process.scan_id

            mzs = self.scan_to_process.mzs
            if len(mzs) > 0:  # check that ms1 scan is not empty
                default_range = [(self.min_mz, self.max_mz)]
                locations = DiaWindows(mzs, default_range, self.dia_design,
                                       self.window_type, self.kaufmann_design,
                                       self.extra_bins,
                                       self.num_windows).locations
                for loc in locations:
                    mz = []
                    isolation_width = []
                    intensity = []
                    for sub_loc in loc[0]:
                        mz.append(sum(sub_loc) / 2)
                        isolation_width.append(sub_loc[1] - sub_loc[0])
                        intensity.append(0)
                    dda_scan_params = get_dda_scan_param(
                        mz, intensity, precursor_scan_id, isolation_width,
                        mz_tol, rt_tol,
                        agc_target=self.ms2_agc_target,
                        max_it=self.ms2_max_it,
                        collision_energy=self.ms2_collision_energy,
                        orbitrap_resolution=self.ms2_orbitrap_resolution,
                        activation_type=self.ms2_activation_type,
                        mass_analyser=self.ms2_mass_analyser,
                        isolation_mode=self.ms2_isolation_mode,
                        polarity=self.environment.mass_spec.ionisation_mode)

                    # push this dda scan to the mass spec queue
                    new_tasks.append(dda_scan_params)
            else:
                locations = []

            # make the MS1 scan
            task = get_default_scan_params(
                polarity=self.environment.mass_spec.ionisation_mode,
                agc_target=self.ms1_agc_target,
                max_it=self.ms1_max_it,
                collision_energy=self.ms1_collision_energy,
                orbitrap_resolution=self.ms1_orbitrap_resolution)

            new_tasks.append(task)

            self.scan_number += len(locations) + 1
            self.next_processed_scan_id = self.scan_number
        return new_tasks
