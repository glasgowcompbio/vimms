from loguru import logger

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW
from vimms.Controller import Controller
from vimms.DIA import DiaWindows
from vimms.MassSpec import ScanParameters


########################################################################################################################
# DIA Controllers
########################################################################################################################

class TreeController(Controller):
    def __init__(self, dia_design, window_type, kaufmann_design, extra_bins, num_windows=None):
        super().__init__()
        self.dia_design = dia_design
        self.window_type = window_type
        self.kaufmann_design = kaufmann_design
        self.extra_bins = extra_bins
        self.num_windows = num_windows

    def handle_acquisition_open(self):
        logger.info('Acquisition open')

    def handle_acquisition_closing(self):
        logger.info('Acquisition closing')

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.last_ms1_scan is not None:

            rt = self.last_ms1_scan.rt

            # then get the last ms1 scan, select bin walls and create scan locations
            mzs = self.last_ms1_scan.mzs
            default_range = [DEFAULT_MS1_SCAN_WINDOW]  # TODO: this should maybe come from somewhere else?
            locations = DiaWindows(mzs, default_range, self.dia_design, self.window_type, self.kaufmann_design,
                                   self.extra_bins, self.num_windows).locations
            logger.debug('Window locations {}'.format(locations))
            for i in range(len(locations)):  # define isolation window around the selected precursor ions
                isolation_windows = locations[i]
                dda_scan_params = ScanParameters()
                dda_scan_params.set(ScanParameters.MS_LEVEL, 2)
                dda_scan_params.set(ScanParameters.ISOLATION_WINDOWS, isolation_windows)
                new_tasks.append(dda_scan_params)  # push this dda scan to the mass spec queue

            # set this ms1 scan as has been processed
            self.last_ms1_scan = None
        return new_tasks
