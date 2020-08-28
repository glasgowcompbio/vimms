from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, INITIAL_SCAN_ID
from vimms.MassSpec import ScanParameters

from vimms.Controller import Controller

class AIF(Controller):
    def __init__(self,min_mz,max_mz,
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):

        super().__init__()
        self.min_mz = min_mz # scan from this mz
        self.max_mz = max_mz # scan to this mz

        # advanced parameters
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution

        self.scan_number = self.initial_scan_id

    # method required by super-class
    def update_state_after_scan(self, last_scan):
        pass

    # method required by super-class
    def reset(self):
        pass

    def _process_scan(self, scan):
        # method called when a scan arrives that requires action
        # normally means that we should schedule some more
        # in DIA we don't need to actually look at the peaks
        # in the scan, just schedule the next block

        # For all ions fragmentation, when we receive the last scan of the previous block
        # we make a new block
        # each block is an MS1 scan followed by an MS2 scan where the MS2 fragmens everything
        scans = []

        if self.scan_to_process is not None:
            
            # make the MS2 scan
            dda_scan_params = ScanParameters()
            dda_scan_params.set(ScanParameters.MS_LEVEL, 2)
            isolation_windows = [[[self.min_mz,self.max_mz]]] # why tripple brackets needed?
            dda_scan_params.set(ScanParameters.ISOLATION_WINDOWS, isolation_windows)
            dda_scan_params.set(ScanParameters.COLLISION_ENERGY,self.ms2_collision_energy)
            dda_scan_params.set(ScanParameters.AGC_TARGET,self.ms2_agc_target)
            dda_scan_params.set(ScanParameters.MAX_IT,self.ms2_max_it)
            dda_scan_params.set(ScanParameters.ORBITRAP_RESOLUTION,self.ms2_orbitrap_resolution)
            dda_scan_params.set(ScanParameters.PRECURSOR_MZ, 0.5*(self.max_mz + self.min_mz))
            dda_scan_params.set(ScanParameters.FIRST_MASS, self.min_mz)
            dda_scan_params.set(ScanParameters.LAST_MASS, self.max_mz)
            # dda_scan_params.set(ScanParameters.CURRENT_TOP_N,10) # time sampling fix see iss18


            scans.append(dda_scan_params)
            self.scan_number += 1 # increase every time we make a scan

            # make the MS1 scan
            task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                            max_it=self.ms1_max_it,
                                                            collision_energy=self.ms1_collision_energy,
                                                            orbitrap_resolution=self.ms1_orbitrap_resolution)
            task.set(ScanParameters.FIRST_MASS, self.min_mz)
            task.set(ScanParameters.LAST_MASS, self.max_mz)
            # task.set(ScanParameters.CURRENT_TOP_N, 10) # time sampling fix see iss18

            scans.append(task)
            self.scan_number += 1
            self.next_processed_scan_id = self.scan_number


            # set this ms1 scan as has been processed
            self.scan_to_process = None

        return scans
        

class SWATH(Controller):
    def __init__(self, min_mz, max_mz, 
                  num_windows=1, scan_overlap=0,
                  ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                  ms1_max_it=DEFAULT_MS1_MAXIT,
                  ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                  ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION):
        super().__init__()
        self.num_windows = num_windows
        self.scan_overlap = scan_overlap
        self.min_mz = min_mz # scan from this mz
        self.max_mz = max_mz # scan to this mz
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution
        self.scan_number = self.initial_scan_id
        self.exp_info = [] # experimental information - isolation windows

    def handle_acquisition_open(self):
        logger.info('Acquisition open')

    def handle_acquisition_closing(self):
        logger.info('Acquisition closing')

    def reset(self):
        pass

    def update_state_after_scan(self, last_scan):
        # add noise for those scan wihout peaks, e.g. keep scan in mzML file
        if last_scan.num_peaks == 0:
            num_peaks = 10 # create 10 peaks
            noise_peaks = self._create_noise_peaks(last_scan, num_peaks)
            self.scans[last_scan.ms_level][-1].mzs = noise_peaks[0]
            self.scans[last_scan.ms_level][-1].intensities = noise_peaks[1]
            self.scans[last_scan.ms_level][-1].num_peaks = num_peaks

    def _create_noise_peaks(self, last_scan, num):
        isolation_windows = last_scan.scan_params.get(ScanParameters.ISOLATION_WINDOWS)
        iso_min = isolation_windows[0][0][0]  # lower bound isolation window, in Da
        iso_max = isolation_windows[0][0][1]  # upper bound isolation window, in Da
        scan_mzs = np.random.uniform(iso_min, iso_max, num)
        scan_intensities = np.random.uniform(5, 100, num)
        return scan_mzs, scan_intensities

    def _process_scan(self, scan):
        new_tasks = []
        if self.scan_to_process is not None:
            # if there's a previous ms1 scan to process
            # then get the last ms1 scan, select bin walls and create scan locations
            mzs = self.last_scan.mzs
            default_range = [(self.min_mz, self.max_mz)]
            locations = DiaWindows(mzs, default_range, "basic", "even", None,
                                   0, self.num_windows).locations
            logger.debug('Window locations {}'.format(locations))
            for i in range(len(locations)):  # define isolation window around the selected precursor ions
                adjust_min = -self.scan_overlap/2
                adjust_max = self.scan_overlap/2                
                if i == 0:
                    adjust_min = 0
                elif i == len(locations)-1:
                    adjust_max = 0
                isolation_windows = locations[i]
                assert self.scan_overlap/2 < isolation_windows[0][0][1] - isolation_windows[0][0][0]
                isowindows_with_offset = [[(isolation_windows[0][0][0] + adjust_min, isolation_windows[0][0][1] + adjust_max)]]
                logger.info('%d\tSWATH\t%.1f\t%.1f' %(i+1, isowindows_with_offset[0][0][0], isowindows_with_offset[0][0][1]) )
                if scan.scan_id == self.initial_scan_id:
                    if i == 0:
                        fist_mz = isowindows_with_offset[0][0][0]
                        self.exp_info.append((0,0,0))
                    if i == len(locations)-1:
                        self.exp_info[0] = (0, fist_mz, isowindows_with_offset[0][0][1])
                    self.exp_info.append((i+1, isowindows_with_offset[0][0][0], isowindows_with_offset[0][0][1]))
                dda_scan_params = ScanParameters()
                dda_scan_params.set(ScanParameters.MS_LEVEL, 2)
                dda_scan_params.set(ScanParameters.ISOLATION_WINDOWS, isowindows_with_offset)
                dda_scan_params.set(ScanParameters.COLLISION_ENERGY,DEFAULT_MS2_COLLISION_ENERGY)
                dda_scan_params.set(ScanParameters.FIRST_MASS, self.min_mz)
                dda_scan_params.set(ScanParameters.LAST_MASS, self.max_mz)

                new_tasks.append(dda_scan_params)  # push this dda scan to the mass spec queue

            # set this ms1 scan as has been processed
            #self.last_ms1_scan = None
            # make the MS1 scan
            task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                            max_it=self.ms1_max_it,
                                                            collision_energy=self.ms1_collision_energy,
                                                            orbitrap_resolution=self.ms1_orbitrap_resolution)

            new_tasks.append(task)

            self.scan_number += len(locations) + 1
            self.next_processed_scan_id = self.scan_number
        return new_tasks