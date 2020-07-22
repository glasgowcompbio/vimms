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
        # make the SM1 scan
        task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                        max_it=self.ms1_max_it,
                                                        collision_energy=self.ms1_collision_energy,
                                                        orbitrap_resolution=self.ms1_orbitrap_resolution)
        task.set(ScanParameters.FIRST_MASS, self.min_mz)
        task.set(ScanParameters.LAST_MASS, self.max_mz)
        # task.set(ScanParameters.CURRENT_TOP_N, 10) # time sampling fix see iss18

        scans.append(task)
        self.scan_number += 1 # increase every time we make a scan
        
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
        self.next_processed_scan_id = self.scan_number
        self.scan_number += 1
        return scans
        