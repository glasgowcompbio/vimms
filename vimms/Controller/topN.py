import numpy as np
from loguru import logger
from ms_deisotope import (
    PenalizedMSDeconVFitter,
    AveragineDeconvoluter,
)
from ms_deisotope.deconvolution import deconvolute_peaks
from ms_deisotope.deconvolution.utils import prepare_peaklist

from vimms.Common import DUMMY_PRECURSOR_MZ
from vimms.Controller.base import Controller
from vimms.Exclusion import TopNExclusion, WeightedDEWExclusion


class TopNController(Controller):
    """
    A controller that implements the standard Top-N DDA fragmentation strategy.
    Does an MS1 scan followed by N fragmentation scans of
    the peaks with the highest intensity that are not excluded
    """

    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                 min_ms1_intensity,
                 ms1_shift=0, initial_exclusion_list=None, advanced_params=None,
                 force_N=False, exclude_after_n_times=1, exclude_t0=0,
                 deisotope=False, charge_range=(2, 3), min_fit_score=80, penalty_factor=1.5,
                 use_quick_charge=False):
        """
        Initialise the Top-N controller

        Args:
            ionisation_mode: ionisation mode, either POSITIVE or NEGATIVE
            N: the number of highest-intensity precursor ions to fragment
            isolation_width: isolation width in Dalton
            mz_tol: m/z tolerance -- m/z tolerance for dynamic exclusion window
            rt_tol: RT tolerance -- RT tolerance for dynamic exclusion window
            min_ms1_intensity: the minimum intensity to fragment a precursor ion
            ms1_shift: advanced parameter -- best to leave it.
            initial_exclusion_list: initial list of exclusion boxes
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
            force_N: whether to always force N fragmentations.
            exclude_after_n_times; allow ions to NOT be excluded up to n_times.
            exclude_t0: time for initial exclusion check.
            deisotope: whether to perform isotopic deconvolution, necessary for proteomics.
            charge_range: the charge state of ions to keep.
            min_fit_score: minimum score to keep from doing isotope deconvolution.
            penalty_factor: penalty factor for scoring during isotope deconvolution.
        """
        super().__init__(advanced_params=advanced_params)

        self.ionisation_mode = ionisation_mode

        # the top N ions to fragment
        self.N = N

        # the isolation width (in Dalton) to select a precursor ion
        self.isolation_width = isolation_width

        # the m/z window (ppm) to prevent the same precursor ion to be
        # fragmented again
        self.mz_tol = mz_tol

        # the rt window to prevent the same precursor ion to be
        # fragmented again
        self.rt_tol = rt_tol

        # minimum ms1 intensity to fragment
        self.min_ms1_intensity = min_ms1_intensity

        # number of scans to move ms1 scan forward in list of new_tasks
        self.ms1_shift = ms1_shift

        # force it to do N MS2 scans regardless
        self.force_N = force_N

        if self.force_N and ms1_shift > 0:
            logger.warning(
                "Setting force_N to True with non-zero shift can lead to "
                "strange behaviour")

        self.exclusion = TopNExclusion(self.mz_tol, self.rt_tol,
                                       exclude_after_n_times=exclude_after_n_times,
                                       exclude_t0=exclude_t0,
                                       initial_exclusion_list=initial_exclusion_list)

        # for isotope filtering using ms_deisotope
        self.deisotope = deisotope
        self.charge_range = charge_range
        self.min_fit_score = min_fit_score
        self.penalty_factor = penalty_factor
        self.use_quick_charge = use_quick_charge

        if self.deisotope:
            scorer = PenalizedMSDeconVFitter(
                minimum_score=self.min_fit_score,
                penalty_factor=self.penalty_factor,
                mass_error_tolerance=0.00002
            )
            self.dc = {'scorer': scorer}

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        fragmented_count = 0

        if self.scan_to_process is not None:
            assert self.scan_to_process == scan

            # original scan data
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            assert mzs.shape == intensities.shape
            rt = self.scan_to_process.rt

            # perform isotope deconvolution, necessary for proteomics data
            if self.deisotope:
                mzs, intensities = self._deisotope(mzs, intensities)

            # loop over points in decreasing intensity
            idx = np.argsort(intensities)[::-1]

            done_ms1 = False
            ms2_tasks = []
            for i in idx:
                mz = mzs[i]
                intensity = intensities[i]

                # stopping criteria is after we've fragmented N ions or
                # we found ion < min_intensity
                if fragmented_count >= self.N:
                    break

                if intensity < self.min_ms1_intensity:
                    logger.debug(
                        'Time %f Minimum intensity threshold %f reached '
                        'at %f, %d' % (rt, self.min_ms1_intensity, intensity,
                                       fragmented_count))
                    break

                # skip ion in the dynamic exclusion list of the mass spec
                is_exc, weight = self.exclusion.is_excluded(mz, rt)
                if is_exc:
                    continue

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.get_ms2_scan_params(
                    mz, intensity, precursor_scan_id, self.isolation_width,
                    self.mz_tol, self.rt_tol)
                new_tasks.append(dda_scan_params)
                ms2_tasks.append(dda_scan_params)
                fragmented_count += 1
                self.current_task_id += 1

                # add an ms1 here
                if fragmented_count == self.N - self.ms1_shift:
                    ms1_scan_params = self.get_ms1_scan_params()
                    self.current_task_id += 1
                    self.next_processed_scan_id = self.current_task_id
                    new_tasks.append(ms1_scan_params)
                    done_ms1 = True

            if self.force_N and len(new_tasks) < self.N:
                # add some extra tasks.
                n_tasks_remaining = self.N - len(new_tasks)
                for i in range(n_tasks_remaining):
                    precursor_scan_id = self.scan_to_process.scan_id
                    dda_scan_params = self.get_ms2_scan_params(
                        DUMMY_PRECURSOR_MZ, 100.0, precursor_scan_id,
                        self.isolation_width,
                        self.mz_tol, self.rt_tol)
                    new_tasks.append(dda_scan_params)
                    ms2_tasks.append(dda_scan_params)
                    fragmented_count += 1
                    self.current_task_id += 1

            # if no ms1 has been added, then add at the end
            if not done_ms1:
                # if fragmented_count < self.N - self.ms1_shift:
                ms1_scan_params = self.get_ms1_scan_params()
                self.current_task_id += 1
                self.next_processed_scan_id = self.current_task_id
                new_tasks.append(ms1_scan_params)

            # create new exclusion items based on the scheduled ms2 tasks
            self.exclusion.update(self.scan_to_process, ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def _deisotope(self, mzs, intensities):
        pl = prepare_peaklist((mzs, intensities))
        dt = AveragineDeconvoluter
        ps = deconvolute_peaks(pl, decon_config=self.dc, charge_range=self.charge_range,
                               deconvoluter_type=dt, use_quick_charge=self.use_quick_charge)

        peaks = ps.peak_set.peaks
        mzs = np.array([peak.mz for peak in peaks])
        intensities = np.array([peak.intensity for peak in peaks])
        return mzs, intensities

    def update_state_after_scan(self, scan):
        pass


class ScanItem():
    """
    Represents a scan item object. Used by the WeightedDEW controller to store
    the pair of m/z and intensity values along with their associated weight
    """

    def __init__(self, mz, intensity, weight=1):
        """
        Initialise a ScanItem object
        Args:
            mz: m/z value
            intensity: intensity value
            weight: the weight for this ScanItem
        """
        self.mz = mz
        self.intensity = intensity
        self.weight = weight

    def __lt__(self, other):
        if self.intensity * self.weight <= other.intensity * other.weight:
            return True
        else:
            return False


class WeightedDEWController(TopNController):
    """
    A variant of the Top-N controller, but it uses a linear weight
     for dynamic exclusion window rather than a True/False indicator on whether
     a certain precursor ion is excluded or not. For more details, refer to our paper.
    """

    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                 min_ms1_intensity, ms1_shift=0,
                 exclusion_t_0=15, log_intensity=False,
                 deisotope=False, charge_range=(2, 3), min_fit_score=80, penalty_factor=1.5, use_quick_charge=False,
                 advanced_params=None):
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                         min_ms1_intensity, ms1_shift=ms1_shift,
                         deisotope=deisotope, charge_range=charge_range,
                         min_fit_score=min_fit_score, penalty_factor=penalty_factor, use_quick_charge=use_quick_charge,
                         advanced_params=advanced_params)
        self.log_intensity = log_intensity
        self.exclusion = WeightedDEWExclusion(mz_tol, rt_tol, exclusion_t_0)

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        fragmented_count = 0
        if self.scan_to_process is not None:
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt

            # perform isotope deconvolution, necessary for proteomics data
            if self.deisotope:
                mzs, intensities = self._deisotope(mzs, intensities)

            if not self.log_intensity:
                mzi = [ScanItem(mz, intensities[i]) for i, mz in enumerate(mzs)
                       if
                       intensities[i] >= self.min_ms1_intensity]
            else:
                # take log of intensities for peak scoring
                mzi = [ScanItem(mz, np.log(intensities[i])) for i, mz in
                       enumerate(mzs) if
                       intensities[i] >= self.min_ms1_intensity]

            for si in mzi:
                is_exc, weight = self.exclusion.is_excluded(si.mz, rt)
                si.weight = weight

            mzi.sort(reverse=True)

            done_ms1 = False
            ms2_tasks = []
            for i in range(len(mzi)):
                # mz = mzi[i].mz
                # intensity = mzi[i].intensity
                # stopping criteria is after we've fragmented N ions or we
                # found ion < min_intensity
                if fragmented_count >= self.N:
                    break

                mz = mzi[i].mz
                if not self.log_intensity:
                    intensity = mzi[i].intensity
                else:
                    intensity = np.exp(mzi[i].intensity)

                # if 138 <= mz <= 138.5:
                #     print(mz,intensity,mzi[i].weight)

                if mzi[i].weight == 0.0:
                    logger.debug(
                        'Time %f no ions left reached at %f, %d' % (
                            rt, intensity, fragmented_count))
                    break

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.get_ms2_scan_params(
                    mz, intensity, precursor_scan_id,
                    self.isolation_width, self.mz_tol, self.rt_tol)
                new_tasks.append(dda_scan_params)
                ms2_tasks.append(dda_scan_params)
                fragmented_count += 1
                self.current_task_id += 1

                # add an ms1 here
                if fragmented_count == self.N - self.ms1_shift:
                    ms1_scan_params = self.get_ms1_scan_params()
                    self.current_task_id += 1
                    self.next_processed_scan_id = self.current_task_id
                    new_tasks.append(ms1_scan_params)
                    done_ms1 = True

            # if no ms1 has been added, then add at the end
            if not done_ms1:
                # if fragmented_count < self.N - self.ms1_shift:
                ms1_scan_params = self.get_ms1_scan_params()
                self.current_task_id += 1
                self.next_processed_scan_id = self.current_task_id
                new_tasks.append(ms1_scan_params)

            # create new exclusion items based on the scheduled ms2 tasks
            self.exclusion.update(self.scan_to_process, ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks
