"""
This file describes controllers that build regions-of-interests (ROIs) in real-time
and use that as additional information to decide which precursor ions to fragment.
"""
import copy
from copy import deepcopy

import numpy as np
from loguru import logger

from vimms.Common import ROI_EXCLUSION_DEW, ROI_EXCLUSION_WEIGHTED_DEW
from vimms.Controller.topN import TopNController
from vimms.Exclusion import (
    MinIntensityFilter, LengthFilter, SmartROIFilter,
    WeightedDEWExclusion, DEWFilter,
    WeightedDEWFilter
)
from vimms.MassSpec import Scan
from vimms.Roi import RoiBuilder


class RoiController(TopNController):
    """
    An ROI based controller with multiple options
    """

    def __init__(self, ionisation_mode, isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 smartroi_params=None,
                 min_roi_length_for_fragmentation=0,
                 ms1_shift=0,
                 advanced_params=None,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None,
                 deisotope=False,
                 charge_range=(2, 3),
                 min_fit_score=80,
                 penalty_factor=1.5,
                 use_quick_charge=False):
        """
        Initialise an ROI-based controller
        Args:
            ionisation_mode: ionisation mode, either POSITIVE or NEGATIVE
            isolation_width: isolation width in Dalton
            N: the number of highest-score precursor ions to fragment
            mz_tol: m/z tolerance -- m/z tolerance for dynamic exclusion window
            rt_tol: RT tolerance -- RT tolerance for dynamic exclusion window
            min_ms1_intensity: the minimum intensity to fragment a precursor ion
            roi_params: an instance of [vimms.Roi.RoiBuilderParams][] that describes
                        how to build ROIs in real time based on incoming scans.
            smartroi_params: an instance of [vimms.Roi.SmartRoiParams][]. If provided, then
                             the SmartROI rules (as described in the paper) will be used to select
                             which ROI to fragment. Otherwise set to None to use standard ROIs.
            min_roi_length_for_fragmentation: how long a ROI should be before it can be fragmented.
            ms1_shift: advanced parameter -- best to leave it.
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
            exclusion_method: an instance of [vimms.Exclusion.TopNExclusion][] or its subclasses,
                              used to describe how to perform dynamic exclusion so that precursors
                              that have been fragmented are not fragmented again.
            exclusion_t_0: parameter for WeightedDEW exclusion (refer to paper for details).
            deisotope: whether to perform isotopic deconvolution, necessary for proteomics.
            charge_range: the charge state of ions to keep.
            min_fit_score: minimum score to keep from doing isotope deconvolution.
            penalty_factor: penalty factor for scoring during isotope deconvolution.
        """
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                         min_ms1_intensity, ms1_shift=ms1_shift,
                         deisotope=deisotope, charge_range=charge_range,
                         min_fit_score=min_fit_score, penalty_factor=penalty_factor, use_quick_charge=use_quick_charge,
                         advanced_params=advanced_params)
        self.min_roi_length_for_fragmentation = min_roi_length_for_fragmentation  # noqa
        self.roi_builder = RoiBuilder(roi_params, smartroi_params=smartroi_params)

        self.exclusion_method = exclusion_method
        assert self.exclusion_method in [ROI_EXCLUSION_DEW,
                                         ROI_EXCLUSION_WEIGHTED_DEW]
        if self.exclusion_method == ROI_EXCLUSION_WEIGHTED_DEW:
            assert exclusion_t_0 is not None, 'Must be a number'
            assert exclusion_t_0 < rt_tol, 'Impossible combination'
            self.exclusion = WeightedDEWExclusion(mz_tol, rt_tol, exclusion_t_0)

        self.exclusion_t_0 = exclusion_t_0

    def schedule_ms1(self, new_tasks):
        """
        Schedule a new MS1 scan by creating a new default MS1 [vimms.Common.ScanParameters][].
        Args:
            new_tasks: the list of new tasks in the environment

        Returns: None

        """
        ms1_scan_params = self.get_ms1_scan_params()
        self.current_task_id += 1
        self.next_processed_scan_id = self.current_task_id
        new_tasks.append(ms1_scan_params)

    class MS2Scheduler():
        """
        A class that performs MS2 scheduling of tasks
        """

        def __init__(self, parent):
            """
            Initialises an MS2 scheduler
            Args:
                parent: the parent controller class
            """
            self.parent = parent
            self.fragmented_count = 0

        def schedule_ms2s(self, new_tasks, ms2_tasks, mz, intensity):
            """
            Schedule a new MS2 scan by creating a new default MS2 [vimms.Common.ScanParameters][].
            Args:
                new_tasks: the list of new tasks in the environment
                ms2_tasks: the list of MS2 tasks in the environment
                mz: the precursor m/z to fragment
                intensity: the precusor intensity to fragment

            Returns: None

            """
            precursor_scan_id = self.parent.scan_to_process.scan_id
            dda_scan_params = self.parent.get_ms2_scan_params(
                mz, intensity, precursor_scan_id, self.parent.isolation_width,
                self.parent.mz_tol, self.parent.rt_tol)
            new_tasks.append(dda_scan_params)
            ms2_tasks.append(dda_scan_params)
            self.parent.current_task_id += 1
            self.fragmented_count += 1

    def _set_fragmented(self, i, roi_id, rt, intensity):
        self.roi_builder.set_fragmented(self.current_task_id, i, roi_id, rt, intensity)

    def _process_scan(self, scan):
        if self.scan_to_process is not None:
            assert self.scan_to_process == scan

            # perform isotope deconvolution, necessary for proteomics data
            if self.deisotope:
                mzs = self.scan_to_process.mzs
                intensities = self.scan_to_process.intensities
                assert mzs.shape == intensities.shape
                mzs, intensities = self._deisotope(mzs, intensities)
                scan = Scan(scan.scan_id, mzs, intensities, scan.ms_level, scan.rt)

            # keep growing ROIs if we encounter a new ms1 scan
            self.roi_builder.update_roi(scan)
            new_tasks, ms2_tasks = [], []
            rt = self.scan_to_process.rt

            done_ms1, ms2s, scores = False, self.MS2Scheduler(self), self._get_scores()
            for i in np.argsort(scores)[::-1]:
                # stopping criteria is done based on the scores
                if scores[i] <= 0:
                    break

                mz, intensity, roi_id = self.roi_builder.get_mz_intensity(i)
                ms2s.schedule_ms2s(new_tasks, ms2_tasks, mz, intensity)
                self._set_fragmented(i, roi_id, rt, intensity)

                if ms2s.fragmented_count == self.N - self.ms1_shift:
                    self.schedule_ms1(new_tasks)
                    done_ms1 = True

            # if no ms1 has been added, then add at the end
            if not done_ms1:
                self.schedule_ms1(new_tasks)

            # create new exclusion items based on the scheduled ms2 tasks
            if self.exclusion is not None:
                self.exclusion.update(self.scan_to_process, ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
            return new_tasks

        elif scan.ms_level == 2:  # add ms2 scans to Rois
            self.roi_builder.add_scan_to_roi(scan)
            return []

    ###########################################################################
    # Scoring functions
    ###########################################################################

    def _log_roi_intensities(self):
        """
        Scores ROI by their log intensity values

        Returns: a numpy array of log intensity scores

        """
        return np.log(self.roi_builder.current_roi_intensities)

    def _min_intensity_filter(self):
        """
        Filter ROIs by minimum intensity threshold

        Returns: indicators whether ROIs pass the check

        """
        f = MinIntensityFilter(self.min_ms1_intensity)
        return f.filter(self.roi_builder.current_roi_intensities)

    def _time_filter(self):
        """
        Filter ROIs by dynamic exclusion

        Returns: indicators whether ROIs pass the check

        """
        if self.exclusion_method == ROI_EXCLUSION_DEW:
            f = DEWFilter(self.rt_tol)
            return f.filter(self.scan_to_process.rt, self.roi_builder.live_roi)

        elif self.exclusion_method == ROI_EXCLUSION_WEIGHTED_DEW:
            f = WeightedDEWFilter(self.exclusion)
            return f.filter(self.scan_to_process.rt, self.roi_builder.live_roi)

    def _length_filter(self):
        """
        Filter ROI based on their length (>= min_roi_length_for_fragmentation)

        Returns: indicators whether ROIs pass the check

        """
        f = LengthFilter(self.min_roi_length_for_fragmentation)
        return f.filter(self.roi_builder.current_roi_length)

    def _smartroi_filter(self):
        """
        Filter ROI based on the SmartROI rules

        Returns: indicators whether ROIs pass the check

        """
        f = SmartROIFilter()
        return f.filter(self.roi_builder.live_roi)

    def _score_filters(self):
        """
        Combine various filtering criteria

        Returns: the combined scoring criteria

        """
        return self._min_intensity_filter() * self._time_filter() * self._length_filter()

    def _get_dda_scores(self):
        """
        Compute DDA scores = log roi intensities * the different scoring criteria
        Returns:

        """
        return self._log_roi_intensities() * self._score_filters()

    def _get_top_N_scores(self, scores):
        """
        Select the topN highest scores and set the rest to 0.

        Args:
            scores: a numpy array of scores

        Returns: same scores, but keeping only the top-N largest values.

        """
        if len(scores) > self.N:  # number of fragmentation events filter
            scores[scores.argsort()[:(len(scores) - self.N)]] = 0
        return scores

    def _get_scores(self):
        """
        Computes scores used to rank precursor ions to fragment

        Returns: an array of scores. Highest scoring ions should be selected first.

        """
        NotImplementedError()


class TopN_SmartRoiController(RoiController):
    """
    A ROI-based controller that implements the Top-N selection with SmartROI rules.
    This is used in the paper 'Rapid Development ...'
    """

    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 smartroi_params,
                 min_roi_length_for_fragmentation=0,
                 ms1_shift=0,
                 advanced_params=None,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None,
                 deisotope=False,
                 charge_range=(2, 3),
                 min_fit_score=80,
                 penalty_factor=1.5,
                 use_quick_charge=False):
        """
        Initialise the Top-N SmartROI controller.

        Args:
            ionisation_mode: ionisation mode, either POSITIVE or NEGATIVE
            isolation_width: isolation width in Dalton
            N: the number of highest-score precursor ions to fragment
            mz_tol: m/z tolerance -- m/z tolerance for dynamic exclusion window
            rt_tol: RT tolerance -- RT tolerance for dynamic exclusion window
            min_ms1_intensity: the minimum intensity to fragment a precursor ion
            roi_params: an instance of [vimms.Roi.RoiBuilderParams][] that describes
                        how to build ROIs in real time based on incoming scans.
            smartroi_params: an instance of [vimms.Roi.SmartRoiParams][]. If provided, then
                             the SmartROI rules (as described in the paper) will be used to select
                             which ROI to fragment. Otherwise set to None to use standard ROIs.
            min_roi_length_for_fragmentation: how long a ROI should be before it can be fragmented.
            ms1_shift: advanced parameter -- best to leave it.
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
            exclusion_method: an instance of [vimms.Exclusion.TopNExclusion][] or its subclasses,
                              used to describe how to perform dynamic exclusion so that precursors
                              that have been fragmented are not fragmented again.
            exclusion_t_0: parameter for WeightedDEW exclusion (refer to paper for details).
            deisotope: whether to perform isotopic deconvolution, necessary for proteomics.
            charge_range: the charge state of ions to keep.
            min_fit_score: minimum score to keep from doing isotope deconvolution.
            penalty_factor: penalty factor for scoring during isotope deconvolution.
        """
        super().__init__(ionisation_mode, isolation_width,
                         N,
                         mz_tol,
                         rt_tol,
                         min_ms1_intensity,
                         roi_params,
                         smartroi_params=smartroi_params,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         ms1_shift=ms1_shift,
                         advanced_params=advanced_params,
                         exclusion_method=exclusion_method,
                         exclusion_t_0=exclusion_t_0,
                         deisotope=deisotope,
                         charge_range=charge_range,
                         min_fit_score=min_fit_score,
                         penalty_factor=penalty_factor,
                         use_quick_charge=use_quick_charge)

    def _get_dda_scores(self):
        return self._log_roi_intensities() * self._min_intensity_filter() * \
            self._smartroi_filter()

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class TopN_RoiController(RoiController):
    """
    A ROI-based controller that implements the Top-N selection.
    """

    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 min_roi_length_for_fragmentation=0,
                 ms1_shift=0,
                 advanced_params=None,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None,
                 deisotope=False,
                 charge_range=(2, 3),
                 min_fit_score=80,
                 penalty_factor=1.5,
                 use_quick_charge=False):
        """
        Initialise the Top-N SmartROI controller.

        Args:
            ionisation_mode: ionisation mode, either POSITIVE or NEGATIVE
            isolation_width: isolation width in Dalton
            N: the number of highest-score precursor ions to fragment
            mz_tol: m/z tolerance -- m/z tolerance for dynamic exclusion window
            rt_tol: RT tolerance -- RT tolerance for dynamic exclusion window
            min_ms1_intensity: the minimum intensity to fragment a precursor ion
            roi_params: an instance of [vimms.Roi.RoiBuilderParams][] that describes
                        how to build ROIs in real time based on incoming scans.
            min_roi_length_for_fragmentation: how long a ROI should be before it can be fragmented.
            ms1_shift: advanced parameter -- best to leave it.
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
            exclusion_method: an instance of [vimms.Exclusion.TopNExclusion][] or its subclasses,
                              used to describe how to perform dynamic exclusion so that precursors
                              that have been fragmented are not fragmented again.
            exclusion_t_0: parameter for WeightedDEW exclusion (refer to paper for details).
            deisotope: whether to perform isotopic deconvolution, necessary for proteomics.
            charge_range: the charge state of ions to keep.
            min_fit_score: minimum score to keep from doing isotope deconvolution.
            penalty_factor: penalty factor for scoring during isotope deconvolution.
        """
        super().__init__(ionisation_mode,
                         isolation_width,
                         N,
                         mz_tol,
                         rt_tol,
                         min_ms1_intensity,
                         roi_params,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         ms1_shift=ms1_shift,
                         advanced_params=advanced_params,
                         exclusion_method=exclusion_method,
                         exclusion_t_0=exclusion_t_0,
                         deisotope=deisotope,
                         charge_range=charge_range,
                         min_fit_score=min_fit_score,
                         penalty_factor=penalty_factor,
                         use_quick_charge=use_quick_charge)

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores
