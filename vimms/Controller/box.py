"""
A controller based on the notion of generic boxes.

Note: this module is still under development and might change significantly.
"""

from copy import deepcopy

import numpy as np

from vimms.Common import ROI_EXCLUSION_DEW, GRID_CONTROLLER_SCORING_PARAMS, \
    ROI_TYPE_SMART
from vimms.Controller.roi import RoiController
from vimms.Roi import RoiBuilder


class GridController(RoiController):
    """
    A multi-sample controller that use a grid to track which regions-of-interests (ROIs)
    have been fragmented across multiple injections.
    """

    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 grid,
                 smartroi_params=None,
                 min_roi_length_for_fragmentation=0,
                 ms1_shift=0,
                 min_rt_width=0.01,
                 min_mz_width=0.00001,
                 advanced_params=None,
                 register_all_roi=False,
                 scoring_params=GRID_CONTROLLER_SCORING_PARAMS,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None):
        """
        Create a grid controller.

        Args:
            ionisation_mode: ionisation mode, either POSITIVE or NEGATIVE
            isolation_width: isolation width in Dalton
            N: the number of highest-score precursor ions to fragment
            mz_tol: m/z tolerance -- m/z tolerance for dynamic exclusion window
            rt_tol: RT tolerance -- RT tolerance for dynamic exclusion window
            min_ms1_intensity: the minimum intensity to fragment a precursor ion
            roi_params: an instance of [vimms.Roi.RoiBuilderParams][] that describes
                        how to build ROIs in real time based on incoming scans.
            grid: an instance of grid object.
            smartroi_params: an instance of [vimms.Roi.SmartRoiParams][]. If provided, then
                             the SmartROI rules (as described in the paper) will be used to select
                             which ROI to fragment. Otherwise set to None to use standard ROIs.
            min_roi_length_for_fragmentation: how long a ROI should be before it can be fragmented.
            ms1_shift: advanced parameter -- best to leave it.
            min_rt_width: minimum RT width when converting a ROI to a box
            min_mz_width: minimum RT width when converting a ROI to a box
            advanced_params: an [vimms.Controller.base.AdvancedParams][] object that contains
                             advanced parameters to control the mass spec. If left to None,
                             default values will be used.
            register_all_roi: whether to register all ROIs or not
            scoring_params: a dictionary of parameters used when calculating scores
            exclusion_method: an instance of [vimms.Exclusion.TopNExclusion][] or its subclasses,
                              used to describe how to perform dynamic exclusion so that precursors
                              that have been fragmented are not fragmented again.
            exclusion_t_0: parameter for WeightedDEW exclusion (refer to paper for details).
        """
        super().__init__(ionisation_mode,
                         isolation_width,
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
                         exclusion_t_0=exclusion_t_0)

        self.roi_builder = RoiBuilder(roi_params, smartroi_params=smartroi_params, grid=grid)
        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        self.grid = grid  # helps us understand previous RoIs
        self.register_all_roi = register_all_roi

        self.scoring_params = scoring_params
        self.dda_weight = scoring_params['dda_weight']
        self.overlap_weight = scoring_params['overlap_weight']
        self.smartroi_weight = scoring_params['smartroi_weight']
        self.smartroi_score_add = scoring_params['smartroi_score_add']

    def update_state_after_scan(self, scan):
        super().update_state_after_scan(scan)
        self.grid.send_training_data(scan)

    def _scale(self, scores):
        """
        Scale scores by its maximum value so it goes from 0 to 1

        Args:
            scores: a numpy array of scores

        Returns: the scaled score array

        """
        if len(scores) > 0 and max(scores) > 0:
            scores = scores / max(scores)
        return scores

    def _get_scores(self):
        non_overlaps = self._overlap_scores()
        if self.roi_builder.roi_type == ROI_TYPE_SMART:  # smart ROI scoring
            smartroi_scores = self._smartroi_filter()
            dda_scores = self._log_roi_intensities() * self._min_intensity_filter()

            if self.smartroi_score_add:  # add the scores
                dda_scores = self._scale(dda_scores)
                final_scores = (self.dda_weight * dda_scores) + (
                        self.smartroi_weight * smartroi_scores) + (
                                       self.overlap_weight * non_overlaps)

            else:
                # multiply them, this might not work well because a lot of
                # the smartroi scores are 0s
                final_scores = dda_scores * smartroi_scores * non_overlaps

        else:  # normal ROI
            dda_scores = self._get_dda_scores()
            final_scores = dda_scores * non_overlaps

        # print(final_scores)
        return self._get_top_N_scores(final_scores)

    def after_injection_cleanup(self):
        self.grid.update_after_injection()


class NonOverlapController(GridController):
    """
    A controller that implements the `non-overlapping` idea to determine how regions-of-interests
    should be fragmented across injections.
    """
    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        non_overlaps = np.array(
            [self.grid.non_overlap(
                r.to_box(self.min_rt_width, self.min_mz_width,
                         rt_shift=(-fn(r)[0]))) for
                r in self.roi_builder.live_roi])
        return non_overlaps


class IntensityNonOverlapController(GridController):
    """
    A variant of the non-overlap controller but it takes into account intensity changes.
    """
    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        scores = np.log([self.grid.intensity_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width,
                     rt_shift=(-fn(r)[0])),
            self.roi_builder.current_roi_intensities[i],
            self.scoring_params) for i, r in
            enumerate(self.roi_builder.live_roi)])
        return scores


class FlexibleNonOverlapController(GridController):
    """
    TODO: this class can probably be removed.
    """
    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 grid,
                 smartroi_params=None,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0,
                 min_rt_width=0.01,
                 min_mz_width=0.00001,
                 advanced_params=None,
                 register_all_roi=False,
                 scoring_params=GRID_CONTROLLER_SCORING_PARAMS,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None):  # weighted dew parameters
        super().__init__(
            ionisation_mode,
            isolation_width,
            N,
            mz_tol,
            rt_tol,
            min_ms1_intensity,
            roi_params,
            grid,
            smartroi_params=smartroi_params,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            ms1_shift=ms1_shift,
            min_rt_width=min_rt_width,
            min_mz_width=min_mz_width,
            advanced_params=advanced_params,
            register_all_roi=register_all_roi,
            scoring_params=scoring_params,
            exclusion_method=exclusion_method,
            exclusion_t_0=exclusion_t_0)
        self.scoring_params = scoring_params
        if self.scoring_params['theta3'] != 0 and self.register_all_roi is False:
            print('Warning: register_all_roi should be set to True id theta3 is not 0')

    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        scores = [self.grid.flexible_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width,
                     rt_shift=(-fn(r)[0])),
            self.roi_builder.current_roi_intensities[i],
            self.scoring_params) for i, r in
            enumerate(self.roi_builder.live_roi)]
        return scores


class CaseControlNonOverlapController(GridController):
    """
    Case-control non-overlap controller
    """
    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 grid,
                 smartroi_params=None,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0,
                 min_rt_width=0.01,
                 min_mz_width=0.00001,
                 advanced_params=None,
                 register_all_roi=False,
                 scoring_params=GRID_CONTROLLER_SCORING_PARAMS,
                 exclusion_method=ROI_EXCLUSION_DEW,
                 exclusion_t_0=None):
        super().__init__(
            ionisation_mode,
            isolation_width,
            N,
            mz_tol,
            rt_tol,
            min_ms1_intensity,
            roi_params,
            grid,
            smartroi_params=smartroi_params,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            ms1_shift=ms1_shift,
            min_rt_width=min_rt_width,
            min_mz_width=min_mz_width,
            advanced_params=advanced_params,
            register_all_roi=register_all_roi,
            scoring_params=scoring_params,
            exclusion_method=exclusion_method,
            exclusion_t_0=exclusion_t_0)
        self.scoring_params = scoring_params
        if self.scoring_params['theta3'] != 0 and self.register_all_roi is False:
            print('Warning: register_all_roi should be set to True id theta3 is not 0')

    def _get_scores(self):
        fn = self.grid.get_estimator()
        scores = [self.grid.case_control_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width,
                     rt_shift=(-fn(r)[0])), self.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.live_roi)]
        return self._get_top_N_scores(scores * self._score_filters())


class TopNBoxRoiController2(GridController):
    """
    TODO: This class can probably be removed too?
    """
    def __init__(self,
                 ionisation_mode,
                 isolation_width,
                 N,
                 mz_tol,
                 rt_tol,
                 min_ms1_intensity,
                 roi_params,
                 grid,
                 smartroi_params=None,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0,
                 min_rt_width=0.01,
                 min_mz_width=0.00001,
                 advanced_params=None,
                 boxes_params=None,
                 boxes=None,
                 boxes_intensity=None,
                 boxes_pvalues=None,
                 box_min_rt_width=0.01,
                 box_min_mz_width=0.01):
        self.boxes_params = boxes_params
        self.boxes = boxes
        # the intensity the boxes have been fragmented at before
        self.boxes_intensity = boxes_intensity
        self.boxes_pvalues = boxes_pvalues
        self.box_min_rt_width = box_min_rt_width
        self.box_min_mz_width = box_min_mz_width
        super().__init__(
            ionisation_mode,
            isolation_width,
            N,
            mz_tol,
            rt_tol,
            min_ms1_intensity,
            roi_params,
            grid,
            smartroi_params=smartroi_params,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            ms1_shift=ms1_shift,
            min_rt_width=min_rt_width,
            min_mz_width=min_mz_width,
            advanced_params=advanced_params)

    def _get_scores(self):
        if self.boxes is not None:
            # calculate dda stuff
            log_intensities = np.log(self.roi_builder.current_roi_intensities)
            intensity_filter = (np.array(
                self.roi_builder.current_roi_intensities) >
                                self.min_ms1_intensity)
            time_filter = (1 - np.array(
                self.roi_builder.live_roi_fragmented).astype(int))
            time_filter[time_filter == 0] = (
                    (self.scan_to_process.rt -
                     np.array(self.roi_builder.live_roi_last_rt)[
                         time_filter == 0]) > self.rt_tol)
            # calculate overlap stuff
            initial_scores = []
            copy_boxes = deepcopy(self.boxes)
            for box in copy_boxes:
                box.pt2.x = min(box.pt2.x, max(self.last_ms1_rt, box.pt1.x))
            prev_intensity = np.maximum(np.log(np.array(self.boxes_intensity)),
                                        [0 for i in self.boxes_intensity])
            box_fragmented = (np.array(self.boxes_intensity) == 0) * 1
            for i in range(len(log_intensities)):
                overlaps = np.array(
                    self.roi_builder.live_roi[i].get_boxes_overlap(
                        copy_boxes, self.box_min_rt_width,
                        self.box_min_mz_width))
                # new peaks not in list of boxes
                new_peaks_score = max(
                    0, (1 - sum(overlaps))) * log_intensities[i]
                # previously fragmented peaks
                old_peaks_score1 = sum(
                    overlaps * (log_intensities[i] - prev_intensity) * (
                            1 - box_fragmented))
                # peaks seen before, but not fragmented
                old_peaks_score2 = sum(
                    overlaps * log_intensities[i] * box_fragmented)
                if self.boxes_pvalues is not None:
                    # based on p values, previously fragmented
                    p_value_scores1 = sum(
                        overlaps * (log_intensities[i] - prev_intensity) * (
                                1 - np.array(self.boxes_pvalues)))
                    # based on p values, not previously fragmented
                    p_value_scores2 = sum(overlaps * log_intensities[i] * (
                            1 - np.array(self.boxes_pvalues)))
                # get the score
                score = self.boxes_params['theta1'] * new_peaks_score
                score += self.boxes_params['theta2'] * old_peaks_score1
                score += self.boxes_params['theta3'] * old_peaks_score2
                if self.boxes_pvalues is not None:
                    score += self.boxes_params['theta4'] * p_value_scores1
                    score += self.boxes_params['theta5'] * p_value_scores2
                score *= time_filter[i]
                # check intensity meets minimal requirement
                score *= intensity_filter
                score *= (score > self.boxes_params[
                    'min_score'])  # check meets min score
                initial_scores.append(score[0])
            initial_scores = np.array(initial_scores)
        else:
            initial_scores = self._get_dda_scores()

        scores = self._get_top_N_scores(initial_scores)
        return scores
