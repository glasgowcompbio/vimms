from copy import deepcopy

import numpy as np

from vimms.Common import ROI_EXCLUSION_DEW, GRID_CONTROLLER_SCORING_PARAMS
from vimms.Controller.roi import RoiController, RoiBuilder


class GridController(RoiController):

    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False, scoring_params=GRID_CONTROLLER_SCORING_PARAMS,
                 roi_type=RoiBuilder.ROI_TYPE_NORMAL, reset_length_seconds=1e6,  # smartroi parameters
                 intensity_increase_factor=10, drop_perc=0.1 / 100,  # smartroi parameters
                 exclusion_method=ROI_EXCLUSION_DEW, exclusion_t_0=None):  # weighted dew parameters
        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
            min_roi_length, N, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            length_units=length_units, ms1_shift=ms1_shift, params=params,
            exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0
        )
        self.roi_builder = RoiBuilder(mz_tol, rt_tol, min_roi_intensity, min_roi_length,
                                      min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                                      reset_length_seconds=reset_length_seconds,
                                      intensity_increase_factor=intensity_increase_factor,
                                      drop_perc=drop_perc,
                                      length_units=length_units, roi_type=roi_type,
                                      grid=grid, register_all_roi=register_all_roi)

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
        if len(scores) > 0 and max(scores) > 0:
            scores = scores / max(scores)
        return scores

    def _get_scores(self):
        non_overlaps = self._overlap_scores()
        if self.roi_builder.roi_type == RoiBuilder.ROI_TYPE_SMART:  # smart ROI scoring
            smartroi_scores = self._smartroi_filter()
            dda_scores = self._log_roi_intensities() * self._min_intensity_filter()

            if self.smartroi_score_add:  # add the scores
                dda_scores = self._scale(dda_scores)
                final_scores = (self.dda_weight * dda_scores) + (self.smartroi_weight * smartroi_scores) + (
                        self.overlap_weight * non_overlaps)

            else:  # multiply them, this wouldn't work well because a lot of the smartroi scores are 0s
                final_scores = dda_scores * smartroi_scores * non_overlaps

        else:  # normal ROI
            dda_scores = self._get_dda_scores()
            final_scores = dda_scores * non_overlaps

        # print(final_scores)
        return self._get_top_N_scores(final_scores)

    def after_injection_cleanup(self):
        self.grid.update_after_injection()


class NonOverlapController(GridController):
    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        non_overlaps = np.array(
            [self.grid.non_overlap(r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0]))) for
             r in self.roi_builder.live_roi])
        return non_overlaps


class IntensityNonOverlapController(GridController):
    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        scores = np.log([self.grid.intensity_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])),
            self.roi_builder.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.roi_builder.live_roi)])
        return scores


class FlexibleNonOverlapController(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False, scoring_params={'theta1': 1, 'theta2': 0, 'theta3': 0},
                 roi_type=RoiBuilder.ROI_TYPE_NORMAL, reset_length_seconds=1e6,  # smartroi parameters
                 intensity_increase_factor=10, drop_perc=0.1 / 100,  # smartroi parameters
                 exclusion_method=ROI_EXCLUSION_DEW, exclusion_t_0=None):  # weighted dew parameters
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, grid, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params, register_all_roi=register_all_roi,
                         roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                         intensity_increase_factor=intensity_increase_factor, drop_perc=drop_perc,
                         exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
        self.scoring_params = scoring_params
        if self.scoring_params['theta3'] != 0 and self.register_all_roi is False:
            print('Warning: register_all_roi should be set to True id theta3 is not 0')

    def _overlap_scores(self):
        fn = self.grid.get_estimator()
        scores = [self.grid.flexible_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])),
            self.roi_builder.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.roi_builder.live_roi)]
        return scores


class CaseControlNonOverlapController(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False,
                 scoring_params={'theta1': 1, 'theta2': 0, 'theta3': 0, 'theta4': 0},
                 roi_type=RoiBuilder.ROI_TYPE_NORMAL, reset_length_seconds=1e6,  # smartroi parameters
                 intensity_increase_factor=10, drop_perc=0.1 / 100,  # smartroi parameters
                 exclusion_method=ROI_EXCLUSION_DEW, exclusion_t_0=None):  # weighted dew parameters
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, grid, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params, register_all_roi=register_all_roi,
                         roi_type=roi_type, reset_length_seconds=reset_length_seconds,
                         intensity_increase_factor=intensity_increase_factor, drop_perc=drop_perc,
                         exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)
        self.scoring_params = scoring_params
        if self.scoring_params['theta3'] != 0 and self.register_all_roi is False:
            print('Warning: register_all_roi should be set to True id theta3 is not 0')

    def _get_scores(self):
        fn = self.grid.get_estimator()
        scores = [self.grid.case_control_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])), self.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.live_roi)]
        return self._get_top_N_scores(scores * self._score_filters())


class TopNBoxRoiController2(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity, min_roi_length,
                 N, grid, boxes_params=None, boxes=None, boxes_intensity=None, boxes_pvalues=None, rt_tol=10,
                 min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0, min_rt_width=0.01,
                 min_mz_width=0.01, params=None, box_min_rt_width=0.01, box_min_mz_width=0.01):
        self.boxes_params = boxes_params
        self.boxes = boxes
        self.boxes_intensity = boxes_intensity  # the intensity the boxes have been fragmented at before
        self.boxes_pvalues = boxes_pvalues
        self.box_min_rt_width = box_min_rt_width
        self.box_min_mz_width = box_min_mz_width
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity, min_roi_length,
                         N, grid=grid, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params)

    def _get_scores(self):
        if self.boxes is not None:
            # calculate dda stuff
            log_intensities = np.log(self.roi_builder.current_roi_intensities)
            intensity_filter = (np.array(self.roi_builder.current_roi_intensities) > self.min_ms1_intensity)
            time_filter = (1 - np.array(self.roi_builder.live_roi_fragmented).astype(int))
            time_filter[time_filter == 0] = (
                    (self.scan_to_process.rt - np.array(self.roi_builder.live_roi_last_rt)[
                        time_filter == 0]) > self.rt_tol)
            # calculate overlap stuff
            initial_scores = []
            copy_boxes = deepcopy(self.boxes)
            for box in copy_boxes:
                box.pt2.x = min(box.pt2.x, max(self.last_ms1_rt, box.pt1.x))
            prev_intensity = np.maximum(np.log(np.array(self.boxes_intensity)), [0 for i in self.boxes_intensity])
            box_fragmented = (np.array(self.boxes_intensity) == 0) * 1
            for i in range(len(log_intensities)):
                overlaps = np.array(self.roi_builder.live_roi[i].get_boxes_overlap(copy_boxes, self.box_min_rt_width,
                                                                                   self.box_min_mz_width))
                # new peaks not in list of boxes
                new_peaks_score = max(0, (1 - sum(overlaps))) * log_intensities[i]
                # previously fragmented peaks
                old_peaks_score1 = sum(overlaps * (log_intensities[i] - prev_intensity) * (1 - box_fragmented))
                # peaks seen before, but not fragmented
                old_peaks_score2 = sum(overlaps * log_intensities[i] * box_fragmented)
                if self.boxes_pvalues is not None:
                    # based on p values, previously fragmented
                    p_value_scores1 = sum(
                        overlaps * (log_intensities[i] - prev_intensity) * (1 - np.array(self.boxes_pvalues)))
                    # based on p values, not previously fragmented
                    p_value_scores2 = sum(overlaps * log_intensities[i] * (1 - np.array(self.boxes_pvalues)))
                # get the score
                score = self.boxes_params['theta1'] * new_peaks_score
                score += self.boxes_params['theta2'] * old_peaks_score1
                score += self.boxes_params['theta3'] * old_peaks_score2
                if self.boxes_pvalues is not None:
                    score += self.boxes_params['theta4'] * p_value_scores1
                    score += self.boxes_params['theta5'] * p_value_scores2
                score *= time_filter[i]
                score *= intensity_filter  # check intensity meets minimal requirement
                score *= (score > self.boxes_params['min_score'])  # check meets min score
                initial_scores.append(score[0])
            initial_scores = np.array(initial_scores)
        else:
            initial_scores = self._get_dda_scores()

        scores = self._get_top_N_scores(initial_scores)
        return scores
