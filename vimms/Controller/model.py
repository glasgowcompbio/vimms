import numpy as np

from vimms.Controller import RoiController


class ModelRoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol,
                 min_ms1_intensity, min_roi_intensity,
                 min_roi_length, boxes, p_values, N=None, rt_tol=10,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0, advanced_params=None,
                 box_min_rt_width=0.01, box_min_mz_width=0.01):
        self.boxes = boxes
        self.p_values = np.array(p_values)
        self.box_min_rt_width = box_min_rt_width
        self.box_min_mz_width = box_min_mz_width

        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
            min_roi_intensity, min_roi_length, N, rt_tol=rt_tol,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            ms1_shift=ms1_shift, advanced_params=advanced_params)


class FullPrioritisationModelRoiController(ModelRoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol,
                 min_ms1_intensity, min_roi_intensity,
                 min_roi_length, boxes, p_values, N=None, rt_tol=10,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0, advanced_params=None,
                 box_min_rt_width=0.01, box_min_mz_width=0.01):
        super().__init__(ionisation_mode, isolation_width, mz_tol,
                         min_ms1_intensity, min_roi_intensity,
                         min_roi_length, boxes, p_values, N, rt_tol,
                         min_roi_length_for_fragmentation,
                         ms1_shift, advanced_params,
                         box_min_rt_width, box_min_mz_width)

        self.p_values_order = np.argsort(
            -np.array(self.p_values))  # this is highest to lowest

    def _get_scores(self):
        dda_scores = self._get_dda_scores()
        overlap_scores = []
        for i in range(len(dda_scores)):
            overlaps = np.array(
                self.live_roi[i].get_boxes_overlap(
                    self.boxes, self.box_min_rt_width, self.box_min_mz_width))
            overlap_scores.append(overlaps * self.p_values_order)
        initial_scores = dda_scores * overlap_scores
        scores = self._get_top_N_scores(initial_scores)
        return scores


class TopNBoxModelRoiController(ModelRoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol,
                 min_ms1_intensity, min_roi_intensity,
                 min_roi_length, boxes, p_values, N=None, rt_tol=10,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0, advanced_params=None,
                 box_min_rt_width=0.01, box_min_mz_width=0.01):
        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
            min_roi_intensity, min_roi_length, boxes, p_values, N, rt_tol,
            min_roi_length_for_fragmentation, ms1_shift,
            advanced_params, box_min_rt_width, box_min_mz_width)

        # this is highest to lowest
        self.p_values_order = np.argsort(-np.array(self.p_values))

    def _get_scores(self):
        dda_scores = self._get_dda_scores()
        overlap_scores = []
        for i in range(len(dda_scores)):
            overlaps = np.array(self.live_roi[i].get_boxes_overlap(
                self.boxes, self.box_min_rt_width, self.box_min_mz_width))
            max_pvalue = self.p_values[np.where(overlaps > 0.0)]
            overlap_scores.append(1 + 1 - max_pvalue)
        initial_scores = dda_scores * overlap_scores
        scores = self._get_top_N_scores(initial_scores)
        return scores
