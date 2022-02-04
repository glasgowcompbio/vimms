from copy import deepcopy

import numpy as np
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Common import ROI_EXCLUSION_DEW, ROI_EXCLUSION_WEIGHTED_DEW, MZ_UNITS_PPM
from vimms.Controller.topN import TopNController
from vimms.Exclusion import MinIntensityFilter, LengthFilter, SmartROIFilter, WeightedDEWExclusion, DEWFilter, \
    WeightedDEWFilter
from vimms.Roi import RoiBuilder


class RoiController(TopNController):
    """
    An ROI based controller with multiple options
    """

    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 params=None, exclusion_method=ROI_EXCLUSION_DEW, exclusion_t_0=None, mz_units=MZ_UNITS_PPM):
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift=ms1_shift,
                         params=params)
        self.min_roi_length_for_fragmentation = min_roi_length_for_fragmentation
        self.roi_builder = RoiBuilder(mz_tol, rt_tol, min_roi_intensity, min_roi_length,
                                      length_units=length_units, roi_type=RoiBuilder.ROI_TYPE_NORMAL, mz_units=mz_units)

        self.exclusion_method = exclusion_method
        assert self.exclusion_method in [ROI_EXCLUSION_DEW, ROI_EXCLUSION_WEIGHTED_DEW]
        if self.exclusion_method == ROI_EXCLUSION_WEIGHTED_DEW:
            assert exclusion_t_0 is not None, 'Must be a number'
            assert exclusion_t_0 < rt_tol, 'Impossible combination'
            self.exclusion = WeightedDEWExclusion(rt_tol, exclusion_t_0)

        self.exclusion_t_0 = exclusion_t_0

    def schedule_ms1(self, new_tasks):
        ms1_scan_params = self.get_ms1_scan_params()
        self.current_task_id += 1
        self.next_processed_scan_id = self.current_task_id
        logger.debug('Created the next processed scan %d' % (self.next_processed_scan_id))
        new_tasks.append(ms1_scan_params)

    class MS2Scheduler():
        def __init__(self, parent):
            self.parent = parent
            self.fragmented_count = 0

        def schedule_ms2s(self, new_tasks, ms2_tasks, mz, intensity):
            precursor_scan_id = self.parent.scan_to_process.scan_id
            dda_scan_params = self.parent.get_ms2_scan_params(mz, intensity, precursor_scan_id,
                                                              self.parent.isolation_width, self.parent.mz_tol,
                                                              self.parent.rt_tol)
            new_tasks.append(dda_scan_params)
            ms2_tasks.append(dda_scan_params)
            self.parent.current_task_id += 1
            self.fragmented_count += 1

    def _process_scan(self, scan):
        if self.scan_to_process is not None:
            # keep growing ROIs if we encounter a new ms1 scan
            self.roi_builder.update_roi(scan)
            new_tasks, ms2_tasks = [], []
            rt = self.scan_to_process.rt

            done_ms1, ms2s, scores = False, self.MS2Scheduler(self), self._get_scores()
            for i in np.argsort(scores)[::-1]:
                if scores[i] <= 0:  # stopping criteria is done based on the scores
                    logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                    break

                mz, intensity, roi_id = self.roi_builder.get_mz_intensity(i)
                ms2s.schedule_ms2s(new_tasks, ms2_tasks, mz, intensity)
                self.roi_builder.set_fragmented(self.current_task_id, i, roi_id, rt, intensity)

                if ms2s.fragmented_count == self.N - self.ms1_shift:
                    self.schedule_ms1(new_tasks)
                    done_ms1 = True

            # if no ms1 has been added, then add at the end
            if not done_ms1: self.schedule_ms1(new_tasks)

            # create new exclusion items based on the scheduled ms2 tasks
            if self.exclusion is not None:
                self.exclusion.update(self.scan_to_process, ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
            return new_tasks

        elif scan.ms_level == 2:  # add ms2 scans to Rois
            self.roi_builder.add_scan_to_roi(scan)
            return []

    ####################################################################################################################
    # Scoring functions
    ####################################################################################################################

    def _log_roi_intensities(self):
        return np.log(self.roi_builder.current_roi_intensities)

    def _min_intensity_filter(self):
        f = MinIntensityFilter(self.min_ms1_intensity)
        return f.filter(self.roi_builder.current_roi_intensities)

    def _time_filter(self):
        if self.exclusion_method == ROI_EXCLUSION_DEW:
            f = DEWFilter(self.rt_tol)
            return f.filter(self.scan_to_process.rt, self.roi_builder.live_roi_last_rt)
        elif self.exclusion_method == ROI_EXCLUSION_WEIGHTED_DEW:
            f = WeightedDEWFilter(self.exclusion)
            return f.filter(self.scan_to_process.rt, self.roi_builder.live_roi_last_rt, self.roi_builder.live_roi)

    def _length_filter(self):
        f = LengthFilter(self.min_roi_length_for_fragmentation)
        return f.filter(self.roi_builder.current_roi_length)

    def _smartroi_filter(self):
        f = SmartROIFilter()
        return f.filter(self.roi_builder.live_roi)

    def _score_filters(self):
        return self._min_intensity_filter() * self._time_filter() * self._length_filter()

    def _get_dda_scores(self):
        return self._log_roi_intensities() * self._score_filters()

    def _get_top_N_scores(self, scores):
        if len(scores) > self.N:  # number of fragmentation events filter
            scores[scores.argsort()[:(len(scores) - self.N)]] = 0
        return scores

    def _get_scores(self):
        NotImplementedError()


class TopN_SmartRoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, rt_tol=10, min_roi_length_for_fragmentation=1,
                 reset_length_seconds=100, intensity_increase_factor=2, length_units="scans",
                 drop_perc=0.01, ms1_shift=0, params=None):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift,
                         params=params)
        self.roi_builder = RoiBuilder(mz_tol, rt_tol, min_roi_intensity, min_roi_length,
                                      reset_length_seconds=reset_length_seconds,
                                      intensity_increase_factor=intensity_increase_factor, drop_perc=drop_perc,
                                      length_units=length_units, roi_type=RoiBuilder.ROI_TYPE_SMART)

    def _get_dda_scores(self):
        return self._log_roi_intensities() * self._min_intensity_filter() * self._smartroi_filter()

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class TopN_RoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, params=None, exclusion_method=ROI_EXCLUSION_DEW, exclusion_t_0=None):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, params=params,
                         exclusion_method=exclusion_method, exclusion_t_0=exclusion_t_0)

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class TopNBoxRoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, boxes_params=None, boxes=None, boxes_intensity=None, boxes_pvalues=None, N=None,
                 rt_tol=10,
                 min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0, params=None,
                 box_min_rt_width=0.01, box_min_mz_width=0.01):

        self.boxes_params = boxes_params
        self.boxes = boxes
        self.boxes_intensity = boxes_intensity  # the intensity the boxes have been fragmented at before
        self.boxes_pvalues = boxes_pvalues
        self.box_min_rt_width = box_min_rt_width
        self.box_min_mz_width = box_min_mz_width
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, params=params)

    def _get_scores(self):
        if self.boxes is not None:
            # calculate dda stuff
            log_intensities = self._log_roi_intensities()
            intensity_filter = self._min_intensity_filter()
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


########################################################################################################################
# Other Functions
########################################################################################################################


def get_peak_status(mzs, rt, boxes, scores, model_scores=None, box_mz_tol=10):
    if model_scores is not None:
        list1 = list(filter(lambda x: x[0].rt_range_in_seconds[0] <= rt <= x[0].rt_range_in_seconds[1],
                            zip(boxes, scores, model_scores)))
        model_score_status = []
    else:
        list1 = list(filter(lambda x: x[0].rt_range_in_seconds[0] <= rt <= x[0].rt_range_in_seconds[1],
                            zip(boxes, scores)))
        model_score_status = None
    peak_status = []
    for mz in mzs:
        list2 = list(filter(lambda x: x[0].mz_range[0] * (1 - box_mz_tol / 1e6) <= mz <=
                                      x[0].mz_range[1] * (1 + box_mz_tol / 1e6), list1))
        if list2 == []:
            peak_status.append(-1)
            if model_scores is not None:
                model_score_status.append(1)
        else:
            scores = [x[1] for x in list2]
            peak_status.append(min(scores))
            if model_scores is not None:
                m_scores = [x[2] for x in list2]
                model_score_status.append(max(m_scores))
    return peak_status, model_score_status


def get_box_intensity(mzml_file, boxes):
    intensities = [0 for i in range(len(boxes))]
    mzs = [None for i in range(len(boxes))]
    box_ids = range(len(boxes))
    mz_file = MZMLFile(mzml_file)
    for scan in mz_file.scans:
        if scan.ms_level == 2:
            continue
        rt = scan.rt_in_seconds
        zipped_boxes = list(
            filter(lambda x: x[0].rt_range_in_seconds[0] <= rt <= x[0].rt_range_in_seconds[1], zip(boxes, box_ids)))
        if not zipped_boxes:
            continue
        for mzint in scan.peaks:
            mz = mzint[0]
            sub_boxes = list(filter(lambda x: x[0].mz_range[0] <= mz <= x[0].mz_range[1], zipped_boxes))
            if not sub_boxes:
                continue
            for box in sub_boxes:
                intensity = mzint[1]
                if intensity > intensities[box[1]]:
                    intensities[box[1]] = intensity
                    mzs[box[1]] = mz
    return intensities, mzs
