from abc import abstractmethod

import numpy as np
from loguru import logger
from copy import deepcopy
import bisect

from vimms.Controller.roi import RoiController
from vimms.Roi import match, Roi


class GridController(RoiController):

    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False):
        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
            min_roi_length, N, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            length_units=length_units, ms1_shift=ms1_shift, params=params
        )

        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        self.grid = grid  # helps us understand previous RoIs
        self.register_all_roi = register_all_roi

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
        if self.scan_to_process is None: return []
        # keep growing ROIs if we encounter a new ms1 scan
        self._update_roi(scan)
        self.current_roi_mzs = [roi.mz_list[-1] for roi in self.live_roi]
        self.current_roi_intensities = [roi.intensity_list[-1] for roi in self.live_roi]
        new_tasks, ms2_tasks = [], []
        rt = self.scan_to_process.rt

        # FIXME: only the 'scans' mode seems to work on the real mass spec (IAPI), why??
        if self.length_units == "scans":
            self.current_roi_length = np.array([roi.n for roi in self.live_roi])
        else:
            self.current_roi_length = np.array([roi.length_in_seconds for roi in self.live_roi])

        done_ms1, ms2s, scores = False, self.MS2Scheduler(self), self._get_scores()
        for i in np.argsort(scores)[::-1]:
            if scores[i] <= 0:  # stopping criteria is done based on the scores
                logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                break

            mz, intensity = self.current_roi_mzs[i], self.current_roi_intensities[i]
            self.live_roi_fragmented[i] = True
            self.live_roi_last_rt[i] = rt
            if self.register_all_roi is False:
                self.grid.register_roi(self.live_roi[i])
            self.live_roi[i].max_fragmentation_intensity = max(self.live_roi[i].max_fragmentation_intensity, intensity)

            ms2s.schedule_ms2s(new_tasks, ms2_tasks, mz, intensity)
            if ms2s.fragmented_count == self.N - self.ms1_shift:
                self.schedule_ms1(new_tasks)
                done_ms1 = True

        # if no ms1 has been added, then add at the end
        if not done_ms1: self.schedule_ms1(new_tasks)

        # create new exclusion items based on the scheduled ms2 tasks
        self.exclusion.update(self.scan_to_process, ms2_tasks)

        self.scan_to_process = None  # set this ms1 scan as has been processed

        return new_tasks

    def _update_roi(self, new_scan):
        if new_scan.ms_level == 1:
            order = np.argsort(self.live_roi)
            self.live_roi.sort()
            self.live_roi_fragmented = np.array(self.live_roi_fragmented)[order].tolist()
            self.live_roi_last_rt = np.array(self.live_roi_last_rt)[order].tolist()
            current_ms1_scan_rt = new_scan.rt
            not_grew = set(self.live_roi)
            for idx in range(len(new_scan.intensities)):
                intensity = new_scan.intensities[idx]
                mz = new_scan.mzs[idx]
                if intensity >= self.min_roi_intensity:
                    match_roi = match(Roi(mz, 0, 0), self.live_roi, self.mz_tol, mz_units=self.mz_units)
                    if match_roi:
                        match_roi.add(mz, current_ms1_scan_rt, intensity)
                        if match_roi in not_grew:
                            not_grew.remove(match_roi)
                    else:
                        new_roi = Roi(mz, current_ms1_scan_rt, intensity, self.roi_id_counter)
                        self.roi_id_counter += 1
                        bisect.insort_right(self.live_roi, new_roi)
                        self.live_roi_fragmented.insert(self.live_roi.index(new_roi), False)
                        self.live_roi_last_rt.insert(self.live_roi.index(new_roi), None)
                        if self.register_all_roi:
                            self.grid.register_roi(new_roi)

            for roi in not_grew:
                if self.length_units == "scans":
                    if roi.n >= self.min_roi_length:
                        self.dead_roi.append(roi)
                    else:
                        self.junk_roi.append(roi)
                else:
                    if roi.length_in_seconds >= self.min_roi_length:
                        self.dead_roi.append(roi)
                    else:
                        self.junk_roi.append(roi)

                pos = self.live_roi.index(roi)
                del self.live_roi[pos]
                del self.live_roi_fragmented[pos]
                del self.live_roi_last_rt[pos]

    def update_state_after_scan(self, scan):
        super().update_state_after_scan(scan)
        self.grid.send_training_data(scan)

    @abstractmethod
    def _get_scores(self):
        pass

    def after_injection_cleanup(self):
        self.grid.update_after_injection()


class NonOverlapController(GridController):
    def _get_scores(self):
        fn = self.grid.get_estimator()
        non_overlaps = [self.grid.non_overlap(r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0]))) for
                        r in self.live_roi]
        return self._get_top_N_scores(self._get_dda_scores() * non_overlaps)


class IntensityNonOverlapController(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False, scoring_params={'theta1': 1}):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, grid, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params, register_all_roi=register_all_roi)
        self.scoring_params = scoring_params

    def _get_scores(self):
        fn = self.grid.get_estimator()
        scores = np.log([self.grid.intensity_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])), self.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.live_roi)])
        return self._get_top_N_scores(scores * self._score_filters())


class FlexibleNonOverlapController(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
             min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
             ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
             params=None, register_all_roi=False, scoring_params={'theta1': 1, 'theta2': 0, 'theta3': 0}):

        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
             min_roi_length, N, grid, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params, register_all_roi=register_all_roi)
        self.scoring_params = scoring_params
        if self.scoring_params['theta3'] != 0 and self.register_all_roi is False:
            print('Warning: register_all_roi should be set to True id theta3 is not 0')

    def _get_scores(self):
        fn = self.grid.get_estimator()
        scores = [self.grid.flexible_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])), self.current_roi_intensities[i],
            self.scoring_params) for i, r in enumerate(self.live_roi)]
        return self._get_top_N_scores(scores * self._score_filters())


class CaseControlNonOverlapController(GridController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None, register_all_roi=False,
                 scoring_params={'theta1': 1, 'theta2': 0, 'theta3': 0, 'theta4': 0}):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, grid, rt_tol=rt_tol,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, ms1_shift=ms1_shift, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width, params=params, register_all_roi=register_all_roi)
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
            log_intensities = np.log(self.current_roi_intensities)
            intensity_filter = (np.array(self.current_roi_intensities) > self.min_ms1_intensity)
            time_filter = (1 - np.array(self.live_roi_fragmented).astype(int))
            time_filter[time_filter == 0] = (
                    (self.scan_to_process.rt - np.array(self.live_roi_last_rt)[time_filter == 0]) > self.rt_tol)
            # calculate overlap stuff
            initial_scores = []
            copy_boxes = deepcopy(self.boxes)
            for box in copy_boxes:
                box.pt2.x = min(box.pt2.x, max(self.last_ms1_rt, box.pt1.x))
            prev_intensity = np.maximum(np.log(np.array(self.boxes_intensity)), [0 for i in self.boxes_intensity])
            box_fragmented = (np.array(self.boxes_intensity) == 0) * 1
            for i in range(len(log_intensities)):
                overlaps = np.array(self.live_roi[i].get_boxes_overlap(copy_boxes, self.box_min_rt_width,
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

