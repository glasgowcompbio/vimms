from abc import abstractmethod

import numpy as np
from loguru import logger

from vimms.Controller.roi import RoiController


class GridController(RoiController):

    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, grid, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 ms1_shift=0, min_rt_width=0.01, min_mz_width=0.01,
                 params=None):
        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
            min_roi_length, N, rt_tol=rt_tol, min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            length_units=length_units, ms1_shift=ms1_shift, params=params
        )

        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        self.grid = grid  # helps us understand previous RoIs

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
    def _get_scores(self):
        fn = self.grid.get_estimator()
        scores = np.log([self.grid.intensity_non_overlap(
            r.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-fn(r)[0])), self.current_roi_intensities[i]) for
                         i, r in enumerate(self.live_roi)])
        return self._get_top_N_scores(scores * self._score_filters())
