from vimms.Controller.base import Controller
from vimms.Roi import match, Roi, SmartRoi

import numpy as np
import bisect
from loguru import logger


# For controller that consider doing an ms1 scans because of RT drift uncertainty, do this by forcing ms2 scores below 0


class BaseController(Controller):
    def __init__(self, ionisation_mode, ms1_shift=0, add_extra_ms1=False, params=None):
        super().__init__(params=params)

        self.N = None
        self.ionisation_mode = ionisation_mode
        self.add_extra_ms1 = add_extra_ms1
        self.ms1_shift = ms1_shift

    def _process_scan(self, scan):
        new_tasks = []  # this gets updated in _get_ms2_scan
        self.fragmented_count = 0  # this gets updated in _get_ms2_scan
        if self.scan_to_process is not None:
            self._update_rois(scan)  # TODO: check that the exclusion list isnt meant to update here
            self.current_scores = self._get_scores(scan)  # this is the individual scoring system
            idx = np.argsort(self.current_scores)[::-1]

            self.done_ms1 = False

            for i in idx:
                ms2_tasks = self._get_next_scans(i)  # gets ms2 scan, returns None if ms2 scan not chosen
                # in above we update the fragmentation_count and things like N
                new_tasks.extend(ms2_tasks)
                self._update_exclusion(ms2_tasks)  # updates exlcusion list
                # consider adding ms1
                if len(ms2_tasks) == 0 and self.done_ms1 is False:
                    ms1_task = self._get_ms1_scan()
                    new_tasks.append(ms1_task)
                if self.fragmented_count == self.N - self.ms1_shift:
                    ms1_task = self._get_ms1_scan()
                    new_tasks.append(ms1_task)
                if self.add_extra_ms1 and self.done_ms1 is False:
                    ms1_task = self._get_ms1_scan()
                    new_tasks.append(ms1_task)

                # break if we have no ms2_tasks
                if len(ms2_tasks) == 0:
                    break  # stop looping through scans
                # update scores if we havent taken break. Generally does nothing, people useful for some future methods
                self._update_scores(ms2_tasks)
            self.scan_to_process = None
        return new_tasks

    def _get_ms1_scan(self):
        ms1_scan_params = self.get_ms1_scan_params()
        self.current_task_id += 1
        self.next_processed_scan_id = self.current_task_id
        logger.debug('Created the next processed scan %d' % (self.next_processed_scan_id))
        self.done_ms1 = True
        return ms1_scan_params

    def _update_rois(self, scan):
        NotImplementedError()

    def _get_scores(self, scan):
        return []

    def _get_current_values(self, i):
        return NotImplementedError()

    def _get_next_scans(self, i):
        new_tasks = []
        mz, intensity, score = self._get_current_values(i)
        # TODO: check score, return [] if less than 0
        # TODO else: add a new scan
        self.fragmented_count += len(new_tasks)
        self.current_task_id += len(new_tasks)
        return new_tasks

    def _update_exclusion(self, ms2_tasks):
        NotImplementedError()  # TODO: To be implemented

    def _update_scores(self, new_tasks):
        pass

    def _get_topn_scores(self, scores, N):
        if len(scores) > N:  # number of fragmentation events filter
            scores[scores.argsort()[:(len(scores) - N)]] = 0
        return scores


class PrecursorController2(BaseController):
    def __init__(self, ionisation_mode, params=None):
        super().__init__(ionisation_mode=ionisation_mode, params=params)

    def _get_current_values(self, i):
        mz = self.scan_to_process.mzs[i]
        intensity = self.scan_to_process.intensities[i]
        score = self.current_scores[i]
        return mz, intensity, score


class RoiController2(BaseController):
    def __init__(self, ionisation_mode, mz_tol, min_roi_intensity, min_roi_length, min_roi_length_for_fragmentation,
                 length_units, params=None):
        super().__init__(ionisation_mode=ionisation_mode, params=params)

        # ROI stuff
        self.mz_tol = mz_tol
        self.min_roi_intensity = min_roi_intensity
        self.mz_units = 'ppm'
        self.min_roi_length = min_roi_length
        self.min_roi_length_for_fragmentation = min_roi_length_for_fragmentation
        self.length_units = length_units

        # Create ROI
        self.live_roi = []
        self.dead_roi = []
        self.junk_roi = []
        self.live_roi_fragmented = []
        self.live_roi_last_rt = []  # last fragmentation time of ROI

    def _get_current_values(self, i):
        mz = self.live_roi[i].mz_list[-1]
        intensity = self.live_roi[i].intensity_list[-1]
        score = self.current_scores[i]
        return mz, intensity, score

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

                        new_roi = self._create_new_roi(self, mz, current_ms1_scan_rt, intensity)
                        bisect.insort_right(self.live_roi, new_roi)
                        self.live_roi_fragmented.insert(self.live_roi.index(new_roi), False)
                        self.live_roi_last_rt.insert(self.live_roi.index(new_roi), None)

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

    def _create_new_roi(self, mz, current_ms1_scan_rt, intensity):
        new_roi = Roi(mz, current_ms1_scan_rt, intensity)
        return new_roi


class SmartRoiController2(RoiController2):
    def __init__(self, ionisation_mode, mz_tol, min_roi_intensity, min_roi_length, min_roi_length_for_fragmentation,
             length_units, reset_length_seconds, intensity_increase_factor, rt_tol, drop_perc, params=None):
        super().__init__(ionisation_mode=ionisation_mode, mz_tol=mz_tol, min_roi_intensity=min_roi_intensity,
                         min_roi_length=min_roi_length,
                         min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
                         length_units=length_units, params=params)

        self.reset_length_seconds = reset_length_seconds
        self.intensity_increase_factor = intensity_increase_factor
        self.rt_tol = rt_tol
        self.drop_perc = drop_perc

    def _create_new_roi(self, mz, current_ms1_scan_rt, intensity):  # this is overwritten here to update the ROI type
        new_roi = SmartRoi(mz, current_ms1_scan_rt, intensity, self.min_roi_length_for_fragmentation,
                 self.reset_length_seconds, self.intensity_increase_factor, self.rt_tol,
                 drop_perc=self.drop_perc)
        return new_roi

# Precursor Controllers

class TopNController2(PrecursorController2):
    NotImplementedError()

    def _get_scores(self, scan):
        NotImplementedError()


class WeightDewController2(PrecursorController2):
    NotImplementedError()

    def _get_scores(self, scan):
        NotImplementedError()


class TargettedController2(PrecursorController2):
    NotImplementedError()

    def _get_scores(self, scan):
        NotImplementedError()

# Roi Controllers

class TopNRoiController2(RoiController2):
    NotImplementedError()

    def _get_scores(self, scan):
        NotImplementedError()


# Smart Roi Controllers

class TopNSmartRoiController2(SmartRoiController2):
    NotImplementedError()

    def _get_scores(self, scan):
        NotImplementedError()
