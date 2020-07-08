import bisect

import numpy as np
from loguru import logger
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from vimms.Controller import TopNController
from vimms.PeakDetector import calculate_window_change
from vimms.Roi import match, Roi, SmartRoi
from vimms.Common import *

from ms2_matching import MZMLFile, load_picked_boxes, map_boxes_to_scans

class RoiController(TopNController):
    """
    An ROI based controller with multiple options
    """

    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 # advanced parameters
                ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                ms1_max_it = DEFAULT_MS1_MAXIT,
                ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                ms2_max_it = DEFAULT_MS2_MAXIT,
                ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_agc_target, ms1_shift,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)

        # ROI stuff
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

    def _process_scan(self, scan):
        # keep growing ROIs if we encounter a new ms1 scan
        self._update_roi(scan)

        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.scan_to_process is not None:
            self.current_roi_mzs = [roi.mz_list[-1] for roi in self.live_roi]
            self.current_roi_intensities = [roi.intensity_list[-1] for roi in self.live_roi]

            # FIXME: only the 'scans' mode seems to work on the real mass spec (IAPI), why??
            if self.length_units == "scans":
                self.current_roi_length = np.array([roi.n for roi in self.live_roi])
            else:
                self.current_roi_length = np.array([roi.length_in_seconds for roi in self.live_roi])

            rt = self.scan_to_process.rt

            # loop over points in decreasing score
            # t0 = time()
            scores = self._get_scores()
            # logger.debug(time()-t0)
            idx = np.argsort(scores)[::-1]
            for i in idx:
                mz = self.current_roi_mzs[i]
                intensity = self.current_roi_intensities[i]

                # stopping criteria is done based on the scores
                if scores[i] <= 0:
                    logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                    break

                # updated fragmented list and times
                self.live_roi_fragmented[i] = True
                self.live_roi_last_rt[i] = rt

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                      self.isolation_width, self.mz_tol, self.rt_tol)
                new_tasks.append(dda_scan_params)

            ms1_scan_params = self.environment.get_default_scan_params()
            ms1_insert_position = max(len(new_tasks) - self.ms1_shift, 0)
            new_tasks.insert(ms1_insert_position, ms1_scan_params)
            num_scans = (len(self.scans[1]) + len(self.scans[2]) + ms1_insert_position + self.pending_tasks)
            self.next_processed_scan_id = num_scans

            # create temp exclusion items
            tasks = new_tasks[(ms1_insert_position + 1):]
            self.temp_exclusion_list = self._update_temp_exclusion_list(tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def update_state_after_scan(self, last_scan):
        # add precursor info based on the current scan produced
        # NOT doing the dynamic exclusion window thing
        self._add_precursor_info(last_scan)

    def reset(self):
        super().reset()
        self.live_roi = []
        self.dead_roi = []
        self.junk_roi = []
        self.live_roi_fragmented = []
        self.live_roi_last_rt = []  # last fragmentation time of ROI

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
                        new_roi = Roi(mz, current_ms1_scan_rt, intensity)
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

    def _get_scores(self):
        NotImplementedError()

    def _get_dda_scores(self):
        scores = np.log(self.current_roi_intensities)  # log intensities
        scores *= (np.log(self.current_roi_intensities) > np.log(self.min_ms1_intensity))  # intensity filter
        time_filter = (1 - np.array(self.live_roi_fragmented).astype(int))
        time_filter[time_filter == 0] = (
                (self.scan_to_process.rt - np.array(self.live_roi_last_rt)[time_filter == 0]) > self.rt_tol)
        scores *= time_filter
        scores *= (self.current_roi_length >= self.min_roi_length_for_fragmentation)
        return scores

    def _get_top_N_scores(self, scores):
        if len(scores) > self.N:  # number of fragmentation events filter
            scores[scores.argsort()[:(len(scores) - self.N)]] = 0
        return scores


class SmartRoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1,
                 reset_length_seconds=100, intensity_increase_factor=2, length_units="scans",
                 drop_perc=0.01, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift,
                         ms1_agc_target, ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
        self.reset_length_seconds = reset_length_seconds
        self.intensity_increase_factor = intensity_increase_factor
        self.drop_perc = drop_perc

    def _process_scan(self, scan):
        # keep growing ROIs if we encounter a new ms1 scan
        self._update_roi(scan)

        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.scan_to_process is not None:
            self.current_roi_mzs = [roi.mz_list[-1] for roi in self.live_roi]
            self.current_roi_intensities = [roi.get_max_intensity() for roi in self.live_roi]
            self.current_rt = self.scan_to_process.rt

            # FIXME: only the 'scans' mode seems to work on the real mass spec (IAPI), why??
            if self.length_units == "scans":
                self.current_roi_length = np.array([roi.n for roi in self.live_roi])
            else:
                self.current_roi_length = np.array([roi.length_in_seconds for roi in self.live_roi])

            # loop over points in decreasing score
            scores = self._get_scores()
            idx = np.argsort(scores)[::-1]
            for i in idx:
                mz = self.current_roi_mzs[i]
                intensity = self.current_roi_intensities[i]

                # stopping criteria is done based on the scores
                if scores[i] <= 0:
                    logger.debug('Time %f Top-%d ions have been selected' % (self.current_rt, self.N))
                    break

                # updated fragmented list and times
                self.live_roi_fragmented[i] = True
                self.live_roi_last_rt[i] = self.current_rt

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                      self.isolation_width, self.mz_tol, self.rt_tol)
                new_tasks.append(dda_scan_params)
                self.live_roi[i].fragmented()

            ms1_scan_params = self.environment.get_default_scan_params()
            ms1_insert_position = max(len(new_tasks) - self.ms1_shift, 0)
            new_tasks.insert(ms1_insert_position, ms1_scan_params)
            num_scans = (len(self.scans[1]) + len(self.scans[2]) + ms1_insert_position + self.pending_tasks)
            self.next_processed_scan_id = num_scans

            # create temp exclusion items
            tasks = new_tasks[(ms1_insert_position + 1):]
            self.temp_exclusion_list = self._update_temp_exclusion_list(tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
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
                    match_roi = match(SmartRoi(mz, 0, 0, self.min_roi_length_for_fragmentation,
                                               self.reset_length_seconds, self.intensity_increase_factor, self.rt_tol),
                                      self.live_roi, self.mz_tol, mz_units=self.mz_units)
                    if match_roi:
                        match_roi.add(mz, current_ms1_scan_rt, intensity)
                        if match_roi in not_grew:
                            not_grew.remove(match_roi)
                    else:
                        new_roi = SmartRoi(mz, current_ms1_scan_rt, intensity, self.min_roi_length_for_fragmentation,
                                           self.reset_length_seconds, self.intensity_increase_factor, self.rt_tol, drop_perc = self.drop_perc)
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

    def _get_dda_scores(self):
        scores = np.log(self.current_roi_intensities)  # log intensities
        scores *= (np.log(self.current_roi_intensities) > np.log(self.min_ms1_intensity))  # intensity filter
        scores *= ([roi.get_can_fragment() for roi in self.live_roi])
        return scores


########################################################################################################################
# Extended ROI Controllers
########################################################################################################################


class TopN_SmartRoiController(SmartRoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1,
                 reset_length_seconds=100, intensity_increase_factor=2, length_units="scans", drop_perc=0.01, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation,
                         reset_length_seconds, intensity_increase_factor, length_units, drop_perc, ms1_shift,
                         ms1_agc_target, ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target,
                         ms2_max_it, ms2_collision_energy, ms2_orbitrap_resolution)

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class TopN_RoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class DsDA_RoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length=1, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans",
                 peak_df=None, peak_scores=None, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
        self.peak_df = peak_df
        self.peak_score = peak_scores

    def _get_dsda_scores(self):
        dsda_scores = [1 for i in self.live_roi]  # TODO: implement DSDA peak scores
        return dsda_scores

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        initial_scores *= self._get_dsda_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class Probability_RoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 probability_method, model_params,  # controller specific parameters
                 min_roi_length=1, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
        self.probability_method = probability_method
        self.model_params = model_params

    def _get_prob_scores(self):
        prob_scores = []
        for roi in self.live_roi:
            if len(roi.intensity_list) < min(self.probability_method.roi_change_n,
                                             self.min_roi_length_for_fragmentation):
                prob_scores.append(0)
            else:
                change = calculate_window_change(roi.intensity_list, self.probability_method.roi_change_n)
                probs = self.probability_method.predict(change)
                prob_scores.append(sum(self.model_params * probs))
        return prob_scores

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        initial_scores *= self._get_prob_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class LocalModel_RoiController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 model_input_len, score_params,  # controller specific parameters
                 min_roi_length=1, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
        self.model = linear_model.LinearRegression(fit_intercept=False)
        self.poly_features = PolynomialFeatures(degree=2)
        self.score_params = score_params
        self.model_input_len = model_input_len

    def _get_model_scores(self):
        model_scores = []
        for roi in self.live_roi:
            if len(roi.rt_list) < self.model_input_len:
                model_scores.append(0)
            else:
                x = np.array(roi.rt_list[-self.model_input_len:-1]).reshape(-1, 1)
                x_poly = self.poly_features.fit_transform(x)
                y = np.array(roi.intensity_list[-self.model_input_len:-1]).reshape(-1, 1)
                fitted_model = self.model.fit(x_poly, y)
                beta1, beta2 = fitted_model.coef_[[1, 2]]
                model_score = self.score_params[0] * abs(beta1) + self.score_params[1] * abs(beta2)
                model_score.append(model_score)
        return model_scores

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        initial_scores *= self._get_model_scores()
        scores = self._get_top_N_scores(initial_scores)
        return scores


class Repeated_SmartRoiController(SmartRoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1,
                 reset_length_seconds=100, intensity_increase_factor=2, length_units="scans", drop_perc=0.01,
                 peak_boxes=[], peak_box_scores=[], box_increase_factor=2, box_decrease_factor=0, box_mz_tol=10,
                 ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, rt_tol, min_roi_length_for_fragmentation,reset_length_seconds,
                       intensity_increase_factor, length_units, drop_perc, ms1_shift, ms1_agc_target, ms1_max_it,
                       ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it, ms2_collision_energy,
                       ms2_orbitrap_resolution)
        self.peak_boxes = peak_boxes
        self.peak_box_scores = peak_box_scores
        self.box_increase_factor = box_increase_factor
        self.box_decrease_factor = box_decrease_factor
        self.box_mz_tol = box_mz_tol

    def _get_scores(self):
        initial_scores = np.array(self._get_dda_scores())
        if self.peak_boxes:
            # 1 if in fragmented peak
            # 0 if in un-fragmented peak
            # -1 if not in any peak
            roi_status, model_score_status = self._get_roi_peak_box_status()
            roi_status = np.array(roi_status)
            # peak boxes already fragmented
            initial_scores[np.where(roi_status == 1)] *= self.box_decrease_factor
            # peak boxes not already fragmented
            initial_scores[np.where(roi_status == 0)] *= self.box_increase_factor
        scores = self._get_top_N_scores(initial_scores)
        return scores

    def _get_roi_peak_box_status(self):
        roi_statuses, model_score_status = get_peak_status(self.current_roi_mzs, self.current_rt, self.peak_boxes,
                                                           self.peak_box_scores, box_mz_tol=self.box_mz_tol)
        return roi_statuses, model_score_status


class CaseControl_SmartRoiController(Repeated_SmartRoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N=None, rt_tol=10, min_roi_length_for_fragmentation=1,
                 reset_length_seconds=100, intensity_increase_factor=2, length_units="scans", drop_perc=0.01,
                 peak_boxes=[], peak_box_scores=[], box_increase_factor=2, box_decrease_factor=0, box_mz_tol=10,
                 coef_scale=1, model_scores=None, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                        min_roi_length, N, rt_tol, min_roi_length_for_fragmentation,reset_length_seconds,
                        intensity_increase_factor, length_units, drop_perc, peak_boxes, peak_box_scores,
                        box_increase_factor, box_decrease_factor, box_mz_tol, ms1_shift, ms1_agc_target, ms1_max_it,
                        ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it, ms2_collision_energy,
                        ms2_orbitrap_resolution)
        self.coef_scale = coef_scale
        self.model_scores = model_scores

    def _get_scores(self):
        initial_scores = self._get_dda_scores()
        if self.peak_boxes:
            # 1 if in fragmented peak
            # 0 if in un-fragmented peak
            # -1 if not in any peak
            roi_status, model_score_status = self._get_roi_peak_box_status()
            # peak boxes already fragmented
            initial_scores[roi_status == 1] *= self.box_decrease_factor
            # peak boxes not already fragmented
            initial_scores[roi_status == 0] *= self.box_increase_factor
            if model_score_status is not None:
                initial_scores *= model_score_status * self.coef_scale

        scores = self._get_top_N_scores(initial_scores)
        return scores

    def _get_roi_peak_box_status(self):
        roi_statuses, model_score_status = get_peak_status(self.current_roi_mzs, self.current_rt, self.peak_boxes,
                                                           self.peak_box_scores, self.model_scores,
                                                           box_mz_tol=self.box_mz_tol)
        return roi_statuses, model_score_status


class Classifier_RoiController(RoiController):  # TODO: Needs properly implementing, but working roughly in principle
    def __init__(self, ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                 roi_picking_model, roi_param_dict,  # controller specific parameters
                 min_roi_length=1, N=None, rt_tol=10, min_roi_length_for_fragmentation=1, length_units="scans", ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target = DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it = DEFAULT_MS1_MAXIT,
                 ms1_collision_energy = DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution = DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target = DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it = DEFAULT_MS2_MAXIT,
                 ms2_collision_energy = DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution = DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, isolation_width, mz_tol, min_ms1_intensity, min_roi_intensity,
                         min_roi_length, N, rt_tol, min_roi_length_for_fragmentation, length_units, ms1_shift, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
        self.roi_picking_model = roi_picking_model
        self.roi_param_dict = roi_param_dict

    # def _get_roi_scores(self):
    #     if not self.live_roi:
    #         return []
    #     roi_scores = []
    #     for roi in self.live_roi:
    #         if len(roi.mz_list) < self.min_roi_length_for_fragmentation:
    #             roi_scores.append(0)
    #         else:
    #             roi_df = get_roi_classification_params([roi], self.roi_param_dict)
    #             roi_scores.append(self.roi_picking_model.predict_proba(roi_df)[0][1])
    #     return roi_scores

    # def _get_scores(self):
    #     initial_scores = self._get_dda_scores()
    #     initial_scores *= self._get_roi_scores()
    #     scores = self._get_top_N_scores(initial_scores)
    #     return scoresó

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
        zipped_boxes = list(filter(lambda x: x[0].rt_range_in_seconds[0] <= rt <= x[0].rt_range_in_seconds[1], zip(boxes, box_ids)))
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

